"""
SPLADE v2 Learned Sparse Encoder

State-of-the-art learned sparse retrieval model producing sparse term vectors
with neural term expansion and importance weighting.

Features:
    - SPLADE (Sparse Lexical AnD Expansion) architecture
    - MLM-based term importance scoring with ReLU + log activation
    - Query/document asymmetric encoding with separate prefixes
    - Efficient batched GPU inference with dynamic padding
    - Top-k pruning for memory-efficient sparse vectors

Architecture:
    - Backbone: Transformer encoder (BERT/DistilBERT)
    - Aggregation: Max-pooling over token representations
    - Activation: log(1 + ReLU(x)) for sparse, non-negative weights
    - Output: Vocabulary-sized sparse vector (30522 for BERT vocab)

Performance (MSMARCO Dev):
    - MRR@10: ~0.368 (SPLADE v2)
    - Recall@1000: ~0.979
    - Index time: ~50ms/doc (GPU)
    - Query time: ~10ms (GPU)

References:
    - SPLADE: Formal et al., SIGIR 2021
    - SPLADE v2: Formal et al., arXiv 2021
    - Efficient SPLADE: Lassance & Clinchant, SIGIR 2022
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Final,
    Optional,
    Sequence,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

from vector_search.core.sparse_types import SparseVector


# =============================================================================
# CONSTANTS
# =============================================================================
# Default SPLADE model from Hugging Face Hub
DEFAULT_SPLADE_MODEL: Final[str] = "naver/splade-cocondenser-ensembledistil"

# Maximum sequence length for SPLADE
MAX_SEQUENCE_LENGTH: Final[int] = 256

# Default top-k pruning (keep only top-k weights)
DEFAULT_TOP_K: Final[int] = 256

# Minimum weight threshold for pruning
MIN_WEIGHT_THRESHOLD: Final[float] = 1e-4

# Query/Document prefixes for asymmetric encoding
QUERY_PREFIX: Final[str] = ""
DOC_PREFIX: Final[str] = ""


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class SPLADEConfig:
    """
    SPLADE encoder configuration.
    
    Attributes:
        model_name: Hugging Face model identifier
        max_length: Maximum token sequence length
        top_k: Keep only top-k weights per vector (None = keep all)
        min_weight: Minimum weight threshold for pruning
        device: Compute device ('cuda', 'cpu', or 'auto')
        batch_size: Maximum batch size for inference
        use_fp16: Enable FP16 inference for faster GPU computation
        query_prefix: Text prefix for query encoding
        doc_prefix: Text prefix for document encoding
    """
    model_name: str = DEFAULT_SPLADE_MODEL
    max_length: int = MAX_SEQUENCE_LENGTH
    top_k: Optional[int] = DEFAULT_TOP_K
    min_weight: float = MIN_WEIGHT_THRESHOLD
    device: str = "auto"
    batch_size: int = 32
    use_fp16: bool = True
    query_prefix: str = QUERY_PREFIX
    doc_prefix: str = DOC_PREFIX


# =============================================================================
# SPLADE ENCODER
# =============================================================================
class SPLADEEncoder:
    """
    SPLADE v2 learned sparse encoder for hybrid retrieval.
    
    Produces vocabulary-sized sparse vectors where each dimension
    represents a term's importance score. Achieves term expansion
    by predicting non-zero weights for semantically related terms
    not present in the input text.
    
    Thread Safety:
        - Encoder is thread-safe for read operations
        - Model loading is lazy and thread-safe via lock
        - Inference uses thread pool for async compatibility
    
    Memory Layout:
        - Model weights: ~250MB (DistilBERT base)
        - Per-vector storage: 8 * top_k bytes (uint32 + float32)
        - GPU memory: ~500MB during inference
    
    Example:
        >>> encoder = SPLADEEncoder(SPLADEConfig())
        >>> query_vec = await encoder.encode_query("what is machine learning")
        >>> doc_vec = await encoder.encode_document("ML is a type of AI...")
        >>> score = query_vec.dot(doc_vec)
    """
    
    __slots__ = (
        '_config', '_model', '_tokenizer', '_device', 
        '_executor', '_lock', '_initialized'
    )
    
    def __init__(self, config: Optional[SPLADEConfig] = None) -> None:
        """
        Initialize SPLADE encoder.
        
        Args:
            config: Encoder configuration (uses defaults if None)
        """
        self._config = config or SPLADEConfig()
        self._model: Optional["AutoModelForMaskedLM"] = None
        self._tokenizer: Optional["AutoTokenizer"] = None
        self._device: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = asyncio.Lock()
        self._initialized = False
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def config(self) -> SPLADEConfig:
        """Get encoder configuration."""
        return self._config
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (dimension of sparse vectors)."""
        if self._tokenizer is None:
            return 30522  # Default BERT vocab size
        return self._tokenizer.vocab_size
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._initialized
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    async def initialize(self) -> None:
        """
        Lazy-load model and tokenizer.
        
        Thread-safe initialization with async lock.
        """
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._load_model)
            self._initialized = True
    
    def _load_model(self) -> None:
        """Synchronous model loading (run in executor)."""
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        
        # Determine device
        if self._config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._config.device
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_name,
            use_fast=True,
        )
        
        # Load model
        self._model = AutoModelForMaskedLM.from_pretrained(
            self._config.model_name,
            torch_dtype=torch.float16 if self._config.use_fp16 else torch.float32,
        )
        self._model.eval()
        self._model.to(self._device)
        
        # Disable gradient computation
        for param in self._model.parameters():
            param.requires_grad = False
    
    # -------------------------------------------------------------------------
    # Encoding Methods
    # -------------------------------------------------------------------------
    async def encode_query(self, text: str) -> SparseVector:
        """
        Encode query text to sparse vector.
        
        Args:
            text: Query text to encode
            
        Returns:
            SparseVector with term weights
            
        Note: Applies query_prefix before encoding.
        """
        await self.initialize()
        prefixed = f"{self._config.query_prefix}{text}"
        return await self._encode_single(prefixed)
    
    async def encode_document(self, text: str) -> SparseVector:
        """
        Encode document text to sparse vector.
        
        Args:
            text: Document text to encode
            
        Returns:
            SparseVector with term weights
            
        Note: Applies doc_prefix before encoding.
        """
        await self.initialize()
        prefixed = f"{self._config.doc_prefix}{text}"
        return await self._encode_single(prefixed)
    
    async def encode_queries_batch(
        self, 
        texts: Sequence[str],
    ) -> list[SparseVector]:
        """
        Batch encode multiple queries.
        
        Args:
            texts: List of query texts
            
        Returns:
            List of SparseVectors in same order
        """
        await self.initialize()
        prefixed = [f"{self._config.query_prefix}{t}" for t in texts]
        return await self._encode_batch(prefixed)
    
    async def encode_documents_batch(
        self,
        texts: Sequence[str],
    ) -> list[SparseVector]:
        """
        Batch encode multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of SparseVectors in same order
        """
        await self.initialize()
        prefixed = [f"{self._config.doc_prefix}{t}" for t in texts]
        return await self._encode_batch(prefixed)
    
    async def _encode_single(self, text: str) -> SparseVector:
        """Internal single text encoding."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            self._encode_sync, 
            [text]
        ).__getitem__(0)
    
    async def _encode_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        """Internal batch encoding with chunking."""
        if not texts:
            return []
        
        loop = asyncio.get_running_loop()
        results: list[SparseVector] = []
        
        # Process in chunks to avoid OOM
        batch_size = self._config.batch_size
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            chunk_results = await loop.run_in_executor(
                self._executor,
                self._encode_sync,
                chunk,
            )
            results.extend(chunk_results)
        
        return results
    
    def _encode_sync(self, texts: Sequence[str]) -> list[SparseVector]:
        """
        Synchronous batch encoding (run in executor).
        
        Implements SPLADE forward pass:
            1. Tokenize input texts
            2. Forward through MLM head
            3. Apply ReLU and log scaling
            4. Max-pool over sequence length
            5. Convert to sparse representation
        """
        import torch
        
        # Tokenize
        inputs = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # SPLADE activation: log(1 + ReLU(x))
        # Clamp to avoid log(0) and numerical instability
        activated = torch.log1p(torch.clamp(logits, min=0))
        
        # Max-pool over sequence (masked by attention)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        activated = activated * attention_mask
        sparse_reps = torch.max(activated, dim=1).values  # [batch, vocab_size]
        
        # Convert to SparseVector
        sparse_reps = sparse_reps.cpu().numpy().astype(np.float32)
        
        results = []
        for rep in sparse_reps:
            sv = SparseVector.from_dense(
                rep,
                top_k=self._config.top_k,
                threshold=self._config.min_weight,
            )
            results.append(sv)
        
        return results
    
    # -------------------------------------------------------------------------
    # Term Decoding Utilities
    # -------------------------------------------------------------------------
    def decode_terms(self, sparse_vec: SparseVector, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Decode sparse vector to human-readable term-weight pairs.
        
        Args:
            sparse_vec: Sparse vector to decode
            top_n: Number of top terms to return
            
        Returns:
            List of (term, weight) tuples sorted by weight descending
        """
        if self._tokenizer is None:
            raise RuntimeError("Encoder not initialized")
        
        # Get top-n by weight
        if sparse_vec.nnz <= top_n:
            indices = np.argsort(sparse_vec.weights)[::-1]
        else:
            indices = np.argpartition(sparse_vec.weights, -top_n)[-top_n:]
            indices = indices[np.argsort(sparse_vec.weights[indices])[::-1]]
        
        terms = []
        for idx in indices:
            term_id = int(sparse_vec.term_ids[idx])
            weight = float(sparse_vec.weights[idx])
            token = self._tokenizer.decode([term_id])
            terms.append((token.strip(), weight))
        
        return terms
    
    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------
    async def __aenter__(self) -> "SPLADEEncoder":
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._executor.shutdown(wait=False)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================
def create_splade_encoder(
    model_name: Optional[str] = None,
    device: str = "auto",
    top_k: int = DEFAULT_TOP_K,
) -> SPLADEEncoder:
    """
    Factory function to create SPLADE encoder with common settings.
    
    Args:
        model_name: Hugging Face model name (None = default SPLADE)
        device: Compute device ('cuda', 'cpu', 'auto')
        top_k: Maximum non-zero weights per vector
        
    Returns:
        Configured SPLADEEncoder instance
    """
    config = SPLADEConfig(
        model_name=model_name or DEFAULT_SPLADE_MODEL,
        device=device,
        top_k=top_k,
    )
    return SPLADEEncoder(config)


# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    "SPLADEConfig",
    "SPLADEEncoder",
    "create_splade_encoder",
    "DEFAULT_SPLADE_MODEL",
    "MAX_SEQUENCE_LENGTH",
    "DEFAULT_TOP_K",
]
