"""
HuggingFace Embeddings: Transformers/SentenceTransformers Integration

Supports:
    - sentence-transformers models (all-MiniLM, all-mpnet, etc.)
    - HuggingFace transformers with custom pooling
    - Local models and Hugging Face Hub models

Features:
    - Automatic model download and caching
    - GPU acceleration when available
    - Configurable pooling strategies
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

from vector_search.core.errors import Err, Ok, Result
from vector_search.core.types import EmbeddingVector
from vector_search.embeddings.base import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """
    HuggingFace embeddings using sentence-transformers.
    
    Example:
        embedder = HuggingFaceEmbedder(model="all-MiniLM-L6-v2")
        result = embedder.embed("Hello world")
    """
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        pooling: Literal["mean", "cls", "max"] = "mean",
    ) -> None:
        """
        Initialize HuggingFace embedder.
        
        Args:
            model: Model name or path
            device: Device ("cpu", "cuda", "mps", or None for auto)
            normalize: Whether to L2-normalize embeddings
            pooling: Pooling strategy for token embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install vectorsearch[huggingface]"
            )
        
        # Load model
        self._model = SentenceTransformer(model, device=device)
        self._normalize = normalize
        self._pooling = pooling
        
        # Get dimension from model
        dim = self._model.get_sentence_embedding_dimension()
        super().__init__(model_name=model, dimension=dim)
    
    def embed(self, text: str) -> Result[EmbeddingVector, str]:
        """Embed single text."""
        result = self.embed_batch([text])
        if result.is_err():
            return Err(result.error)  # type: ignore
        return Ok(result.unwrap()[0])
    
    def embed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """
        Embed batch of texts.
        
        Uses sentence-transformers batch encoding for efficiency.
        """
        try:
            # Encode with sentence-transformers
            embeddings = self._model.encode(
                list(texts),
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
            )
            
            # Convert to EmbeddingVectors
            vectors = [
                EmbeddingVector.from_numpy(emb.astype(np.float32))
                for emb in embeddings
            ]
            
            return Ok(vectors)
            
        except Exception as e:
            return Err(f"HuggingFace embedding failed: {str(e)}")
    
    async def aembed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """
        Async embed batch.
        
        Note: sentence-transformers is sync, so this runs in thread pool.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)
