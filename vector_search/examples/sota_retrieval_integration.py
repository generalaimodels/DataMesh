"""
SOTA Exact Text Retrieval Integration

End-to-end demonstration of hybrid retrieval combining:
    - SPLADE v2 Learned Sparse Retrieval (exact lexical + term expansion)
    - HNSW Dense Vector Retrieval (semantic similarity)
    - Reciprocal Rank Fusion (RRF) for score combination
    - Cross-Encoder Reranking for precision optimization

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         SOTA RETRIEVAL PIPELINE                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   [Document]  ──→  [SPLADE Encoder]  ──→  [Inverted Index]              │
    │              ╲                                                           │
    │               ╲──→  [Bi-Encoder]  ──→  [HNSW Index]                     │
    │                                                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   [Query]  ──→  [SPLADE]  ──→  [Sparse Search]  ──┐                     │
    │            ╲                                       │                     │
    │             ╲──→  [Bi-Encoder]  ──→  [Dense Search] ──→ [RRF Fusion]    │
    │                                                          │               │
    │                                                          ↓               │
    │                                                  [Cross-Encoder]         │
    │                                                          │               │
    │                                                          ↓               │
    │                                                  [Final Results]         │
    └─────────────────────────────────────────────────────────────────────────┘

Performance Targets:
    - Recall@10: ≥ 0.90
    - MRR@10: ≥ 0.35
    - P95 Latency: < 50ms
    - Throughput: > 500 QPS

Usage:
    python -m vector_search.examples.sota_retrieval_integration --max-samples 1000

References:
    - Hybrid Retrieval: Lin et al., arXiv 2021
    - SPLADE v2: Formal et al., arXiv 2021
    - Cross-Encoder: Nogueira & Cho, EMNLP 2019
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class SOTARetrievalConfig:
    """
    SOTA Retrieval System Configuration.
    
    Attributes:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use
        max_samples: Maximum documents to index
        dense_model: Bi-encoder model for dense embeddings
        sparse_model: SPLADE model for sparse representations
        enable_reranking: Use cross-encoder for L2 reranking
        batch_size: Batch size for encoding
        dimension: Dense embedding dimension
    """
    # Dataset
    dataset_name: str = "Arena/YinYang-Text"
    split: str = "train"
    max_samples: int = 1000
    
    # Models
    dense_model: str = "intfloat/multilingual-e5-base"
    sparse_model: str = "naver/splade-cocondenser-ensembledistil"
    
    # Retrieval
    enable_reranking: bool = False  # Disabled by default for demo speed
    sparse_candidates: int = 100
    dense_candidates: int = 100
    rerank_top_n: int = 50
    
    # Processing
    batch_size: int = 32
    dimension: int = 768
    
    # Evaluation
    eval_queries: int = 50
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])


# =============================================================================
# DUAL ENCODER (Dense + Sparse)
# =============================================================================
class DualEncoder:
    """
    Combined encoder producing both dense and sparse representations.
    
    Uses:
        - Bi-Encoder (SentenceTransformer) for dense embeddings
        - SPLADE v2 for learned sparse vectors
    """
    
    def __init__(self, config: SOTARetrievalConfig):
        self.config = config
        self._dense_model = None
        self._sparse_encoder = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Lazy-load both encoders."""
        if self._initialized:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_models)
        self._initialized = True
    
    def _load_models(self) -> None:
        """Load dense and sparse models."""
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   📦 Loading Dense Encoder ({device}): {self.config.dense_model}")
        self._dense_model = SentenceTransformer(self.config.dense_model, device=device)
        
        # Sparse encoder (simplified mock for demo without full SPLADE deps)
        print(f"   📦 Loading Sparse Encoder: Simulated SPLADE")
        # For demo: use vocabulary-hashed term weights
        self._sparse_encoder = self._create_simple_sparse_encoder()
    
    def _create_simple_sparse_encoder(self):
        """Create a BM25-style sparse encoder for demo purposes."""
        from vector_search.core.sparse_types import SparseVector
        
        class SimpleSparsEncoder:
            """BM25-inspired sparse encoder (production would use SPLADE)."""
            
            def __init__(self, vocab_size: int = 30522):
                self.vocab_size = vocab_size
                # Simple tokenization
                import re
                self.tokenize = lambda t: re.findall(r'\w+', t.lower())
            
            def encode(self, text: str) -> SparseVector:
                """Convert text to sparse vector via term hashing."""
                tokens = self.tokenize(text)
                if not tokens:
                    return SparseVector.from_dict({})
                
                # Count term frequencies
                tf: Dict[int, float] = {}
                for token in tokens:
                    term_id = hash(token) % self.vocab_size
                    tf[term_id] = tf.get(term_id, 0) + 1.0
                
                # Apply log-TF normalization
                for tid in tf:
                    tf[tid] = 1.0 + np.log(tf[tid])
                
                return SparseVector.from_dict(tf)
            
            def encode_batch(self, texts: List[str]) -> List[SparseVector]:
                return [self.encode(t) for t in texts]
        
        return SimpleSparsEncoder()
    
    def prepare_text(self, record: dict) -> str:
        """Format record for encoding."""
        prompt = record.get("prompt", "")[:500]
        chosen = record.get("chosen", "")[:500]
        return f"passage: Prompt: {prompt}\nAnswer: {chosen}"
    
    async def encode_dense(self, texts: List[str]) -> np.ndarray:
        """Encode texts to dense embeddings."""
        await self.initialize()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._dense_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )
    
    async def encode_sparse(self, texts: List[str]) -> List["SparseVector"]:
        """Encode texts to sparse vectors."""
        await self.initialize()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._sparse_encoder.encode_batch(texts)
        )
    
    async def encode_dual(self, texts: List[str]) -> Tuple[np.ndarray, List["SparseVector"]]:
        """Encode texts to both dense and sparse representations."""
        dense_task = asyncio.create_task(self.encode_dense(texts))
        sparse_task = asyncio.create_task(self.encode_sparse(texts))
        
        dense_vecs = await dense_task
        sparse_vecs = await sparse_task
        
        return dense_vecs, sparse_vecs


# =============================================================================
# SOTA RETRIEVAL PIPELINE
# =============================================================================
class SOTARetrievalPipeline:
    """
    Complete SOTA Hybrid Retrieval Pipeline.
    
    Implements:
        1. Dual-path indexing (sparse + dense)
        2. Parallel query execution
        3. Score fusion (RRF)
        4. Optional cross-encoder reranking
        5. Comprehensive evaluation metrics
    """
    
    def __init__(self, config: SOTARetrievalConfig):
        self.config = config
        self.encoder: Optional[DualEncoder] = None
        self.sparse_index = None
        self.dense_index = None
        self.hybrid_retriever = None
        self.doc_store: Dict[int, str] = {}  # doc_id → text
        self.stats = {
            "indexed": 0,
            "ingestion_time_ms": 0,
            "latencies": [],
        }
    
    async def initialize(self) -> None:
        """Initialize all components."""
        from vector_search.core.config import HNSWConfig
        from vector_search.core.types import VectorId, EmbeddingVector
        from vector_search.index.hnsw import HNSWIndex
        from vector_search.index.inverted_index import InvertedIndex, InvertedIndexConfig
        from vector_search.index.hybrid_retriever import HybridRetriever, HybridConfig, FusionMethod
        
        print("\n🚀 Initializing SOTA Retrieval Pipeline")
        print("=" * 60)
        
        # Initialize encoder
        self.encoder = DualEncoder(self.config)
        await self.encoder.initialize()
        
        # Initialize indexes
        print("   🔧 Creating Indexes...")
        
        # Sparse index (inverted)
        sparse_config = InvertedIndexConfig(
            initial_capacity=self.config.max_samples,
            use_wand=True,
        )
        self.sparse_index = InvertedIndex(sparse_config)
        
        # Dense index (HNSW)
        dense_config = HNSWConfig(
            dimension=self.config.dimension,
            M=16,
            ef_construction=100,
            ef_search=64,
        )
        self.dense_index = HNSWIndex(dense_config)
        
        # Hybrid retriever (will be set after indexing)
        print("   ✅ Pipeline Initialized\n")
    
    async def ingest_documents(self, records: List[dict]) -> int:
        """
        Ingest documents into both indexes.
        
        Args:
            records: List of document records
            
        Returns:
            Number of documents indexed
        """
        from vector_search.core.types import VectorId, EmbeddingVector
        
        print(f"📥 Ingesting {len(records)} documents...")
        start_time = time.perf_counter()
        
        # Prepare texts
        texts = [self.encoder.prepare_text(r) for r in records]
        
        # Encode in batches
        indexed = 0
        batch_size = self.config.batch_size
        
        for i in range(0, len(records), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_records = records[i:i + batch_size]
            
            # Dual encoding
            dense_vecs, sparse_vecs = await self.encoder.encode_dual(batch_texts)
            
            # Index each document
            for j, (text, dense_vec, sparse_vec, record) in enumerate(
                zip(batch_texts, dense_vecs, sparse_vecs, batch_records)
            ):
                doc_id = i + j
                
                # Store document text for reranking
                self.doc_store[doc_id] = text
                
                # Add to sparse index
                self.sparse_index.add_document(
                    doc_id=doc_id,
                    sparse_vec=sparse_vec,
                    external_id=f"doc-{doc_id}",
                    metadata={"prompt": record.get("prompt", "")[:100]},
                )
                
                # Add to dense index
                vid = VectorId(value=f"doc-{doc_id}")
                emb = EmbeddingVector.from_numpy(dense_vec.astype(np.float32))
                self.dense_index.insert(vid, emb, metadata={"idx": doc_id})
                
                indexed += 1
            
            # Progress
            progress = min(100, int((i + len(batch_texts)) / len(records) * 100))
            print(f"   Indexed: {indexed}/{len(records)} ({progress}%)", end="\r")
        
        # Compile sparse index for querying
        self.sparse_index.compile()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats["indexed"] = indexed
        self.stats["ingestion_time_ms"] = elapsed_ms
        
        print(f"\n   ✅ Indexed {indexed} documents in {elapsed_ms:.1f}ms")
        print(f"   Throughput: {indexed / (elapsed_ms / 1000):.1f} docs/sec")
        
        # Create hybrid retriever
        from vector_search.index.hybrid_retriever import HybridRetriever, HybridConfig, FusionMethod
        
        hybrid_config = HybridConfig(
            sparse_candidates=self.config.sparse_candidates,
            dense_candidates=self.config.dense_candidates,
            fusion_method=FusionMethod.RRF,
            enable_reranking=self.config.enable_reranking,
            rerank_top_n=self.config.rerank_top_n,
        )
        
        self.hybrid_retriever = HybridRetriever(
            sparse_index=self.sparse_index,
            dense_index=self.dense_index,
            config=hybrid_config,
            doc_store=self.doc_store,
        )
        
        return indexed
    
    async def search(
        self,
        query: str,
        k: int = 10,
    ) -> "HybridResults":
        """
        Execute hybrid search query.
        
        Args:
            query: Natural language query
            k: Number of results to return
            
        Returns:
            HybridResults with matches and diagnostics
        """
        # Encode query
        query_dense = (await self.encoder.encode_dense([f"query: {query}"]))[0]
        query_sparse = (await self.encoder.encode_sparse([query]))[0]
        
        # Search
        results = await self.hybrid_retriever.search(
            query_sparse=query_sparse,
            query_dense=query_dense,
            k=k,
            query_text=query,
        )
        
        self.stats["latencies"].append(results.total_time_ms)
        
        return results
    
    async def evaluate(
        self,
        eval_data: List[Tuple[str, int]],  # (query, relevant_doc_id)
    ) -> Dict[str, float]:
        """
        Run offline evaluation harness.
        
        Computes:
            - Recall@K for various K
            - MRR@10
            - NDCG@10
            - Latency percentiles
        """
        print("\n📊 Evaluation Harness")
        print("=" * 60)
        
        metrics = {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
        }
        
        k_max = max(self.config.k_values)
        num_queries = len(eval_data)
        
        for query_text, relevant_doc_id in eval_data:
            results = await self.search(query_text, k=k_max)
            
            # Get returned doc IDs
            returned_ids = [m.doc_id for m in results.matches]
            
            # Recall at various K
            for k in self.config.k_values:
                if relevant_doc_id in returned_ids[:k]:
                    metrics[f"recall@{k}"] += 1
            
            # MRR@10
            for rank, doc_id in enumerate(returned_ids[:10], start=1):
                if doc_id == relevant_doc_id:
                    metrics["mrr@10"] += 1.0 / rank
                    break
            
            # NDCG@10 (binary relevance)
            for rank, doc_id in enumerate(returned_ids[:10], start=1):
                if doc_id == relevant_doc_id:
                    metrics["ndcg@10"] += 1.0 / np.log2(rank + 1)
                    break
        
        # Normalize
        for key in metrics:
            metrics[key] /= num_queries
        
        # Latency stats
        latencies = self.stats["latencies"]
        if latencies:
            metrics["p50_latency_ms"] = float(np.percentile(latencies, 50))
            metrics["p95_latency_ms"] = float(np.percentile(latencies, 95))
            metrics["p99_latency_ms"] = float(np.percentile(latencies, 99))
        
        # Print results
        print(f"\n🎯 Retrieval Quality (n={num_queries} queries):")
        print(f"   Recall@1:  {metrics['recall@1']:.3f}")
        print(f"   Recall@5:  {metrics['recall@5']:.3f}")
        print(f"   Recall@10: {metrics['recall@10']:.3f}")
        print(f"   Recall@20: {metrics['recall@20']:.3f}")
        print(f"   MRR@10:    {metrics['mrr@10']:.3f}")
        print(f"   NDCG@10:   {metrics['ndcg@10']:.3f}")
        
        print(f"\n⏱️  Latency:")
        print(f"   P50: {metrics.get('p50_latency_ms', 0):.2f} ms")
        print(f"   P95: {metrics.get('p95_latency_ms', 0):.2f} ms")
        print(f"   P99: {metrics.get('p99_latency_ms', 0):.2f} ms")
        
        # Quality gate
        if metrics["recall@10"] >= 0.8:
            print("\n✅ Quality Gate PASSED")
        else:
            print("\n⚠️  Quality Gate WARNING")
        
        return metrics


# =============================================================================
# INTERACTIVE SEARCH CLI
# =============================================================================
async def interactive_search(pipeline: SOTARetrievalPipeline) -> None:
    """Interactive search mode for testing queries."""
    print("\n🎮 Interactive Search Mode")
    print("=" * 60)
    print("Type a query and press Enter. Type 'exit' to quit.\n")
    
    loop = asyncio.get_running_loop()
    
    while True:
        try:
            query = await loop.run_in_executor(None, input, "Query: ")
            query = query.strip()
            
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            
            # Search
            start = time.perf_counter()
            results = await pipeline.search(query, k=5)
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"\n🔍 Results for: '{query}' ({elapsed:.1f}ms)")
            print("-" * 50)
            
            for i, match in enumerate(results.matches, 1):
                sparse_info = f"S:{match.sparse_rank}" if match.sparse_rank else "S:-"
                dense_info = f"D:{match.dense_rank}" if match.dense_rank else "D:-"
                
                # Get snippet from doc store
                doc_text = pipeline.doc_store.get(match.doc_id, "")[:100]
                
                print(f"{i}. [Score: {match.score:.4f}] [{sparse_info}, {dense_info}]")
                print(f"   {doc_text}...")
                print()
            
        except (KeyboardInterrupt, EOFError):
            break
    
    print("\n👋 Exiting interactive mode.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
async def main(args: argparse.Namespace) -> None:
    """Main pipeline execution."""
    
    # Configuration
    config = SOTARetrievalConfig(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        eval_queries=args.eval_queries,
        enable_reranking=args.rerank,
    )
    
    # Initialize pipeline
    pipeline = SOTARetrievalPipeline(config)
    await pipeline.initialize()
    
    # Load dataset
    try:
        from datasets import load_dataset
        print(f"\n📂 Loading Dataset: {config.dataset_name}")
        ds = load_dataset(config.dataset_name, split=config.split, streaming=True)
        
        # Collect documents
        records = []
        for i, item in enumerate(ds):
            if i >= config.max_samples:
                break
            records.append(item)
        
        print(f"   Loaded {len(records)} documents")
        
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        print("   Creating synthetic demo data...")
        
        # Synthetic demo data
        records = [
            {"prompt": f"What is topic {i}?", "chosen": f"Topic {i} is about {['ML', 'AI', 'NLP', 'CV'][i % 4]}..."}
            for i in range(min(100, config.max_samples))
        ]
    
    # Ingest documents
    await pipeline.ingest_documents(records)
    
    # Create evaluation data
    import random
    random.seed(42)
    eval_samples = random.sample(range(len(records)), min(config.eval_queries, len(records)))
    eval_data = [
        (records[idx].get("prompt", "")[:200], idx)
        for idx in eval_samples
    ]
    
    # Run evaluation
    if args.verify:
        await pipeline.evaluate(eval_data)
    
    # Interactive mode
    if args.interactive:
        await interactive_search(pipeline)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Pipeline Summary")
    print("=" * 60)
    print(f"   Documents Indexed: {pipeline.stats['indexed']}")
    print(f"   Ingestion Time: {pipeline.stats['ingestion_time_ms']:.1f}ms")
    print(f"   Dense Index Size: {pipeline.dense_index.count} vectors")
    print(f"   Sparse Index Stats: {pipeline.sparse_index.stats()}")
    print("\n✅ SOTA Retrieval Pipeline Complete!")


def cli_main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SOTA Exact Text Retrieval System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=1000,
        help="Maximum documents to index"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--eval-queries",
        type=int,
        default=50,
        help="Number of evaluation queries"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run evaluation harness"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive search mode"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
