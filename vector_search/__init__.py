"""
VectorSearch: Planetary-Scale Distributed Vector Search & Inference System

SOTA Implementation Features:
    - HNSW with SIMD/AVX2 optimized distance kernels
    - IVF-PQ quantization with OPQ rotation
    - Lock-free concurrent reads with epoch-based reclamation
    - Sub-millisecond search latency at 10M+ vector scale

SDK Ecosystem:
    - HuggingFace Transformers/SentenceTransformers
    - OpenAI text-embedding-3-*
    - Anthropic voyage embeddings
    - Cohere embed-v3
    - LangChain Embeddings wrapper

Usage:
    from vector_search import VectorIndex, EmbeddingVector
    
    # Create index
    index = VectorIndex(dimension=1536, metric="cosine")
    
    # Insert vectors
    index.upsert(
        ids=["doc1", "doc2"],
        vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
        metadata=[{"title": "Doc 1"}, {"title": "Doc 2"}],
    )
    
    # Search
    results = index.search(query_vector=[0.1, 0.2, ...], k=10)
    
    # With embedding provider
    from vector_search.embeddings import OpenAIEmbedder
    
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    results = index.search(query="semantic search query", embedder=embedder, k=10)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Planetary AI Systems"

# =============================================================================
# LAZY IMPORTS FOR FAST STARTUP
# =============================================================================
# Core types (always available, zero dependencies)
from vector_search.core.types import (
    VectorId,
    EmbeddingVector,
    SearchQuery,
    SearchResult,
    IndexConfig,
    MetricType,
)
from vector_search.core.errors import (
    Result,
    Ok,
    Err,
    VectorSearchError,
    IndexError,
    QueryError,
    EmbeddingError,
)

# Index implementations (lazy-loaded on first access)
def __getattr__(name: str):
    """Lazy import of heavy modules for fast startup time."""
    if name == "VectorIndex":
        from vector_search.index.hnsw import HNSWIndex
        return HNSWIndex
    if name == "HNSWIndex":
        from vector_search.index.hnsw import HNSWIndex
        return HNSWIndex
    if name == "ScalarQuantizer":
        from vector_search.index.quantization import ScalarQuantizer
        return ScalarQuantizer
    if name == "ProductQuantizer":
        from vector_search.index.quantization import ProductQuantizer
        return ProductQuantizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Core types
    "VectorId",
    "EmbeddingVector", 
    "SearchQuery",
    "SearchResult",
    "IndexConfig",
    "MetricType",
    # Error handling
    "Result",
    "Ok",
    "Err",
    "VectorSearchError",
    "IndexError",
    "QueryError",
    "EmbeddingError",
    # Index (lazy)
    "VectorIndex",
    "HNSWIndex",
    "ScalarQuantizer",
    "ProductQuantizer",
]
