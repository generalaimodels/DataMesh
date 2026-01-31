"""
Index Module: SOTA Vector Indexing Algorithms

Provides:
    - HNSWIndex: SIMD-optimized graph-based ANN
    - InvertedIndex: Block-Max WAND sparse retrieval
    - HybridRetriever: Sparse + Dense fusion
    - Distance kernels: Cosine, L2, Inner Product
"""

from vector_search.index.distance import (
    cosine_similarity,
    cosine_distance,
    l2_distance,
    inner_product,
    normalize_vector,
)
from vector_search.index.hnsw import HNSWIndex
from vector_search.index.inverted_index import InvertedIndex, InvertedIndexConfig
from vector_search.index.hybrid_retriever import (
    HybridRetriever,
    HybridConfig,
    FusionMethod,
    HybridMatch,
    HybridResults,
)

__all__ = [
    # Dense Index
    "HNSWIndex",
    # Sparse Index
    "InvertedIndex",
    "InvertedIndexConfig",
    # Hybrid
    "HybridRetriever",
    "HybridConfig",
    "FusionMethod",
    "HybridMatch",
    "HybridResults",
    # Distance
    "cosine_similarity",
    "cosine_distance",
    "l2_distance",
    "inner_product",
    "normalize_vector",
]

