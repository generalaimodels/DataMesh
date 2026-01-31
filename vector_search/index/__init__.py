"""
Index Module: SOTA Vector Indexing Algorithms

Provides:
    - HNSWIndex: SIMD-optimized graph-based ANN
    - Distance kernels: Cosine, L2, Inner Product
    - Quantization: SQ, PQ, OPQ
"""

from vector_search.index.distance import (
    cosine_similarity,
    cosine_distance,
    l2_distance,
    inner_product,
    normalize_vector,
)
from vector_search.index.hnsw import HNSWIndex

__all__ = [
    "HNSWIndex",
    "cosine_similarity",
    "cosine_distance",
    "l2_distance",
    "inner_product",
    "normalize_vector",
]
