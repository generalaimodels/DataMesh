"""
Tests Module: Unit and Integration Tests

Test Coverage:
    - Core types (VectorId, EmbeddingVector, SearchResult)
    - HNSW index (insert, search, delete)
    - Distance functions (cosine, L2, IP)
    - Embeddings (mock embedder)
"""

from vector_search.tests.test_types import *
from vector_search.tests.test_hnsw import *
from vector_search.tests.test_distance import *
