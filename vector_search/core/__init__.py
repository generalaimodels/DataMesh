"""
Core Module: Types, Errors, and Configuration

Self-contained module with zero external dependencies beyond numpy.
Provides the foundational abstractions for the entire vector search system.
"""

from vector_search.core.types import (
    VectorId,
    EmbeddingVector,
    SearchQuery,
    SearchResult,
    SearchResults,
    IndexConfig,
    MetricType,
    QuantizationType,
    IndexStats,
)
from vector_search.core.errors import (
    Result,
    Ok,
    Err,
    VectorSearchError,
    IndexError,
    QueryError,
    EmbeddingError,
    ConfigError,
    StorageError,
)
from vector_search.core.config import (
    HNSWConfig,
    QuantizationConfig,
    ServerConfig,
)
from vector_search.core.protocols import (
    VectorIndexProtocol,
    EmbedderProtocol,
    StorageProtocol,
)

__all__ = [
    # Types
    "VectorId",
    "EmbeddingVector",
    "SearchQuery",
    "SearchResult",
    "SearchResults",
    "IndexConfig",
    "MetricType",
    "QuantizationType",
    "IndexStats",
    # Errors
    "Result",
    "Ok",
    "Err",
    "VectorSearchError",
    "IndexError",
    "QueryError",
    "EmbeddingError",
    "ConfigError",
    "StorageError",
    # Config
    "HNSWConfig",
    "QuantizationConfig",
    "ServerConfig",
    # Protocols
    "VectorIndexProtocol",
    "EmbedderProtocol",
    "StorageProtocol",
]
