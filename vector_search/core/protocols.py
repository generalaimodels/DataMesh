"""
Protocol Definitions: Structural Subtyping for Pluggable Backends

Defines abstract interfaces for:
    - VectorIndexProtocol: Vector insert/search/delete
    - EmbedderProtocol: Text-to-vector embedding
    - StorageProtocol: Index persistence
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:
    from vector_search.core.types import (
        EmbeddingVector,
        IndexStats,
        SearchQuery,
        SearchResults,
        VectorId,
    )
    from vector_search.core.errors import Result


# =============================================================================
# VECTOR INDEX PROTOCOL
# =============================================================================
@runtime_checkable
class VectorIndexProtocol(Protocol):
    """
    Protocol for vector index implementations.
    
    Implementations:
        - HNSWIndex: In-memory graph-based search
        - DiskANNIndex: SSD-optimized graph search
        - FlatIndex: Brute-force exact search
    """
    
    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        ...
    
    @property
    def count(self) -> int:
        """Number of indexed vectors."""
        ...
    
    @abstractmethod
    def insert(
        self,
        id: "VectorId",
        vector: "EmbeddingVector",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Result[None, str]":
        """Insert single vector."""
        ...
    
    @abstractmethod
    def insert_batch(
        self,
        ids: Sequence["VectorId"],
        vectors: Sequence["EmbeddingVector"],
        metadata: Optional[Sequence[dict[str, Any]]] = None,
    ) -> "Result[int, str]":
        """Insert batch of vectors. Returns count inserted."""
        ...
    
    @abstractmethod
    def search(self, query: "SearchQuery") -> "Result[SearchResults, str]":
        """Search for similar vectors."""
        ...
    
    @abstractmethod
    def delete(self, id: "VectorId") -> "Result[bool, str]":
        """Delete vector by ID. Returns True if existed."""
        ...
    
    @abstractmethod
    def get(self, id: "VectorId") -> "Result[EmbeddingVector, str]":
        """Get vector by ID."""
        ...
    
    @abstractmethod
    def stats(self) -> "IndexStats":
        """Get index statistics."""
        ...


# =============================================================================
# EMBEDDER PROTOCOL
# =============================================================================
@runtime_checkable
class EmbedderProtocol(Protocol):
    """
    Protocol for embedding providers.
    
    Implementations:
        - OpenAIEmbedder
        - HuggingFaceEmbedder
        - CohereEmbedder
        - AnthropicEmbedder (via Voyage)
    """
    
    @property
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...
    
    @property
    def model_name(self) -> str:
        """Model identifier."""
        ...
    
    @abstractmethod
    def embed(self, text: str) -> "Result[EmbeddingVector, str]":
        """Embed single text."""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> "Result[list[EmbeddingVector], str]":
        """Embed batch of texts."""
        ...
    
    @abstractmethod
    async def aembed(self, text: str) -> "Result[EmbeddingVector, str]":
        """Async embed single text."""
        ...
    
    @abstractmethod
    async def aembed_batch(self, texts: Sequence[str]) -> "Result[list[EmbeddingVector], str]":
        """Async embed batch of texts."""
        ...


# =============================================================================
# STORAGE PROTOCOL
# =============================================================================
@runtime_checkable
class StorageProtocol(Protocol):
    """
    Protocol for index persistence.
    
    Implementations:
        - FileStorage: Local file system
        - S3Storage: AWS S3
        - GCSStorage: Google Cloud Storage
    """
    
    @abstractmethod
    def save(self, path: str, data: bytes) -> "Result[None, str]":
        """Save data to path."""
        ...
    
    @abstractmethod
    def load(self, path: str) -> "Result[bytes, str]":
        """Load data from path."""
        ...
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...
    
    @abstractmethod
    def delete(self, path: str) -> "Result[None, str]":
        """Delete path."""
        ...
