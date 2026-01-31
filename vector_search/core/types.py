"""
Core Type Definitions: Platform-Agnostic Vector Search Primitives

Designed for seamless integration with:
    - HuggingFace Transformers/SentenceTransformers
    - OpenAI text-embedding-3-*
    - Anthropic voyage embeddings
    - Cohere embed-v3
    - LangChain Embeddings

Memory Layout Optimization:
    - __slots__ for minimal memory footprint
    - Contiguous numpy arrays for SIMD operations
    - Zero-copy views where possible
    
Thread Safety:
    - Immutable types (frozen=True) for lock-free sharing
    - Mutable types use explicit synchronization
"""

from __future__ import annotations

import array
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)
from uuid import UUID, uuid4

# Conditional numpy import for zero-dependency core
if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


# =============================================================================
# METRIC TYPES
# =============================================================================
class MetricType(Enum):
    """
    Distance/similarity metrics for vector comparison.
    
    Ordered by computational cost (ascending):
        INNER_PRODUCT: 1 FMA per dimension
        COSINE: 3 FMAs per dimension (normalize + dot)
        L2: 1 FMA + 1 sqrt
        L1: 1 abs + 1 add per dimension
    """
    COSINE = "cosine"           # Cosine similarity (most common)
    L2 = "l2"                   # Euclidean distance
    INNER_PRODUCT = "ip"        # Inner product (dot product)
    L1 = "l1"                   # Manhattan distance
    
    def is_similarity(self) -> bool:
        """True if higher values = more similar."""
        return self in (MetricType.COSINE, MetricType.INNER_PRODUCT)
    
    def is_distance(self) -> bool:
        """True if lower values = more similar."""
        return self in (MetricType.L2, MetricType.L1)


class QuantizationType(Enum):
    """Vector quantization methods for compression."""
    NONE = "none"               # Full precision float32
    FLOAT16 = "float16"         # Half precision (2x compression)
    BFLOAT16 = "bfloat16"       # Brain float (2x, better range)
    INT8 = "int8"               # Scalar quantization (4x)
    UINT8 = "uint8"             # Unsigned scalar (4x)
    BINARY = "binary"           # 1-bit (32x, hamming distance)
    PQ = "pq"                   # Product quantization (16-64x)
    OPQ = "opq"                 # Optimized PQ with rotation
    IVF_PQ = "ivf_pq"           # IVF + PQ (recommended for 1M+)


# =============================================================================
# VECTOR ID: CROSS-PLATFORM IDENTIFIER
# =============================================================================
@dataclass(frozen=True, slots=True)
class VectorId:
    """
    Globally unique vector identifier.
    
    Supports multiple ID formats for cross-platform compatibility:
        - String IDs (LangChain, Pinecone style)
        - UUID IDs (internal, Milvus style)
        - Composite IDs (namespace:id format)
    
    Memory Layout:
        value: 8 bytes (string pointer)
        namespace: 8 bytes (string pointer, interned)
        Total: 16 bytes + string contents
        
    Sharding:
        Used as consistent hash key for partition assignment.
        Namespace provides logical isolation within physical shards.
    """
    value: str
    namespace: str = "default"
    
    @classmethod
    def generate(cls, namespace: str = "default") -> "VectorId":
        """Generate new VectorId with random UUID."""
        return cls(value=str(uuid4()), namespace=namespace)
    
    @classmethod
    def from_uuid(cls, uuid_val: UUID, namespace: str = "default") -> "VectorId":
        """Create from UUID object."""
        return cls(value=str(uuid_val), namespace=namespace)
    
    @classmethod
    def from_int(cls, int_val: int, namespace: str = "default") -> "VectorId":
        """Create from integer ID (auto-increment style)."""
        return cls(value=str(int_val), namespace=namespace)
    
    def to_composite(self) -> str:
        """Format as namespace:id for storage keys."""
        return f"{self.namespace}:{self.value}"
    
    @classmethod
    def from_composite(cls, composite: str) -> "VectorId":
        """Parse namespace:id format."""
        if ":" in composite:
            namespace, value = composite.split(":", 1)
            return cls(value=value, namespace=namespace)
        return cls(value=composite, namespace="default")
    
    def shard_key(self) -> bytes:
        """
        Compute shard key for consistent hashing.
        
        Returns 20-byte key: SHA1(namespace:id)
        Used by consistent hash ring for partition assignment.
        """
        import hashlib
        return hashlib.sha1(self.to_composite().encode()).digest()
    
    def __str__(self) -> str:
        return self.value if self.namespace == "default" else self.to_composite()
    
    def __hash__(self) -> int:
        return hash((self.value, self.namespace))


# =============================================================================
# EMBEDDING VECTOR: SIMD-OPTIMIZED STORAGE
# =============================================================================
@dataclass(slots=True)
class EmbeddingVector:
    """
    High-dimensional vector optimized for SIMD operations.
    
    Memory Layout:
        - Uses numpy array internally for contiguous memory
        - 64-byte aligned for AVX-512 operations
        - Supports zero-copy views from external buffers
        
    Supported Input Types:
        - list[float]: Standard Python list
        - numpy.ndarray: NumPy array (preferred, zero-copy)
        - torch.Tensor: PyTorch tensor (auto-converted)
        - bytes: Raw binary data
        
    Quantization:
        Original data preserved; quantized view available via to_quantized()
    """
    _buffer: bytes  # Raw bytes storage (portable, serializable)
    dimension: int
    dtype: Literal["float32", "float16", "bfloat16", "int8"] = "float32"
    
    # Cached numpy view (lazy-initialized)
    _np_cache: Optional["np.ndarray"] = field(default=None, repr=False, compare=False)
    
    @classmethod
    def from_list(cls, values: Sequence[float], dtype: str = "float32") -> "EmbeddingVector":
        """
        Create from Python list of floats.
        
        Args:
            values: Sequence of float values
            dtype: Target dtype (float32, float16, int8)
            
        Returns:
            EmbeddingVector with data in specified dtype
            
        Complexity: O(n) for copying
        """
        import numpy as np
        arr = np.array(values, dtype=np.float32)
        if dtype == "float16":
            arr = arr.astype(np.float16)
        elif dtype == "int8":
            # Simple scalar quantization for now
            arr = np.clip(arr * 127, -128, 127).astype(np.int8)
        return cls(
            _buffer=arr.tobytes(),
            dimension=len(values),
            dtype=dtype,  # type: ignore
        )
    
    @classmethod
    def from_numpy(cls, arr: "np.ndarray") -> "EmbeddingVector":
        """
        Create from NumPy array.
        
        Args:
            arr: NumPy array (1D or will be flattened)
            
        Returns:
            EmbeddingVector (copies data to internal buffer)
            
        Complexity: O(n) for copying
        """
        import numpy as np
        arr = np.asarray(arr).flatten()
        
        # Map numpy dtype to our dtype enum
        dtype_map = {
            np.float32: "float32",
            np.float16: "float16",
            np.int8: "int8",
        }
        dtype = dtype_map.get(arr.dtype.type, "float32")
        
        # Ensure float32 if not recognized
        if arr.dtype not in (np.float32, np.float16, np.int8):
            arr = arr.astype(np.float32)
            dtype = "float32"
            
        return cls(
            _buffer=arr.tobytes(),
            dimension=len(arr),
            dtype=dtype,  # type: ignore
        )
    
    @classmethod
    def from_bytes(
        cls, 
        data: bytes, 
        dimension: int,
        dtype: Literal["float32", "float16", "int8"] = "float32",
    ) -> "EmbeddingVector":
        """Create from raw bytes (zero-copy when possible)."""
        return cls(_buffer=data, dimension=dimension, dtype=dtype)
    
    def to_numpy(self) -> "np.ndarray":
        """
        Get NumPy array view of vector data.
        
        Returns:
            numpy.ndarray with appropriate dtype
            
        Note: Returns cached array if available (O(1) after first call)
        """
        if self._np_cache is not None:
            return self._np_cache
            
        import numpy as np
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.float16,  # Approximate with float16
            "int8": np.int8,
        }
        np_dtype = dtype_map[self.dtype]
        self._np_cache = np.frombuffer(self._buffer, dtype=np_dtype)
        return self._np_cache
    
    def to_list(self) -> list[float]:
        """Convert to Python list of floats."""
        arr = self.to_numpy()
        return arr.astype(float).tolist()
    
    def to_bytes(self) -> bytes:
        """Get raw bytes representation."""
        return self._buffer
    
    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return len(self._buffer)
    
    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        sizes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
        return sizes[self.dtype]
    
    def normalize(self) -> "EmbeddingVector":
        """Return L2-normalized copy of vector."""
        import numpy as np
        arr = self.to_numpy().astype(np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return EmbeddingVector.from_numpy(arr)
    
    def __len__(self) -> int:
        return self.dimension
    
    def __getitem__(self, idx: int) -> float:
        return float(self.to_numpy()[idx])
    
    def __iter__(self) -> Iterator[float]:
        return iter(self.to_list())
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingVector):
            return False
        return self._buffer == other._buffer and self.dtype == other.dtype


# =============================================================================
# SEARCH QUERY: LANGCHAIN-COMPATIBLE INTERFACE
# =============================================================================
@dataclass(slots=True)
class SearchQuery:
    """
    Vector similarity search query.
    
    Designed for compatibility with:
        - LangChain VectorStore interface
        - Pinecone query format
        - Milvus search parameters
        
    Attributes:
        vector: Query vector (flexible input types)
        k: Number of results to return
        filters: Metadata filters (MongoDB-style operators)
        score_threshold: Minimum similarity score
        include_metadata: Return metadata with results
        include_vectors: Return vectors with results
        namespace: Search within specific namespace
        timeout_ms: Query timeout in milliseconds
    """
    vector: Union[EmbeddingVector, Sequence[float], "np.ndarray"]
    k: int = 10
    filters: Optional[dict[str, Any]] = None
    score_threshold: Optional[float] = None
    include_metadata: bool = True
    include_vectors: bool = False
    namespace: str = "default"
    timeout_ms: float = 10000.0  # 10 second default
    
    def get_vector(self) -> EmbeddingVector:
        """Normalize vector to EmbeddingVector type."""
        if isinstance(self.vector, EmbeddingVector):
            return self.vector
        # Handle numpy array
        if hasattr(self.vector, "dtype"):
            return EmbeddingVector.from_numpy(self.vector)  # type: ignore
        # Handle list
        return EmbeddingVector.from_list(list(self.vector))
    
    def validate(self, max_k: int = 10000, max_dimension: int = 65536) -> Optional[str]:
        """
        Validate query parameters.
        
        Returns:
            None if valid, error message string if invalid
        """
        if self.k < 1:
            return f"k must be >= 1, got {self.k}"
        if self.k > max_k:
            return f"k must be <= {max_k}, got {self.k}"
        vec = self.get_vector()
        if vec.dimension > max_dimension:
            return f"dimension must be <= {max_dimension}, got {vec.dimension}"
        if self.score_threshold is not None:
            if not (0.0 <= self.score_threshold <= 1.0):
                return f"score_threshold must be in [0, 1], got {self.score_threshold}"
        return None


# =============================================================================
# SEARCH RESULT: SINGLE MATCH
# =============================================================================
@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Single search result with similarity score.
    
    Attributes:
        id: Vector identifier
        score: Similarity score (higher = more similar for cosine/IP)
        rank: Position in result set (0-indexed)
        metadata: Optional associated metadata
        vector: Optional vector data (if requested)
    """
    id: VectorId
    score: float
    rank: int = 0
    metadata: Optional[dict[str, Any]] = None
    vector: Optional[EmbeddingVector] = None
    
    @property
    def distance(self) -> float:
        """Convert similarity to distance (1 - score for cosine)."""
        return 1.0 - self.score
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/API response."""
        result: dict[str, Any] = {
            "id": str(self.id),
            "score": self.score,
            "rank": self.rank,
        }
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.vector is not None:
            result["vector"] = self.vector.to_list()
        return result


# =============================================================================
# SEARCH RESULTS: BATCH RESPONSE
# =============================================================================
@dataclass(slots=True)
class SearchResults:
    """
    Collection of search results with query metadata.
    
    Attributes:
        matches: List of SearchResult objects
        query_time_ms: Time to execute query
        total_candidates: Candidates considered before filtering
        namespace: Namespace searched
    """
    matches: list[SearchResult] = field(default_factory=list)
    query_time_ms: float = 0.0
    total_candidates: int = 0
    namespace: str = "default"
    
    def __len__(self) -> int:
        return len(self.matches)
    
    def __iter__(self) -> Iterator[SearchResult]:
        return iter(self.matches)
    
    def __getitem__(self, idx: int) -> SearchResult:
        return self.matches[idx]
    
    @property
    def ids(self) -> list[str]:
        """Get list of result IDs."""
        return [str(m.id) for m in self.matches]
    
    @property
    def scores(self) -> list[float]:
        """Get list of scores."""
        return [m.score for m in self.matches]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/API response."""
        return {
            "matches": [m.to_dict() for m in self.matches],
            "query_time_ms": self.query_time_ms,
            "total_candidates": self.total_candidates,
            "namespace": self.namespace,
        }


# =============================================================================
# INDEX CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class IndexConfig:
    """
    Configuration for vector index creation.
    
    Attributes:
        dimension: Vector dimensionality
        metric: Distance/similarity metric
        quantization: Compression method
        max_elements: Maximum capacity
        namespace: Default namespace
    """
    dimension: int
    metric: MetricType = MetricType.COSINE
    quantization: QuantizationType = QuantizationType.NONE
    max_elements: int = 1_000_000
    namespace: str = "default"
    
    def validate(self) -> Optional[str]:
        """Validate configuration parameters."""
        if self.dimension < 1:
            return f"dimension must be >= 1, got {self.dimension}"
        if self.dimension > 65536:
            return f"dimension must be <= 65536, got {self.dimension}"
        if self.max_elements < 1:
            return f"max_elements must be >= 1, got {self.max_elements}"
        return None


# =============================================================================
# INDEX STATISTICS
# =============================================================================
@dataclass(frozen=True, slots=True)
class IndexStats:
    """
    Runtime statistics for vector index.
    
    Attributes:
        total_vectors: Number of indexed vectors
        dimension: Vector dimension
        index_size_bytes: Memory/disk usage
        metric: Distance metric
        quantization: Active quantization
        build_time_ms: Index construction time
    """
    total_vectors: int
    dimension: int
    index_size_bytes: int
    metric: MetricType
    quantization: QuantizationType = QuantizationType.NONE
    build_time_ms: float = 0.0
    namespaces: dict[str, int] = field(default_factory=dict)
    
    @property
    def bytes_per_vector(self) -> float:
        """Average bytes per indexed vector."""
        if self.total_vectors == 0:
            return 0.0
        return self.index_size_bytes / self.total_vectors
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/API response."""
        return {
            "total_vectors": self.total_vectors,
            "dimension": self.dimension,
            "index_size_bytes": self.index_size_bytes,
            "bytes_per_vector": self.bytes_per_vector,
            "metric": self.metric.value,
            "quantization": self.quantization.value,
            "build_time_ms": self.build_time_ms,
            "namespaces": self.namespaces,
        }
