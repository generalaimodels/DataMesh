"""
Database Protocol Definitions: Unified Storage Abstraction Layer

Provides structural subtyping protocols (PEP 544) for pluggable database backends:
- DatabaseProtocol: Base async CRUD operations with Result monad
- TransactionProtocol: ACID transaction semantics with rollback
- ReplicationProtocol: CP/AP replication strategy interface
- ShardedProtocol: Consistent hashing for distributed shards

Design Principles:
    - Zero-exception control flow via Result[T, E] monad
    - Async-first for non-blocking I/O (io_uring/epoll compatible)
    - Protocol classes for structural subtyping (duck typing with type safety)
    - Memory-efficient: __slots__ where applicable

Complexity Analysis:
    - All protocol methods: O(1) dispatch overhead
    - Actual complexity determined by concrete implementations

Author: Planetary AI Systems
License: MIT
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import UUID

from datamesh.core.types import Result, Ok, Err, EntityId


# =============================================================================
# TYPE VARIABLES
# =============================================================================
K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type
T = TypeVar("T")  # Generic type


# =============================================================================
# CONSISTENCY LEVEL
# =============================================================================
class ConsistencyLevel(Enum):
    """
    Consistency levels for distributed operations.
    
    Ordered by increasing consistency guarantees (and latency).
    """
    ONE = auto()          # Single replica acknowledgment
    QUORUM = auto()       # Majority replicas (N/2 + 1)
    ALL = auto()          # All replicas must acknowledge
    LOCAL_QUORUM = auto() # Quorum within local datacenter
    EACH_QUORUM = auto()  # Quorum in each datacenter
    SERIAL = auto()       # Linearizable (Paxos/Raft)
    
    def min_replicas(self, replication_factor: int) -> int:
        """Minimum replicas required for this consistency level."""
        if self == ConsistencyLevel.ONE:
            return 1
        elif self == ConsistencyLevel.QUORUM or self == ConsistencyLevel.LOCAL_QUORUM:
            return (replication_factor // 2) + 1
        elif self == ConsistencyLevel.ALL:
            return replication_factor
        elif self == ConsistencyLevel.EACH_QUORUM:
            return (replication_factor // 2) + 1  # Per datacenter
        elif self == ConsistencyLevel.SERIAL:
            return (replication_factor // 2) + 1
        return 1


# =============================================================================
# ISOLATION LEVEL
# =============================================================================
class IsolationLevel(Enum):
    """
    Transaction isolation levels (SQL standard + Snapshot).
    
    Ordered by increasing isolation (and potential for blocking).
    """
    READ_UNCOMMITTED = auto()  # Dirty reads allowed
    READ_COMMITTED = auto()    # Only committed data visible
    REPEATABLE_READ = auto()   # Consistent snapshot within transaction
    SNAPSHOT = auto()          # MVCC-based snapshot isolation
    SERIALIZABLE = auto()      # Full serializability


# =============================================================================
# OPERATION TYPE
# =============================================================================
class OperationType(Enum):
    """Database operation types for logging and metrics."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SCAN = "scan"
    BATCH = "batch"


# =============================================================================
# OPERATION RESULT METADATA
# =============================================================================
@dataclass(frozen=True, slots=True)
class OperationMetadata:
    """
    Metadata returned with every database operation.
    
    Immutable to prevent accidental modification after return.
    Uses __slots__ for memory efficiency (~40% reduction).
    """
    operation: OperationType
    latency_ns: int              # Nanosecond precision
    affected_rows: int = 0
    version: int = 0             # For OCC
    consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    shard_id: Optional[str] = None
    replica_id: Optional[str] = None
    
    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return self.latency_ns / 1_000_000
    
    @property
    def latency_us(self) -> float:
        """Latency in microseconds."""
        return self.latency_ns / 1_000


# =============================================================================
# DATABASE PROTOCOL: BASE CRUD OPERATIONS
# =============================================================================
@runtime_checkable
class DatabaseProtocol(Protocol[K, V]):
    """
    Base protocol for async database operations.
    
    All methods return Result[T, str] for zero-exception control flow.
    Implementations should be fully async for non-blocking I/O.
    
    Type Parameters:
        K: Key type (typically str, UUID, or composite key)
        V: Value type (typically dataclass or dict)
    
    Example:
        class MyStore(DatabaseProtocol[str, UserData]):
            async def get(self, key: str) -> Result[UserData, str]:
                ...
    """
    
    @abstractmethod
    async def get(
        self,
        key: K,
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[tuple[V, OperationMetadata], str]:
        """
        Retrieve value by key.
        
        Args:
            key: Unique identifier for the record
            consistency: Read consistency level
            
        Returns:
            Ok((value, metadata)): Record found
            Err(message): Not found or error
            
        Complexity: O(1) for hash-based, O(log N) for tree-based
        """
        ...
    
    @abstractmethod
    async def put(
        self,
        key: K,
        value: V,
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Insert or update value by key (upsert semantics).
        
        Args:
            key: Unique identifier for the record
            value: Data to store
            consistency: Write consistency level
            
        Returns:
            Ok(metadata): Operation successful
            Err(message): Write failed
            
        Complexity: O(1) amortized for hash-based
        """
        ...
    
    @abstractmethod
    async def delete(
        self,
        key: K,
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Delete record by key.
        
        Args:
            key: Unique identifier for the record
            consistency: Write consistency level
            
        Returns:
            Ok(metadata): Deletion successful (or key not found)
            Err(message): Deletion failed
            
        Complexity: O(1) for hash-based
        """
        ...
    
    @abstractmethod
    async def exists(
        self,
        key: K,
        consistency: ConsistencyLevel = ConsistencyLevel.ONE,
    ) -> Result[bool, str]:
        """
        Check if key exists without fetching value.
        
        More efficient than get() when value is not needed.
        
        Complexity: O(1)
        """
        ...
    
    @abstractmethod
    async def scan(
        self,
        prefix: Optional[K] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Result[tuple[list[tuple[K, V]], Optional[str]], str]:
        """
        Scan records with optional prefix filter.
        
        Args:
            prefix: Key prefix for filtering (optional)
            limit: Maximum records to return
            cursor: Pagination cursor from previous scan
            
        Returns:
            Ok((records, next_cursor)): List of (key, value) pairs
            Err(message): Scan failed
            
        Complexity: O(limit) assuming efficient prefix scan
        """
        ...


# =============================================================================
# BATCH OPERATIONS PROTOCOL
# =============================================================================
@runtime_checkable
class BatchProtocol(Protocol[K, V]):
    """
    Protocol for batch operations (multi-get, multi-put).
    
    Batch operations amortize network overhead for bulk access patterns.
    Implementations should use pipelining where available.
    """
    
    @abstractmethod
    async def multi_get(
        self,
        keys: list[K],
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[dict[K, V], str]:
        """
        Retrieve multiple values by keys.
        
        Missing keys are omitted from result dict.
        
        Complexity: O(len(keys)) with pipelining
        """
        ...
    
    @abstractmethod
    async def multi_put(
        self,
        items: dict[K, V],
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Insert or update multiple values atomically.
        
        All-or-nothing semantics if supported by backend.
        
        Complexity: O(len(items)) with pipelining
        """
        ...
    
    @abstractmethod
    async def multi_delete(
        self,
        keys: list[K],
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Delete multiple records atomically.
        
        Complexity: O(len(keys))
        """
        ...


# =============================================================================
# TRANSACTION PROTOCOL
# =============================================================================
@runtime_checkable
class TransactionProtocol(Protocol[K, V]):
    """
    Protocol for ACID transaction support.
    
    Provides begin/commit/rollback semantics with configurable isolation.
    Implementations should support optimistic concurrency control (OCC).
    """
    
    @abstractmethod
    async def begin_transaction(
        self,
        isolation: IsolationLevel = IsolationLevel.SNAPSHOT,
        timeout_ms: int = 30000,
    ) -> Result["TransactionHandle[K, V]", str]:
        """
        Begin new transaction.
        
        Args:
            isolation: Transaction isolation level
            timeout_ms: Transaction timeout
            
        Returns:
            Ok(handle): Transaction handle for operations
            Err(message): Failed to begin transaction
        """
        ...
    
    @abstractmethod
    async def commit(
        self,
        handle: "TransactionHandle[K, V]",
    ) -> Result[OperationMetadata, str]:
        """
        Commit transaction.
        
        Returns:
            Ok(metadata): Commit successful
            Err(message): Commit failed (conflict, timeout, etc.)
        """
        ...
    
    @abstractmethod
    async def rollback(
        self,
        handle: "TransactionHandle[K, V]",
    ) -> Result[None, str]:
        """
        Rollback transaction.
        
        All changes made within transaction are discarded.
        """
        ...


# =============================================================================
# TRANSACTION HANDLE
# =============================================================================
@dataclass
class TransactionHandle(Generic[K, V]):
    """
    Handle for an active transaction.
    
    Provides scoped operations within transaction context.
    Supports context manager protocol for automatic cleanup.
    """
    transaction_id: UUID
    isolation: IsolationLevel
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_ms: int = 30000
    _operations: list[tuple[OperationType, K, Optional[V]]] = field(
        default_factory=list, repr=False
    )
    _committed: bool = field(default=False, repr=False)
    _rolled_back: bool = field(default=False, repr=False)
    
    @property
    def is_active(self) -> bool:
        """Check if transaction is still active."""
        if self._committed or self._rolled_back:
            return False
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds() * 1000
        return elapsed < self.timeout_ms
    
    def record_operation(
        self,
        op_type: OperationType,
        key: K,
        value: Optional[V] = None,
    ) -> None:
        """Record operation for potential rollback."""
        if not self.is_active:
            raise RuntimeError("Transaction is not active")
        self._operations.append((op_type, key, value))
    
    def mark_committed(self) -> None:
        """Mark transaction as committed."""
        self._committed = True
    
    def mark_rolled_back(self) -> None:
        """Mark transaction as rolled back."""
        self._rolled_back = True


# =============================================================================
# REPLICATION PROTOCOL
# =============================================================================
@runtime_checkable
class ReplicationProtocol(Protocol):
    """
    Protocol for replication management.
    
    Supports both CP (strong consistency) and AP (eventual consistency) modes.
    Implementations handle replica placement and failure recovery.
    """
    
    @abstractmethod
    async def get_replication_factor(self) -> int:
        """Get current replication factor."""
        ...
    
    @abstractmethod
    async def get_replica_status(
        self,
        shard_id: str,
    ) -> Result[list["ReplicaInfo"], str]:
        """
        Get status of all replicas for a shard.
        
        Returns list of replica info including lag and health.
        """
        ...
    
    @abstractmethod
    async def wait_for_replication(
        self,
        key: Any,
        target_replicas: int,
        timeout_ms: int = 5000,
    ) -> Result[bool, str]:
        """
        Wait for write to replicate to target number of replicas.
        
        Used for read-your-writes consistency.
        """
        ...


# =============================================================================
# REPLICA INFO
# =============================================================================
@dataclass(frozen=True, slots=True)
class ReplicaInfo:
    """Information about a single replica."""
    replica_id: str
    shard_id: str
    region: str
    is_leader: bool
    lag_ms: int
    last_heartbeat: datetime
    status: str  # "healthy", "degraded", "offline"


# =============================================================================
# SHARDED PROTOCOL
# =============================================================================
@runtime_checkable
class ShardedProtocol(Protocol[K]):
    """
    Protocol for sharded/partitioned databases.
    
    Uses consistent hashing for key-to-shard mapping.
    Supports virtual nodes for balanced distribution.
    """
    
    @abstractmethod
    def get_shard_for_key(self, key: K) -> str:
        """
        Determine shard ID for given key.
        
        Uses consistent hashing with virtual nodes.
        
        Complexity: O(log N) where N is virtual node count
        """
        ...
    
    @abstractmethod
    async def get_shard_info(
        self,
        shard_id: str,
    ) -> Result["ShardInfo", str]:
        """Get information about a specific shard."""
        ...
    
    @abstractmethod
    async def list_shards(self) -> Result[list["ShardInfo"], str]:
        """List all shards in the cluster."""
        ...
    
    @abstractmethod
    async def rebalance_shards(
        self,
        target_distribution: Optional[dict[str, float]] = None,
    ) -> Result[None, str]:
        """
        Trigger shard rebalancing.
        
        Used when nodes are added/removed from cluster.
        """
        ...


# =============================================================================
# SHARD INFO
# =============================================================================
@dataclass(frozen=True, slots=True)
class ShardInfo:
    """Information about a single shard."""
    shard_id: str
    key_range_start: str
    key_range_end: str
    leader_node: str
    replica_nodes: tuple[str, ...]
    record_count: int
    size_bytes: int
    status: str  # "active", "splitting", "merging", "draining"


# =============================================================================
# TIME-TO-LIVE PROTOCOL
# =============================================================================
@runtime_checkable
class TTLProtocol(Protocol[K, V]):
    """
    Protocol for TTL (Time-To-Live) support.
    
    Enables automatic expiration of records.
    Used for session caches and temporary data.
    """
    
    @abstractmethod
    async def put_with_ttl(
        self,
        key: K,
        value: V,
        ttl_seconds: int,
    ) -> Result[OperationMetadata, str]:
        """
        Insert value with automatic expiration.
        
        Record is automatically deleted after ttl_seconds.
        """
        ...
    
    @abstractmethod
    async def get_ttl(
        self,
        key: K,
    ) -> Result[Optional[int], str]:
        """
        Get remaining TTL for key in seconds.
        
        Returns None if key has no TTL or doesn't exist.
        """
        ...
    
    @abstractmethod
    async def extend_ttl(
        self,
        key: K,
        additional_seconds: int,
    ) -> Result[int, str]:
        """
        Extend TTL for existing key.
        
        Returns new TTL in seconds.
        """
        ...
    
    @abstractmethod
    async def remove_ttl(
        self,
        key: K,
    ) -> Result[None, str]:
        """Remove TTL from key (make persistent)."""
        ...


# =============================================================================
# STREAMING PROTOCOL
# =============================================================================
@runtime_checkable
class StreamingProtocol(Protocol[K, V]):
    """
    Protocol for streaming large result sets.
    
    Uses async generators for memory-efficient iteration.
    Supports backpressure via async iteration.
    """
    
    @abstractmethod
    def stream_scan(
        self,
        prefix: Optional[K] = None,
        batch_size: int = 100,
    ) -> AsyncIterator[tuple[K, V]]:
        """
        Stream records with optional prefix filter.
        
        Yields records one at a time for memory efficiency.
        Batch size controls internal buffering.
        
        Example:
            async for key, value in store.stream_scan():
                process(key, value)
        """
        ...
    
    @abstractmethod
    def stream_changes(
        self,
        since: Optional[datetime] = None,
    ) -> AsyncIterator[tuple[OperationType, K, Optional[V]]]:
        """
        Stream change events (CDC - Change Data Capture).
        
        Yields (operation, key, value) tuples.
        Value is None for DELETE operations.
        """
        ...


# =============================================================================
# VERSIONED PROTOCOL (OCC)
# =============================================================================
@runtime_checkable
class VersionedProtocol(Protocol[K, V]):
    """
    Protocol for versioned records (Optimistic Concurrency Control).
    
    Each record has a version number that increments on update.
    Compare-and-swap semantics prevent lost updates.
    """
    
    @abstractmethod
    async def get_with_version(
        self,
        key: K,
    ) -> Result[tuple[V, int], str]:
        """
        Get value with its current version.
        
        Returns (value, version) tuple.
        """
        ...
    
    @abstractmethod
    async def put_if_version(
        self,
        key: K,
        value: V,
        expected_version: int,
    ) -> Result[int, str]:
        """
        Update value only if version matches.
        
        Args:
            key: Record key
            value: New value
            expected_version: Version from previous read
            
        Returns:
            Ok(new_version): Update successful
            Err("version_mismatch"): Concurrent modification detected
        """
        ...
    
    @abstractmethod
    async def delete_if_version(
        self,
        key: K,
        expected_version: int,
    ) -> Result[None, str]:
        """
        Delete record only if version matches.
        
        Prevents accidental deletion of modified records.
        """
        ...


# =============================================================================
# COMPOSITE STORE INTERFACE (Type Alias)
# =============================================================================
# Note: In Python 3.12+, combining multiple Protocol classes with Generic
# type parameters requires special handling. For practical usage, implementations
# should inherit from the individual protocols they support.
#
# Example implementation:
#   class MyStore(
#       DatabaseProtocol[str, dict],
#       BatchProtocol[str, dict],
#       TTLProtocol[str, dict],
#       VersionedProtocol[str, dict],
#   ):
#       ...
