"""
In-Memory Database Backends: Development and Testing Implementations

Provides production-grade in-memory implementations of database protocols:
- InMemoryCPStore: Spanner-compatible with OCC and transactions
- InMemoryAPStore: Redis-compatible with TTL and pub/sub
- InMemoryObjectStore: S3-compatible blob storage

Design Principles:
    - Full protocol compliance for seamless production swap
    - Thread-safe operations via asyncio locks
    - Realistic latency simulation for performance testing
    - Memory-efficient with automatic cleanup

Performance Characteristics:
    - Get/Put/Delete: O(1) average case
    - Scan: O(k) where k is result limit
    - Transaction: O(n) where n is operation count

Author: Planetary AI Systems
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, TypeVar
from uuid import UUID, uuid4

# Type variables for generic classes
KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type

from datamesh.core.types import Result, Ok, Err, EntityId

from datamesh.storage.protocols import (
    ConsistencyLevel,
    IsolationLevel,
    OperationType,
    OperationMetadata,
    TransactionHandle,
    ReplicaInfo,
    ShardInfo,
    DatabaseProtocol,
    BatchProtocol,
    TransactionProtocol,
    TTLProtocol,
    VersionedProtocol,
    StreamingProtocol,
)


# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_SIMULATED_LATENCY_NS: int = 50_000  # 50 microseconds
MAX_SCAN_LIMIT: int = 10_000
DEFAULT_TTL_SECONDS: int = 86400  # 24 hours


# =============================================================================
# VERSIONED RECORD
# =============================================================================
@dataclass
class VersionedRecord:
    """
    Internal record with version tracking for OCC.
    
    Uses monotonic version counter for conflict detection.
    Stores creation and modification timestamps.
    """
    value: Any
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if record has expired based on TTL."""
        if self.ttl_expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.ttl_expires_at


# =============================================================================
# IN-MEMORY CP STORE (SPANNER-COMPATIBLE)
# =============================================================================
class InMemoryCPStore:
    """
    In-memory Control Plane store with strong consistency.
    
    Features:
        - Optimistic Concurrency Control (OCC)
        - Serializable transactions
        - Version tracking for all records
        - Simulated replication delay
    
    Thread Safety:
        All operations are protected by asyncio.Lock for
        concurrent access safety within async context.
    
    Example:
        store = InMemoryCPStore[str, UserData]()
        
        # Basic CRUD
        await store.put("user:123", user_data)
        result = await store.get("user:123")
        
        # Transaction
        txn = await store.begin_transaction()
        await store.put("user:123", updated_data, transaction=txn)
        await store.commit(txn)
    """
    
    __slots__ = (
        "_data",
        "_lock",
        "_transactions",
        "_version_counter",
        "_change_log",
        "_simulate_latency",
    )
    
    def __init__(
        self,
        simulate_latency: bool = False,
    ) -> None:
        """
        Initialize in-memory CP store.
        
        Args:
            simulate_latency: If True, add realistic delays
        """
        self._data: Dict[Any, VersionedRecord] = {}
        self._lock = asyncio.Lock()
        self._transactions: Dict[UUID, Dict[Any, VersionedRecord]] = {}
        self._version_counter: int = 0
        self._change_log: List[Tuple[OperationType, Any, Optional[Any], datetime]] = []
        self._simulate_latency = simulate_latency
    
    async def _simulate_network_latency(self) -> None:
        """Simulate network/storage latency for realistic testing."""
        if self._simulate_latency:
            await asyncio.sleep(DEFAULT_SIMULATED_LATENCY_NS / 1_000_000_000)
    
    def _measure_time(self) -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()
    
    # -------------------------------------------------------------------------
    # DatabaseProtocol Implementation
    # -------------------------------------------------------------------------
    
    async def get(
        self,
        key: Any,
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[Tuple[Any, OperationMetadata], str]:
        """
        Retrieve value by key with version metadata.
        
        Complexity: O(1) average case (hash lookup)
        """
        start_ns = self._measure_time()
        await self._simulate_network_latency()
        
        async with self._lock:
            record = self._data.get(key)
            
            if record is None:
                return Err(f"Key not found: {key}")
            
            if record.is_expired():
                del self._data[key]
                return Err(f"Key expired: {key}")
            
            metadata = OperationMetadata(
                operation=OperationType.READ,
                latency_ns=self._measure_time() - start_ns,
                affected_rows=1,
                version=record.version,
                consistency=consistency,
            )
            
            return Ok((record.value, metadata))
    
    async def put(
        self,
        key: Any,
        value: Any,
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Insert or update value (upsert semantics).
        
        Complexity: O(1) average case
        """
        start_ns = self._measure_time()
        await self._simulate_network_latency()
        
        async with self._lock:
            existing = self._data.get(key)
            
            if existing is not None:
                # Update existing record
                new_version = existing.version + 1
                self._data[key] = VersionedRecord(
                    value=value,
                    version=new_version,
                    created_at=existing.created_at,
                    updated_at=datetime.now(timezone.utc),
                )
                op_type = OperationType.UPDATE
            else:
                # Create new record
                self._version_counter += 1
                self._data[key] = VersionedRecord(
                    value=value,
                    version=1,
                )
                new_version = 1
                op_type = OperationType.CREATE
            
            # Log change for CDC
            self._change_log.append((
                op_type,
                key,
                value,
                datetime.now(timezone.utc),
            ))
            
            metadata = OperationMetadata(
                operation=op_type,
                latency_ns=self._measure_time() - start_ns,
                affected_rows=1,
                version=new_version,
                consistency=consistency,
            )
            
            return Ok(metadata)
    
    async def delete(
        self,
        key: Any,
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Delete record by key.
        
        Complexity: O(1)
        """
        start_ns = self._measure_time()
        await self._simulate_network_latency()
        
        async with self._lock:
            if key in self._data:
                del self._data[key]
                affected = 1
                
                # Log deletion
                self._change_log.append((
                    OperationType.DELETE,
                    key,
                    None,
                    datetime.now(timezone.utc),
                ))
            else:
                affected = 0
            
            metadata = OperationMetadata(
                operation=OperationType.DELETE,
                latency_ns=self._measure_time() - start_ns,
                affected_rows=affected,
                consistency=consistency,
            )
            
            return Ok(metadata)
    
    async def exists(
        self,
        key: Any,
        consistency: ConsistencyLevel = ConsistencyLevel.ONE,
    ) -> Result[bool, str]:
        """Check if key exists."""
        async with self._lock:
            record = self._data.get(key)
            if record is None:
                return Ok(False)
            if record.is_expired():
                del self._data[key]
                return Ok(False)
            return Ok(True)
    
    async def scan(
        self,
        prefix: Optional[Any] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Result[Tuple[List[Tuple[Any, Any]], Optional[str]], str]:
        """
        Scan records with optional prefix filter.
        
        Complexity: O(N) for full scan, O(k) for limited scan
        """
        start_ns = self._measure_time()
        await self._simulate_network_latency()
        
        limit = min(limit, MAX_SCAN_LIMIT)
        
        async with self._lock:
            # Get all keys, filter by prefix if specified
            keys = list(self._data.keys())
            
            if prefix is not None:
                prefix_str = str(prefix)
                keys = [k for k in keys if str(k).startswith(prefix_str)]
            
            # Sort for deterministic pagination
            keys.sort(key=str)
            
            # Apply cursor
            start_idx = 0
            if cursor is not None:
                try:
                    start_idx = int(cursor)
                except ValueError:
                    start_idx = 0
            
            # Slice results
            end_idx = start_idx + limit
            result_keys = keys[start_idx:end_idx]
            
            # Build results, filtering expired
            results: List[Tuple[Any, Any]] = []
            for key in result_keys:
                record = self._data.get(key)
                if record is not None and not record.is_expired():
                    results.append((key, record.value))
            
            # Determine next cursor
            next_cursor = str(end_idx) if end_idx < len(keys) else None
            
            return Ok((results, next_cursor))
    
    # -------------------------------------------------------------------------
    # BatchProtocol Implementation
    # -------------------------------------------------------------------------
    
    async def multi_get(
        self,
        keys: List[Any],
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[Dict[Any, Any], str]:
        """
        Retrieve multiple values by keys.
        
        Complexity: O(len(keys))
        """
        await self._simulate_network_latency()
        
        async with self._lock:
            results: Dict[Any, Any] = {}
            for key in keys:
                record = self._data.get(key)
                if record is not None and not record.is_expired():
                    results[key] = record.value
            
            return Ok(results)
    
    async def multi_put(
        self,
        items: Dict[Any, Any],
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Insert or update multiple values atomically.
        
        Complexity: O(len(items))
        """
        start_ns = self._measure_time()
        await self._simulate_network_latency()
        
        async with self._lock:
            for key, value in items.items():
                existing = self._data.get(key)
                if existing is not None:
                    self._data[key] = VersionedRecord(
                        value=value,
                        version=existing.version + 1,
                        created_at=existing.created_at,
                        updated_at=datetime.now(timezone.utc),
                    )
                else:
                    self._data[key] = VersionedRecord(value=value, version=1)
            
            metadata = OperationMetadata(
                operation=OperationType.BATCH,
                latency_ns=self._measure_time() - start_ns,
                affected_rows=len(items),
                consistency=consistency,
            )
            
            return Ok(metadata)
    
    async def multi_delete(
        self,
        keys: List[Any],
        consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
    ) -> Result[OperationMetadata, str]:
        """
        Delete multiple records atomically.
        
        Complexity: O(len(keys))
        """
        start_ns = self._measure_time()
        await self._simulate_network_latency()
        
        async with self._lock:
            deleted = 0
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    deleted += 1
            
            metadata = OperationMetadata(
                operation=OperationType.BATCH,
                latency_ns=self._measure_time() - start_ns,
                affected_rows=deleted,
                consistency=consistency,
            )
            
            return Ok(metadata)
    
    # -------------------------------------------------------------------------
    # TransactionProtocol Implementation
    # -------------------------------------------------------------------------
    
    async def begin_transaction(
        self,
        isolation: IsolationLevel = IsolationLevel.SNAPSHOT,
        timeout_ms: int = 30000,
    ) -> Result[TransactionHandle[Any, Any], str]:
        """
        Begin new transaction with snapshot isolation.
        
        Creates a snapshot of current data for read operations.
        """
        txn_id = uuid4()
        
        async with self._lock:
            # Create snapshot of current data
            snapshot = {k: VersionedRecord(
                value=v.value,
                version=v.version,
                created_at=v.created_at,
                updated_at=v.updated_at,
            ) for k, v in self._data.items() if not v.is_expired()}
            
            self._transactions[txn_id] = snapshot
        
        handle = TransactionHandle[Any, Any](
            transaction_id=txn_id,
            isolation=isolation,
            timeout_ms=timeout_ms,
        )
        
        return Ok(handle)
    
    async def commit(
        self,
        handle: TransactionHandle[Any, Any],
    ) -> Result[OperationMetadata, str]:
        """
        Commit transaction with OCC validation.
        
        Checks for version conflicts before applying changes.
        """
        start_ns = self._measure_time()
        
        if not handle.is_active:
            return Err("Transaction is not active")
        
        async with self._lock:
            snapshot = self._transactions.get(handle.transaction_id)
            if snapshot is None:
                return Err("Transaction not found")
            
            # Validate no conflicts (OCC)
            for op_type, key, value in handle._operations:
                if op_type in (OperationType.UPDATE, OperationType.DELETE):
                    current = self._data.get(key)
                    original = snapshot.get(key)
                    
                    if current is not None and original is not None:
                        if current.version != original.version:
                            # Version conflict detected
                            handle.mark_rolled_back()
                            del self._transactions[handle.transaction_id]
                            return Err(f"Version conflict on key: {key}")
            
            # Apply all operations
            for op_type, key, value in handle._operations:
                if op_type == OperationType.CREATE or op_type == OperationType.UPDATE:
                    existing = self._data.get(key)
                    if existing:
                        self._data[key] = VersionedRecord(
                            value=value,
                            version=existing.version + 1,
                            created_at=existing.created_at,
                        )
                    else:
                        self._data[key] = VersionedRecord(value=value, version=1)
                elif op_type == OperationType.DELETE:
                    if key in self._data:
                        del self._data[key]
            
            # Cleanup
            handle.mark_committed()
            del self._transactions[handle.transaction_id]
            
            metadata = OperationMetadata(
                operation=OperationType.BATCH,
                latency_ns=self._measure_time() - start_ns,
                affected_rows=len(handle._operations),
            )
            
            return Ok(metadata)
    
    async def rollback(
        self,
        handle: TransactionHandle[Any, Any],
    ) -> Result[None, str]:
        """Rollback transaction, discarding all changes."""
        async with self._lock:
            if handle.transaction_id in self._transactions:
                del self._transactions[handle.transaction_id]
            handle.mark_rolled_back()
            return Ok(None)
    
    # -------------------------------------------------------------------------
    # VersionedProtocol Implementation
    # -------------------------------------------------------------------------
    
    async def get_with_version(
        self,
        key: Any,
    ) -> Result[Tuple[Any, int], str]:
        """Get value with version number."""
        async with self._lock:
            record = self._data.get(key)
            if record is None or record.is_expired():
                return Err(f"Key not found: {key}")
            return Ok((record.value, record.version))
    
    async def put_if_version(
        self,
        key: Any,
        value: Any,
        expected_version: int,
    ) -> Result[int, str]:
        """
        Update only if version matches (CAS operation).
        
        Returns new version on success.
        """
        async with self._lock:
            record = self._data.get(key)
            
            if record is None:
                if expected_version != 0:
                    return Err("version_mismatch")
                self._data[key] = VersionedRecord(value=value, version=1)
                return Ok(1)
            
            if record.version != expected_version:
                return Err("version_mismatch")
            
            new_version = record.version + 1
            self._data[key] = VersionedRecord(
                value=value,
                version=new_version,
                created_at=record.created_at,
            )
            
            return Ok(new_version)
    
    async def delete_if_version(
        self,
        key: Any,
        expected_version: int,
    ) -> Result[None, str]:
        """Delete only if version matches."""
        async with self._lock:
            record = self._data.get(key)
            
            if record is None:
                return Err(f"Key not found: {key}")
            
            if record.version != expected_version:
                return Err("version_mismatch")
            
            del self._data[key]
            return Ok(None)
    
    # -------------------------------------------------------------------------
    # TTLProtocol Implementation
    # -------------------------------------------------------------------------
    
    async def put_with_ttl(
        self,
        key: Any,
        value: Any,
        ttl_seconds: int,
    ) -> Result[OperationMetadata, str]:
        """Insert value with automatic expiration."""
        start_ns = self._measure_time()
        
        async with self._lock:
            expires_at = datetime.now(timezone.utc)
            from datetime import timedelta
            expires_at = expires_at + timedelta(seconds=ttl_seconds)
            
            existing = self._data.get(key)
            if existing:
                self._data[key] = VersionedRecord(
                    value=value,
                    version=existing.version + 1,
                    created_at=existing.created_at,
                    ttl_expires_at=expires_at,
                )
            else:
                self._data[key] = VersionedRecord(
                    value=value,
                    version=1,
                    ttl_expires_at=expires_at,
                )
            
            metadata = OperationMetadata(
                operation=OperationType.CREATE,
                latency_ns=self._measure_time() - start_ns,
                affected_rows=1,
            )
            
            return Ok(metadata)
    
    async def get_ttl(
        self,
        key: Any,
    ) -> Result[Optional[int], str]:
        """Get remaining TTL in seconds."""
        async with self._lock:
            record = self._data.get(key)
            if record is None:
                return Ok(None)
            
            if record.ttl_expires_at is None:
                return Ok(None)
            
            remaining = (record.ttl_expires_at - datetime.now(timezone.utc)).total_seconds()
            return Ok(max(0, int(remaining)))
    
    async def extend_ttl(
        self,
        key: Any,
        additional_seconds: int,
    ) -> Result[int, str]:
        """Extend TTL by additional seconds."""
        async with self._lock:
            record = self._data.get(key)
            if record is None:
                return Err(f"Key not found: {key}")
            
            from datetime import timedelta
            if record.ttl_expires_at is None:
                new_expires = datetime.now(timezone.utc) + timedelta(seconds=additional_seconds)
            else:
                new_expires = record.ttl_expires_at + timedelta(seconds=additional_seconds)
            
            record.ttl_expires_at = new_expires
            remaining = (new_expires - datetime.now(timezone.utc)).total_seconds()
            return Ok(int(remaining))
    
    async def remove_ttl(
        self,
        key: Any,
    ) -> Result[None, str]:
        """Remove TTL from key."""
        async with self._lock:
            record = self._data.get(key)
            if record is None:
                return Err(f"Key not found: {key}")
            record.ttl_expires_at = None
            return Ok(None)
    
    # -------------------------------------------------------------------------
    # StreamingProtocol Implementation
    # -------------------------------------------------------------------------
    
    async def stream_scan(
        self,
        prefix: Optional[Any] = None,
        batch_size: int = 100,
    ) -> AsyncIterator[Tuple[Any, Any]]:
        """Stream records with memory-efficient iteration."""
        cursor: Optional[str] = None
        
        while True:
            result = await self.scan(prefix=prefix, limit=batch_size, cursor=cursor)
            if result.is_err():
                break
            
            records, next_cursor = result.unwrap()
            
            for key, value in records:
                yield (key, value)
            
            if next_cursor is None:
                break
            cursor = next_cursor
    
    async def stream_changes(
        self,
        since: Optional[datetime] = None,
    ) -> AsyncIterator[Tuple[OperationType, Any, Optional[Any]]]:
        """Stream change events (CDC)."""
        for op_type, key, value, timestamp in self._change_log:
            if since is None or timestamp > since:
                yield (op_type, key, value)
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    async def clear(self) -> None:
        """Clear all data (for testing)."""
        async with self._lock:
            self._data.clear()
            self._transactions.clear()
            self._change_log.clear()
    
    async def count(self) -> int:
        """Get total record count."""
        async with self._lock:
            return len(self._data)
    
    async def cleanup_expired(self) -> int:
        """Remove all expired records. Returns count removed."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._data.items()
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._data[key]
            return len(expired_keys)


# =============================================================================
# IN-MEMORY AP STORE (REDIS-COMPATIBLE)
# =============================================================================
class InMemoryAPStore(InMemoryCPStore):
    """
    In-memory Availability Plane store for hot data.
    
    Extends CPStore with:
        - LRU eviction when capacity exceeded
        - Pub/Sub for real-time events
        - Optimized for high throughput
    
    Example:
        store = InMemoryAPStore[str, SessionPayload](max_entries=10000)
        
        await store.put("session:123", payload)
        await store.subscribe("session:*", callback)
    """
    
    __slots__ = (
        "_max_entries",
        "_max_bytes",
        "_lru_order",
        "_subscribers",
        "_current_bytes",
    )
    
    def __init__(
        self,
        max_entries: int = 10000,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        simulate_latency: bool = False,
    ) -> None:
        super().__init__(simulate_latency=simulate_latency)
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._lru_order: OrderedDict[Any, None] = OrderedDict()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._current_bytes = 0
    
    async def put(
        self,
        key: Any,
        value: Any,
        consistency: ConsistencyLevel = ConsistencyLevel.ONE,
    ) -> Result[OperationMetadata, str]:
        """Put with LRU tracking and capacity management."""
        # Evict if needed
        async with self._lock:
            while len(self._data) >= self._max_entries:
                # Evict LRU entry
                if self._lru_order:
                    oldest_key = next(iter(self._lru_order))
                    del self._lru_order[oldest_key]
                    if oldest_key in self._data:
                        del self._data[oldest_key]
                else:
                    break
        
        result = await super().put(key, value, consistency)
        
        if result.is_ok():
            async with self._lock:
                # Update LRU order
                if key in self._lru_order:
                    self._lru_order.move_to_end(key)
                else:
                    self._lru_order[key] = None
                
                # Notify subscribers
                await self._notify_subscribers(key, value)
        
        return result
    
    async def get(
        self,
        key: Any,
        consistency: ConsistencyLevel = ConsistencyLevel.ONE,
    ) -> Result[Tuple[Any, OperationMetadata], str]:
        """Get with LRU update."""
        result = await super().get(key, consistency)
        
        if result.is_ok():
            async with self._lock:
                if key in self._lru_order:
                    self._lru_order.move_to_end(key)
        
        return result
    
    async def subscribe(
        self,
        pattern: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """Subscribe to key pattern changes."""
        async with self._lock:
            if pattern not in self._subscribers:
                self._subscribers[pattern] = []
            self._subscribers[pattern].append(callback)
    
    async def unsubscribe(
        self,
        pattern: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """Unsubscribe from pattern."""
        async with self._lock:
            if pattern in self._subscribers:
                self._subscribers[pattern] = [
                    cb for cb in self._subscribers[pattern]
                    if cb != callback
                ]
    
    async def _notify_subscribers(
        self,
        key: Any,
        value: Any,
    ) -> None:
        """Notify all matching subscribers."""
        key_str = str(key)
        for pattern, callbacks in self._subscribers.items():
            if self._matches_pattern(key_str, pattern):
                for callback in callbacks:
                    try:
                        callback(key, value)
                    except Exception:
                        pass  # Don't let subscriber errors affect store
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Simple wildcard pattern matching."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return key == pattern


# =============================================================================
# IN-MEMORY OBJECT STORE (S3-COMPATIBLE)
# =============================================================================
@dataclass
class ObjectMetadata:
    """Metadata for stored objects."""
    key: str
    size_bytes: int
    content_type: str
    etag: str
    created_at: datetime
    metadata: Dict[str, str] = field(default_factory=dict)


class InMemoryObjectStore:
    """
    In-memory object store for blob storage.
    
    S3-compatible interface for:
        - Large blob storage (history snapshots, artifacts)
        - Content-addressable storage
        - Immutable objects
    
    Example:
        store = InMemoryObjectStore()
        
        # Store object
        await store.put_object("snapshots/2024-01/entity-123.parquet", data)
        
        # Retrieve
        result = await store.get_object("snapshots/2024-01/entity-123.parquet")
    """
    
    __slots__ = ("_objects", "_metadata", "_lock")
    
    def __init__(self) -> None:
        self._objects: Dict[str, bytes] = {}
        self._metadata: Dict[str, ObjectMetadata] = {}
        self._lock = asyncio.Lock()
    
    async def put_object(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Result[ObjectMetadata, str]:
        """
        Store object with metadata.
        
        Computes ETag (MD5 hash) for content verification.
        """
        async with self._lock:
            # Compute ETag
            etag = hashlib.md5(data).hexdigest()
            
            obj_metadata = ObjectMetadata(
                key=key,
                size_bytes=len(data),
                content_type=content_type,
                etag=etag,
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
            
            self._objects[key] = data
            self._metadata[key] = obj_metadata
            
            return Ok(obj_metadata)
    
    async def get_object(
        self,
        key: str,
    ) -> Result[Tuple[bytes, ObjectMetadata], str]:
        """Retrieve object with metadata."""
        async with self._lock:
            if key not in self._objects:
                return Err(f"Object not found: {key}")
            
            return Ok((self._objects[key], self._metadata[key]))
    
    async def head_object(
        self,
        key: str,
    ) -> Result[ObjectMetadata, str]:
        """Get object metadata without data."""
        async with self._lock:
            if key not in self._metadata:
                return Err(f"Object not found: {key}")
            return Ok(self._metadata[key])
    
    async def delete_object(
        self,
        key: str,
    ) -> Result[None, str]:
        """Delete object."""
        async with self._lock:
            if key in self._objects:
                del self._objects[key]
                del self._metadata[key]
            return Ok(None)
    
    async def list_objects(
        self,
        prefix: str = "",
        limit: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> Result[Tuple[List[ObjectMetadata], Optional[str]], str]:
        """List objects with prefix filter."""
        async with self._lock:
            keys = sorted([k for k in self._metadata.keys() if k.startswith(prefix)])
            
            start_idx = 0
            if continuation_token:
                try:
                    start_idx = int(continuation_token)
                except ValueError:
                    start_idx = 0
            
            end_idx = start_idx + limit
            result_keys = keys[start_idx:end_idx]
            
            results = [self._metadata[k] for k in result_keys]
            next_token = str(end_idx) if end_idx < len(keys) else None
            
            return Ok((results, next_token))
    
    async def copy_object(
        self,
        source_key: str,
        dest_key: str,
    ) -> Result[ObjectMetadata, str]:
        """Copy object to new key."""
        async with self._lock:
            if source_key not in self._objects:
                return Err(f"Source not found: {source_key}")
            
            data = self._objects[source_key]
            source_meta = self._metadata[source_key]
            
            new_meta = ObjectMetadata(
                key=dest_key,
                size_bytes=source_meta.size_bytes,
                content_type=source_meta.content_type,
                etag=source_meta.etag,
                created_at=datetime.now(timezone.utc),
                metadata=dict(source_meta.metadata),
            )
            
            self._objects[dest_key] = data
            self._metadata[dest_key] = new_meta
            
            return Ok(new_meta)
    
    async def get_object_range(
        self,
        key: str,
        start_byte: int,
        end_byte: int,
    ) -> Result[bytes, str]:
        """Get byte range from object (for streaming)."""
        async with self._lock:
            if key not in self._objects:
                return Err(f"Object not found: {key}")
            
            data = self._objects[key]
            return Ok(data[start_byte:end_byte])
    
    async def count(self) -> int:
        """Get total object count."""
        async with self._lock:
            return len(self._objects)
    
    async def total_size_bytes(self) -> int:
        """Get total size of all objects."""
        async with self._lock:
            return sum(len(data) for data in self._objects.values())
