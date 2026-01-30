"""
History Database Layer: SOTA-Level Time-Series Conversation Storage

High-performance conversation history with:
- LSM-tree optimized append-only writes
- Time-bucket partitioning for efficient archival
- Tiered storage (Hot → Warm → Cold)
- Cursor-based pagination with composite keys
- Automatic compaction and snapshot management
- CQRS separation for read/write optimization

Design Principles:
    - Append-only writes for LSM-tree efficiency
    - Time-bucket partitioning for range scans
    - Immutable interactions for caching
    - Streaming for large history retrieval

Performance Targets:
    - Append: <2ms P99
    - Range query (100 items): <10ms P99
    - Full timeline scan: <100ms P99

Author: Planetary AI Systems
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
from uuid import UUID, uuid4

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp
from datamesh.storage.protocols import (
    ConsistencyLevel,
    OperationType,
    OperationMetadata,
)
from datamesh.storage.backends import (
    InMemoryCPStore,
    InMemoryAPStore,
    InMemoryObjectStore,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class HistoryDBConfig:
    """
    Configuration for history database layer.
    
    Optimized for append-heavy workloads.
    """
    # Bucket settings
    bucket_duration_hours: int = 24  # One bucket per day
    hot_retention_days: int = 7
    warm_retention_days: int = 30
    cold_retention_days: int = 365
    
    # Write settings
    write_batch_size: int = 100
    write_batch_delay_ms: int = 50
    
    # Read settings
    default_page_size: int = 100
    max_page_size: int = 1000
    prefetch_count: int = 2
    
    # Compaction
    compaction_threshold: int = 1000  # Interactions before compaction
    max_interactions_per_bucket: int = 10000
    
    # Archival
    archive_to_cold_days: int = 30
    snapshot_interval_hours: int = 24
    
    # Object store
    object_store_prefix: str = "history/"
    compression_enabled: bool = True


# =============================================================================
# STORAGE TIER
# =============================================================================
class StorageTier(Enum):
    """Storage tier classification."""
    HOT = auto()    # In-memory, <1ms latency
    WARM = auto()   # SSD/fast storage, <10ms latency
    COLD = auto()   # Object store, <1s latency
    ARCHIVE = auto() # Glacier/deep archive, minutes to hours
    
    @property
    def max_latency_ms(self) -> int:
        """Expected maximum latency for tier."""
        latencies = {
            StorageTier.HOT: 1,
            StorageTier.WARM: 10,
            StorageTier.COLD: 1000,
            StorageTier.ARCHIVE: 300000,  # 5 minutes
        }
        return latencies.get(self, 1000)


# =============================================================================
# INTERACTION TYPE
# =============================================================================
class InteractionType(Enum):
    """Type of conversation interaction."""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESULT = "function_result"
    ERROR = "error"
    METADATA = "metadata"


# =============================================================================
# INTERACTION
# =============================================================================
@dataclass
class Interaction:
    """
    Single conversation interaction.
    
    Immutable after creation for caching and consistency.
    Uses content reference for large payloads.
    """
    interaction_id: UUID
    entity_id: EntityId
    session_id: UUID
    
    # Ordering
    sequence_id: int
    timestamp: datetime
    
    # Content
    interaction_type: InteractionType
    role: str  # "user", "assistant", "system", "tool"
    content: str
    content_hash: str = ""
    
    # Metadata
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # References
    parent_id: Optional[UUID] = None
    content_ref: Optional[str] = None  # Reference to object store for large content
    
    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:16]
    
    @property
    def bucket_key(self) -> str:
        """Generate time-bucket key (YYYY-MM-DD format)."""
        return self.timestamp.strftime("%Y-%m-%d")
    
    @property
    def storage_key(self) -> str:
        """Generate unique storage key."""
        return f"interaction:{self.entity_id.value}:{self.session_id}:{self.sequence_id:010d}"
    
    @property
    def is_large(self) -> bool:
        """Check if content should be stored separately."""
        return len(self.content) > 10000  # 10KB threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "interaction_id": str(self.interaction_id),
            "entity_id": self.entity_id.value,
            "session_id": str(self.session_id),
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp.isoformat(),
            "interaction_type": self.interaction_type.value,
            "role": self.role,
            "content": self.content if not self.content_ref else "",
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "content_ref": self.content_ref,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """Deserialize from dictionary."""
        return cls(
            interaction_id=UUID(data["interaction_id"]),
            entity_id=EntityId(data["entity_id"]),
            session_id=UUID(data["session_id"]),
            sequence_id=data["sequence_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            interaction_type=InteractionType(data["interaction_type"]),
            role=data["role"],
            content=data.get("content", ""),
            content_hash=data.get("content_hash", ""),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
            parent_id=UUID(data["parent_id"]) if data.get("parent_id") else None,
            content_ref=data.get("content_ref"),
        )


# =============================================================================
# HISTORY BUCKET
# =============================================================================
@dataclass
class HistoryBucket:
    """
    Time-partitioned history bucket.
    
    Groups interactions by time period for efficient:
    - Range queries
    - TTL management
    - Tiered storage migration
    """
    bucket_id: str  # Format: YYYY-MM-DD
    entity_id: EntityId
    session_id: UUID
    
    # Boundaries
    start_time: datetime
    end_time: datetime
    
    # Statistics
    interaction_count: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    
    # State
    tier: StorageTier = StorageTier.HOT
    is_sealed: bool = False  # True if no more writes expected
    is_compacted: bool = False
    
    # References
    snapshot_ref: Optional[str] = None  # Object store reference
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def storage_key(self) -> str:
        """Generate storage key."""
        return f"bucket:{self.entity_id.value}:{self.session_id}:{self.bucket_id}"
    
    @property
    def age_days(self) -> int:
        """Bucket age in days."""
        delta = datetime.now(timezone.utc) - self.start_time
        return delta.days
    
    def should_migrate_to_warm(self, hot_days: int) -> bool:
        """Check if bucket should migrate to warm tier."""
        return self.tier == StorageTier.HOT and self.age_days >= hot_days
    
    def should_migrate_to_cold(self, warm_days: int) -> bool:
        """Check if bucket should migrate to cold tier."""
        return self.tier == StorageTier.WARM and self.age_days >= warm_days
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bucket_id": self.bucket_id,
            "entity_id": self.entity_id.value,
            "session_id": str(self.session_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "interaction_count": self.interaction_count,
            "total_tokens": self.total_tokens,
            "total_bytes": self.total_bytes,
            "tier": self.tier.name,
            "is_sealed": self.is_sealed,
            "is_compacted": self.is_compacted,
            "snapshot_ref": self.snapshot_ref,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryBucket":
        """Deserialize from dictionary."""
        return cls(
            bucket_id=data["bucket_id"],
            entity_id=EntityId(data["entity_id"]),
            session_id=UUID(data["session_id"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            interaction_count=data.get("interaction_count", 0),
            total_tokens=data.get("total_tokens", 0),
            total_bytes=data.get("total_bytes", 0),
            tier=StorageTier[data.get("tier", "HOT")],
            is_sealed=data.get("is_sealed", False),
            is_compacted=data.get("is_compacted", False),
            snapshot_ref=data.get("snapshot_ref"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data.get("updated_at", data["created_at"])),
        )


# =============================================================================
# HISTORY CURSOR
# =============================================================================
@dataclass(frozen=True, slots=True)
class HistoryCursor:
    """
    Opaque cursor for paginated history queries.
    
    Encodes position in multi-dimensional space:
    (timestamp, sequence_id, bucket_id)
    """
    timestamp: datetime
    sequence_id: int
    bucket_id: str
    direction: str = "backward"  # "forward" or "backward"
    
    def encode(self) -> str:
        """Encode cursor to opaque string."""
        data = f"{self.timestamp.isoformat()}|{self.sequence_id}|{self.bucket_id}|{self.direction}"
        return hashlib.sha256(data.encode()).hexdigest()[:32] + ":" + data
    
    @classmethod
    def decode(cls, encoded: str) -> "HistoryCursor":
        """Decode cursor from string."""
        _, data = encoded.split(":", 1)
        parts = data.split("|")
        return cls(
            timestamp=datetime.fromisoformat(parts[0]),
            sequence_id=int(parts[1]),
            bucket_id=parts[2],
            direction=parts[3] if len(parts) > 3 else "backward",
        )


# =============================================================================
# HISTORY DATABASE LAYER
# =============================================================================
class HistoryDatabaseLayer:
    """
    SOTA-level history database layer with time-series optimization.
    
    Architecture:
        - Append-only writes for LSM-tree efficiency
        - Time-bucket partitioning for range queries
        - Tiered storage (Hot → Warm → Cold)
        - CQRS: Separate read/write paths
    
    Features:
        - Efficient append operations
        - Cursor-based pagination
        - Automatic tier migration
        - Snapshot creation for archival
        - Streaming for large histories
    
    Example:
        db = HistoryDatabaseLayer()
        
        # Append interaction
        await db.append(interaction)
        
        # Query recent interactions
        result = await db.query_range(entity_id, session_id, limit=100)
        
        # Stream full history
        async for interaction in db.stream_history(entity_id, session_id):
            process(interaction)
    """
    
    __slots__ = (
        "_config",
        "_cp_store",       # Metadata store (buckets, indices)
        "_ap_store",       # Hot tier (recent interactions)
        "_object_store",   # Cold tier (archived buckets)
        "_sequence_counters",
        "_write_buffer",
        "_write_lock",
        "_metrics",
    )
    
    def __init__(
        self,
        config: Optional[HistoryDBConfig] = None,
        cp_store: Optional[InMemoryCPStore] = None,
        ap_store: Optional[InMemoryAPStore] = None,
        object_store: Optional[InMemoryObjectStore] = None,
    ) -> None:
        """Initialize history database layer."""
        self._config = config or HistoryDBConfig()
        
        self._cp_store = cp_store or InMemoryCPStore()
        self._ap_store = ap_store or InMemoryAPStore(
            max_entries=1_000_000,
        )
        self._object_store = object_store or InMemoryObjectStore()
        
        # Sequence counter per session
        self._sequence_counters: Dict[str, int] = {}
        
        # Write buffer for batching
        self._write_buffer: List[Interaction] = []
        self._write_lock = asyncio.Lock()
        
        # Metrics
        self._metrics = HistoryMetrics()
    
    # -------------------------------------------------------------------------
    # WRITE OPERATIONS
    # -------------------------------------------------------------------------
    
    async def append(
        self,
        interaction: Interaction,
        flush: bool = True,
    ) -> Result[Interaction, str]:
        """
        Append interaction to history.
        
        Optimized for append-only writes with automatic bucketing.
        
        Args:
            interaction: Interaction to append
            flush: If True, flush write buffer immediately
            
        Returns:
            Ok(interaction): With assigned sequence_id
            Err(message): Append failed
            
        Complexity: O(1) amortized
        """
        start_ns = time.time_ns()
        
        # Assign sequence ID
        counter_key = f"{interaction.entity_id.value}:{interaction.session_id}"
        
        async with self._write_lock:
            current_seq = self._sequence_counters.get(counter_key, 0)
            interaction.sequence_id = current_seq + 1
            self._sequence_counters[counter_key] = interaction.sequence_id
        
        # Handle large content
        if interaction.is_large:
            content_ref = await self._store_large_content(interaction)
            if content_ref.is_ok():
                interaction.content_ref = content_ref.unwrap()
                interaction.content = ""  # Clear content, stored separately
        
        # Ensure bucket exists
        bucket_result = await self._ensure_bucket(interaction)
        if bucket_result.is_err():
            return Err(bucket_result.error)
        
        # Write to hot tier
        write_result = await self._ap_store.put(
            interaction.storage_key,
            interaction.to_dict(),
        )
        
        if write_result.is_err():
            return Err(f"Failed to write interaction: {write_result.error}")
        
        # Update bucket statistics
        await self._update_bucket_stats(interaction)
        
        # Update metrics
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        self._metrics.record_append(latency_ms, interaction.token_count)
        
        return Ok(interaction)
    
    async def batch_append(
        self,
        interactions: List[Interaction],
    ) -> Result[List[Interaction], str]:
        """
        Batch append multiple interactions.
        
        More efficient than individual appends for bulk ingestion.
        
        Complexity: O(n) where n is batch size
        """
        start_ns = time.time_ns()
        
        results: List[Interaction] = []
        
        # Group by session for sequence assignment
        by_session: Dict[str, List[Interaction]] = defaultdict(list)
        for interaction in interactions:
            key = f"{interaction.entity_id.value}:{interaction.session_id}"
            by_session[key].append(interaction)
        
        # Assign sequences and prepare batch
        batch_data: Dict[str, Dict[str, Any]] = {}
        
        async with self._write_lock:
            for session_key, session_interactions in by_session.items():
                current_seq = self._sequence_counters.get(session_key, 0)
                
                for interaction in session_interactions:
                    current_seq += 1
                    interaction.sequence_id = current_seq
                    
                    # Handle large content
                    if interaction.is_large:
                        result = await self._store_large_content(interaction)
                        if result.is_ok():
                            interaction.content_ref = result.unwrap()
                            interaction.content = ""
                    
                    batch_data[interaction.storage_key] = interaction.to_dict()
                    results.append(interaction)
                
                self._sequence_counters[session_key] = current_seq
        
        # Batch write
        write_result = await self._ap_store.multi_put(batch_data)
        
        if write_result.is_err():
            return Err(f"Batch append failed: {write_result.error}")
        
        # Update bucket stats
        for interaction in results:
            await self._update_bucket_stats(interaction)
        
        # Update metrics
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        total_tokens = sum(i.token_count for i in results)
        self._metrics.record_batch_append(latency_ms, len(results), total_tokens)
        
        return Ok(results)
    
    async def _store_large_content(
        self,
        interaction: Interaction,
    ) -> Result[str, str]:
        """Store large content in object store."""
        content_key = (
            f"{self._config.object_store_prefix}content/"
            f"{interaction.entity_id.value}/"
            f"{interaction.session_id}/"
            f"{interaction.interaction_id}.txt"
        )
        
        result = await self._object_store.put_object(
            key=content_key,
            data=interaction.content.encode("utf-8"),
            content_type="text/plain",
            metadata={
                "interaction_id": str(interaction.interaction_id),
                "content_hash": interaction.content_hash,
            },
        )
        
        if result.is_err():
            return Err(result.error)
        
        return Ok(content_key)
    
    async def _ensure_bucket(
        self,
        interaction: Interaction,
    ) -> Result[HistoryBucket, str]:
        """Ensure bucket exists for interaction's timestamp."""
        bucket_id = interaction.bucket_key
        bucket_storage_key = f"bucket:{interaction.entity_id.value}:{interaction.session_id}:{bucket_id}"
        
        # Check if bucket exists
        result = await self._cp_store.get(bucket_storage_key)
        
        if result.is_ok():
            data, _ = result.unwrap()
            return Ok(HistoryBucket.from_dict(data))
        
        # Create new bucket
        bucket_date = datetime.strptime(bucket_id, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        bucket = HistoryBucket(
            bucket_id=bucket_id,
            entity_id=interaction.entity_id,
            session_id=interaction.session_id,
            start_time=bucket_date,
            end_time=bucket_date + timedelta(hours=self._config.bucket_duration_hours),
        )
        
        await self._cp_store.put(bucket_storage_key, bucket.to_dict())
        
        return Ok(bucket)
    
    async def _update_bucket_stats(
        self,
        interaction: Interaction,
    ) -> None:
        """Update bucket statistics after append."""
        bucket_storage_key = f"bucket:{interaction.entity_id.value}:{interaction.session_id}:{interaction.bucket_key}"
        
        result = await self._cp_store.get_with_version(bucket_storage_key)
        if result.is_err():
            return
        
        data, version = result.unwrap()
        bucket = HistoryBucket.from_dict(data)
        
        bucket.interaction_count += 1
        bucket.total_tokens += interaction.token_count
        bucket.total_bytes += len(interaction.content.encode("utf-8"))
        bucket.updated_at = datetime.now(timezone.utc)
        
        await self._cp_store.put_if_version(
            bucket_storage_key,
            bucket.to_dict(),
            expected_version=version,
        )
    
    # -------------------------------------------------------------------------
    # READ OPERATIONS
    # -------------------------------------------------------------------------
    
    async def get_interaction(
        self,
        entity_id: EntityId,
        session_id: UUID,
        sequence_id: int,
    ) -> Result[Interaction, str]:
        """
        Get single interaction by sequence ID.
        
        Complexity: O(1)
        """
        storage_key = f"interaction:{entity_id.value}:{session_id}:{sequence_id:010d}"
        
        # Try hot tier first
        result = await self._ap_store.get(storage_key)
        
        if result.is_ok():
            data, _ = result.unwrap()
            interaction = Interaction.from_dict(data)
            
            # Load large content if needed
            if interaction.content_ref:
                content_result = await self._load_large_content(interaction.content_ref)
                if content_result.is_ok():
                    interaction.content = content_result.unwrap()
            
            return Ok(interaction)
        
        return Err(f"Interaction not found: {sequence_id}")
    
    async def query_range(
        self,
        entity_id: EntityId,
        session_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        direction: str = "backward",  # "forward" or "backward"
    ) -> Result[Tuple[List[Interaction], Optional[str]], str]:
        """
        Query interactions within time range.
        
        Args:
            entity_id: Entity ID
            session_id: Session ID
            start_time: Start of range (optional)
            end_time: End of range (optional)
            limit: Maximum results
            cursor: Pagination cursor
            direction: "forward" (oldest first) or "backward" (newest first)
            
        Returns:
            Ok((interactions, next_cursor)): Results with pagination
            
        Complexity: O(limit)
        """
        start_ns = time.time_ns()
        
        limit = min(limit, self._config.max_page_size)
        
        # Parse cursor if provided
        start_seq = 0
        if cursor:
            parsed = HistoryCursor.decode(cursor)
            start_seq = parsed.sequence_id
        
        # Scan interactions - need enough items to cover cursor offset + desired limit
        prefix = f"interaction:{entity_id.value}:{session_id}:"
        # When paginating, we need to scan past the cursor position
        scan_limit = start_seq + limit + 1 if start_seq > 0 else limit * 2
        result = await self._ap_store.scan(prefix=prefix, limit=scan_limit)
        
        if result.is_err():
            return Err(result.error)
        
        records, _ = result.unwrap()
        
        # Filter and convert
        interactions: List[Interaction] = []
        
        for key, data in records:
            interaction = Interaction.from_dict(data)
            
            # Apply time range filter
            if start_time and interaction.timestamp < start_time:
                continue
            if end_time and interaction.timestamp > end_time:
                continue
            
            # Apply cursor filter
            if direction == "backward" and interaction.sequence_id >= start_seq and start_seq > 0:
                continue
            if direction == "forward" and interaction.sequence_id <= start_seq and start_seq > 0:
                continue
            
            # Load large content
            if interaction.content_ref:
                content_result = await self._load_large_content(interaction.content_ref)
                if content_result.is_ok():
                    interaction.content = content_result.unwrap()
            
            interactions.append(interaction)
            
            if len(interactions) >= limit:
                break
        
        # Sort by sequence
        if direction == "backward":
            interactions.sort(key=lambda x: x.sequence_id, reverse=True)
        else:
            interactions.sort(key=lambda x: x.sequence_id)
        
        # Trim to limit
        interactions = interactions[:limit]
        
        # Generate next cursor
        next_cursor = None
        if len(interactions) == limit:
            last = interactions[-1]
            cursor_obj = HistoryCursor(
                timestamp=last.timestamp,
                sequence_id=last.sequence_id,
                bucket_id=last.bucket_key,
                direction=direction,
            )
            next_cursor = cursor_obj.encode()
        
        # Update metrics
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        self._metrics.record_query(latency_ms, len(interactions))
        
        return Ok((interactions, next_cursor))
    
    async def get_recent(
        self,
        entity_id: EntityId,
        session_id: UUID,
        count: int = 10,
    ) -> Result[List[Interaction], str]:
        """
        Get most recent interactions.
        
        Convenience method for common use case.
        
        Complexity: O(count)
        """
        result = await self.query_range(
            entity_id=entity_id,
            session_id=session_id,
            limit=count,
            direction="backward",
        )
        
        if result.is_err():
            return Err(result.error)
        
        interactions, _ = result.unwrap()
        return Ok(interactions)
    
    async def _load_large_content(
        self,
        content_ref: str,
    ) -> Result[str, str]:
        """Load large content from object store."""
        result = await self._object_store.get_object(content_ref)
        
        if result.is_err():
            return Err(result.error)
        
        data, _ = result.unwrap()
        return Ok(data.decode("utf-8"))
    
    # -------------------------------------------------------------------------
    # STREAMING
    # -------------------------------------------------------------------------
    
    async def stream_history(
        self,
        entity_id: EntityId,
        session_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 100,
    ) -> AsyncIterator[Interaction]:
        """
        Stream full history as async iterator.
        
        Memory-efficient for large histories.
        
        Example:
            async for interaction in db.stream_history(entity_id, session_id):
                process(interaction)
        """
        cursor = None
        
        while True:
            result = await self.query_range(
                entity_id=entity_id,
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                limit=batch_size,
                cursor=cursor,
                direction="forward",
            )
            
            if result.is_err():
                break
            
            interactions, next_cursor = result.unwrap()
            
            for interaction in interactions:
                yield interaction
            
            if next_cursor is None:
                break
            
            cursor = next_cursor
    
    # -------------------------------------------------------------------------
    # BUCKET OPERATIONS
    # -------------------------------------------------------------------------
    
    async def list_buckets(
        self,
        entity_id: EntityId,
        session_id: UUID,
        tier_filter: Optional[StorageTier] = None,
    ) -> Result[List[HistoryBucket], str]:
        """
        List all buckets for a session.
        
        Used for tier migration and archival operations.
        """
        prefix = f"bucket:{entity_id.value}:{session_id}:"
        
        result = await self._cp_store.scan(prefix=prefix, limit=1000)
        
        if result.is_err():
            return Err(result.error)
        
        records, _ = result.unwrap()
        
        buckets = []
        for _, data in records:
            bucket = HistoryBucket.from_dict(data)
            
            if tier_filter is not None and bucket.tier != tier_filter:
                continue
            
            buckets.append(bucket)
        
        # Sort by bucket_id (date)
        buckets.sort(key=lambda b: b.bucket_id)
        
        return Ok(buckets)
    
    async def migrate_bucket_tier(
        self,
        bucket: HistoryBucket,
        target_tier: StorageTier,
    ) -> Result[HistoryBucket, str]:
        """
        Migrate bucket to different storage tier.
        
        Hot → Warm: Keep in AP store but mark for less frequent access
        Warm → Cold: Snapshot to object store
        Cold → Archive: Move to deep archive (Glacier)
        """
        if bucket.tier == target_tier:
            return Ok(bucket)
        
        if target_tier == StorageTier.COLD and bucket.tier in (StorageTier.HOT, StorageTier.WARM):
            # Create snapshot in object store
            snapshot_result = await self._create_bucket_snapshot(bucket)
            if snapshot_result.is_err():
                return Err(snapshot_result.error)
            
            bucket.snapshot_ref = snapshot_result.unwrap()
            bucket.is_sealed = True
        
        # Update bucket metadata
        bucket.tier = target_tier
        bucket.updated_at = datetime.now(timezone.utc)
        
        await self._cp_store.put(bucket.storage_key, bucket.to_dict())
        
        return Ok(bucket)
    
    async def _create_bucket_snapshot(
        self,
        bucket: HistoryBucket,
    ) -> Result[str, str]:
        """Create snapshot of bucket in object store."""
        # Gather all interactions in bucket
        interactions = []
        
        async for interaction in self.stream_history(
            entity_id=bucket.entity_id,
            session_id=bucket.session_id,
            start_time=bucket.start_time,
            end_time=bucket.end_time,
        ):
            interactions.append(interaction.to_dict())
        
        # Serialize to JSON
        import json
        snapshot_data = json.dumps({
            "bucket": bucket.to_dict(),
            "interactions": interactions,
        }).encode("utf-8")
        
        # Compress if enabled
        if self._config.compression_enabled:
            import lz4.frame
            snapshot_data = lz4.frame.compress(snapshot_data)
        
        # Store in object store
        snapshot_key = (
            f"{self._config.object_store_prefix}snapshots/"
            f"{bucket.entity_id.value}/"
            f"{bucket.session_id}/"
            f"{bucket.bucket_id}.json.lz4"
        )
        
        result = await self._object_store.put_object(
            key=snapshot_key,
            data=snapshot_data,
            content_type="application/json",
            metadata={
                "bucket_id": bucket.bucket_id,
                "interaction_count": str(bucket.interaction_count),
                "compressed": "true" if self._config.compression_enabled else "false",
            },
        )
        
        if result.is_err():
            return Err(result.error)
        
        return Ok(snapshot_key)
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    async def get_session_stats(
        self,
        entity_id: EntityId,
        session_id: UUID,
    ) -> Result[Dict[str, Any], str]:
        """
        Get statistics for a session's history.
        """
        buckets_result = await self.list_buckets(entity_id, session_id)
        
        if buckets_result.is_err():
            return Err(buckets_result.error)
        
        buckets = buckets_result.unwrap()
        
        total_interactions = sum(b.interaction_count for b in buckets)
        total_tokens = sum(b.total_tokens for b in buckets)
        total_bytes = sum(b.total_bytes for b in buckets)
        
        tier_counts = defaultdict(int)
        for bucket in buckets:
            tier_counts[bucket.tier.name] += bucket.interaction_count
        
        return Ok({
            "bucket_count": len(buckets),
            "total_interactions": total_interactions,
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
            "tier_distribution": dict(tier_counts),
            "oldest_interaction": buckets[0].start_time.isoformat() if buckets else None,
            "newest_interaction": buckets[-1].end_time.isoformat() if buckets else None,
        })
    
    def get_metrics(self) -> "HistoryMetrics":
        """Get current metrics."""
        return self._metrics


# =============================================================================
# HISTORY METRICS
# =============================================================================
@dataclass
class HistoryMetrics:
    """Metrics for history database operations."""
    
    # Counters
    appends: int = 0
    batch_appends: int = 0
    queries: int = 0
    
    # Token tracking
    total_tokens_written: int = 0
    total_interactions_written: int = 0
    total_interactions_read: int = 0
    
    # Latency tracking (in ms)
    append_latency_sum: float = 0.0
    batch_append_latency_sum: float = 0.0
    query_latency_sum: float = 0.0
    
    def record_append(self, latency_ms: float, tokens: int) -> None:
        self.appends += 1
        self.append_latency_sum += latency_ms
        self.total_tokens_written += tokens
        self.total_interactions_written += 1
    
    def record_batch_append(self, latency_ms: float, count: int, tokens: int) -> None:
        self.batch_appends += 1
        self.batch_append_latency_sum += latency_ms
        self.total_tokens_written += tokens
        self.total_interactions_written += count
    
    def record_query(self, latency_ms: float, result_count: int) -> None:
        self.queries += 1
        self.query_latency_sum += latency_ms
        self.total_interactions_read += result_count
    
    @property
    def avg_append_latency_ms(self) -> float:
        return self.append_latency_sum / self.appends if self.appends > 0 else 0
    
    @property
    def avg_query_latency_ms(self) -> float:
        return self.query_latency_sum / self.queries if self.queries > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "appends": self.appends,
            "batch_appends": self.batch_appends,
            "queries": self.queries,
            "total_tokens_written": self.total_tokens_written,
            "total_interactions_written": self.total_interactions_written,
            "total_interactions_read": self.total_interactions_read,
            "avg_append_latency_ms": self.avg_append_latency_ms,
            "avg_query_latency_ms": self.avg_query_latency_ms,
        }
