"""
Conversation Timeline: Time-Series Interaction Storage

Provides append-only time-series storage for conversation interactions:
- Optimized for sequential writes (LSM-friendly)
- Time-bucket partitioning for efficient pruning
- Reverse chronological ordering for timeline queries
- Embedding support for semantic history retrieval

Storage Model:
    conversation_history (
        entity_id UUID,
        session_id UUID,
        bucket_id STRING,         -- YYYY-MM for TTL efficiency
        sequence_id BIGINT,       -- Monotonic within session
        interaction_type ENUM,    -- PROMPT | RESPONSE | SYSTEM
        content_ref STRING,       -- S3 URI with byte-range
        embedding_vector VECTOR,  -- For semantic retrieval
        token_count INT,
        created_at TIMESTAMP,
        PRIMARY KEY ((entity_id, bucket_id), sequence_id DESC)
    )
"""

from __future__ import annotations

import asyncio
import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional, Sequence, Iterator
from uuid import UUID

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, ConversationId, Timestamp,
    ContentHash, EmbeddingVector,
)
from datamesh.core.errors import StorageError


# =============================================================================
# INTERACTION TYPES
# =============================================================================
class InteractionType(Enum):
    """Type of conversation interaction."""
    PROMPT = auto()       # User input
    RESPONSE = auto()     # Model output
    SYSTEM = auto()       # System messages, instructions
    TOOL_CALL = auto()    # Function/tool invocations
    TOOL_RESULT = auto()  # Function/tool results
    ANNOTATION = auto()   # Human annotations, feedback
    
    @property
    def is_user_generated(self) -> bool:
        """Check if interaction is user-generated."""
        return self in (InteractionType.PROMPT, InteractionType.ANNOTATION)
    
    @property
    def is_model_generated(self) -> bool:
        """Check if interaction is model-generated."""
        return self in (InteractionType.RESPONSE, InteractionType.TOOL_CALL)


# =============================================================================
# INTERACTION MODEL
# =============================================================================
@dataclass(slots=True)
class Interaction:
    """
    Single conversation interaction.
    
    Immutable after creation to ensure timeline integrity.
    Content stored by reference for large payloads.
    """
    # Identity
    entity_id: EntityId
    session_id: UUID
    sequence_id: int  # Monotonic within session
    
    # Type and content
    interaction_type: InteractionType
    content: bytes  # Inline for small content
    content_ref: Optional[str] = None  # S3 URI for large content
    
    # Metadata
    bucket_id: str = ""  # YYYY-MM format
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    token_count: int = 0
    
    # Optional embeddings for semantic retrieval
    embedding: Optional[EmbeddingVector] = None
    
    # Extended metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Compute bucket_id from created_at."""
        if not self.bucket_id:
            self.bucket_id = self.created_at.strftime("%Y-%m")
    
    @property
    def composite_key(self) -> tuple[UUID, str, int]:
        """Composite key for storage: (entity_id, bucket_id, sequence_id)."""
        return (self.entity_id.value, self.bucket_id, self.sequence_id)
    
    @property
    def is_large(self) -> bool:
        """Check if content is stored by reference."""
        return self.content_ref is not None
    
    @property
    def content_size(self) -> int:
        """Content size in bytes."""
        return len(self.content)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entity_id": str(self.entity_id.value),
            "session_id": str(self.session_id),
            "sequence_id": self.sequence_id,
            "interaction_type": self.interaction_type.name,
            "content": self.content.hex() if len(self.content) < 1024 else None,
            "content_ref": self.content_ref,
            "bucket_id": self.bucket_id,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
            "has_embedding": self.embedding is not None,
            "metadata": self.metadata,
        }


# =============================================================================
# TIMELINE STATISTICS
# =============================================================================
@dataclass
class TimelineStats:
    """Statistics for conversation timeline."""
    total_interactions: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    bucket_count: int = 0
    oldest_interaction: Optional[datetime] = None
    newest_interaction: Optional[datetime] = None
    
    @property
    def avg_tokens_per_interaction(self) -> float:
        """Average tokens per interaction."""
        if self.total_interactions == 0:
            return 0.0
        return self.total_tokens / self.total_interactions


# =============================================================================
# CONVERSATION TIMELINE
# =============================================================================
class ConversationTimeline:
    """
    Time-series storage for conversation interactions.
    
    Features:
        - Append-only writes for LSM optimization
        - Time-bucket partitioning for efficient pruning
        - Reverse chronological ordering for timeline queries
        - Semantic retrieval via embedding vectors
    
    Usage:
        timeline = ConversationTimeline(entity_id)
        
        # Append interaction
        interaction = Interaction(
            entity_id=entity_id,
            session_id=session_id,
            sequence_id=0,
            interaction_type=InteractionType.PROMPT,
            content=b"Hello, world!",
        )
        await timeline.append(interaction)
        
        # Query recent history
        result = await timeline.get_recent(limit=10)
    
    Thread Safety:
        All operations are async-safe via internal locking.
    """
    
    __slots__ = (
        "_entity_id", "_interactions", "_sequence_counter",
        "_buckets", "_stats", "_lock",
    )
    
    def __init__(self, entity_id: EntityId) -> None:
        self._entity_id = entity_id
        # In-memory storage: bucket_id -> list of interactions (sorted by sequence_id desc)
        self._interactions: dict[str, list[Interaction]] = {}
        self._sequence_counter: int = 0
        self._buckets: list[str] = []  # Sorted bucket IDs
        self._stats = TimelineStats()
        self._lock = asyncio.Lock()
    
    async def append(
        self,
        interaction: Interaction,
    ) -> Result[int, StorageError]:
        """
        Append interaction to timeline.
        
        Returns the assigned sequence_id.
        """
        async with self._lock:
            # Validate entity_id
            if interaction.entity_id.value != self._entity_id.value:
                return Err(StorageError.constraint_violation(
                    constraint="fk_entity_id",
                    table="conversation_history",
                ))
            
            # Assign sequence_id if not set
            if interaction.sequence_id == 0:
                self._sequence_counter += 1
                interaction.sequence_id = self._sequence_counter
            else:
                self._sequence_counter = max(
                    self._sequence_counter,
                    interaction.sequence_id,
                )
            
            # Get or create bucket
            bucket_id = interaction.bucket_id
            if bucket_id not in self._interactions:
                self._interactions[bucket_id] = []
                bisect.insort(self._buckets, bucket_id)
                self._stats.bucket_count = len(self._buckets)
            
            # Insert in sorted order (by sequence_id descending)
            bucket = self._interactions[bucket_id]
            # Find insertion point for descending order
            idx = 0
            for i, item in enumerate(bucket):
                if interaction.sequence_id > item.sequence_id:
                    idx = i
                    break
                idx = i + 1
            bucket.insert(idx, interaction)
            
            # Update stats
            self._stats.total_interactions += 1
            self._stats.total_tokens += interaction.token_count
            self._stats.total_bytes += interaction.content_size
            
            if self._stats.oldest_interaction is None:
                self._stats.oldest_interaction = interaction.created_at
            self._stats.oldest_interaction = min(
                self._stats.oldest_interaction,
                interaction.created_at,
            )
            self._stats.newest_interaction = max(
                self._stats.newest_interaction or interaction.created_at,
                interaction.created_at,
            )
            
            return Ok(interaction.sequence_id)
    
    async def get_recent(
        self,
        limit: int = 10,
        before_sequence: Optional[int] = None,
    ) -> Result[list[Interaction], StorageError]:
        """
        Get most recent interactions.
        
        Args:
            limit: Maximum interactions to return
            before_sequence: Only return interactions before this sequence_id
        """
        async with self._lock:
            result: list[Interaction] = []
            
            # Iterate buckets in reverse chronological order
            for bucket_id in reversed(self._buckets):
                bucket = self._interactions[bucket_id]
                
                for interaction in bucket:
                    if before_sequence and interaction.sequence_id >= before_sequence:
                        continue
                    result.append(interaction)
                    if len(result) >= limit:
                        return Ok(result)
            
            return Ok(result)
    
    async def get_by_session(
        self,
        session_id: UUID,
        limit: int = 100,
    ) -> Result[list[Interaction], StorageError]:
        """Get interactions for specific session."""
        async with self._lock:
            result: list[Interaction] = []
            
            for bucket in self._interactions.values():
                for interaction in bucket:
                    if interaction.session_id == session_id:
                        result.append(interaction)
                        if len(result) >= limit:
                            return Ok(result)
            
            # Sort by sequence_id descending
            result.sort(key=lambda i: i.sequence_id, reverse=True)
            return Ok(result[:limit])
    
    async def get_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> Result[list[Interaction], StorageError]:
        """Get interactions in time range."""
        async with self._lock:
            result: list[Interaction] = []
            
            for bucket in self._interactions.values():
                for interaction in bucket:
                    if start_time <= interaction.created_at <= end_time:
                        result.append(interaction)
                        if len(result) >= limit:
                            return Ok(result)
            
            result.sort(key=lambda i: i.created_at, reverse=True)
            return Ok(result)
    
    async def get_by_type(
        self,
        interaction_type: InteractionType,
        limit: int = 100,
    ) -> Result[list[Interaction], StorageError]:
        """Get interactions by type."""
        async with self._lock:
            result: list[Interaction] = []
            
            for bucket in self._interactions.values():
                for interaction in bucket:
                    if interaction.interaction_type == interaction_type:
                        result.append(interaction)
                        if len(result) >= limit:
                            return Ok(result)
            
            return Ok(result)
    
    async def count_tokens(
        self,
        session_id: Optional[UUID] = None,
    ) -> int:
        """Count total tokens, optionally filtered by session."""
        async with self._lock:
            if session_id is None:
                return self._stats.total_tokens
            
            total = 0
            for bucket in self._interactions.values():
                for interaction in bucket:
                    if interaction.session_id == session_id:
                        total += interaction.token_count
            return total
    
    async def get_bucket(
        self,
        bucket_id: str,
    ) -> Result[list[Interaction], StorageError]:
        """Get all interactions in a specific bucket."""
        async with self._lock:
            bucket = self._interactions.get(bucket_id, [])
            return Ok(list(bucket))
    
    async def delete_bucket(
        self,
        bucket_id: str,
    ) -> Result[int, StorageError]:
        """Delete entire bucket. Returns count of deleted interactions."""
        async with self._lock:
            if bucket_id not in self._interactions:
                return Ok(0)
            
            bucket = self._interactions.pop(bucket_id)
            count = len(bucket)
            
            # Update stats
            self._stats.total_interactions -= count
            for interaction in bucket:
                self._stats.total_tokens -= interaction.token_count
                self._stats.total_bytes -= interaction.content_size
            
            if bucket_id in self._buckets:
                self._buckets.remove(bucket_id)
            self._stats.bucket_count = len(self._buckets)
            
            return Ok(count)
    
    async def prune_before(
        self,
        cutoff_time: datetime,
    ) -> Result[int, StorageError]:
        """
        Delete all interactions before cutoff time.
        
        Operates at bucket granularity for efficiency.
        """
        async with self._lock:
            cutoff_bucket = cutoff_time.strftime("%Y-%m")
            buckets_to_delete = [
                b for b in self._buckets if b < cutoff_bucket
            ]
            
            total_deleted = 0
            for bucket_id in buckets_to_delete:
                bucket = self._interactions.pop(bucket_id, [])
                total_deleted += len(bucket)
                
                for interaction in bucket:
                    self._stats.total_tokens -= interaction.token_count
                    self._stats.total_bytes -= interaction.content_size
                
                self._buckets.remove(bucket_id)
            
            self._stats.total_interactions -= total_deleted
            self._stats.bucket_count = len(self._buckets)
            
            return Ok(total_deleted)
    
    def iterate_all(self) -> Iterator[Interaction]:
        """
        Iterate all interactions in reverse chronological order.
        
        Note: Not async-safe, use only for offline processing.
        """
        for bucket_id in reversed(self._buckets):
            yield from self._interactions[bucket_id]
    
    @property
    def entity_id(self) -> EntityId:
        """Entity ID for this timeline."""
        return self._entity_id
    
    @property
    def stats(self) -> TimelineStats:
        """Timeline statistics."""
        return self._stats
    
    @property
    def bucket_ids(self) -> list[str]:
        """List of bucket IDs."""
        return list(self._buckets)
