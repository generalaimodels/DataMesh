"""
Working Set: Sliding Window Buffer for Short-Term Memory

Provides:
- Redis ZSET-like sorted set with sequence_id scoring
- O(log N) range queries for context extraction
- Atomic pop operations for eviction
- Configurable size limits (count and bytes)

Design:
    Working set stores recent interactions in sorted order.
    Used for:
    - Context window construction
    - Recency-based retrieval
    - Eviction to long-term memory
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence
from uuid import UUID

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class WorkingSetConfig:
    """Configuration for working set."""
    max_items: int = 20           # Maximum number of items
    max_bytes: int = 65536        # Maximum total bytes (64KB)
    max_tokens: int = 8192        # Maximum total tokens
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if > 1KB


# =============================================================================
# WORKING SET ITEM
# =============================================================================
@dataclass(slots=True)
class WorkingSetItem:
    """
    Single item in the working set.
    
    Stores interaction content with scoring for retrieval.
    """
    item_id: str
    sequence_id: int  # Primary sort key (higher = more recent)
    content: bytes
    token_count: int = 0
    
    # Metadata
    role: str = "user"  # user, assistant, system, tool
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Compression
    is_compressed: bool = False
    original_size: int = 0
    
    @property
    def size_bytes(self) -> int:
        """Content size in bytes."""
        return len(self.content)
    
    @property
    def effective_size(self) -> int:
        """Original size before compression."""
        return self.original_size if self.is_compressed else self.size_bytes
    
    def compress(self) -> None:
        """Compress content with LZ4."""
        if self.is_compressed or self.size_bytes < 1024:
            return
        
        import lz4.frame
        self.original_size = self.size_bytes
        self.content = lz4.frame.compress(self.content)
        self.is_compressed = True
    
    def decompress(self) -> bytes:
        """Get decompressed content."""
        if not self.is_compressed:
            return self.content
        
        import lz4.frame
        return lz4.frame.decompress(self.content)


# =============================================================================
# WORKING SET
# =============================================================================
class WorkingSet:
    """
    Sliding window buffer for short-term memory.
    
    Features:
        - Sorted set semantics (like Redis ZSET)
        - O(log N) range queries by sequence_id
        - Configurable size limits
        - Eviction callbacks for LTM consolidation
    
    Usage:
        ws = WorkingSet(entity_id, config=WorkingSetConfig(max_items=20))
        
        # Add item
        item = WorkingSetItem(
            item_id="msg-1",
            sequence_id=1,
            content=b"Hello, world!",
            token_count=3,
        )
        await ws.add(item)
        
        # Get recent items
        recent = await ws.get_range(start_seq=0, limit=10)
        
        # Pop oldest for eviction
        evicted = await ws.pop_oldest(count=5)
    """
    
    __slots__ = (
        "_entity_id", "_config", "_items", "_sequence_index",
        "_total_bytes", "_total_tokens", "_lock",
        "_eviction_callbacks",
    )
    
    def __init__(
        self,
        entity_id: EntityId,
        config: Optional[WorkingSetConfig] = None,
    ) -> None:
        self._entity_id = entity_id
        self._config = config or WorkingSetConfig()
        self._items: dict[str, WorkingSetItem] = {}
        self._sequence_index: list[str] = []  # item_ids sorted by sequence_id
        self._total_bytes = 0
        self._total_tokens = 0
        self._lock = asyncio.Lock()
        self._eviction_callbacks: list[Callable[[list[WorkingSetItem]], None]] = []
    
    def on_eviction(
        self,
        callback: Callable[[list[WorkingSetItem]], None],
    ) -> None:
        """Register callback for eviction events."""
        self._eviction_callbacks.append(callback)
    
    async def add(
        self,
        item: WorkingSetItem,
    ) -> Result[None, str]:
        """
        Add item to working set.
        
        May trigger eviction if limits exceeded.
        """
        async with self._lock:
            # Check for duplicate
            if item.item_id in self._items:
                return Err(f"Item {item.item_id} already exists")
            
            # Compress if configured
            if self._config.enable_compression:
                if item.size_bytes >= self._config.compression_threshold:
                    item.compress()
            
            # Add item
            self._items[item.item_id] = item
            self._total_bytes += item.size_bytes
            self._total_tokens += item.token_count
            
            # Insert in sorted order by sequence_id
            # Binary search for insertion point
            import bisect
            bisect.insort(
                self._sequence_index,
                item.item_id,
                key=lambda iid: self._items[iid].sequence_id,
            )
            
            # Check limits and evict if necessary
            await self._enforce_limits()
            
            return Ok(None)
    
    async def get(self, item_id: str) -> Optional[WorkingSetItem]:
        """Get item by ID."""
        async with self._lock:
            return self._items.get(item_id)
    
    async def get_range(
        self,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
        limit: int = 100,
        descending: bool = True,
    ) -> list[WorkingSetItem]:
        """
        Get items in sequence range.
        
        Args:
            start_seq: Minimum sequence_id (inclusive)
            end_seq: Maximum sequence_id (inclusive)
            limit: Maximum items to return
            descending: If True, return newest first
        """
        async with self._lock:
            items = list(self._items.values())
            
            # Filter by range
            if start_seq is not None:
                items = [i for i in items if i.sequence_id >= start_seq]
            if end_seq is not None:
                items = [i for i in items if i.sequence_id <= end_seq]
            
            # Sort
            items.sort(key=lambda i: i.sequence_id, reverse=descending)
            
            return items[:limit]
    
    async def get_recent(self, limit: int = 10) -> list[WorkingSetItem]:
        """Get most recent items."""
        return await self.get_range(limit=limit, descending=True)
    
    async def get_oldest(self, limit: int = 10) -> list[WorkingSetItem]:
        """Get oldest items."""
        return await self.get_range(limit=limit, descending=False)
    
    async def pop_oldest(self, count: int = 1) -> list[WorkingSetItem]:
        """
        Remove and return oldest items.
        
        Used for eviction to long-term memory.
        """
        async with self._lock:
            evicted: list[WorkingSetItem] = []
            
            for _ in range(min(count, len(self._sequence_index))):
                item_id = self._sequence_index.pop(0)
                item = self._items.pop(item_id)
                self._total_bytes -= item.size_bytes
                self._total_tokens -= item.token_count
                evicted.append(item)
            
            # Notify callbacks
            if evicted and self._eviction_callbacks:
                for callback in self._eviction_callbacks:
                    try:
                        callback(evicted)
                    except Exception:
                        pass
            
            return evicted
    
    async def remove(self, item_id: str) -> bool:
        """Remove specific item."""
        async with self._lock:
            if item_id not in self._items:
                return False
            
            item = self._items.pop(item_id)
            self._sequence_index.remove(item_id)
            self._total_bytes -= item.size_bytes
            self._total_tokens -= item.token_count
            
            return True
    
    async def clear(self) -> int:
        """Clear all items. Returns count removed."""
        async with self._lock:
            count = len(self._items)
            self._items.clear()
            self._sequence_index.clear()
            self._total_bytes = 0
            self._total_tokens = 0
            return count
    
    async def _enforce_limits(self) -> None:
        """Evict items if limits exceeded."""
        evicted: list[WorkingSetItem] = []
        
        # Evict by count
        while len(self._items) > self._config.max_items:
            item_id = self._sequence_index.pop(0)
            item = self._items.pop(item_id)
            self._total_bytes -= item.size_bytes
            self._total_tokens -= item.token_count
            evicted.append(item)
        
        # Evict by bytes
        while self._total_bytes > self._config.max_bytes and self._sequence_index:
            item_id = self._sequence_index.pop(0)
            item = self._items.pop(item_id)
            self._total_bytes -= item.size_bytes
            self._total_tokens -= item.token_count
            evicted.append(item)
        
        # Evict by tokens
        while self._total_tokens > self._config.max_tokens and self._sequence_index:
            item_id = self._sequence_index.pop(0)
            item = self._items.pop(item_id)
            self._total_bytes -= item.size_bytes
            self._total_tokens -= item.token_count
            evicted.append(item)
        
        # Notify callbacks
        if evicted and self._eviction_callbacks:
            for callback in self._eviction_callbacks:
                try:
                    callback(evicted)
                except Exception:
                    pass
    
    @property
    def entity_id(self) -> EntityId:
        """Entity ID for this working set."""
        return self._entity_id
    
    @property
    def count(self) -> int:
        """Item count."""
        return len(self._items)
    
    @property
    def total_bytes(self) -> int:
        """Total bytes stored."""
        return self._total_bytes
    
    @property
    def total_tokens(self) -> int:
        """Total tokens stored."""
        return self._total_tokens
    
    @property
    def is_empty(self) -> bool:
        """Check if working set is empty."""
        return len(self._items) == 0
    
    @property
    def max_sequence_id(self) -> Optional[int]:
        """Maximum sequence ID in set."""
        if not self._sequence_index:
            return None
        return self._items[self._sequence_index[-1]].sequence_id
    
    @property
    def min_sequence_id(self) -> Optional[int]:
        """Minimum sequence ID in set."""
        if not self._sequence_index:
            return None
        return self._items[self._sequence_index[0]].sequence_id
