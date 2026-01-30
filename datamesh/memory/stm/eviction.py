"""
Eviction Manager: Policy-Based Eviction for STM

Provides:
- Eviction policies (LRU, LFU, TTL-based)
- Eviction event generation for LTM consolidation
- Configurable thresholds and watermarks
- Backpressure signaling

Design:
    Eviction is the bridge between STM and LTM.
    When items are evicted from working set,
    the consolidation pipeline is notified
    to persist semantically important content.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Optional
from uuid import uuid4

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp


# =============================================================================
# EVICTION POLICY
# =============================================================================
class EvictionPolicy(Enum):
    """Policy for selecting items to evict."""
    LRU = auto()       # Least Recently Used
    LFU = auto()       # Least Frequently Used
    FIFO = auto()      # First In First Out
    TTL = auto()       # Time-To-Live based
    SIZE = auto()      # Largest items first
    PRIORITY = auto()  # Lowest priority first
    
    def describe(self) -> str:
        """Human-readable description."""
        return {
            EvictionPolicy.LRU: "Evict least recently accessed items",
            EvictionPolicy.LFU: "Evict least frequently accessed items",
            EvictionPolicy.FIFO: "Evict oldest items first",
            EvictionPolicy.TTL: "Evict items that have expired",
            EvictionPolicy.SIZE: "Evict largest items first",
            EvictionPolicy.PRIORITY: "Evict lowest priority items first",
        }[self]


# =============================================================================
# EVICTION EVENT
# =============================================================================
@dataclass(frozen=True, slots=True)
class EvictionEvent:
    """
    Event generated when items are evicted from STM.
    
    Used to trigger LTM consolidation.
    """
    event_id: str
    entity_id: EntityId
    evicted_items: tuple[str, ...]  # Item IDs
    reason: str
    policy: EvictionPolicy
    timestamp: Timestamp
    total_bytes: int
    total_tokens: int
    
    # Consolidation hints
    should_consolidate: bool = True
    priority: int = 1  # 1-10, higher = more important
    
    @property
    def item_count(self) -> int:
        """Number of evicted items."""
        return len(self.evicted_items)


# =============================================================================
# EVICTION CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class EvictionConfig:
    """Configuration for eviction manager."""
    policy: EvictionPolicy = EvictionPolicy.LRU
    high_watermark_ratio: float = 0.9   # Start evicting at 90%
    low_watermark_ratio: float = 0.7    # Stop evicting at 70%
    ttl_seconds: int = 3600             # Default TTL for TTL policy
    batch_size: int = 10                # Items to evict per batch
    consolidation_threshold: int = 100  # Tokens to trigger consolidation


# =============================================================================
# EVICTION MANAGER
# =============================================================================
class EvictionManager:
    """
    Manages eviction policies for short-term memory.
    
    Features:
        - Configurable eviction policies
        - High/low watermark thresholds
        - Event generation for consolidation
        - Backpressure signaling
    
    Usage:
        manager = EvictionManager(
            entity_id=entity_id,
            config=EvictionConfig(policy=EvictionPolicy.LRU),
        )
        manager.on_eviction(lambda e: consolidation_pipeline.enqueue(e))
        
        # Check if eviction needed
        if manager.should_evict(current_bytes, max_bytes):
            events = await manager.plan_eviction(items, target_bytes)
            for event in events:
                await execute_eviction(event)
    """
    
    __slots__ = (
        "_entity_id", "_config", "_event_handlers",
        "_access_counts", "_access_times", "_lock",
    )
    
    def __init__(
        self,
        entity_id: EntityId,
        config: Optional[EvictionConfig] = None,
    ) -> None:
        self._entity_id = entity_id
        self._config = config or EvictionConfig()
        self._event_handlers: list[Callable[[EvictionEvent], None]] = []
        self._access_counts: dict[str, int] = {}
        self._access_times: dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    def on_eviction(
        self,
        handler: Callable[[EvictionEvent], None],
    ) -> None:
        """Register eviction event handler."""
        self._event_handlers.append(handler)
    
    def record_access(self, item_id: str) -> None:
        """Record item access for LRU/LFU tracking."""
        now = datetime.now(timezone.utc)
        self._access_times[item_id] = now
        self._access_counts[item_id] = self._access_counts.get(item_id, 0) + 1
    
    def should_evict(
        self,
        current: int,
        maximum: int,
    ) -> bool:
        """Check if eviction should occur based on watermarks."""
        if maximum <= 0:
            return False
        ratio = current / maximum
        return ratio >= self._config.high_watermark_ratio
    
    def target_after_eviction(
        self,
        current: int,
        maximum: int,
    ) -> int:
        """Calculate target size after eviction."""
        return int(maximum * self._config.low_watermark_ratio)
    
    async def plan_eviction(
        self,
        items: list[Any],  # WorkingSetItems
        bytes_to_evict: int,
    ) -> list[EvictionEvent]:
        """
        Plan eviction based on policy.
        
        Returns list of eviction events to execute.
        """
        async with self._lock:
            if not items:
                return []
            
            # Sort items by eviction priority
            sorted_items = self._sort_by_policy(items)
            
            # Select items to evict
            to_evict: list[Any] = []
            evicted_bytes = 0
            evicted_tokens = 0
            
            for item in sorted_items:
                to_evict.append(item)
                evicted_bytes += item.size_bytes
                evicted_tokens += item.token_count
                
                if evicted_bytes >= bytes_to_evict:
                    break
            
            if not to_evict:
                return []
            
            # Create eviction event
            event = EvictionEvent(
                event_id=str(uuid4()),
                entity_id=self._entity_id,
                evicted_items=tuple(i.item_id for i in to_evict),
                reason=f"Capacity exceeded, policy: {self._config.policy.name}",
                policy=self._config.policy,
                timestamp=Timestamp.now(),
                total_bytes=evicted_bytes,
                total_tokens=evicted_tokens,
                should_consolidate=(evicted_tokens >= self._config.consolidation_threshold),
                priority=self._calculate_priority(to_evict),
            )
            
            # Notify handlers
            for handler in self._event_handlers:
                try:
                    handler(event)
                except Exception:
                    pass
            
            return [event]
    
    def _sort_by_policy(self, items: list[Any]) -> list[Any]:
        """Sort items by eviction policy (first = evict first)."""
        policy = self._config.policy
        
        if policy == EvictionPolicy.LRU:
            return sorted(
                items,
                key=lambda i: self._access_times.get(
                    i.item_id,
                    datetime.min.replace(tzinfo=timezone.utc),
                ),
            )
        
        if policy == EvictionPolicy.LFU:
            return sorted(
                items,
                key=lambda i: self._access_counts.get(i.item_id, 0),
            )
        
        if policy == EvictionPolicy.FIFO:
            return sorted(items, key=lambda i: i.sequence_id)
        
        if policy == EvictionPolicy.SIZE:
            return sorted(items, key=lambda i: i.size_bytes, reverse=True)
        
        # Default: FIFO
        return sorted(items, key=lambda i: i.sequence_id)
    
    def _calculate_priority(self, items: list[Any]) -> int:
        """Calculate consolidation priority for evicted items."""
        # Higher priority for more tokens
        total_tokens = sum(i.token_count for i in items)
        
        if total_tokens > 1000:
            return 10
        if total_tokens > 500:
            return 7
        if total_tokens > 100:
            return 5
        return 3
    
    def cleanup_tracking(self, item_ids: list[str]) -> None:
        """Remove tracking data for evicted items."""
        for item_id in item_ids:
            self._access_counts.pop(item_id, None)
            self._access_times.pop(item_id, None)
    
    @property
    def config(self) -> EvictionConfig:
        """Current configuration."""
        return self._config
