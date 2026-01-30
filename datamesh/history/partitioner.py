"""
History Partitioner: Time-Bucket Partitioning Strategy

Provides time-based partitioning for conversation history:
- Hot/Warm/Cold tier classification
- Automatic bucket rotation
- Partition pruning for expired data
- TTL enforcement at bucket granularity

Partitioning Strategy:
    - Hot (0-90 days): In-memory + fast storage
    - Warm (90 days - 2 years): Compressed columnar
    - Cold (>2 years): Archival storage (Glacier-like)

Bucket Granularity:
    - HOURLY: For high-frequency analytics
    - DAILY: Default for most use cases
    - WEEKLY: For low-volume entities
    - MONTHLY: For cost optimization
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Optional, Sequence

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp


# =============================================================================
# BUCKET GRANULARITY
# =============================================================================
class BucketGranularity(Enum):
    """Time granularity for history buckets."""
    HOURLY = auto()   # %Y-%m-%d-%H
    DAILY = auto()    # %Y-%m-%d
    WEEKLY = auto()   # %Y-W%W
    MONTHLY = auto()  # %Y-%m
    
    @property
    def format_string(self) -> str:
        """datetime format string for this granularity."""
        return {
            BucketGranularity.HOURLY: "%Y-%m-%d-%H",
            BucketGranularity.DAILY: "%Y-%m-%d",
            BucketGranularity.WEEKLY: "%Y-W%W",
            BucketGranularity.MONTHLY: "%Y-%m",
        }[self]
    
    @property
    def duration_days(self) -> float:
        """Approximate duration in days."""
        return {
            BucketGranularity.HOURLY: 1/24,
            BucketGranularity.DAILY: 1,
            BucketGranularity.WEEKLY: 7,
            BucketGranularity.MONTHLY: 30,
        }[self]


# =============================================================================
# TIER CLASSIFICATION
# =============================================================================
class TierClassification(Enum):
    """Storage tier for history data."""
    HOT = auto()    # High-performance, high-cost
    WARM = auto()   # Balanced performance/cost
    COLD = auto()   # Low-cost archival
    FROZEN = auto() # Minimal-cost, slow retrieval
    
    @property
    def retention_days(self) -> int:
        """Typical retention at each tier."""
        return {
            TierClassification.HOT: 90,
            TierClassification.WARM: 365 * 2,
            TierClassification.COLD: 365 * 7,
            TierClassification.FROZEN: -1,  # Indefinite
        }[self]
    
    @property
    def access_latency_hint(self) -> str:
        """Expected access latency."""
        return {
            TierClassification.HOT: "<10ms",
            TierClassification.WARM: "<100ms",
            TierClassification.COLD: "<10s",
            TierClassification.FROZEN: "minutes-hours",
        }[self]


# =============================================================================
# HISTORY BUCKET
# =============================================================================
@dataclass(slots=True)
class HistoryBucket:
    """
    Represents a time-partitioned history bucket.
    
    Buckets are immutable units for:
    - TTL enforcement (delete entire bucket)
    - Compaction (compress bucket contents)
    - Archival (move bucket to cold storage)
    """
    bucket_id: str  # Formatted time identifier
    entity_id: EntityId
    granularity: BucketGranularity
    tier: TierClassification = TierClassification.HOT
    
    # Time bounds
    start_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    end_time: Optional[datetime] = None
    
    # Bucket metadata
    interaction_count: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    is_sealed: bool = False  # Sealed buckets are immutable
    
    # Storage references
    storage_path: Optional[str] = None  # Path in current tier
    archive_path: Optional[str] = None  # Path in archive tier
    
    @property
    def age_days(self) -> float:
        """Bucket age in days."""
        now = datetime.now(timezone.utc)
        return (now - self.start_time).total_seconds() / 86400
    
    @property
    def should_compact(self) -> bool:
        """Check if bucket should be compacted."""
        return (
            self.is_sealed
            and self.tier == TierClassification.HOT
            and self.age_days > 7
        )
    
    @property
    def should_archive(self) -> bool:
        """Check if bucket should move to colder tier."""
        if self.tier == TierClassification.HOT:
            return self.age_days > 90
        if self.tier == TierClassification.WARM:
            return self.age_days > 365 * 2
        return False
    
    def seal(self) -> None:
        """Seal bucket (make immutable)."""
        self.is_sealed = True
        self.end_time = datetime.now(timezone.utc)
    
    def promote(self, new_tier: TierClassification) -> None:
        """Move bucket to different tier."""
        self.tier = new_tier


# =============================================================================
# HISTORY PARTITIONER
# =============================================================================
class HistoryPartitioner:
    """
    Manages time-based partitioning for conversation history.
    
    Features:
        - Automatic bucket creation based on granularity
        - Tier classification based on age
        - TTL enforcement via bucket pruning
        - Bucket sealing for immutability
    
    Usage:
        partitioner = HistoryPartitioner(
            granularity=BucketGranularity.MONTHLY,
            hot_retention_days=90,
        )
        
        # Get bucket for timestamp
        bucket = partitioner.get_or_create_bucket(entity_id, timestamp)
        
        # Classify tier based on age
        tier = partitioner.classify_tier(bucket)
        
        # Prune expired buckets
        expired = await partitioner.prune_expired(entity_id)
    """
    
    __slots__ = (
        "_granularity", "_hot_retention_days", "_warm_retention_days",
        "_cold_retention_days", "_buckets", "_lock",
    )
    
    def __init__(
        self,
        granularity: BucketGranularity = BucketGranularity.MONTHLY,
        hot_retention_days: int = 90,
        warm_retention_days: int = 365 * 2,
        cold_retention_days: int = 365 * 7,
    ) -> None:
        self._granularity = granularity
        self._hot_retention_days = hot_retention_days
        self._warm_retention_days = warm_retention_days
        self._cold_retention_days = cold_retention_days
        # Storage: (entity_id, bucket_id) -> HistoryBucket
        self._buckets: dict[tuple[str, str], HistoryBucket] = {}
        self._lock = asyncio.Lock()
    
    def compute_bucket_id(self, timestamp: datetime) -> str:
        """Compute bucket ID for given timestamp."""
        return timestamp.strftime(self._granularity.format_string)
    
    def get_or_create_bucket(
        self,
        entity_id: EntityId,
        timestamp: Optional[datetime] = None,
    ) -> HistoryBucket:
        """Get existing bucket or create new one."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        bucket_id = self.compute_bucket_id(timestamp)
        key = (str(entity_id.value), bucket_id)
        
        if key not in self._buckets:
            self._buckets[key] = HistoryBucket(
                bucket_id=bucket_id,
                entity_id=entity_id,
                granularity=self._granularity,
                start_time=timestamp,
            )
        
        return self._buckets[key]
    
    def classify_tier(self, bucket: HistoryBucket) -> TierClassification:
        """Classify bucket's storage tier based on age."""
        age_days = bucket.age_days
        
        if age_days <= self._hot_retention_days:
            return TierClassification.HOT
        if age_days <= self._warm_retention_days:
            return TierClassification.WARM
        if age_days <= self._cold_retention_days:
            return TierClassification.COLD
        return TierClassification.FROZEN
    
    async def update_bucket_stats(
        self,
        entity_id: EntityId,
        bucket_id: str,
        interaction_count_delta: int = 0,
        token_delta: int = 0,
        byte_delta: int = 0,
    ) -> None:
        """Update bucket statistics."""
        async with self._lock:
            key = (str(entity_id.value), bucket_id)
            bucket = self._buckets.get(key)
            
            if bucket:
                bucket.interaction_count += interaction_count_delta
                bucket.total_tokens += token_delta
                bucket.total_bytes += byte_delta
    
    async def seal_bucket(
        self,
        entity_id: EntityId,
        bucket_id: str,
    ) -> Result[bool, str]:
        """Seal bucket (make immutable)."""
        async with self._lock:
            key = (str(entity_id.value), bucket_id)
            bucket = self._buckets.get(key)
            
            if not bucket:
                return Err(f"Bucket {bucket_id} not found")
            
            if bucket.is_sealed:
                return Ok(False)  # Already sealed
            
            bucket.seal()
            return Ok(True)
    
    async def get_buckets_for_entity(
        self,
        entity_id: EntityId,
        tier_filter: Optional[TierClassification] = None,
    ) -> list[HistoryBucket]:
        """Get all buckets for an entity."""
        async with self._lock:
            entity_key = str(entity_id.value)
            buckets = [
                b for (eid, _), b in self._buckets.items()
                if eid == entity_key
            ]
            
            if tier_filter:
                buckets = [b for b in buckets if b.tier == tier_filter]
            
            # Sort by start_time descending
            buckets.sort(key=lambda b: b.start_time, reverse=True)
            return buckets
    
    async def get_buckets_needing_compaction(self) -> list[HistoryBucket]:
        """Get all buckets that need compaction."""
        async with self._lock:
            return [b for b in self._buckets.values() if b.should_compact]
    
    async def get_buckets_needing_archival(self) -> list[HistoryBucket]:
        """Get all buckets that need tier migration."""
        async with self._lock:
            return [b for b in self._buckets.values() if b.should_archive]
    
    async def prune_expired(
        self,
        entity_id: EntityId,
    ) -> Result[list[str], str]:
        """
        Prune buckets exceeding cold retention.
        
        Returns list of pruned bucket IDs.
        """
        async with self._lock:
            entity_key = str(entity_id.value)
            cutoff_days = self._cold_retention_days
            
            to_prune = [
                (key, bucket) for key, bucket in self._buckets.items()
                if key[0] == entity_key
                and bucket.age_days > cutoff_days
            ]
            
            pruned_ids = []
            for key, bucket in to_prune:
                del self._buckets[key]
                pruned_ids.append(bucket.bucket_id)
            
            return Ok(pruned_ids)
    
    async def get_bucket_range(
        self,
        entity_id: EntityId,
        start_time: datetime,
        end_time: datetime,
    ) -> list[HistoryBucket]:
        """Get buckets overlapping time range."""
        async with self._lock:
            entity_key = str(entity_id.value)
            return [
                b for (eid, _), b in self._buckets.items()
                if eid == entity_key
                and b.start_time <= end_time
                and (b.end_time is None or b.end_time >= start_time)
            ]
    
    @property
    def granularity(self) -> BucketGranularity:
        """Configured granularity."""
        return self._granularity
    
    @property
    def total_buckets(self) -> int:
        """Total bucket count."""
        return len(self._buckets)
