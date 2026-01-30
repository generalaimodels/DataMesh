"""
Time-Based Partitioner: Temporal Data Segmentation

Partitions data by time for:
- Efficient TTL enforcement (GDPR right-to-erasure)
- Hot/warm/cold tiering
- Time-range query optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Iterator, Optional

from datamesh.core.types import Timestamp


class PartitionGranularity(Enum):
    """Time partition granularity."""
    HOURLY = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()


@dataclass(frozen=True, slots=True, order=True)
class Partition:
    """Time partition identifier."""
    start: datetime
    end: datetime
    granularity: PartitionGranularity
    
    @property
    def name(self) -> str:
        """Generate partition name for storage."""
        if self.granularity == PartitionGranularity.HOURLY:
            return self.start.strftime("%Y%m%d_%H")
        elif self.granularity == PartitionGranularity.DAILY:
            return self.start.strftime("%Y%m%d")
        elif self.granularity == PartitionGranularity.WEEKLY:
            return self.start.strftime("%Y_W%W")
        else:  # MONTHLY
            return self.start.strftime("%Y%m")
    
    def contains(self, dt: datetime) -> bool:
        """Check if datetime falls within partition."""
        return self.start <= dt < self.end
    
    @property
    def duration(self) -> timedelta:
        """Partition time span."""
        return self.end - self.start


class TimePartitioner:
    """
    Manages time-based data partitioning.
    
    Supports:
    - Partition assignment for timestamps
    - Range queries across partitions
    - TTL-based partition dropping
    
    Usage:
        partitioner = TimePartitioner(PartitionGranularity.DAILY)
        
        partition = partitioner.get_partition(datetime.now())
        
        for p in partitioner.get_range(start, end):
            query_partition(p)
    """
    
    __slots__ = ("_granularity", "_retention_days")
    
    def __init__(
        self,
        granularity: PartitionGranularity = PartitionGranularity.MONTHLY,
        retention_days: Optional[int] = None,
    ) -> None:
        self._granularity = granularity
        self._retention_days = retention_days
    
    def get_partition(self, dt: datetime) -> Partition:
        """Get partition containing datetime."""
        start = self._truncate(dt)
        end = self._next_boundary(start)
        
        return Partition(
            start=start,
            end=end,
            granularity=self._granularity,
        )
    
    def get_partition_for_timestamp(self, ts: Timestamp) -> Partition:
        """Get partition for Timestamp type."""
        dt = datetime.fromtimestamp(ts.seconds, tz=timezone.utc)
        return self.get_partition(dt)
    
    def get_range(
        self,
        start: datetime,
        end: datetime,
    ) -> Iterator[Partition]:
        """
        Iterate partitions overlapping time range.
        
        Yields partitions in chronological order.
        """
        current = self._truncate(start)
        
        while current < end:
            next_boundary = self._next_boundary(current)
            yield Partition(
                start=current,
                end=next_boundary,
                granularity=self._granularity,
            )
            current = next_boundary
    
    def get_expired_before(self, cutoff: datetime) -> Iterator[Partition]:
        """Get partitions that should be dropped based on retention."""
        if self._retention_days is None:
            return
        
        retention_cutoff = cutoff - timedelta(days=self._retention_days)
        
        # Generate partitions from epoch to cutoff
        # In practice, you'd track created partitions
        current = self._truncate(retention_cutoff - timedelta(days=365))
        
        while current < retention_cutoff:
            next_boundary = self._next_boundary(current)
            if next_boundary <= retention_cutoff:
                yield Partition(
                    start=current,
                    end=next_boundary,
                    granularity=self._granularity,
                )
            current = next_boundary
    
    def _truncate(self, dt: datetime) -> datetime:
        """Truncate datetime to partition boundary."""
        if self._granularity == PartitionGranularity.HOURLY:
            return dt.replace(minute=0, second=0, microsecond=0)
        elif self._granularity == PartitionGranularity.DAILY:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif self._granularity == PartitionGranularity.WEEKLY:
            # Start of ISO week (Monday)
            days_since_monday = dt.weekday()
            monday = dt - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # MONTHLY
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _next_boundary(self, dt: datetime) -> datetime:
        """Get next partition boundary after datetime."""
        if self._granularity == PartitionGranularity.HOURLY:
            return dt + timedelta(hours=1)
        elif self._granularity == PartitionGranularity.DAILY:
            return dt + timedelta(days=1)
        elif self._granularity == PartitionGranularity.WEEKLY:
            return dt + timedelta(weeks=1)
        else:  # MONTHLY
            # Move to first day of next month
            if dt.month == 12:
                return dt.replace(year=dt.year + 1, month=1)
            return dt.replace(month=dt.month + 1)
    
    @property
    def granularity(self) -> PartitionGranularity:
        return self._granularity
    
    @property
    def retention_days(self) -> Optional[int]:
        return self._retention_days
