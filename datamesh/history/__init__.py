"""
History Manager Module: Temporal Data Architecture

Provides:
- Timeline: Time-series interaction storage
- Partitioner: Time-bucket partitioning strategy
- Compactor: Archival and compaction pipeline
- Cursor: Cursor-based pagination
- Snapshot: Immutable snapshot control

Architecture:
- Command Side: Append-only writes with time-bucket partitioning
- Query Side: Materialized views for recent history, timeline, analytics
- Hot/Warm/Cold tiering for cost-efficient storage
"""

from datamesh.history.timeline import (
    ConversationTimeline,
    Interaction,
    InteractionType,
    TimelineStats,
)
from datamesh.history.partitioner import (
    HistoryPartitioner,
    HistoryBucket,
    BucketGranularity,
    TierClassification,
)
from datamesh.history.compactor import (
    HistoryCompactor,
    CompactionJob,
    CompactionStats,
    ArchivalFormat,
)
from datamesh.history.cursor import (
    HistoryCursor,
    CursorEncoder,
    PaginatedResult,
)
from datamesh.history.snapshot import (
    HistorySnapshot,
    SnapshotManager,
    SnapshotMetadata,
)

__all__ = [
    # Timeline
    "ConversationTimeline",
    "Interaction",
    "InteractionType",
    "TimelineStats",
    # Partitioner
    "HistoryPartitioner",
    "HistoryBucket",
    "BucketGranularity",
    "TierClassification",
    # Compactor
    "HistoryCompactor",
    "CompactionJob",
    "CompactionStats",
    "ArchivalFormat",
    # Cursor
    "HistoryCursor",
    "CursorEncoder",
    "PaginatedResult",
    # Snapshot
    "HistorySnapshot",
    "SnapshotManager",
    "SnapshotMetadata",
]
