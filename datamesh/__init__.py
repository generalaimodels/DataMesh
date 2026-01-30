"""
Planetary-Scale Conversational Data Mesh Architecture

A polyglot persistence system governed by CAP theorem partitioning:
- CP Subsystem: Metadata control plane (PostgreSQL) with serializable isolation
- AP Subsystem: Content data plane (SQLite + LSM) for high-throughput writes
- Object Storage: Content-addressable storage for large artifacts
- Session Manager: Distributed state orchestration with leasing
- History Manager: Temporal data architecture with partitioning
- Memory Manager: Hierarchical cognitive storage (STM/LTM)

Performance Targets:
- Write throughput: 1M rows/sec
- Read latency (cached): P99 < 50ms
- Read latency (uncached): P99 < 500ms
- Availability: 99.999% (metadata), 99.99% (content)

Author: Planetary Data Mesh Engineering
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Planetary Data Mesh Engineering"

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================
from datamesh.core.types import (
    Result,
    Ok,
    Err,
    EntityId,
    ConversationId,
    InstructionId,
    Timestamp,
    ByteRange,
    ComplianceTier,
    GeoRegion,
)
from datamesh.core.errors import (
    DataMeshError,
    StorageError,
    IngestionError,
    QueryError,
    ShardingError,
)
from datamesh.core.config import DataMeshConfig

# Session Manager exports
from datamesh.session import (
    SessionState,
    SessionStateMachine,
    SessionRegistry,
    SessionCache,
    DistributedLease,
    SessionAffinity,
)

# History Manager exports
from datamesh.history import (
    ConversationTimeline,
    HistoryPartitioner,
    HistoryCompactor,
    HistoryCursor,
    SnapshotManager,
)

# Memory Manager exports
from datamesh.memory import (
    WorkingSet,
    ContextWindow,
    VectorStore,
    GraphStore,
    EpisodicMemory,
    ConsolidationPipeline,
    MemoryHierarchy,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Result monad
    "Result",
    "Ok",
    "Err",
    # Identity types
    "EntityId",
    "ConversationId",
    "InstructionId",
    "Timestamp",
    "ByteRange",
    # Enums
    "ComplianceTier",
    "GeoRegion",
    # Errors
    "DataMeshError",
    "StorageError",
    "IngestionError",
    "QueryError",
    "ShardingError",
    # Config
    "DataMeshConfig",
    # Session Manager
    "SessionState",
    "SessionStateMachine",
    "SessionRegistry",
    "SessionCache",
    "DistributedLease",
    "SessionAffinity",
    # History Manager
    "ConversationTimeline",
    "HistoryPartitioner",
    "HistoryCompactor",
    "HistoryCursor",
    "SnapshotManager",
    # Memory Manager
    "WorkingSet",
    "ContextWindow",
    "VectorStore",
    "GraphStore",
    "EpisodicMemory",
    "ConsolidationPipeline",
    "MemoryHierarchy",
]

