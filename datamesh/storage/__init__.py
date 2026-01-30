"""
Storage Module: Unified Database Abstraction Layer

Provides:
- Protocol definitions for pluggable backends
- In-memory implementations for development/testing
- CP (Control Plane) and AP (Availability Plane) subsystems
- Object store for blob storage
"""

# Protocol definitions
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

# In-memory backends
from datamesh.storage.backends import (
    InMemoryCPStore,
    InMemoryAPStore,
    InMemoryObjectStore,
    ObjectMetadata,
)

# Legacy CP/AP subsystems
from datamesh.storage.cp.engine import CPEngine
from datamesh.storage.cp.repositories import (
    ConversationRepository,
    InstructionRepository,
)
from datamesh.storage.ap.engine import APEngine
from datamesh.storage.ap.repositories import ResponseRepository

__all__ = [
    # Protocols
    "ConsistencyLevel",
    "IsolationLevel",
    "OperationType",
    "OperationMetadata",
    "TransactionHandle",
    "ReplicaInfo",
    "ShardInfo",
    "DatabaseProtocol",
    "BatchProtocol",
    "TransactionProtocol",
    "TTLProtocol",
    "VersionedProtocol",
    "StreamingProtocol",
    # In-memory backends
    "InMemoryCPStore",
    "InMemoryAPStore",
    "InMemoryObjectStore",
    "ObjectMetadata",
    # Legacy
    "CPEngine",
    "APEngine",
    "ConversationRepository",
    "InstructionRepository",
    "ResponseRepository",
]
