"""
Session Manager Module: Distributed State Orchestration

Provides:
- SessionStateMachine: Lifecycle FSM with guard conditions
- SessionRegistry: CP-tier metadata storage
- SessionCache: AP-tier hot session payload
- DistributedLease: Redlock-style distributed locking
- SessionAffinity: Geo-affinity and consistent hashing

Architecture:
- Hot Path: In-memory cache with 24h TTL
- Control Plane: Persistent registry with optimistic locking
- Coordination: Lease-based distributed locks
"""

from datamesh.session.state_machine import (
    SessionState,
    SessionStateMachine,
    SessionTransition,
    TransitionGuard,
)
from datamesh.session.registry import (
    SessionRegistry,
    SessionMetadata,
)
from datamesh.session.cache import (
    SessionCache,
    SessionPayload,
    CacheStats,
)
from datamesh.session.lease import (
    DistributedLease,
    LeaseHandle,
    LeaseConfig,
)
from datamesh.session.affinity import (
    SessionAffinity,
    SessionAffinity as SessionAffinityManager,  # Alias for backwards compatibility
    AffinityConfig,
    NodeHealth,
)

__all__ = [
    # State Machine
    "SessionState",
    "SessionStateMachine",
    "SessionTransition",
    "TransitionGuard",
    # Registry
    "SessionRegistry",
    "SessionMetadata",
    # Cache
    "SessionCache",
    "SessionPayload",
    "CacheStats",
    # Lease
    "DistributedLease",
    "LeaseHandle",
    "LeaseConfig",
    # Affinity
    "SessionAffinity",
    "SessionAffinityManager",
    "AffinityConfig",
    "NodeHealth",
]
