"""
Storage Module: Unified Database Abstraction Layer
===================================================

Provides:
- Protocol definitions for pluggable backends
- In-memory implementations for development/testing
- Production backends (Redis, S3, Spanner)
- Factory functions for backend selection
- CP (Control Plane) and AP (Availability Plane) subsystems
- Object store for blob storage

Design Principles:
-----------------
1. **Backend Agnostic**: Same interface for in-memory and production
2. **Factory Pattern**: Runtime backend selection via configuration
3. **Lazy Loading**: Production dependencies loaded only when needed
4. **Result Monad**: No exceptions for control flow

Example:
    >>> # Development (in-memory)
    >>> cp_store = create_cp_store()
    >>> ap_store = create_ap_store()
    
    >>> # Production (configured)
    >>> from datamesh.storage.config import RedisConfig
    >>> ap_store = create_ap_store(RedisConfig(host="redis.prod"))
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

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

# In-memory backends (always available)
from datamesh.storage.backends import (
    InMemoryCPStore,
    InMemoryAPStore,
    InMemoryObjectStore,
    ObjectMetadata,
)

# Configuration
from datamesh.storage.config import (
    BackendType,
    RedisMode,
    ConsistencyPreference,
    RedisConfig,
    S3Config,
    SpannerConfig,
    StorageConfig,
)

# Lazy imports for production backends
if TYPE_CHECKING:
    from datamesh.storage.redis_store import RedisAPStore
    from datamesh.storage.s3_store import S3ObjectStore


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cp_store(
    config: Optional[SpannerConfig] = None,
) -> Any:
    """
    Create Control Plane store.
    
    Factory function for CP-tier storage (strong consistency).
    Returns in-memory implementation by default.
    
    Args:
        config: Optional Spanner configuration for production.
    
    Returns:
        InMemoryCPStore: If config is None (development).
        SpannerCPStore: If config is provided (future).
    
    Example:
        >>> # Development
        >>> store = create_cp_store()
        
        >>> # Production (Spanner)
        >>> store = create_cp_store(SpannerConfig(...))
    """
    if config is not None:
        # Spanner backend not yet implemented
        # from datamesh.storage.spanner_store import SpannerCPStore
        # return SpannerCPStore(config)
        raise NotImplementedError(
            "Spanner backend not yet implemented. Use in-memory for now."
        )
    
    return InMemoryCPStore()


def create_ap_store(
    config: Optional[RedisConfig] = None,
    max_entries: int = 100_000,
    max_bytes: int = 1024 * 1024 * 1024,  # 1 GiB
) -> Any:
    """
    Create Availability Plane store.
    
    Factory function for AP-tier storage (high availability).
    Returns in-memory implementation by default.
    
    Args:
        config: Optional Redis configuration for production.
        max_entries: Maximum entries for in-memory store.
        max_bytes: Maximum bytes for in-memory store.
    
    Returns:
        InMemoryAPStore: If config is None (development).
        RedisAPStore: If config is provided (production).
    
    Example:
        >>> # Development
        >>> store = create_ap_store()
        
        >>> # Production (Redis)
        >>> store = create_ap_store(RedisConfig(host="redis.prod"))
    """
    if config is not None:
        from datamesh.storage.redis_store import RedisAPStore
        return RedisAPStore(config)
    
    return InMemoryAPStore(max_entries=max_entries, max_bytes=max_bytes)


def create_object_store(
    config: Optional[S3Config] = None,
) -> Any:
    """
    Create Object Store.
    
    Factory function for blob storage (large objects).
    Returns in-memory implementation by default.
    
    Args:
        config: Optional S3 configuration for production.
    
    Returns:
        InMemoryObjectStore: If config is None (development).
        S3ObjectStore: If config is provided (production).
    
    Example:
        >>> # Development
        >>> store = create_object_store()
        
        >>> # Production (S3)
        >>> store = create_object_store(S3Config(bucket_name="my-bucket"))
    """
    if config is not None:
        from datamesh.storage.s3_store import S3ObjectStore
        return S3ObjectStore(config)
    
    return InMemoryObjectStore()


def create_stores_from_config(config: StorageConfig) -> dict:
    """
    Create all stores from unified configuration.
    
    Args:
        config: StorageConfig with backend specifications.
    
    Returns:
        Dict with keys: "cp_store", "ap_store", "object_store"
    
    Example:
        >>> config = StorageConfig.for_development()
        >>> stores = create_stores_from_config(config)
        >>> cp_store = stores["cp_store"]
    """
    cp_store = create_cp_store(config.spanner_config)
    ap_store = create_ap_store(config.redis_config)
    object_store = create_object_store(config.s3_config)
    
    return {
        "cp_store": cp_store,
        "ap_store": ap_store,
        "object_store": object_store,
    }


# =============================================================================
# LEGACY SUBSYSTEMS
# =============================================================================

# Legacy CP/AP subsystems (for backward compatibility)
from datamesh.storage.cp.engine import CPEngine
from datamesh.storage.cp.repositories import (
    ConversationRepository,
    InstructionRepository,
)
from datamesh.storage.ap.engine import APEngine
from datamesh.storage.ap.repositories import ResponseRepository


# =============================================================================
# MODULE EXPORTS
# =============================================================================

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
    # Configuration
    "BackendType",
    "RedisMode",
    "ConsistencyPreference",
    "RedisConfig",
    "S3Config",
    "SpannerConfig",
    "StorageConfig",
    # In-memory backends
    "InMemoryCPStore",
    "InMemoryAPStore",
    "InMemoryObjectStore",
    "ObjectMetadata",
    # Factory functions
    "create_cp_store",
    "create_ap_store",
    "create_object_store",
    "create_stores_from_config",
    # Legacy
    "CPEngine",
    "APEngine",
    "ConversationRepository",
    "InstructionRepository",
    "ResponseRepository",
]

