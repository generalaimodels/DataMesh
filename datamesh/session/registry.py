"""
Session Registry: CP-Tier Metadata Storage

Provides durable session metadata storage with:
- Composite primary key (entity_id, session_id)
- Optimistic concurrency via consistency_boundary_version
- Soft delete support for regulatory compliance
- Time-range queries for analytics

Storage Model:
    Session metadata resides in CP tier (PostgreSQL/Spanner)
    for strong consistency guarantees. Hot payload data is
    stored separately in AP tier (SessionCache).

Schema:
    session_registry (
        entity_id UUID,
        session_id UUID,
        current_state ENUM,
        context_window_ref STRING,
        memory_consolidation_checkpoint TIMESTAMP,
        geo_affinity STRING,
        consistency_boundary_version INT64,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        terminated_at TIMESTAMP NULL,
        PRIMARY KEY (entity_id, session_id)
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, Timestamp, GeoRegion,
    PaginationCursor,
)
from datamesh.core.errors import StorageError
from datamesh.session.state_machine import SessionState


# =============================================================================
# SESSION METADATA MODEL
# =============================================================================
@dataclass(slots=True)
class SessionMetadata:
    """
    CP-tier session metadata record.
    
    Contains durable session state and references to
    AP-tier payload storage. Immutable fields are frozen
    after creation; mutable fields track session evolution.
    """
    # Immutable identity
    entity_id: EntityId
    session_id: UUID
    
    # Mutable state
    current_state: SessionState = SessionState.INITIATED
    context_window_ref: Optional[str] = None  # AP-tier cache key
    memory_consolidation_checkpoint: Optional[Timestamp] = None
    geo_affinity: GeoRegion = GeoRegion.US_EAST
    consistency_boundary_version: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    terminated_at: Optional[datetime] = None
    
    # Extended metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        if self.terminated_at and self.current_state != SessionState.TERMINATED:
            raise ValueError("terminated_at set but state is not TERMINATED")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "entity_id": str(self.entity_id.value),
            "session_id": str(self.session_id),
            "current_state": self.current_state.name,
            "context_window_ref": self.context_window_ref,
            "memory_consolidation_checkpoint": (
                self.memory_consolidation_checkpoint.nanos
                if self.memory_consolidation_checkpoint else None
            ),
            "geo_affinity": self.geo_affinity.value,
            "consistency_boundary_version": self.consistency_boundary_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "terminated_at": (
                self.terminated_at.isoformat()
                if self.terminated_at else None
            ),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        """Deserialize from dictionary."""
        return cls(
            entity_id=EntityId(value=UUID(data["entity_id"])),
            session_id=UUID(data["session_id"]),
            current_state=SessionState[data["current_state"]],
            context_window_ref=data.get("context_window_ref"),
            memory_consolidation_checkpoint=(
                Timestamp(nanos=data["memory_consolidation_checkpoint"])
                if data.get("memory_consolidation_checkpoint") else None
            ),
            geo_affinity=GeoRegion(data.get("geo_affinity", "US_EAST")),
            consistency_boundary_version=data.get("consistency_boundary_version", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            terminated_at=(
                datetime.fromisoformat(data["terminated_at"])
                if data.get("terminated_at") else None
            ),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def is_active(self) -> bool:
        """Check if session is in active state."""
        return self.current_state == SessionState.ACTIVE
    
    @property
    def is_terminated(self) -> bool:
        """Check if session is terminated."""
        return self.current_state == SessionState.TERMINATED
    
    @property
    def cache_key(self) -> str:
        """Generate AP-tier cache key."""
        return f"session:{self.entity_id.value}:{self.session_id}"


# =============================================================================
# SESSION REGISTRY (IN-MEMORY IMPLEMENTATION)
# =============================================================================
class SessionRegistry:
    """
    CP-tier session metadata repository.
    
    Provides CRUD operations with:
    - Optimistic concurrency control
    - Cursor-based pagination
    - Time-range queries
    - Soft delete support
    
    Note: This implementation uses in-memory storage.
    Production would use PostgreSQL/Spanner via CPEngine.
    
    Usage:
        registry = SessionRegistry()
        
        # Create session
        metadata = SessionMetadata(
            entity_id=EntityId.generate(),
            session_id=uuid4(),
        )
        result = await registry.create(metadata)
        
        # Update with optimistic locking
        metadata.current_state = SessionState.ACTIVE
        result = await registry.update(metadata, expected_version=0)
    """
    
    __slots__ = ("_store", "_lock")
    
    def __init__(self) -> None:
        # In-memory store: (entity_id, session_id) -> SessionMetadata
        self._store: dict[tuple[UUID, UUID], SessionMetadata] = {}
        self._lock = asyncio.Lock()
    
    async def create(
        self,
        metadata: SessionMetadata,
    ) -> Result[SessionMetadata, StorageError]:
        """
        Create new session metadata.
        
        Returns Err if session already exists.
        """
        async with self._lock:
            key = (metadata.entity_id.value, metadata.session_id)
            
            if key in self._store:
                return Err(StorageError.constraint_violation(
                    constraint="pk_session_registry",
                    table="session_registry",
                ))
            
            # Initialize version and timestamps
            metadata.consistency_boundary_version = 0
            metadata.created_at = datetime.now(timezone.utc)
            metadata.updated_at = metadata.created_at
            metadata.context_window_ref = metadata.cache_key
            
            self._store[key] = metadata
            return Ok(metadata)
    
    async def get(
        self,
        entity_id: EntityId,
        session_id: UUID,
    ) -> Result[Optional[SessionMetadata], StorageError]:
        """
        Retrieve session metadata by composite key.
        
        Returns None if not found (not an error).
        """
        async with self._lock:
            key = (entity_id.value, session_id)
            metadata = self._store.get(key)
            return Ok(metadata)
    
    async def update(
        self,
        metadata: SessionMetadata,
        expected_version: Optional[int] = None,
    ) -> Result[SessionMetadata, StorageError]:
        """
        Update session metadata with optimistic locking.
        
        Args:
            metadata: Updated metadata object
            expected_version: Expected current version (for OCC)
        
        Returns:
            Err if version mismatch (concurrent modification)
        """
        async with self._lock:
            key = (metadata.entity_id.value, metadata.session_id)
            
            existing = self._store.get(key)
            if existing is None:
                return Err(StorageError.constraint_violation(
                    constraint="fk_session_exists",
                    table="session_registry",
                ))
            
            # Optimistic concurrency check
            if expected_version is not None:
                if existing.consistency_boundary_version != expected_version:
                    return Err(StorageError.serialization_conflict(
                        transaction_id=f"session:{metadata.session_id}",
                    ))
            
            # Update version and timestamp
            metadata.consistency_boundary_version = (
                existing.consistency_boundary_version + 1
            )
            metadata.updated_at = datetime.now(timezone.utc)
            
            # Handle termination
            if metadata.current_state == SessionState.TERMINATED:
                metadata.terminated_at = datetime.now(timezone.utc)
            
            self._store[key] = metadata
            return Ok(metadata)
    
    async def delete(
        self,
        entity_id: EntityId,
        session_id: UUID,
        soft: bool = True,
    ) -> Result[bool, StorageError]:
        """
        Delete session metadata.
        
        Args:
            soft: If True, mark as terminated instead of removing
        """
        async with self._lock:
            key = (entity_id.value, session_id)
            
            if key not in self._store:
                return Ok(False)
            
            if soft:
                metadata = self._store[key]
                metadata.current_state = SessionState.TERMINATED
                metadata.terminated_at = datetime.now(timezone.utc)
                metadata.consistency_boundary_version += 1
            else:
                del self._store[key]
            
            return Ok(True)
    
    async def list_by_entity(
        self,
        entity_id: EntityId,
        limit: int = 100,
        include_terminated: bool = False,
    ) -> Result[list[SessionMetadata], StorageError]:
        """
        List sessions for entity.
        
        Ordered by created_at descending (most recent first).
        """
        async with self._lock:
            sessions = [
                m for key, m in self._store.items()
                if key[0] == entity_id.value
                and (include_terminated or not m.is_terminated)
            ]
            
            # Sort by created_at descending
            sessions.sort(key=lambda s: s.created_at, reverse=True)
            
            return Ok(sessions[:limit])
    
    async def list_active(
        self,
        limit: int = 1000,
    ) -> Result[list[SessionMetadata], StorageError]:
        """List all active sessions across entities."""
        async with self._lock:
            active = [
                m for m in self._store.values()
                if m.is_active
            ]
            return Ok(active[:limit])
    
    async def count_by_entity(
        self,
        entity_id: EntityId,
        include_terminated: bool = False,
    ) -> Result[int, StorageError]:
        """Count sessions for entity."""
        async with self._lock:
            count = sum(
                1 for key, m in self._store.items()
                if key[0] == entity_id.value
                and (include_terminated or not m.is_terminated)
            )
            return Ok(count)
    
    async def get_by_state(
        self,
        state: SessionState,
        limit: int = 100,
    ) -> Result[list[SessionMetadata], StorageError]:
        """List sessions in specific state."""
        async with self._lock:
            matching = [
                m for m in self._store.values()
                if m.current_state == state
            ]
            return Ok(matching[:limit])
    
    @property
    def total_sessions(self) -> int:
        """Total session count (for metrics)."""
        return len(self._store)
