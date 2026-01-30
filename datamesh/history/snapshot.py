"""
History Snapshot: Immutable Snapshot Control

Provides:
- Copy-on-write snapshot creation
- SHA-256 content validation
- Regulatory compliance (GDPR, data retention)
- Snapshot comparison and diff

Design:
    Snapshots are point-in-time immutable captures:
    - Monthly snapshots for compliance
    - On-demand snapshots before migrations
    - Verified with content hashes
    - Archived to object storage
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp, ContentHash


# =============================================================================
# SNAPSHOT STATUS
# =============================================================================
class SnapshotStatus(Enum):
    """Status of snapshot creation."""
    PENDING = auto()     # Scheduled but not started
    CREATING = auto()    # In progress
    VERIFYING = auto()   # Validating content hash
    COMPLETED = auto()   # Successfully created
    FAILED = auto()      # Creation failed


# =============================================================================
# SNAPSHOT METADATA
# =============================================================================
@dataclass(slots=True)
class SnapshotMetadata:
    """
    Metadata for a history snapshot.
    
    Tracks creation status and content verification.
    """
    snapshot_id: UUID = field(default_factory=uuid4)
    entity_id: Optional[EntityId] = None
    
    # Timing
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    
    # Content info
    bucket_ids: list[str] = field(default_factory=list)
    interaction_count: int = 0
    total_bytes: int = 0
    
    # Verification
    content_hash: Optional[str] = None  # SHA-256 of snapshot
    status: SnapshotStatus = SnapshotStatus.PENDING
    error_message: Optional[str] = None
    
    # Storage
    storage_path: Optional[str] = None
    archive_tier: str = "standard"  # standard, glacier, deep_archive
    
    # Compliance
    retention_days: int = 365 * 7  # 7 years default
    compliance_tags: list[str] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if snapshot is complete."""
        return self.status == SnapshotStatus.COMPLETED
    
    @property
    def is_verified(self) -> bool:
        """Check if snapshot has verified hash."""
        return self.is_complete and self.content_hash is not None
    
    @property
    def age_days(self) -> float:
        """Snapshot age in days."""
        if self.completed_at is None:
            return 0.0
        now = datetime.now(timezone.utc)
        return (now - self.completed_at).total_seconds() / 86400
    
    @property
    def expires_at(self) -> Optional[datetime]:
        """Expiration date based on retention."""
        if self.completed_at is None:
            return None
        from datetime import timedelta
        return self.completed_at + timedelta(days=self.retention_days)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "snapshot_id": str(self.snapshot_id),
            "entity_id": str(self.entity_id.value) if self.entity_id else None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "bucket_ids": self.bucket_ids,
            "interaction_count": self.interaction_count,
            "total_bytes": self.total_bytes,
            "content_hash": self.content_hash,
            "status": self.status.name,
            "storage_path": self.storage_path,
            "retention_days": self.retention_days,
        }


# =============================================================================
# HISTORY SNAPSHOT
# =============================================================================
@dataclass
class HistorySnapshot:
    """
    Immutable snapshot of conversation history.
    
    Contains serialized history data and verification info.
    """
    metadata: SnapshotMetadata
    data: bytes = b""  # Serialized snapshot content
    
    @property
    def snapshot_id(self) -> UUID:
        """Snapshot ID."""
        return self.metadata.snapshot_id
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of snapshot data."""
        return hashlib.sha256(self.data).hexdigest()
    
    def verify(self) -> Result[bool, str]:
        """Verify snapshot integrity."""
        if not self.metadata.content_hash:
            return Err("Snapshot has no content hash")
        
        computed = self.compute_hash()
        if computed != self.metadata.content_hash:
            return Err(
                f"Hash mismatch: expected {self.metadata.content_hash}, "
                f"got {computed}"
            )
        
        return Ok(True)


# =============================================================================
# SNAPSHOT MANAGER
# =============================================================================
class SnapshotManager:
    """
    Manages history snapshot lifecycle.
    
    Features:
        - Scheduled and on-demand snapshot creation
        - Content verification with SHA-256
        - Retention policy enforcement
        - Archive tier management
    
    Usage:
        manager = SnapshotManager()
        
        # Create snapshot
        result = await manager.create_snapshot(
            entity_id=entity_id,
            bucket_ids=["2024-01", "2024-02"],
        )
        
        # Verify snapshot
        snapshot = await manager.get_snapshot(snapshot_id)
        verification = snapshot.verify()
    """
    
    __slots__ = (
        "_snapshots", "_lock", "_storage_backend",
        "_default_retention_days",
    )
    
    def __init__(
        self,
        storage_backend: Optional[Any] = None,
        default_retention_days: int = 365 * 7,
    ) -> None:
        self._snapshots: dict[UUID, HistorySnapshot] = {}
        self._lock = asyncio.Lock()
        self._storage_backend = storage_backend
        self._default_retention_days = default_retention_days
    
    async def create_snapshot(
        self,
        entity_id: EntityId,
        bucket_ids: list[str],
        data: bytes = b"",
        compliance_tags: Optional[list[str]] = None,
    ) -> Result[SnapshotMetadata, str]:
        """
        Create new history snapshot.
        
        In production, this would:
        1. Lock the buckets
        2. Serialize all interactions
        3. Compute content hash
        4. Upload to object storage
        5. Update metadata
        """
        async with self._lock:
            metadata = SnapshotMetadata(
                entity_id=entity_id,
                bucket_ids=bucket_ids,
                retention_days=self._default_retention_days,
                compliance_tags=compliance_tags or [],
            )
            
            try:
                metadata.status = SnapshotStatus.CREATING
                
                # Simulate snapshot creation
                # In production: serialize interactions from buckets
                snapshot_data = data or self._serialize_placeholder(
                    entity_id, bucket_ids
                )
                
                metadata.total_bytes = len(snapshot_data)
                metadata.status = SnapshotStatus.VERIFYING
                
                # Compute hash
                content_hash = hashlib.sha256(snapshot_data).hexdigest()
                metadata.content_hash = content_hash
                
                # Generate storage path
                metadata.storage_path = (
                    f"snapshots/{entity_id.value}/"
                    f"{metadata.snapshot_id}.snapshot"
                )
                
                # Create snapshot object
                snapshot = HistorySnapshot(
                    metadata=metadata,
                    data=snapshot_data,
                )
                
                # Store (in production: upload to S3)
                self._snapshots[metadata.snapshot_id] = snapshot
                
                metadata.status = SnapshotStatus.COMPLETED
                metadata.completed_at = datetime.now(timezone.utc)
                
                return Ok(metadata)
                
            except Exception as e:
                metadata.status = SnapshotStatus.FAILED
                metadata.error_message = str(e)
                return Err(f"Snapshot creation failed: {e}")
    
    def _serialize_placeholder(
        self,
        entity_id: EntityId,
        bucket_ids: list[str],
    ) -> bytes:
        """Create placeholder snapshot data."""
        import json
        data = {
            "entity_id": str(entity_id.value),
            "bucket_ids": bucket_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interactions": [],
        }
        return json.dumps(data).encode("utf-8")
    
    async def get_snapshot(
        self,
        snapshot_id: UUID,
    ) -> Result[Optional[HistorySnapshot], str]:
        """Retrieve snapshot by ID."""
        async with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            return Ok(snapshot)
    
    async def get_metadata(
        self,
        snapshot_id: UUID,
    ) -> Result[Optional[SnapshotMetadata], str]:
        """Get snapshot metadata only (no data)."""
        async with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            if snapshot:
                return Ok(snapshot.metadata)
            return Ok(None)
    
    async def verify_snapshot(
        self,
        snapshot_id: UUID,
    ) -> Result[bool, str]:
        """Verify snapshot integrity."""
        async with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                return Err(f"Snapshot {snapshot_id} not found")
            
            return snapshot.verify()
    
    async def list_snapshots(
        self,
        entity_id: Optional[EntityId] = None,
        status_filter: Optional[SnapshotStatus] = None,
        limit: int = 100,
    ) -> list[SnapshotMetadata]:
        """List snapshots with optional filtering."""
        async with self._lock:
            snapshots = list(self._snapshots.values())
            
            if entity_id:
                snapshots = [
                    s for s in snapshots
                    if s.metadata.entity_id
                    and s.metadata.entity_id.value == entity_id.value
                ]
            
            if status_filter:
                snapshots = [
                    s for s in snapshots
                    if s.metadata.status == status_filter
                ]
            
            # Sort by created_at descending
            snapshots.sort(
                key=lambda s: s.metadata.created_at,
                reverse=True,
            )
            
            return [s.metadata for s in snapshots[:limit]]
    
    async def delete_snapshot(
        self,
        snapshot_id: UUID,
    ) -> Result[bool, str]:
        """Delete snapshot."""
        async with self._lock:
            if snapshot_id not in self._snapshots:
                return Ok(False)
            
            del self._snapshots[snapshot_id]
            return Ok(True)
    
    async def cleanup_expired(self) -> list[UUID]:
        """Remove expired snapshots based on retention policy."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            expired: list[UUID] = []
            
            for snapshot_id, snapshot in list(self._snapshots.items()):
                expires = snapshot.metadata.expires_at
                if expires and now > expires:
                    del self._snapshots[snapshot_id]
                    expired.append(snapshot_id)
            
            return expired
    
    async def compare_snapshots(
        self,
        snapshot_a_id: UUID,
        snapshot_b_id: UUID,
    ) -> Result[dict[str, Any], str]:
        """Compare two snapshots."""
        async with self._lock:
            a = self._snapshots.get(snapshot_a_id)
            b = self._snapshots.get(snapshot_b_id)
            
            if not a or not b:
                return Err("One or both snapshots not found")
            
            return Ok({
                "snapshot_a": str(snapshot_a_id),
                "snapshot_b": str(snapshot_b_id),
                "same_hash": a.metadata.content_hash == b.metadata.content_hash,
                "size_diff": len(b.data) - len(a.data),
                "bucket_diff": {
                    "added": [
                        bid for bid in b.metadata.bucket_ids
                        if bid not in a.metadata.bucket_ids
                    ],
                    "removed": [
                        bid for bid in a.metadata.bucket_ids
                        if bid not in b.metadata.bucket_ids
                    ],
                },
            })
    
    @property
    def snapshot_count(self) -> int:
        """Total snapshot count."""
        return len(self._snapshots)
