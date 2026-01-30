"""
Session Cache: AP-Tier Hot Session Payload

Provides high-performance session payload storage:
- In-memory hash structure (Redis-like semantics)
- TTL-based expiration (24h default, renewable via heartbeat)
- Compressed context serialization (msgpack + lz4)
- O(1) access for active sessions

Data Model:
    Key: session:{entity_id}:{session_id}
    Fields:
        - active_turn_count: INT
        - accumulated_tokens: INT
        - last_interaction_ts: TIMESTAMP
        - compressed_context: BYTES (LZ4-compressed msgpack)
        - metadata: BYTES

Design:
    AP tier optimizes for availability over consistency.
    Payload may be slightly stale; registry is source of truth.
    Eviction triggers memory consolidation to LTM.
"""

from __future__ import annotations

import asyncio
import hashlib
import lz4.frame
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from uuid import UUID

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, Timestamp,
)
from datamesh.core.errors import StorageError


# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_TTL_SECONDS: int = 86400  # 24 hours
MAX_PAYLOAD_BYTES: int = 1024 * 1024  # 1MB limit
COMPRESSION_THRESHOLD: int = 1024  # Compress if > 1KB


# =============================================================================
# SESSION PAYLOAD MODEL
# =============================================================================
@dataclass(slots=True)
class SessionPayload:
    """
    AP-tier session payload.
    
    Contains hot session data for active interactions.
    Compressed for memory efficiency.
    """
    # Identity
    entity_id: EntityId
    session_id: UUID
    
    # Counters
    active_turn_count: int = 0
    accumulated_tokens: int = 0
    
    # Timestamps
    created_at: Timestamp = field(default_factory=Timestamp.now)
    last_interaction_ts: Timestamp = field(default_factory=Timestamp.now)
    expires_at: Timestamp = field(
        default_factory=lambda: Timestamp(
            nanos=Timestamp.now().nanos + DEFAULT_TTL_SECONDS * 1_000_000_000
        )
    )
    
    # Context data (serialized turns)
    context_data: bytes = b""  # Raw or compressed
    is_compressed: bool = False
    
    # Extended metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def cache_key(self) -> str:
        """Generate cache key."""
        return f"session:{self.entity_id.value}:{self.session_id}"
    
    @property
    def is_expired(self) -> bool:
        """Check if payload has expired."""
        return Timestamp.now().nanos >= self.expires_at.nanos
    
    @property
    def ttl_remaining_seconds(self) -> float:
        """Remaining TTL in seconds."""
        remaining_nanos = self.expires_at.nanos - Timestamp.now().nanos
        return max(0, remaining_nanos / 1_000_000_000)
    
    def renew(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Extend TTL (heartbeat)."""
        self.expires_at = Timestamp(
            nanos=Timestamp.now().nanos + ttl_seconds * 1_000_000_000
        )
    
    def touch(self) -> None:
        """Update last interaction timestamp."""
        self.last_interaction_ts = Timestamp.now()
    
    def compress_context(self) -> None:
        """Compress context data with LZ4."""
        if self.is_compressed or len(self.context_data) < COMPRESSION_THRESHOLD:
            return
        
        self.context_data = lz4.frame.compress(self.context_data)
        self.is_compressed = True
    
    def decompress_context(self) -> bytes:
        """Get decompressed context data."""
        if not self.is_compressed:
            return self.context_data
        return lz4.frame.decompress(self.context_data)
    
    def set_context(self, data: bytes, compress: bool = True) -> None:
        """Set context data with optional compression."""
        if compress and len(data) >= COMPRESSION_THRESHOLD:
            self.context_data = lz4.frame.compress(data)
            self.is_compressed = True
        else:
            self.context_data = data
            self.is_compressed = False
    
    def serialize(self) -> bytes:
        """Serialize payload to bytes."""
        # Header: version (1) + flags (1) + turn_count (4) + tokens (4) + timestamps (8x3)
        flags = 0x01 if self.is_compressed else 0x00
        header = struct.pack(
            ">BBII QQQ",
            0x01,  # Version
            flags,
            self.active_turn_count,
            self.accumulated_tokens,
            self.created_at.nanos,
            self.last_interaction_ts.nanos,
            self.expires_at.nanos,
        )
        
        # Entity ID (16 bytes) + Session ID (16 bytes)
        ids = self.entity_id.value.bytes + self.session_id.bytes
        
        # Context length (4 bytes) + context
        context_header = struct.pack(">I", len(self.context_data))
        
        return header + ids + context_header + self.context_data
    
    @classmethod
    def deserialize(cls, data: bytes) -> SessionPayload:
        """Deserialize payload from bytes."""
        # Parse header
        version, flags, turn_count, tokens, created_ns, last_ns, expires_ns = (
            struct.unpack(">BBII QQQ", data[:34])
        )
        
        # Parse IDs
        entity_bytes = data[34:50]
        session_bytes = data[50:66]
        
        # Parse context
        context_len = struct.unpack(">I", data[66:70])[0]
        context_data = data[70:70 + context_len]
        
        return cls(
            entity_id=EntityId(value=UUID(bytes=entity_bytes)),
            session_id=UUID(bytes=session_bytes),
            active_turn_count=turn_count,
            accumulated_tokens=tokens,
            created_at=Timestamp(nanos=created_ns),
            last_interaction_ts=Timestamp(nanos=last_ns),
            expires_at=Timestamp(nanos=expires_ns),
            context_data=context_data,
            is_compressed=(flags & 0x01) != 0,
        )


# =============================================================================
# CACHE STATISTICS
# =============================================================================
@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_bytes: int = 0
    entry_count: int = 0
    compression_ratio: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# SESSION CACHE IMPLEMENTATION
# =============================================================================
class SessionCache:
    """
    AP-tier hot session cache.
    
    Provides O(1) access to active session payloads with:
    - TTL-based expiration
    - LRU eviction when capacity exceeded
    - Compression for memory efficiency
    - Eviction callbacks for LTM consolidation
    
    Usage:
        cache = SessionCache(max_entries=10000)
        cache.on_eviction(lambda p: consolidate_to_ltm(p))
        
        # Store payload
        payload = SessionPayload(entity_id=eid, session_id=sid)
        await cache.put(payload)
        
        # Retrieve with TTL renewal
        result = await cache.get(cache_key, renew_ttl=True)
    """
    
    __slots__ = (
        "_store", "_max_entries", "_max_bytes",
        "_stats", "_eviction_callbacks", "_lock",
        "_access_order",
    )
    
    def __init__(
        self,
        max_entries: int = 10000,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
    ) -> None:
        self._store: dict[str, SessionPayload] = {}
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._stats = CacheStats()
        self._eviction_callbacks: list[Callable[[SessionPayload], None]] = []
        self._lock = asyncio.Lock()
        self._access_order: list[str] = []  # LRU tracking
    
    def on_eviction(
        self,
        callback: Callable[[SessionPayload], None],
    ) -> None:
        """Register callback for eviction events."""
        self._eviction_callbacks.append(callback)
    
    async def put(
        self,
        payload: SessionPayload,
        compress: bool = True,
    ) -> Result[None, StorageError]:
        """
        Store session payload.
        
        Evicts LRU entries if capacity exceeded.
        """
        async with self._lock:
            key = payload.cache_key
            
            # Compress if beneficial
            if compress:
                payload.compress_context()
            
            # Check size limit
            serialized = payload.serialize()
            if len(serialized) > MAX_PAYLOAD_BYTES:
                return Err(StorageError.disk_full(
                    path="session_cache",
                    required_bytes=len(serialized),
                    available_bytes=MAX_PAYLOAD_BYTES,
                ))
            
            # Evict if necessary
            while (
                len(self._store) >= self._max_entries
                or self._stats.total_bytes + len(serialized) > self._max_bytes
            ):
                if not self._access_order:
                    break
                await self._evict_lru()
            
            # Update or insert
            if key in self._store:
                old_payload = self._store[key]
                self._stats.total_bytes -= len(old_payload.serialize())
            
            self._store[key] = payload
            self._stats.total_bytes += len(serialized)
            self._stats.entry_count = len(self._store)
            
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return Ok(None)
    
    async def get(
        self,
        key: str,
        renew_ttl: bool = True,
    ) -> Result[Optional[SessionPayload], StorageError]:
        """
        Retrieve session payload.
        
        Args:
            key: Cache key
            renew_ttl: Extend TTL on access (heartbeat)
        """
        async with self._lock:
            payload = self._store.get(key)
            
            if payload is None:
                self._stats.misses += 1
                return Ok(None)
            
            # Check expiration
            if payload.is_expired:
                await self._expire(key, payload)
                self._stats.misses += 1
                return Ok(None)
            
            self._stats.hits += 1
            
            # Renew TTL
            if renew_ttl:
                payload.renew()
            
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return Ok(payload)
    
    async def delete(self, key: str) -> Result[bool, StorageError]:
        """Remove payload from cache."""
        async with self._lock:
            if key not in self._store:
                return Ok(False)
            
            payload = self._store.pop(key)
            self._stats.total_bytes -= len(payload.serialize())
            self._stats.entry_count = len(self._store)
            
            if key in self._access_order:
                self._access_order.remove(key)
            
            return Ok(True)
    
    async def touch(self, key: str) -> Result[bool, StorageError]:
        """Update last interaction timestamp (heartbeat)."""
        async with self._lock:
            payload = self._store.get(key)
            if payload is None:
                return Ok(False)
            
            if payload.is_expired:
                await self._expire(key, payload)
                return Ok(False)
            
            payload.touch()
            payload.renew()
            
            # Update LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return Ok(True)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return
        
        key = self._access_order.pop(0)
        payload = self._store.pop(key, None)
        
        if payload:
            self._stats.total_bytes -= len(payload.serialize())
            self._stats.evictions += 1
            self._stats.entry_count = len(self._store)
            
            # Notify callbacks
            for callback in self._eviction_callbacks:
                try:
                    callback(payload)
                except Exception:
                    pass
    
    async def _expire(self, key: str, payload: SessionPayload) -> None:
        """Handle expired entry."""
        self._store.pop(key, None)
        self._stats.total_bytes -= len(payload.serialize())
        self._stats.expirations += 1
        self._stats.entry_count = len(self._store)
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        # Notify callbacks (expired entries also trigger consolidation)
        for callback in self._eviction_callbacks:
            try:
                callback(payload)
            except Exception:
                pass
    
    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        async with self._lock:
            expired_keys = [
                k for k, p in self._store.items()
                if p.is_expired
            ]
            
            for key in expired_keys:
                payload = self._store.pop(key)
                await self._expire(key, payload)
            
            return len(expired_keys)
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def contains(self, key: str) -> bool:
        """Check if key exists (without updating LRU)."""
        return key in self._store
