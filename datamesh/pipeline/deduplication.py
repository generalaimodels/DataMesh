"""
Deduplication Service: Idempotency Key Management

Provides exactly-once semantics via:
- Idempotency key validation
- Request hash computation
- TTL-based key expiration
- Duplicate detection with original response
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from datamesh.core.types import Result, Ok, Err, ContentHash, Timestamp, EntityId
from datamesh.core.errors import IngestionError
from datamesh.storage.cp.repositories import IdempotencyRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeduplicationResult:
    """Result of deduplication check."""
    is_duplicate: bool
    original_response: Optional[dict[str, Any]] = None
    idempotency_key: Optional[str] = None


class DeduplicationService:
    """
    Manages request deduplication via idempotency keys.
    
    Two-tier deduplication:
    1. Idempotency-Key header (explicit, client-provided)
    2. Content hash (implicit, computed from request body)
    
    Usage:
        service = DeduplicationService(repo)
        
        # Check before processing
        result = await service.check(idempotency_key, request_data)
        if result.is_duplicate:
            return result.original_response
        
        # Process request...
        
        # Record completion
        await service.complete(idempotency_key, response_data)
    """
    
    __slots__ = ("_repo", "_ttl_hours", "_local_cache", "_cache_lock")
    
    def __init__(
        self,
        repo: IdempotencyRepository,
        ttl_hours: int = 24,
    ) -> None:
        self._repo = repo
        self._ttl_hours = ttl_hours
        
        # Local LRU cache for hot keys
        self._local_cache: dict[str, tuple[Timestamp, dict[str, Any]]] = {}
        self._cache_lock = asyncio.Lock()
    
    async def check(
        self,
        idempotency_key: str,
        request_data: bytes,
        entity_id: Optional[EntityId] = None,
    ) -> Result[DeduplicationResult, IngestionError]:
        """
        Check if request is duplicate.
        
        Args:
            idempotency_key: Client-provided unique key
            request_data: Request body for hash computation
            entity_id: Optional entity for scoped lookups
            
        Returns:
            DeduplicationResult indicating if duplicate
        """
        # Validate key format
        if not idempotency_key or len(idempotency_key) > 64:
            return Err(IngestionError.validation_failed(
                field="idempotency_key",
                value=idempotency_key[:20] if idempotency_key else "",
                reason="Key must be 1-64 characters",
            ))
        
        # Check local cache first
        async with self._cache_lock:
            if idempotency_key in self._local_cache:
                ts, response = self._local_cache[idempotency_key]
                # Check TTL
                age_hours = (Timestamp.now() - ts) / (3600 * 1_000_000_000)
                if age_hours < self._ttl_hours:
                    return Ok(DeduplicationResult(
                        is_duplicate=True,
                        original_response=response,
                        idempotency_key=idempotency_key,
                    ))
                else:
                    del self._local_cache[idempotency_key]
        
        # Compute request hash
        request_hash = ContentHash.compute(request_data)
        
        # Check database
        result = await self._repo.check_or_set(
            key=idempotency_key,
            request_hash=request_hash,
            entity_id=entity_id,
        )
        
        if result.is_err():
            return result
        
        existing = result.unwrap()
        
        if existing is None:
            # New request, key was inserted
            return Ok(DeduplicationResult(
                is_duplicate=False,
                idempotency_key=idempotency_key,
            ))
        
        # Duplicate found
        return Ok(DeduplicationResult(
            is_duplicate=True,
            original_response=existing.get("result_payload"),
            idempotency_key=idempotency_key,
        ))
    
    async def check_by_content(
        self,
        content: bytes,
    ) -> Result[DeduplicationResult, IngestionError]:
        """
        Check for duplicate by content hash only.
        
        Used when no idempotency key provided.
        """
        content_hash = ContentHash.compute(content)
        
        # Use content hash as implicit idempotency key
        implicit_key = f"content:{content_hash.to_hex()[:32]}"
        
        return await self.check(implicit_key, content)
    
    async def complete(
        self,
        idempotency_key: str,
        response_data: dict[str, Any],
        status: str = "SUCCESS",
    ) -> Result[None, IngestionError]:
        """
        Mark request as completed with response.
        
        Response is returned for subsequent duplicate requests.
        """
        # Update database
        result = await self._repo.complete(
            key=idempotency_key,
            status=status,
            payload=response_data,
        )
        
        if result.is_err():
            return Err(IngestionError.validation_failed(
                field="idempotency_key",
                value=idempotency_key,
                reason="Failed to complete",
            ))
        
        # Update local cache
        async with self._cache_lock:
            self._local_cache[idempotency_key] = (Timestamp.now(), response_data)
            
            # Evict old entries (simple LRU)
            if len(self._local_cache) > 10000:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._local_cache.keys(),
                    key=lambda k: self._local_cache[k][0].nanos,
                )
                for key in sorted_keys[:1000]:
                    del self._local_cache[key]
        
        return Ok(None)
    
    async def cleanup_expired(self) -> int:
        """Remove expired idempotency keys."""
        # Cleanup database
        result = await self._repo.cleanup_expired()
        db_count = result.unwrap() if result.is_ok() else 0
        
        # Cleanup local cache
        now = Timestamp.now()
        ttl_nanos = self._ttl_hours * 3600 * 1_000_000_000
        
        async with self._cache_lock:
            expired = [
                key for key, (ts, _) in self._local_cache.items()
                if (now - ts) > ttl_nanos
            ]
            for key in expired:
                del self._local_cache[key]
        
        return db_count + len(expired)
    
    @staticmethod
    def compute_request_hash(
        method: str,
        path: str,
        body: bytes,
        headers: Optional[dict[str, str]] = None,
    ) -> ContentHash:
        """
        Compute deterministic hash of request.
        
        Includes method, path, and body for uniqueness.
        Excludes non-deterministic headers.
        """
        # Canonical request format
        canonical = {
            "method": method.upper(),
            "path": path,
            "body_hash": hashlib.sha256(body).hexdigest(),
        }
        
        # Include idempotency-relevant headers only
        if headers:
            for key in ["content-type", "accept"]:
                if key in headers:
                    canonical[f"header:{key}"] = headers[key]
        
        canonical_bytes = json.dumps(canonical, sort_keys=True).encode()
        return ContentHash.compute(canonical_bytes)
