"""
History Cursor: Cursor-Based Pagination

Provides:
- Composite cursor encoding (bucket_id, sequence_id)
- Base64 encoding for URL-safe transport
- Bidirectional traversal support
- Cursor validation and expiration

Design:
    Cursor-based pagination is superior to offset-based:
    - Consistent results during concurrent writes
    - O(1) seek instead of O(n) skip
    - Stable ordering with clustered indexes

Cursor Format:
    {
        "v": 1,                    // Version
        "b": "2024-01",            // Bucket ID
        "s": 12345,                // Sequence ID
        "d": "desc",               // Direction
        "e": 1704067200            // Expiration timestamp
    }
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional, Sequence, TypeVar, Generic

from datamesh.core.types import Result, Ok, Err


# =============================================================================
# CONSTANTS
# =============================================================================
CURSOR_VERSION: int = 1
DEFAULT_CURSOR_TTL_SECONDS: int = 3600  # 1 hour
MAX_PAGE_SIZE: int = 100
CURSOR_HMAC_SECRET: bytes = b"datamesh-cursor-v1"  # In production, use env var


# =============================================================================
# PAGINATION DIRECTION
# =============================================================================
class PaginationDirection(Enum):
    """Direction for cursor traversal."""
    FORWARD = "asc"   # Oldest to newest
    BACKWARD = "desc" # Newest to oldest (default for timelines)


# =============================================================================
# HISTORY CURSOR
# =============================================================================
@dataclass(frozen=True, slots=True)
class HistoryCursor:
    """
    Immutable cursor for history pagination.
    
    Encodes position in the timeline for continuation queries.
    HMAC signature prevents tampering.
    """
    bucket_id: str
    sequence_id: int
    direction: PaginationDirection = PaginationDirection.BACKWARD
    expires_at: int = 0  # Unix timestamp
    signature: str = ""  # HMAC-SHA256 truncated
    
    @classmethod
    def create(
        cls,
        bucket_id: str,
        sequence_id: int,
        direction: PaginationDirection = PaginationDirection.BACKWARD,
        ttl_seconds: int = DEFAULT_CURSOR_TTL_SECONDS,
    ) -> HistoryCursor:
        """Create new cursor with signature."""
        expires_at = int(time.time()) + ttl_seconds
        
        # Compute signature
        payload = f"{CURSOR_VERSION}:{bucket_id}:{sequence_id}:{direction.value}:{expires_at}"
        signature = hashlib.sha256(
            CURSOR_HMAC_SECRET + payload.encode()
        ).hexdigest()[:16]
        
        return cls(
            bucket_id=bucket_id,
            sequence_id=sequence_id,
            direction=direction,
            expires_at=expires_at,
            signature=signature,
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if cursor has expired."""
        return time.time() > self.expires_at
    
    def validate(self) -> Result[None, str]:
        """Validate cursor signature and expiration."""
        if self.is_expired:
            return Err("Cursor has expired")
        
        # Verify signature
        payload = f"{CURSOR_VERSION}:{self.bucket_id}:{self.sequence_id}:{self.direction.value}:{self.expires_at}"
        expected = hashlib.sha256(
            CURSOR_HMAC_SECRET + payload.encode()
        ).hexdigest()[:16]
        
        if self.signature != expected:
            return Err("Invalid cursor signature")
        
        return Ok(None)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "v": CURSOR_VERSION,
            "b": self.bucket_id,
            "s": self.sequence_id,
            "d": self.direction.value,
            "e": self.expires_at,
            "h": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryCursor:
        """Deserialize from dictionary."""
        return cls(
            bucket_id=data["b"],
            sequence_id=data["s"],
            direction=PaginationDirection(data["d"]),
            expires_at=data["e"],
            signature=data["h"],
        )


# =============================================================================
# CURSOR ENCODER
# =============================================================================
class CursorEncoder:
    """
    Encodes/decodes cursors for transport.
    
    Uses URL-safe Base64 encoding for HTTP compatibility.
    """
    
    @staticmethod
    def encode(cursor: HistoryCursor) -> str:
        """Encode cursor to URL-safe string."""
        data = cursor.to_dict()
        json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(json_bytes).decode("ascii")
    
    @staticmethod
    def decode(encoded: str) -> Result[HistoryCursor, str]:
        """Decode cursor from string."""
        try:
            # Pad if necessary
            padding = 4 - (len(encoded) % 4)
            if padding != 4:
                encoded += "=" * padding
            
            json_bytes = base64.urlsafe_b64decode(encoded)
            data = json.loads(json_bytes)
            
            # Version check
            if data.get("v", 0) != CURSOR_VERSION:
                return Err(f"Unsupported cursor version: {data.get('v')}")
            
            cursor = HistoryCursor.from_dict(data)
            
            # Validate
            validation = cursor.validate()
            if validation.is_err():
                return Err(validation.error)
            
            return Ok(cursor)
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return Err(f"Invalid cursor format: {e}")


# =============================================================================
# PAGINATED RESULT
# =============================================================================
T = TypeVar("T")


@dataclass(slots=True)
class PaginatedResult(Generic[T]):
    """
    Paginated query result.
    
    Contains items and optional cursor for next page.
    """
    items: list[T]
    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None
    has_more: bool = False
    total_count: Optional[int] = None  # If available without scan
    
    @property
    def count(self) -> int:
        """Number of items in this page."""
        return len(self.items)
    
    @property
    def is_empty(self) -> bool:
        """Check if result is empty."""
        return len(self.items) == 0
    
    def map(self, func: Any) -> PaginatedResult[Any]:
        """Transform items with function."""
        return PaginatedResult(
            items=[func(item) for item in self.items],
            next_cursor=self.next_cursor,
            prev_cursor=self.prev_cursor,
            has_more=self.has_more,
            total_count=self.total_count,
        )


# =============================================================================
# PAGINATION HELPERS
# =============================================================================
def create_next_cursor(
    bucket_id: str,
    sequence_id: int,
    direction: PaginationDirection = PaginationDirection.BACKWARD,
) -> Optional[str]:
    """Create encoded cursor for next page."""
    cursor = HistoryCursor.create(
        bucket_id=bucket_id,
        sequence_id=sequence_id,
        direction=direction,
    )
    return CursorEncoder.encode(cursor)


def parse_cursor(
    encoded: Optional[str],
) -> Result[Optional[HistoryCursor], str]:
    """Parse optional cursor string."""
    if encoded is None or encoded == "":
        return Ok(None)
    
    result = CursorEncoder.decode(encoded)
    if result.is_err():
        return Err(result.error)
    
    return Ok(result.unwrap())


def validate_page_size(
    limit: int,
    max_limit: int = MAX_PAGE_SIZE,
) -> int:
    """Validate and normalize page size."""
    if limit <= 0:
        return max_limit
    return min(limit, max_limit)
