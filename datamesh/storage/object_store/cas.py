"""
Content-Addressable Storage (CAS)

Provides deduplication and idempotency via SHA-256 content hashing.
Objects are stored by their content hash, ensuring:
- Automatic deduplication (same content = same key)
- Idempotent writes (re-uploading same content is no-op)
- Integrity verification (hash mismatch = corruption)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from datamesh.core.types import Result, Ok, Err, ByteRange, ContentHash
from datamesh.core.errors import StorageError
from datamesh.core.config import ObjectStoreConfig
from datamesh.storage.object_store.backends import (
    StorageBackend,
    FileSystemBackend,
    S3Backend,
    ObjectMetadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CASObject:
    """Content-addressed object reference."""
    hash: ContentHash
    size_bytes: int
    compressed: bool
    
    @property
    def key(self) -> str:
        """Storage key derived from content hash."""
        return self.hash.to_hex()


class ContentAddressableStorage:
    """
    Content-addressable storage layer.
    
    Wraps storage backend with:
    - Automatic content hashing
    - Deduplication on write
    - Integrity verification on read
    
    Usage:
        cas = ContentAddressableStorage(config)
        
        # Store object
        result = await cas.put(b"hello world")
        cas_obj = result.unwrap()
        
        # Retrieve by hash
        data = await cas.get(cas_obj.hash)
    """
    
    def __init__(self, config: ObjectStoreConfig) -> None:
        self._config = config
        self._backend = self._create_backend(config)
    
    def _create_backend(self, config: ObjectStoreConfig) -> StorageBackend:
        """Create appropriate backend based on config."""
        if config.backend == "s3":
            return S3Backend(config)
        return FileSystemBackend(config)
    
    async def put(
        self,
        data: bytes,
        compress: Optional[bool] = None,
    ) -> Result[CASObject, StorageError]:
        """
        Store data and return content-addressed reference.
        
        Idempotent: re-storing same data returns same hash.
        
        Args:
            data: Raw bytes to store
            compress: Force compression on/off (None = auto)
            
        Returns:
            CASObject with hash and metadata
        """
        # Compute content hash
        content_hash = ContentHash.compute(data)
        key = content_hash.to_hex()
        
        # Check if already exists (deduplication)
        exists_result = await self._backend.exists(key)
        if exists_result.is_ok() and exists_result.unwrap():
            head_result = await self._backend.head(key)
            if head_result.is_ok() and head_result.unwrap():
                meta = head_result.unwrap()
                return Ok(CASObject(
                    hash=content_hash,
                    size_bytes=meta.size_bytes,
                    compressed=meta.compressed,
                ))
        
        # Determine compression
        should_compress = compress
        if should_compress is None:
            should_compress = self._config.compression_enabled and len(data) > 1024
        
        # Store object
        result = await self._backend.put(key, data, compress=should_compress)
        
        if result.is_err():
            return result
        
        meta = result.unwrap()
        return Ok(CASObject(
            hash=content_hash,
            size_bytes=meta.size_bytes,
            compressed=meta.compressed,
        ))
    
    async def get(
        self,
        content_hash: ContentHash,
        byte_range: Optional[ByteRange] = None,
        verify: bool = True,
    ) -> Result[bytes, StorageError]:
        """
        Retrieve data by content hash.
        
        Args:
            content_hash: SHA-256 hash of content
            byte_range: Optional partial read range
            verify: Verify hash on read (default True)
            
        Returns:
            Raw bytes (decompressed if needed)
        """
        key = content_hash.to_hex()
        
        result = await self._backend.get(key, byte_range)
        
        if result.is_err():
            return result
        
        data = result.unwrap()
        
        # Verify integrity (skip for partial reads)
        if verify and byte_range is None:
            actual_hash = ContentHash.compute(data)
            if actual_hash.digest != content_hash.digest:
                return Err(StorageError.corruption(
                    description="Content hash mismatch",
                    affected_range=key,
                ))
        
        return Ok(data)
    
    async def delete(self, content_hash: ContentHash) -> Result[bool, StorageError]:
        """
        Delete object by content hash.
        
        Note: Deleting by hash may affect multiple references
        if content was stored multiple times with different metadata.
        """
        key = content_hash.to_hex()
        return await self._backend.delete(key)
    
    async def exists(self, content_hash: ContentHash) -> Result[bool, StorageError]:
        """Check if content exists."""
        key = content_hash.to_hex()
        return await self._backend.exists(key)
    
    async def head(self, content_hash: ContentHash) -> Result[Optional[CASObject], StorageError]:
        """Get object metadata without body."""
        key = content_hash.to_hex()
        
        result = await self._backend.head(key)
        
        if result.is_err():
            return result
        
        meta = result.unwrap()
        if meta is None:
            return Ok(None)
        
        return Ok(CASObject(
            hash=content_hash,
            size_bytes=meta.size_bytes,
            compressed=meta.compressed,
        ))
    
    async def copy(
        self,
        source_hash: ContentHash,
        target_storage: ContentAddressableStorage,
    ) -> Result[CASObject, StorageError]:
        """Copy object to another CAS instance."""
        # Get from source
        result = await self.get(source_hash)
        if result.is_err():
            return result
        
        # Put to target
        return await target_storage.put(result.unwrap())
