"""
Storage Backends: FileSystem and S3-compatible object storage.

Provides abstract interface for:
- Put object (with optional compression)
- Get object (with byte range support)
- Delete object
- Check existence
"""

from __future__ import annotations

import hashlib
import io
import logging
import lzma
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datamesh.core.types import Result, Ok, Err, ByteRange, ContentHash
from datamesh.core.errors import StorageError
from datamesh.core.config import ObjectStoreConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectMetadata:
    """Metadata for stored object."""
    key: str
    size_bytes: int
    content_hash: ContentHash
    compressed: bool
    content_type: str = "application/octet-stream"


class StorageBackend(ABC):
    """Abstract storage backend interface."""
    
    @abstractmethod
    async def put(
        self,
        key: str,
        data: bytes,
        compress: bool = False,
    ) -> Result[ObjectMetadata, StorageError]:
        """Store object."""
        pass
    
    @abstractmethod
    async def get(
        self,
        key: str,
        byte_range: Optional[ByteRange] = None,
    ) -> Result[bytes, StorageError]:
        """Retrieve object or byte range."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> Result[bool, StorageError]:
        """Delete object."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> Result[bool, StorageError]:
        """Check if object exists."""
        pass
    
    @abstractmethod
    async def head(self, key: str) -> Result[Optional[ObjectMetadata], StorageError]:
        """Get object metadata without body."""
        pass


class FileSystemBackend(StorageBackend):
    """
    Local filesystem storage backend.
    
    Objects stored at: {data_dir}/{key}
    Compressed objects: {data_dir}/{key}.lzma
    """
    
    def __init__(self, config: ObjectStoreConfig) -> None:
        self._data_dir = config.data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
    
    def _key_to_path(self, key: str, compressed: bool = False) -> Path:
        """Convert key to filesystem path."""
        # Use hash-based directory structure for scalability
        key_hash = hashlib.md5(key.encode()).hexdigest()
        subdir = self._data_dir / key_hash[:2] / key_hash[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        
        suffix = ".lzma" if compressed else ""
        return subdir / f"{key_hash}{suffix}"
    
    async def put(
        self,
        key: str,
        data: bytes,
        compress: bool = False,
    ) -> Result[ObjectMetadata, StorageError]:
        """Store object to filesystem."""
        try:
            content_hash = ContentHash.compute(data)
            
            if compress and len(data) > 1024:  # Only compress > 1KB
                compressed_data = lzma.compress(data)
                path = self._key_to_path(key, compressed=True)
                path.write_bytes(compressed_data)
                size = len(compressed_data)
            else:
                path = self._key_to_path(key, compressed=False)
                path.write_bytes(data)
                size = len(data)
                compress = False
            
            return Ok(ObjectMetadata(
                key=key,
                size_bytes=size,
                content_hash=content_hash,
                compressed=compress,
            ))
            
        except OSError as e:
            return Err(StorageError.disk_full(
                path=str(self._data_dir),
                required_bytes=len(data),
                available_bytes=0,
            ))
    
    async def get(
        self,
        key: str,
        byte_range: Optional[ByteRange] = None,
    ) -> Result[bytes, StorageError]:
        """Retrieve object from filesystem."""
        try:
            # Check for compressed version first
            compressed_path = self._key_to_path(key, compressed=True)
            if compressed_path.exists():
                data = lzma.decompress(compressed_path.read_bytes())
            else:
                path = self._key_to_path(key, compressed=False)
                if not path.exists():
                    return Err(StorageError.connection_failed(
                        host=str(path),
                        port=0,
                    ))
                data = path.read_bytes()
            
            if byte_range:
                data = data[byte_range.start:byte_range.end + 1]
            
            return Ok(data)
            
        except OSError as e:
            return Err(StorageError.connection_failed(
                host=str(self._data_dir),
                port=0,
                cause=e,
            ))
    
    async def delete(self, key: str) -> Result[bool, StorageError]:
        """Delete object from filesystem."""
        try:
            for compressed in [True, False]:
                path = self._key_to_path(key, compressed=compressed)
                if path.exists():
                    path.unlink()
                    return Ok(True)
            return Ok(False)
            
        except OSError as e:
            return Err(StorageError.connection_failed(
                host=str(self._data_dir),
                port=0,
                cause=e,
            ))
    
    async def exists(self, key: str) -> Result[bool, StorageError]:
        """Check if object exists."""
        compressed_path = self._key_to_path(key, compressed=True)
        path = self._key_to_path(key, compressed=False)
        return Ok(compressed_path.exists() or path.exists())
    
    async def head(self, key: str) -> Result[Optional[ObjectMetadata], StorageError]:
        """Get object metadata."""
        for compressed in [True, False]:
            path = self._key_to_path(key, compressed=compressed)
            if path.exists():
                stat = path.stat()
                return Ok(ObjectMetadata(
                    key=key,
                    size_bytes=stat.st_size,
                    content_hash=ContentHash(digest=b"\x00" * 32),  # Unknown without read
                    compressed=compressed,
                ))
        return Ok(None)


class S3Backend(StorageBackend):
    """
    S3-compatible storage backend.
    
    Supports MinIO, Ceph, AWS S3, and compatible services.
    """
    
    def __init__(self, config: ObjectStoreConfig) -> None:
        self._config = config
        self._client = None
    
    async def _get_client(self):
        """Lazy initialize S3 client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    endpoint_url=self._config.s3_endpoint,
                    aws_access_key_id=self._config.s3_access_key,
                    aws_secret_access_key=self._config.s3_secret_key,
                    region_name=self._config.s3_region,
                )
            except ImportError:
                logger.warning("boto3 not available, S3 backend disabled")
        return self._client
    
    async def put(
        self,
        key: str,
        data: bytes,
        compress: bool = False,
    ) -> Result[ObjectMetadata, StorageError]:
        """Store object to S3."""
        client = await self._get_client()
        if client is None:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
            ))
        
        try:
            content_hash = ContentHash.compute(data)
            
            if compress and len(data) > 1024:
                data = lzma.compress(data)
                key = f"{key}.lzma"
            else:
                compress = False
            
            client.put_object(
                Bucket=self._config.s3_bucket,
                Key=key,
                Body=data,
            )
            
            return Ok(ObjectMetadata(
                key=key,
                size_bytes=len(data),
                content_hash=content_hash,
                compressed=compress,
            ))
            
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
                cause=e,
            ))
    
    async def get(
        self,
        key: str,
        byte_range: Optional[ByteRange] = None,
    ) -> Result[bytes, StorageError]:
        """Retrieve object from S3."""
        client = await self._get_client()
        if client is None:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
            ))
        
        try:
            kwargs = {"Bucket": self._config.s3_bucket, "Key": key}
            
            if byte_range:
                kwargs["Range"] = byte_range.to_http_header()
            
            response = client.get_object(**kwargs)
            data = response["Body"].read()
            
            if key.endswith(".lzma"):
                data = lzma.decompress(data)
            
            return Ok(data)
            
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
                cause=e,
            ))
    
    async def delete(self, key: str) -> Result[bool, StorageError]:
        """Delete object from S3."""
        client = await self._get_client()
        if client is None:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
            ))
        
        try:
            client.delete_object(
                Bucket=self._config.s3_bucket,
                Key=key,
            )
            return Ok(True)
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
                cause=e,
            ))
    
    async def exists(self, key: str) -> Result[bool, StorageError]:
        """Check if object exists in S3."""
        result = await self.head(key)
        if result.is_err():
            return result
        return Ok(result.unwrap() is not None)
    
    async def head(self, key: str) -> Result[Optional[ObjectMetadata], StorageError]:
        """Get object metadata from S3."""
        client = await self._get_client()
        if client is None:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
            ))
        
        try:
            response = client.head_object(
                Bucket=self._config.s3_bucket,
                Key=key,
            )
            
            return Ok(ObjectMetadata(
                key=key,
                size_bytes=response["ContentLength"],
                content_hash=ContentHash(digest=b"\x00" * 32),
                compressed=key.endswith(".lzma"),
            ))
        except client.exceptions.NoSuchKey:
            return Ok(None)
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="s3",
                port=443,
                cause=e,
            ))
