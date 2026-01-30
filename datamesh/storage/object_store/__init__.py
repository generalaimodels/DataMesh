"""
Object Storage module: Content-addressable storage for large artifacts.
"""

from datamesh.storage.object_store.cas import ContentAddressableStorage
from datamesh.storage.object_store.backends import (
    StorageBackend,
    FileSystemBackend,
    S3Backend,
)

__all__ = [
    "ContentAddressableStorage",
    "StorageBackend",
    "FileSystemBackend",
    "S3Backend",
]
