"""
Core module: Type definitions, error hierarchy, and configuration.

This module provides the foundational abstractions for the data mesh:
- Result/Either monads for zero-exception control flow
- Exhaustive error hierarchy with pattern matching support
- Configuration management with validation
"""

from datamesh.core.types import (
    Result,
    Ok,
    Err,
    EntityId,
    ConversationId,
    InstructionId,
    Timestamp,
    ByteRange,
    ComplianceTier,
    GeoRegion,
)
from datamesh.core.errors import (
    DataMeshError,
    StorageError,
    IngestionError,
    QueryError,
    ShardingError,
)
from datamesh.core.config import DataMeshConfig

__all__ = [
    "Result",
    "Ok",
    "Err",
    "EntityId",
    "ConversationId",
    "InstructionId",
    "Timestamp",
    "ByteRange",
    "ComplianceTier",
    "GeoRegion",
    "DataMeshError",
    "StorageError",
    "IngestionError",
    "QueryError",
    "ShardingError",
    "DataMeshConfig",
]
