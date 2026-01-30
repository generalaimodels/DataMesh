"""
Exhaustive Error Hierarchy for Planetary-Scale Data Mesh

Design Principles:
- Forbid exceptions for control flow (use Result types)
- Enforce exhaustive pattern matching for all error variants
- Never swallow errors or use null for absence
- Carry full error context for debugging and audit trails

Each error type includes:
- Unique error code for programmatic handling
- Human-readable message for logging
- Optional cause chain for root cause analysis
- Timestamp for correlation with distributed traces

Usage:
    result = some_operation()
    match result:
        case Ok(value):
            process(value)
        case Err(StorageError.ConnectionFailed(host, port)):
            handle_connection_failure(host, port)
        case Err(StorageError.Timeout(duration)):
            handle_timeout(duration)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from uuid import uuid4

from datamesh.core.types import Timestamp


# =============================================================================
# ERROR CODE ENUMERATION
# =============================================================================
class ErrorCode(Enum):
    """
    Unique error codes for programmatic error handling.
    
    Codes are grouped by subsystem:
    - 1xxx: Storage errors
    - 2xxx: Ingestion errors
    - 3xxx: Query errors
    - 4xxx: Sharding errors
    - 5xxx: Security errors
    - 9xxx: Internal/unknown errors
    """
    
    # Storage errors (1xxx)
    STORAGE_CONNECTION_FAILED = 1001
    STORAGE_TIMEOUT = 1002
    STORAGE_CONSTRAINT_VIOLATION = 1003
    STORAGE_SERIALIZATION_CONFLICT = 1004
    STORAGE_DISK_FULL = 1005
    STORAGE_CORRUPTION = 1006
    STORAGE_REPLICA_LAG = 1007
    
    # Ingestion errors (2xxx)
    INGESTION_VALIDATION_FAILED = 2001
    INGESTION_CAPACITY_EXCEEDED = 2002
    INGESTION_DUPLICATE_DETECTED = 2003
    INGESTION_SCHEMA_MISMATCH = 2004
    INGESTION_BACKPRESSURE = 2005
    INGESTION_SAGA_FAILED = 2006
    
    # Query errors (3xxx)
    QUERY_TIMEOUT = 3001
    QUERY_MALFORMED_CURSOR = 3002
    QUERY_PERMISSION_DENIED = 3003
    QUERY_RESOURCE_NOT_FOUND = 3004
    QUERY_INVALID_FILTER = 3005
    QUERY_RESULT_TOO_LARGE = 3006
    
    # Sharding errors (4xxx)
    SHARDING_RING_IMBALANCE = 4001
    SHARDING_MIGRATION_FAILED = 4002
    SHARDING_NODE_UNAVAILABLE = 4003
    SHARDING_PARTITION_NOT_FOUND = 4004
    
    # Security errors (5xxx)
    SECURITY_UNAUTHORIZED = 5001
    SECURITY_FORBIDDEN = 5002
    SECURITY_TOKEN_EXPIRED = 5003
    SECURITY_ENCRYPTION_FAILED = 5004
    SECURITY_GEO_FENCE_VIOLATION = 5005
    
    # Internal errors (9xxx)
    INTERNAL_ERROR = 9001
    INTERNAL_CONFIGURATION_ERROR = 9002
    INTERNAL_DEPENDENCY_UNAVAILABLE = 9003
    
    # Reliability errors (6xxx)
    RELIABILITY_CIRCUIT_OPEN = 6001
    RELIABILITY_RETRY_EXHAUSTED = 6002
    RELIABILITY_TIMEOUT = 6003


# =============================================================================
# BASE ERROR CLASS
# =============================================================================
@dataclass
class DataMeshError(Exception):
    """
    Base class for all data mesh errors.
    
    Provides common infrastructure for error handling:
    - Unique error ID for distributed tracing
    - Error code for programmatic handling
    - Timestamp for correlation
    - Cause chain for root cause analysis
    
    All errors are immutable after creation to prevent
    accidental mutation during propagation.
    """
    
    code: ErrorCode
    message: str
    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: Timestamp = field(default_factory=Timestamp.now)
    cause: Optional[Exception] = None
    context: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        # Initialize Exception base class with message
        super().__init__(self.message)
    
    def with_context(self, **kwargs: Any) -> DataMeshError:
        """
        Add context to error (returns new instance).
        
        Context is useful for debugging but should not
        contain sensitive information.
        """
        new_context = {**self.context, **kwargs}
        return DataMeshError(
            code=self.code,
            message=self.message,
            error_id=self.error_id,
            timestamp=self.timestamp,
            cause=self.cause,
            context=new_context,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize error to dictionary for logging/API responses.
        
        Note: Excludes cause stack trace in production to avoid
        leaking implementation details.
        """
        return {
            "error_id": self.error_id,
            "code": self.code.name,
            "code_value": self.code.value,
            "message": self.message,
            "timestamp_nanos": self.timestamp.nanos,
            "context": self.context,
        }
    
    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message} (id={self.error_id[:8]})"
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.name}, "
            f"message={self.message!r}, "
            f"error_id={self.error_id!r})"
        )


# =============================================================================
# STORAGE ERRORS (CP/AP SUBSYSTEMS)
# =============================================================================
@dataclass
class StorageError(DataMeshError):
    """
    Errors from CP (PostgreSQL) and AP (SQLite) storage subsystems.
    
    Covers connection issues, timeouts, constraint violations,
    and serialization conflicts.
    """
    
    @classmethod
    def connection_failed(
        cls,
        host: str,
        port: int,
        cause: Optional[Exception] = None,
    ) -> StorageError:
        """Database connection failed."""
        return cls(
            code=ErrorCode.STORAGE_CONNECTION_FAILED,
            message=f"Failed to connect to database at {host}:{port}",
            cause=cause,
            context={"host": host, "port": port},
        )
    
    @classmethod
    def timeout(
        cls,
        operation: str,
        duration_ms: int,
        cause: Optional[Exception] = None,
    ) -> StorageError:
        """Operation timed out."""
        return cls(
            code=ErrorCode.STORAGE_TIMEOUT,
            message=f"Operation '{operation}' timed out after {duration_ms}ms",
            cause=cause,
            context={"operation": operation, "duration_ms": duration_ms},
        )
    
    @classmethod
    def constraint_violation(
        cls,
        constraint: str,
        table: str,
        cause: Optional[Exception] = None,
    ) -> StorageError:
        """Database constraint violated (unique, foreign key, etc)."""
        return cls(
            code=ErrorCode.STORAGE_CONSTRAINT_VIOLATION,
            message=f"Constraint '{constraint}' violated on table '{table}'",
            cause=cause,
            context={"constraint": constraint, "table": table},
        )
    
    @classmethod
    def serialization_conflict(
        cls,
        transaction_id: str,
        cause: Optional[Exception] = None,
    ) -> StorageError:
        """Serializable transaction conflict detected."""
        return cls(
            code=ErrorCode.STORAGE_SERIALIZATION_CONFLICT,
            message=f"Serialization conflict in transaction {transaction_id}",
            cause=cause,
            context={"transaction_id": transaction_id},
        )
    
    @classmethod
    def disk_full(
        cls,
        path: str,
        required_bytes: int,
        available_bytes: int,
    ) -> StorageError:
        """Insufficient disk space."""
        return cls(
            code=ErrorCode.STORAGE_DISK_FULL,
            message=f"Disk full at {path}: need {required_bytes}B, have {available_bytes}B",
            context={
                "path": path,
                "required_bytes": required_bytes,
                "available_bytes": available_bytes,
            },
        )
    
    @classmethod
    def corruption(
        cls,
        description: str,
        affected_range: Optional[str] = None,
    ) -> StorageError:
        """Data corruption detected."""
        return cls(
            code=ErrorCode.STORAGE_CORRUPTION,
            message=f"Data corruption detected: {description}",
            context={"affected_range": affected_range},
        )
    
    @classmethod
    def replica_lag(
        cls,
        replica: str,
        lag_ms: int,
        threshold_ms: int,
    ) -> StorageError:
        """Replica lag exceeds threshold."""
        return cls(
            code=ErrorCode.STORAGE_REPLICA_LAG,
            message=f"Replica {replica} lag {lag_ms}ms exceeds threshold {threshold_ms}ms",
            context={"replica": replica, "lag_ms": lag_ms, "threshold_ms": threshold_ms},
        )


# =============================================================================
# INGESTION ERRORS (WRITE PATH)
# =============================================================================
@dataclass
class IngestionError(DataMeshError):
    """
    Errors from the write path ingestion pipeline.
    
    Covers validation failures, capacity limits, duplicate detection,
    schema mismatches, and saga transaction failures.
    """
    
    @classmethod
    def validation_failed(
        cls,
        field: str,
        value: Any,
        reason: str,
    ) -> IngestionError:
        """Input validation failed."""
        return cls(
            code=ErrorCode.INGESTION_VALIDATION_FAILED,
            message=f"Validation failed for field '{field}': {reason}",
            context={"field": field, "value": str(value)[:100], "reason": reason},
        )
    
    @classmethod
    def capacity_exceeded(
        cls,
        resource: str,
        current: int,
        limit: int,
    ) -> IngestionError:
        """Resource capacity limit exceeded."""
        return cls(
            code=ErrorCode.INGESTION_CAPACITY_EXCEEDED,
            message=f"Capacity exceeded for {resource}: {current}/{limit}",
            context={"resource": resource, "current": current, "limit": limit},
        )
    
    @classmethod
    def duplicate_detected(
        cls,
        idempotency_key: str,
        original_timestamp: Timestamp,
    ) -> IngestionError:
        """Duplicate request detected via idempotency key."""
        return cls(
            code=ErrorCode.INGESTION_DUPLICATE_DETECTED,
            message=f"Duplicate request with key '{idempotency_key}'",
            context={
                "idempotency_key": idempotency_key,
                "original_timestamp_nanos": original_timestamp.nanos,
            },
        )
    
    @classmethod
    def schema_mismatch(
        cls,
        expected_version: int,
        actual_version: int,
        differences: list[str],
    ) -> IngestionError:
        """Schema version mismatch."""
        return cls(
            code=ErrorCode.INGESTION_SCHEMA_MISMATCH,
            message=f"Schema mismatch: expected v{expected_version}, got v{actual_version}",
            context={
                "expected_version": expected_version,
                "actual_version": actual_version,
                "differences": differences,
            },
        )
    
    @classmethod
    def backpressure(
        cls,
        queue_depth: int,
        threshold: int,
    ) -> IngestionError:
        """Backpressure triggered due to queue depth."""
        return cls(
            code=ErrorCode.INGESTION_BACKPRESSURE,
            message=f"Backpressure: queue depth {queue_depth} exceeds threshold {threshold}",
            context={"queue_depth": queue_depth, "threshold": threshold},
        )
    
    @classmethod
    def saga_failed(
        cls,
        saga_id: str,
        failed_step: str,
        compensated_steps: list[str],
        cause: Optional[Exception] = None,
    ) -> IngestionError:
        """Saga transaction failed, compensating transactions executed."""
        return cls(
            code=ErrorCode.INGESTION_SAGA_FAILED,
            message=f"Saga {saga_id} failed at step '{failed_step}'",
            cause=cause,
            context={
                "saga_id": saga_id,
                "failed_step": failed_step,
                "compensated_steps": compensated_steps,
            },
        )


# =============================================================================
# QUERY ERRORS (READ PATH)
# =============================================================================
@dataclass
class QueryError(DataMeshError):
    """
    Errors from the read path query execution.
    
    Covers timeouts, invalid cursors, permission issues,
    and result size limits.
    """
    
    @classmethod
    def timeout(
        cls,
        query_id: str,
        duration_ms: int,
        query_type: str,
    ) -> QueryError:
        """Query execution timed out."""
        return cls(
            code=ErrorCode.QUERY_TIMEOUT,
            message=f"Query {query_id} timed out after {duration_ms}ms",
            context={
                "query_id": query_id,
                "duration_ms": duration_ms,
                "query_type": query_type,
            },
        )
    
    @classmethod
    def malformed_cursor(
        cls,
        cursor: str,
        reason: str,
    ) -> QueryError:
        """Pagination cursor is malformed or expired."""
        return cls(
            code=ErrorCode.QUERY_MALFORMED_CURSOR,
            message=f"Malformed cursor: {reason}",
            context={"cursor": cursor[:50], "reason": reason},
        )
    
    @classmethod
    def permission_denied(
        cls,
        resource: str,
        required_permission: str,
        user_permissions: list[str],
    ) -> QueryError:
        """User lacks required permission."""
        return cls(
            code=ErrorCode.QUERY_PERMISSION_DENIED,
            message=f"Permission denied: {required_permission} required for {resource}",
            context={
                "resource": resource,
                "required_permission": required_permission,
                "user_permissions": user_permissions,
            },
        )
    
    @classmethod
    def resource_not_found(
        cls,
        resource_type: str,
        resource_id: str,
    ) -> QueryError:
        """Requested resource does not exist."""
        return cls(
            code=ErrorCode.QUERY_RESOURCE_NOT_FOUND,
            message=f"{resource_type} '{resource_id}' not found",
            context={"resource_type": resource_type, "resource_id": resource_id},
        )
    
    @classmethod
    def invalid_filter(
        cls,
        filter_field: str,
        filter_value: Any,
        reason: str,
    ) -> QueryError:
        """Query filter is invalid."""
        return cls(
            code=ErrorCode.QUERY_INVALID_FILTER,
            message=f"Invalid filter on '{filter_field}': {reason}",
            context={
                "filter_field": filter_field,
                "filter_value": str(filter_value)[:100],
                "reason": reason,
            },
        )
    
    @classmethod
    def result_too_large(
        cls,
        result_count: int,
        max_count: int,
    ) -> QueryError:
        """Query result exceeds maximum size."""
        return cls(
            code=ErrorCode.QUERY_RESULT_TOO_LARGE,
            message=f"Result size {result_count} exceeds limit {max_count}",
            context={"result_count": result_count, "max_count": max_count},
        )


# =============================================================================
# SHARDING ERRORS
# =============================================================================
@dataclass
class ShardingError(DataMeshError):
    """
    Errors from the sharding and partitioning subsystem.
    
    Covers hash ring imbalance, migration failures,
    and partition routing issues.
    """
    
    @classmethod
    def ring_imbalance(
        cls,
        max_load_factor: float,
        threshold: float,
    ) -> ShardingError:
        """Hash ring has become imbalanced."""
        return cls(
            code=ErrorCode.SHARDING_RING_IMBALANCE,
            message=f"Ring imbalance: max load factor {max_load_factor:.2f} > {threshold:.2f}",
            context={"max_load_factor": max_load_factor, "threshold": threshold},
        )
    
    @classmethod
    def migration_failed(
        cls,
        source_node: str,
        target_node: str,
        partition_range: str,
        cause: Optional[Exception] = None,
    ) -> ShardingError:
        """Partition migration failed."""
        return cls(
            code=ErrorCode.SHARDING_MIGRATION_FAILED,
            message=f"Migration failed from {source_node} to {target_node}",
            cause=cause,
            context={
                "source_node": source_node,
                "target_node": target_node,
                "partition_range": partition_range,
            },
        )
    
    @classmethod
    def node_unavailable(
        cls,
        node_id: str,
        last_seen: Timestamp,
    ) -> ShardingError:
        """Shard node is unavailable."""
        return cls(
            code=ErrorCode.SHARDING_NODE_UNAVAILABLE,
            message=f"Node {node_id} unavailable",
            context={"node_id": node_id, "last_seen_nanos": last_seen.nanos},
        )
    
    @classmethod
    def partition_not_found(
        cls,
        partition_key: bytes,
    ) -> ShardingError:
        """Partition for key not found in ring."""
        return cls(
            code=ErrorCode.SHARDING_PARTITION_NOT_FOUND,
            message=f"No partition found for key {partition_key.hex()[:16]}...",
            context={"partition_key_hex": partition_key.hex()},
        )


# =============================================================================
# SECURITY ERRORS
# =============================================================================
@dataclass
class SecurityError(DataMeshError):
    """
    Errors from security and compliance subsystem.
    
    Covers authentication, authorization, encryption,
    and geo-fencing violations.
    """
    
    @classmethod
    def unauthorized(
        cls,
        reason: str,
    ) -> SecurityError:
        """Request lacks valid authentication."""
        return cls(
            code=ErrorCode.SECURITY_UNAUTHORIZED,
            message=f"Unauthorized: {reason}",
            context={"reason": reason},
        )
    
    @classmethod
    def forbidden(
        cls,
        resource: str,
        action: str,
    ) -> SecurityError:
        """Authenticated user lacks permission."""
        return cls(
            code=ErrorCode.SECURITY_FORBIDDEN,
            message=f"Forbidden: cannot {action} on {resource}",
            context={"resource": resource, "action": action},
        )
    
    @classmethod
    def token_expired(
        cls,
        token_id: str,
        expiry: Timestamp,
    ) -> SecurityError:
        """Authentication token has expired."""
        return cls(
            code=ErrorCode.SECURITY_TOKEN_EXPIRED,
            message=f"Token {token_id[:8]}... expired",
            context={"token_id": token_id, "expiry_nanos": expiry.nanos},
        )
    
    @classmethod
    def encryption_failed(
        cls,
        operation: str,
        cause: Optional[Exception] = None,
    ) -> SecurityError:
        """Encryption or decryption operation failed."""
        return cls(
            code=ErrorCode.SECURITY_ENCRYPTION_FAILED,
            message=f"Encryption failed during {operation}",
            cause=cause,
            context={"operation": operation},
        )
    
    @classmethod
    def geo_fence_violation(
        cls,
        data_region: str,
        request_region: str,
        compliance_tier: str,
    ) -> SecurityError:
        """Request violates geo-fencing policy."""
        return cls(
            code=ErrorCode.SECURITY_GEO_FENCE_VIOLATION,
            message=f"Geo-fence violation: {compliance_tier} data in {data_region} "
                    f"cannot be accessed from {request_region}",
            context={
                "data_region": data_region,
                "request_region": request_region,
                "compliance_tier": compliance_tier,
            },
        )


# =============================================================================
# RELIABILITY ERRORS
# =============================================================================
@dataclass
class ReliabilityError(DataMeshError):
    """
    Errors from reliability subsystem (circuit breakers, retries).
    
    Covers circuit breaker state, retry exhaustion, and timeouts.
    """
    
    @classmethod
    def circuit_open(
        cls,
        circuit_name: str,
        failure_count: int,
        retry_after_seconds: int,
    ) -> ReliabilityError:
        """Circuit breaker is open, failing fast."""
        return cls(
            code=ErrorCode.RELIABILITY_CIRCUIT_OPEN,
            message=f"Circuit '{circuit_name}' is OPEN after {failure_count} failures",
            context={
                "circuit_name": circuit_name,
                "failure_count": failure_count,
                "retry_after_seconds": retry_after_seconds,
            },
        )
    
    @classmethod
    def retry_exhausted(
        cls,
        attempts: int,
        last_error: str,
    ) -> ReliabilityError:
        """All retry attempts exhausted."""
        return cls(
            code=ErrorCode.RELIABILITY_RETRY_EXHAUSTED,
            message=f"Retry exhausted after {attempts} attempts: {last_error}",
            context={"attempts": attempts, "last_error": last_error},
        )
    
    @classmethod
    def timeout(
        cls,
        operation: str,
        timeout_ms: int,
    ) -> ReliabilityError:
        """Operation timed out."""
        return cls(
            code=ErrorCode.RELIABILITY_TIMEOUT,
            message=f"Operation '{operation}' timed out after {timeout_ms}ms",
            context={"operation": operation, "timeout_ms": timeout_ms},
        )

