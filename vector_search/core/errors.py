"""
Result Monad & Error Types: Zero-Exception Control Flow

Implements Rust-inspired Result[T, E] monad for deterministic error handling.
All vector search operations return Result types instead of raising exceptions.

Design Principles:
    - Exhaustive Error Handling: Pattern matching on all error variants
    - Zero-Cost Abstraction: No runtime overhead for successful operations  
    - Type Safety: Static type checking for error propagation
    - Composability: Monadic bind (flat_map) for chaining fallible operations

Memory Layout:
    Ok[T]:  8 bytes (vtable) + sizeof(T) 
    Err[E]: 8 bytes (vtable) + sizeof(E)
    
Performance:
    - is_ok/is_err: O(1) branch-free via isinstance
    - unwrap: O(1) direct attribute access
    - map/flat_map: O(f) where f is transform complexity
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    NoReturn,
    Optional,
    TypeVar,
    Union,
    final,
    overload,
)


# =============================================================================
# TYPE VARIABLES
# =============================================================================
T = TypeVar("T")  # Success value type
E = TypeVar("E")  # Error type  
U = TypeVar("U")  # Transform result type


# =============================================================================
# RESULT MONAD: SUCCESS VARIANT
# =============================================================================
@final
@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """
    Success variant of Result monad.
    
    Immutable, hashable container for successful computation results.
    Uses __slots__ for memory efficiency (~40% reduction vs dict-based).
    
    Memory Layout (64-bit):
        _value: 8 bytes (pointer to T)
        Total: 8 bytes + sizeof(T)
        
    Thread Safety:
        Immutable after construction - safe for concurrent access.
        
    Example:
        result: Result[int, str] = Ok(42)
        if result.is_ok():
            value = result.unwrap()  # Type-safe extraction
    """
    _value: T
    
    def is_ok(self) -> Literal[True]:
        """O(1) success check - branch-free via type narrowing."""
        return True
    
    def is_err(self) -> Literal[False]:
        """O(1) error check - branch-free via type narrowing."""
        return False
    
    def unwrap(self) -> T:
        """
        Extract value. Safe to call after is_ok() check.
        
        Returns:
            T: The wrapped success value
            
        Complexity: O(1) direct attribute access
        """
        return self._value
    
    def unwrap_or(self, default: T) -> T:
        """Return value, ignoring default. Complexity: O(1)."""
        return self._value
    
    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Return value without calling fallback. Complexity: O(1)."""
        return self._value
    
    def expect(self, msg: str) -> T:
        """Unwrap with custom panic message (never panics for Ok)."""
        return self._value
    
    def map(self, fn: Callable[[T], U]) -> "Ok[U]":
        """
        Apply transformation to success value.
        
        Complexity: O(f) where f is complexity of fn
        
        Example:
            Ok(5).map(lambda x: x * 2)  # Ok(10)
        """
        return Ok(fn(self._value))
    
    def map_err(self, fn: Callable[[Any], Any]) -> "Ok[T]":
        """No-op on success variant - returns self unchanged."""
        return self
    
    def flat_map(self, fn: Callable[[T], "Result[U, Any]"]) -> "Result[U, Any]":
        """
        Monadic bind for chaining fallible operations.
        
        Complexity: O(f) where f is complexity of fn
        
        Example:
            def parse_int(s: str) -> Result[int, str]:
                try:
                    return Ok(int(s))
                except ValueError:
                    return Err(f"Invalid int: {s}")
                    
            Ok("42").flat_map(parse_int)  # Ok(42)
        """
        return fn(self._value)
    
    def and_then(self, fn: Callable[[T], "Result[U, Any]"]) -> "Result[U, Any]":
        """Alias for flat_map - Rust naming convention."""
        return fn(self._value)
    
    def or_else(self, fn: Callable[[Any], "Result[T, Any]"]) -> "Ok[T]":
        """No-op on success - returns self unchanged."""
        return self
    
    @property
    def value(self) -> T:
        """Property accessor for the wrapped value."""
        return self._value
    
    @property
    def error(self) -> None:
        """No error on success variant."""
        return None
    
    def __repr__(self) -> str:
        return f"Ok({self._value!r})"
    
    def __bool__(self) -> Literal[True]:
        """Ok is always truthy."""
        return True


# =============================================================================
# RESULT MONAD: ERROR VARIANT
# =============================================================================
@final
@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """
    Error variant of Result monad.
    
    Immutable container for error information.
    Carries full error context for exhaustive handling.
    
    Memory Layout (64-bit):
        _error: 8 bytes (pointer to E)
        Total: 8 bytes + sizeof(E)
        
    Example:
        result: Result[int, str] = Err("Division by zero")
        if result.is_err():
            print(f"Error: {result.error}")
    """
    _error: E
    
    def is_ok(self) -> Literal[False]:
        """O(1) success check."""
        return False
    
    def is_err(self) -> Literal[True]:
        """O(1) error check."""
        return True
    
    def unwrap(self) -> NoReturn:
        """
        Attempting to unwrap an error is a programming error.
        
        Raises:
            RuntimeError: Always, with error context
        """
        raise RuntimeError(f"Called unwrap() on Err: {self._error}")
    
    def unwrap_or(self, default: T) -> T:
        """Return default value on error. Complexity: O(1)."""
        return default
    
    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Compute fallback value on error. Complexity: O(f)."""
        return f()
    
    def expect(self, msg: str) -> NoReturn:
        """Panic with custom message."""
        raise RuntimeError(f"{msg}: {self._error}")
    
    def map(self, fn: Callable[[Any], U]) -> "Err[E]":
        """No-op on error variant - propagates error unchanged."""
        return self
    
    def map_err(self, fn: Callable[[E], U]) -> "Err[U]":
        """Transform error value."""
        return Err(fn(self._error))
    
    def flat_map(self, fn: Callable[[Any], "Result[U, E]"]) -> "Err[E]":
        """Propagate error through monadic chain."""
        return self
    
    def and_then(self, fn: Callable[[Any], "Result[U, E]"]) -> "Err[E]":
        """Alias for flat_map - propagates error."""
        return self
    
    def or_else(self, fn: Callable[[E], "Result[T, Any]"]) -> "Result[T, Any]":
        """Try recovery on error."""
        return fn(self._error)
    
    @property
    def value(self) -> None:
        """No value on error variant."""
        return None
    
    @property
    def error(self) -> E:
        """Property accessor for the error."""
        return self._error
    
    def __repr__(self) -> str:
        return f"Err({self._error!r})"
    
    def __bool__(self) -> Literal[False]:
        """Err is always falsy."""
        return False


# Union type for pattern matching
Result = Union[Ok[T], Err[E]]


# =============================================================================
# ERROR TAXONOMY: STRUCTURED ERROR HIERARCHY
# =============================================================================
class ErrorCode(Enum):
    """
    Canonical error codes for categorization and metrics.
    
    Ranges:
        1000-1999: Index errors
        2000-2999: Query errors
        3000-3999: Embedding errors
        4000-4999: Storage errors
        5000-5999: Configuration errors
        9000-9999: Internal errors
    """
    # Index errors (1000-1999)
    INDEX_NOT_FOUND = 1001
    INDEX_ALREADY_EXISTS = 1002
    INDEX_CAPACITY_EXCEEDED = 1003
    INDEX_DIMENSION_MISMATCH = 1004
    INDEX_CORRUPTED = 1005
    
    # Query errors (2000-2999)
    QUERY_INVALID_VECTOR = 2001
    QUERY_INVALID_K = 2002
    QUERY_TIMEOUT = 2003
    QUERY_FILTER_INVALID = 2004
    
    # Embedding errors (3000-3999)
    EMBEDDING_PROVIDER_ERROR = 3001
    EMBEDDING_RATE_LIMITED = 3002
    EMBEDDING_INVALID_INPUT = 3003
    EMBEDDING_DIMENSION_MISMATCH = 3004
    
    # Storage errors (4000-4999)
    STORAGE_READ_ERROR = 4001
    STORAGE_WRITE_ERROR = 4002
    STORAGE_NOT_FOUND = 4003
    STORAGE_CORRUPTED = 4004
    
    # Configuration errors (5000-5999)
    CONFIG_INVALID = 5001
    CONFIG_MISSING_REQUIRED = 5002
    
    # Internal errors (9000-9999)
    INTERNAL_ERROR = 9001
    NOT_IMPLEMENTED = 9002


@dataclass(frozen=True, slots=True)
class VectorSearchError:
    """
    Base error type for all vector search operations.
    
    Structured error with:
        - Unique error code for categorization
        - Human-readable message
        - Machine-readable details
        - Optional cause chain
        - Timestamp for debugging
        
    Memory Layout:
        code: 4 bytes (enum)
        message: 8 bytes (pointer)
        details: 8 bytes (pointer)  
        cause: 8 bytes (pointer)
        timestamp: 8 bytes (float)
        Total: ~36 bytes + string/dict contents
    """
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    cause: Optional["VectorSearchError"] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/transmission."""
        return {
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "details": self.details,
            "cause": self.cause.to_dict() if self.cause else None,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def with_cause(self, cause: "VectorSearchError") -> "VectorSearchError":
        """Chain errors for root cause analysis."""
        return VectorSearchError(
            code=self.code,
            message=self.message,
            details=self.details,
            cause=cause,
            timestamp=self.timestamp,
        )


# =============================================================================
# SPECIALIZED ERROR TYPES (Convenience constructors)
# =============================================================================
class IndexError(VectorSearchError):
    """Error during index operations (insert, delete, build)."""
    
    @classmethod
    def not_found(cls, index_name: str) -> "IndexError":
        return cls(
            code=ErrorCode.INDEX_NOT_FOUND,
            message=f"Index '{index_name}' not found",
            details={"index_name": index_name},
        )
    
    @classmethod
    def dimension_mismatch(cls, expected: int, actual: int) -> "IndexError":
        return cls(
            code=ErrorCode.INDEX_DIMENSION_MISMATCH,
            message=f"Dimension mismatch: expected {expected}, got {actual}",
            details={"expected": expected, "actual": actual},
        )
    
    @classmethod
    def capacity_exceeded(cls, current: int, max_capacity: int) -> "IndexError":
        return cls(
            code=ErrorCode.INDEX_CAPACITY_EXCEEDED,
            message=f"Index capacity exceeded: {current}/{max_capacity}",
            details={"current": current, "max_capacity": max_capacity},
        )


class QueryError(VectorSearchError):
    """Error during query/search operations."""
    
    @classmethod
    def invalid_vector(cls, reason: str) -> "QueryError":
        return cls(
            code=ErrorCode.QUERY_INVALID_VECTOR,
            message=f"Invalid query vector: {reason}",
            details={"reason": reason},
        )
    
    @classmethod
    def invalid_k(cls, k: int, max_k: int) -> "QueryError":
        return cls(
            code=ErrorCode.QUERY_INVALID_K,
            message=f"Invalid k={k}, must be in [1, {max_k}]",
            details={"k": k, "max_k": max_k},
        )
    
    @classmethod
    def timeout(cls, timeout_ms: float) -> "QueryError":
        return cls(
            code=ErrorCode.QUERY_TIMEOUT,
            message=f"Query timed out after {timeout_ms}ms",
            details={"timeout_ms": timeout_ms},
        )


class EmbeddingError(VectorSearchError):
    """Error during embedding generation."""
    
    @classmethod
    def provider_error(cls, provider: str, message: str) -> "EmbeddingError":
        return cls(
            code=ErrorCode.EMBEDDING_PROVIDER_ERROR,
            message=f"Embedding provider '{provider}' error: {message}",
            details={"provider": provider, "provider_message": message},
        )
    
    @classmethod
    def rate_limited(cls, provider: str, retry_after: Optional[float] = None) -> "EmbeddingError":
        return cls(
            code=ErrorCode.EMBEDDING_RATE_LIMITED,
            message=f"Rate limited by '{provider}'",
            details={"provider": provider, "retry_after_seconds": retry_after},
        )


class ConfigError(VectorSearchError):
    """Error in configuration."""
    
    @classmethod
    def invalid(cls, param: str, value: Any, reason: str) -> "ConfigError":
        return cls(
            code=ErrorCode.CONFIG_INVALID,
            message=f"Invalid config '{param}': {reason}",
            details={"param": param, "value": value, "reason": reason},
        )
    
    @classmethod
    def missing(cls, param: str) -> "ConfigError":
        return cls(
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
            message=f"Missing required config: {param}",
            details={"param": param},
        )


class StorageError(VectorSearchError):
    """Error in storage operations."""
    
    @classmethod
    def read_error(cls, path: str, reason: str) -> "StorageError":
        return cls(
            code=ErrorCode.STORAGE_READ_ERROR,
            message=f"Failed to read '{path}': {reason}",
            details={"path": path, "reason": reason},
        )
    
    @classmethod
    def write_error(cls, path: str, reason: str) -> "StorageError":
        return cls(
            code=ErrorCode.STORAGE_WRITE_ERROR,
            message=f"Failed to write '{path}': {reason}",
            details={"path": path, "reason": reason},
        )
