"""
Circuit Breaker: Fault Tolerance for External Dependencies

Implements Hystrix-style circuit breaker:
- CLOSED: Normal operation, requests flow through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing recovery, limited requests allowed

Prevents cascade failures by failing fast when downstream is unhealthy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar

from datamesh.core.types import Result, Ok, Err
from datamesh.core.errors import ReliabilityError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker state."""
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Failing fast
    HALF_OPEN = auto()  # Testing recovery


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    state: CircuitState
    failures: int
    successes: int
    consecutive_failures: int
    last_failure_time: Optional[float]
    last_state_change: float
    total_requests: int
    rejected_requests: int


class CircuitBreaker:
    """
    Circuit breaker with configurable thresholds.
    
    Usage:
        breaker = CircuitBreaker("database")
        
        result = await breaker.call(async_operation)
        
        # Or with decorator
        @breaker.wrap
        async def fetch_data():
            ...
    """
    
    __slots__ = (
        "_name", "_failure_threshold", "_success_threshold",
        "_timeout_seconds", "_state", "_failures", "_successes",
        "_consecutive_failures", "_last_failure_time",
        "_last_state_change", "_lock", "_total_requests",
        "_rejected_requests", "_fallback",
    )
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 30.0,
        fallback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        Args:
            name: Identifier for logging/metrics
            failure_threshold: Failures before opening circuit
            success_threshold: Successes in half-open to close
            timeout_seconds: Time before testing recovery
            fallback: Optional fallback function
        """
        self._name = name
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout_seconds = timeout_seconds
        self._fallback = fallback
        
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._consecutive_failures = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change = time.monotonic()
        self._total_requests = 0
        self._rejected_requests = 0
        self._lock = asyncio.Lock()
    
    async def call(
        self,
        func: Callable[[], T],
    ) -> Result[T, ReliabilityError]:
        """
        Execute function through circuit breaker.
        
        Returns:
            Ok with result or Err with CircuitOpenError
        """
        async with self._lock:
            self._total_requests += 1
            
            # Check state transitions
            self._check_state_transition()
            
            if self._state == CircuitState.OPEN:
                self._rejected_requests += 1
                logger.warning(f"Circuit '{self._name}' is OPEN, rejecting request")
                
                if self._fallback:
                    return Ok(self._fallback())
                
                return Err(ReliabilityError.circuit_open(
                    circuit_name=self._name,
                    failure_count=self._failures,
                    retry_after_seconds=int(self._time_until_half_open()),
                ))
        
        # Execute outside lock
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            
            await self._on_success()
            return Ok(result)
            
        except Exception as e:
            await self._on_failure()
            
            if self._fallback:
                return Ok(self._fallback())
            
            return Err(ReliabilityError.circuit_open(
                circuit_name=self._name,
                failure_count=self._consecutive_failures,
                retry_after_seconds=int(self._timeout_seconds),
            ))
    
    async def _on_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._successes += 1
            self._consecutive_failures = 0
            
            if self._state == CircuitState.HALF_OPEN:
                if self._successes >= self._success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    async def _on_failure(self) -> None:
        """Record failed call."""
        async with self._lock:
            self._failures += 1
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()
            
            if self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open reopens
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self._failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            if self._time_until_half_open() <= 0:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.monotonic()
        
        if new_state == CircuitState.CLOSED:
            self._failures = 0
            self._consecutive_failures = 0
            self._successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._successes = 0
        
        logger.info(f"Circuit '{self._name}': {old_state.name} → {new_state.name}")
    
    def _time_until_half_open(self) -> float:
        """Time remaining before testing recovery."""
        if self._last_failure_time is None:
            return 0
        
        elapsed = time.monotonic() - self._last_failure_time
        return max(0, self._timeout_seconds - elapsed)
    
    def wrap(self, func: Callable[..., T]) -> Callable[..., Result[T, ReliabilityError]]:
        """Decorator to wrap function with circuit breaker."""
        async def wrapper(*args: Any, **kwargs: Any) -> Result[T, ReliabilityError]:
            return await self.call(lambda: func(*args, **kwargs))
        return wrapper
    
    def force_open(self) -> None:
        """Manually open circuit."""
        self._transition_to(CircuitState.OPEN)
    
    def force_close(self) -> None:
        """Manually close circuit."""
        self._transition_to(CircuitState.CLOSED)
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    @property
    def stats(self) -> CircuitStats:
        return CircuitStats(
            state=self._state,
            failures=self._failures,
            successes=self._successes,
            consecutive_failures=self._consecutive_failures,
            last_failure_time=self._last_failure_time,
            last_state_change=self._last_state_change,
            total_requests=self._total_requests,
            rejected_requests=self._rejected_requests,
        )


class CircuitBreakerRegistry:
    """Global registry of circuit breakers."""
    
    _breakers: dict[str, CircuitBreaker] = {}
    
    @classmethod
    def get(cls, name: str, **kwargs: Any) -> CircuitBreaker:
        """Get or create circuit breaker by name."""
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(name, **kwargs)
        return cls._breakers[name]
    
    @classmethod
    def all_stats(cls) -> dict[str, CircuitStats]:
        """Get stats for all breakers."""
        return {name: cb.stats for name, cb in cls._breakers.items()}
