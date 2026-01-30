"""
Retry Policy: Exponential Backoff with Jitter

Implements robust retry strategy:
- Exponential backoff: 100ms × 2^n
- Full jitter: random(0, backoff) to prevent thundering herd
- Max retries: 3 for idempotent, 0 for non-idempotent

Timeout tiers: Connection (5s), Request (30s), Global (60s)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence, Type, TypeVar

from datamesh.core.types import Result, Ok, Err
from datamesh.core.errors import ReliabilityError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Retry configuration."""
    
    max_retries: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 10000
    exponential_base: float = 2.0
    jitter: bool = True  # Full jitter
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: tuple[Type[Exception], ...] = ()
    
    # Timeout tiers
    connection_timeout_s: float = 5.0
    request_timeout_s: float = 30.0
    global_timeout_s: float = 60.0
    
    @classmethod
    def default(cls) -> RetryPolicy:
        """Default retry policy."""
        return cls()
    
    @classmethod
    def no_retry(cls) -> RetryPolicy:
        """No retries (for non-idempotent operations)."""
        return cls(max_retries=0)
    
    @classmethod
    def aggressive(cls) -> RetryPolicy:
        """Aggressive retries for critical operations."""
        return cls(
            max_retries=5,
            base_delay_ms=50,
            max_delay_ms=30000,
        )


@dataclass
class RetryStats:
    """Retry attempt statistics."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay_ms: float = 0.0
    last_error: Optional[str] = None


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    policy: Optional[RetryPolicy] = None,
) -> Result[T, ReliabilityError]:
    """
    Execute async function with retry and exponential backoff.
    
    Args:
        func: Async function to execute
        policy: Retry configuration (default if None)
        
    Returns:
        Ok with result or Err after exhausting retries
    """
    if policy is None:
        policy = RetryPolicy.default()
    
    stats = RetryStats()
    last_exception: Optional[Exception] = None
    
    global_deadline = time.monotonic() + policy.global_timeout_s
    
    for attempt in range(policy.max_retries + 1):
        stats.total_attempts += 1
        
        # Check global timeout
        if time.monotonic() >= global_deadline:
            return Err(ReliabilityError.retry_exhausted(
                attempts=stats.total_attempts,
                last_error=str(last_exception) if last_exception else "Timeout",
            ))
        
        try:
            # Apply request timeout
            result = await asyncio.wait_for(
                func(),
                timeout=policy.request_timeout_s,
            )
            stats.successful_attempts += 1
            return Ok(result)
            
        except asyncio.TimeoutError as e:
            last_exception = e
            stats.failed_attempts += 1
            logger.debug(f"Attempt {attempt + 1} timed out")
            
        except policy.non_retryable_exceptions as e:
            # Non-retryable: fail immediately
            return Err(ReliabilityError.retry_exhausted(
                attempts=stats.total_attempts,
                last_error=str(e),
            ))
            
        except policy.retryable_exceptions as e:
            last_exception = e
            stats.failed_attempts += 1
            stats.last_error = str(e)
            logger.debug(f"Attempt {attempt + 1} failed: {e}")
        
        # Calculate backoff delay
        if attempt < policy.max_retries:
            delay = calculate_backoff(
                attempt=attempt,
                base_delay_ms=policy.base_delay_ms,
                max_delay_ms=policy.max_delay_ms,
                exponential_base=policy.exponential_base,
                jitter=policy.jitter,
            )
            stats.total_delay_ms += delay
            
            logger.debug(f"Retrying in {delay}ms (attempt {attempt + 2})")
            await asyncio.sleep(delay / 1000)
    
    return Err(ReliabilityError.retry_exhausted(
        attempts=stats.total_attempts,
        last_error=str(last_exception) if last_exception else "Unknown error",
    ))


def calculate_backoff(
    attempt: int,
    base_delay_ms: int,
    max_delay_ms: int,
    exponential_base: float,
    jitter: bool,
) -> float:
    """
    Calculate backoff delay with optional jitter.
    
    Full jitter: random(0, min(cap, base * 2^attempt))
    """
    # Exponential delay
    delay = min(max_delay_ms, base_delay_ms * (exponential_base ** attempt))
    
    # Full jitter
    if jitter:
        delay = random.uniform(0, delay)
    
    return delay


class RetryContext:
    """
    Context manager for retry operations.
    
    Usage:
        async with RetryContext(policy) as ctx:
            for attempt in ctx.attempts():
                try:
                    result = await risky_operation()
                    ctx.success()
                    break
                except Exception as e:
                    await ctx.fail(e)
    """
    
    __slots__ = ("_policy", "_attempt", "_last_error", "_succeeded")
    
    def __init__(self, policy: Optional[RetryPolicy] = None) -> None:
        self._policy = policy or RetryPolicy.default()
        self._attempt = 0
        self._last_error: Optional[Exception] = None
        self._succeeded = False
    
    async def __aenter__(self) -> RetryContext:
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        pass
    
    def attempts(self) -> range:
        """Iterator over retry attempts."""
        return range(self._policy.max_retries + 1)
    
    def success(self) -> None:
        """Mark operation as successful."""
        self._succeeded = True
    
    async def fail(self, error: Exception) -> None:
        """Record failure and wait for backoff."""
        self._last_error = error
        self._attempt += 1
        
        if self._attempt <= self._policy.max_retries:
            delay = calculate_backoff(
                attempt=self._attempt - 1,
                base_delay_ms=self._policy.base_delay_ms,
                max_delay_ms=self._policy.max_delay_ms,
                exponential_base=self._policy.exponential_base,
                jitter=self._policy.jitter,
            )
            await asyncio.sleep(delay / 1000)
    
    @property
    def succeeded(self) -> bool:
        return self._succeeded
    
    @property
    def last_error(self) -> Optional[Exception]:
        return self._last_error
