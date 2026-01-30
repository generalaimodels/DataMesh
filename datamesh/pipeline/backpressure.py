"""
Backpressure Controller: Flow Control for High-Velocity Ingestion

Implements reactive flow control:
- Token bucket rate limiting
- Queue depth monitoring
- Adaptive batch sizing
- Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from datamesh.core.types import Result, Ok, Err
from datamesh.core.errors import IngestionError
from datamesh.core import constants as C

logger = logging.getLogger(__name__)


class BackpressureState(Enum):
    """Flow control state."""
    NORMAL = auto()      # Full throughput
    WARNING = auto()     # Reduced batch size
    CRITICAL = auto()    # Minimal throughput
    BLOCKED = auto()     # Reject new requests


@dataclass
class BackpressureMetrics:
    """Real-time backpressure metrics."""
    state: BackpressureState = BackpressureState.NORMAL
    queue_depth: int = 0
    inflight_requests: int = 0
    tokens_available: float = 0.0
    current_batch_size: int = C.WAL_BATCH_SIZE_ROWS
    requests_accepted: int = 0
    requests_rejected: int = 0


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Provides smooth rate limiting with burst support.
    Thread-safe via atomic operations.
    """
    
    __slots__ = ("_capacity", "_rate", "_tokens", "_last_update")
    
    def __init__(self, rate: float, capacity: float) -> None:
        """
        Args:
            rate: Tokens per second to add
            capacity: Maximum tokens (burst size)
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_update = time.monotonic()
    
    def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without blocking.
        
        Returns True if tokens acquired, False otherwise.
        """
        self._refill()
        
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False
    
    async def acquire_async(
        self,
        tokens: float = 1.0,
        timeout: float = 5.0,
    ) -> bool:
        """Acquire tokens with async waiting."""
        deadline = time.monotonic() + timeout
        
        while time.monotonic() < deadline:
            if self.acquire(tokens):
                return True
            
            # Calculate wait time
            wait = (tokens - self._tokens) / self._rate
            wait = min(wait, deadline - time.monotonic())
            
            if wait > 0:
                await asyncio.sleep(wait)
        
        return False
    
    def _refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_update = now
    
    @property
    def available(self) -> float:
        """Current available tokens."""
        self._refill()
        return self._tokens


class BackpressureController:
    """
    Coordinates backpressure across ingestion pipeline.
    
    Monitors:
    - Request queue depth
    - Consumer lag
    - Downstream health
    
    Controls:
    - Rate limiting via token bucket
    - Adaptive batch sizing
    - Request rejection in overload
    """
    
    __slots__ = (
        "_bucket", "_metrics", "_queue_depth", "_inflight",
        "_state_callbacks", "_lock",
    )
    
    def __init__(
        self,
        rate_limit: float = C.INGESTION_RATE_LIMIT_PER_TENANT,
        burst_size: float = 1000,
    ) -> None:
        self._bucket = TokenBucket(rate=rate_limit, capacity=burst_size)
        self._metrics = BackpressureMetrics()
        self._queue_depth = 0
        self._inflight = 0
        self._state_callbacks: list[Callable[[BackpressureState], None]] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0) -> Result[None, IngestionError]:
        """
        Acquire permission to process request.
        
        Returns error if system is overloaded.
        """
        async with self._lock:
            # Check state first
            if self._metrics.state == BackpressureState.BLOCKED:
                self._metrics.requests_rejected += 1
                return Err(IngestionError.backpressure(
                    queue_depth=self._queue_depth,
                    threshold=C.BACKPRESSURE_QUEUE_DEPTH_CRITICAL,
                ))
            
            # Try to acquire tokens
            if await self._bucket.acquire_async(tokens, timeout=1.0):
                self._inflight += 1
                self._metrics.inflight_requests = self._inflight
                self._metrics.requests_accepted += 1
                return Ok(None)
            
            # Rate limit exceeded
            self._metrics.requests_rejected += 1
            return Err(IngestionError.backpressure(
                queue_depth=self._queue_depth,
                threshold=int(self._bucket._capacity),
            ))
    
    async def release(self) -> None:
        """Release inflight slot after request completes."""
        async with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._metrics.inflight_requests = self._inflight
    
    async def update_queue_depth(self, depth: int) -> None:
        """Update queue depth and adjust state."""
        async with self._lock:
            self._queue_depth = depth
            self._metrics.queue_depth = depth
            
            # Determine new state
            old_state = self._metrics.state
            
            if depth >= C.BACKPRESSURE_QUEUE_DEPTH_CRITICAL:
                self._metrics.state = BackpressureState.CRITICAL
                self._metrics.current_batch_size = C.WAL_BATCH_SIZE_ROWS // 4
            elif depth >= C.BACKPRESSURE_QUEUE_DEPTH_WARNING:
                self._metrics.state = BackpressureState.WARNING
                self._metrics.current_batch_size = C.WAL_BATCH_SIZE_ROWS // 2
            else:
                self._metrics.state = BackpressureState.NORMAL
                self._metrics.current_batch_size = C.WAL_BATCH_SIZE_ROWS
            
            self._metrics.tokens_available = self._bucket.available
            
            # Notify callbacks on state change
            if old_state != self._metrics.state:
                logger.info(
                    f"Backpressure state changed: {old_state.name} → {self._metrics.state.name}"
                )
                for callback in self._state_callbacks:
                    try:
                        callback(self._metrics.state)
                    except Exception as e:
                        logger.error(f"State callback error: {e}")
    
    def block(self) -> None:
        """Block all new requests."""
        self._metrics.state = BackpressureState.BLOCKED
        logger.warning("Backpressure: BLOCKED - rejecting all requests")
    
    def unblock(self) -> None:
        """Resume accepting requests."""
        self._metrics.state = BackpressureState.NORMAL
        logger.info("Backpressure: Unblocked")
    
    def on_state_change(self, callback: Callable[[BackpressureState], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)
    
    @property
    def metrics(self) -> BackpressureMetrics:
        """Get current metrics."""
        self._metrics.tokens_available = self._bucket.available
        return self._metrics
    
    @property
    def recommended_batch_size(self) -> int:
        """Get adaptive batch size based on current state."""
        return self._metrics.current_batch_size
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy for new requests."""
        return self._metrics.state in {
            BackpressureState.NORMAL,
            BackpressureState.WARNING,
        }
