"""
Distributed Tracing: OpenTelemetry-Compatible Span Management

Provides request correlation and latency breakdown:
- Span contexts with trace/span IDs
- Parent-child relationships
- Baggage propagation
- Sampling control

Sampling: 100% for errors, 1% for success paths.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    """Span completion status."""
    UNSET = auto()
    OK = auto()
    ERROR = auto()


@dataclass
class SpanContext:
    """Immutable span context for propagation."""
    trace_id: str  # 32-char hex
    span_id: str   # 16-char hex
    parent_span_id: Optional[str] = None
    sampled: bool = True
    baggage: dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def generate(cls, parent: Optional[SpanContext] = None) -> SpanContext:
        """Generate new span context."""
        if parent:
            return cls(
                trace_id=parent.trace_id,
                span_id=secrets.token_hex(8),
                parent_span_id=parent.span_id,
                sampled=parent.sampled,
                baggage=dict(parent.baggage),
            )
        return cls(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            sampled=True,
        )
    
    def to_headers(self) -> dict[str, str]:
        """Export context as W3C Trace Context headers."""
        # traceparent: version-trace_id-span_id-flags
        flags = "01" if self.sampled else "00"
        traceparent = f"00-{self.trace_id}-{self.span_id}-{flags}"
        
        headers = {"traceparent": traceparent}
        
        if self.baggage:
            # tracestate/baggage header
            baggage_items = [f"{k}={v}" for k, v in self.baggage.items()]
            headers["baggage"] = ",".join(baggage_items)
        
        return headers
    
    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional[SpanContext]:
        """Parse context from W3C Trace Context headers."""
        traceparent = headers.get("traceparent", "")
        
        if not traceparent:
            return None
        
        parts = traceparent.split("-")
        if len(parts) != 4:
            return None
        
        try:
            version, trace_id, span_id, flags = parts
            sampled = flags == "01"
            
            baggage = {}
            if "baggage" in headers:
                for item in headers["baggage"].split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        baggage[k.strip()] = v.strip()
            
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                sampled=sampled,
                baggage=baggage,
            )
        except Exception:
            return None


@dataclass
class Span:
    """
    Individual trace span.
    
    Represents a unit of work with timing and metadata.
    """
    name: str
    context: SpanContext
    start_time_ns: int
    end_time_ns: Optional[int] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)
    
    @property
    def duration_ns(self) -> Optional[int]:
        """Span duration in nanoseconds."""
        if self.end_time_ns:
            return self.end_time_ns - self.start_time_ns
        return None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Span duration in milliseconds."""
        if self.duration_ns:
            return self.duration_ns / 1_000_000
        return None
    
    def set_attribute(self, key: str, value: Any) -> Span:
        """Add attribute to span."""
        self.attributes[key] = value
        return self
    
    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> Span:
        """Add timestamped event."""
        self.events.append((time.time_ns(), name, attributes or {}))
        return self
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> Span:
        """Set span status."""
        self.status = status
        if message:
            self.attributes["status_message"] = message
        return self
    
    def end(self, status: Optional[SpanStatus] = None) -> None:
        """End span timing."""
        self.end_time_ns = time.time_ns()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    def to_dict(self) -> dict[str, Any]:
        """Export span as dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "status": self.status.name,
            "attributes": self.attributes,
            "events": [
                {"time_ns": t, "name": n, "attributes": a}
                for t, n, a in self.events
            ],
        }


# Context variable for current span
_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)


class Tracer:
    """
    Distributed tracer for request correlation.
    
    Usage:
        tracer = Tracer("datamesh")
        
        with tracer.start_span("ingest") as span:
            span.set_attribute("entity_id", str(entity_id))
            process_request()
    """
    
    __slots__ = (
        "_service_name", "_sample_rate", "_exporter",
        "_spans", "_lock",
    )
    
    _instance: Optional[Tracer] = None
    
    def __init__(
        self,
        service_name: str,
        sample_rate: float = 1.0,
        exporter: Optional[Callable[[Span], None]] = None,
    ) -> None:
        self._service_name = service_name
        self._sample_rate = sample_rate
        self._exporter = exporter
        self._spans: list[Span] = []
        self._lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, service_name: str = "datamesh") -> Tracer:
        """Get singleton tracer."""
        if cls._instance is None:
            cls._instance = cls(service_name)
        return cls._instance
    
    @contextmanager
    def start_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """
        Start a new span as context manager.
        
        Automatically links to current span if no parent specified.
        """
        # Get parent from context if not provided
        if parent is None:
            current = _current_span.get()
            if current:
                parent = current.context
        
        # Generate context
        context = SpanContext.generate(parent)
        
        # Determine sampling
        if not self._should_sample(context):
            context = SpanContext(
                trace_id=context.trace_id,
                span_id=context.span_id,
                parent_span_id=context.parent_span_id,
                sampled=False,
            )
        
        # Create span
        span = Span(
            name=name,
            context=context,
            start_time_ns=time.time_ns(),
            attributes=attributes or {},
        )
        span.attributes["service.name"] = self._service_name
        
        # Set as current
        token = _current_span.set(span)
        
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            span.end()
            _current_span.reset(token)
            
            if context.sampled:
                self._export(span)
    
    def _should_sample(self, context: SpanContext) -> bool:
        """Determine if span should be sampled."""
        # If parent is sampled, sample this span
        if context.parent_span_id and context.sampled:
            return True
        
        # Random sampling
        return secrets.randbelow(1000) < int(self._sample_rate * 1000)
    
    def _export(self, span: Span) -> None:
        """Export completed span."""
        with self._lock:
            self._spans.append(span)
            
            # Keep bounded
            if len(self._spans) > 10000:
                self._spans = self._spans[-5000:]
        
        if self._exporter:
            try:
                self._exporter(span)
            except Exception as e:
                logger.warning(f"Span export failed: {e}")
    
    @staticmethod
    def get_current_span() -> Optional[Span]:
        """Get current span from context."""
        return _current_span.get()
    
    @staticmethod
    def get_current_trace_id() -> Optional[str]:
        """Get current trace ID."""
        span = _current_span.get()
        return span.context.trace_id if span else None
    
    def get_recent_spans(self, limit: int = 100) -> list[Span]:
        """Get recent completed spans."""
        with self._lock:
            return list(self._spans[-limit:])
    
    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject current span context into carrier."""
        span = _current_span.get()
        if span:
            carrier.update(span.context.to_headers())
    
    def extract_context(self, carrier: dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from carrier."""
        return SpanContext.from_headers(carrier)
