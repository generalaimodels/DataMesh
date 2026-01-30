"""
Structured Logging: JSON-Formatted with Correlation IDs

Provides:
- JSON-formatted log output
- Automatic trace/span ID injection
- Log level filtering
- Context propagation

Designed for centralized log aggregation (ELK, Loki).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Optional, TextIO

from datamesh.observability.tracing import Tracer


class LogLevel(IntEnum):
    """Log level enumeration."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# Context variable for request-scoped fields
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: str
    level: str
    message: str
    logger_name: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "@timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "logger": self.logger_name,
        }
        
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        
        data.update(self.extra)
        
        return json.dumps(data, default=str)


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter with trace correlation.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Get trace context
        trace_id = None
        span_id = None
        
        current_span = Tracer.get_current_span()
        if current_span:
            trace_id = current_span.context.trace_id
            span_id = current_span.context.span_id
        
        # Get context variables
        extra = dict(_log_context.get())
        
        # Add record extras
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename",
                "funcName", "levelname", "levelno", "lineno",
                "module", "msecs", "pathname", "process",
                "processName", "relativeCreated", "stack_info",
                "exc_info", "exc_text", "thread", "threadName",
            }:
                extra[key] = value
        
        # Handle exception info
        if record.exc_info:
            extra["exception"] = self.formatException(record.exc_info)
        
        log_record = LogRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            trace_id=trace_id,
            span_id=span_id,
            extra=extra,
        )
        
        return log_record.to_json()


class StructuredLogger:
    """
    Structured logger with context propagation.
    
    Usage:
        logger = StructuredLogger("datamesh.ingestion")
        
        with logger.context(entity_id="123"):
            logger.info("Processing request")
    """
    
    __slots__ = ("_logger", "_default_extra")
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
        self._default_extra: dict[str, Any] = {}
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        extra = {**self._default_extra, **kwargs}
        self._logger.log(level.value, message, extra=extra)
    
    def with_extra(self, **kwargs: Any) -> StructuredLogger:
        """Create child logger with additional default fields."""
        new_logger = StructuredLogger(self._logger.name)
        new_logger._logger = self._logger
        new_logger._default_extra = {**self._default_extra, **kwargs}
        return new_logger
    
    @staticmethod
    def context(**kwargs: Any):
        """Context manager for request-scoped fields."""
        return _LogContext(kwargs)


class _LogContext:
    """Context manager for adding fields to all logs."""
    
    __slots__ = ("_fields", "_token", "_old")
    
    def __init__(self, fields: dict[str, Any]) -> None:
        self._fields = fields
        self._token = None
        self._old: dict[str, Any] = {}
    
    def __enter__(self) -> _LogContext:
        self._old = _log_context.get()
        new_context = {**self._old, **self._fields}
        self._token = _log_context.set(new_context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        if self._token:
            _log_context.reset(self._token)


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    json_output: bool = True,
    stream: Optional[TextIO] = None,
) -> None:
    """
    Configure root logger for structured logging.
    
    Args:
        level: Minimum log level
        json_output: Use JSON formatting
        stream: Output stream (default: stderr)
    """
    root = logging.getLogger()
    root.setLevel(level.value)
    
    # Remove existing handlers
    root.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level.value)
    
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
    
    root.addHandler(handler)
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
