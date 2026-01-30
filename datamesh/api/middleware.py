"""
API Middleware: Cross-Cutting Concerns

Provides:
- RateLimitMiddleware: Token bucket rate limiting
- AuthMiddleware: JWT validation
- TracingMiddleware: Request tracing
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Optional

from datamesh.api.router import Request, Response, Handler
from datamesh.observability.tracing import Tracer, SpanContext
from datamesh.observability.metrics import MetricsCollector, Counter, Histogram
from datamesh.pipeline.backpressure import TokenBucket


class RateLimitMiddleware:
    """
    Token bucket rate limiter middleware.
    
    Limits requests per tenant (identified by X-Tenant-ID header).
    """
    
    __slots__ = ("_rate", "_burst", "_buckets", "_default_bucket")
    
    def __init__(
        self,
        rate: float = 1000.0,  # requests per second
        burst: float = 2000.0, # burst capacity
    ) -> None:
        self._rate = rate
        self._burst = burst
        self._buckets: dict[str, TokenBucket] = {}
        self._default_bucket = TokenBucket(rate, burst)
    
    def _get_bucket(self, tenant_id: str) -> TokenBucket:
        """Get or create bucket for tenant."""
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = TokenBucket(self._rate, self._burst)
        return self._buckets[tenant_id]
    
    async def __call__(
        self,
        request: Request,
        handler: Handler,
    ) -> Response:
        """Apply rate limiting."""
        tenant_id = request.header("x-tenant-id", "default")
        bucket = self._get_bucket(tenant_id)
        
        if not bucket.acquire():
            return Response.json(
                {
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": 1,
                },
                status=429,
                headers={"Retry-After": "1"},
            )
        
        return await handler(request)


class AuthMiddleware:
    """
    JWT authentication middleware.
    
    Validates JWT tokens in Authorization header.
    """
    
    __slots__ = ("_secret", "_algorithms", "_skip_paths")
    
    def __init__(
        self,
        secret: str,
        algorithms: tuple[str, ...] = ("HS256",),
        skip_paths: tuple[str, ...] = ("/health", "/metrics"),
    ) -> None:
        self._secret = secret
        self._algorithms = algorithms
        self._skip_paths = skip_paths
    
    async def __call__(
        self,
        request: Request,
        handler: Handler,
    ) -> Response:
        """Validate authentication."""
        # Skip auth for certain paths
        if request.path in self._skip_paths:
            return await handler(request)
        
        auth_header = request.header("authorization", "")
        
        if not auth_header.startswith("Bearer "):
            return Response.json(
                {"error": "Missing or invalid Authorization header"},
                status=401,
            )
        
        token = auth_header[7:]
        
        try:
            # Simple JWT validation (no external deps)
            claims = self._validate_jwt(token)
            
            # Add claims to request
            request.headers["x-user-id"] = claims.get("sub", "")
            request.headers["x-tenant-id"] = claims.get("tenant_id", "default")
            
            return await handler(request)
            
        except Exception as e:
            return Response.json(
                {"error": f"Invalid token: {e}"},
                status=401,
            )
    
    def _validate_jwt(self, token: str) -> dict[str, Any]:
        """
        Validate JWT token (simplified).
        
        In production, use a proper JWT library.
        """
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        
        header_b64, payload_b64, signature = parts
        
        # Decode payload (simplified - use proper base64url in production)
        import base64
        padding = 4 - len(payload_b64) % 4
        payload_b64 += "=" * padding
        payload_json = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(payload_json)
        
        # Check expiration
        if "exp" in claims and claims["exp"] < time.time():
            raise ValueError("Token expired")
        
        # Verify signature (simplified HMAC)
        expected_sig = hmac.new(
            self._secret.encode(),
            f"{header_b64}.{payload_b64}".encode(),
            hashlib.sha256,
        ).digest()
        
        # In production, compare with actual signature
        
        return claims


class TracingMiddleware:
    """
    Distributed tracing middleware.
    
    Creates spans for each request and propagates trace context.
    """
    
    __slots__ = ("_tracer", "_metrics")
    
    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self._tracer = tracer or Tracer.get_instance()
        self._metrics = metrics or MetricsCollector.get_instance()
        
        # Initialize metrics
        self._request_counter = self._metrics.counter(
            "http_requests_total",
            label_names=["method", "path", "status"],
        )
        self._request_latency = self._metrics.histogram(
            "http_request_duration_seconds",
            label_names=["method", "path"],
        )
    
    async def __call__(
        self,
        request: Request,
        handler: Handler,
    ) -> Response:
        """Create span and record metrics."""
        # Extract parent context from headers
        parent = self._tracer.extract_context(request.headers)
        
        span_name = f"{request.method} {request.path}"
        
        async with self._tracer.start_span(span_name, parent=parent) as span:
            # Add request attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", request.path)
            span.set_attribute("http.user_agent", request.header("user-agent", ""))
            
            start_time = time.perf_counter()
            
            try:
                response = await handler(request)
                
                # Add response attributes
                span.set_attribute("http.status_code", response.status)
                
                return response
                
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise
                
            finally:
                duration = time.perf_counter() - start_time
                
                # Record metrics
                self._request_counter.inc(
                    method=request.method,
                    path=request.path,
                    status=str(response.status if 'response' in locals() else 500),
                )
                self._request_latency.observe(
                    duration,
                    method=request.method,
                    path=request.path,
                )


class CorsMiddleware:
    """
    CORS middleware for cross-origin requests.
    """
    
    __slots__ = ("_origins", "_methods", "_headers")
    
    def __init__(
        self,
        allowed_origins: tuple[str, ...] = ("*",),
        allowed_methods: tuple[str, ...] = ("GET", "POST", "PUT", "DELETE", "OPTIONS"),
        allowed_headers: tuple[str, ...] = ("*",),
    ) -> None:
        self._origins = allowed_origins
        self._methods = allowed_methods
        self._headers = allowed_headers
    
    async def __call__(
        self,
        request: Request,
        handler: Handler,
    ) -> Response:
        """Add CORS headers."""
        origin = request.header("origin", "*")
        
        # Check if origin is allowed
        if "*" not in self._origins and origin not in self._origins:
            return Response.error("Origin not allowed", status=403)
        
        # Handle preflight
        if request.method == "OPTIONS":
            return Response(
                status=204,
                headers=self._cors_headers(origin),
            )
        
        response = await handler(request)
        
        # Add CORS headers to response
        for key, value in self._cors_headers(origin).items():
            response.headers[key] = value
        
        return response
    
    def _cors_headers(self, origin: str) -> dict[str, str]:
        """Generate CORS headers."""
        return {
            "Access-Control-Allow-Origin": origin if "*" not in self._origins else "*",
            "Access-Control-Allow-Methods": ", ".join(self._methods),
            "Access-Control-Allow-Headers": ", ".join(self._headers),
            "Access-Control-Max-Age": "86400",
        }
