"""
HTTP Router: Request Routing and Handler Dispatch

Provides lightweight routing without external dependencies.
Supports:
- Path parameter extraction
- Query string parsing
- Content negotiation
- Method-based dispatch
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional, Sequence
from urllib.parse import parse_qs, urlparse


@dataclass
class Request:
    """HTTP request representation."""
    method: str
    path: str
    query_params: dict[str, list[str]] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    path_params: dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_raw(
        cls,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes = b"",
    ) -> Request:
        """Parse request from raw HTTP data."""
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        return cls(
            method=method.upper(),
            path=parsed.path,
            query_params=query_params,
            headers={k.lower(): v for k, v in headers.items()},
            body=body,
        )
    
    def json(self) -> Any:
        """Parse body as JSON."""
        if not self.body:
            return None
        return json.loads(self.body)
    
    def query(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get first query parameter value."""
        values = self.query_params.get(key, [])
        return values[0] if values else default
    
    def header(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get header value (case-insensitive)."""
        return self.headers.get(key.lower(), default)


@dataclass
class Response:
    """HTTP response representation."""
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def json(
        cls,
        data: Any,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> Response:
        """Create JSON response."""
        body = json.dumps(data, default=str).encode()
        h = headers or {}
        h["content-type"] = "application/json"
        return cls(status=status, body=body, headers=h)
    
    @classmethod
    def error(cls, message: str, status: int = 400) -> Response:
        """Create error response."""
        return cls.json({"error": message}, status=status)
    
    @classmethod
    def not_found(cls) -> Response:
        """Create 404 response."""
        return cls.error("Not found", status=404)
    
    @classmethod
    def method_not_allowed(cls) -> Response:
        """Create 405 response."""
        return cls.error("Method not allowed", status=405)


# Handler function signature
Handler = Callable[[Request], Awaitable[Response]]

# Middleware function signature
Middleware = Callable[[Request, Handler], Awaitable[Response]]


@dataclass
class Route:
    """Route definition."""
    method: str
    pattern: re.Pattern
    handler: Handler
    param_names: list[str]
    
    @classmethod
    def create(cls, method: str, path: str, handler: Handler) -> Route:
        """Create route from path pattern."""
        # Convert path params to regex groups
        param_names: list[str] = []
        
        def replace_param(match: re.Match) -> str:
            param_names.append(match.group(1))
            return r"(?P<" + match.group(1) + r">[^/]+)"
        
        pattern_str = re.sub(r"\{(\w+)\}", replace_param, path)
        pattern_str = f"^{pattern_str}$"
        
        return cls(
            method=method.upper(),
            pattern=re.compile(pattern_str),
            handler=handler,
            param_names=param_names,
        )
    
    def match(self, method: str, path: str) -> Optional[dict[str, str]]:
        """Match request against route."""
        if method.upper() != self.method:
            return None
        
        match = self.pattern.match(path)
        if not match:
            return None
        
        return match.groupdict()


class DataMeshRouter:
    """
    HTTP request router.
    
    Usage:
        router = DataMeshRouter()
        
        @router.get("/api/conversations/{id}")
        async def get_conversation(request: Request) -> Response:
            conv_id = request.path_params["id"]
            ...
        
        response = await router.dispatch(request)
    """
    
    __slots__ = ("_routes", "_middleware", "_prefix")
    
    def __init__(self, prefix: str = "") -> None:
        self._routes: list[Route] = []
        self._middleware: list[Middleware] = []
        self._prefix = prefix
    
    def route(
        self,
        path: str,
        methods: Sequence[str] = ("GET",),
    ) -> Callable[[Handler], Handler]:
        """Register route decorator."""
        def decorator(handler: Handler) -> Handler:
            full_path = self._prefix + path
            for method in methods:
                route = Route.create(method, full_path, handler)
                self._routes.append(route)
            return handler
        return decorator
    
    def get(self, path: str) -> Callable[[Handler], Handler]:
        """Register GET route."""
        return self.route(path, ["GET"])
    
    def post(self, path: str) -> Callable[[Handler], Handler]:
        """Register POST route."""
        return self.route(path, ["POST"])
    
    def put(self, path: str) -> Callable[[Handler], Handler]:
        """Register PUT route."""
        return self.route(path, ["PUT"])
    
    def delete(self, path: str) -> Callable[[Handler], Handler]:
        """Register DELETE route."""
        return self.route(path, ["DELETE"])
    
    def use(self, middleware: Middleware) -> None:
        """Add middleware."""
        self._middleware.append(middleware)
    
    def include(self, router: DataMeshRouter) -> None:
        """Include routes from another router."""
        self._routes.extend(router._routes)
    
    async def dispatch(self, request: Request) -> Response:
        """Route request to handler."""
        # Find matching route
        handler: Optional[Handler] = None
        
        for route in self._routes:
            params = route.match(request.method, request.path)
            if params is not None:
                request.path_params = params
                handler = route.handler
                break
        
        if handler is None:
            # Check if path exists but wrong method
            for route in self._routes:
                if route.pattern.match(request.path):
                    return Response.method_not_allowed()
            return Response.not_found()
        
        # Apply middleware chain
        final_handler = handler
        for mw in reversed(self._middleware):
            final_handler = self._wrap_middleware(mw, final_handler)
        
        try:
            return await final_handler(request)
        except Exception as e:
            return Response.error(str(e), status=500)
    
    def _wrap_middleware(
        self,
        middleware: Middleware,
        handler: Handler,
    ) -> Handler:
        """Wrap handler with middleware."""
        async def wrapped(request: Request) -> Response:
            return await middleware(request, handler)
        return wrapped
