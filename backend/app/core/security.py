"""Input validation utilities, an IP-based rate limiter for the
unauthenticated auth surface, and security headers applied to every
response.

This module was previously dead code: the broken `backend.app.core`
import path made it un-importable, and nothing in main.py ever registered
its middlewares. It's now wired in two places:
  - main.py registers security_headers_middleware() on every response.
  - the /auth routes depend on rate_limit_dependency() (see
    routes/auth.py) to put an IP-based sliding-window cap on the
    unauthenticated brute-force surface (login/signup), which has no
    client identity to key a per-client limiter on.

Authenticated requests are rate-limited separately, per identity, in
core/auth.py (API keys and JWT users) — this module deliberately does not
apply an IP cap to them, so a NAT'd office sharing one IP can't starve
itself of the app it's already authenticated to.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from app.core.config import settings
from app.core.exceptions import RateLimitExceededError


class RateLimiter:
    """Simple in-memory sliding-window rate limiter, keyed by any string
    identity (an IP address for the unauthenticated auth surface).

    Thread- and event-loop-safe: a plain threading.Lock guards the shared
    dict (the check-and-record must be atomic), so the same instance works
    whether the dependency runs in a thread or on the event loop.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        """Return True if a request from client_id is allowed, else False.

        Async so the internal asyncio lock keeps concurrent check-and-
        record atomic even across threads dispatched onto the loop.
        """
        now = time.time()
        cutoff = now - self.window_seconds
        async with self._lock:
            pending = self.requests[client_id]
            pending[:] = [req_time for req_time in pending if req_time > cutoff]
            if len(pending) >= self.max_requests:
                return False
            pending.append(now)
            return True


def get_client_ip(request: Request) -> str:
    """Best-effort client IP, honoring the X-Forwarded-For header only
    when the direct TCP peer is a configured trusted proxy (Settings.
    trusted_proxy_ips), and degrading to a stable placeholder when even
    request.client is None (e.g. some test clients / ASGI servers).

    X-Forwarded-For is attacker-controlled on any request that reaches
    this process directly (no proxy in front) or through an untrusted
    intermediary — trusting it unconditionally would let a client rotate
    a fake header value per request to defeat rate_limit_dependency's
    per-IP brute-force limiter on /auth/login and /auth/signup. With no
    trusted proxy configured (the default), X-Forwarded-For is ignored
    entirely and request.client.host is used instead.
    """
    peer = request.client.host if request.client is not None else None
    if peer is not None and peer in settings.trusted_proxy_ips_set:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # XFF can be a comma-separated chain: "client, proxy1, proxy2".
            # Only trust the leftmost entry (the original client) — still
            # only reached once the direct peer itself is a trusted proxy.
            first = forwarded.split(",")[0].strip()
            if first:
                return first
    if peer is not None:
        return peer
    return "unknown"


async def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency: reject a request from this IP if it's exceeded
    the unauthenticated-auth rate limit. Applied to /auth/signup and
    /auth/login (see routes/auth.py) — the only endpoints without an
    identity to key a per-client limiter on. Skipped when
    Settings.rate_limit_enabled is False."""
    if not settings.rate_limit_enabled:
        return
    client_id = get_client_ip(request)
    if not await rate_limiter.is_allowed(client_id):
        raise RateLimitExceededError("Too many requests. Please try again later.")


# Global per-IP rate limiter instance for the unauthenticated auth surface.
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_per_minute,
    window_seconds=settings.rate_limit_window_seconds,
)


# Security headers middleware — registered on the app in main.py so every
# response (including errors) carries a baseline of hardening headers.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

# This backend is a JSON API — every response but the three paths below is
# either JSON or an error page, never markup that runs a browser-executed
# script. A strict CSP is still worth sending (the FastAPI-generated
# /openapi.json page, a debug=True traceback page, or any future HTML
# response inherit it for free), but /docs and /redoc are FastAPI's own
# Swagger UI / ReDoc pages, which load their JS/CSS from a CDN
# (cdn.jsdelivr.net) and run inline bootstrap scripts — a strict CSP would
# break both, and neither page reflects user input, so it's not the
# surface this header is defending.
_CSP_EXEMPT_PREFIXES = ("/docs", "/redoc")
_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)


async def security_headers_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Add security headers to all responses."""
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    if not request.url.path.startswith(_CSP_EXEMPT_PREFIXES):
        response.headers["Content-Security-Policy"] = _CONTENT_SECURITY_POLICY
    # HSTS is only meaningful (and only safe to advertise) over HTTPS.
    # Including it over plain HTTP would let a MITM strip the header, so
    # gate it on the request scheme.
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
