"""Per-client API key authentication with basic rate limiting, plus
individual user login (JWT) as a second, parallel auth path.

A small table of keys (sha256_hash -> client_name) loaded from
Settings.api_key_table. Clients send their key in the X-API-Key header.
On success, the client_name is stored in request.state.client_name for
downstream audit logging, alongside request.state.tenant_id and
request.state.role (both None when the DB is disabled — see
tenant_service.resolve_tenant). This is require_api_key, unchanged from
before JWT login existed — it's still exactly what non-browser/service
clients (scripts, CI, a future public API) use.

require_auth (below) is what routers actually depend on: it checks for
an `Authorization: Bearer <jwt>` header first (individual user login,
see app/services/user_service.py and app/api/v1/routes/auth.py) and
falls through to require_api_key unchanged if that header is absent.
Either path ends up populating the same request.state fields
(client_name/tenant_id/role) that permissions.py and every audit log
already read — callers don't need to know or care which one fired.

Lookup is O(1) via dict keyed by hash — no linear scan, no bcrypt cost
(bcrypt is used for user *passwords* in user_service.py, a deliberately
different, slower primitive — see that module's docstring).

Rate limiting: simple per-client sliding window (default 60 req/min per key).
Uses an in-memory store with thread-safe operations. Not persisted —
resets on restart, which is acceptable for a free-tier demo.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field

from fastapi import Header, Request

from app.core.config import settings
from app.core.exceptions import RateLimitExceededError, UnauthorizedError
from app.core.request_context import set_client_name
from app.services.tenant_service import find_api_key, resolve_tenant
from app.services.user_service import decode_access_token, get_user_by_id

logger = logging.getLogger(__name__)


@dataclass
class _RateLimitBucket:
    """Sliding-window request counter for a single client."""

    requests: list[float] = field(default_factory=list)  # timestamps
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check_and_record(self, limit: int, window_sec: int) -> bool:
        """Return True if request allowed, False if rate limited.

        Removes timestamps older than window_sec, then checks count.
        """
        now = time.time()
        with self.lock:
            # Drop expired
            cutoff = now - window_sec
            self.requests = [ts for ts in self.requests if ts > cutoff]
            if len(self.requests) >= limit:
                return False
            self.requests.append(now)
            return True


# Global rate limit store (in-memory, per client_name). Capped to
# _RATE_LIMIT_STORE_MAX distinct identities so a process that has seen many
# different API keys (a real concern over a long-running deployment) doesn't
# grow this dict without bound — the oldest idle identity is evicted on a
# miss once the cap is hit. The sliding-window timestamps within each bucket
# are still pruned per-check; this cap is only about the number of tracked
# *identities*, not their request counts.
_RATE_LIMIT_STORE_MAX = 1000
_rate_limit_store: dict[str, _RateLimitBucket] = {}
_rate_limit_lock = threading.Lock()


def _check_rate_limit(client_name: str) -> bool:
    """Check and record a request for the given client. Returns True if allowed.

    Reads its limit/window from Settings (rate_limit_per_minute /
    rate_limit_window_seconds) rather than a hardcoded constant, so a
    deployment tuning RATE_LIMIT_PER_MINUTES sees the cap apply to both the
    per-identity (this) limiter and the per-IP limiter in security.py.
    """
    with _rate_limit_lock:
        bucket = _rate_limit_store.get(client_name)
        if bucket is None:
            if len(_rate_limit_store) >= _RATE_LIMIT_STORE_MAX:
                # Evict the identity with the oldest most-recent request so
                # the store can't grow without bound. Each bucket keeps its
                # own timestamps in insertion order via check_and_record's
                # append, so the oldest request time is the first one still
                # present; fall back to any key for the truly-stale corner.
                oldest = min(
                    _rate_limit_store,
                    key=lambda name: (
                        _rate_limit_store[name].requests[0]
                        if _rate_limit_store[name].requests
                        else 0.0
                    ),
                    default=client_name,
                )
                _rate_limit_store.pop(oldest, None)
            bucket = _RateLimitBucket()
            _rate_limit_store[client_name] = bucket
    return bucket.check_and_record(
        settings.rate_limit_per_minute,
        settings.rate_limit_window_seconds,
    )


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key is None:
        logger.warning(
            "audit_event",
            extra={
                "extra_fields": {
                    "event": "auth_failed",
                    "path": request.url.path,
                    "reason": "missing_header",
                }
            },
        )
        raise UnauthorizedError("Missing or invalid API key.")

    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    key_table = settings.api_key_table
    client_name = key_table.get(key_hash)
    if client_name is None:
        # Not an env-configured key — try the DB api_keys table (covers
        # keys added directly to PostgreSQL, not present in API_KEYS env).
        db_lookup = find_api_key(key_hash)
        if db_lookup is not None:
            client_name, _tenant_id, _role = db_lookup
    if client_name is not None:
        # Check rate limit
        if not _check_rate_limit(client_name):
            logger.warning(
                "audit_event",
                extra={
                    "extra_fields": {
                        "event": "rate_limited",
                        "path": request.url.path,
                        "client": client_name,
                    }
                },
            )
            raise RateLimitExceededError("Rate limit exceeded. Please slow down.")
        # Resolve the tenant (and role) for this client (creating the row
        # on first sight when the DB is enabled); both None when the DB is
        # disabled, in which case downstream persistence AND role-gating
        # are also disabled (see documents.py's delete route).
        resolved = resolve_tenant(client_name)
        tenant_id, role = resolved if resolved is not None else (None, None)
        request.state.client_name = client_name
        request.state.tenant_id = tenant_id
        request.state.role = role
        set_client_name(client_name)
        return

    logger.warning(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "auth_failed",
                "path": request.url.path,
                "reason": "invalid_key",
            }
        },
    )
    raise UnauthorizedError("Missing or invalid API key.")


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """The dependency routers actually declare. Tries JWT first
    (Authorization: Bearer <token>, individual user login); falls
    through to require_api_key unchanged if that header is absent, so
    every existing API-key caller keeps working exactly as before.

    A *present but invalid* Bearer token is rejected outright rather
    than silently falling through to try X-API-Key too — a client that
    sent a bearer token meant to authenticate with it, and a silent
    fallback would mask an expired/malformed token as some other kind
    of failure.
    """
    if authorization is not None and authorization.lower().startswith("bearer "):
        token = authorization[len("bearer ") :].strip()
        user_id = decode_access_token(token)
        resolved = get_user_by_id(user_id) if user_id is not None else None
        if resolved is None:
            logger.warning(
                "audit_event",
                extra={
                    "extra_fields": {
                        "event": "auth_failed",
                        "path": request.url.path,
                        "reason": "invalid_token",
                    }
                },
            )
            raise UnauthorizedError("Invalid or expired token.")

        email, tenant_id, role = resolved
        # Rate limit the JWT path too — the README's "60 req/min per
        # identity" applies to both auth methods, and an unthrottled JWT
        # path would let a stolen/loose token hammer the API the same way
        # the (now IP-limited) login endpoint would have been hammered.
        if not _check_rate_limit(email):
            logger.warning(
                "audit_event",
                extra={
                    "extra_fields": {
                        "event": "rate_limited",
                        "path": request.url.path,
                        "client": email,
                    }
                },
            )
            raise RateLimitExceededError("Rate limit exceeded. Please slow down.")
        request.state.client_name = email
        request.state.tenant_id = tenant_id
        request.state.role = role
        request.state.user_id = user_id
        request.state.auth_method = "jwt"
        set_client_name(email)
        return

    require_api_key(request, x_api_key)
    request.state.user_id = None
    request.state.auth_method = "api_key"
