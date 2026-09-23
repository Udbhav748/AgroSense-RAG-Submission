"""Individual login-based accounts (User) and JWT issuance/verification.

Distinct from tenant_service.py, which resolves an API-key client_name
to a tenant. This module additionally *creates* a tenant: each signup
gets its own personal Tenant (1:1), created in the same transaction as
the User row. That's what makes per-user privacy free — Document/
ChatSession are already scoped by tenant_id, so a dedicated tenant per
user isolates them from everyone else using the exact same columns and
ownership-check code that already exists for API-key clients. See
app/api/v1/routes/auth.py for the routes that call this module, and
app/core/auth.py's require_auth for how an issued JWT gets back to
(tenant_id, role) on every subsequent request.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, db_enabled
from app.core.exceptions import (
    AccountLockedError,
    AuthConfigurationError,
    DatabaseNotConfiguredError,
    EmailAlreadyRegisteredError,
    UnauthorizedError,
)
from app.models.db_models import Tenant, User
from app.services.tenant_service import effective_role

logger = logging.getLogger(__name__)


@dataclass
class _LoginAttemptBucket:
    """Sliding-window failed-login counter for a single email — mirrors
    core/auth.py's _RateLimitBucket, but keyed by the account under
    attack rather than the caller, so it survives an attacker rotating
    IPs (or, pre-TRUSTED_PROXY_IPS, spoofing X-Forwarded-For)."""

    failures: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_locked(self, max_attempts: int, window_sec: int) -> bool:
        now = time.time()
        with self.lock:
            cutoff = now - window_sec
            self.failures = [ts for ts in self.failures if ts > cutoff]
            return len(self.failures) >= max_attempts

    def record_failure(self) -> None:
        with self.lock:
            self.failures.append(time.time())


# Global per-email lockout store (in-memory, mirrors core/auth.py's
# _rate_limit_store). Capped to _FAILED_LOGIN_STORE_MAX distinct emails
# so an attacker submitting many different addresses can't grow this
# dict without bound — the oldest idle entry is evicted on a miss once
# the cap is hit.
_FAILED_LOGIN_STORE_MAX = 1000
_failed_login_store: dict[str, _LoginAttemptBucket] = {}
_failed_login_lock = threading.Lock()


def _get_or_create_bucket(email: str) -> _LoginAttemptBucket:
    with _failed_login_lock:
        bucket = _failed_login_store.get(email)
        if bucket is None:
            if len(_failed_login_store) >= _FAILED_LOGIN_STORE_MAX:
                oldest = min(
                    _failed_login_store,
                    key=lambda e: (
                        _failed_login_store[e].failures[0]
                        if _failed_login_store[e].failures
                        else 0.0
                    ),
                    default=email,
                )
                _failed_login_store.pop(oldest, None)
            bucket = _LoginAttemptBucket()
            _failed_login_store[email] = bucket
        return bucket


def _is_locked_out(email: str) -> bool:
    return _get_or_create_bucket(email).is_locked(
        settings.login_lockout_max_attempts, settings.login_lockout_window_seconds
    )


def _record_failed_attempt(email: str) -> None:
    _get_or_create_bucket(email).record_failure()


def _clear_failed_attempts(email: str) -> None:
    with _failed_login_lock:
        _failed_login_store.pop(email, None)


def _session() -> Session:
    if not db_enabled():
        # Unlike tenant_service.py's identically-shaped _session() (whose
        # callers all check db_enabled() themselves first, so this branch
        # never actually fires there), every public function in this
        # module calls straight through to here — so this is the one
        # place that actually raises when DATABASE_URL is unset. A real
        # AppError, not a bare RuntimeError: individual user login being
        # unavailable in a DB-less deployment is a config gap the caller
        # should see clearly (503), not a generic 500.
        raise DatabaseNotConfiguredError(
            "Individual user login requires DATABASE_URL to be configured."
        )
    if SessionLocal is None:
        # db_enabled() being True guarantees this in practice (engine and
        # SessionLocal are always set together — see app/core/database.py),
        # but mypy can't see that invariant across the db_enabled() call.
        raise DatabaseNotConfiguredError(
            "Individual user login requires DATABASE_URL to be configured."
        )
    return SessionLocal()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_user(email: str, password: str) -> tuple[int, int, str]:
    """Create a personal Tenant + User for a new signup, in one
    transaction. Returns (user_id, tenant_id, role). Raises
    EmailAlreadyRegisteredError if the email is already taken.

    slug is a generated uuid, not the raw email — Tenant.slug already
    has its own uniqueness/format expectations from the API-key world
    (client_name values), and a generated id keeps this path from
    colliding with or depending on that format.
    """
    with _session() as db:
        existing = db.query(User).filter(User.email == email).first()
        if existing is not None:
            raise EmailAlreadyRegisteredError(f"An account with email {email} already exists.")

        tenant = Tenant(slug=f"user-{uuid.uuid4()}", name=email)
        db.add(tenant)
        db.flush()  # assigns tenant.id without committing yet

        user = User(email=email, password_hash=_hash_password(password), tenant_id=tenant.id)
        db.add(user)
        db.commit()
        db.refresh(user)
        db.refresh(tenant)

        logger.info(
            "user_created",
            extra={"extra_fields": {"user_id": user.id, "tenant_id": tenant.id}},
        )
        return user.id, tenant.id, effective_role(email, tenant.role)


def authenticate_user(email: str, password: str) -> tuple[int, int, str]:
    """Verify email/password. Returns (user_id, tenant_id, role).

    Raises AccountLockedError before touching the database at all if
    this email has too many recent failures (Settings.
    login_lockout_max_attempts/login_lockout_window_seconds) — checked
    first, and independent of _session()'s DATABASE_URL guard, so a
    locked-out account is rejected the same way regardless of DB state,
    and a repeated attack doesn't cost a query + bcrypt comparison each
    time. Raises UnauthorizedError on any other mismatch — deliberately
    the same error and message for "no such email" and "wrong password"
    so a caller can't enumerate registered emails from the response
    alone.
    """
    if _is_locked_out(email):
        raise AccountLockedError("Too many failed login attempts. Try again in a few minutes.")

    with _session() as db:
        user = db.query(User).filter(User.email == email).first()
        if user is None or not _verify_password(password, user.password_hash):
            _record_failed_attempt(email)
            raise UnauthorizedError("Invalid email or password.")

        _clear_failed_attempts(email)
        tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
        stored_role = tenant.role if tenant is not None else "member"
        return user.id, user.tenant_id, effective_role(email, stored_role)


def get_user_by_id(user_id: int) -> tuple[str, int, str] | None:
    """Return (email, tenant_id, role) for GET /auth/me, or None if the
    user (or its tenant) no longer exists."""
    with _session() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            return None
        tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
        stored_role = tenant.role if tenant is not None else "member"
        return user.email, user.tenant_id, effective_role(user.email, stored_role)


def create_access_token(user_id: int, email: str) -> str:
    """Issue a JWT identifying this user. Raises AuthConfigurationError
    if JWT_SECRET_KEY isn't set — fails loud rather than signing with an
    empty/weak secret, the same posture missing-API-key checks already
    use elsewhere in this codebase."""
    if not settings.jwt_secret_key:
        raise AuthConfigurationError(
            "JWT_SECRET_KEY is not configured; individual user login is unavailable."
        )
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expiry_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    """Return the user_id encoded in a valid, unexpired token, or None
    if the token is missing, expired, malformed, signed with a different
    secret, or was issued before the user's most recent POST /auth/logout
    (see revoke_all_tokens below). Never raises — require_auth treats
    None as "not a JWT-authenticated request" and falls through to the
    API-key path.

    The revocation check only runs when the DB is enabled: a JWT issued
    at all implies DB-backed login was available at issuance time (see
    create_user/authenticate_user's own _session() guard), but a
    deployment could disable DATABASE_URL between issuance and use, and
    degrading to "no revocation possible" (same as never having called
    logout) is safer than raising here — require_auth would otherwise
    turn a config change into a hard failure for every existing session.
    """
    if not settings.jwt_secret_key:
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = int(payload["sub"])
        issued_at = payload["iat"]
    except (jwt.PyJWTError, KeyError, ValueError):
        return None

    if db_enabled():
        with _session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if user is not None and user.tokens_revoked_after is not None:
                issued_at_dt = datetime.fromtimestamp(issued_at, tz=UTC)
                if issued_at_dt < user.tokens_revoked_after:
                    return None

    return user_id


def revoke_all_tokens(user_id: int) -> None:
    """Invalidate every JWT previously issued to this user, effective
    immediately — called by POST /auth/logout. Sets
    User.tokens_revoked_after to now(); decode_access_token rejects any
    token whose `iat` predates it. One column, not a growing per-token
    blocklist: this app issues one active token shape per user (no
    refresh tokens), so "kill everything issued before this instant" is
    the correct, simplest semantic for "log out".

    A no-op (not an error) when the DB is disabled or the user doesn't
    exist — logout degrading to "client-side token discard only" (the
    previous behavior) is the right fallback, not a failure the caller
    needs to handle.
    """
    if not db_enabled():
        return
    with _session() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            return
        user.tokens_revoked_after = datetime.now(UTC)
        db.commit()
