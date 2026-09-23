"""In-memory chat session store.

Stores conversation history keyed by session_id. This is an in-memory
store — not persisted to disk — because the current deployment target
(Render free tier) has an ephemeral filesystem that wipes on every
spin-down/redeploy (see docs/OPERATIONS.md "The constraint that shapes
everything below: an ephemeral filesystem"). A file-backed store would
face the exact same problem and wouldn't buy more durability than
in-memory on this environment; it would just add complexity for a
durability guarantee this deployment can't actually provide.

If/when the deployment moves to a platform with a real persistent volume
or a database, this module is the single place to swap in a persistent
backend (e.g. Redis, PostgreSQL, or even a JSON file on a mounted disk).
The public API (get_history, append_turn, create_session) would stay the
same.

Retention: bounded by max_sessions (default 1000) with LRU eviction.
No TTL — session count is capped, not time-based, to avoid OOM on
memory-constrained deployments (Render free tier: 512MB). This is a
deliberate simplification over a full TTL system; the cap alone prevents
unbounded growth.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, cast

logger = logging.getLogger(__name__)


class HistoryTurn(TypedDict):
    role: str  # "user" | "assistant"
    content: str


class SessionStore(Protocol):
    """Structural interface shared by InMemorySessionStore (below) and
    PostgresSessionStore (services/postgres_session_store.py) — the two
    backends get_session_store() picks between. A Protocol rather than a
    shared base class: both already implement matching method signatures
    independently, so this just gives get_session_store()'s return value
    (and the module-level singleton) a real static type instead of forcing
    either concrete class on callers."""

    def create_session(self, tenant_id: int | None = None) -> str: ...

    # dict[str, Any], not HistoryTurn: this is the shape callers (query.py,
    # rag_service.py) actually consume — ChatRequest.history is already
    # `list[dict[str, Any]] | None` (see app/models/schemas.py) and gets
    # assigned into the same local as this method's result when a session
    # has no server-side history yet, so the two must match exactly.
    # HistoryTurn remains InMemorySessionStore's internal storage shape.
    def get_history(self, session_id: str) -> list[dict[str, Any]] | None: ...

    def append_turn(self, session_id: str, role: str, content: str) -> bool: ...

    def get_or_create_session(
        self, session_id: str | None, tenant_id: int | None = None
    ) -> str: ...

    def delete_session(self, session_id: str) -> bool: ...

    def session_exists(self, session_id: str) -> bool: ...

    def get_session_count(self) -> int: ...


@dataclass
class Session:
    session_id: str
    tenant_id: int | None = None
    history: list[HistoryTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class InMemorySessionStore:
    """Thread-safe in-memory session store for chat history.

    Bounded by max_sessions with LRU eviction to prevent unbounded
    memory growth — the direct cause of the Render free-tier OOM.
    Each session's history is also capped at max_turns_per_session
    to prevent a few heavy sessions from exhausting memory.
    """

    def __init__(self, max_sessions: int = 1000, max_turns_per_session: int = 50):
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if max_turns_per_session <= 0:
            raise ValueError("max_turns_per_session must be positive")
        self._max_sessions = max_sessions
        self._max_turns_per_session = max_turns_per_session
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _evict_lru_if_needed(self) -> None:
        """Evict least-recently-accessed session if at capacity.

        Called with lock held.
        """
        if len(self._sessions) >= self._max_sessions:
            # Find session with oldest last_accessed
            lru_session_id = min(
                self._sessions.keys(), key=lambda sid: self._sessions[sid].last_accessed
            )
            self._sessions.pop(lru_session_id)
            logger.info(
                "session_evicted_lru",
                extra={
                    "extra_fields": {
                        "session_id": lru_session_id,
                        "max_sessions": self._max_sessions,
                    }
                },
            )

    def create_session(self, tenant_id: int | None = None) -> str:
        """Create a new empty session and return its session_id.

        Evicts LRU session if at capacity. tenant_id is stored on the
        session so get_or_create_session can refuse to hand it back to a
        different tenant later — memory bounds are still by max_sessions,
        not partitioned by tenant, only ownership is tracked.
        """
        session_id = str(uuid.uuid4())
        with self._lock:
            self._evict_lru_if_needed()
            self._sessions[session_id] = Session(session_id=session_id, tenant_id=tenant_id)
        logger.info("session_created", extra={"extra_fields": {"session_id": session_id}})
        return session_id

    def get_history(self, session_id: str) -> list[dict[str, Any]] | None:
        """Return history for session_id, or None if not found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.last_accessed = time.time()
            # Return a copy to prevent external mutation. HistoryTurn is a
            # TypedDict (a plain dict at runtime, just with fixed keys), so
            # this cast only widens the static type back to the
            # SessionStore Protocol's dict[str, Any] shape.
            return cast("list[dict[str, Any]]", list(session.history))

    def append_turn(self, session_id: str, role: str, content: str) -> bool:
        """Append a turn to the session's history. Returns False if session not found.

        Trims oldest turns if history exceeds max_turns_per_session.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.history.append({"role": role, "content": content})
            # Trim to max_turns_per_session (keep most recent)
            if len(session.history) > self._max_turns_per_session:
                session.history = session.history[-self._max_turns_per_session :]
            session.last_accessed = time.time()
        return True

    def get_or_create_session(self, session_id: str | None, tenant_id: int | None = None) -> str:
        """If session_id is provided, exists, and its owner isn't
        verifiably a different tenant, return it. Otherwise create a new
        session.

        A session whose stored tenant_id is known and doesn't match the
        requester's is never handed back — without this, knowing (or
        guessing) another tenant's session_id would read and extend their
        conversation history. Ownership is only enforced when both sides
        are known (session.tenant_id and the requester's tenant_id);
        either being None (DB disabled — no multi-tenancy in effect, or a
        session created before tenant tracking existed) skips the check,
        same "unknown isn't a mismatch" rule used everywhere else tenant
        ownership is checked in this app. On a mismatch, this transparently
        starts a fresh session rather than raising — the same recovery
        path an unknown/expired session_id already takes.
        """
        if session_id is not None:
            with self._lock:
                session = self._sessions.get(session_id)
                if session is not None and (
                    session.tenant_id is None or tenant_id is None or session.tenant_id == tenant_id
                ):
                    return session_id
        return self.create_session(tenant_id=tenant_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def get_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Global singleton instance
_session_store: SessionStore | None = None
_session_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Return the global session store instance (created on first call).

    Returns the PostgreSQL-backed store when the DB is enabled, else the
    in-memory store — both implement the same interface (create_session,
    get_history, append_turn, get_or_create_session, delete_session,
    session_exists, get_session_count), so callers don't care which they
    get. Thread-safe lazy initialization using double-checked locking.
    """
    global _session_store
    from app.core.database import db_enabled

    if db_enabled():
        from app.services.postgres_session_store import PostgresSessionStore

        if _session_store is None:
            with _session_store_lock:
                if _session_store is None:
                    _session_store = PostgresSessionStore()
        return _session_store

    if _session_store is None:
        with _session_store_lock:
            if _session_store is None:
                _session_store = InMemorySessionStore()
    return _session_store
