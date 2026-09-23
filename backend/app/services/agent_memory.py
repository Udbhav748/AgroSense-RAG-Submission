"""Persistent agent memory: per-session conversation turns and durable
key/value facts that survive history trimming and give the pipeline a
working memory across turns.

Two halves, mirroring the checklist's "ConversationMemory + FactMemory":

- turns — the last N user/assistant exchanges per session (bounded, same
  LRU discipline as InMemorySessionStore). The live prompt already carries
  recent turns via session_store, so this store is the durable/
  supplementary half: recorded alongside the main pipeline, retrievable
  for inspection, and the substrate for future cross-session recall.
- facts — durable key/value claims extracted from the assistant's answers
  (key, value, confidence, source chunk ids). build_context() formats
  these into a "Remembered from earlier in this conversation" block that
  ChatService injects into the router and generation prompts, so a
  follow-up like "and what about the other document?" can draw on what
  was established earlier without re-searching.

In-memory and bounded, exactly like session_store and approval_service:
the deployment target (Render free tier) has an ephemeral filesystem, so a
file/DB-backed store would add complexity for a durability guarantee that
environment can't provide. Bounds mirror the session store's (max_sessions
LRU eviction; max_turns and max_facts per session) so memory can't grow
without limit.

Failure-safe by construction: every operation is best-effort and never
raises into the pipeline — memory is a quality enhancement, never a new
failure mode.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class _MemoryFact:
    """One durable fact remembered for a session: a short key, the claim
    itself, the extraction confidence, and which retrieved chunks (if any)
    it was grounded on. updated_at is the recency signal for the
    per-session fact LRU bound."""

    key: str
    value: str
    confidence: str = "medium"  # "high" | "medium" | "low"
    source_chunk_ids: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


@dataclass
class _SessionMemory:
    """Everything AgentMemory remembers for one session. facts is keyed by
    the (lowercased) fact key so upserting is an O(1) update, not a scan."""

    session_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)  # [{"role", "content"}]
    facts: dict[str, _MemoryFact] = field(default_factory=dict)
    last_accessed: float = field(default_factory=time.time)


class AgentMemory:
    """Thread-safe, bounded per-session agent memory.

    Same shape as InMemorySessionStore: a lock-guarded dict keyed by
    session_id with LRU eviction at max_sessions and per-session caps on
    turns and facts. A session's memory is created lazily on first write
    (reads of an unknown session return empty results, never raise).
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        max_turns_per_session: int = 10,
        max_facts_per_session: int = 20,
    ):
        if max_sessions <= 0 or max_turns_per_session <= 0 or max_facts_per_session <= 0:
            raise ValueError("agent memory bounds must all be positive")
        self._max_sessions = max_sessions
        self._max_turns_per_session = max_turns_per_session
        self._max_facts_per_session = max_facts_per_session
        self._sessions: dict[str, _SessionMemory] = {}
        self._lock = threading.Lock()

    # -- internal helpers (callers hold the lock) ---------------------------

    def _evict_session_lru_if_needed(self) -> None:
        """Evict the least-recently-accessed session once at capacity."""
        if len(self._sessions) >= self._max_sessions:
            lru_id = min(self._sessions.keys(), key=lambda sid: self._sessions[sid].last_accessed)
            self._sessions.pop(lru_id)
            logger.info(
                "agent_memory_session_evicted_lru",
                extra={"extra_fields": {"session_id": lru_id, "max_sessions": self._max_sessions}},
            )

    def _evict_fact_lru_if_needed(self, session: _SessionMemory) -> None:
        """Evict the least-recently-updated fact once a session is at its
        fact cap (oldest-established fact goes first)."""
        if len(session.facts) >= self._max_facts_per_session:
            lru_key = min(session.facts.keys(), key=lambda k: session.facts[k].updated_at)
            session.facts.pop(lru_key)
            logger.info(
                "agent_memory_fact_evicted_lru",
                extra={
                    "extra_fields": {
                        "session_id": session.session_id,
                        "max_facts_per_session": self._max_facts_per_session,
                    }
                },
            )

    def _get_locked(self, session_id: str) -> _SessionMemory:
        """Return the session's memory, creating it (with LRU eviction) on
        first touch. Caller must hold the lock."""
        session = self._sessions.get(session_id)
        if session is None:
            self._evict_session_lru_if_needed()
            session = _SessionMemory(session_id=session_id)
            self._sessions[session_id] = session
        return session

    # -- turns ---------------------------------------------------------------

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Record one turn, trimming to the session's turn cap (oldest
        first). Best-effort: unknown sessions are created lazily."""
        with self._lock:
            session = self._get_locked(session_id)
            session.turns.append({"role": role, "content": content})
            if len(session.turns) > self._max_turns_per_session:
                session.turns = session.turns[-self._max_turns_per_session :]
            session.last_accessed = time.time()

    def get_recent_turns(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the most recent turns (oldest-first), capped at limit
        (default: the session's turn cap). [] for an unknown session."""
        limit = limit or self._max_turns_per_session
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_accessed = time.time()
            return list(session.turns[-limit:])

    # -- facts ---------------------------------------------------------------

    def upsert_fact(
        self,
        session_id: str,
        key: str,
        value: str,
        confidence: str = "medium",
        source_chunk_ids: list[str] | None = None,
    ) -> None:
        """Store or update a fact for a session. Keys are normalized to
        lowercase; empty keys/values are silently ignored. An existing key
        is updated in place (value, confidence, source chunks, recency) —
        the freshest claim wins, and the fact stays at the top of the
        per-session LRU order."""
        key = str(key).strip().lower()
        value = str(value).strip()
        if not key or not value:
            return
        confidence = confidence if confidence in ("high", "medium", "low") else "medium"
        with self._lock:
            session = self._get_locked(session_id)
            existing = session.facts.get(key)
            if existing is not None:
                existing.value = value
                existing.confidence = confidence
                existing.source_chunk_ids = list(source_chunk_ids or [])
                existing.updated_at = time.time()
            else:
                self._evict_fact_lru_if_needed(session)
                session.facts[key] = _MemoryFact(
                    key=key,
                    value=value,
                    confidence=confidence,
                    source_chunk_ids=list(source_chunk_ids or []),
                    updated_at=time.time(),
                )
            session.last_accessed = time.time()

    def get_fact(self, session_id: str, key: str) -> dict[str, Any] | None:
        """Return one fact as a dict (key/value/confidence/
        source_chunk_ids), or None if the session or key is unknown."""
        key = str(key).strip().lower()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.last_accessed = time.time()
            fact = session.facts.get(key)
            if fact is None:
                return None
            return {
                "key": fact.key,
                "value": fact.value,
                "confidence": fact.confidence,
                "source_chunk_ids": list(fact.source_chunk_ids),
            }

    def get_all_facts(self, session_id: str) -> list[dict[str, Any]]:
        """Return every fact for a session (insertion order). [] for an
        unknown session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_accessed = time.time()
            return [
                {
                    "key": fact.key,
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "source_chunk_ids": list(fact.source_chunk_ids),
                }
                for fact in session.facts.values()
            ]

    # -- prompt injection ------------------------------------------------------

    def build_context(self, session_id: str) -> str:
        """Format the session's remembered facts as a prompt block, or ""
        when nothing is remembered yet. The conversation turns are NOT
        included here — they already flow into the prompt through the live
        history passed by the route; this block is the durable fact layer
        on top of them."""
        facts = self.get_all_facts(session_id)
        if not facts:
            return ""
        lines = ["Remembered from earlier in this conversation (established facts):"]
        for fact in facts:
            lines.append(f"- {fact['key']}: {fact['value']}")
        return "\n".join(lines)

    # -- lifecycle -------------------------------------------------------------

    def delete_session(self, session_id: str) -> bool:
        """Forget everything remembered for a session. True if it existed."""
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


# Global singleton instance, same lazy thread-safe pattern as the session
# store (session_store.py) and approval queue (approval_service.py).
_agent_memory: AgentMemory | None = None
_agent_memory_lock = threading.Lock()


def get_agent_memory() -> AgentMemory:
    """Return the process-wide agent memory, created once on first call
    with the configured bounds. Shared across sessions/requests the same
    way get_session_store() is — memory keyed by session_id inside it."""
    global _agent_memory
    if _agent_memory is None:
        with _agent_memory_lock:
            if _agent_memory is None:
                _agent_memory = AgentMemory(
                    max_sessions=settings.agent_memory_max_sessions,
                    max_turns_per_session=settings.agent_memory_max_turns_per_session,
                    max_facts_per_session=settings.agent_memory_max_facts_per_session,
                )
    return _agent_memory
