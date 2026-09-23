"""DB-backed listing/ownership/title for chat sessions — the pieces
`session_store.py`/`postgres_session_store.py` (turn-by-turn history for
the *active* conversation) don't provide. Mirrors
`document_repository.py`'s shape exactly: every function is a no-op /
returns a sensible default when the DB is disabled, so the history-list
routes degrade the same way document listing already does.

ENCRYPTION AT REST (Module 10 gap-closure follow-up): `ChatSession.title`
is populated verbatim from a user's first message in the session (see
`set_session_title_if_unset`) -- it is real, sensitive user-authored
content sitting in the same table as the already-encrypted
`ChatTurn.content`, so it gets the identical AES-256-GCM treatment via
the same shared `app.core.encryption.encrypt_text_field`/
`decrypt_text_field` helpers `postgres_session_store.py` uses for
content, with the same `session_id`-bound associated data and the same
legacy-plaintext-passthrough backward compatibility. `title` differs
from `content` in one way: it's nullable (no title until a first turn
exists), so encryption is only applied when a value is actually being
set/read, never to `None` itself.
"""

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, db_enabled
from app.core.encryption import decrypt_text_field, encrypt_text_field
from app.models.db_models import ChatSession

logger = logging.getLogger(__name__)


def _decrypt_title(stored: str | None, *, session_id: str) -> str | None:
    if stored is None:
        return None
    return decrypt_text_field(stored, associated_data=session_id, key_b64=settings.encryption_key_b64)


def _session() -> Session:
    if SessionLocal is None:
        raise RuntimeError("Database not configured")
    return SessionLocal()


def list_sessions(tenant_id: int | None) -> list[dict[str, Any]]:
    """All sessions for a tenant, most-recently-accessed first. []
    when the DB is disabled or tenant_id is None (no scope to list
    under — matches list_documents(None)'s own behavior for a caller
    with no resolvable tenant, not "everything")."""
    if not db_enabled() or tenant_id is None:
        return []

    with _session() as db:
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.tenant_id == tenant_id)
            .order_by(ChatSession.last_accessed_at.desc())
            .all()
        )
        return [
            {
                "session_id": s.session_id,
                "title": _decrypt_title(s.title, session_id=s.session_id),
                "created_at": s.created_at,
                "last_accessed_at": s.last_accessed_at,
            }
            for s in sessions
        ]


def get_session_owner(session_id: str) -> int | None:
    """tenant_id that owns session_id, or None if unknown — either the
    DB is disabled or no such session exists. Callers treat "unknown"
    as "can't verify, don't block," the same convention
    document_repository.get_document_owner already uses."""
    if not db_enabled():
        return None
    with _session() as db:
        session = db.get(ChatSession, session_id)
        return session.tenant_id if session is not None else None


def set_session_title_if_unset(session_id: str, title: str) -> None:
    """Best-effort: populate title from the first user turn, once — a
    failure here must never fail the chat request it's piggybacking on.

    The stored value is encrypted (see module docstring) via the same
    fail-closed primitive `content` uses. A missing/invalid encryption
    key surfaces as EncryptionKeyMissingError from encrypt_text_field,
    which this function's existing broad except deliberately catches —
    consistent with "best-effort," title simply stays unset in that
    case. It is never written as plaintext as a fallback.
    """
    if not db_enabled():
        return
    try:
        with _session() as db:
            session = db.get(ChatSession, session_id)
            if session is not None and session.title is None:
                encrypted_title = encrypt_text_field(
                    title[:200], associated_data=session_id, key_b64=settings.encryption_key_b64
                )
                session.title = encrypted_title
                db.commit()
    except Exception as exc:
        logger.warning(
            "session_title_set_failed",
            extra={"extra_fields": {"session_id": session_id, "error": str(exc)}},
        )
