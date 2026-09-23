"""Integration tests for encryption-at-rest of ChatSession.title
(app/services/session_repository.py) -- the encryption-scope expansion
that extends the existing AES-256-GCM primitive (app/core/encryption.py,
already covering ChatTurn.content via postgres_session_store.py) to the
other genuinely sensitive text column in the same table.

Mirrors tests/test_postgres_session_store_encryption.py's fixtures and
methodology exactly (real SQLite-backed SQLAlchemy session, real
INSERT/SELECT, not a mock of the storage layer) so both encrypted
columns are proven with the same rigor.
"""

import base64

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.core.encryption import EncryptionIntegrityError, EncryptionKeyMissingError, generate_key
from app.models.db_models import ChatSession, ChatTurn  # noqa: F401 -- registers tables on Base
from app.services import session_repository as repo_module
from app.services.session_repository import list_sessions, set_session_title_if_unset


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(repo_module, "SessionLocal", session_local)
    monkeypatch.setattr(repo_module, "db_enabled", lambda: True)
    return session_local


@pytest.fixture
def encryption_key(monkeypatch):
    key = generate_key()
    monkeypatch.setattr(settings, "encryption_key_b64", key)
    return key


def _create_session(session_local, tenant_id: int = 1) -> str:
    import uuid
    from datetime import UTC, datetime

    session_id = str(uuid.uuid4())
    with session_local() as db:
        db.add(
            ChatSession(
                session_id=session_id,
                tenant_id=tenant_id,
                created_at=datetime.now(UTC),
                last_accessed_at=datetime.now(UTC),
            )
        )
        db.commit()
    return session_id


def _raw_title(session_local, session_id: str) -> str | None:
    with session_local() as db:
        return db.get(ChatSession, session_id).title


class TestEncryptedWrite:
    def test_persisted_title_is_not_plaintext(self, sqlite_session_local, encryption_key):
        session_id = _create_session(sqlite_session_local)
        sensitive_title = "My social security number is 123-45-6789"

        set_session_title_if_unset(session_id, sensitive_title)

        raw_stored = _raw_title(sqlite_session_local, session_id)
        assert raw_stored is not None
        assert sensitive_title not in raw_stored
        assert "123-45-6789" not in raw_stored
        assert raw_stored.startswith("enc1:")


class TestDecryptRoundTrip:
    def test_list_sessions_returns_decrypted_title(self, sqlite_session_local, encryption_key):
        session_id = _create_session(sqlite_session_local, tenant_id=7)
        original = "What does the document say about apple scab treatment?"

        set_session_title_if_unset(session_id, original)
        sessions = list_sessions(tenant_id=7)

        assert len(sessions) == 1
        assert sessions[0]["title"] == original

    def test_title_only_set_once(self, sqlite_session_local, encryption_key):
        session_id = _create_session(sqlite_session_local, tenant_id=1)
        set_session_title_if_unset(session_id, "first title")
        set_session_title_if_unset(session_id, "second title should be ignored")

        sessions = list_sessions(tenant_id=1)
        assert sessions[0]["title"] == "first title"

    def test_null_title_stays_null_and_is_not_an_error(self, sqlite_session_local, encryption_key):
        _create_session(sqlite_session_local, tenant_id=1)
        sessions = list_sessions(tenant_id=1)
        assert sessions[0]["title"] is None


class TestWrongKey:
    def test_decrypting_with_a_different_key_fails_safely(self, sqlite_session_local, encryption_key):
        from app.core.encryption import decrypt_bytes

        session_id = _create_session(sqlite_session_local)
        set_session_title_if_unset(session_id, "secret title")

        raw_stored = _raw_title(sqlite_session_local, session_id)
        ciphertext = base64.b64decode(raw_stored[len("enc1:"):])

        wrong_key = generate_key()
        with pytest.raises(EncryptionIntegrityError):
            decrypt_bytes(ciphertext, key_b64=wrong_key, associated_data=session_id.encode())


class TestTamperedCiphertext:
    def test_tampered_title_fails_to_decrypt(self, sqlite_session_local, encryption_key):
        session_id = _create_session(sqlite_session_local, tenant_id=1)
        set_session_title_if_unset(session_id, "untampered title")

        with sqlite_session_local() as db:
            session = db.get(ChatSession, session_id)
            corrupted = session.title[:-4] + ("0" if session.title[-4] != "0" else "1") + session.title[-3:]
            session.title = corrupted
            db.commit()

        with pytest.raises(EncryptionIntegrityError):
            list_sessions(tenant_id=1)


class TestMissingKey:
    def test_title_write_fails_closed_without_a_configured_key(self, sqlite_session_local, monkeypatch):
        monkeypatch.setattr(settings, "encryption_key_b64", None)
        session_id = _create_session(sqlite_session_local, tenant_id=1)

        # Best-effort by design (title-setting must never fail the chat
        # request it piggybacks on) -- but the missing-key failure is
        # swallowed, never bypassed by writing plaintext instead.
        set_session_title_if_unset(session_id, "should never be persisted")

        assert _raw_title(sqlite_session_local, session_id) is None

    def test_title_read_fails_closed_if_key_removed_after_writing(self, sqlite_session_local, encryption_key, monkeypatch):
        session_id = _create_session(sqlite_session_local, tenant_id=1)
        set_session_title_if_unset(session_id, "written while key was present")

        monkeypatch.setattr(settings, "encryption_key_b64", None)

        with pytest.raises(EncryptionKeyMissingError):
            list_sessions(tenant_id=1)


class TestSessionIsolation:
    def test_ciphertext_from_one_session_cannot_be_decrypted_under_another(self, sqlite_session_local, encryption_key):
        from app.core.encryption import decrypt_bytes

        session_a = _create_session(sqlite_session_local, tenant_id=1)
        session_b = _create_session(sqlite_session_local, tenant_id=1)
        set_session_title_if_unset(session_a, "session A's private title")

        raw_a = _raw_title(sqlite_session_local, session_a)
        ciphertext = base64.b64decode(raw_a[len("enc1:"):])

        with pytest.raises(EncryptionIntegrityError):
            decrypt_bytes(ciphertext, key_b64=encryption_key, associated_data=session_b.encode())  # wrong session_id as AAD


class TestBackwardCompatibilityWithLegacyPlaintextRows:
    def test_legacy_plaintext_title_reads_back_unchanged(self, sqlite_session_local, encryption_key):
        session_id = _create_session(sqlite_session_local, tenant_id=1)

        with sqlite_session_local() as db:
            session = db.get(ChatSession, session_id)
            session.title = "a legacy plaintext title"
            db.commit()

        sessions = list_sessions(tenant_id=1)
        assert sessions[0]["title"] == "a legacy plaintext title"

    def test_new_title_writes_are_always_encrypted_even_when_legacy_titles_exist(self, sqlite_session_local, encryption_key):
        legacy_session = _create_session(sqlite_session_local, tenant_id=1)
        with sqlite_session_local() as db:
            session = db.get(ChatSession, legacy_session)
            session.title = "legacy plaintext title"
            db.commit()

        new_session = _create_session(sqlite_session_local, tenant_id=1)
        set_session_title_if_unset(new_session, "new title, must be encrypted")

        assert _raw_title(sqlite_session_local, legacy_session) == "legacy plaintext title"
        assert _raw_title(sqlite_session_local, new_session).startswith("enc1:")

        sessions = {s["session_id"]: s["title"] for s in list_sessions(tenant_id=1)}
        assert sessions[legacy_session] == "legacy plaintext title"
        assert sessions[new_session] == "new title, must be encrypted"


class TestPlaintextLeakage:
    def test_synthetic_marker_never_appears_in_raw_storage(self, sqlite_session_local, encryption_key):
        marker = "TEST_SENSITIVE_VALUE_9f3e7c21"
        session_id = _create_session(sqlite_session_local, tenant_id=1)

        set_session_title_if_unset(session_id, f"note about {marker}")

        raw_stored = _raw_title(sqlite_session_local, session_id)
        assert marker not in raw_stored

        sessions = list_sessions(tenant_id=1)
        assert marker in sessions[0]["title"]  # recovered correctly through the application path
