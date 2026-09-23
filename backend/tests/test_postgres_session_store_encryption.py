"""Integration tests for encryption-at-rest in PostgresSessionStore
(app/services/postgres_session_store.py) -- the Module 10 gap-closure
that wires the existing AES-256-GCM primitive (app/core/encryption.py)
into REAL persistent storage, not just the primitive in isolation.

Uses a fresh in-memory SQLite database per test (via the same
declarative Base the real Postgres models use) so these exercise the
actual repository/storage implementation -- real INSERT/SELECT through
SQLAlchemy, real ChatSession/ChatTurn rows -- not a mock of the storage
layer. This is also the first test coverage postgres_session_store.py
has had at all.
"""

import base64

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.core.encryption import EncryptionIntegrityError, EncryptionKeyMissingError, generate_key
from app.models.db_models import ChatSession, ChatTurn  # noqa: F401 -- registers tables on Base
from app.services import postgres_session_store as pss_module
from app.services.postgres_session_store import PostgresSessionStore


@pytest.fixture
def sqlite_session_local(monkeypatch):
    """A fresh in-memory SQLite engine per test, wired into
    postgres_session_store's module-level SessionLocal reference (the
    exact name _session() reads) so the store's real code path runs
    against a real, isolated database -- not the shared test DB, and
    never the real Postgres deployment."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(pss_module, "SessionLocal", session_local)
    return session_local


@pytest.fixture
def encryption_key(monkeypatch):
    key = generate_key()
    monkeypatch.setattr(settings, "encryption_key_b64", key)
    return key


def _raw_content_column(session_local, session_id: str) -> list[str]:
    """Reads ChatTurn.content directly via a raw session -- bypassing
    PostgresSessionStore.get_history entirely -- to inspect exactly what
    is actually persisted, not what the store's own decrypt path returns."""
    with session_local() as db:
        session = db.get(ChatSession, session_id)
        return [turn.content for turn in session.turns]


class TestEncryptedWrite:
    def test_persisted_content_is_not_plaintext(self, sqlite_session_local, encryption_key):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)
        sensitive_text = "My social security number is 123-45-6789, please remember it."

        store.append_turn(session_id, "user", sensitive_text)

        raw_values = _raw_content_column(sqlite_session_local, session_id)
        assert len(raw_values) == 1
        raw_stored = raw_values[0]

        # Application value: "My social security number is 123-45-6789, please remember it."
        # Raw persisted value (example shape, redacted length): "enc1:<base64 ciphertext>..."
        assert sensitive_text not in raw_stored
        assert "123-45-6789" not in raw_stored
        assert raw_stored.startswith("enc1:")


class TestDecryptRoundTrip:
    def test_read_through_normal_path_returns_original_plaintext(self, sqlite_session_local, encryption_key):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)
        original = "What does the document say about apple scab treatment?"

        store.append_turn(session_id, "user", original)
        history = store.get_history(session_id)

        assert history == [{"role": "user", "content": original}]

    def test_multiple_turns_round_trip_independently(self, sqlite_session_local, encryption_key):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)
        store.append_turn(session_id, "user", "first message")
        store.append_turn(session_id, "assistant", "first reply")
        store.append_turn(session_id, "user", "second message")

        history = store.get_history(session_id)

        assert [t["content"] for t in history] == ["first message", "first reply", "second message"]


class TestWrongKey:
    def test_decrypting_with_a_different_key_fails_safely(self, sqlite_session_local, encryption_key):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)
        store.append_turn(session_id, "user", "secret content")

        raw_stored = _raw_content_column(sqlite_session_local, session_id)[0]
        ciphertext = base64.b64decode(raw_stored[len("enc1:"):])

        from app.core.encryption import decrypt_bytes

        wrong_key = generate_key()
        with pytest.raises(EncryptionIntegrityError):
            decrypt_bytes(ciphertext, key_b64=wrong_key, associated_data=session_id.encode())


class TestTamperedCiphertext:
    def test_tampered_content_fails_to_decrypt(self, sqlite_session_local, encryption_key):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)
        store.append_turn(session_id, "user", "untampered content")

        with sqlite_session_local() as db:
            session = db.get(ChatSession, session_id)
            turn = session.turns[0]
            # Flip a character deep in the ciphertext -- corrupting the
            # AES-GCM auth tag or ciphertext body, not just the "enc1:" marker.
            corrupted = turn.content[:-4] + ("0" if turn.content[-4] != "0" else "1") + turn.content[-3:]
            turn.content = corrupted
            db.commit()

        with pytest.raises(EncryptionIntegrityError):
            store.get_history(session_id)


class TestMissingKey:
    def test_write_fails_closed_without_a_configured_key(self, sqlite_session_local, monkeypatch):
        monkeypatch.setattr(settings, "encryption_key_b64", None)
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)

        with pytest.raises(EncryptionKeyMissingError):
            store.append_turn(session_id, "user", "should never be persisted")

        # Nothing was silently written as plaintext -- the row simply doesn't exist.
        raw_values = _raw_content_column(sqlite_session_local, session_id)
        assert raw_values == []

    def test_read_fails_closed_if_key_removed_after_writing(self, sqlite_session_local, encryption_key, monkeypatch):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)
        store.append_turn(session_id, "user", "written while key was present")

        monkeypatch.setattr(settings, "encryption_key_b64", None)

        with pytest.raises(EncryptionKeyMissingError):
            store.get_history(session_id)


class TestSessionIsolation:
    def test_ciphertext_from_one_session_cannot_be_decrypted_under_another(self, sqlite_session_local, encryption_key):
        """The session_id is bound as AES-GCM associated data -- a
        ciphertext genuinely written for session A must not decrypt
        successfully if presented as if it belonged to session B, even
        with the correct key. This is an extra guarantee on top of the
        existing session_id foreign key / tenant ownership checks."""
        store = PostgresSessionStore()
        session_a = store.create_session(tenant_id=1)
        session_b = store.create_session(tenant_id=1)
        store.append_turn(session_a, "user", "session A's private content")

        raw_a = _raw_content_column(sqlite_session_local, session_a)[0]
        ciphertext = base64.b64decode(raw_a[len("enc1:"):])

        from app.core.encryption import decrypt_bytes

        with pytest.raises(EncryptionIntegrityError):
            decrypt_bytes(ciphertext, key_b64=encryption_key, associated_data=session_b.encode())  # wrong session_id as AAD

    def test_cross_tenant_session_lookup_still_denied_independent_of_encryption(self, sqlite_session_local, encryption_key):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)
        store.append_turn(session_id, "user", "tenant 1's content")

        # get_or_create_session's existing ownership rule: a tenant_id
        # mismatch never hands back someone else's session -- unchanged
        # by the encryption integration.
        resolved = store.get_or_create_session(session_id, tenant_id=2)
        assert resolved != session_id


class TestBackwardCompatibilityWithLegacyPlaintextRows:
    def test_legacy_plaintext_row_without_marker_reads_back_unchanged(self, sqlite_session_local, encryption_key):
        """A row written before this encryption integration existed (no
        'enc1:' prefix) must remain readable -- no migration is forced,
        and a plaintext row is never mistaken for ciphertext."""
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)

        with sqlite_session_local() as db:
            session = db.get(ChatSession, session_id)
            from datetime import UTC, datetime

            session.turns.append(ChatTurn(role="user", content="a legacy plaintext row", created_at=datetime.now(UTC)))
            db.commit()

        history = store.get_history(session_id)
        assert history == [{"role": "user", "content": "a legacy plaintext row"}]

    def test_new_writes_are_always_encrypted_even_when_legacy_rows_exist(self, sqlite_session_local, encryption_key):
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=1)

        with sqlite_session_local() as db:
            session = db.get(ChatSession, session_id)
            from datetime import UTC, datetime

            session.turns.append(ChatTurn(role="user", content="legacy plaintext", created_at=datetime.now(UTC)))
            db.commit()

        store.append_turn(session_id, "assistant", "new reply, must be encrypted")

        raw_values = _raw_content_column(sqlite_session_local, session_id)
        assert raw_values[0] == "legacy plaintext"  # unchanged, still readable
        assert raw_values[1].startswith("enc1:")  # new write is encrypted, never plaintext

        history = store.get_history(session_id)
        assert [t["content"] for t in history] == ["legacy plaintext", "new reply, must be encrypted"]


class TestNormalFlowStillWorksWithEncryptionEnabled:
    def test_create_append_read_delete_flow_unaffected(self, sqlite_session_local, encryption_key):
        """The existing session lifecycle (unchanged by this integration)
        still works end-to-end with encryption enabled."""
        store = PostgresSessionStore()
        session_id = store.create_session(tenant_id=5)
        assert store.session_exists(session_id)

        store.append_turn(session_id, "user", "hello")
        store.append_turn(session_id, "assistant", "hi there")
        assert len(store.get_history(session_id)) == 2

        assert store.delete_session(session_id) is True
        assert store.session_exists(session_id) is False
