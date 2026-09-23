"""Unit tests for the DB-independent pieces of user_service.py: password
hashing, JWT issuance/verification, per-email login lockout, and JWT
revocation-on-logout. The DB-touching functions (create_user,
authenticate_user, get_user_by_id) are exercised at the route level
instead (tests/test_auth_routes.py), matching this repo's existing
convention of mocking at the service-function boundary rather than
running against a real database (see test_main.py's tenant-ownership
tests for the same pattern). The lockout tests below are an exception
that proves the rule: _is_locked_out is checked in authenticate_user
*before* _session() is ever called, so it's exercised directly, with
DATABASE_URL forced off exactly like the rest of this suite (see
conftest.py) — proving the lockout fires even with no DB configured is
the point, not a limitation of the test. The revocation tests fake the
one DB round-trip decode_access_token/revoke_all_tokens make, the same
way test_main.py fakes the vector store rather than standing up FAISS.
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import settings
from app.core.exceptions import AccountLockedError, AuthConfigurationError
from app.services import user_service
from app.services.user_service import (
    _hash_password,
    _is_locked_out,
    _verify_password,
    create_access_token,
    decode_access_token,
    revoke_all_tokens,
)


class TestPasswordHashing:
    def test_correct_password_verifies(self):
        password_hash = _hash_password("correct-horse-battery-staple")
        assert _verify_password("correct-horse-battery-staple", password_hash) is True

    def test_wrong_password_does_not_verify(self):
        password_hash = _hash_password("correct-horse-battery-staple")
        assert _verify_password("wrong-password", password_hash) is False

    def test_same_password_hashes_differently_each_time(self):
        # bcrypt salts per-call — two hashes of the same password must
        # never be equal, or the salt isn't doing its job.
        first = _hash_password("same-password")
        second = _hash_password("same-password")
        assert first != second
        assert _verify_password("same-password", first) is True
        assert _verify_password("same-password", second) is True


class TestJWT:
    @pytest.fixture(autouse=True)
    def _jwt_secret(self, monkeypatch):
        monkeypatch.setattr(settings, "jwt_secret_key", "test-secret-key")
        monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
        monkeypatch.setattr(settings, "jwt_expiry_minutes", 1440)

    def test_round_trip(self):
        token = create_access_token(user_id=42, email="a@example.com")
        assert decode_access_token(token) == 42

    def test_decode_with_wrong_secret_returns_none(self, monkeypatch):
        token = create_access_token(user_id=42, email="a@example.com")
        monkeypatch.setattr(settings, "jwt_secret_key", "a-different-secret")
        assert decode_access_token(token) is None

    def test_decode_garbage_token_returns_none(self):
        assert decode_access_token("not-a-real-jwt") is None

    def test_decode_expired_token_returns_none(self, monkeypatch):
        monkeypatch.setattr(settings, "jwt_expiry_minutes", -1)  # already expired
        token = create_access_token(user_id=42, email="a@example.com")
        assert decode_access_token(token) is None

    def test_create_without_secret_raises_auth_configuration_error(self, monkeypatch):
        monkeypatch.setattr(settings, "jwt_secret_key", "")
        with pytest.raises(AuthConfigurationError):
            create_access_token(user_id=42, email="a@example.com")

    def test_decode_without_secret_returns_none_not_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "jwt_secret_key", "")
        assert decode_access_token("anything") is None


class TestLoginLockout:
    @pytest.fixture(autouse=True)
    def _clean_store(self):
        user_service._failed_login_store.clear()
        yield
        user_service._failed_login_store.clear()

    @pytest.fixture(autouse=True)
    def _lockout_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "login_lockout_max_attempts", 3)
        monkeypatch.setattr(settings, "login_lockout_window_seconds", 900)

    def test_not_locked_before_max_attempts(self):
        for _ in range(2):
            user_service._record_failed_attempt("a@example.com")
        assert _is_locked_out("a@example.com") is False

    def test_locked_at_max_attempts(self):
        for _ in range(3):
            user_service._record_failed_attempt("a@example.com")
        assert _is_locked_out("a@example.com") is True

    def test_lockout_is_per_email(self):
        for _ in range(3):
            user_service._record_failed_attempt("a@example.com")
        assert _is_locked_out("b@example.com") is False

    def test_clear_resets_lockout(self):
        for _ in range(3):
            user_service._record_failed_attempt("a@example.com")
        user_service._clear_failed_attempts("a@example.com")
        assert _is_locked_out("a@example.com") is False

    def test_failures_outside_window_are_not_counted(self, monkeypatch):
        # Collapse the window to 0s: every failure is immediately "in the
        # past" relative to itself, so is_locked's own pruning drops them
        # all on the very next check — proves the sliding window actually
        # ages entries out, not just that clearing works.
        monkeypatch.setattr(settings, "login_lockout_window_seconds", 0)
        for _ in range(5):
            user_service._record_failed_attempt("a@example.com")
        assert _is_locked_out("a@example.com") is False

    def test_authenticate_user_raises_locked_before_touching_db(self):
        # DATABASE_URL is forced off for the whole suite (conftest.py) —
        # authenticate_user would normally raise DatabaseNotConfiguredError
        # the moment it calls _session(). Getting AccountLockedError
        # instead proves the lockout check runs first.
        for _ in range(3):
            user_service._record_failed_attempt("locked@example.com")
        with pytest.raises(AccountLockedError):
            user_service.authenticate_user("locked@example.com", "whatever")


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeSession:
    """Stands in for a SQLAlchemy Session across `with _session() as db`,
    returning a fixed User row for every query — enough to exercise
    decode_access_token/revoke_all_tokens' one round-trip without a real
    database, mirroring test_main.py's FakeVectorStore approach."""

    def __init__(self, user):
        self._user = user
        self.committed = False

    def query(self, model):
        return _FakeQuery(self._user)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def commit(self):
        self.committed = True


class _FakeUser:
    def __init__(self, user_id, tokens_revoked_after=None):
        self.id = user_id
        self.tokens_revoked_after = tokens_revoked_after


class TestTokenRevocation:
    @pytest.fixture(autouse=True)
    def _jwt_secret(self, monkeypatch):
        monkeypatch.setattr(settings, "jwt_secret_key", "test-secret-key")
        monkeypatch.setattr(settings, "jwt_algorithm", "HS256")
        monkeypatch.setattr(settings, "jwt_expiry_minutes", 1440)

    def test_decode_ignores_revocation_when_db_disabled(self, monkeypatch):
        # db_enabled() is False for the whole suite already (conftest.py)
        # — this asserts decode_access_token doesn't even try the check,
        # so a token stays valid exactly as it did before this feature.
        monkeypatch.setattr(user_service, "db_enabled", lambda: False)
        token = create_access_token(user_id=42, email="a@example.com")
        assert decode_access_token(token) == 42

    def test_decode_valid_when_never_revoked(self, monkeypatch):
        monkeypatch.setattr(user_service, "db_enabled", lambda: True)
        fake_user = _FakeUser(42, tokens_revoked_after=None)
        monkeypatch.setattr(user_service, "_session", lambda: _FakeSession(fake_user))

        token = create_access_token(user_id=42, email="a@example.com")
        assert decode_access_token(token) == 42

    def test_decode_rejects_token_issued_before_revocation(self, monkeypatch):
        monkeypatch.setattr(user_service, "db_enabled", lambda: True)
        token = create_access_token(user_id=42, email="a@example.com")
        # Revoked one minute *after* the token above was issued.
        fake_user = _FakeUser(42, tokens_revoked_after=datetime.now(UTC) + timedelta(minutes=1))
        monkeypatch.setattr(user_service, "_session", lambda: _FakeSession(fake_user))

        assert decode_access_token(token) is None

    def test_decode_accepts_token_issued_after_revocation(self, monkeypatch):
        monkeypatch.setattr(user_service, "db_enabled", lambda: True)
        # Revoked one minute *before* a token freshly issued now.
        fake_user = _FakeUser(42, tokens_revoked_after=datetime.now(UTC) - timedelta(minutes=1))
        monkeypatch.setattr(user_service, "_session", lambda: _FakeSession(fake_user))

        token = create_access_token(user_id=42, email="a@example.com")
        assert decode_access_token(token) == 42

    def test_revoke_all_tokens_noop_when_db_disabled(self, monkeypatch):
        monkeypatch.setattr(user_service, "db_enabled", lambda: False)
        # Would raise DatabaseNotConfiguredError if this fell through to
        # _session() — must return cleanly instead.
        revoke_all_tokens(42)

    def test_revoke_all_tokens_sets_timestamp_and_commits(self, monkeypatch):
        monkeypatch.setattr(user_service, "db_enabled", lambda: True)
        fake_user = _FakeUser(42, tokens_revoked_after=None)
        fake_session = _FakeSession(fake_user)
        monkeypatch.setattr(user_service, "_session", lambda: fake_session)

        revoke_all_tokens(42)

        assert fake_user.tokens_revoked_after is not None
        assert fake_session.committed is True

    def test_revoke_all_tokens_noop_when_user_missing(self, monkeypatch):
        monkeypatch.setattr(user_service, "db_enabled", lambda: True)
        fake_session = _FakeSession(None)
        monkeypatch.setattr(user_service, "_session", lambda: fake_session)

        revoke_all_tokens(999)  # does not raise
        assert fake_session.committed is False
