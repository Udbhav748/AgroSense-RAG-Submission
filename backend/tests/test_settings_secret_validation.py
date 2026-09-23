"""Gap-closure: secrets management was documented but not code-enforced --
nothing previously stopped a DEBUG=false (production) run from using the
exact placeholder values backend/.env.example ships with. These tests
pin the new fail-closed behavior in
Settings._reject_weak_secrets_in_production (app/core/config.py).
"""

import pytest

from app.core.config import Settings


def _base_kwargs(**overrides):
    kwargs = {
        "gemini_api_key": "real-looking-gemini-key-abcdef123456",
        "api_key": "a-genuinely-strong-random-api-key-value",
    }
    kwargs.update(overrides)
    return kwargs


class TestDebugModeBypassesValidation:
    def test_debug_true_allows_placeholder_api_key(self):
        settings = Settings(**_base_kwargs(debug=True, api_key="change-me"))
        assert settings.api_key == "change-me"

    def test_debug_true_allows_short_jwt_secret(self):
        settings = Settings(**_base_kwargs(debug=True, jwt_secret_key="short"))
        assert settings.jwt_secret_key == "short"


class TestProductionModeRejectsWeakApiKey:
    def test_placeholder_api_key_raises(self):
        with pytest.raises(ValueError, match="API_KEY"):
            Settings(**_base_kwargs(debug=False, api_key="local-dev-api-key-change-me"))

    def test_short_api_key_raises(self):
        with pytest.raises(ValueError, match="API_KEY"):
            Settings(**_base_kwargs(debug=False, api_key="short-key"))

    def test_strong_api_key_passes(self):
        settings = Settings(**_base_kwargs(debug=False, api_key="a-genuinely-strong-random-api-key-value"))
        assert settings.debug is False


class TestProductionModeRejectsWeakApiKeysMap:
    def test_weak_entry_in_api_keys_json_raises(self):
        with pytest.raises(ValueError, match="API_KEYS entry"):
            Settings(**_base_kwargs(debug=False, api_keys='{"client-a": "short"}'))

    def test_strong_entries_in_api_keys_json_pass(self):
        settings = Settings(
            **_base_kwargs(debug=False, api_keys='{"client-a": "a-genuinely-strong-random-key-value-1"}')
        )
        assert settings.api_keys

    def test_weak_single_api_key_is_ignored_when_api_keys_map_supersedes_it(self):
        """API_KEYS (plural) supersedes the single API_KEY entirely once
        set (see api_key_hash_map) -- a leftover weak single api_key in
        the environment is inert, not a real exposure, and must not fail
        a run that's actually using strong per-client keys. Regression
        test for a real bug caught during this feature's own regression
        run: this exact scenario crashed a real subprocess
        (test_load_concurrency_eval.py's uvicorn smoke test) before the
        fix."""
        settings = Settings(
            **_base_kwargs(
                debug=False,
                api_key="short",  # weak, but irrelevant once api_keys is set
                api_keys='{"client-a": "a-genuinely-strong-random-key-value-1"}',
            )
        )
        assert settings.api_keys


class TestProductionModeRejectsWeakJwtSecret:
    def test_short_jwt_secret_raises(self):
        with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
            Settings(**_base_kwargs(debug=False, jwt_secret_key="too-short"))

    def test_placeholder_jwt_secret_raises(self):
        with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
            Settings(**_base_kwargs(debug=False, jwt_secret_key="changeme"))

    def test_empty_jwt_secret_is_allowed(self):
        """jwt_secret_key is a genuinely optional feature (individual-user
        login) -- an empty value in production just means that feature is
        unavailable (create_access_token raises its own clear error at
        use-time), not a misconfiguration to fail startup over."""
        settings = Settings(**_base_kwargs(debug=False, jwt_secret_key=""))
        assert settings.jwt_secret_key == ""

    def test_strong_jwt_secret_passes(self):
        settings = Settings(
            **_base_kwargs(debug=False, jwt_secret_key="a-genuinely-strong-random-jwt-signing-secret-value")
        )
        assert settings.jwt_secret_key


class TestProductionModeRejectsDevDatabasePassword:
    def test_dev_password_in_database_url_raises(self):
        with pytest.raises(ValueError, match="DATABASE_URL"):
            Settings(
                **_base_kwargs(
                    debug=False,
                    database_url="postgresql+psycopg2://insightai:insightai-dev-password@localhost:5432/insightai",
                )
            )

    def test_real_password_in_database_url_passes(self):
        settings = Settings(
            **_base_kwargs(
                debug=False,
                database_url="postgresql+psycopg2://insightai:a-real-production-password@db.example.com:5432/insightai",
            )
        )
        assert settings.database_url


class TestMultipleProblemsReportedTogether:
    def test_all_problems_listed_in_one_error(self):
        with pytest.raises(ValueError) as exc_info:
            Settings(**_base_kwargs(debug=False, api_key="short", jwt_secret_key="weak"))
        message = str(exc_info.value)
        assert "API_KEY" in message
        assert "JWT_SECRET_KEY" in message
