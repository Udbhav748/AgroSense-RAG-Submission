"""Shared pytest helpers for the backend test suite.

SCHEMA_COMPLIANCE is a tiny shared counter: assert_matches_schema() bumps
it every time a test validates a response body against one of the app's
Pydantic response models, and pytest_terminal_summary prints the
resulting compliance rate in the final test report.
"""

import os

# Force DB-disabled for the whole test session, regardless of whatever
# backend/.env has DATABASE_URL set to for live/manual use. Every test in
# this suite that needs DB-backed behavior (tenant/role resolution,
# session listing, ownership checks, ...) already mocks the specific
# function it needs (resolve_tenant, get_document_owner, etc.) rather
# than hitting a real database — that's this suite's one, consistent
# convention, not an exception. A real DATABASE_URL in the ambient
# environment has broken this twice already (once from a stray local
# Postgres/SQLite value, once from that value silently making individual
# user login "work" in tests that assumed it couldn't) — this line makes
# that class of bug structurally impossible instead of trusting everyone
# sharing this environment to remember to unset it by hand. Must run
# before any `app.*` import anywhere (conftest.py is always the first
# thing pytest loads in this directory, before test collection), since
# Settings() and database.py's engine are both built once at import time.
os.environ["DATABASE_URL"] = ""
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("API_KEY", "test-secret-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-for-testing-only-1234567890")
# Explicit, not inherited from backend/.env: Settings._reject_weak_secrets_in_production
# (app/core/config.py) refuses to construct with placeholder-looking
# secrets when debug=False. The test suite intentionally uses short,
# obviously-fake secrets above, so it must declare itself non-production
# on its own -- relying on the ambient .env's DEBUG=true would make the
# whole suite fail closed (breaking on Settings() import, before any test
# even runs) the moment .env is absent or that line changes, e.g. in a
# clean CI checkout.
os.environ.setdefault("DEBUG", "true")

import pytest

SCHEMA_COMPLIANCE = {"total": 0, "passed": 0}


@pytest.fixture(autouse=True)
def _reset_semantic_cache():
    """Clear the global semantic query cache before each test."""
    from app.services.cache_service import cache_service

    cache_service.invalidate()
    yield
    cache_service.invalidate()


@pytest.fixture(autouse=True)
def _reset_rate_limit_store():
    """Clear the global per-client rate-limit buckets before each test.

    auth.py keeps its sliding-window counters in a module-global store
    that is never reset during a process lifetime. TestSecurity's own
    low_limit fixture clears it, but the rest of the suite makes far more
    than the default 60 authenticated requests per client per minute —
    without a reset here, whichever test happens to run once that bucket
    fills gets an unrelated 401. Clearing before every test isolates the
    rate limiter to whatever that one test actually does.
    """
    from app.core.auth import _rate_limit_store

    _rate_limit_store.clear()
    yield
    _rate_limit_store.clear()


def assert_matches_schema(model_cls, payload: dict):
    """Validate payload against a Pydantic response model.

    Raises (failing the calling test) if payload doesn't conform. Only
    counts toward SCHEMA_COMPLIANCE once validation has actually run.
    """
    SCHEMA_COMPLIANCE["total"] += 1
    model_cls.model_validate(payload)
    SCHEMA_COMPLIANCE["passed"] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    total = SCHEMA_COMPLIANCE["total"]
    passed = SCHEMA_COMPLIANCE["passed"]
    rate = (passed / total * 100) if total else 0.0
    terminalreporter.write_sep("=", f"Schema Compliance Rate: {passed}/{total} ({rate:.1f}%)")
