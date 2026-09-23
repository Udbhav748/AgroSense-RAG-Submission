"""SQLAlchemy engine/session wiring for PostgreSQL persistence.

Optional by design: if Settings.database_url is empty, `engine` and
`SessionLocal` are None and `db_enabled()` returns False. Every consumer
(session store, document repository, auth) checks db_enabled() and falls
back to the legacy in-memory/file behavior, so an ephemeral deployment
that can't host a real Postgres keeps working exactly as before. See
app/core/config.py's database_url docstring.
"""

import logging
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models (see app/models/db_models.py)."""


# No engine/session when the DB is disabled — every caller guards on
# db_enabled() before touching these. Models imported against this module
# still work without a live connection because they only need the
# declarative metadata, not an engine.
engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def _probe_tcp(host: str, port: int, timeout_sec: float = 1.0) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except Exception:
        return False


def _setup_sqlite_engine() -> tuple[Engine, sessionmaker[Session]]:
    backend_dir = Path(__file__).resolve().parents[2]
    sqlite_path = backend_dir / "db.sqlite3"
    sqlite_url = f"sqlite:///{sqlite_path.as_posix()}"
    sq_engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
    )
    sq_session = sessionmaker(bind=sq_engine, autocommit=False, autoflush=False)
    return sq_engine, sq_session


if settings.database_url:
    if settings.database_url.startswith("sqlite"):
        try:
            engine = create_engine(
                settings.database_url,
                connect_args={"check_same_thread": False},
            )
            SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        except Exception:
            engine, SessionLocal = _setup_sqlite_engine()
    elif settings.database_url.startswith("postgresql"):
        from urllib.parse import urlparse

        parsed = urlparse(settings.database_url)
        pg_host = parsed.hostname or "localhost"
        pg_port = parsed.port or 5432
        if _probe_tcp(pg_host, pg_port, timeout_sec=0.8):
            try:
                engine = create_engine(
                    settings.database_url,
                    pool_pre_ping=True,
                    connect_args={"connect_timeout": settings.database_connect_timeout_seconds},
                )
                SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
            except Exception:
                engine, SessionLocal = _setup_sqlite_engine()
        else:
            logger.info(
                "db_postgres_offline_using_sqlite",
                extra={
                    "extra_fields": {
                        "host": f"{pg_host}:{pg_port}",
                        "hint": "PostgreSQL is offline. Seamlessly utilizing local SQLite persistence.",
                    }
                },
            )
            engine, SessionLocal = _setup_sqlite_engine()
    else:
        engine, SessionLocal = _setup_sqlite_engine()
else:
    engine, SessionLocal = _setup_sqlite_engine()


def db_enabled() -> bool:
    """True if a DATABASE_URL is configured and we can serve sessions."""
    return engine is not None and SessionLocal is not None


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped session (only valid when
    db_enabled(); routes that may run with the DB disabled must guard
    before depending on this)."""
    # Checked directly on SessionLocal (rather than via db_enabled()) so
    # mypy can narrow it from `sessionmaker[Session] | None` in this same
    # scope — engine/SessionLocal are always set together (see the
    # `if settings.database_url:` block above), so this is equivalent to
    # db_enabled() being False.
    if SessionLocal is None:
        raise RuntimeError("Database not configured (DATABASE_URL is empty)")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations() -> None:
    """Apply any pending Alembic migrations to the configured database.

    create_all() only creates tables that don't exist yet — it can never run
    an ALTER TABLE on an existing one and has zero awareness of Alembic's
    version history. A written migration that never reaches the live database
    (the documents.collection incident) breaks every query on that table with
    a 500, and it will recur for every future migration unless they're applied
    automatically. This runs `alembic upgrade head` programmatically via
    Alembic's API, so startup itself applies whatever is pending.

    Idempotent: `upgrade head` on an already-current database is a no-op.
    """
    from alembic import command
    from alembic.config import Config

    backend_dir = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend_dir / "alembic.ini"))
    # Make the script location absolute so it resolves regardless of CWD.
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    # env.py calls fileConfig(config.config_file_name) which would clobber
    # this process's configured structured logging; it checks the
    # `skip_logging_setup` attribute first (see alembic/env.py).
    cfg.attributes["skip_logging_setup"] = True
    command.upgrade(cfg, "head")


def init_db() -> None:
    """Bring the schema up to date on every startup, then create tables.

    If PostgreSQL is unreachable, seamlessly activates local SQLite (db.sqlite3)
    so authentication, user accounts, and sessions work out-of-the-box with $0 cost
    and zero external database requirements.
    """
    global engine, SessionLocal
    import app.models.db_models  # noqa: F401  ensure models are registered on Base

    if engine is None or SessionLocal is None or engine.dialect.name == "sqlite":
        if engine is None or SessionLocal is None:
            engine, SessionLocal = _setup_sqlite_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("db_ready_sqlite", extra={"extra_fields": {"database": str(engine.url)}})
        return

    try:
        run_migrations()
    except OperationalError as exc:
        host = _database_host_for_log()
        logger.warning(
            "db_postgres_unreachable_switching_to_sqlite",
            extra={
                "extra_fields": {
                    "host": host,
                    "hint": f"Postgres unreachable at {host}. Seamlessly operating with local SQLite database (db.sqlite3).",
                    "error": str(exc),
                }
            },
        )
        engine, SessionLocal = _setup_sqlite_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("db_ready_sqlite", extra={"extra_fields": {"database": str(engine.url)}})
        return
    except Exception as exc:
        if _has_alembic_history():
            # Not a no-history fallback: a real migration failure on a DB
            # Alembic already manages. Don't paper over it with create_all.
            logger.error(
                "db_migrations_failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise
        logger.warning(
            "db_no_alembic_history_using_create_all",
            extra={"extra_fields": {"reason": str(exc)}},
        )
        # engine is guaranteed non-None here: init_db() returned early above
        # unless db_enabled() was True, and engine/SessionLocal are only ever
        # set together. Narrowed explicitly (rather than relied on across the
        # try/except) since mypy doesn't track that invariant through the
        # intervening calls.
        if engine is None:
            raise RuntimeError(
                "Database engine not initialized despite db_enabled() check"
            ) from exc
        Base.metadata.create_all(bind=engine)
        _stamp_migrations_at_head()
    logger.info(
        "db_ready", extra={"extra_fields": {"url_scheme": settings.database_url.split("://")[0]}}
    )


def _stamp_migrations_at_head() -> None:
    """Record the DB as being at migration head (alembic_version table).

    Called after the create_all() fallback so a pre-Alembic database enters
    Alembic's version history: without this, every future startup would take
    the same create_all fallback and the next real migration would be skipped
    silently forever.
    """
    from alembic import command
    from alembic.config import Config

    backend_dir = Path(__file__).resolve().parents[2]
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    cfg.attributes["skip_logging_setup"] = True
    command.stamp(cfg, "head")


def _has_alembic_history() -> bool:
    """True if the target DB has ever been through Alembic (an
    alembic_version table exists). A fresh DB that only ever used create_all()
    has no such table — that's the no-history fallback case in init_db."""
    # Only called from init_db()'s except block, which is only reached after
    # db_enabled() confirmed engine is set — but that invariant isn't visible
    # to mypy across the function boundary, so it's re-checked here.
    if engine is None:
        raise RuntimeError("Database engine not initialized despite db_enabled() check")
    return bool(inspect(engine).has_table("alembic_version"))


def _database_host_for_log() -> str:
    """Best-effort host[:port] from the DATABASE_URL, for the unreachable-DB
    error line. Never let logging itself raise on a malformed URL."""
    try:
        return settings.database_url.split("@")[-1].split("/")[0]
    except Exception:
        return settings.database_url or "<unknown>"
