"""Alembic migration environment.

Reads the DATABASE_URL from the app's own Settings (app.core.config) so
migrations always target the same database the app connects to — the
sqlalchemy.url in alembic.ini is left blank on purpose and overridden
here.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make backend/ importable so `from app...` works regardless of CWD.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.core.database import Base  # noqa: E402
from app.models import db_models  # noqa: E402,F401  register models on Base

config = context.config

# When migrations run inside the app process (app/core/database.py's
# run_migrations, on every startup), the alembic.ini logging config must NOT
# be applied — fileConfig would clobber the app's already-configured
# structured (JSON) logging. The CLI path (`alembic upgrade head`) sets no
# attribute, so it keeps the ini's console logging exactly as before.
if config.config_file_name is not None and not config.attributes.get("skip_logging_setup"):
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout, no DB)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to the DB and execute)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
