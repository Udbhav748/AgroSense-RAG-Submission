"""Best-effort request usage logging to PostgreSQL.

The middleware in app/main.py calls record_usage() after each request; it
never blocks or fails the request — a DB failure just drops the log line
(with a log warning) and the response proceeds untouched.
"""

import logging

from app.core.database import SessionLocal, db_enabled
from app.models.db_models import UsageLog

logger = logging.getLogger(__name__)


def record_usage(
    *,
    request_id: str,
    tenant_id: int | None,
    client_name: str | None,
    path: str,
    method: str,
    status_code: int,
    latency_ms: float,
) -> None:
    if not db_enabled():
        return
    assert SessionLocal is not None  # db_enabled() guarantees this
    try:
        with SessionLocal() as db:
            db.add(
                UsageLog(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    client_name=client_name,
                    path=path,
                    method=method,
                    status_code=status_code,
                    latency_ms=latency_ms,
                )
            )
            db.commit()
    except Exception as exc:
        logger.warning("usage_log_write_failed", extra={"extra_fields": {"error": str(exc)}})
