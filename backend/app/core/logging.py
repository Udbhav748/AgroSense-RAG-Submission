"""Structured (JSON) logging configuration for the application.

Call configure_logging() once at startup. Elsewhere, use
logging.getLogger(__name__) and pass extra={"extra_fields": {...}} to attach
structured context to a log line.
"""

import json
import logging
import sys
from datetime import UTC, datetime

from app.core.config import settings
from app.core.request_context import request_id_var


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Set by the request-ID middleware (see main.py) for the duration
        # of the request; every log line emitted while handling it picks
        # this up automatically, with no call site passing it explicitly.
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id

        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if settings.debug else logging.INFO)
