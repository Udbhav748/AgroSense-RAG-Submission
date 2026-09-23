"""Registers global FastAPI exception handlers.

Domain errors (AppError and subclasses) are translated into their mapped
HTTP status code. Anything else is logged and returned as a generic 500,
so unexpected exceptions never leak internals to the client.

Response shape includes structured error code for frontend handling:
{
  "detail": "Human-readable message",
  "error_code": "STABLE_ERROR_CODE",
  "taxonomy_category": "auth|validation|tool|llm|rate_limit|..."
}
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.metrics import get_metrics

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        get_metrics().record_error(
            taxonomy_category=exc.taxonomy_category, status_code=exc.status_code
        )
        logger.warning(
            "request_failed",
            extra={
                "extra_fields": {
                    "path": request.url.path,
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "taxonomy_category": exc.taxonomy_category,
                }
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "error_code": exc.error_code,
                "taxonomy_category": exc.taxonomy_category,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        get_metrics().record_error(taxonomy_category="internal", status_code=500)
        logger.error(
            "unhandled_exception",
            exc_info=exc,
            extra={"extra_fields": {"path": request.url.path}},
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "error_code": "INTERNAL_ERROR",
                "taxonomy_category": "internal",
            },
        )
