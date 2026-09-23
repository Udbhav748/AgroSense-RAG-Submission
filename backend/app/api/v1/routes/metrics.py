"""Live Prometheus metrics endpoint.

GET /metrics serves the in-process registry (app/core/metrics.py) in the
Prometheus text exposition format, for a real scraper to pull. The one
offline rollup the app already has (monitoring/log_aggregate.py) reads
captured logs after the fact; this endpoint is the fully live counterpart
— same underlying events (request durations, tool invocations, LLM
token/cost, loop caps, retrieval timeouts, errors), scraped at any moment.

Access control mirrors /health but one step tighter: unauthenticated by
default (any status_code is a metric; metrics carry no payload data, and
the route cost is trivial), but a METRICS_BEARER_TOKEN (when set) is
required via Authorization: Bearer, so a publicly-exposed deployment can
keep its metrics private without giving scrapers a full API key. The
token is compared via hmac.compare_digest, never logged, and a mismatch
returns 401 — indistinguishable from an unset token to a scraper.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, Response, status

from app.core.config import settings
from app.core.metrics import get_metrics
from app.services.metrics import export_prometheus_metrics

router = APIRouter()


@router.get(
    "/metrics",
    tags=["Health"],
    summary="Live metrics in Prometheus text exposition format",
    responses={401: {"description": "Invalid or missing bearer token"}},
)
def metrics_endpoint(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> Response:
    """Serve the in-process metric registry as Prometheus text format."""
    if settings.metrics_bearer_token:
        expected = f"Bearer {settings.metrics_bearer_token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            return Response(
                content='{"detail": "Missing or invalid bearer token", '
                '"error_code": "METRICS_UNAUTHORIZED", "taxonomy_category": "auth"}',
                status_code=status.HTTP_401_UNAUTHORIZED,
                media_type="application/json",
            )

    core_body = get_metrics().render()
    rag_body = export_prometheus_metrics()
    body = core_body + rag_body
    # text/plain with the Prometheus version param; scrapers accept either
    # this or OpenMetrics (a separate Accept negotiation we don't need).
    return Response(
        content=body,
        status_code=status.HTTP_200_OK,
        media_type="text/plain; version=0.0.4",
    )
