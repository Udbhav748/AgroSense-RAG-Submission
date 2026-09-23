import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import (
    admin,
    approvals,
    auth,
    documents,
    health,
    query,
)
from app.api.v1.routes import (
    metrics as metrics_routes,
)
from app.core.config import settings
from app.core.database import db_enabled, init_db
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.core.metrics import RequestTimer, get_metrics
from app.core.request_context import request_id_var
from app.core.security import security_headers_middleware
from app.services import s3_sync_service
from app.services.demo_seed_service import seed_if_empty
from app.services.tenant_service import seed_keys_from_settings
from app.services.usage_service import record_usage

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # On the Lambda deployment (Settings.s3_sync_enabled), restore whatever
    # was last synced before anything else touches local disk — a no-op
    # everywhere else. Must run before query.get_vector_store() below,
    # which triggers a synchronous FAISSVectorStore().load() off whatever
    # is on disk at that point.
    s3_sync_service.download_all()

    # Bring the PostgreSQL schema up-to-date and mirror env API keys into
    # the DB (both no-ops when DATABASE_URL is unset). Then seed the vector
    # store — no-op whenever the store already has vectors (local dev with
    # a persisted docker-compose volume, restored from S3 sync above, or
    # any platform with real persistent storage) — only fires on a
    # genuinely empty store, e.g. Render's free tier or a fresh Lambda
    # environment with nothing synced yet. See demo_seed_service.py.
    init_db()
    seed_keys_from_settings()
    await seed_if_empty(query.get_vector_store())

    # Pre-warm embedding model in background thread so first user queries are instantaneous
    import threading

    from app.services.embedding_service import get_embedding_model
    threading.Thread(target=get_embedding_model, daemon=True).start()

    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Apply the security-headers middleware (core/security.py) to every
    response — one more Starlette-style wrapper because main.py already
    uses the @app.middleware("http") decorator shape for its other
    cross-cutting concerns, so registering this the same way keeps the
    file's middleware section consistent."""
    return await security_headers_middleware(request, call_next)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Generate (or propagate) an X-Request-ID for every request.

    Stored in request_id_var for the duration of the request so every log
    line emitted while handling it — including from error handlers —
    carries the same id automatically (see JSONFormatter in core/logging.py).
    Also echoed back as a response header so callers can correlate their
    own logs against the server's.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def usage_log_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Record one usage_log row per request when PostgreSQL is configured.

    Registered after request_id_middleware, so it wraps it: when the
    finally block runs, request_id_var is set and the auth dependency has
    populated request.state.client_name/tenant_id for authenticated routes
    (absent for unauthenticated/failed requests — tenant_id stays null).
    Best-effort: a DB failure drops the log line, never the response.
    """
    if not db_enabled():
        return await call_next(request)

    start = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        # call_next can raise (an unhandled exception escapes the error
        # handlers); in that case there's no response yet — skip the usage
        # log line rather than reading an unbound `response` and masking
        # the original exception with a NameError. status_code defaults to
        # 500 for the failed-request accounting, matching metrics_middleware.
        record_usage(
            request_id=request_id_var.get() or request.headers.get("X-Request-ID") or "",
            tenant_id=getattr(request.state, "tenant_id", None),
            client_name=getattr(request.state, "client_name", None),
            path=request.url.path,
            method=request.method,
            status_code=response.status_code if response is not None else 500,
            latency_ms=round(latency_ms, 2),
        )
    # Reachable only when call_next above didn't raise (an exception would
    # have propagated out of the finally block instead) — response is
    # always set by then.
    assert response is not None
    return response


@app.middleware("http")
async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Record one http_requests_total + latency histogram per request.

    Lives in-app so a Prometheus-style scraper can pull live metrics from
    GET /metrics with nothing but this process running — the online
    counterpart to the offline log rollups (monitoring/log_aggregate.py).
    The error-taxonomy breakdown is recorded separately by
    error_handlers.py's record_error(); this middleware covers every
    response, healthy or not.

    Exceptions from call_next (e.g. a client disconnecting mid-stream) are
    recorded as a 500 so the request is never invisible to the scraper —
    the exception still propagates, as it would without this middleware.
    """
    timer = RequestTimer(get_metrics(), request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        timer.finish(500)
        raise
    timer.finish(response.status_code)
    return response


register_exception_handlers(app)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(admin.router)
app.include_router(approvals.router)
app.include_router(metrics_routes.router)
