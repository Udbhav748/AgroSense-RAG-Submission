"""Per-node tracing and the canonical workflow lifecycle event names.

Nodes call `emit_node_trace` around their delegate call so every execution
produces one structured log line with trace_id/request_id/node/status/
latency/error — reusing the project's existing structured logging
(logger.info/warning with `extra={"extra_fields": ...}`), not a second,
incompatible log format. The lifecycle event name constants are the single
source of truth for what `/chat/stream`'s SSE `trace` events and internal
node tracing both emit, so the two can't drift apart.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.metrics import get_metrics

# Canonical lifecycle event names. Emitted by nodes/graph.py and mirrored
# into SSE `trace` events by the streaming endpoint — one shared vocabulary.
WORKFLOW_STARTED = "workflow_started"
REQUEST_VALIDATED = "request_validated"
PLANNER_STARTED = "planner_started"
PLANNER_COMPLETED = "planner_completed"
RETRIEVAL_STARTED = "retrieval_started"
RETRIEVAL_COMPLETED = "retrieval_completed"
RERANKING_STARTED = "reranking_started"
RERANKING_COMPLETED = "reranking_completed"
RETRIEVAL_GRADED = "retrieval_graded"
VISION_STARTED = "vision_started"
VISION_COMPLETED = "vision_completed"
WEB_SEARCH_STARTED = "web_search_started"
WEB_SEARCH_COMPLETED = "web_search_completed"
GENERATION_STARTED = "generation_started"
GENERATION_COMPLETED = "generation_completed"
VALIDATION_STARTED = "validation_started"
VALIDATION_FAILED = "validation_failed"
REFLECTION_STARTED = "reflection_started"
APPROVAL_REQUIRED = "approval_required"
APPROVAL_RECEIVED = "approval_received"
FINALIZED = "finalized"
WORKFLOW_COMPLETED = "workflow_completed"
WORKFLOW_FAILED = "workflow_failed"

logger = logging.getLogger(__name__)


@dataclass
class NodeTrace:
    """One node execution's trace record — matches PDF section 10's shape."""

    trace_id: str | None
    request_id: str | None
    node: str
    status: str  # "success" | "failure"
    latency_ms: float
    error_type: str | None = None
    retry_count: int = 0

    def as_extra_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "node": self.node,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "retry_count": self.retry_count,
        }
        if self.error_type:
            fields["error_type"] = self.error_type
        return fields


def emit_node_trace(
    *,
    trace_id: str | None,
    request_id: str | None,
    node: str,
    status: str,
    latency_ms: float,
    error_type: str | None = None,
    retry_count: int = 0,
    extra: dict[str, object] | None = None,
) -> NodeTrace:
    """Log one structured `agent_node_trace` line and return the record so
    callers (graph.py) can also feed it into node_timings/metrics without
    recomputing anything."""
    trace = NodeTrace(
        trace_id=trace_id,
        request_id=request_id,
        node=node,
        status=status,
        latency_ms=latency_ms,
        error_type=error_type,
        retry_count=retry_count,
    )
    fields = trace.as_extra_fields()
    if extra:
        fields.update(extra)
    log_fn = logger.info if status == "success" else logger.warning
    log_fn("agent_node_trace", extra={"extra_fields": fields})
    get_metrics().record_agent_node_execution(
        node=node, status=status, latency_seconds=latency_ms / 1000.0, error_type=error_type
    )
    return trace


class NodeTimer:
    """Small context manager so nodes don't hand-roll perf_counter math."""

    def __init__(self) -> None:
        self._start = 0.0
        self.latency_ms = 0.0

    def __enter__(self) -> NodeTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.latency_ms = (time.perf_counter() - self._start) * 1000
