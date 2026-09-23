"""Evaluation-only telemetry capture: extracts the real per-request node
execution sequence and LLM cost from the EXISTING structured log lines
(`agent_node_trace` from agent_graph/events.py::emit_node_trace,
`chat_query_handled` from rag_service.py::ChatService._respond) via a
temporary logging.Handler attached only for the duration of one
evaluation call.

This is deliberately NOT a production code change: no new field on
AgentState or ChatResponse, no new API surface. It reads exactly what the
app already logs in production (the same lines `metrics_report.py` parses
offline from a log file) — just captured in-process instead of from a
file, so eval runners can attribute per-case node sequences and cost
without waiting for a log-file round-trip. Attaching a log handler adds
no per-request overhead outside of an active `capture_telemetry()` block,
and is removed immediately after each call.

No sensitive content is captured: `agent_node_trace`'s fields are
node/status/latency/error_type/trace_id/request_id only (see
NodeTrace.as_extra_fields) and `chat_query_handled`'s are aggregate
counts (query_length, steps_taken, token counts, cost) — never raw query
text, answer text, or document content.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

_NODE_TRACE_LOGGER = "app.services.agent_graph.events"
_CHAT_HANDLED_LOGGER = "app.services.rag_service"


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@dataclass
class CapturedTelemetry:
    node_sequence: list[str] = field(default_factory=list)
    node_statuses: list[dict] = field(default_factory=list)
    chat_query_handled: dict | None = None

    @property
    def steps_taken(self) -> int | None:
        return self.chat_query_handled.get("steps_taken") if self.chat_query_handled else None

    @property
    def estimated_cost_usd(self) -> float | None:
        return self.chat_query_handled.get("estimated_cost_usd") if self.chat_query_handled else None

    @property
    def total_tokens(self) -> int | None:
        return self.chat_query_handled.get("total_tokens") if self.chat_query_handled else None

    @property
    def llm_calls(self) -> int | None:
        return self.chat_query_handled.get("llm_calls") if self.chat_query_handled else None


@contextmanager
def capture_telemetry() -> Iterator[_CaptureHandler]:
    """Attach a capture handler to the two loggers that already emit the
    structured lines this module reads, for the duration of the `with`
    block. Use `extract(handler)` afterward to get a CapturedTelemetry.
    """
    handler = _CaptureHandler()
    handler.setLevel(logging.INFO)
    loggers = [logging.getLogger(_NODE_TRACE_LOGGER), logging.getLogger(_CHAT_HANDLED_LOGGER)]
    previous_levels = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        for lg in loggers:
            lg.removeHandler(handler)
        for lg, level in previous_levels:
            lg.setLevel(level)


def extract(handler: _CaptureHandler) -> CapturedTelemetry:
    result = CapturedTelemetry()
    for record in handler.records:
        fields = getattr(record, "extra_fields", None)
        if not isinstance(fields, dict):
            continue
        if record.getMessage() == "agent_node_trace" and "node" in fields:
            result.node_sequence.append(fields["node"])
            result.node_statuses.append({"node": fields["node"], "status": fields.get("status")})
        elif record.getMessage() == "chat_query_handled":
            result.chat_query_handled = dict(fields)
    return result
