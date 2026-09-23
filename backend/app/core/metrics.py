"""Dependency-free, in-process Prometheus metrics registry.

Powers GET /metrics (app/api/v1/routes/metrics.py) in the standard
Prometheus text exposition format, so a real scraper — Prometheus,
Grafana Cloud, or a CloudWatch agent pointing at the endpoint — can pull
live metrics with no sidecar, no push gateway, and no new infrastructure
service. That keeps the "one process + one DB, free tier" deployment
story intact: metrics are collected exactly like /health is polled.

Instrumentation lives at the existing log-emit call sites (the same
places monitoring/log_aggregate.py and eval/metrics_report.py read offline),
so the live scraper and the offline log rollup agree by construction:

- HTTP requests: emitted by a middleware in app/main.py
  (http_requests_total, http_request_duration_seconds by method/path/status).
- Tools: app/services/tool_registry.py's @track_tool wrapper
  (tool_invocations_total, tool_invocation_duration_seconds).
- LLM generations: app/services/gemini_client.py / groq_client.py
  (llm_generations_total, llm_tokens_total, llm_cost_usd_total).
- Corrective loop cap: app/services/rag_service.py (loop_capped_total).
- Retrieval degradation: app/services/retrieval_service.py
  (retrieval_timeouts_total).
- Errors: app/core/error_handlers.py (errors_total by taxonomy_category).

Why not the `prometheus_client` package (or a hosted vendor) instead of
owning ~120 lines? This project's explicit monitoring philosophy is
"dependency-free stand-ins" (backend/monitoring/README.md, and the hard
constraints in docs/EXTERNAL_FEATURES_PLAN.md: zero ongoing cost, freely
deployable, minimal dependencies). The Prometheus wire format for
counters, gauges, and bucketed histograms is small and stable enough to
trust to a hand-rolled registry at this app's scale, and the numeric
semantics still match the Prometheus data model exactly, so nothing about
how a downstream scraper consumes it differs from the real library.

Concurrency: FastAPI executes sync routes on a threadpool while async
routes run on the event loop, so every registry mutation goes through a
single lock — the same threading story as FAISSVectorStore's lock.
Cardinality is bounded: counters/gauges key by (name, labels); HTTP paths
are normalized (UUIDs and numeric ids -> "{id}"); histogram buckets are a
fixed set. Nothing accumulates unbounded per-request state.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from typing import Any

# Fixed histogram buckets in seconds, chosen to span this app's realistic
# request range (sub-millisecond /health checks up to multi-tool, multi-
# second chat answers). Prometheus semantics: each bucket holds the
# cumulative count of observations <= its le (less-or-equal) upper bound,
# +Inf catches the tail, and p50/p95/p99 are derived by the scraper via
# histogram_quantile — matching how the offline rollup reports latency.
HISTOGRAM_BUCKETS_SECONDS: tuple[float, ...] = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


def _escape_label_value(value: Any) -> str:
    """Escape a Prometheus label value per the text exposition format."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _format_labels(labels: dict[str, Any] | None) -> str:
    """Format labels as {k="v",k2="v2"} or the empty string when absent."""
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def _label_key(labels: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Normalize a labels dict into a hashable, order-insensitive key."""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


# Normalizes /documents/550e8400-e29b-41d4-a716-446655440000 -> /documents/{id}
# so request counts and latency histograms stay low-cardinality instead of
# growing one series per document id.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def normalize_path(path: str) -> str:
    parts = []
    for segment in path.split("/"):
        if _UUID_RE.match(segment) or segment.isdigit():
            segment = "{id}"
        parts.append(segment)
    return "/".join(parts)


class _Histogram:
    """Fixed-bucket cumulative histogram per the Prometheus model."""

    __slots__ = ("cumulative", "sum", "count")

    def __init__(self) -> None:
        self.cumulative: list[float] = [0.0] * len(HISTOGRAM_BUCKETS_SECONDS)
        self.sum = 0.0
        self.count = 0.0

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1.0
        for i, bound in enumerate(HISTOGRAM_BUCKETS_SECONDS):
            if value <= bound:
                self.cumulative[i] += 1.0
        # +Inf bucket is rendered as count (see Metrics.render) — never
        # stored separately, since it's definitionally every observation.


class Metrics:
    """Thread-safe registry of counters, gauges, and histograms.

    render() serializes to the Prometheus text exposition format. reset()
    wipes all state — used by tests to isolate cases against the module
    singleton; never part of the request path.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(
            float
        )
        self._gauges: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(
            float
        )
        self._histograms: defaultdict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = (
            defaultdict(_Histogram)
        )
        self._started_at = time.time()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    # --- Recording -------------------------------------------------------

    def inc_counter(
        self, name: str, labels: dict[str, Any] | None = None, amount: float = 1.0
    ) -> None:
        with self._lock:
            self._counters[(name, _label_key(labels))] += amount

    def set_gauge(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._gauges[(name, _label_key(labels))] = value

    def observe_duration(
        self, name: str, value_seconds: float, labels: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._histograms[(name, _label_key(labels))].observe(value_seconds)

    # --- Convenience groups ---------------------------------------------

    def record_llm_generation(
        self,
        *,
        provider: str,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        """One LLM generation: counters for calls, tokens, and cost."""
        labels = {"provider": provider}
        if model:
            labels["model"] = model
        self.inc_counter("llm_generations_total", labels)
        self.inc_counter(
            "llm_tokens_total", {"provider": provider, "type": "prompt"}, prompt_tokens
        )
        self.inc_counter(
            "llm_tokens_total", {"provider": provider, "type": "completion"}, completion_tokens
        )
        self.inc_counter("llm_cost_usd_total", {"provider": provider}, estimated_cost_usd)

    def record_tool_invocation(self, *, tool: str, success: bool, latency_ms: float) -> None:
        """One @track_tool invocation: call count by outcome + latency hist."""
        self.inc_counter(
            "tool_invocations_total", {"tool": tool, "result": "success" if success else "error"}
        )
        self.observe_duration(
            "tool_invocation_duration_seconds", latency_ms / 1000.0, {"tool": tool}
        )

    def record_error(self, *, taxonomy_category: str, status_code: int) -> None:
        self.inc_counter(
            "errors_total",
            {"taxonomy_category": taxonomy_category, "status_code": str(status_code)},
        )

    def record_retrieval_grade(self, grade: str) -> None:
        """One retrieval grading outcome — the distribution of good/weak/
        insufficient grades the RAG pipeline dashboard's primary panel
        draws (see monitoring/grafana/dashboards/rag_pipeline.json)."""
        self.inc_counter("retrieval_grades_total", {"grade": grade})

    def record_web_search_fallback(self, *, stage: str) -> None:
        """One corrective-loop web-search fallback (stage: which path fired
        it — 'retrieval' when a weak grade pulled web context in before
        generation, 'reflection'/'research' for the retry paths). Feeds the
        web-search fallback rate panel in rag_pipeline.json."""
        self.inc_counter("web_search_fallbacks_total", {"stage": stage})

    def record_agent_handoff(self, *, frm: str, to: str) -> None:
        """One agent handoff (e.g. router -> research, retrieval_grader ->
        research). Feeds the agent-metrics dashboard's handoff breakdown."""
        self.inc_counter("agent_handoffs_total", {"from": frm, "to": to})

    # --- Explicit agent workflow (Phase 1) --------------------------------
    #
    # Instrumented from agent_graph/events.py::emit_node_trace (per-node)
    # and agent_graph/engine.py::CompiledGraph.run()/stream() (per-workflow-
    # run), so every graph execution — the production chat graph and the
    # older create_rag_agent_graph — gets this for free. Labels are
    # intentionally low-cardinality (node/status/action/error_type only —
    # never raw query text, user id, or document content), per the same
    # cardinality discipline as normalize_path() above.

    def record_agent_workflow_started(self) -> None:
        self.inc_counter("agent_workflow_started_total")

    def record_agent_workflow_completed(self, *, status: str) -> None:
        """status: 'completed' -> success counter, anything else (e.g.
        'failed') -> the failure counter. Never both for one run."""
        if status == "completed":
            self.inc_counter("agent_workflow_completed_total")
        else:
            self.inc_counter("agent_workflow_failed_total")

    def record_agent_workflow_duration(self, seconds: float) -> None:
        self.observe_duration("agent_workflow_duration_seconds", seconds)

    def record_agent_steps(self, count: int) -> None:
        """Total node executions in one workflow run — feeds the "Average
        Steps" evaluation metric alongside the offline eval harness."""
        self.inc_counter("agent_steps_total", amount=float(count))

    def record_agent_node_execution(
        self, *, node: str, status: str, latency_seconds: float, error_type: str | None = None
    ) -> None:
        """One node execution: status is 'success' or 'failure'."""
        self.inc_counter("agent_node_executions_total", {"node": node, "status": status})
        if status == "failure":
            failure_labels: dict[str, Any] = {"node": node}
            if error_type:
                failure_labels["error_type"] = error_type
            self.inc_counter("agent_node_failures_total", failure_labels)
        self.observe_duration("agent_node_latency_seconds", latency_seconds, {"node": node})

    def record_agent_reflection(self) -> None:
        self.inc_counter("agent_reflections_total")

    def record_agent_retry(self, *, node: str) -> None:
        self.inc_counter("agent_retries_total", {"node": node})

    def record_agent_loop_limit_hit(self) -> None:
        """A workflow run hit the graph's max_steps cap without reaching
        END — the Loop Rate evaluation metric's numerator."""
        self.inc_counter("agent_loop_limit_hits_total")

    def record_agent_approval(self, *, event: str, action: str) -> None:
        """event: 'required' | 'approved' | 'rejected'."""
        self.inc_counter(f"agent_approval_{event}_total", {"action": action})

    def agent_workflow_summary(self) -> dict[str, float | None]:
        """Aggregate the explicit-workflow counters/histograms above into
        the PDF's Workflow Completion Rate / Node Success Rate / Average
        Node Latency / Loop Rate / Average Steps — computed from whatever
        has actually been recorded in this registry since the last reset
        (backend/eval/run_eval.py resets before a run and reads this after;
        a live deployment reads it cumulative since process start via
        GET /metrics's raw counters). Never fabricated: a rate is None,
        not 0, when its denominator is zero (no workflows recorded yet).
        """
        with self._lock:
            completed = self._counters.get(("agent_workflow_completed_total", ()), 0.0)
            failed = self._counters.get(("agent_workflow_failed_total", ()), 0.0)
            total_workflows = completed + failed

            node_success = 0.0
            node_total = 0.0
            for (name, label_key), value in self._counters.items():
                if name != "agent_node_executions_total":
                    continue
                node_total += value
                if dict(label_key).get("status") == "success":
                    node_success += value

            latency_sum = 0.0
            latency_count = 0.0
            for (name, _label_key), hist in self._histograms.items():
                if name == "agent_node_latency_seconds":
                    latency_sum += hist.sum
                    latency_count += hist.count

            loop_hits = self._counters.get(("agent_loop_limit_hits_total", ()), 0.0)
            steps_total = self._counters.get(("agent_steps_total", ()), 0.0)

        return {
            "workflow_completion_rate": (completed / total_workflows) if total_workflows else None,
            "node_success_rate": (node_success / node_total) if node_total else None,
            "average_node_latency_ms": (
                (latency_sum / latency_count) * 1000.0 if latency_count else None
            ),
            "loop_rate": (loop_hits / total_workflows) if total_workflows else None,
            "average_steps": (steps_total / total_workflows) if total_workflows else None,
            "total_workflows": total_workflows,
        }

    def record_vectors(self, count: int) -> None:
        """Set the total number of vectors currently indexed (FAISS index
        size, or pgvector row count when PGVECTOR_ENABLED). A gauge so the
        system dashboard's "vector count" stat reflects the live store
        rather than accumulating over time."""
        self.set_gauge("insightai_vectors_total", float(count))

    # --- Rendering -------------------------------------------------------

    def render(self) -> str:
        """Serialize every metric to the Prometheus text exposition format."""
        with self._lock:
            lines: list[str] = []

            lines.append(
                "# HELP insightai_uptime_seconds Seconds since the application process started."
            )
            lines.append("# TYPE insightai_uptime_seconds gauge")
            lines.append(f"insightai_uptime_seconds {time.time() - self._started_at:.3f}")

            for (name, label_key), value in sorted(self._counters.items()):
                labels = dict(label_key)
                lines.append(f"# HELP {name} Counter incremented by observed events.")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{_format_labels(labels)} {value:.0f}")

            for (name, label_key), value in sorted(self._gauges.items()):
                labels = dict(label_key)
                lines.append(f"# HELP {name} Gauge set to the last observed value.")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{_format_labels(labels)} {value:.3f}")

            for (name, label_key), hist in sorted(self._histograms.items()):
                labels = dict(label_key)
                lines.append(f"# HELP {name} Observation durations in seconds.")
                lines.append(f"# TYPE {name} histogram")
                for i, bound in enumerate(HISTOGRAM_BUCKETS_SECONDS):
                    bucket_labels = dict(labels)
                    bucket_labels["le"] = f"{bound:g}"
                    lines.append(
                        f"{name}_bucket{_format_labels(bucket_labels)} {hist.cumulative[i]:.0f}"
                    )
                # Prometheus: le="+Inf" always equals count — every
                # observation falls into it regardless of the largest
                # finite bound.
                inf_labels = dict(labels)
                inf_labels["le"] = "+Inf"
                lines.append(f"{name}_bucket{_format_labels(inf_labels)} {hist.count:.0f}")
                lines.append(f"{name}_sum{_format_labels(labels)} {hist.sum:.3f}")
                lines.append(f"{name}_count{_format_labels(labels)} {hist.count:.0f}")

            return "\n".join(lines) + "\n"


_metrics = Metrics()


def get_metrics() -> Metrics:
    """The process-wide metrics registry (one per process, module singleton)."""
    return _metrics


def reset_metrics() -> None:
    """Wipe all recorded state. Used by tests for isolation only."""
    _metrics.reset()


class RequestTimer:
    """Measures one HTTP request (duration from construction to finish()).

    Used by main.py's metrics middleware. finish() records the request
    total and latency; the middleware always calls it, including on error
    paths (the error taxonomy breakdown is recorded separately by
    error_handlers.py's record_error()).
    """

    __slots__ = ("_metrics", "_start", "_method", "_path")

    def __init__(self, metric: Metrics, method: str, path: str) -> None:
        self._metrics = metric
        self._method = method
        self._path = normalize_path(path)
        self._start = time.perf_counter()

    def finish(self, status_code: int) -> None:
        duration = time.perf_counter() - self._start
        self._metrics.inc_counter(
            "http_requests_total",
            {"method": self._method, "path": self._path, "status": str(status_code)},
        )
        self._metrics.observe_duration(
            "http_request_duration_seconds",
            duration,
            {"method": self._method, "path": self._path},
        )
