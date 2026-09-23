#!/usr/bin/env python
"""Module 10 gap-closure (P6): the single authoritative observability
metric report -- ties together the instrumentation that already exists
(structured JSON logging, request_id/trace_id, agent_node_trace,
tool_invocation, LLM token/cost events, the Prometheus-style /metrics
registry) with the offline aggregation (monitoring/log_aggregate.py,
monitoring/dashboard.py), the bounded local availability measurement
(run_availability_eval.py), and the automated-alerting validation
(app/core/alerting.py) -- into ONE machine-readable artifact, and is
explicit everywhere about which layer each number came from.

Traffic source: the SAME real, controlled 35-request sample
run_observability_eval.py already uses (30 successful POST /chat + 5
error-path DELETE /documents/{missing_id}), through the actual FastAPI
app via TestClient, LLM/embedding calls mocked (no API quota consumed,
mirrors tests/test_main.py's own fixture). What's new here is that this
run captures EVERY structured log line the app emits during that
traffic (not just latency), and feeds it through the same offline
aggregation the real monitoring/ scripts use against a real captured
app.log -- so the numbers in this report are the real aggregation
logic exercised against a real (if synthetic-traffic) log stream, not
hand-computed.

Usage (from backend/):
    python eval/module10/runners/run_observability_final_eval.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.alerting import AlertRule, Comparator, MockNotificationSink, build_default_engine  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.models.document import EmbeddedChunk  # noqa: E402
from app.services.cache_service import cache_service  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402
from eval.module10 import config  # noqa: E402
from monitoring.dashboard import _endpoint_breakdown, _requests_per_minute, _retry_activity  # noqa: E402
from monitoring.log_aggregate import aggregate  # noqa: E402

FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)
DOCUMENT_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
N_SUCCESS_REQUESTS = 30
N_ERROR_REQUESTS = 5


def _distinct_fake_embedding(query: str) -> list[float]:
    """A per-query embedding used only to keep retrieval deterministic
    and offline (no live sentence-transformers call). NOT relied on to
    prevent cache collisions -- see cache_service.clear() calls in the
    traffic loop below for that; a real finding while building this
    report was that a single constant fake embedding across all 30
    requests, PLUS embeddings that happened to still land within the
    real SemanticQueryCache's 0.96 cosine-similarity threshold even when
    varied, collapsed most "different" requests into cache hits that
    never reach finalizer_node/ChatService._respond() and therefore
    never log "chat_query_handled" -- see
    tests/test_observability_cache_gap.py and this report's own
    methodology/limitations for the full, disclosed finding.
    """
    idx = abs(hash(query)) % (FAKE_EMBEDDING_DIM - 1) + 1
    magnitude = 0.3 + (abs(hash(query)) % 100) / 200.0  # 0.3..0.795, varies per query
    vec = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)
    vec[idx] = magnitude
    return vec


class _FakeLLMClient:
    def generate(self, prompt: str) -> str:
        return "A grounded answer citing the retrieved context [1]."

    def generate_stream(self, prompt: str):
        yield self.generate(prompt)

    def generate_structured(self, prompt: str) -> str:
        return '{"answer": "structured answer", "sources": []}'


class _AllRecordsCaptureHandler(logging.Handler):
    """Captures every structured log record emitted during the `with`
    block, reshaped into the exact dict schema a real captured app.log
    line has (see app/core/logging.py::JSONFormatter) -- {timestamp,
    message, **extra_fields} -- so monitoring/log_aggregate.py's
    aggregate() (written to parse real log FILES) can be run against
    this in-memory list unmodified, with no separate "eval mode" parser
    to maintain."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            entry.update(extra_fields)
        self.records.append(entry)


def _run_controlled_traffic_and_capture_logs() -> tuple[list[dict], dict]:
    import tempfile

    handler = _AllRecordsCaptureHandler()
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    timings = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = FAISSVectorStore(index_path=tmp_path / "index.faiss", metadata_path=tmp_path / "metadata.json")
            store.create_index(dimension=FAKE_EMBEDDING_DIM)
            store.add_embeddings(
                [
                    EmbeddedChunk(
                        chunk_id="chunk-1",
                        document_id=DOCUMENT_ID,
                        embedding=FAKE_EMBEDDING,
                        metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "Apple scab treatment: sulfur or captan."},
                    )
                ]
            )
            fake_llm = _FakeLLMClient()

            with (
                patch("app.api.v1.routes.query.get_vector_store", lambda: store),
                patch("app.api.v1.routes.documents.get_vector_store", lambda: store),
                patch("app.api.v1.routes.query.get_llm_client", lambda: fake_llm),
                patch("app.services.retrieval_service.embed_query", _distinct_fake_embedding),
                patch("app.services.hybrid_search.embed_query", _distinct_fake_embedding),
                # cache_service._get_embedding() falls back to
                # app.services.embedding_service.embed_query directly
                # (a third import site, distinct from the two above) when
                # no explicit query_embedding is passed -- cache_lookup_node
                # doesn't pass one, so this must be patched too, both to
                # avoid a live sentence-transformers model load here and
                # to control whether the semantic cache treats these
                # requests as duplicates (see _distinct_fake_embedding's
                # docstring for what happened before this was added).
                patch("app.services.embedding_service.embed_query", _distinct_fake_embedding),
                TestClient(app) as client,
            ):
                headers = {"X-API-Key": settings.api_key}
                start = time.perf_counter()
                for i in range(N_SUCCESS_REQUESTS):
                    # Clear the response cache before every request so each
                    # of these 30 "different" queries genuinely exercises
                    # the full retrieval->generation pipeline for this
                    # measurement, rather than the real SemanticQueryCache
                    # (correctly) treating near-duplicate phrasing as a hit
                    # -- see _distinct_fake_embedding's docstring and
                    # tests/test_observability_cache_gap.py for that real,
                    # disclosed finding.
                    cache_service.clear()
                    client.post("/chat", json={"query": f"What treats apple scab? (req {i})"}, headers=headers)
                for i in range(N_ERROR_REQUESTS):
                    client.delete(f"/documents/nonexistent-{i}", params={"confirm": "true"}, headers=headers)
                timings["wall_clock_seconds"] = round(time.perf_counter() - start, 4)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)

    return handler.records, timings


def _validate_alerting(agg: dict) -> dict:
    """TASK 5/6: metric input -> threshold evaluation -> alert triggered
    -> payload produced, run against BOTH the real measured aggregate
    (expected: no alert, since the controlled sample is healthy) and a
    deliberately synthetic breach scenario (expected: alert) -- proving
    the path runs, not just that it imports."""
    results = []

    # 1) error rate above threshold (synthetic breach -- the real sample is healthy)
    sink = MockNotificationSink()
    engine = build_default_engine(sink)
    events = engine.evaluate("error_rate", 0.50, now=0.0)
    results.append({
        "scenario": "error_rate_above_threshold",
        "threshold": 0.05,
        "input_metric_value": 0.50,
        "expected_alert": True,
        "actual_alert_triggered": len(events) == 1 and events[0].kind == "alert",
        "payload_produced": bool(sink.delivered),
    })

    # 2) latency above threshold (synthetic breach)
    sink = MockNotificationSink()
    engine = build_default_engine(sink)
    events = engine.evaluate("p95_latency_seconds", 10.0, now=0.0)
    results.append({
        "scenario": "p95_latency_above_threshold",
        "threshold": 3.0,
        "input_metric_value": 10.0,
        "expected_alert": True,
        "actual_alert_triggered": len(events) == 1 and events[0].kind == "alert",
        "payload_produced": bool(sink.delivered),
    })

    # 3) metric below a minimum threshold (health_check_success)
    sink = MockNotificationSink()
    engine = build_default_engine(sink)
    events = engine.evaluate("health_check_success", 0.0, now=0.0)
    results.append({
        "scenario": "health_check_success_below_minimum",
        "threshold": 0.5,
        "input_metric_value": 0.0,
        "expected_alert": True,
        "actual_alert_triggered": len(events) == 1 and events[0].kind == "alert",
        "payload_produced": bool(sink.delivered),
    })

    # 4) no alert when values remain within threshold -- fed the REAL
    # measured error rate from this run's own controlled traffic sample.
    sink = MockNotificationSink()
    engine = build_default_engine(sink)
    events = engine.evaluate("error_rate", agg["error_rate"], now=0.0)
    results.append({
        "scenario": "real_measured_error_rate_within_threshold",
        "threshold": 0.05,
        "input_metric_value": agg["error_rate"],
        "expected_alert": agg["error_rate"] > 0.05,
        "actual_alert_triggered": len(events) == 1,
        "payload_produced": bool(sink.delivered),
    })

    all_as_expected = all(r["actual_alert_triggered"] == r["expected_alert"] for r in results)
    return {
        "results": results,
        "all_scenarios_behaved_as_expected": all_as_expected,
        "note": "Scenarios 1-3 use deliberately synthetic breach values (documented as such) to prove "
        "the threshold-evaluation-to-alert-payload path actually runs; scenario 4 uses this run's own "
        "real measured error_rate to prove the healthy/no-alert path also runs against real data, not "
        "only synthetic inputs.",
    }


def _validate_dashboard(records: list[dict], agg: dict) -> dict:
    """TASK 7: run monitoring/dashboard.py's own view functions against
    the same captured records and confirm each required view is present
    and non-crashing -- not a fake screenshot."""
    retry = _retry_activity(records)
    endpoints = _endpoint_breakdown(records)
    rpm = _requests_per_minute(records)

    required_views = {
        "availability": "error_rate" in agg and "requests" in agg,
        "latency": all(k in agg for k in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms")),
        "error_rate": "error_rate" in agg,
        "tool_success": "tool_success_rate" in agg,
        "retry_activity": "retry_events" in retry,
        "requests": rpm is not None,
        "token_usage": "total_tokens" in agg,
        "cost": "estimated_cost_usd" in agg,
    }
    return {
        "required_views_present": required_views,
        "all_required_views_present": all(required_views.values()),
        "retry_activity": retry,
        "endpoint_breakdown": endpoints,
        "requests_per_minute": rpm,
        "validated_against": "the same captured records this report's aggregate metrics come from "
        "(see traffic_source below) -- not a separate/hypothetical log file.",
    }


def _run_availability_as_subprocess(*, port: int, duration_seconds: float, interval_seconds: float) -> dict:
    """Runs the availability measurement as a genuinely separate OS
    process rather than calling run_availability_measurement() in-process.

    This process already holds an open sqlite3 connection (via the
    TestClient app instantiated for the traffic-capture step above) and
    a loaded sentence-transformers/torch state; spawning the probed
    uvicorn subprocess from inside this same process caused it to not
    become healthy within the startup timeout on this run (most likely
    file-lock contention on the shared backend/db.sqlite3 file) --
    running it as a fully separate `python run_availability_eval.py`
    invocation, exactly as a human operator would from the command line,
    avoids that in-process contention entirely and is what actually
    produced a real 1.0 availability result when tried standalone.
    """
    import re
    import subprocess as _subprocess

    backend_root = Path(__file__).resolve().parents[3]
    proc = _subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_availability_eval.py"),
            "--port", str(port),
            "--duration-seconds", str(duration_seconds),
            "--interval-seconds", str(interval_seconds),
        ],
        cwd=str(backend_root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    match = re.search(r"Saved: (.+\.json)", proc.stdout)
    if not match:
        return {
            "started": False,
            "error": "run_availability_eval.py subprocess did not report a saved report path.",
            "stdout_tail": proc.stdout[-500:],
            "stderr_tail": proc.stderr[-500:],
        }
    import json as _json

    saved_path = Path(match.group(1).strip())
    return _json.loads(saved_path.read_text(encoding="utf-8"))


def main() -> None:
    records, timings = _run_controlled_traffic_and_capture_logs()
    agg = aggregate(records)
    dashboard_result = _validate_dashboard(records, agg)
    alert_result = _validate_alerting(agg)

    availability_result = _run_availability_as_subprocess(port=8813, duration_seconds=15, interval_seconds=1)

    report = {
        "metadata": {
            **config.run_metadata(sample_count=agg["requests"], dataset_version="observability_v2"),
            "evaluator": "eval/module10/runners/run_observability_final_eval.py",
        },
        "traffic_source": {
            "description": "30 successful POST /chat + 5 error-path DELETE /documents/{missing_id} "
            "through the real FastAPI app via TestClient, LLM/embedding calls mocked (no API quota "
            "consumed) -- mirrors tests/test_main.py's own fixture pattern and "
            "run_observability_eval.py's existing sample.",
            "n_requests": N_SUCCESS_REQUESTS + N_ERROR_REQUESTS,
            "wall_clock_seconds": timings["wall_clock_seconds"],
            "time_window": f"{report_time_window(records)}",
        },
        "instrumentation_audit": {
            "real_runtime_instrumentation": [
                "structured JSON logging (app/core/logging.py)",
                "request_id/trace_id propagation (app/core/request_context.py, main.py middleware)",
                "agent_node_trace per-node tracing (app/services/agent_graph/events.py)",
                "tool_invocation logging (app/services/tool_registry.py's @track_tool)",
                "LLM token/cost logging (llm_generation_completed, gemini_client.py/groq_client.py)",
                "live GET /metrics Prometheus-style registry (app/core/metrics.py)",
            ],
            "offline_aggregation": [
                "monitoring/log_aggregate.py::aggregate() -- pull-on-demand over a log file",
                "monitoring/dashboard.py -- read-only text dashboard over the same aggregate()",
                "eval/metrics_report.py -- eval-harness-side metrics rollup",
            ],
            "automated_alerting": [
                "app/core/alerting.py::AlertEngine -- real, tested, debounced threshold engine "
                "(this pass's TASK 5 proves it runs end to end); NOT continuously invoked against a "
                "live target -- no scheduled job currently calls AlertEngine.evaluate() periodically.",
                "monitoring/log_aggregate.py's own simpler _breaches()+send_alert() webhook path -- "
                "a SEPARATE, simpler mechanism than AlertEngine (no debounce/recovery events), used by "
                "the CLI script directly. Both exist; they are not the same code path -- documented "
                "here to avoid implying they're unified when they are not.",
            ],
            "currently_inert_without_a_deployment": [
                ".github/workflows/health-monitor.yml -- inert by design, no persistent deployment to "
                "poll (pre-existing, unchanged by this pass)",
                "AlertEngine is not wired to any periodic real-metrics poll -- it is exercised here and "
                "in tests/test_alerting.py / tests/test_alert_engine_integration.py, not in a live loop.",
            ],
            "manual_only": [
                "monitoring/uptime_check.py and this pass's run_availability_eval.py are invoked "
                "on demand, not on a running schedule against a live target.",
            ],
        },
        "aggregate_metrics": agg,
        "aggregate_error_rate": {
            "value": agg["error_rate"],
            "formula": "failed_requests / total_requests",
            "by_taxonomy_category": agg["error_rate_by_category"],
            "note": "Both the single aggregate number and the taxonomy breakdown are always reported "
            "together -- the aggregate does not replace the category breakdown (TASK 3).",
        },
        "latency_percentiles_ms": {
            "p50": agg["p50_latency_ms"],
            "p95": agg["p95_latency_ms"],
            "p99": agg["p99_latency_ms"],
        },
        "tool_and_retry": {
            "tool_attempts": agg["tool_attempts"],
            "tool_successes": agg["tool_successes"],
            "tool_success_rate": agg["tool_success_rate"],
            "timed_out_calls": agg["timed_out_calls"],
            "loop_capped_rate": agg["loop_capped_rate"],
            **dashboard_result["retry_activity"],
        },
        "token_and_cost": {
            "total_tokens": agg["total_tokens"],
            "estimated_cost_usd": agg["estimated_cost_usd"],
            "note": "0 in this run because the LLM client is mocked (no live API call) -- real "
            "token/cost figures from live runs are reported separately in "
            "eval/module10/reports/provider_ab_eval_*.json and docs/MODULE10_RESULTS.md's latency/cost "
            "sections, not fabricated here.",
        },
        "availability": availability_result,
        "alerting_validation": alert_result,
        "dashboard_validation": dashboard_result,
        "prompt_logging_status": {
            "log_prompt_content_default": False,
            "log_prompt_content_current_setting": settings.log_prompt_content,
            "prompt_version_always_recorded": True,
            "boundary": "Exact prompt content is a controlled/debug-only capture "
            "(Settings.log_prompt_content, off by default) -- see "
            "tests/test_prompt_capture_boundary.py. Never automatically enabled by this pass.",
        },
        "limitations": [
            "Traffic is a controlled local TestClient sample with a mocked LLM -- application overhead "
            "latency (routing, retrieval, graph execution) is real; live LLM provider round-trip "
            "latency is not included here (see the Provider A-B evaluation for that).",
            "35 requests on a single local process is not a production-scale sample.",
            "Availability is a bounded local measurement (see the 'availability' block's own label) -- "
            "not production availability, no SLO is claimed.",
            "AlertEngine's automated path is validated here to prove it runs correctly against real "
            "and synthetic inputs -- it is not continuously/automatically invoked against a live "
            "target on a schedule; no such deployment exists for this project.",
            "No hosted dashboard/durable monitoring service exists -- monitoring/dashboard.py is a "
            "dependency-free, on-demand terminal view over a log file, validated here against real "
            "captured records, not a Grafana-style live service.",
            "No long-lived SLO measurement exists or is claimed.",
            "No centralized logging service exists -- logs are process-local stdout JSON lines.",
        ],
    }

    path = config.save_report(report, name="observability_final")
    print(f"Saved: {path}")
    print(f"Error rate: {agg['error_rate']}  P50={agg['p50_latency_ms']}ms P95={agg['p95_latency_ms']}ms P99={agg['p99_latency_ms']}ms")
    print(f"Availability: {availability_result.get('availability')}")
    print(f"Alerting scenarios as expected: {alert_result['all_scenarios_behaved_as_expected']}")
    print(f"Dashboard required views present: {dashboard_result['all_required_views_present']}")


def report_time_window(records: list[dict]) -> str:
    timestamps = [r["timestamp"] for r in records if r.get("timestamp")]
    if not timestamps:
        return "n/a"
    return f"{min(timestamps)} -> {max(timestamps)}"


if __name__ == "__main__":
    main()
