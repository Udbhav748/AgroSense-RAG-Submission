"""Text dashboard over the backend's JSON logs.

A dependency-free, terminal-rendered dashboard (the checklist's
"Dashboards" gap, closed locally — a stand-in for a hosted Grafana-style
stack, consistent with log_aggregate.py and metrics_report.py being
stand-ins for real observability). Reuses log_aggregate.aggregate for the
rollups so the two tools can never drift apart on the numbers, and adds
the views a dashboard needs and an aggregate doesn't: requests per minute,
per-endpoint breakdown, retry activity, and the availability-window view.

Usage (from backend/):
    python monitoring/dashboard.py app.log
    python monitoring/dashboard.py app.log --window-min 30
    python monitoring/dashboard.py app.log --json      # machine-readable

Exit code is always 0 — this is a read-only view, not an alert gate
(that's log_aggregate.py's job). See docs/CHECKLIST.md §10.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

# Make the backend root importable regardless of how this script is run
# (`python monitoring/dashboard.py` puts monitoring/ on sys.path, not
# backend/). Reuses log_aggregate's rollup so the two tools can never
# drift apart on the numbers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.log_aggregate import _load_records, aggregate  # noqa: E402


def _time_window(records: list[dict], window_min: float) -> list[dict]:
    if window_min <= 0:
        return records
    cutoff = datetime.now(UTC).timestamp() - window_min * 60
    kept = []
    for rec in records:
        ts = rec.get("timestamp")
        if not ts:
            continue
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.timestamp() >= cutoff:
            kept.append(rec)
    return kept


def _requests_per_minute(records: list[dict]) -> float:
    buckets: Counter[str] = Counter()
    for rec in records:
        msg = rec.get("message")
        if msg not in ("chat_query_handled", "request_failed"):
            continue
        ts = rec.get("timestamp")
        if not ts:
            continue
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        buckets[parsed.strftime("%Y-%m-%d %H:%M")] += 1
    if not buckets:
        return 0.0
    span_minutes = max(1, len(buckets))
    return sum(buckets.values()) / span_minutes


def _endpoint_breakdown(records: list[dict]) -> dict[str, int]:
    # /chat/stream emits trace + answer_chunk events with no per-endpoint
    # marker, so this breakdown keys on the request outcome events whose
    # message implies the endpoint they served.
    counters: dict[str, int] = defaultdict(int)
    for rec in records:
        msg = rec.get("message")
        if msg == "chat_query_handled":
            counters["chat_query_handled"] += 1
        elif msg == "request_failed":
            counters["request_failed"] += 1
        elif msg == "document_processing_completed":
            counters["upload_completed"] += 1
        elif msg == "diagnose_response_sent":
            counters["diagnose_completed"] += 1
    return dict(counters)


# Every retry-capable call this app makes: (retry event, its matching
# completion event). Mirrors eval/metrics_report.py's
# _RETRY_EVENT_FAMILIES — kept as its own copy rather than importing
# across the eval/monitoring boundary, consistent with how these two
# scripts already duplicate rather than share their small helpers.
_RETRY_EVENT_FAMILIES = [
    ("llm_generation_retrying", "llm_generation_completed"),
    ("web_search_retrying", "web_search_completed"),
    ("embed_query_retrying", "embed_query_completed"),
    ("vision_request_retrying", "vision_diagnosis_completed"),
]


def _retry_activity(records: list[dict]) -> dict:
    retried_ids: set[str] = set()
    success_ids: set[str] = set()
    retry_events = 0
    providers: Counter = Counter()

    for retry_message, completed_message in _RETRY_EVENT_FAMILIES:
        retries = [r for r in records if r.get("message") == retry_message]
        if not retries:
            continue
        completed_ids = {
            r.get("request_id") for r in records if r.get("message") == completed_message and r.get("request_id")
        }
        retry_events += len(retries)
        providers.update(r.get("provider", "unknown") for r in retries)
        for r in retries:
            request_id = r.get("request_id")
            if not request_id:
                continue
            # Namespaced by family — one request can retry more than one
            # kind of call, so the same request_id could otherwise
            # collide across families.
            namespaced_id = f"{retry_message}:{request_id}"
            retried_ids.add(namespaced_id)
            if request_id in completed_ids:
                success_ids.add(namespaced_id)

    return {
        "retry_events": retry_events,
        "retried_requests": len(retried_ids),
        "retry_success_rate": round(len(success_ids) / len(retried_ids), 4) if retried_ids else 0.0,
        "providers": dict(providers),
    }


def _print_dashboard(records: list[dict], agg: dict, window_min: float) -> None:
    retry = _retry_activity(records)
    requests_per_min = _requests_per_minute(records)
    endpoints = _endpoint_breakdown(records)
    first_ts = next((r.get("timestamp") for r in records if r.get("timestamp")), "n/a")
    last_ts = next((r.get("timestamp") for r in reversed(records) if r.get("timestamp")), "n/a")

    window_label = f" (last {window_min:.0f}m)" if window_min > 0 else ""
    print("=" * 64)
    print("AgroSense-RAG - text dashboard")
    print("=" * 64)
    print(f"records scanned : {agg['records_scanned']}{window_label}")
    print(f"window         : {first_ts}  ->  {last_ts}")
    print(f"throughput     : ~{requests_per_min:.2f} requests/min")
    print()
    print("-- Availability --")
    print(f"requests       : {agg['requests']}")
    print(f"failures       : {agg['failures']}")
    print(f"error rate     : {agg['error_rate']:.4f}")
    print(f"workflow complete: {agg['workflow_completion_rate']:.4f}")
    print(f"by category    : {agg['error_rate_by_category'] or 'none'}")
    print(f"endpoints      : {endpoints or 'none'}")
    print()
    print("-- Latency (processing_duration, ms) --")
    print(f"p50={agg['p50_latency_ms']}  p95={agg['p95_latency_ms']}  p99={agg['p99_latency_ms']}")
    print()
    print("-- Agent / loop --")
    print(f"loop capped rate : {agg['loop_capped_rate']:.4f}")
    print(f"tool success rate: {agg['tool_success_rate'] or 'no tool events'}")
    print(f"timeout rate     : {agg['timeout_rate']:.4f}  ({agg['timed_out_calls']} timed-out calls)")
    print(f"node success rate: {agg['node_success_rate']:.4f}  ({agg['agent_completions']} agent + {agg['tool_calls']} tool executions)")
    print(f"node latency (ms): avg={agg['avg_node_latency_ms']}  agent_avg={agg['avg_agent_latency_ms']}  tool_avg={agg['avg_tool_latency_ms']}")
    print(f"handoff accuracy : {agg['agent_handoff_accuracy']:.4f}  ({agg['correct_handoffs']}/{agg['handoffs']} handoffs)")
    print(f"agent idle time  : avg={agg['agent_idle_time_ms']}ms over {agg['agent_idle_gaps']} gaps")
    print()
    print("-- LLM / retries --")
    print(f"retry events     : {retry['retry_events']}  (providers: {retry['providers'] or 'none'})")
    print(f"retried requests : {retry['retried_requests']}")
    print(f"retry success    : {retry['retry_success_rate']:.4f}")
    print(f"tokens / cost    : {agg['total_tokens']} / ${agg['estimated_cost_usd']:.6f}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", help="Path to a captured backend JSON log file.")
    parser.add_argument("--window-min", type=float, default=0.0, help="Only show records within the trailing N minutes (0 = all).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the text dashboard.")
    args = parser.parse_args()

    path = Path(args.logfile)
    if not path.is_file():
        raise SystemExit(f"Log file not found: {path}")

    records = _time_window(_load_records(str(path)), args.window_min)
    if not records:
        raise SystemExit("No records in the window.")
    agg = aggregate(records, window_min=args.window_min)

    if args.json:
        print(json.dumps({"aggregate": agg, "retry": _retry_activity(records),
                          "requests_per_min": _requests_per_minute(records),
                          "endpoints": _endpoint_breakdown(records)}))
    else:
        _print_dashboard(records, agg, args.window_min)
    sys.exit(0)


if __name__ == "__main__":
    main()
