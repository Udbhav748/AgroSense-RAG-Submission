"""Rollup + alert thresholds for the backend's JSON logs.

A tiny, dependency-free stand-in for a hosted monitoring/alerting stack:
aggregates a captured log file (same format as metrics_report.py consumes)
into a windowed health rollup — error rate, latency percentiles, token/cost
totals — and fails (exit 1) if any configured threshold is breached, which a
scheduled job (cron / GitHub Actions) can turn into an alert.

This is explicitly NOT a replacement for Prometheus/Grafana: it's
pull-on-demand over a file, with no long-term storage and no dashboard.
It can push an alert though — see --alert-webhook-url /
monitoring/alert_webhook.py — a best-effort Slack-compatible webhook
POST when any threshold is breached, optional and off by default. See
docs/CHECKLIST.md §10.

Usage (from backend/):
    python monitoring/log_aggregate.py app.log
    python monitoring/log_aggregate.py app.log --window-min 30 --json
    python monitoring/log_aggregate.py app.log --alert-webhook-url https://hooks.slack.com/services/...

Exit codes: 0 = within thresholds, 1 = threshold breached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

# Needed whether this runs as a standalone script (`python
# monitoring/log_aggregate.py` puts monitoring/ on sys.path, not
# backend/) or gets imported under pytest — same reasoning as
# dashboard.py's identical line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.alert_webhook import send_alert  # noqa: E402

# Default thresholds (set each via --*. Values are the alert limit; breaching
# any of them flips the run to "alert").
DEFAULT_ERROR_RATE = 0.05       # fraction of requests that failed
DEFAULT_P95_LATENCY_MS = 5000.0 # p95 of processing_duration
DEFAULT_LOOP_CAPPED_RATE = 0.2  # fraction of requests that hit the loop cap
DEFAULT_TIMEOUT_RATE = 0.1      # fraction of tool calls that timed out
DEFAULT_MIN_NODE_SUCCESS_RATE = 0.8         # agents + tools combined
DEFAULT_MIN_WORKFLOW_COMPLETION_RATE = 0.9  # fraction of requests that reached a final state
DEFAULT_MIN_HANDOFF_ACCURACY = 0.7          # only checked when handoffs > 0

# agent_completed outcomes that mean the agent failed to produce a usable
# result — as opposed to a deliberate, config-gated skip (research
# agent's "disabled"/"pending_approval") or a degrade that still
# returned a usable decision (router's "fallback"). Everything not
# listed here counts as a successful node execution, so a new agent or
# outcome defaults to "success" rather than needing this set updated
# first — the same "unknown isn't a failure" bias this app already uses
# for tenant/ownership checks (see rag_service.py, document_repository.py).
_AGENT_FAILURE_OUTCOMES = {"no_results", "synthesis_failed"}


def _load_records(path: str) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _agent_metrics(records: list[dict]) -> dict:
    """Node Success Rate, Average Node Latency, Agent Handoff Accuracy,
    and Agent Idle Time — all derived from agent_events.py's
    agent_started/agent_completed/agent_handoff events (see
    app/services/agent_events.py, router_agent.py, research_agent.py).
    These events exist specifically for this purpose, but nothing
    computed the actual metrics from them until now.

    "Node" spans both agents and tools (matching the LangGraph vocabulary
    the checklist uses) — agent_completed and tool_invocation are both
    "one execution unit finished," success or fail.
    """
    agent_completions = [r for r in records if r.get("message") == "agent_completed"]
    tool_calls = [r for r in records if r.get("message") == "tool_invocation"]
    handoffs = [r for r in records if r.get("message") == "agent_handoff"]

    agent_latencies = [float(r["duration_ms"]) for r in agent_completions if r.get("duration_ms") is not None]
    tool_latencies = [float(r["latency_ms"]) for r in tool_calls if r.get("latency_ms") is not None]
    all_latencies = agent_latencies + tool_latencies

    successful_agents = sum(1 for r in agent_completions if r.get("outcome") not in _AGENT_FAILURE_OUTCOMES)
    successful_tools = sum(1 for r in tool_calls if r.get("success") is True)
    total_nodes = len(agent_completions) + len(tool_calls)
    successful_nodes = successful_agents + successful_tools
    node_success_rate = successful_nodes / total_nodes if total_nodes else 0.0

    # Agent Handoff Accuracy: a handoff is "correct" if the agent it
    # handed off to (to_agent) went on to actually complete successfully
    # for the same request — the only outcome-based signal available for
    # "was this the right agent to hand off to." Correlated by
    # request_id; the to_agent's first completion after the handoff is
    # taken as its outcome.
    completions_by_request: dict[str, list[dict]] = defaultdict(list)
    for r in agent_completions:
        request_id = r.get("request_id")
        if request_id:
            completions_by_request[request_id].append(r)

    correct_handoffs = 0
    for handoff in handoffs:
        request_id = handoff.get("request_id")
        to_agent = handoff.get("to_agent")
        if not request_id or not to_agent:
            continue
        match = next(
            (c for c in completions_by_request.get(request_id, []) if c.get("agent") == to_agent),
            None,
        )
        if match is not None and match.get("outcome") not in _AGENT_FAILURE_OUTCOMES:
            correct_handoffs += 1
    handoff_accuracy = correct_handoffs / len(handoffs) if handoffs else 0.0

    # Agent Idle Time: the gap between one agent's agent_completed and
    # the next agent_started for a *different* agent, within the same
    # request — time spent waiting between a handoff and the next agent
    # actually picking up the task. Requires request_id (set by the
    # request middleware) to group events by request; off-request calls
    # have no "next agent" to wait for and are naturally excluded.
    lifecycle_events: dict[str, list[tuple[str, str, datetime]]] = defaultdict(list)
    for r in records:
        msg = r.get("message")
        if msg not in ("agent_started", "agent_completed"):
            continue
        request_id = r.get("request_id")
        agent = r.get("agent")
        ts = r.get("timestamp")
        if not request_id or not agent or not ts:
            continue
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        lifecycle_events[request_id].append((msg, agent, parsed))

    idle_gaps_ms: list[float] = []
    for events in lifecycle_events.values():
        events.sort(key=lambda e: e[2])
        pending_agent: str | None = None
        pending_completed_at: datetime | None = None
        for msg, agent, ts in events:
            if msg == "agent_completed":
                pending_agent, pending_completed_at = agent, ts
            elif msg == "agent_started":
                if pending_completed_at is not None and agent != pending_agent:
                    gap_ms = (ts - pending_completed_at).total_seconds() * 1000
                    if gap_ms >= 0:
                        idle_gaps_ms.append(gap_ms)
                pending_agent, pending_completed_at = None, None

    return {
        "agent_completions": len(agent_completions),
        "tool_calls": len(tool_calls),
        "node_success_rate": round(node_success_rate, 4),
        "avg_node_latency_ms": round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0,
        "avg_agent_latency_ms": round(sum(agent_latencies) / len(agent_latencies), 2) if agent_latencies else 0.0,
        "avg_tool_latency_ms": round(sum(tool_latencies) / len(tool_latencies), 2) if tool_latencies else 0.0,
        "handoffs": len(handoffs),
        "correct_handoffs": correct_handoffs,
        "agent_handoff_accuracy": round(handoff_accuracy, 4),
        "agent_idle_time_ms": round(sum(idle_gaps_ms) / len(idle_gaps_ms), 2) if idle_gaps_ms else 0.0,
        "agent_idle_gaps": len(idle_gaps_ms),
    }


def aggregate(records: list[dict], window_min: float = 0.0) -> dict:
    # Optionally narrow to the trailing window by timestamp field.
    if window_min > 0:
        from datetime import timedelta

        cutoff = datetime.now(UTC) - timedelta(minutes=window_min)
        filtered = []
        for rec in records:
            ts = rec.get("timestamp")
            if not ts:
                continue
            try:
                parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed >= cutoff:
                filtered.append(rec)
        records = filtered

    total_requests = 0
    failures = 0
    by_category = {}
    durations = []
    loop_capped = 0
    tokens_in = tokens_out = 0
    cost = 0.0
    tool_attempts = {}  # tool -> total invocations
    tool_successes = {}
    timed_out_tool_calls = 0
    # retrieve()'s own timeout (retrieval_service._search_with_timeout)
    # degrades to an empty result list rather than raising, so it's
    # still a *successful* tool_invocation — its timeout is only visible
    # via this separate event, counted alongside the tool_invocation
    # `timed_out` flag below (see tool_registry._looks_like_timeout).
    retrieval_timeouts = 0

    for rec in records:
        msg = rec.get("message", "")
        if msg == "request_failed":
            total_requests += 1
            failures += 1
            cat = rec.get("taxonomy_category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
        elif msg == "chat_query_handled":
            total_requests += 1
            duration = rec.get("processing_duration")
            if duration is not None:
                durations.append(float(duration))
        if msg == "loop_capped":
            loop_capped += 1
        if msg == "llm_generation_completed":
            tokens_in += int(rec.get("prompt_tokens", 0) or 0)
            tokens_out += int(rec.get("completion_tokens", 0) or 0)
            cost += float(rec.get("estimated_cost_usd", 0.0) or 0.0)
        if msg == "tool_invocation":
            tool = rec.get("tool", "unknown")
            tool_attempts[tool] = tool_attempts.get(tool, 0) + 1
            if rec.get("success") is True:
                tool_successes[tool] = tool_successes.get(tool, 0) + 1
            if rec.get("timed_out") is True:
                timed_out_tool_calls += 1
        if msg == "retrieval_timed_out":
            retrieval_timeouts += 1

    error_rate = failures / total_requests if total_requests else 0.0
    # Workflow Completion Rate: a "workflow" is one full /chat request;
    # "completed" means it reached a final state without erroring — the
    # complement of error_rate, given its own explicit name since the
    # checklist calls it out as a distinct metric.
    workflow_completion_rate = 1 - error_rate if total_requests else 0.0
    loop_capped_rate = loop_capped / total_requests if total_requests else 0.0
    tool_success_rate = {
        tool: (tool_successes.get(tool, 0) / attempts if attempts else 0.0)
        for tool, attempts in tool_attempts.items()
    }
    total_tool_calls = sum(tool_attempts.values())
    timed_out_calls = timed_out_tool_calls + retrieval_timeouts
    timeout_rate = timed_out_calls / total_tool_calls if total_tool_calls else 0.0
    return {
        "records_scanned": len(records),
        "requests": total_requests,
        "failures": failures,
        "error_rate": round(error_rate, 4),
        "workflow_completion_rate": round(workflow_completion_rate, 4),
        "error_rate_by_category": by_category,
        "p50_latency_ms": round(_percentile(durations, 50), 1),
        "p95_latency_ms": round(_percentile(durations, 95), 1),
        "p99_latency_ms": round(_percentile(durations, 99), 1),
        "loop_capped_rate": round(loop_capped_rate, 4),
        "tool_attempts": tool_attempts,
        "tool_successes": tool_successes,
        "tool_success_rate": tool_success_rate,
        "timed_out_calls": timed_out_calls,
        "timeout_rate": round(timeout_rate, 4),
        "total_tokens": tokens_in + tokens_out,
        "estimated_cost_usd": round(cost, 6),
        **_agent_metrics(records),
    }


def _breaches(
    agg: dict,
    max_error: float,
    max_p95: float,
    max_loop: float,
    min_tool_success: float = 0.8,
    max_timeout_rate: float = DEFAULT_TIMEOUT_RATE,
    min_node_success: float = DEFAULT_MIN_NODE_SUCCESS_RATE,
    min_workflow_completion: float = DEFAULT_MIN_WORKFLOW_COMPLETION_RATE,
    min_handoff_accuracy: float = DEFAULT_MIN_HANDOFF_ACCURACY,
) -> list[str]:
    problems = []
    if agg["error_rate"] > max_error:
        problems.append(f"error_rate {agg['error_rate']:.4f} > {max_error}")
    if agg["p95_latency_ms"] > max_p95:
        problems.append(f"p95_latency {agg['p95_latency_ms']:.1f}ms > {max_p95}ms")
    if agg["loop_capped_rate"] > max_loop:
        problems.append(f"loop_capped_rate {agg['loop_capped_rate']:.4f} > {max_loop}")
    if agg["timeout_rate"] > max_timeout_rate:
        problems.append(f"timeout_rate {agg['timeout_rate']:.4f} > {max_timeout_rate}")
    for tool, rate in agg["tool_success_rate"].items():
        if rate < min_tool_success:
            problems.append(f"tool_success_rate({tool}) {rate:.4f} < {min_tool_success}")
    # Guarded by their own counts, not just `requests`/`handoffs > 0` —
    # an empty-of-that-signal window (e.g. agent_routing_enabled=false,
    # so zero handoffs ever) must never look like a breach.
    if agg["requests"] > 0 and agg["workflow_completion_rate"] < min_workflow_completion:
        problems.append(f"workflow_completion_rate {agg['workflow_completion_rate']:.4f} < {min_workflow_completion}")
    if agg["agent_completions"] + agg["tool_calls"] > 0 and agg["node_success_rate"] < min_node_success:
        problems.append(f"node_success_rate {agg['node_success_rate']:.4f} < {min_node_success}")
    if agg["handoffs"] > 0 and agg["agent_handoff_accuracy"] < min_handoff_accuracy:
        problems.append(f"agent_handoff_accuracy {agg['agent_handoff_accuracy']:.4f} < {min_handoff_accuracy}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", help="Path to a captured backend JSON log file.")
    parser.add_argument("--window-min", type=float, default=0.0, help="Only aggregate records within the trailing N minutes (0 = all).")
    parser.add_argument("--max-error-rate", type=float, default=DEFAULT_ERROR_RATE)
    parser.add_argument("--max-p95-latency-ms", type=float, default=DEFAULT_P95_LATENCY_MS)
    parser.add_argument("--max-loop-capped-rate", type=float, default=DEFAULT_LOOP_CAPPED_RATE)
    parser.add_argument("--min-tool-success-rate", type=float, default=0.8)
    parser.add_argument("--max-timeout-rate", type=float, default=DEFAULT_TIMEOUT_RATE)
    parser.add_argument("--min-node-success-rate", type=float, default=DEFAULT_MIN_NODE_SUCCESS_RATE)
    parser.add_argument("--min-workflow-completion-rate", type=float, default=DEFAULT_MIN_WORKFLOW_COMPLETION_RATE)
    parser.add_argument("--min-handoff-accuracy", type=float, default=DEFAULT_MIN_HANDOFF_ACCURACY)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text.")
    parser.add_argument(
        "--alert-webhook-url",
        default=os.environ.get("ALERT_WEBHOOK_URL"),
        help="Slack-compatible webhook URL to POST to when a threshold is breached. Unset = no-op.",
    )
    args = parser.parse_args()

    records = _load_records(args.logfile)
    agg = aggregate(records, window_min=args.window_min)
    problems = _breaches(
        agg,
        args.max_error_rate,
        args.max_p95_latency_ms,
        args.max_loop_capped_rate,
        args.min_tool_success_rate,
        args.max_timeout_rate,
        args.min_node_success_rate,
        args.min_workflow_completion_rate,
        args.min_handoff_accuracy,
    )

    if problems:
        send_alert(
            args.alert_webhook_url,
            f"AgroSense-RAG log_aggregate alert ({agg['requests']} requests scanned):\n"
            + "\n".join(f"- {p}" for p in problems),
        )

    if args.json:
        print(json.dumps({"aggregate": agg, "alerts": problems, "alert": bool(problems)}))
    else:
        print(f"records scanned : {agg['records_scanned']}")
        print(f"requests        : {agg['requests']}  failures: {agg['failures']}  "
              f"error_rate: {agg['error_rate']:.4f}  by_category: {agg['error_rate_by_category']}")
        print(f"workflow complete: {agg['workflow_completion_rate']:.4f}")
        print(f"latency (ms)    : p50={agg['p50_latency_ms']}  p95={agg['p95_latency_ms']}  p99={agg['p99_latency_ms']}")
        print(f"loop capped rate: {agg['loop_capped_rate']:.4f}")
        print(f"tool success    : {agg['tool_success_rate']}")
        print(f"timeout rate    : {agg['timeout_rate']:.4f}  ({agg['timed_out_calls']}/{sum(agg['tool_attempts'].values()) or 0} tool calls)")
        print(f"node success    : {agg['node_success_rate']:.4f}  ({agg['agent_completions']} agent + {agg['tool_calls']} tool executions)")
        print(f"node latency (ms): avg={agg['avg_node_latency_ms']}  agent_avg={agg['avg_agent_latency_ms']}  tool_avg={agg['avg_tool_latency_ms']}")
        print(f"handoff accuracy: {agg['agent_handoff_accuracy']:.4f}  ({agg['correct_handoffs']}/{agg['handoffs']} handoffs)")
        print(f"agent idle time : avg={agg['agent_idle_time_ms']}ms over {agg['agent_idle_gaps']} gaps")
        print(f"tokens / cost   : {agg['total_tokens']} / ${agg['estimated_cost_usd']:.6f}")
        if problems:
            print("ALERTS BREACHED:")
            for p in problems:
                print(f"  - {p}")
        else:
            print("Within thresholds.")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
