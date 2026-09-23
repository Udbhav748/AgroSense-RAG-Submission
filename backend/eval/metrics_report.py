"""Parses the backend's JSON logs and prints a lightweight metrics report:
latency percentiles, error rate by taxonomy_category, corrective-RAG Loop
Count and Average Steps, Retry Success Rate and Timeout Rate across every
retry-capable call, total LLM token usage/cost, Acceptance Rate from chat
feedback, per-criterion Rubric Scores, and Inter-Annotator Agreement
across reviewers who scored the same response.

This is a stand-in for real observability, not a replacement for it — in
production you'd ship these same structured log lines to something like
Prometheus/Grafana (metrics + dashboards + alerting) instead of grepping
log files after the fact. This script exists so the numbers are
inspectable locally without standing up that infrastructure.

Each line of input is expected to be one JSON object, as emitted by
app.core.logging.JSONFormatter — i.e. the backend's own stdout.

Usage (from backend/):
    uvicorn app.main:app | tee app.log   # in one terminal, generate traffic
    python eval/metrics_report.py app.log   # in another

    # or pipe directly:
    cat app.log | python eval/metrics_report.py

Acceptance Rate, Rubric Scores, and Inter-Annotator Agreement are read
separately, from backend/feedback/feedback.jsonl (written by
POST /chat/feedback — see app/services/feedback_service.py), not from the
log file argument above:
    python eval/metrics_report.py app.log --feedback-file feedback/feedback.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Mirrors Settings.feedback_dir_name / feedback_filename's defaults
# (app/core/config.py) without importing app.core.config itself — this
# script is otherwise dependency-free (no .env/Settings needed to just
# parse log files), and the default here is only a convenience;
# --feedback-file overrides it for a non-default FEEDBACK_DIR_NAME.
DEFAULT_FEEDBACK_PATH = Path(__file__).resolve().parents[1] / "feedback" / "feedback.jsonl"


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy/scipy dependency for a script this small."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def parse_log_lines(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Non-JSON lines (e.g. a stray print, uvicorn's startup banner
            # if it slipped into the same stream) are skipped rather than
            # crashing the whole report over one bad line.
            continue
    return records


def report_latency(records: list[dict]) -> None:
    by_event = defaultdict(list)
    for record in records:
        duration = record.get("processing_duration")
        if isinstance(duration, (int, float)):
            by_event[record.get("message", "unknown")].append(duration)

    all_durations = [duration for durations in by_event.values() for duration in durations]

    print("=== Latency (processing_duration, seconds) ===")
    if not all_durations:
        print("No processing_duration values found in the logs.\n")
        return

    print(
        f"overall (n={len(all_durations)}): "
        f"P50={_percentile(all_durations, 50):.4f}  "
        f"P95={_percentile(all_durations, 95):.4f}  "
        f"P99={_percentile(all_durations, 99):.4f}"
    )
    for event, durations in sorted(by_event.items()):
        print(
            f"  {event:<32s} (n={len(durations):>4d}): "
            f"P50={_percentile(durations, 50):.4f}  "
            f"P95={_percentile(durations, 95):.4f}  "
            f"P99={_percentile(durations, 99):.4f}"
        )
    print()


def report_error_rate_by_category(records: list[dict]) -> None:
    errors = [record for record in records if record.get("message") == "request_failed"]

    print("=== Error rate by taxonomy_category ===")
    if not errors:
        print("No request_failed entries found in the logs.\n")
        return

    counts: dict[str, int] = defaultdict(int)
    for error in errors:
        counts[error.get("taxonomy_category", "unknown")] += 1

    total = len(errors)
    for category, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {category:<12s} {count:>4d}  ({count / total * 100:.1f}% of errors)")
    print(f"  {'total':<12s} {total:>4d}")
    print()


def report_loop_count_and_avg_steps(records: list[dict]) -> None:
    """Average Steps and loop_capped rate, from two events ChatService
    already logs — no new logging needed, this only aggregates fields
    that exist today:

    - chat_query_handled (ChatService._respond, one per /chat or
      /chat/stream response): its steps_taken field is averaged across
      every request regardless of query_type, matching how the field is
      already surfaced as one number per response (see the README's API
      reference).
    - loop_capped (ChatService._correct/_correct_streamed,
      rag_service.py): logged specifically when the corrective loop hits
      Settings-independent _MAX_LLM_CALLS (3) generate() calls without
      resolving — i.e. the loop was forced to stop and return whatever
      answer it had, rather than looping freely. This is a narrower,
      more precise signal than "did the loop run at all" (most
      reflection/web-fallback iterations resolve well under the cap and
      never log this event) — it specifically flags requests where the
      cap, not the model being satisfied, ended the loop.

    The fraction is loop_capped events over chat_query_handled requests:
    both are counted straight from the same parsed log stream, so no
    correlation by request id is needed (there's exactly one
    chat_query_handled per request, and at most one loop_capped per
    request since hitting the cap ends the loop immediately).
    """
    handled = [record for record in records if record.get("message") == "chat_query_handled"]
    capped = [record for record in records if record.get("message") == "loop_capped"]

    print("=== Loop Count / Average Steps ===")
    if not handled:
        print("No chat_query_handled entries found in the logs.\n")
        return

    all_steps = [record.get("steps_taken", 0) for record in handled]
    avg_steps = sum(all_steps) / len(all_steps)
    capped_rate = len(capped) / len(handled)

    print(f"  requests:            {len(handled)}")
    print(f"  average steps:       {avg_steps:.2f}")
    print(f"  loop_capped events:  {len(capped)}  ({capped_rate * 100:.1f}% of requests)")
    print()


# Every retry-capable call this app makes: (retry event, its matching
# completion event, a human label). Each retries via tenacity and logs a
# `<name>_retrying` event (before_sleep) plus a `<name>_completed` event
# on eventual success. LLM generation was the only family tracked here
# for a while; web search, query embedding, and vision/diagnose all
# gained their own retry+completion events since (see
# web_search_service.py, embedding_service.py, vision_client.py) but
# weren't wired into this report until now.
_RETRY_EVENT_FAMILIES = [
    ("llm_generation_retrying", "llm_generation_completed", "LLM generation"),
    ("web_search_retrying", "web_search_completed", "web search"),
    ("embed_query_retrying", "embed_query_completed", "query embedding"),
    ("vision_request_retrying", "vision_diagnosis_completed", "vision/diagnose"),
]


def _correlate_retry_outcomes(
    records: list[dict], retry_message: str, completed_message: str
) -> tuple[set[str], set[str], int]:
    """Correlate one call family's retry events to their eventual outcome
    by request_id (set by the request middleware on every log line). A
    call that logged >=1 retry and >=1 matching completion counts as a
    retry success; retried with no completion counts as a retry failure.
    Records without a request_id (off-request calls) fall back to a time
    window — a completion within RETRY_WINDOW_SECONDS of the retry counts
    as its outcome. Returns (success_ids, failure_ids, retry_event_count).
    """
    retries = [record for record in records if record.get("message") == retry_message]
    completed = [record for record in records if record.get("message") == completed_message]
    if not retries:
        return set(), set(), 0

    completed_ids = {record.get("request_id") for record in completed if record.get("request_id")}
    success_ids: set[str] = set()
    failure_ids: set[str] = set()

    for record in retries:
        request_id = record.get("request_id")
        if request_id:
            (success_ids if request_id in completed_ids else failure_ids).add(request_id)
        else:
            retry_time = record.get("timestamp")
            completed_time = next(
                (c.get("timestamp") for c in completed if _within_window(retry_time, c.get("timestamp"))),
                None,
            )
            (success_ids if completed_time else failure_ids).add(f"unid:{retry_time or 'unknown'}")

    return success_ids, failure_ids, len(retries)


def report_retry_success_rate(records: list[dict]) -> None:
    """Retry Success Rate across every retry-capable call this app makes
    (see _RETRY_EVENT_FAMILIES) — both overall and broken down per call
    type, since a family with a much lower rate than the others is
    exactly what the breakdown is for catching.

    The checklist's Retry Success Rate: the fraction of retried calls
    that eventually succeeded.
    """
    print("=== Retry Success Rate ===")

    all_success: set[str] = set()
    all_failure: set[str] = set()
    total_retry_events = 0
    any_retries = False

    for retry_message, completed_message, label in _RETRY_EVENT_FAMILIES:
        success_ids, failure_ids, retry_count = _correlate_retry_outcomes(
            records, retry_message, completed_message
        )
        total_retry_events += retry_count
        if retry_count == 0:
            continue
        any_retries = True
        # Namespaced by family before merging into the overall totals — a
        # single request can retry more than one kind of call, so the
        # same request_id could otherwise collide across families.
        all_success |= {f"{retry_message}:{rid}" for rid in success_ids}
        all_failure |= {f"{retry_message}:{rid}" for rid in failure_ids}
        family_total = len(success_ids) + len(failure_ids)
        family_rate = len(success_ids) / family_total if family_total else 0.0
        print(
            f"  {label:<15s}: {retry_count} retry events, {family_total} retried calls, "
            f"rate {family_rate:.4f} ({len(success_ids)}/{family_total})"
        )

    if not any_retries:
        print("No retry events found in the logs.\n")
        return

    total = len(all_success) + len(all_failure)
    success_rate = len(all_success) / total if total else 0.0
    print(f"  total retry events:       {total_retry_events}")
    print(f"  total retried calls:      {total}")
    print(f"  Retry Success Rate:       {success_rate:.4f} ({success_rate * 100:.1f}%)")
    print()


RETRY_WINDOW_SECONDS = 60.0


def _within_window(a: str | None, b: str | None) -> bool:
    """True if ISO timestamps a and b are within RETRY_WINDOW_SECONDS.

    String comparison is unsafe across offsets, so parse both to epoch
    seconds. Any parse failure returns False (conservative: no match).
    """
    if not a or not b:
        return False
    try:
        a_ts = datetime.fromisoformat(a.replace("Z", "+00:00")).timestamp()
        b_ts = datetime.fromisoformat(b.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return False
    return abs(a_ts - b_ts) <= RETRY_WINDOW_SECONDS


def report_timeout_rate(records: list[dict]) -> None:
    """Timeout Rate = Timed-Out Calls / Total Calls, over the same
    tool_invocation event stream Tool/API Success Rate is computed from
    (see tool_registry.track_tool).

    Two signal sources, since a timeout doesn't always mean the call
    failed:
    - tool_invocation records with timed_out=True (see
      tool_registry._looks_like_timeout) — a tool call that raised, and
      the underlying error looks like a timeout.
    - retrieval_timed_out events — retrieve()'s own timeout
      (retrieval_service._search_with_timeout) degrades to an empty
      result list rather than raising, so it's still a *successful*
      tool_invocation and wouldn't otherwise be counted; this is the only
      way that case is visible at all.

    Total Calls is every tool_invocation, matching the denominator
    Tool/API Success Rate already uses, so the two rates stay comparable.
    """
    tool_calls = [record for record in records if record.get("message") == "tool_invocation"]
    retrieval_timeouts = [record for record in records if record.get("message") == "retrieval_timed_out"]

    print("=== Timeout Rate ===")
    if not tool_calls:
        print("No tool_invocation entries found in the logs.\n")
        return

    timed_out_tool_calls = sum(1 for record in tool_calls if record.get("timed_out") is True)
    timed_out_calls = timed_out_tool_calls + len(retrieval_timeouts)
    total_calls = len(tool_calls)
    timeout_rate = timed_out_calls / total_calls if total_calls else 0.0

    by_tool: dict[str, int] = defaultdict(int)
    for record in tool_calls:
        if record.get("timed_out") is True:
            by_tool[record.get("tool", "unknown")] += 1
    if retrieval_timeouts:
        by_tool["retrieval"] += len(retrieval_timeouts)

    print(f"  total calls:              {total_calls}")
    print(f"  timed-out calls:          {timed_out_calls}")
    print(f"  by tool:                  {dict(by_tool) or 'none'}")
    print(f"  Timeout Rate:             {timeout_rate:.4f} ({timeout_rate * 100:.1f}%)")
    print()


def report_agent_workflow_metrics(records: list[dict]) -> None:
    """Phase 1 explicit-workflow metrics, parsed from the same structured
    log stream as everything else in this file — no new logging needed:

    - agent_node_trace (agent_graph/events.py::emit_node_trace, one per
      node execution): status/latency_ms feed Node Success Rate and
      Average Node Latency.
    - request_validated (agent_graph/nodes.py::validate_request_node, one
      per workflow run — including runs that fail validation): the total-
      workflow-runs denominator for Workflow Completion Rate / Loop Rate /
      Average Steps.
    - finalized (agent_graph/nodes.py::finalizer_node): termination_reason
      == "success" is a completed workflow. Note a cache-hit short-circuits
      straight to END without calling finalizer_node, so it's absent from
      this specific event but still counted as a completed run via its own
      request_validated + the graph's completed/failed workflow tracking
      in the live metrics registry (GET /metrics) — this offline log-based
      view is a secondary, inspectable cross-check, not the source of
      truth for a live deployment.
    - graph_cycle_capped_max_steps (agent_graph/engine.py): one per
      workflow run that hit max_steps without reaching END — Loop Rate's
      numerator.
    """
    node_traces = [record for record in records if record.get("message") == "agent_node_trace"]
    validated = [record for record in records if record.get("message") == "request_validated"]
    finalized = [record for record in records if record.get("message") == "finalized"]
    loop_capped = [
        record for record in records if record.get("message") == "graph_cycle_capped_max_steps"
    ]

    print("=== Explicit Workflow Metrics (agent_graph) ===")
    if not node_traces and not validated:
        print("No agent_graph log entries found in the logs.\n")
        return

    total_workflows = len(validated)
    successful_workflows = sum(1 for record in finalized if record.get("termination_reason") == "success")
    workflow_completion_rate = successful_workflows / total_workflows if total_workflows else 0.0

    node_success = sum(1 for record in node_traces if record.get("status") == "success")
    node_total = len(node_traces)
    node_success_rate = node_success / node_total if node_total else 0.0

    latencies = [
        record.get("latency_ms", 0.0)
        for record in node_traces
        if isinstance(record.get("latency_ms"), int | float)
    ]
    avg_node_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    loop_rate = len(loop_capped) / total_workflows if total_workflows else 0.0

    by_node: dict[str, list[float]] = defaultdict(list)
    node_status: dict[str, list[bool]] = defaultdict(list)
    for record in node_traces:
        node = record.get("node", "unknown")
        if isinstance(record.get("latency_ms"), int | float):
            by_node[node].append(record["latency_ms"])
        node_status[node].append(record.get("status") == "success")

    print(f"  workflow runs:            {total_workflows}")
    print(f"  Workflow Completion Rate: {workflow_completion_rate:.4f} ({successful_workflows}/{total_workflows})")
    print(f"  Node Success Rate:        {node_success_rate:.4f} ({node_success}/{node_total})")
    print(f"  Average Node Latency:     {avg_node_latency_ms:.2f}ms  (n={node_total})")
    print(f"  Loop Rate:                {loop_rate:.4f} ({len(loop_capped)}/{total_workflows})")
    print("  Per-node breakdown:")
    for node in sorted(by_node):
        node_lat = by_node[node]
        statuses = node_status[node]
        node_rate = sum(statuses) / len(statuses) if statuses else 0.0
        avg_lat = sum(node_lat) / len(node_lat) if node_lat else 0.0
        print(
            f"    {node:<22s} success={node_rate:.4f}  avg_latency={avg_lat:.2f}ms  (n={len(statuses)})"
        )
    print()


def report_tokens_and_cost(records: list[dict]) -> None:
    generations = [
        record for record in records if record.get("message") == "llm_generation_completed"
    ]

    print("=== Token usage & estimated cost ===")
    if not generations:
        print("No llm_generation_completed entries found in the logs.\n")
        return

    prompt_tokens = sum(record.get("prompt_tokens", 0) or 0 for record in generations)
    completion_tokens = sum(record.get("completion_tokens", 0) or 0 for record in generations)
    total_tokens = sum(record.get("total_tokens", 0) or 0 for record in generations)
    total_cost = sum(record.get("estimated_cost_usd", 0) or 0 for record in generations)

    print(f"  generations:        {len(generations)}")
    print(f"  prompt tokens:      {prompt_tokens}")
    print(f"  completion tokens:  {completion_tokens}")
    print(f"  total tokens:       {total_tokens}")
    print(f"  estimated cost:     ${total_cost:.6f}")
    print()


def report_acceptance_rate(feedback_path: Path) -> None:
    """Acceptance Rate = thumbs-up ÷ total feedback events — the LLMOps
    acceptance-rate metric. Reads backend/feedback/feedback.jsonl, one
    JSON feedback event per line (see app/services/feedback_service.py)."""
    print("=== Acceptance Rate (chat feedback) ===")
    if not feedback_path.is_file():
        print(f"No feedback file found at {feedback_path}.\n")
        return

    lines = feedback_path.read_text(encoding="utf-8").splitlines()
    events = parse_log_lines(lines)
    ratings = [event.get("rating") for event in events]
    ups = sum(1 for rating in ratings if rating == "up")
    downs = sum(1 for rating in ratings if rating == "down")
    total = ups + downs

    if total == 0:
        print("No feedback events found.\n")
        return

    acceptance_rate = ups / total
    print(f"  up:               {ups}")
    print(f"  down:             {downs}")
    print(f"  total:            {total}")
    print(f"  Acceptance Rate:  {acceptance_rate:.4f} ({acceptance_rate * 100:.1f}%)")
    print()


# The seven answer-quality-checklist criteria a reviewer can score 1-5 on
# (see RubricScores, app/models/schemas.py) — kept in one place so the
# two report functions below can't drift on which fields they expect.
RUBRIC_CRITERIA = [
    "correctness",
    "helpfulness",
    "completeness",
    "safety",
    "tone",
    "groundedness",
    "citation_quality",
]


def report_rubric_scores(feedback_path: Path) -> None:
    """Per-criterion average of the optional 1-5 rubric a reviewer can
    attach to feedback (see RubricScores). Only rubric-bearing events
    contribute — plain thumbs-up/down feedback has no rubric and is
    covered by Acceptance Rate instead.
    """
    print("=== Rubric Scores (1-5 per criterion) ===")
    if not feedback_path.is_file():
        print(f"No feedback file found at {feedback_path}.\n")
        return

    lines = feedback_path.read_text(encoding="utf-8").splitlines()
    events = parse_log_lines(lines)
    rubrics = [event["rubric"] for event in events if event.get("rubric")]

    if not rubrics:
        print("No rubric-bearing feedback events found.\n")
        return

    for criterion in RUBRIC_CRITERIA:
        scores = [rubric[criterion] for rubric in rubrics if criterion in rubric]
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        print(f"  {criterion:<18s}: {avg:.2f} / 5  (n={len(scores)})")
    print(f"  rubric reviews:     {len(rubrics)}")
    print()


def report_inter_annotator_agreement(feedback_path: Path) -> None:
    """Inter-Annotator Agreement — the checklist's "Basic agreement":
    Agreement Rate = Matching Reviewer Judgments / Total Judgments.

    Operationalized as pairwise percent agreement on `rating` (up/down):
    for every pair of distinct reviewers (by reviewer_id) who both judged
    the same message_id, one "judgment comparison" is whether their
    ratings matched. Agreement Rate = matching pairs / total pairs —
    the standard percent-agreement definition of inter-rater reliability.

    reviewer_id is resolved server-side from the authenticated caller
    (see query.py's submit_feedback), never client-supplied — events
    without one (feedback submitted before rubric support existed, or
    with no resolvable identity) are excluded, since "who agrees with
    whom" is undefined without knowing who "who" is. A message_id judged
    by only one reviewer contributes no pairs either — agreement is
    inherently about *multiple* reviewers of the same output.
    """
    print("=== Inter-Annotator Agreement ===")
    if not feedback_path.is_file():
        print(f"No feedback file found at {feedback_path}.\n")
        return

    lines = feedback_path.read_text(encoding="utf-8").splitlines()
    events = parse_log_lines(lines)

    # message_id -> {reviewer_id: latest rating} — a reviewer who
    # re-submits feedback on the same message overwrites their own
    # earlier judgment rather than counting twice.
    by_message: dict[str, dict[str, str]] = defaultdict(dict)
    for event in events:
        reviewer_id = event.get("reviewer_id")
        message_id = event.get("message_id")
        rating = event.get("rating")
        if not reviewer_id or not message_id or not rating:
            continue
        by_message[message_id][reviewer_id] = rating

    matching_pairs = 0
    total_pairs = 0
    multi_reviewer_messages = 0
    for reviewer_ratings in by_message.values():
        if len(reviewer_ratings) < 2:
            continue
        multi_reviewer_messages += 1
        ratings = list(reviewer_ratings.values())
        for i in range(len(ratings)):
            for j in range(i + 1, len(ratings)):
                total_pairs += 1
                if ratings[i] == ratings[j]:
                    matching_pairs += 1

    if total_pairs == 0:
        print(
            "No message judged by 2+ distinct reviewers found — Inter-Annotator "
            "Agreement needs multiple reviewers on the same output.\n"
        )
        return

    agreement_rate = matching_pairs / total_pairs
    print(f"  messages with 2+ reviewers: {multi_reviewer_messages}")
    print(f"  reviewer pairs compared:    {total_pairs}")
    print(f"  matching pairs:             {matching_pairs}")
    print(f"  Agreement Rate:             {agreement_rate:.4f} ({agreement_rate * 100:.1f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_file",
        nargs="?",
        help="Path to a JSON log file (one JSON object per line). Reads stdin if omitted.",
    )
    parser.add_argument(
        "--feedback-file",
        default=str(DEFAULT_FEEDBACK_PATH),
        help=f"Path to the feedback JSONL file (default: {DEFAULT_FEEDBACK_PATH}).",
    )
    args = parser.parse_args()

    if args.log_file:
        path = Path(args.log_file)
        if not path.is_file():
            raise SystemExit(f"Log file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = sys.stdin.readlines()

    records = parse_log_lines(lines)
    if not records:
        raise SystemExit("No parseable JSON log lines found.")

    print(f"Parsed {len(records)} log lines.\n")
    report_latency(records)
    report_error_rate_by_category(records)
    report_loop_count_and_avg_steps(records)
    report_agent_workflow_metrics(records)
    report_retry_success_rate(records)
    report_timeout_rate(records)
    report_tokens_and_cost(records)
    report_acceptance_rate(Path(args.feedback_file))
    report_rubric_scores(Path(args.feedback_file))
    report_inter_annotator_agreement(Path(args.feedback_file))


if __name__ == "__main__":
    main()
