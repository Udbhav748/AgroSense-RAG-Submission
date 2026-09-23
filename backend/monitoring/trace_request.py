"""Reconstructs and prints the full step-by-step trace for one request,
from the backend's JSON logs.

Every log line already carries request_id (set by the request-id
middleware, read by app.core.logging.JSONFormatter — see
app/core/request_context.py) and a timestamp, so a complete per-request
trace is *reconstructable* from the raw log file. This is the tool that
actually reconstructs it — the data existed (planning, retrieval,
grading, tool calls, generation, and any agent handoffs all already log
one event each, tagged with the same request_id), but nothing before
this let a developer ask "show me everything that happened for this one
request, in order" as a single command; the alternative was hand-grep
and manually sort by timestamp.

Usage (from backend/):
    python monitoring/trace_request.py app.log <request_id>
    python monitoring/trace_request.py app.log <request_id> --json

    # Don't know the request_id? List recent requests first:
    python monitoring/trace_request.py app.log --list
    python monitoring/trace_request.py app.log --list -n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Make the backend root importable regardless of how this script is run
# (`python monitoring/trace_request.py` puts monitoring/ on sys.path, not
# backend/). Reuses log_aggregate's file loader so the two tools can
# never drift on how a log line is parsed — the same pattern
# dashboard.py already uses for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.log_aggregate import _load_records  # noqa: E402

# Fields already surfaced structurally (as the trace's per-event header)
# — not worth repeating in that event's own detail dump.
_SUPPRESSED_FIELDS = {"timestamp", "level", "message", "request_id", "logger"}

# One request's outcome is marked by exactly one of these — what --list
# enumerates, so it shows one row per request, not one per log line.
_OUTCOME_MESSAGES = {"chat_query_handled", "request_failed", "diagnose_response_sent"}

_EPOCH = datetime.min.replace(tzinfo=UTC)


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sort_key(record: dict) -> datetime:
    # A record with a missing/unparseable timestamp sorts first rather
    # than being dropped — still visible, just flagged by its position.
    return _parse_timestamp(record.get("timestamp")) or _EPOCH


def trace(records: list[dict], request_id: str) -> list[dict]:
    """Every record for request_id, in chronological order."""
    matching = [r for r in records if r.get("request_id") == request_id]
    return sorted(matching, key=_sort_key)


def print_trace(records: list[dict], request_id: str) -> None:
    events = trace(records, request_id)

    print(f"=== Trace for request_id={request_id} ===")
    if not events:
        print("No log records found for this request_id.\n")
        return

    first_ts = _parse_timestamp(events[0].get("timestamp"))
    last_ts = _parse_timestamp(events[-1].get("timestamp"))
    duration_note = ""
    if first_ts is not None and last_ts is not None:
        duration_note = f", {(last_ts - first_ts).total_seconds() * 1000:.1f}ms total"
    print(f"{len(events)} events{duration_note}\n")

    for index, record in enumerate(events, 1):
        timestamp = record.get("timestamp", "?")
        message = record.get("message", "?")
        print(f"[{index:>2}] {timestamp}  {message}")
        for key, value in record.items():
            if key in _SUPPRESSED_FIELDS:
                continue
            print(f"     {key}: {value}")
    print()


def list_requests(records: list[dict], limit: int) -> None:
    outcomes = [
        record for record in records if record.get("message") in _OUTCOME_MESSAGES and record.get("request_id")
    ]
    outcomes.sort(key=_sort_key, reverse=True)

    print(f"=== Recent requests (showing up to {limit}) ===")
    if not outcomes:
        print("No request_id-tagged outcome events found in the logs.\n")
        return

    for record in outcomes[:limit]:
        timestamp = record.get("timestamp", "?")
        request_id = record.get("request_id")
        message = record.get("message")
        detail = ""
        if message == "request_failed":
            detail = f"  ({record.get('taxonomy_category', 'unknown')})"
        print(f"  {timestamp}  {request_id}  {message}{detail}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfile", help="Path to a captured backend JSON log file.")
    parser.add_argument(
        "request_id", nargs="?", help="The request_id to trace. Omit with --list to browse recent requests."
    )
    parser.add_argument("--list", action="store_true", help="List recent request_ids instead of tracing one.")
    parser.add_argument("-n", "--limit", type=int, default=20, help="How many recent requests --list shows (default 20).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text (trace mode only).")
    args = parser.parse_args()

    path = Path(args.logfile)
    if not path.is_file():
        raise SystemExit(f"Log file not found: {path}")
    records = _load_records(str(path))

    if args.list:
        list_requests(records, args.limit)
        return

    if not args.request_id:
        raise SystemExit("Provide a request_id to trace, or use --list to browse recent requests.")

    if args.json:
        print(json.dumps({"request_id": args.request_id, "events": trace(records, args.request_id)}))
    else:
        print_trace(records, args.request_id)


if __name__ == "__main__":
    main()
