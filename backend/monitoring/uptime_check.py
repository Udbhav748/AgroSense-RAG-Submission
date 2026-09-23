"""Availability / uptime checker for the deployed services.

Polls the live backend (/health) and frontend (/) and reports reachability,
status code, and latency. Designed to be run periodically (cron, GitHub
Actions schedule, or a local loop) so a "down" event is visible instead of
silent — a lightweight stand-in for a hosted uptime monitor, which the
Render free tier doesn't include.

Usage:
    python monitoring/uptime_check.py                    # once, defaults
    python monitoring/uptime_check.py --loop 300 --json  # every 300s, JSON lines
    python monitoring/uptime_check.py --backend https://agrosense-rag-backend.onrender.com

Exit code is 0 if all targets are healthy, 1 if any check failed (so a
scheduled job can alert on a non-zero exit).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# Same reasoning as dashboard.py's identical line: needed whether this
# runs as a standalone script (`python monitoring/uptime_check.py` puts
# monitoring/ on sys.path, not backend/) or gets imported under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.alert_webhook import send_alert  # noqa: E402

DEFAULT_BACKEND = "http://localhost:8000"
DEFAULT_FRONTEND = "http://localhost:5173"

# Persists the running availability percentage across checks so a scheduled
# run can report trailing availability rather than only the last probe.
STATE_FILE = Path(__file__).parent / "uptime_state.json"

TIMEOUT_SECONDS = 15


def _probe(url: str) -> tuple[bool, int | None, float]:
    """Returns (healthy, status_code, latency_ms)."""
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
            latency = (time.monotonic() - start) * 1000
            return 200 <= resp.status < 400, resp.status, latency
    except urllib.error.HTTPError as exc:
        latency = (time.monotonic() - start) * 1000
        # 4xx/5xx are "reachable but failing"; count them as unhealthy.
        return False, exc.code, latency
    except Exception:
        latency = (time.monotonic() - start) * 1000
        return False, None, latency


def _load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        # State persistence is best-effort; a read-only FS shouldn't fail the check.
        pass


def check_all(backend: str, frontend: str) -> dict:
    state = _load_state()
    now = datetime.now(UTC).isoformat()

    targets = {"backend": backend, "frontend": frontend}
    results = {}
    all_healthy = True
    for name, url in targets.items():
        healthy, status, latency = _probe(url)
        all_healthy = all_healthy and healthy
        # Trailing availability = healthy probes / total probes, persisted
        # across invocations so scheduled runs accumulate a real percentage.
        record = state.get(name, {"healthy": 0, "total": 0})
        record["total"] += 1
        if healthy:
            record["healthy"] += 1
        availability = record["healthy"] / record["total"] if record["total"] else 0.0
        state[name] = record
        results[name] = {
            "url": url,
            "healthy": healthy,
            "status_code": status,
            "latency_ms": round(latency, 1),
            "trailing_availability": round(availability, 4),
        }

    _save_state(state)
    return {
        "checked_at": now,
        "all_healthy": all_healthy,
        "targets": results,
    }


def _print_human(report: dict) -> None:
    print(f"[{report['checked_at']}] overall: {'HEALTHY' if report['all_healthy'] else 'DEGRADED'}")
    for name, t in report["targets"].items():
        status = "ok" if t["healthy"] else f"FAIL ({t['status_code'] or 'no-response'})"
        print(
            f"  {name:>9}: {status}  latency={t['latency_ms']:>7.1f}ms  "
            f"availability={t['trailing_availability']*100:.1f}%"
        )


def _alert_text(report: dict) -> str:
    down = [
        f"{name} ({t['status_code'] or 'no-response'})"
        for name, t in report["targets"].items()
        if not t["healthy"]
    ]
    return f"AgroSense-RAG uptime alert: {', '.join(down)} unreachable at {report['checked_at']}."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("BACKEND_URL", DEFAULT_BACKEND), help=f"Backend base URL (default: {DEFAULT_BACKEND}).")
    parser.add_argument("--frontend", default=os.environ.get("FRONTEND_URL", DEFAULT_FRONTEND), help=f"Frontend URL (default: {DEFAULT_FRONTEND}).")
    parser.add_argument("--loop", type=int, default=0, help="Poll every N seconds instead of once. 0 = run once.")
    parser.add_argument("--json", action="store_true", help="Emit one JSON line per check instead of human text.")
    parser.add_argument(
        "--alert-webhook-url",
        default=os.environ.get("ALERT_WEBHOOK_URL"),
        help="Slack-compatible webhook URL to POST to when a target is down. Unset = no-op.",
    )
    args = parser.parse_args()

    while True:
        report = check_all(args.backend, args.frontend)
        if args.json:
            print(json.dumps(report))
            sys.stdout.flush()
        else:
            _print_human(report)
        if not report["all_healthy"]:
            # Best-effort, un-debounced: a --loop run that stays down
            # alerts on every iteration, not just the first. Accepted
            # for the same reason the rest of this package stays simple
            # — see monitoring/README.md's Known Limitations. A single
            # invocation (e.g. from a CI schedule) only runs once anyway,
            # so this only matters for a long-lived local --loop.
            send_alert(args.alert_webhook_url, _alert_text(report))
        if args.loop <= 0:
            break
        time.sleep(args.loop)

    sys.exit(0 if report["all_healthy"] else 1)


if __name__ == "__main__":
    main()
