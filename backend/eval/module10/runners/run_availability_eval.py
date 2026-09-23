#!/usr/bin/env python
"""Module 10 gap-closure (P6), TASK 4: a reproducible BOUNDED LOCAL
SERVICE AVAILABILITY MEASUREMENT.

Starts the real FastAPI app as a real `uvicorn` subprocess bound to
localhost (not TestClient -- an actual running HTTP server, so this
exercises the real socket/process/startup path a deployed instance
would), then issues repeated real HTTP GET /health probes over a
bounded interval, and computes:

    Availability = successful probes / total probes

This is explicitly NOT "production availability" and is never labeled
as such -- there is no persistent production deployment for this
project (see docs/CHECKLIST.md's Availability row). It is a short
smoke/validation measurement of THIS machine's local service process
over a bounded window, useful for confirming the health endpoint and
probe mechanics work end to end -- not an SLO, and not extrapolated
into one.

Deliberately reuses monitoring.uptime_check._probe() (the same probe
function the real uptime checker uses) rather than reimplementing HTTP
polling, so this measurement and the "real" tool stay consistent. It
does NOT call monitoring.uptime_check.check_all(), since that persists
trailing state into monitoring/uptime_state.json -- this bounded local
run must not corrupt that file's real trailing-availability bookkeeping
with test noise.

Usage (from backend/):
    python eval/module10/runners/run_availability_eval.py
    python eval/module10/runners/run_availability_eval.py --duration-seconds 30 --interval-seconds 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from monitoring.uptime_check import _probe  # noqa: E402
from eval.module10 import config  # noqa: E402

DEFAULT_PORT = 8811
DEFAULT_DURATION_SECONDS = 20
DEFAULT_INTERVAL_SECONDS = 1.0
STARTUP_TIMEOUT_SECONDS = 90


def _wait_for_startup(url: str, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        healthy, _, _ = _probe(url)
        if healthy:
            return True
        time.sleep(0.5)
    return False


def run_availability_measurement(
    *, port: int, duration_seconds: float, interval_seconds: float
) -> dict:
    backend_root = Path(__file__).resolve().parents[3]
    health_url = f"http://127.0.0.1:{port}/health"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(backend_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        started = _wait_for_startup(health_url, STARTUP_TIMEOUT_SECONDS)
        if not started:
            return {
                "error": "The local uvicorn process did not become healthy within "
                f"{STARTUP_TIMEOUT_SECONDS}s -- no probes were run.",
                "started": False,
            }

        probes: list[dict] = []
        deadline = time.monotonic() + duration_seconds
        while time.monotonic() < deadline:
            healthy, status_code, latency_ms = _probe(health_url)
            probes.append({"healthy": healthy, "status_code": status_code, "latency_ms": round(latency_ms, 2)})
            time.sleep(interval_seconds)

        total = len(probes)
        successful = sum(1 for p in probes if p["healthy"])
        failed = total - successful
        latencies = [p["latency_ms"] for p in probes]

        return {
            "started": True,
            "endpoint_tested": health_url,
            "total_probes": total,
            "successful_probes": successful,
            "failed_probes": failed,
            "availability": round(successful / total, 4) if total else None,
            "test_duration_seconds": duration_seconds,
            "probe_interval_seconds": interval_seconds,
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "min": round(min(latencies), 2) if latencies else None,
                "max": round(max(latencies), 2) if latencies else None,
            },
            "per_probe": probes,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    args = parser.parse_args()

    result = run_availability_measurement(
        port=args.port, duration_seconds=args.duration_seconds, interval_seconds=args.interval_seconds
    )

    report = {
        "metadata": config.run_metadata(sample_count=result.get("total_probes", 0), dataset_version="availability_v1"),
        "label": "bounded local service availability measurement",
        "methodology": (
            "A real `uvicorn app.main:app` subprocess is started on localhost, then GET /health is "
            "probed repeatedly over a bounded local wall-clock window using the same "
            "monitoring.uptime_check._probe() function the real uptime checker uses. "
            "Availability = successful probes / total probes. This is NOT production availability -- "
            "there is no persistent production deployment for this project -- and this short window is "
            "a smoke/validation measurement, not an SLO."
        ),
        **result,
        "limitations": [
            "Single machine, single short local run -- not a distributed or long-running SLO "
            "measurement.",
            "Measures only the health endpoint's reachability/latency, not full request-path "
            "availability under real traffic (see the observability report's error_rate for that).",
            "The uvicorn process, OS, and network stack are all local to this machine -- results do "
            "not generalize to a real hosted deployment's availability characteristics.",
        ],
    }

    path = config.save_report(report, name="availability_bounded_local")
    print(f"Saved: {path}")
    if result.get("started"):
        print(f"Availability: {result['availability']} ({result['successful_probes']}/{result['total_probes']} probes)")
    else:
        print(f"FAILED TO START: {result.get('error')}")


if __name__ == "__main__":
    main()
