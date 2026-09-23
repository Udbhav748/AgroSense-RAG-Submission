#!/usr/bin/env python
"""Orchestrates every module10 runner and backend/eval/regression_check.py
into one full_audit_<ts>.json.

Usage (from backend/):
    python eval/module10/runners/run_full_module10.py
    python eval/module10/runners/run_full_module10.py --skip-live
        (runs only run_failure_eval.py, the one fully-offline runner)

This calls out to each runner as a subprocess (same pattern as
backend/eval/nightly_eval.py's own subprocess orchestration of run_eval.py)
rather than importing and re-running their logic in-process, so a crash in
one runner doesn't take down the others and each still writes its own
independent timestamped artifact in reports/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.module10 import config  # noqa: E402

RUNNERS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = RUNNERS_DIR.parents[2]


def _run(script: str) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(RUNNERS_DIR / script)],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    ok = result.returncode == 0
    output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    return ok, output


def _latest_report(prefix: str) -> dict | None:
    matches = sorted(config.REPORTS_DIR.glob(f"{prefix}_*.json"))
    if not matches:
        return None
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-live", action="store_true", help="Only run the offline failure-injection runner")
    args = parser.parse_args()

    runs = [("run_failure_eval.py", "failure_eval")]
    if not args.skip_live:
        runs = [
            ("run_rag_eval.py", "rag_eval"),
            ("run_agent_eval.py", "agent_eval"),
            ("run_security_eval.py", "security_eval"),
            ("run_memory_eval.py", "memory_eval"),
            *runs,
        ]

    outcomes = {}
    for script, prefix in runs:
        print(f"=== Running {script} ===")
        ok, output = _run(script)
        print(output[-2000:])
        outcomes[prefix] = {"ok": ok, "report": _latest_report(prefix) if ok else None}
        if not ok:
            print(f"!!! {script} FAILED (exit non-zero) — see output above. Continuing with remaining runners.")

    full_report = {
        "metadata": config.run_metadata(sample_count=sum(1 for o in outcomes.values() if o["ok"])),
        "runner_outcomes": {name: o["ok"] for name, o in outcomes.items()},
        "reports": {name: o["report"] for name, o in outcomes.items()},
    }
    path = config.save_report(full_report, name="full_audit")
    print(f"\nFull audit saved: {path}")
    for name, o in outcomes.items():
        print(f"  {name}: {'OK' if o['ok'] else 'FAILED'}")


if __name__ == "__main__":
    main()
