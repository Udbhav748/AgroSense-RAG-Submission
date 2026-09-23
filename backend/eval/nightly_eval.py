"""Nightly evaluation wrapper: run the eval harness, gate against the
committed baseline, and emit a machine-readable summary.

This is the CI-facing entry point for the scheduled nightly eval job (see
.github/workflows/nightly-eval.yml) and the opt-in startup hook
(NIGHTLY_EVAL_ENABLED=true). It wires the existing pieces together:

1. eval/run_eval.py — runs every dataset entry through ChatService and
   produces the full result report (JSON, saved under eval/results/).
2. eval/regression_check.py — compares the fresh report against a
   committed baseline and fails (non-zero exit) on any tracked metric
   regressing beyond tolerance.
3. A short summary report (JSON) written to eval/results/latest.json,
   capturing the headline metrics and whether the gate passed — the
   artifact the scheduled workflow surfaces.

Exit codes (matching regression_check.py): 0 = no regression, 1 =
regression detected, 2 = usage/input error.

Usage (from backend/):
    python eval/nightly_eval.py
    python eval/nightly_eval.py --dataset dataset_v2.json --baseline eval/baselines/v2_groq.json
    python eval/nightly_eval.py --tol 0.03 --delay 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings

_EVAL_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _EVAL_DIR / "results"

# Headline metrics the summary report captures — the same set the Grafana
# dashboards and the regression gate care about, kept in one place so the
# summary stays in sync with what's actually tracked.
HEADLINE_METRICS = [
    "task_success_rate",
    "groundedness_proxy",
    "entailment_groundedness",
    "injection_resistance",
    "false_refusal_rate",
    "data_leak_rate",
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "hit_rate_at_5",
    "citation_accuracy",
    "tool_arg_accuracy",
]


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_EVAL_DIR.parent))
    return result.returncode


def _latest_result() -> Path:
    """Newest run_eval.py result JSON under eval/results/ (excludes the
    latest.json summary itself)."""
    results = [p for p in _RESULTS_DIR.glob("*.json") if p.name != "latest.json"]
    if not results:
        return _RESULTS_DIR / "latest.json"
    return max(results, key=lambda p: p.stat().st_mtime)


def _summarize(report: dict, baseline: dict | None, gate_passed: bool) -> dict:
    summary = {
        "run_at": datetime.now(UTC).isoformat(),
        "dataset": report.get("dataset"),
        "dataset_version": report.get("dataset_version"),
        "llm_provider": report.get("llm_provider"),
        "llm_model_name": report.get("llm_model_name"),
        "prompt_version": report.get("prompt_version"),
        "hybrid_search_enabled": report.get("hybrid_search_enabled"),
        "reranking_enabled": report.get("reranking_enabled"),
        "n_entries": report.get("n_entries"),
        "gate_passed": gate_passed,
        "baseline": baseline.get("dataset_version") if baseline else None,
        "metrics": {},
    }
    for key in HEADLINE_METRICS:
        summary["metrics"][key] = report.get(key)
        if baseline:
            summary["metrics"][f"{key}_baseline"] = baseline.get(key)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset filename inside backend/eval/ (default: Settings.nightly_eval_dataset).",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline JSON (relative to backend/) to gate against (default: Settings.nightly_eval_baseline).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Absolute regression tolerance on the 0-1 scale (default: Settings.nightly_eval_tolerance).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between dataset entries (pass-through to run_eval.py).",
    )
    parser.add_argument(
        "--skip-regression-gate",
        action="store_true",
        help="Run the eval but skip the baseline regression gate (e.g. for a baseline refresh).",
    )
    args = parser.parse_args()

    dataset = args.dataset or settings.nightly_eval_dataset
    baseline_path = args.baseline or settings.nightly_eval_baseline
    tol = args.tol if args.tol is not None else settings.nightly_eval_tolerance

    dataset_path = _EVAL_DIR / dataset
    if not dataset_path.is_file():
        print(f"nightly_eval: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(2)
    baseline_full = _EVAL_DIR.parent / baseline_path
    baseline = None
    if not args.skip_regression_gate:
        if not baseline_full.is_file():
            print(f"nightly_eval: baseline not found: {baseline_full}", file=sys.stderr)
            sys.exit(2)
        try:
            baseline = json.loads(baseline_full.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"nightly_eval: baseline is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(2)

    run_rc = _run(
        ["python", "eval/run_eval.py", "--dataset", dataset, "--delay", str(args.delay)]
    )
    if run_rc != 0:
        print("nightly_eval: eval harness failed.", file=sys.stderr)
        sys.exit(run_rc)

    newest = _latest_result()
    report = json.loads(newest.read_text(encoding="utf-8"))

    if args.skip_regression_gate or baseline is None:
        gate_rc = 0
    else:
        gate_rc = _run(
            [
                "python",
                "eval/regression_check.py",
                "--results",
                str(newest),
                "--baseline",
                baseline_path,
                "--tol",
                str(tol),
            ]
        )

    summary = _summarize(report, baseline, gate_rc == 0)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _RESULTS_DIR / "latest.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nnightly_eval summary written to {summary_path}")

    if gate_rc == 0:
        print("NIGHTLY EVAL GATE: PASSED")
    else:
        print("NIGHTLY EVAL GATE: FAILED — see regressions above.", file=sys.stderr)
    sys.exit(gate_rc)


if __name__ == "__main__":
    main()
