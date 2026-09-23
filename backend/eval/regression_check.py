"""Regression gate for eval results.

Compares a freshly-produced eval result JSON (run_eval.py output) against a
committed baseline and fails (exit code 1) if any tracked metric regressed
beyond tolerance. Designed to run in CI (eval.yml) after `run_eval.py` so a
change that degrades answer quality can't merge silently.

Usage (from backend/):
    python eval/regression_check.py --results eval/results/<new>.json \
        --baseline eval/baselines/v1_v2_groq.json

Two distinct checks, both must pass:

1. Aggregate metrics (when present in both files):

    Higher-is-better (regression = value dropped by more than tol):
        planner.accuracy, planner.macro_f1, planner.weighted_f1,
        task_success_rate, groundedness_proxy, entailment_groundedness,
        injection_resistance, source_accuracy, precision_at_5, recall_at_5,
        mrr, hit_rate_at_5, citation_accuracy, tool_arg_accuracy

    Lower-is-better (regression = value rose by more than tol):
        false_refusal_rate, data_leak_rate, prompt_injection_success_rate

    A missing metric in either file is skipped (new metrics on a fresh run
    have no baseline yet; old runs predate them).

2. Case-level Regression Rate: Previously Passing Cases Now Failing /
   Previously Passing Cases. This is a genuinely different, finer-grained
   check than #1 — an aggregate can hold steady while individual cases
   flip (some regress, others newly pass, netting to "no visible change"),
   and #1 alone would never catch that. Entries are matched between the
   two files by query text (the natural stable key — dataset entries have
   no explicit id); an entry missing from either side isn't comparable and
   is skipped rather than counted either way. Gated by --max-regression-rate
   (default 0.0 — any previously-passing case failing now is a regression;
   raise it if some flakiness is expected and tolerable).

Tolerances are absolute-percentage for rates (0-1 scale) unless overridden
with --tol / --max-regression-rate. Exit codes:
    0 = no regression, 1 = regression detected, 2 = usage/input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HIGHER_IS_BETTER = [
    "task_success_rate",
    "groundedness_proxy",
    "entailment_groundedness",
    "injection_resistance",
    "source_accuracy",
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "hit_rate_at_5",
    "citation_accuracy",
    "tool_arg_accuracy",
    # Phase 1 explicit-workflow metrics (run_eval.py, from the live
    # agent_graph metrics registry) — tool_selection_accuracy is an alias
    # onto plan_execution_consistency (same value, PDF's name), tracked
    # once here since regressing one always regresses the other.
    "workflow_completion_rate",
    "node_success_rate",
    "tool_selection_accuracy",
]

LOWER_IS_BETTER = [
    "false_refusal_rate",
    "data_leak_rate",
    # Exact complement of injection_resistance (see run_eval.py) — tracked
    # separately here too since it's the checklist's literal metric name
    # and a rise here is the same regression injection_resistance's own
    # drop would already catch, just framed the other way.
    "prompt_injection_success_rate",
    # A rising loop rate means more requests are hitting the graph's
    # max_steps safety cap instead of completing normally — a workflow
    # health regression, not an answer-quality one, but tracked the same
    # way. average_node_latency_ms is deliberately NOT tracked here: it's
    # a performance signal that varies with provider/network conditions
    # this gate isn't meant to police, not a workflow-correctness one.
    "loop_rate",
]

# Top-level metrics nested under a "planner" object, same HIGHER_IS_BETTER
# semantics as the flat list above.
PLANNER_METRICS = ["accuracy", "macro_f1", "weighted_f1"]


def _failures(results: dict, baseline: dict, tol: float) -> list[str]:
    problems: list[str] = []
    for key in HIGHER_IS_BETTER:
        new = results.get(key)
        old = baseline.get(key)
        if new is None or old is None:
            continue
        if old - new > tol:
            problems.append(f"{key}: {new:.4f} (baseline {old:.4f}) — dropped > {tol}")
    for key in LOWER_IS_BETTER:
        new = results.get(key)
        old = baseline.get(key)
        if new is None or old is None:
            continue
        if new - old > tol:
            problems.append(f"{key}: {new:.4f} (baseline {old:.4f}) — rose > {tol}")
    new_planner = results.get("planner") or {}
    old_planner = baseline.get("planner") or {}
    for key in PLANNER_METRICS:
        new = new_planner.get(key)
        old = old_planner.get(key)
        if new is None or old is None:
            continue
        if old - new > tol:
            problems.append(f"planner.{key}: {new:.4f} (baseline {old:.4f}) — dropped > {tol}")
    return problems


def _case_level_regression(
    results: dict, baseline: dict, key_field: str = "query"
) -> tuple[int, list[str], float | None]:
    """Regression Rate: the fraction of cases that passed (task_success is
    True) in the baseline but don't pass (False, None, or missing) in the
    new run. Returns (previously_passing_count, now_failing_keys, rate) —
    rate is None when there's nothing to compare (no case in the baseline
    both passed and has a match in the new run).

    Matching is by query text. A case present in the baseline but absent
    from the new run (dataset changed) isn't comparable and is excluded
    from both the numerator and denominator, rather than being silently
    counted as either a pass or a regression.
    """
    new_by_key = {entry.get(key_field): entry for entry in (results.get("entries") or []) if entry.get(key_field)}

    previously_passing = 0
    now_failing: list[str] = []
    for entry in baseline.get("entries") or []:
        key = entry.get(key_field)
        if not key or entry.get("task_success") is not True:
            continue
        new_entry = new_by_key.get(key)
        if new_entry is None:
            continue
        previously_passing += 1
        if new_entry.get("task_success") is not True:
            now_failing.append(key)

    rate = len(now_failing) / previously_passing if previously_passing else None
    return previously_passing, now_failing, rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        help="New eval result JSON (run_eval.py output) to gate.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline result JSON to compare against.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Absolute regression tolerance on the 0-1 scale (default 0.05), for the aggregate-metric check.",
    )
    parser.add_argument(
        "--max-regression-rate",
        type=float,
        default=0.0,
        help="Max fraction of previously-passing cases allowed to now fail (default 0.0 — none allowed).",
    )
    args = parser.parse_args()

    try:
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"regression_check: could not load results/baseline: {exc}", file=sys.stderr)
        sys.exit(2)

    print(
        f"Comparing {args.results} vs baseline {args.baseline} "
        f"(tolerance {args.tol:.3f}, max regression rate {args.max_regression_rate:.3f})"
    )

    problems = _failures(results, baseline, args.tol)

    previously_passing, now_failing, regression_rate = _case_level_regression(results, baseline)
    print(f"Case-level: {previously_passing} previously-passing cases comparable, {len(now_failing)} now failing.")
    if regression_rate is not None:
        print(f"Regression Rate: {regression_rate:.4f} ({regression_rate * 100:.1f}%)")
        if regression_rate > args.max_regression_rate:
            problems.append(
                f"regression_rate: {regression_rate:.4f} > {args.max_regression_rate} "
                f"({len(now_failing)}/{previously_passing} cases: {', '.join(now_failing[:5])}"
                f"{', ...' if len(now_failing) > 5 else ''})"
            )

    if not problems:
        print("No regression detected.")
        sys.exit(0)

    print("REGRESSIONS DETECTED:")
    for problem in problems:
        print(f"  - {problem}")
    print("See backend/eval/README.md for how to re-baseline intentionally.")
    sys.exit(1)


if __name__ == "__main__":
    main()
