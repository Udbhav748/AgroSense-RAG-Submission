"""Regression tests for the Module 10 tool reliability evaluator
(eval/module10/runners/run_tool_reliability_eval.py), which exercises
the REAL @track_tool + @retry-wrapped search_web() production function
with its internal _search_web_core mocked -- no real network call.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_tool_reliability_eval import SCENARIOS, run_scenario  # noqa: E402


def test_all_scenarios_behave_as_expected():
    results = [run_scenario(s) for s in SCENARIOS]
    failures = [r for r in results if not r["scenario_passed"]]
    assert failures == [], f"Retry/timeout behavior regressed for: {failures}"


def test_retry_is_bounded_at_three_attempts():
    """The all-attempts-fail scenario must stop at exactly 3 attempts,
    never retry indefinitely."""
    results = [run_scenario(s) for s in SCENARIOS]
    exhausted = next(r for r in results if r["scenario_id"] == "tr_004")
    assert exhausted["actual_attempts"] == 3
    assert exhausted["actual_success"] is False


def test_timeout_scenarios_that_ultimately_fail_are_correctly_classified():
    """classified_as_timeout reflects the FINAL raised exception only --
    a timeout scenario that recovers via retry (tr_005) raises no final
    exception, so there's nothing to classify; only a timeout scenario
    that exhausts all retries (tr_006) has a final exception to check."""
    results = [run_scenario(s) for s in SCENARIOS]
    for r in results:
        if r["is_timeout_scenario"] and not r["actual_success"]:
            assert r["classified_as_timeout"] is True


def test_transient_failure_recovers_within_retry_budget():
    results = [run_scenario(s) for s in SCENARIOS]
    recovered = next(r for r in results if r["scenario_id"] == "tr_002")
    assert recovered["actual_success"] is True
    assert recovered["actual_attempts"] == 2
