"""Regression tests for the Module 10 diagnose (vision tool) reliability
evaluator (eval/module10/runners/run_diagnose_reliability_eval.py),
which exercises the REAL @track_tool + @retry-wrapped diagnose_image()
production function with only the httpx.post network boundary (and
irrelevant image-processing/self-healing/fallback branches) mocked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_diagnose_reliability_eval import SCENARIOS, run_scenario  # noqa: E402


def test_all_scenarios_behave_as_expected():
    results = [run_scenario(s) for s in SCENARIOS]
    failures = [r for r in results if not r["scenario_passed"]]
    assert failures == [], f"diagnose_image retry/timeout behavior regressed for: {failures}"


def test_retry_is_bounded_at_three_attempts():
    results = [run_scenario(s) for s in SCENARIOS]
    exhausted = next(r for r in results if r["scenario_id"] == "dr_003")
    assert exhausted["actual_attempts"] == 3
    assert exhausted["actual_success"] is False


def test_transient_failure_recovers_within_retry_budget():
    results = [run_scenario(s) for s in SCENARIOS]
    recovered = next(r for r in results if r["scenario_id"] == "dr_002")
    assert recovered["actual_success"] is True
    assert recovered["actual_attempts"] == 2


def test_gemini_fallback_forced_off_so_retry_is_actually_exercised():
    """If the Gemini fallback were allowed to succeed, diagnose_image
    would never raise/retry at all -- this test locks down that the
    harness's mocking actually reaches the retry decorator, not just the
    happy path."""
    results = [run_scenario(s) for s in SCENARIOS]
    all_fail = next(r for r in results if r["scenario_id"] == "dr_003")
    assert all_fail["actual_attempts"] == 3  # proves @retry actually ran 3 times, not short-circuited
