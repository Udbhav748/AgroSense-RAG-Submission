#!/usr/bin/env python
"""Module 10 tool reliability evaluation for `diagnose`, the vision
tool -- extends the same Tool Success/Retry/Timeout measurement
`run_tool_reliability_eval.py` did for `web_search` to the second real
retry-wrapped tool (`app/services/vision_client.py::diagnose_image`,
`@track_tool("diagnose")` + `@retry(stop_after_attempt(3),
wait_fixed(1), retry_if_exception_type(VisionServiceError))`).

This was previously disclosed as NOT attempted (too complex to mock
deterministically without risk) -- this script closes that gap by
mocking every non-retry-relevant branch explicitly rather than avoiding
the function: `generate_leaf_saliency` (image processing, irrelevant to
retry behavior), `is_leafsense_online`/`_try_auto_start_leafsense`
(self-healing, irrelevant here since we force `engine="leafsense"` to
skip the direct-Gemini branch), and `_diagnose_with_gemini_fallback`
(forced to return None, so a simulated LeafSense HTTP failure actually
propagates as VisionServiceError and exercises the real @retry decorator
-- if the fallback were allowed to succeed, no retry would ever be
observed, silently hiding the exact behavior this evaluator exists to
measure).

Usage (from backend/):
    python eval/module10/runners/run_diagnose_reliability_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.exceptions import VisionServiceError  # noqa: E402
from app.services.tool_registry import _looks_like_timeout  # noqa: E402
from app.services.vision_client import diagnose_image  # noqa: E402

_FAKE_RESPONSE_PAYLOAD = {"class": "Apple___healthy", "confidence": 0.95}


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _fake_success(*args, **kwargs) -> _FakeResponse:
    return _FakeResponse(_FAKE_RESPONSE_PAYLOAD)


def _fake_timeout(*args, **kwargs):
    raise httpx.TimeoutException("simulated LeafSense timeout")


def _fake_connection_error(*args, **kwargs):
    raise httpx.ConnectError("simulated connection refused")


SCENARIOS = [
    {
        "id": "dr_001",
        "category": "success_first_attempt",
        "http_behavior": lambda calls: _fake_success(),
        "expect_final_success": True,
        "expect_attempts": 1,
    },
    {
        "id": "dr_002",
        "category": "transient_fail_then_success",
        "http_behavior": lambda calls: _fake_connection_error() if calls["n"] == 1 else _fake_success(),
        "expect_final_success": True,
        "expect_attempts": 2,
    },
    {
        "id": "dr_003",
        "category": "all_attempts_fail",
        "http_behavior": lambda calls: _fake_connection_error(),
        "expect_final_success": False,
        "expect_attempts": 3,
    },
    {
        "id": "dr_004",
        "category": "timeout_first_then_success",
        "http_behavior": lambda calls: _fake_timeout() if calls["n"] == 1 else _fake_success(),
        "expect_final_success": True,
        "expect_attempts": 2,
        "is_timeout_scenario": True,
    },
    {
        "id": "dr_005",
        "category": "timeout_all_attempts",
        "http_behavior": lambda calls: _fake_timeout(),
        "expect_final_success": False,
        "expect_attempts": 3,
        "is_timeout_scenario": True,
    },
]


def run_scenario(scenario: dict) -> dict:
    calls = {"n": 0}

    def _post(*args, **kwargs):
        calls["n"] += 1
        return scenario["http_behavior"](calls)

    exception_raised = None
    with (
        patch("app.services.vision_client.generate_leaf_saliency", return_value={"heatmap_base64": None, "infected_area_percentage": None, "lesion_count": None}),
        patch("app.services.vision_client.is_leafsense_online", return_value=True),
        patch("app.services.vision_client._diagnose_with_gemini_fallback", return_value=None),
        patch("httpx.post", side_effect=_post),
    ):
        try:
            diagnose_image(b"fake-image-bytes", "leaf.jpg", "image/jpeg", engine="leafsense")
            actual_success = True
        except VisionServiceError as exc:
            actual_success = False
            exception_raised = str(exc)

    timed_out = exception_raised is not None and _looks_like_timeout(VisionServiceError(exception_raised))

    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "expect_final_success": scenario["expect_final_success"],
        "actual_success": actual_success,
        "expect_attempts": scenario["expect_attempts"],
        "actual_attempts": calls["n"],
        "scenario_passed": actual_success == scenario["expect_final_success"] and calls["n"] == scenario["expect_attempts"],
        "is_timeout_scenario": scenario.get("is_timeout_scenario", False),
        "classified_as_timeout": timed_out,
    }


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.module10 import config

    results = [run_scenario(s) for s in SCENARIOS]
    n = len(results)
    n_correct = sum(1 for r in results if r["scenario_passed"])

    n_tool_successes = sum(1 for r in results if r["actual_success"])
    tool_success_rate = round(n_tool_successes / n, 4)

    retried = [r for r in results if r["actual_attempts"] > 1]
    retry_success_rate = round(sum(1 for r in retried if r["actual_success"]) / len(retried), 4) if retried else None

    timeout_scenarios = [r for r in results if r["is_timeout_scenario"]]
    timeout_rate = round(len(timeout_scenarios) / n, 4)

    report = {
        "metadata": {
            **config.run_metadata(sample_count=n, dataset_version="diagnose_reliability_v1"),
            "evaluator": "eval/module10/runners/run_diagnose_reliability_eval.py",
            "target": "app.services.vision_client.diagnose_image (real @track_tool + @retry-wrapped production function, engine='leafsense')",
        },
        "methodology": (
            "Exercises the REAL diagnose_image function with only httpx.post (the actual network "
            "boundary) mocked, plus generate_leaf_saliency/is_leafsense_online/"
            "_diagnose_with_gemini_fallback mocked to isolate retry behavior from image-processing, "
            "self-healing, and cross-provider-fallback logic that are irrelevant to what this evaluator "
            "measures. The Gemini fallback is forced to return None specifically so a simulated LeafSense "
            "failure actually propagates as VisionServiceError and reaches the real @retry decorator -- "
            "letting the fallback succeed would silently hide the retry behavior being measured."
        ),
        "tool_success_rate": tool_success_rate,
        "retry_success_rate": retry_success_rate,
        "timeout_rate": timeout_rate,
        "n_scenarios": n,
        "n_scenarios_behaved_as_expected": n_correct,
        "per_scenario": results,
        "limitations": [
            "5 hand-authored scenarios mirroring run_tool_reliability_eval.py's web_search coverage, not "
            "an exhaustive fuzz test.",
            "Only the engine='leafsense' branch is tested (bypasses the direct-Gemini and hybrid-consensus "
            "branches) to isolate retry behavior specifically -- those branches are covered functionally "
            "by existing tests elsewhere (tests/test_vision_client.py if present), not by this evaluator.",
        ],
    }

    path = config.save_report(report, name="diagnose_reliability_final")
    print(f"Saved: {path}")
    print(f"Tool Success Rate: {tool_success_rate}")
    print(f"Retry Success Rate: {retry_success_rate}")
    print(f"Timeout Rate: {timeout_rate}")
    print(f"Scenarios behaving as expected: {n_correct}/{n}")
    failures = [r for r in results if not r["scenario_passed"]]
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  {f}")


if __name__ == "__main__":
    main()
