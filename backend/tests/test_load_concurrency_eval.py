"""Deterministic tests for the Module 10 P7 load/concurrency evaluator's
aggregation logic (eval/module10/runners/run_load_concurrency_final_eval.py).

Mocks the actual HTTP calls (the real benchmark run genuinely spawns a
uvicorn subprocess and issues real HTTP requests -- see
docs/MODULE10_RESULTS.md's Load/Concurrency section and
TestRealHttpSmokeIntegration below for that) so this suite stays fast,
offline, and deterministic while still pinning the RPS/error-rate/
percentile/timeout-categorization formulas against a controlled,
fake-but-realistic sequence of per-request outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_load_concurrency_final_eval import (  # noqa: E402
    _percentile,
    run_level,
    run_rate_limit_burst_scenario,
)


def _fake_result(outcome: str, latency_s: float = 0.01, status_code: int | None = 200) -> dict:
    return {"outcome": outcome, "latency_s": latency_s, "status_code": status_code}


class TestPercentile:
    def test_empty_list_is_zero(self):
        assert _percentile([], 95) == 0.0

    def test_ordering_holds(self):
        values = [float(i) for i in range(1, 101)]
        assert _percentile(values, 50) <= _percentile(values, 95) <= _percentile(values, 99)

    def test_p50_of_small_sorted_list(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0


class TestRunLevelAggregation:
    """Patches _one_request's underlying outcome sequence and confirms
    run_level's RPS/error-rate/success-rate/percentile aggregation
    matches hand-computed expectations for a fixed, controlled set of
    per-request outcomes -- not just that it runs without crashing."""

    def _run_with_outcomes(self, outcomes: list[dict]) -> dict:
        with patch(
            "eval.module10.runners.run_load_concurrency_final_eval._one_request",
            side_effect=outcomes,
        ), patch("httpx.Client") as mock_client_cls, patch(
            "eval.module10.runners.run_load_concurrency_final_eval._resource_sample",
            return_value={"available": False},
        ):
            mock_client_cls.return_value.__enter__.return_value = MagicMock()
            return run_level(
                workload="health",
                concurrency=max(1, len(outcomes)),
                n_requests=len(outcomes),
                base_url="http://127.0.0.1:9",
                api_key="k",
            )

    def test_all_success_gives_zero_error_rate_and_full_success_rate(self):
        outcomes = [_fake_result("success", latency_s=0.01 * i) for i in range(1, 11)]
        result = self._run_with_outcomes(outcomes)
        assert result["requests"] == 10
        assert result["successes"] == 10
        assert result["error_rate"] == 0.0
        assert result["success_rate"] == 1.0
        assert result["timeouts"] == 0
        assert result["http_failures"] == 0

    def test_mixed_outcomes_are_categorized_distinctly_not_lumped_together(self):
        outcomes = [
            _fake_result("success"),
            _fake_result("success"),
            _fake_result("http_failure", status_code=500),
            _fake_result("timeout", status_code=None),
            _fake_result("connection_error", status_code=None),
            _fake_result("unexpected_exception", status_code=None),
        ]
        result = self._run_with_outcomes(outcomes)
        assert result["requests"] == 6
        assert result["successes"] == 2
        assert result["http_failures"] == 1
        assert result["timeouts"] == 1
        assert result["connection_errors"] == 1
        assert result["unexpected_exceptions"] == 1
        # A timeout must never be silently folded into the generic
        # http_failures bucket -- each category has its own independent
        # counter (unlike a naive "status != 200 => failure" tally).
        assert result["timeouts"] == 1 and result["http_failures"] == 1
        assert result["error_rate"] == round(4 / 6, 4)

    def test_latency_percentiles_computed_only_over_successes(self):
        outcomes = [
            _fake_result("success", latency_s=0.010),
            _fake_result("success", latency_s=0.020),
            _fake_result("success", latency_s=0.030),
            _fake_result("timeout", latency_s=999.0),  # must not pollute latency stats
        ]
        result = self._run_with_outcomes(outcomes)
        assert result["latency_ms"]["min"] == 10.0
        assert result["latency_ms"]["max"] == 30.0
        assert result["latency_ms"]["mean"] == 20.0

    def test_rps_is_requests_over_wall_time(self):
        outcomes = [_fake_result("success") for _ in range(5)]
        with patch(
            "eval.module10.runners.run_load_concurrency_final_eval._one_request",
            side_effect=outcomes,
        ), patch("httpx.Client") as mock_client_cls, patch(
            "eval.module10.runners.run_load_concurrency_final_eval._resource_sample",
            return_value={"available": False},
        ), patch(
            "time.perf_counter", side_effect=[0.0, 1.0]
        ):
            mock_client_cls.return_value.__enter__.return_value = MagicMock()
            result = run_level(workload="health", concurrency=5, n_requests=5, base_url="http://x", api_key="k")
        assert result["wall_time_seconds"] == 1.0
        assert result["rps"] == 5.0

    def test_empty_result_set_does_not_raise_and_reports_none_rates(self):
        result = self._run_with_outcomes([])
        assert result["requests"] == 0
        assert result["error_rate"] is None
        assert result["success_rate"] is None
        assert result["latency_ms"]["mean"] is None


class TestRateLimitBurstScenario:
    def test_rate_limited_requests_are_counted_separately_from_other_failures(self):
        outcomes = (
            [_fake_result("success") for _ in range(60)]
            + [_fake_result("http_failure", status_code=429) for _ in range(40)]
        )
        with patch(
            "eval.module10.runners.run_load_concurrency_final_eval._one_request",
            side_effect=outcomes,
        ), patch("httpx.Client") as mock_client_cls, patch(
            "eval.module10.runners.run_load_concurrency_final_eval._probe",
            return_value=(True, 200, 5.0),
        ):
            mock_client_cls.return_value.__enter__.return_value = MagicMock()
            result = run_rate_limit_burst_scenario(base_url="http://127.0.0.1:9", api_key="k")

        assert result["successes"] == 60
        assert result["rate_limited_429"] == 40
        assert result["other_failures"] == 0
        assert result["recovered_after"] is True

    def test_no_rate_limiting_is_reported_honestly_not_assumed(self):
        outcomes = [_fake_result("success") for _ in range(100)]
        with patch(
            "eval.module10.runners.run_load_concurrency_final_eval._one_request",
            side_effect=outcomes,
        ), patch("httpx.Client") as mock_client_cls, patch(
            "eval.module10.runners.run_load_concurrency_final_eval._probe",
            return_value=(True, 200, 5.0),
        ):
            mock_client_cls.return_value.__enter__.return_value = MagicMock()
            result = run_rate_limit_burst_scenario(base_url="http://127.0.0.1:9", api_key="k")

        assert result["rate_limited_429"] == 0
        assert "did not exceed" in result["what_this_demonstrates"]

    def test_service_not_recovered_is_reported_as_false_not_hidden(self):
        outcomes = [_fake_result("success") for _ in range(100)]
        with patch(
            "eval.module10.runners.run_load_concurrency_final_eval._one_request",
            side_effect=outcomes,
        ), patch("httpx.Client") as mock_client_cls, patch(
            "eval.module10.runners.run_load_concurrency_final_eval._probe",
            return_value=(False, None, 0.0),
        ):
            mock_client_cls.return_value.__enter__.return_value = MagicMock()
            result = run_rate_limit_burst_scenario(base_url="http://127.0.0.1:9", api_key="k")

        assert result["recovered_after"] is False


class TestRealHttpSmokeIntegration:
    """TASK 14's required integration-style proof: a real local HTTP
    process is started, concurrent requests are genuinely sent over a
    real socket, results are captured, and /health remains available
    afterward -- bounded and deterministic (tiny concurrency/request
    count, LLM mocked), not the full ladder."""

    def test_real_uvicorn_subprocess_answers_concurrent_health_requests(self):
        import subprocess as _subprocess

        from eval.module10.runners.run_load_concurrency_final_eval import (
            _start_uvicorn,
            _wait_for_startup,
        )

        proc = None
        try:
            proc = _start_uvicorn('{"smoke_test": "smoketestkeyxxxxxxxxxxxxxxxxxxxx"}')
            started = _wait_for_startup("http://127.0.0.1:8814/health", 90.0)
            assert started, "real uvicorn subprocess did not become healthy in time"

            result = run_level(
                workload="health",
                concurrency=3,
                n_requests=6,
                base_url="http://127.0.0.1:8814",
                api_key="smoketestkeyxxxxxxxxxxxxxxxxxxxx",
            )
            assert result["requests"] == 6
            assert result["successes"] == 6  # /health is unauthenticated -- always 200 regardless of key
            assert result["error_rate"] == 0.0

            from monitoring.uptime_check import _probe

            healthy_after, status_after, _ = _probe("http://127.0.0.1:8814/health")
            assert healthy_after is True
            assert status_after == 200
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except _subprocess.TimeoutExpired:
                    proc.kill()
