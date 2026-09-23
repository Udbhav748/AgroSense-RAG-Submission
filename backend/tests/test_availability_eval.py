"""Deterministic tests for the availability-calculation logic in
eval/module10/runners/run_availability_eval.py -- mocks the actual
uvicorn subprocess/HTTP probing (which the real report generation
exercises for real, see docs/MODULE10_RESULTS.md's Availability
section) so this suite stays offline and fast, while still pinning the
Availability = successful / total formula and the bounded-window loop
against a fake but controlled probe sequence.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_availability_eval import run_availability_measurement  # noqa: E402


def _monotonic_stepper(step: float):
    """A deterministic stand-in for time.monotonic(): each call advances
    by `step`, so the number of while-loop iterations in
    run_availability_measurement is exactly controlled rather than
    guessed via a finite side_effect list (which would StopIteration if
    the loop polled time.monotonic() a different number of times than
    expected)."""
    state = {"t": 0.0}

    def _next():
        value = state["t"]
        state["t"] += step
        return value

    return _next


class TestAvailabilityFormula:
    def test_all_healthy_probes_gives_availability_1(self):
        with patch(
            "eval.module10.runners.run_availability_eval._wait_for_startup", return_value=True
        ), patch(
            "eval.module10.runners.run_availability_eval._probe",
            return_value=(True, 200, 5.0),
        ), patch(
            "eval.module10.runners.run_availability_eval.subprocess.Popen"
        ) as mock_popen, patch("time.sleep"), patch(
            "time.monotonic", side_effect=_monotonic_stepper(1.0)
        ):
            mock_popen.return_value.terminate.return_value = None
            mock_popen.return_value.wait.return_value = None
            result = run_availability_measurement(port=1, duration_seconds=3, interval_seconds=1)

        assert result["started"] is True
        assert result["availability"] == 1.0
        assert result["failed_probes"] == 0
        assert result["total_probes"] > 0

    def test_mixed_probes_gives_partial_availability(self):
        probe_sequence = [(True, 200, 5.0), (False, 500, 5.0), (True, 200, 5.0), (False, None, 5.0)]
        with patch(
            "eval.module10.runners.run_availability_eval._wait_for_startup", return_value=True
        ), patch(
            "eval.module10.runners.run_availability_eval._probe", side_effect=probe_sequence
        ), patch(
            "eval.module10.runners.run_availability_eval.subprocess.Popen"
        ) as mock_popen, patch("time.sleep"), patch(
            "time.monotonic", side_effect=_monotonic_stepper(1.0)
        ):
            mock_popen.return_value.terminate.return_value = None
            mock_popen.return_value.wait.return_value = None
            result = run_availability_measurement(port=1, duration_seconds=5, interval_seconds=1)

        assert result["total_probes"] == 4
        assert result["successful_probes"] == 2
        assert result["failed_probes"] == 2
        assert result["availability"] == 0.5

    def test_startup_failure_reports_error_and_runs_no_probes(self):
        with patch(
            "eval.module10.runners.run_availability_eval._wait_for_startup", return_value=False
        ), patch("eval.module10.runners.run_availability_eval.subprocess.Popen") as mock_popen:
            mock_popen.return_value.terminate.return_value = None
            mock_popen.return_value.wait.return_value = None
            result = run_availability_measurement(port=1, duration_seconds=5, interval_seconds=1)

        assert result["started"] is False
        assert "error" in result

    def test_process_is_always_terminated_even_on_success(self):
        with patch(
            "eval.module10.runners.run_availability_eval._wait_for_startup", return_value=True
        ), patch(
            "eval.module10.runners.run_availability_eval._probe", return_value=(True, 200, 1.0)
        ), patch(
            "eval.module10.runners.run_availability_eval.subprocess.Popen"
        ) as mock_popen, patch("time.sleep"), patch(
            "time.monotonic", side_effect=_monotonic_stepper(1.0)
        ):
            mock_popen.return_value.terminate.return_value = None
            mock_popen.return_value.wait.return_value = None
            run_availability_measurement(port=1, duration_seconds=1, interval_seconds=1)

        mock_popen.return_value.terminate.assert_called_once()
