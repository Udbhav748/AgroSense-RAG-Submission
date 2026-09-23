"""Regression test for the Module 10 observability evaluator's
percentile helper (eval/module10/runners/run_observability_eval.py).
The full 35-request TestClient run is exercised by running the script
directly (see docs/REPRODUCE_MODULE10.md), not repeated here to keep
the regular test suite fast -- this test locks down the percentile
math itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_observability_eval import _percentile  # noqa: E402


def test_percentile_basic_ordering():
    values = sorted([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    p50 = _percentile(values, 50)
    p95 = _percentile(values, 95)
    p99 = _percentile(values, 99)
    assert p50 <= p95 <= p99
    assert values[0] <= p50 <= values[-1]


def test_percentile_empty_list_returns_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_single_value():
    assert _percentile([5.0], 50) == 5.0
    assert _percentile([5.0], 99) == 5.0
