"""Regression test for the Module 10 load-test evaluator's percentile
helper (eval/module10/runners/run_load_test.py). The full 4-level
concurrency run is exercised by running the script directly (see
docs/REPRODUCE_MODULE10.md), not repeated here to keep the regular test
suite fast.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_load_test import _percentile  # noqa: E402


def test_percentile_ordering():
    values = sorted([float(i) for i in range(1, 101)])
    assert _percentile(values, 50) <= _percentile(values, 95) <= _percentile(values, 99)


def test_percentile_empty():
    assert _percentile([], 50) == 0.0
