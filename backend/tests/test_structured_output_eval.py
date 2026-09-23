"""Regression tests for the Module 10 structured-output evaluation
dataset (eval/module10/runners/run_structured_output_eval.py). Ensures
the 17-case fixed dataset's parser-correctness stays at 100% -- a
regression here means parse_structured_answer's accept/reject behavior
changed for a documented case shape.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_structured_output_eval import CASES, run_case  # noqa: E402


def test_all_seventeen_cases_behave_as_expected():
    results = [run_case(case) for case in CASES]
    failures = [r for r in results if not r["case_passed"]]
    assert failures == [], f"Parser behavior regressed for: {failures}"


def test_valid_cases_have_correct_field_values():
    results = [run_case(case) for case in CASES]
    for r in results:
        if "field_checks" in r:
            assert r["field_checks"]["answer_correct"], r
            assert r["field_checks"]["sources_correct"], r


def test_dataset_covers_the_documented_failure_categories():
    categories = {c["category"] for c in CASES}
    required = {
        "valid",
        "fenced_json",
        "malformed_json",
        "missing_required_field",
        "wrong_type_answer_is_number",
        "extra_field",
        "provider_failure_empty_response",
    }
    assert required.issubset(categories)
