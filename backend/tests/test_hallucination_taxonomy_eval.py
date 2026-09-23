"""Regression tests for the Module 10 hallucination taxonomy evaluator
(eval/module10/runners/run_hallucination_taxonomy_eval.py). Confirms the
deterministic categorization rule stays stable and the evaluator never
silently claims a category is "ground truth" or fabricates a case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.runners.run_hallucination_taxonomy_eval import (  # noqa: E402
    ADDITIONAL_CASES,
    CATEGORY_LABELS,
    load_faithfulness_cases,
)


def test_loads_all_twenty_faithfulness_cases():
    cases = load_faithfulness_cases()
    assert len(cases) == 20
    assert all(c["category_code"] in CATEGORY_LABELS for c in cases)


def test_generation_error_reply_cases_are_categorized_as_reliability_failure_not_hallucination():
    """The 2 cases previously root-caused as GENERATION_ERROR_REPLY
    (orange-01, pepper-01) must land in category X, never in a
    hallucination/unsupported-claim bucket -- conflating the two would
    misrepresent both metrics (see docs/RAG_BENCHMARK_REPORT.md)."""
    cases = load_faithfulness_cases()
    reliability_failures = [c for c in cases if c["category_code"] == "X"]
    ids = {c["case_id"] for c in reliability_failures}
    assert ids == {"eval-orange-01", "eval-pepper-01"}


def test_additional_cases_reuse_real_prior_evidence_not_fabricated():
    """Each reused human-eval row must cite its real source document,
    never present itself as newly, independently verified."""
    for case in ADDITIONAL_CASES:
        assert "docs/HUMAN_EVAL.md" in case["source"] or "security_eval" in case["source"]


def test_no_case_is_labeled_ground_truth_or_llm_judge():
    cases = load_faithfulness_cases() + ADDITIONAL_CASES
    for case in cases:
        rationale = case["rationale"].lower()
        assert "ground truth" not in rationale
        assert "llm judge" not in rationale
