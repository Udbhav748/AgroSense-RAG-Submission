"""Regression tests for the Module 10 generic tool-argument-validation
dataset (eval/module10/runners/run_tool_validation_eval.py), covering
all 4 real registered tools (retrieval, summarization, web_search,
diagnose) against their actual production input schemas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.tool_registry import TOOL_SCHEMAS  # noqa: E402
from eval.module10.runners.run_tool_validation_eval import CASES, run_case  # noqa: E402


def test_all_cases_behave_as_expected():
    results = [run_case(case) for case in CASES]
    failures = [r for r in results if not r["case_passed"]]
    assert failures == [], f"Tool argument validation regressed for: {failures}"


def test_all_four_registered_tools_are_covered():
    tools_in_dataset = {c["tool"] for c in CASES}
    assert tools_in_dataset == set(TOOL_SCHEMAS.keys())


def test_every_tool_has_at_least_one_valid_and_one_invalid_case():
    for tool in TOOL_SCHEMAS:
        tool_cases = [c for c in CASES if c["tool"] == tool]
        assert any(c["expect_valid"] for c in tool_cases), f"{tool} has no valid case"
        assert any(not c["expect_valid"] for c in tool_cases), f"{tool} has no invalid case"
