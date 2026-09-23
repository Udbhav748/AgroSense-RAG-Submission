"""Module 10 gap-closure: tool-argument accuracy was previously measured
only on the subset of tool calls with a ground-truth value already in
the dataset (summarize's document_id, web_research's approval gate) --
retrieve's genuinely planner-decided crop/collection argument
(extract_crop_context(query), used to scope retrieval) had zero
ground-truth coverage, distinct from top_k/min_score which are
correctly N/A (caller-supplied ChatRequest fields, not planner
decisions).

These tests exercise the REAL eval.module10.runners.run_agent_eval
module (not reimplemented logic) against a real ChatService whose
`_plan()` call is pure/deterministic (keyword-based, no LLM call) --
so this runs fully offline, no live API calls needed.
"""

from __future__ import annotations

from eval.module10 import config
from eval.module10.runners.run_agent_eval import run_tool_argument_cases


class _FakeVectorStore:
    pass


class _FakeLLMClient:
    def generate(self, prompt: str) -> str:
        raise AssertionError("run_tool_argument_cases must not call the LLM -- _plan() is deterministic")


def _make_chat_service():
    from app.services.rag_service import ChatService

    return ChatService(_FakeVectorStore(), _FakeLLMClient())


def _real_dataset_crop_cases() -> list[dict]:
    dataset = config.load_dataset("agent_eval.json")
    return [c for c in dataset["tool_argument_cases"] if c.get("expected_argument") == "crop"]


class TestCropArgumentAccuracyIsMeasured:
    def test_crop_cases_exist_in_the_real_dataset(self):
        """Regression pin: this ground-truth category must not silently
        disappear from the dataset in a future edit."""
        cases = _real_dataset_crop_cases()
        assert len(cases) >= 3, "expected at least 3 real crop-extraction ground-truth cases"

    def test_crop_extraction_scores_as_applicable_not_na(self):
        """The real bug this closes: crop IS a planner-decided argument
        (unlike top_k), so it belongs in the applicable=True denominator,
        not silently excluded like top_k/diagnose/web_research."""
        chat_service = _make_chat_service()
        dataset = config.load_dataset("agent_eval.json")
        result = run_tool_argument_cases(chat_service, dataset["tool_argument_cases"], "some-document-id", [])
        assert result["n_applicable"] > 0
        assert "retrieve" in result["per_tool"]

    def test_correct_crop_extraction_scores_perfectly_on_the_real_dataset(self):
        """The real dataset's crop cases (apple/potato/tomato queries +
        one no-crop negative case) must all extract correctly against
        the REAL extract_crop_context/_plan implementation -- this is
        the actual measurement, not a mocked assertion."""
        chat_service = _make_chat_service()
        dataset = config.load_dataset("agent_eval.json")
        result = run_tool_argument_cases(chat_service, dataset["tool_argument_cases"], "some-document-id", [])
        assert result["per_tool"]["retrieve"] == 1.0

    def test_no_crop_mentioned_extracts_none_not_a_hallucinated_crop(self):
        """Direct check of the negative case (agent_arg_009): a query
        with no crop keyword must not have some crop guessed onto it."""
        chat_service = _make_chat_service()
        plan = chat_service._plan("What is a Work Breakdown Structure?")  # noqa: SLF001
        assert plan.crop is None

    def test_plural_crop_keyword_normalizes_to_singular(self):
        """agent_arg_008: 'tomatoes' (plural, as it appears in real
        queries) must normalize to the 'tomato' collection key used to
        scope retrieval -- a real, previously-unverified normalization
        rule, not assumed correct."""
        chat_service = _make_chat_service()
        plan = chat_service._plan(
            "What fungicides and organic methods treat early blight on tomatoes caused by Alternaria solani?"
        )
        assert plan.crop == "tomato"

    def test_top_k_case_remains_not_applicable(self):
        """Regression pin: top_k must stay N/A (caller-supplied, not
        planner-decided) -- this fix must not accidentally start scoring
        it as if it were a real ground-truth case."""
        chat_service = _make_chat_service()
        dataset = config.load_dataset("agent_eval.json")
        result = run_tool_argument_cases(chat_service, dataset["tool_argument_cases"], "some-document-id", [])
        na_ids = {c["case_id"] for c in result["not_applicable"]}
        assert "agent_arg_003" in na_ids
