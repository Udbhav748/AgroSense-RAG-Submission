"""Module 10 gap-closure: upgrades the disclosed lexical-overlap
groundedness/hallucination proxy with real NLI (entailment) scoring.
No new model is trained -- a pretrained cross-encoder NLI model
(cross-encoder/nli-MiniLM2-L6-H768) is used for inference only, loaded
the same way app/services/reranking_service.py's reranker is.

Fast tests here mock the model to pin the surrounding logic (claim
splitting reuse, per-chunk-max aggregation, threshold decision,
edge cases) without a real model load. One slower test
(TestRealModelIntegration) loads the REAL pretrained model and checks
it correctly distinguishes a genuinely entailed claim from a
genuinely contradicted one -- proving actual model behavior, not just
mocked plumbing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from eval.module10.metrics.nli_faithfulness import (
    ENTAILMENT_THRESHOLD,
    compute_nli_faithfulness,
    entailment_probability,
    get_nli_model,
)


class _FakeChunk:
    def __init__(self, text: str):
        self.text = text


class TestEntailmentProbability:
    def test_softmax_and_label_order_extraction(self, monkeypatch):
        """The model's raw logits are [contradiction, entailment,
        neutral] (verified against real examples in
        TestRealModelIntegration below) -- pin that index 1 is read as
        entailment, with a real softmax applied, not a raw logit."""
        import numpy as np

        fake_model = MagicMock()
        fake_model.predict.return_value = np.array([[-3.78, 3.85, -0.56]])
        get_nli_model.cache_clear()
        monkeypatch.setattr(
            "eval.module10.metrics.nli_faithfulness.get_nli_model", lambda: fake_model
        )

        prob = entailment_probability("A man is eating pizza.", "A man is eating food.")
        assert 0.98 < prob <= 1.0  # this logit gap softmaxes to near-certain entailment


class TestComputeNliFaithfulness:
    def test_empty_answer_scores_zero(self):
        result = compute_nli_faithfulness("", [_FakeChunk("some context")])
        assert result["nli_faithfulness"] == 0.0
        assert result["total_claims"] == 0

    def test_no_retrieved_chunks_scores_zero(self):
        result = compute_nli_faithfulness("A real claim goes here.", [])
        assert result["nli_faithfulness"] == 0.0

    def test_per_chunk_max_not_concatenation(self, monkeypatch):
        """Regression pin for the real methodological choice this module
        makes: score against EACH chunk and take the max, not one
        concatenated giant premise. A claim entailed by chunk 2 only
        (chunk 1 irrelevant) must still be marked supported."""

        def fake_entailment(premise: str, hypothesis: str) -> float:
            return 0.95 if "supports this" in premise else 0.05

        monkeypatch.setattr(
            "eval.module10.metrics.nli_faithfulness.entailment_probability", fake_entailment
        )
        chunks = [_FakeChunk("irrelevant chunk one"), _FakeChunk("this chunk supports this claim")]
        result = compute_nli_faithfulness("This is a real claim that needs sixteen characters.", chunks)
        assert result["total_claims"] == 1
        assert result["supported_claims"] == 1
        assert result["per_claim"][0]["best_matching_chunk_index"] == 1

    def test_threshold_boundary(self, monkeypatch):
        monkeypatch.setattr(
            "eval.module10.metrics.nli_faithfulness.entailment_probability",
            lambda p, h: ENTAILMENT_THRESHOLD - 0.01,
        )
        result = compute_nli_faithfulness("A claim just below the threshold value here.", [_FakeChunk("context")])
        assert result["supported_claims"] == 0

        monkeypatch.setattr(
            "eval.module10.metrics.nli_faithfulness.entailment_probability",
            lambda p, h: ENTAILMENT_THRESHOLD + 0.01,
        )
        result = compute_nli_faithfulness("A claim just above the threshold value here.", [_FakeChunk("context")])
        assert result["supported_claims"] == 1

    def test_mixed_supported_and_unsupported_claims(self, monkeypatch):
        def fake_entailment(premise: str, hypothesis: str) -> float:
            return 0.9 if "supported" in hypothesis else 0.1

        monkeypatch.setattr(
            "eval.module10.metrics.nli_faithfulness.entailment_probability", fake_entailment
        )
        answer = "This claim is supported by context. This unrelated claim is not grounded at all."
        result = compute_nli_faithfulness(answer, [_FakeChunk("some context")])
        assert result["total_claims"] == 2
        assert result["supported_claims"] == 1
        assert result["nli_faithfulness"] == 0.5


class TestRealModelIntegration:
    """Loads the REAL pretrained cross-encoder -- no training, inference
    only, same posture as this project's existing reranker/embedding
    model tests (e.g. test_reranker.py::test_reranker_singleton). Proves
    genuine entailment behavior, not mocked plumbing."""

    def test_real_model_distinguishes_entailment_from_contradiction(self):
        get_nli_model.cache_clear()
        entailed_prob = entailment_probability(
            "The apple orchard was treated with copper-based fungicide to control scab.",
            "The orchard received a copper fungicide treatment.",
        )
        contradicted_prob = entailment_probability(
            "The apple orchard was treated with copper-based fungicide to control scab.",
            "The orchard received no treatment of any kind.",
        )
        assert entailed_prob > 0.7
        assert contradicted_prob < 0.3
        assert entailed_prob > contradicted_prob
