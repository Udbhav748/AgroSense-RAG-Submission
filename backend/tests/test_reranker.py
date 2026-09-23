"""Unit tests for CrossEncoderReranker and hybrid_search_with_rerank:
- Neural cross-encoder scoring and logit calibration into [0.0, 1.0]
- Zero-failure heuristic token-overlap/exact-term alignment fallback
- Candidate pool filtering and top_k preservation
- Compatibility with DocumentChunk and RetrievedChunk
- hybrid_search_with_rerank end-to-end integration
"""

import pytest

from app.models.document import DocumentChunk, RetrievedChunk
from app.services.hybrid_search import hybrid_search_with_rerank
from app.services.reranker import (
    CrossEncoderReranker,
    _calibrate_logits,
    _heuristic_score,
    _sigmoid,
    _tokenize,
    get_reranker,
    rerank,
)


def make_retrieved_chunk(chunk_id: str, text: str, score: float = 0.5, metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=score,
        metadata=metadata or {},
    )


def make_doc_chunk(chunk_id: str, text: str, chunk_index: int = 0, metadata: dict | None = None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=chunk_index,
        text=text,
        metadata=metadata or {},
    )


class TestScoreCalibration:
    def test_sigmoid_monotonicity_and_bounds(self):
        # Extreme negative to extreme positive
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)
        assert _sigmoid(-5.0) < _sigmoid(0.0) < _sigmoid(5.0)
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_calibrate_logits_unit_interval(self):
        logits = [-10.0, -2.5, 0.0, 2.5, 10.0]
        calibrated = _calibrate_logits(logits)
        assert len(calibrated) == len(logits)
        assert all(0.0 <= score <= 1.0 for score in calibrated)
        # Order must be strictly ascending
        for i in range(len(calibrated) - 1):
            assert calibrated[i] < calibrated[i + 1]

    def test_calibrate_empty_list(self):
        assert _calibrate_logits([]) == []


class TestHeuristicScoring:
    def test_empty_query_or_text_returns_zero(self):
        assert _heuristic_score("", "some content") == 0.0
        assert _heuristic_score("query", "") == 0.0
        assert _heuristic_score("   ", "   ") == 0.0

    def test_exact_term_overlap_scores_higher_than_unrelated(self):
        query = "early blight tomato treatment"
        relevant_text = "Early blight in tomato crops can be managed with copper fungicide treatment."
        unrelated_text = "Nitrogen fertilizer application rates for winter wheat cultivation."

        score_rel = _heuristic_score(query, relevant_text)
        score_unrel = _heuristic_score(query, unrelated_text)

        assert score_rel > score_unrel
        assert score_rel > 0.6
        assert score_unrel == 0.0

    def test_phrase_match_bonus(self):
        query = "powdery mildew"
        phrase_text = "Symptoms of powdery mildew include white fungal patches on leaves."
        scattered_text = "The powdery substance was white, and mildew is common in humid weather."

        score_phrase = _heuristic_score(query, phrase_text)
        score_scattered = _heuristic_score(query, scattered_text)

        assert score_phrase > score_scattered

    def test_heuristic_score_bounded_in_unit_range(self):
        query = "apple scab venturia inaequalis fungicide schedule"
        text = "Apple scab caused by Venturia inaequalis requires a strict fungicide schedule."
        score = _heuristic_score(query, text)
        assert 0.0 <= score <= 1.0


class TestCrossEncoderReranker:
    def test_reranker_singleton(self):
        r1 = get_reranker()
        r2 = get_reranker()
        assert r1 is r2

    def test_rerank_with_heuristic_fallback_ranking(self):
        reranker = CrossEncoderReranker()
        # Force fallback mode by leaving _model as None and marking load failed
        reranker._model_load_attempted = True
        reranker._model = None
        reranker._model_load_failed = True

        chunks = [
            make_retrieved_chunk("c1", "Unrelated corn fertilization guidelines", score=0.9),
            make_retrieved_chunk("c2", "Peach bacterial spot management and copper spray schedule", score=0.4),
            make_retrieved_chunk("c3", "Peach cultivation overview and climate requirements", score=0.6),
        ]

        query = "peach bacterial spot spray"
        reranked = reranker.rerank(query, chunks, top_k=2)

        assert len(reranked) == 2
        # c2 has highest term overlap for query
        assert reranked[0].chunk_id == "c2"
        assert "rerank_score" in reranked[0].metadata
        assert 0.0 <= reranked[0].metadata["rerank_score"] <= 1.0

    def test_rerank_candidate_pool_filtering_top_k(self):
        reranker = CrossEncoderReranker()
        reranker._model_load_attempted = True
        reranker._model = None
        reranker._model_load_failed = True

        candidates = [
            make_retrieved_chunk(f"c{i}", f"Tomato leaf mold symptom note {i}") for i in range(10)
        ]
        results = reranker.rerank("tomato leaf mold", candidates, top_k=4)

        assert len(results) == 4
        # Verify scores are monotonically descending
        scores = [c.metadata["rerank_score"] for c in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_rerank_empty_and_zero_top_k(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", [], top_k=5) == []
        assert reranker.rerank("query", [make_retrieved_chunk("c1", "text")], top_k=0) == []
        assert reranker.rerank("query", [make_retrieved_chunk("c1", "text")], top_k=-1) == []

    def test_rerank_document_chunks(self):
        reranker = CrossEncoderReranker()
        reranker._model_load_attempted = True
        reranker._model = None
        reranker._model_load_failed = True

        doc_chunks = [
            make_doc_chunk("d1", "Citrus greening disease vector control psyllids", chunk_index=0),
            make_doc_chunk("d2", "General greenhouse irrigation guide", chunk_index=1),
        ]

        results = reranker.rerank("citrus greening psyllids", doc_chunks, top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], DocumentChunk)
        assert results[0].chunk_id == "d1"
        assert "rerank_score" in results[0].metadata

    def test_module_convenience_rerank(self):
        chunks = [
            make_retrieved_chunk("c1", "Apple cedar rust fungicide timing"),
            make_retrieved_chunk("c2", "Grapevine pruning in winter"),
        ]
        results = rerank("cedar rust apple", chunks, top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_mock_neural_cross_encoder_prediction(self):
        class FakeNeuralCrossEncoder:
            def predict(self, pairs):
                # Return synthetic logits: higher for pair mentioning 'bacterial'
                return [3.5 if "bacterial" in text.lower() else -2.0 for _, text in pairs]

        reranker = CrossEncoderReranker()
        reranker._model = FakeNeuralCrossEncoder()
        reranker._model_load_attempted = True
        reranker._model_load_failed = False

        chunks = [
            make_retrieved_chunk("low_rel", "General crop harvesting practices"),
            make_retrieved_chunk("high_rel", "Bacterial spot on peach foliage treatment"),
        ]

        results = reranker.rerank("bacterial spot", chunks, top_k=2)
        assert results[0].chunk_id == "high_rel"
        assert results[0].metadata["rerank_score"] > results[1].metadata["rerank_score"]
        # Sigmoid(3.5) approx 0.97, Sigmoid(-2.0) approx 0.119
        assert results[0].metadata["rerank_score"] > 0.9
        assert results[1].metadata["rerank_score"] < 0.2


class FakeFAISSStore:
    def __init__(self, semantic_chunks, bm25_chunks):
        self._semantic = semantic_chunks
        self._bm25 = bm25_chunks

    def search(self, query_vector, top_k, tenant_id=None, **kwargs):
        return self._semantic[:top_k]

    def search_bm25(self, query, top_k, tenant_id=None, **kwargs):
        return self._bm25[:top_k]


class TestHybridSearchWithRerank:
    def test_hybrid_search_with_rerank_pipeline(self, monkeypatch):
        monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda q: [0.1, 0.2])

        # Candidate pool of 6 chunks retrieved via FAISS & BM25
        semantic_candidates = [
            make_retrieved_chunk("c1", "Wheat rust rust rust", score=0.9),
            make_retrieved_chunk("c2", "Tomato bacterial canker diagnostics and management", score=0.8),
            make_retrieved_chunk("c3", "Tomato early blight signs and symptoms", score=0.7),
        ]
        bm25_candidates = [
            make_retrieved_chunk("c2", "Tomato bacterial canker diagnostics and management", score=10.0),
            make_retrieved_chunk("c4", "Soil pH testing methodology", score=5.0),
            make_retrieved_chunk("c5", "Greenhouse ventilation setup", score=2.0),
        ]

        store = FakeFAISSStore(semantic_candidates, bm25_candidates)

        query = "bacterial canker tomato"
        # Run hybrid_search_with_rerank asking for top 2 precision results out of candidate_pool=5
        results = hybrid_search_with_rerank(
            query=query,
            vector_store=store,
            top_k=2,
            candidate_pool=5,
            rrf_k=60,
        )

        assert len(results) == 2
        # c2 mentions both 'bacterial canker' and 'tomato' and should be ranked #1
        assert results[0].chunk_id == "c2"
        assert "rerank_score" in results[0].metadata
        assert results[0].metadata["rerank_score"] > 0.5

    def test_hybrid_search_with_rerank_empty_candidates(self, monkeypatch):
        monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda q: [0.1, 0.2])
        store = FakeFAISSStore([], [])
        results = hybrid_search_with_rerank(
            query="empty query",
            vector_store=store,
            top_k=5,
            candidate_pool=20,
        )
        assert results == []
