"""Tests for Feature #4 — the lexical groundedness (hallucination)
detector in the RAG pipeline.

Covers the pure module-level helpers (_content_tokens, _grounding_score,
_detect_hallucination) and the ChatResponse wiring in _respond (the
hallucination_detected / grounding_score fields + metric/log emission).
"""

from app.core.config import settings
from app.models.document import RetrievedChunk, WebSearchResult
from app.services.rag_service import (
    _content_tokens,
    _detect_hallucination,
    _grounding_score,
)


def make_chunk(text="chunk text about the pmp guide and project management", score=0.9):
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text=text,
        score=score,
        metadata={},
    )


def make_web_result(snippet="web snippet about something else entirely"):
    return WebSearchResult(title="t", url="https://example.com/x", snippet=snippet)


class TestContentTokens:
    def test_strips_stopwords_and_case(self):
        tokens = _content_tokens("The Project Management the AND and OR")
        assert "project" in tokens
        assert "management" in tokens
        assert "the" not in tokens
        assert "and" not in tokens

    def test_single_char_tokens_dropped(self):
        assert not _content_tokens("a b c")
        assert "a" not in _content_tokens("a project")

    def test_non_alphabetic_content_kept(self):
        tokens = _content_tokens("phase1 phase-2 PMBOK_v7")
        assert "phase1" in tokens
        assert "phase" in tokens  # 'phase-2' splits on the hyphen
        assert "pmbok" in tokens
        assert "v7" in tokens


class TestGroundingScore:
    def test_full_overlap_is_1(self):
        chunk = make_chunk(text="alpha beta gamma delta epsilon")
        score = _grounding_score("alpha beta gamma delta epsilon", [chunk], [])
        assert score == 1.0

    def test_no_overlap_is_zero(self):
        chunk = make_chunk(text="completely unrelated document words")
        score = _grounding_score("totally different invented vocabulary", [chunk], [])
        assert score == 0.0

    def test_empty_answer_scores_1(self):
        chunk = make_chunk()
        assert _grounding_score("", [chunk], []) == 1.0
        assert _grounding_score("   ", [chunk], []) == 1.0

    def test_no_context_scores_1(self):
        assert _grounding_score("some answer words here", [], []) == 1.0

    def test_web_snippets_count_as_context(self):
        web = make_web_result(snippet="alpha beta gamma delta epsilon")
        score = _grounding_score("alpha beta gamma", [], [web])
        assert score == 1.0


class TestDetectHallucination:
    def test_grounded_answer_not_detected(self, monkeypatch):
        monkeypatch.setattr(settings, "hallucination_detection_enabled", True)
        monkeypatch.setattr(settings, "hallucination_min_answer_chars", 1)
        chunk = make_chunk(text="alpha beta gamma delta epsilon zeta eta theta")
        detected, score = _detect_hallucination("alpha beta gamma delta epsilon", [chunk], [])
        assert detected is False
        assert score == 1.0

    def test_invented_answer_detected(self, monkeypatch):
        monkeypatch.setattr(settings, "hallucination_detection_enabled", True)
        chunk = make_chunk(text="managed project scope schedule budget quality")
        answer = (
            "the eiffel tower was built in 1889 in paris france from wrought iron "
            "and stands three hundred meters tall"
        )
        detected, score = _detect_hallucination(answer, [chunk], [])
        assert detected is True
        assert score is not None and score < settings.hallucination_grounding_threshold

    def test_too_short_answer_skipped(self, monkeypatch):
        monkeypatch.setattr(settings, "hallucination_detection_enabled", True)
        monkeypatch.setattr(settings, "hallucination_min_answer_chars", 200)
        chunk = make_chunk(text="alpha beta")
        detected, score = _detect_hallucination("alpha", [chunk], [])
        assert detected is False
        assert score is None

    def test_no_context_skipped(self, monkeypatch):
        monkeypatch.setattr(settings, "hallucination_detection_enabled", True)
        detected, score = _detect_hallucination("some long enough answer", [], [])
        assert detected is False
        assert score is None

    def test_disabled_config_never_flags(self, monkeypatch):
        monkeypatch.setattr(settings, "hallucination_detection_enabled", False)
        chunk = make_chunk(text="alpha")
        detected, score = _detect_hallucination("zzzz totally unrelated", [chunk], [])
        assert detected is False
        assert score is None

    def test_threshold_boundary_respected(self, monkeypatch):
        monkeypatch.setattr(settings, "hallucination_detection_enabled", True)
        monkeypatch.setattr(settings, "hallucination_grounding_threshold", 0.5)
        monkeypatch.setattr(settings, "hallucination_min_answer_chars", 1)
        # Half of the answer's tokens come from context -> exactly at (0.5)
        # is NOT below the threshold, so not flagged.
        chunk = make_chunk(text="alpha beta")
        detected, score = _detect_hallucination("alpha beta fabricated bogus", [chunk], [])
        assert score == 0.5
        assert detected is False


class TestRespondWiring:
    def test_grounded_answer_passthrough(self, monkeypatch):
        from tests.test_rag_service import make_service

        monkeypatch.setattr(settings, "hallucination_detection_enabled", True)
        service = make_service()
        import time

        chunk = make_chunk(text="nkp alpha beta gamma delta epsilon inventory shelf life")
        response = service._respond(
            answer="alpha beta gamma delta epsilon are the inventory principles in the document",
            retrieved_chunks=[chunk],
            query="test",
            query_type="retrieve",
            tool_used="retrieval",
            steps_taken=2,
            start=time.perf_counter() - 0.1,
        )
        assert response.hallucination_detected is False
        assert response.grounding_score is not None
        assert response.grounding_score >= settings.hallucination_grounding_threshold

    def test_invented_answer_flagged_and_metric_counted(self, monkeypatch):
        from app.core.metrics import get_metrics
        from tests.test_rag_service import make_service

        monkeypatch.setattr(settings, "hallucination_detection_enabled", True)
        get_metrics().reset()
        service = make_service()
        import time

        chunk = make_chunk(text="alpha beta gamma delta epsilon inventory policy")
        response = service._respond(
            answer="the great wall of china was built in the qin dynasty twenty centuries ago "
            "and stretches thousands of kilometers across the northern plains",
            retrieved_chunks=[chunk],
            query="test",
            query_type="retrieve",
            tool_used="retrieval",
            steps_taken=2,
            start=time.perf_counter() - 0.1,
        )
        assert response.hallucination_detected is True
        assert response.grounding_score < settings.hallucination_grounding_threshold
        rendered = get_metrics().render()
        assert "hallucinations_detected_total" in rendered
        assert 'answer_source="documents"' in rendered
