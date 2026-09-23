"""Unit tests for ChatService.handle_diagnose (the image-diagnosis action).

diagnose_image and retrieve are monkeypatched at the rag_service module
level — these tests cover the diagnose action's wiring (prediction ->
retrieval query -> existing corrective RAG loop -> response shape), not
vision_client's HTTP handling (see test_vision_client.py) or retrieval
internals (see test_retrieval_service.py).
"""

import pytest

import app.services.rag_service as rag_service_module
from app.core.exceptions import VisionServiceError
from app.models.document import RetrievedChunk, VisionPrediction
from app.services.prompt_builder import FALLBACK_REPLY
from app.services.rag_service import ChatService


class FakeLLMClient:
    def __init__(self, response="grounded diagnosis answer"):
        self.response = response
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


class FakeVectorStore:
    """handle_diagnose never touches the vector store directly — retrieve()
    is monkeypatched at the module level — only used to satisfy
    ChatService.__init__."""


def make_service(llm_client=None):
    return ChatService(FakeVectorStore(), llm_client or FakeLLMClient())


def make_chunk(text="Bacterial spot symptoms concentrate at the leaf tip.", score=0.9):
    return RetrievedChunk(chunk_id="chunk-1", document_id="doc-1", text=text, score=score, metadata={})


def make_prediction(
    raw_class="Peach___Bacterial_spot",
    crop="peach",
    disease="bacterial spot",
    confidence=0.94,
    low_confidence=False,
):
    return VisionPrediction(
        raw_class=raw_class, crop=crop, disease=disease, confidence=confidence, low_confidence=low_confidence
    )


class TestHandleDiagnoseHappyPath:
    def test_high_confidence_in_corpus_prediction_returns_grounded_answer(self, monkeypatch):
        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: make_prediction())
        captured_retrieve_kwargs = {}

        def _mock_retrieve(query, vector_store, **kwargs):
            captured_retrieve_kwargs.update(kwargs)
            return [make_chunk()]

        monkeypatch.setattr(rag_service_module, "retrieve", _mock_retrieve)

        llm_client = FakeLLMClient(response="This is bacterial spot; here's what to do.")
        service = make_service(llm_client)

        response = service.handle_diagnose(b"fake-image-bytes", "leaf.jpg", "image/jpeg")

        assert response.answer == "This is bacterial spot; here's what to do."
        assert response.tool_used == "diagnose"
        assert response.diagnosis is not None
        assert response.diagnosis.crop == "peach"
        assert captured_retrieve_kwargs.get("collection") == "peach"
        assert response.diagnosis.disease == "bacterial spot"
        assert response.diagnosis.confidence == pytest.approx(0.94)
        assert response.diagnosis.low_confidence is False
        assert len(response.retrieved_chunks) == 1
        assert len(response.sources) == 1

    def test_retrieval_query_includes_crop_to_disambiguate_shared_disease_names(self, monkeypatch):
        # Bacterial_spot exists for both peach and tomato; the query handed
        # to retrieve() must carry the crop, or retrieval could pull the
        # wrong crop's chunks.
        captured_queries = []

        def _fake_retrieve(query, vector_store, **kwargs):
            captured_queries.append(query)
            return [make_chunk()]

        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: make_prediction())
        monkeypatch.setattr(rag_service_module, "retrieve", _fake_retrieve)

        service = make_service()
        service.handle_diagnose(b"fake-image-bytes", "leaf.jpg", "image/jpeg")

        assert captured_queries
        assert "peach" in captured_queries[0]
        assert "bacterial spot" in captured_queries[0]

    def test_accompanying_user_query_is_folded_into_retrieval_query(self, monkeypatch):
        captured_queries = []

        def _fake_retrieve(query, vector_store, **kwargs):
            captured_queries.append(query)
            return [make_chunk()]

        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: make_prediction())
        monkeypatch.setattr(rag_service_module, "retrieve", _fake_retrieve)

        service = make_service()
        service.handle_diagnose(
            b"fake-image-bytes", "leaf.jpg", "image/jpeg", query="is this from poor fertilization?"
        )

        assert "poor fertilization" in captured_queries[0]


class TestHandleDiagnoseLowConfidence:
    def test_low_confidence_flag_is_surfaced_not_hidden(self, monkeypatch):
        low_confidence_prediction = make_prediction(confidence=0.31, low_confidence=True)
        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: low_confidence_prediction)
        monkeypatch.setattr(
            rag_service_module, "retrieve", lambda *a, **k: [make_chunk()]
        )

        service = make_service()
        response = service.handle_diagnose(b"fake-image-bytes", "leaf.jpg", "image/jpeg")

        # Still answers (a low-confidence prediction is a flag, not a
        # refusal) but the flag must be visible in the response.
        assert response.diagnosis.low_confidence is True
        assert response.diagnosis.confidence == pytest.approx(0.31)


class TestHandleDiagnoseVisionServiceFailure:
    def test_vision_service_unreachable_propagates_as_app_error(self, monkeypatch):
        def _raise_vision_error(*a, **k):
            raise VisionServiceError("Could not reach LeafSense at http://localhost:8001: connection refused")

        monkeypatch.setattr(rag_service_module, "diagnose_image", _raise_vision_error)

        service = make_service()
        with pytest.raises(VisionServiceError):
            service.handle_diagnose(b"fake-image-bytes", "leaf.jpg", "image/jpeg")

    def test_vision_service_timeout_propagates_as_app_error(self, monkeypatch):
        def _raise_timeout(*a, **k):
            raise VisionServiceError("LeafSense request timed out after 15s")

        monkeypatch.setattr(rag_service_module, "diagnose_image", _raise_timeout)

        service = make_service()
        with pytest.raises(VisionServiceError):
            service.handle_diagnose(b"fake-image-bytes", "leaf.jpg", "image/jpeg")


class TestHandleDiagnoseOutOfCorpusCrop:
    def test_crop_not_in_corpus_falls_back_without_hallucinating(self, monkeypatch):
        # A real LeafSense class (grape isn't in AgroSense-RAG's 5-crop corpus)
        # — retrieval finds nothing relevant, generation (correctly) can't
        # ground an answer and produces the fixed fallback line rather than
        # inventing grape disease advice from unrelated chunks.
        grape_prediction = make_prediction(raw_class="Grape___Black_rot", crop="grape", disease="black rot")
        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: grape_prediction)
        monkeypatch.setattr(
            rag_service_module, "retrieve", lambda *a, **k: []
        )

        llm_client = FakeLLMClient(response=FALLBACK_REPLY)
        service = make_service(llm_client)

        response = service.handle_diagnose(b"fake-image-bytes", "leaf.jpg", "image/jpeg")

        assert response.answer == FALLBACK_REPLY
        assert response.retrieved_chunks == []
        assert response.diagnosis.crop == "grape"
        assert response.diagnosis.disease == "black rot"
