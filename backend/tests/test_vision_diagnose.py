"""Comprehensive unit and integration tests for LeafSense vision bridge and
diagnose workflows.

Covers:
- LeafSense response mocking (successful diagnosis, low confidence, invalid image, 502/503 HTTP errors, timeout/connection refused).
- Verification that all 38 classes map cleanly.
- End-to-end /chat/diagnose endpoint behavior via TestClient (200 responses, schema compliance, 400/415 validations, and 502 VisionServiceError mapping).
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.exceptions import VisionServiceError
from app.models.document import EmbeddedChunk, RetrievedChunk, VisionPrediction
from app.models.schemas import ChatResponse
from app.services import vision_client as vision_client_module
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.rag_service import ChatService
from app.services.vision_client import CLASS_LABEL_MAP, diagnose_image
from tests.conftest import assert_matches_schema

try:
    from app.main import app
    HAS_MULTIPART = True
except (ImportError, RuntimeError):
    app = None
    HAS_MULTIPART = False

VALID_HEADERS = {"X-API-Key": settings.api_key}

# Complete list of 38 classes as declared in LeafSense/backend/main.py
LEAFSENSE_38_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


class _FakeResponse:
    """Mock httpx.Response for LeafSense HTTP calls."""

    def __init__(self, json_data=None, status_code=200, text=""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost:8001/predict/insightai")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code} error: {self.text}",
                request=request,
                response=response,
            )

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class FakeLLMClient:
    """Mock LLM client for RAG diagnosis answers."""

    def __init__(self, response_text: str = "This is a diagnosis treatment recommendation."):
        self.response_text = response_text
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response_text


# ---------------------------------------------------------------------------
# Unit Tests: vision_client.py
# ---------------------------------------------------------------------------


class TestVisionClientUnit:
    def test_all_38_classes_cleanly_mapped(self):
        """Verify CLASS_LABEL_MAP covers all 38 LeafSense classes without any unmapped entries."""
        assert len(CLASS_LABEL_MAP) == 38
        assert len(LEAFSENSE_38_CLASSES) == 38

        for class_name in LEAFSENSE_38_CLASSES:
            assert class_name in CLASS_LABEL_MAP, f"Missing class in CLASS_LABEL_MAP: {class_name}"
            crop, disease = CLASS_LABEL_MAP[class_name]
            assert crop != "unknown", f"Crop was unmapped for {class_name}"
            assert disease != "", f"Disease was empty for {class_name}"
            assert isinstance(crop, str) and len(crop) > 0
            assert isinstance(disease, str) and len(disease) > 0

    def test_successful_diagnosis_healthy_leaf(self, monkeypatch):
        """Mock successful LeafSense prediction for healthy leaf."""
        monkeypatch.setattr(settings, "vision_confidence_threshold", 0.5)
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse({"class": "Apple___healthy", "confidence": 0.98}),
        )

        prediction = diagnose_image(b"fake-image-bytes", "healthy_apple.jpg", "image/jpeg")

        assert prediction.raw_class == "Apple___healthy"
        assert prediction.crop == "apple"
        assert prediction.disease == "healthy"
        assert prediction.confidence == pytest.approx(0.98)
        assert prediction.low_confidence is False

    def test_successful_diagnosis_diseased_leaf(self, monkeypatch):
        """Mock successful LeafSense prediction for diseased leaf."""
        monkeypatch.setattr(settings, "vision_confidence_threshold", 0.5)
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse({"class": "Tomato___Early_blight", "confidence": 0.92}),
        )

        prediction = diagnose_image(b"fake-image-bytes", "tomato_leaf.jpg", "image/jpeg")

        assert prediction.raw_class == "Tomato___Early_blight"
        assert prediction.crop == "tomato"
        assert prediction.disease == "early blight"
        assert prediction.confidence == pytest.approx(0.92)
        assert prediction.low_confidence is False

    def test_low_confidence_diagnosis_flagged(self, monkeypatch):
        """Mock prediction where confidence is below the threshold."""
        monkeypatch.setattr(settings, "vision_confidence_threshold", 0.60)
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse({"class": "Potato___Late_blight", "confidence": 0.35}),
        )

        prediction = diagnose_image(b"fake-image-bytes", "potato_leaf.jpg", "image/jpeg")

        assert prediction.raw_class == "Potato___Late_blight"
        assert prediction.crop == "potato"
        assert prediction.disease == "late blight"
        assert prediction.confidence == pytest.approx(0.35)
        assert prediction.low_confidence is True

    def test_invalid_image_response_from_leafsense_400(self, monkeypatch):
        """Mock LeafSense returning 400 Bad Request when an invalid image is uploaded."""
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse(
                status_code=400, text='{"detail":"Uploaded file is not a valid image."}'
            ),
        )

        with pytest.raises(VisionServiceError) as exc_info:
            diagnose_image(b"corrupt-data", "bad.jpg", "image/jpeg")

        assert "400" in str(exc_info.value)
        assert "not a valid image" in str(exc_info.value)

    def test_service_unavailable_503(self, monkeypatch):
        """Mock LeafSense returning 503 Service Unavailable when model is not loaded."""
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse(
                status_code=503, text='{"detail":"Model is not loaded."}'
            ),
        )

        with pytest.raises(VisionServiceError) as exc_info:
            diagnose_image(b"fake-bytes", "leaf.jpg", "image/jpeg")

        assert "503" in str(exc_info.value)
        assert "Model is not loaded" in str(exc_info.value)

    def test_service_bad_gateway_502(self, monkeypatch):
        """Mock 502 Bad Gateway response from proxy/service."""
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse(status_code=502, text="Bad Gateway"),
        )

        with pytest.raises(VisionServiceError) as exc_info:
            diagnose_image(b"fake-bytes", "leaf.jpg", "image/jpeg")

        assert "502" in str(exc_info.value)

    def test_connection_timeout(self, monkeypatch):
        """Mock connection timeout to LeafSense."""
        def _raise_timeout(*a, **k):
            raise httpx.TimeoutException("Connection timed out after 15s")

        monkeypatch.setattr(vision_client_module.httpx, "post", _raise_timeout)

        with pytest.raises(VisionServiceError) as exc_info:
            diagnose_image(b"fake-bytes", "leaf.jpg", "image/jpeg")

        assert "timed out" in str(exc_info.value).lower()

    def test_connection_refused_service_not_running(self, monkeypatch):
        """Mock connection refused when LeafSense is not running."""
        def _raise_connect_error(*a, **k):
            raise httpx.ConnectError("Connection refused at http://localhost:8001")

        monkeypatch.setattr(vision_client_module.httpx, "post", _raise_connect_error)

        with pytest.raises(VisionServiceError) as exc_info:
            diagnose_image(b"fake-bytes", "leaf.jpg", "image/jpeg")

        assert "Could not reach LeafSense" in str(exc_info.value)

    def test_malformed_response_json(self, monkeypatch):
        """Mock LeafSense returning an unexpected JSON response structure."""
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse({"wrong_key": "data"}),
        )

        with pytest.raises(VisionServiceError) as exc_info:
            diagnose_image(b"fake-bytes", "leaf.jpg", "image/jpeg")

        assert "unexpected response shape" in str(exc_info.value)

    def test_api_key_header_forwarded_when_configured(self, monkeypatch):
        """Verify X-API-Key header is sent to LeafSense if configured in settings."""
        monkeypatch.setattr(settings, "vision_service_api_key", "secret-leafsense-key")
        captured_headers = {}

        def _fake_post(url, files=None, headers=None, timeout=None):
            captured_headers.update(headers or {})
            return _FakeResponse({"class": "Apple___healthy", "confidence": 0.99})

        monkeypatch.setattr(vision_client_module.httpx, "post", _fake_post)

        diagnose_image(b"fake-bytes", "leaf.jpg", "image/jpeg")
        assert captured_headers.get("X-API-Key") == "secret-leafsense-key"


# ---------------------------------------------------------------------------
# Integration Tests: ChatService.handle_diagnose
# ---------------------------------------------------------------------------


class TestChatServiceDiagnose:
    def test_chat_service_diagnose_success(self, monkeypatch):
        fake_prediction = VisionPrediction(
            raw_class="Tomato___Early_blight",
            crop="tomato",
            disease="early blight",
            confidence=0.95,
            low_confidence=False,
        )
        monkeypatch.setattr(
            "app.services.rag_service.diagnose_image", lambda *a, **k: fake_prediction
        )
        monkeypatch.setattr(
            "app.services.rag_service.retrieve",
            lambda query, vector_store, **k: [
                RetrievedChunk(
                    chunk_id="c1",
                    document_id="d1",
                    text="Treat early blight with copper fungicide.",
                    score=0.92,
                    metadata={"source": "corpus"},
                )
            ],
        )

        service = ChatService(vector_store=None, llm_client=FakeLLMClient("Apply fungicide."))
        response = service.handle_diagnose(b"fake-image", "tomato.jpg", "image/jpeg")

        assert response.tool_used == "diagnose"
        assert response.diagnosis is not None
        assert response.diagnosis.crop == "tomato"
        assert response.diagnosis.disease == "early blight"
        assert response.diagnosis.confidence == pytest.approx(0.95)
        assert response.diagnosis.low_confidence is False
        assert response.answer == "Apply fungicide."

    def test_chat_service_diagnose_service_error_propagates(self, monkeypatch):
        def _raise_error(*a, **k):
            raise VisionServiceError("Could not reach LeafSense at http://localhost:8001")

        monkeypatch.setattr("app.services.rag_service.diagnose_image", _raise_error)

        service = ChatService(vector_store=None, llm_client=FakeLLMClient())
        with pytest.raises(VisionServiceError):
            service.handle_diagnose(b"fake-image", "leaf.jpg", "image/jpeg")

    def test_chat_service_stream_diagnose_success(self, monkeypatch):
        fake_prediction = VisionPrediction(
            raw_class="Tomato___Early_blight",
            crop="tomato",
            disease="early blight",
            confidence=0.95,
            low_confidence=False,
        )
        monkeypatch.setattr(
            "app.services.rag_service.diagnose_image", lambda *a, **k: fake_prediction
        )
        monkeypatch.setattr(
            "app.services.rag_service.retrieve",
            lambda query, vector_store, **k: [
                RetrievedChunk(
                    chunk_id="c1",
                    document_id="d1",
                    text="Treat early blight with copper fungicide.",
                    score=0.92,
                    metadata={"source": "corpus"},
                )
            ],
        )

        class StreamingLLMClient(FakeLLMClient):
            def generate_stream(self, prompt: str):
                yield "Apply "
                yield "copper "
                yield "fungicide."

        service = ChatService(vector_store=None, llm_client=StreamingLLMClient())
        events = list(service.stream_diagnose(b"fake-image", "tomato.jpg", "image/jpeg"))

        assert len(events) >= 5
        # 1. vision analyzing trace event
        assert events[0]["type"] == "trace"
        assert events[0].get("event") == "vision_analyzing" or events[0].get("stage") == "vision_analyzing"
        assert events[0]["payload"]["filename"] == "tomato.jpg"

        # 2. immediate diagnosis prediction
        assert events[1]["type"] == "diagnosis"
        assert events[1]["payload"]["crop"] == "tomato"
        assert events[1]["payload"]["disease"] == "early blight"
        assert events[1]["payload"]["confidence"] == pytest.approx(0.95)

        # 3. retrieval trace
        assert events[2]["type"] == "trace"
        assert events[2].get("event") == "retrieval_completed" or events[2].get("stage") == "retrieval_completed"
        assert events[2]["payload"]["chunks_count"] == 1

        # 4. answer chunk events
        chunk_events = [e for e in events if e["type"] == "answer_chunk"]
        assert len(chunk_events) > 0
        assert "token" in chunk_events[0]["payload"]

        # 5. final done event
        done_event = events[-1]
        assert done_event["type"] == "done"
        assert isinstance(done_event["payload"], ChatResponse)
        assert done_event["payload"].tool_used == "diagnose"
        assert done_event["payload"].diagnosis.crop == "tomato"

    def test_chat_service_stream_diagnose_error_event(self, monkeypatch):
        def _raise_error(*a, **k):
            raise VisionServiceError("Could not reach LeafSense at http://localhost:8001")

        monkeypatch.setattr("app.services.rag_service.diagnose_image", _raise_error)

        service = ChatService(vector_store=None, llm_client=FakeLLMClient())
        events = list(service.stream_diagnose(b"fake-image", "leaf.jpg", "image/jpeg"))

        assert len(events) == 2
        assert events[0]["type"] == "trace"
        assert events[1]["type"] == "error"
        assert "Could not reach LeafSense" in events[1]["detail"]["message"]


# ---------------------------------------------------------------------------
# API Route Tests: POST /api/v1/chat/diagnose with TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def diagnose_api_client(monkeypatch, tmp_path):
    """TestClient wired for /chat/diagnose with mock vector store and LLM."""
    fake_llm = FakeLLMClient("Recommended treatment for early blight.")
    vector_store = FAISSVectorStore(
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    vector_store.create_index(dimension=8)
    vector_store.add_embeddings(
        [
            EmbeddedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                embedding=[1.0] + [0.0] * 7,
                metadata={
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "source": "manual",
                    "text": "Early blight causes dark concentric rings on leaves.",
                },
            )
        ]
    )

    monkeypatch.setattr("app.api.v1.routes.query.get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.api.v1.routes.documents.get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.api.v1.routes.query.get_llm_client", lambda: fake_llm)
    monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda q: [1.0] + [0.0] * 7)
    monkeypatch.setattr("app.services.hybrid_search.embed_query", lambda q: [1.0] + [0.0] * 7)

    from app.services.session_store import InMemorySessionStore
    test_session_store = InMemorySessionStore()
    if not HAS_MULTIPART or app is None:
        pytest.skip("python-multipart not installed on host environment")

    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.skipif(not HAS_MULTIPART, reason="python-multipart not installed on host environment")
class TestDiagnoseEndpointE2E:
    def test_post_diagnose_success(self, diagnose_api_client, monkeypatch):
        """Successful leaf upload and diagnosis returns 200 with schema compliance."""
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse(
                {"class": "Tomato___Early_blight", "confidence": 0.94}
            ),
        )

        response = diagnose_api_client.post(
            "/chat/diagnose",
            headers=VALID_HEADERS,
            files={"image": ("leaf.jpg", b"fake-jpeg-image-bytes", "image/jpeg")},
            data={"query": "What treatment is needed?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert_matches_schema(ChatResponse, body)
        assert body["tool_used"] == "diagnose"
        assert body["diagnosis"]["crop"] == "tomato"
        assert body["diagnosis"]["disease"] == "early blight"
        assert body["diagnosis"]["confidence"] == pytest.approx(0.94)
        assert body["diagnosis"]["low_confidence"] is False

    def test_post_diagnose_stream_success(self, diagnose_api_client, monkeypatch):
        """Streaming diagnose endpoint returns SSE event-stream with real-time chunks."""
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse(
                {"class": "Tomato___Early_blight", "confidence": 0.94}
            ),
        )

        class StreamingLLM(FakeLLMClient):
            def generate_stream(self, prompt: str):
                yield "Treatment: "
                yield "Apply copper spray."

        monkeypatch.setattr("app.api.v1.routes.query.get_llm_client", lambda: StreamingLLM())

        import json

        response = diagnose_api_client.post(
            "/chat/diagnose/stream",
            headers=VALID_HEADERS,
            files={"image": ("leaf.jpg", b"fake-jpeg-image-bytes", "image/jpeg")},
            data={"query": "What fungicide?"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                event_json = line[len("data: "):].strip()
                if event_json:
                    events.append(json.loads(event_json))

        assert len(events) >= 4
        event_types = [e.get("type") for e in events]
        assert "trace" in event_types
        assert "diagnosis" in event_types
        assert "done" in event_types

        diag_event = next(e for e in events if e.get("type") == "diagnosis")
        assert diag_event["payload"]["crop"] == "tomato"
        assert diag_event["payload"]["disease"] == "early blight"

        done_event = next(e for e in events if e.get("type") == "done")
        assert done_event["payload"]["diagnosis"]["crop"] == "tomato"
        assert "session_id" in done_event["payload"]

    def test_post_diagnose_low_confidence(self, diagnose_api_client, monkeypatch):
        """Low-confidence diagnosis returns 200 with low_confidence flag True."""
        monkeypatch.setattr(settings, "vision_confidence_threshold", 0.60)
        monkeypatch.setattr(
            vision_client_module.httpx,
            "post",
            lambda *a, **k: _FakeResponse(
                {"class": "Potato___Early_blight", "confidence": 0.38}
            ),
        )

        response = diagnose_api_client.post(
            "/chat/diagnose",
            headers=VALID_HEADERS,
            files={"image": ("potato.png", b"fake-png-image-bytes", "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert_matches_schema(ChatResponse, body)
        assert body["diagnosis"]["low_confidence"] is True
        assert body["diagnosis"]["crop"] == "potato"

    def test_post_diagnose_invalid_mime_type_rejected(self, diagnose_api_client):
        """Uploading a non-image file (e.g. text/plain) is rejected with 415 or 400."""
        response = diagnose_api_client.post(
            "/chat/diagnose",
            headers=VALID_HEADERS,
            files={"image": ("notes.txt", b"plain text data", "text/plain")},
        )

        assert response.status_code in (400, 415)
        assert "image" in response.json()["detail"].lower()

    def test_post_diagnose_empty_image_rejected(self, diagnose_api_client):
        """Uploading an empty file is rejected with 400."""
        response = diagnose_api_client.post(
            "/chat/diagnose",
            headers=VALID_HEADERS,
            files={"image": ("empty.jpg", b"", "image/jpeg")},
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_post_diagnose_leafsense_unavailable_returns_502(
        self, diagnose_api_client, monkeypatch
    ):
        """When LeafSense is down / connection refused, endpoint returns 502 with VISION_SERVICE_ERROR."""
        def _raise_connect_error(*a, **k):
            raise httpx.ConnectError("Connection refused at http://localhost:8001")

        monkeypatch.setattr(vision_client_module.httpx, "post", _raise_connect_error)

        response = diagnose_api_client.post(
            "/chat/diagnose",
            headers=VALID_HEADERS,
            files={"image": ("leaf.jpg", b"fake-jpeg-image-bytes", "image/jpeg")},
        )

        assert response.status_code == 502
        body = response.json()
        assert "VISION_SERVICE_ERROR" in body.get("error_code", "") or "LeafSense" in body.get(
            "detail", ""
        )

    def test_post_diagnose_leafsense_timeout_returns_502(
        self, diagnose_api_client, monkeypatch
    ):
        """When LeafSense times out, endpoint returns 502 with VISION_SERVICE_ERROR."""
        def _raise_timeout(*a, **k):
            raise httpx.TimeoutException("Request timed out")

        monkeypatch.setattr(vision_client_module.httpx, "post", _raise_timeout)

        response = diagnose_api_client.post(
            "/chat/diagnose",
            headers=VALID_HEADERS,
            files={"image": ("leaf.jpg", b"fake-jpeg-image-bytes", "image/jpeg")},
        )

        assert response.status_code == 502
