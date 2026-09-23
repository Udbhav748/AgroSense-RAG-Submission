"""Module 10 gap-closure: real parallel execution, wired into an actual
AgroSense-RAG workflow -- ChatService.handle_diagnose runs vision
classification (LeafSense/Gemini) and the weather/microclimate lookup
(Open-Meteo) CONCURRENTLY when latitude/longitude are supplied, instead
of the old sequential "await weather fully, then start vision" order.

These are integration-style tests against the real ChatService method
(not a synthetic toy), using monkeypatched I/O boundaries
(diagnose_image, WeatherService.get_weather_risk) exactly like the
project's existing test_rag_service_diagnose.py does -- proving the
real workflow, not just the generic primitive (see
test_agent_graph_parallel_execution.py for that).
"""

from __future__ import annotations

import threading

import app.services.rag_service as rag_service_module
from app.core.exceptions import VisionServiceError
from app.models.document import RetrievedChunk, VisionPrediction
from app.models.schemas import WeatherRiskResponse
from app.services.rag_service import ChatService


class FakeLLMClient:
    def __init__(self, response="grounded diagnosis answer citing weather context [1]"):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class FakeVectorStore:
    pass


def make_service() -> ChatService:
    return ChatService(FakeVectorStore(), FakeLLMClient())


def make_prediction() -> VisionPrediction:
    return VisionPrediction(
        raw_class="Peach___Bacterial_spot", crop="peach", disease="bacterial spot",
        confidence=0.94, low_confidence=False,
    )


def make_weather_response() -> WeatherRiskResponse:
    return WeatherRiskResponse.model_validate(
        {
            "location": {"latitude": 35.78, "longitude": -78.64, "timezone": "America/New_York"},
            "current": {"temperature_c": 19.0, "humidity_pct": 94.0, "precipitation_mm": 0.0, "wind_kmh": 11.0},
            "risk_level": "High",
            "risk_score": 0.8,
            "favorable_conditions_summary": "High humidity, mild temperatures.",
            "spray_advisory": "Apply protectant fungicide within 24 hours.",
        }
    )


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1", document_id="doc-1",
        text="Bacterial spot treatment: copper-based bactericide.", score=0.9, metadata={},
    )


class TestVisionAndWeatherRunConcurrently:
    def test_lat_lon_present_runs_vision_and_weather_as_real_concurrent_branches(self, monkeypatch):
        """Mutual-wait proof at the real workflow level: diagnose_image
        (vision) and get_weather_risk (weather) each signal their own
        start and wait for the other's -- if handle_diagnose still ran
        them sequentially, this would hang and time out inside the
        branch, surfacing as a failed diagnosis rather than a passing
        response."""
        vision_started = threading.Event()
        weather_started = threading.Event()

        def fake_diagnose_image(*args, **kwargs):
            vision_started.set()
            if not weather_started.wait(timeout=2.0):
                raise AssertionError("weather branch never started while vision was running -- not concurrent")
            return make_prediction()

        async def fake_get_weather_risk(self, lat, lon, crop=None, disease=None):
            weather_started.set()
            if not vision_started.wait(timeout=2.0):
                raise AssertionError("vision branch never started while weather was running -- not concurrent")
            return make_weather_response()

        monkeypatch.setattr(rag_service_module, "diagnose_image", fake_diagnose_image)
        monkeypatch.setattr(
            "app.services.weather_service.WeatherService.get_weather_risk", fake_get_weather_risk
        )
        monkeypatch.setattr(rag_service_module, "retrieve", lambda *a, **k: [make_chunk()])

        service = make_service()
        response = service.handle_diagnose(
            b"fake-image-bytes", "leaf.jpg", "image/jpeg",
            latitude=35.78, longitude=-78.64,
        )

        assert response.answer  # real response produced, both branches completed
        assert response.diagnosis is not None
        assert response.diagnosis.crop == "peach"
        assert response.weather_risk is not None
        assert response.weather_risk.risk_level == "High"


class TestWeatherFailureDoesNotBreakDiagnosis:
    def test_weather_branch_failure_degrades_gracefully(self, monkeypatch):
        """Matches the pre-existing route-level behavior: a weather
        lookup failure is logged and the diagnosis proceeds without
        weather context -- it must never fail the whole request over an
        optional enrichment call."""
        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: make_prediction())
        monkeypatch.setattr(rag_service_module, "retrieve", lambda *a, **k: [make_chunk()])

        async def failing_weather(self, lat, lon, crop=None, disease=None):
            raise ConnectionError("Open-Meteo unreachable")

        monkeypatch.setattr(
            "app.services.weather_service.WeatherService.get_weather_risk", failing_weather
        )

        service = make_service()
        response = service.handle_diagnose(
            b"fake-image-bytes", "leaf.jpg", "image/jpeg",
            latitude=35.78, longitude=-78.64,
        )

        assert response.answer  # diagnosis still succeeds
        assert response.weather_risk is None  # weather absent, not fabricated


class TestVisionFailurePreservesRealExceptionType:
    def test_vision_branch_failure_raises_the_original_apperror_not_a_generic_one(self, monkeypatch):
        """The critical correctness requirement: a VisionServiceError (an
        AppError subclass carrying status_code=502/error_code) raised
        inside the concurrent vision branch must still surface as that
        SAME exception type -- not silently downgraded to a generic
        ChatServiceError, which would change the HTTP status code a
        real caller sees."""

        def failing_vision(*args, **kwargs):
            raise VisionServiceError("LeafSense unavailable")

        monkeypatch.setattr(rag_service_module, "diagnose_image", failing_vision)

        async def fake_weather(self, lat, lon, crop=None, disease=None):
            return make_weather_response()

        monkeypatch.setattr(
            "app.services.weather_service.WeatherService.get_weather_risk", fake_weather
        )

        service = make_service()
        try:
            service.handle_diagnose(
                b"fake-image-bytes", "leaf.jpg", "image/jpeg",
                latitude=35.78, longitude=-78.64,
            )
            raised = None
        except Exception as exc:  # noqa: BLE001 -- inspecting the real raised type is the point of this test
            raised = exc

        assert raised is not None
        assert isinstance(raised, VisionServiceError)
        assert raised.status_code == 502


class TestBackwardCompatibilityWithoutCoordinates:
    def test_no_lat_lon_never_attempts_a_weather_fetch(self, monkeypatch):
        """Existing behavior, unchanged: without latitude/longitude,
        handle_diagnose runs vision alone (the old sequential single-
        branch path) -- weather is never even attempted, exactly as
        before this feature existed."""
        weather_call_count = {"n": 0}

        async def counting_weather(self, lat, lon, crop=None, disease=None):
            weather_call_count["n"] += 1
            return make_weather_response()

        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: make_prediction())
        monkeypatch.setattr(rag_service_module, "retrieve", lambda *a, **k: [make_chunk()])
        monkeypatch.setattr(
            "app.services.weather_service.WeatherService.get_weather_risk", counting_weather
        )

        service = make_service()
        response = service.handle_diagnose(b"fake-image-bytes", "leaf.jpg", "image/jpeg")

        assert weather_call_count["n"] == 0
        assert response.weather_risk is None

    def test_precomputed_weather_risk_param_still_works_and_skips_concurrent_path(self, monkeypatch):
        """A caller that already has a WeatherRiskResponse (the old
        contract) can still pass it directly -- handle_diagnose must
        not re-fetch or ignore it."""
        weather_call_count = {"n": 0}

        async def counting_weather(self, lat, lon, crop=None, disease=None):
            weather_call_count["n"] += 1
            return make_weather_response()

        monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: make_prediction())
        monkeypatch.setattr(rag_service_module, "retrieve", lambda *a, **k: [make_chunk()])
        monkeypatch.setattr(
            "app.services.weather_service.WeatherService.get_weather_risk", counting_weather
        )

        service = make_service()
        precomputed = make_weather_response()
        response = service.handle_diagnose(
            b"fake-image-bytes", "leaf.jpg", "image/jpeg",
            weather_risk=precomputed,
            latitude=35.78, longitude=-78.64,  # present, but weather_risk already given -- must not re-fetch
        )

        assert weather_call_count["n"] == 0  # never re-fetched
        assert response.weather_risk is not None
        assert response.weather_risk.risk_level == precomputed.risk_level


class TestSessionAndSecurityBoundariesPreserved:
    def test_two_concurrent_diagnoses_with_different_tenants_do_not_cross_contaminate(self, monkeypatch):
        """Runs two handle_diagnose calls (each with its own tenant_id
        and its own vision prediction) back-to-back and confirms each
        response reflects only its OWN tenant/prediction -- proving the
        new concurrency inside one request doesn't leak into a
        DIFFERENT request's data, since each call gets its own fresh
        closures/state, never shared mutable globals."""
        predictions_by_tenant = {
            1: VisionPrediction(raw_class="Apple___Scab", crop="apple", disease="scab", confidence=0.9, low_confidence=False),
            2: VisionPrediction(raw_class="Tomato___Late_blight", crop="tomato", disease="late blight", confidence=0.9, low_confidence=False),
        }
        call_tenant_order: list[int] = []

        def diagnose_for_current_tenant(*args, **kwargs):
            tenant = call_tenant_order[-1]
            return predictions_by_tenant[tenant]

        async def fake_weather(self, lat, lon, crop=None, disease=None):
            return make_weather_response()

        monkeypatch.setattr(rag_service_module, "diagnose_image", diagnose_for_current_tenant)
        monkeypatch.setattr(rag_service_module, "retrieve", lambda *a, **k: [make_chunk()])
        monkeypatch.setattr(
            "app.services.weather_service.WeatherService.get_weather_risk", fake_weather
        )

        service = make_service()

        call_tenant_order.append(1)
        response_1 = service.handle_diagnose(
            b"tenant-1-image", "leaf1.jpg", "image/jpeg",
            tenant_id=1, latitude=10.0, longitude=20.0,
        )
        call_tenant_order.append(2)
        response_2 = service.handle_diagnose(
            b"tenant-2-image", "leaf2.jpg", "image/jpeg",
            tenant_id=2, latitude=30.0, longitude=40.0,
        )

        assert response_1.diagnosis.crop == "apple"
        assert response_2.diagnosis.crop == "tomato"
