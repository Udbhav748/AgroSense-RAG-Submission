"""Unit and integration tests for WeatherService, disease infection risk modeling,
spray advisory generation, and microclimate RAG integration.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.models.document import EmbeddedChunk, RetrievedChunk, VisionPrediction
from app.models.schemas import ChatResponse, WeatherRiskResponse
from app.services import weather_service as weather_module
from app.services.rag_service import ChatService
from app.services.weather_service import (
    WeatherService,
    calculate_disease_risk,
    generate_spray_advisory,
)
from tests.conftest import assert_matches_schema

try:
    from app.main import app

    HAS_APP = True
except (ImportError, RuntimeError):
    app = None
    HAS_APP = False

VALID_HEADERS = {"X-API-Key": settings.api_key}


class _FakeAsyncResponse:
    """Mock async httpx response for Open-Meteo API."""

    def __init__(self, json_data: dict[str, Any], status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://api.open-meteo.com/v1/forecast")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("Open-Meteo API Error", request=request, response=response)

    def json(self):
        return self._json_data


class FakeVectorStore:
    """Mock vector store for retrieval in diagnose tests."""

    def search(self, query_vector, top_k: int = 3, min_score: float = 0.0, **kwargs):
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-tomato-blight",
                text="Late blight is caused by Phytophthora infestans. Smith periods indicate high infection risk.",
                score=0.92,
                metadata={"rerank_score": 0.92},
            )
        ]

    def search_bm25(self, query: str, top_k: int = 3, **kwargs):
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-tomato-blight",
                text="Late blight is caused by Phytophthora infestans. Smith periods indicate high infection risk.",
                score=0.92,
                metadata={"rerank_score": 0.92},
            )
        ]


class FakeLLMClient:
    """Mock LLM client capturing prompt and returning structured answer."""

    def __init__(self, response_text: str = "Apply chlorothalonil or copper fungicide before rain."):
        self.response_text = response_text
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response_text

    def generate_stream(self, prompt: str):
        self.calls.append(prompt)
        yield "Apply chlorothalonil or copper fungicide before rain."


# ---------------------------------------------------------------------------
# Unit Tests: Pathogen Infection Risk Models & Spray Advisories
# ---------------------------------------------------------------------------


class TestWeatherRiskModeling:
    def test_smith_period_high_humidity_late_blight(self):
        """12 consecutive hours with RH >= 90% and Temp 15-22°C triggers Smith Period."""
        # 12 consecutive hours meeting criteria (18°C, 95% RH) followed by normal weather
        hourly_temps = [18.0] * 12 + [25.0] * 12
        hourly_rh = [95.0] * 12 + [60.0] * 12
        hourly_precip_prob = [40.0] * 24

        weather_data = {
            "latitude": 35.7796,
            "longitude": -78.6382,
            "timezone": "America/New_York",
            "current": {
                "temperature_2m": 18.5,
                "relative_humidity_2m": 94.0,
                "precipitation": 0.0,
                "wind_speed_10m": 8.0,
            },
            "hourly": {
                "temperature_2m": hourly_temps,
                "relative_humidity_2m": hourly_rh,
                "precipitation_probability": hourly_precip_prob,
            },
        }

        response = calculate_disease_risk(weather_data, crop="tomato", disease="Late_blight")

        assert response.risk_level in ["High", "Critical"]
        assert response.risk_score >= 0.85
        assert "Smith Period" in response.favorable_conditions_summary
        assert "Late Blight" in response.favorable_conditions_summary
        assert response.current.temperature_c == pytest.approx(18.5)
        assert response.current.humidity_pct == pytest.approx(94.0)

    def test_smith_period_sub_threshold_moderate_risk(self):
        """6 consecutive hours with RH >= 90% and Temp 15-22°C gives moderate risk."""
        hourly_temps = [19.0] * 6 + [26.0] * 18
        hourly_rh = [92.0] * 6 + [55.0] * 18

        weather_data = {
            "latitude": 35.7796,
            "longitude": -78.6382,
            "timezone": "America/New_York",
            "current": {
                "temperature_2m": 24.0,
                "relative_humidity_2m": 60.0,
                "precipitation": 0.0,
                "wind_speed_10m": 10.0,
            },
            "hourly": {
                "temperature_2m": hourly_temps,
                "relative_humidity_2m": hourly_rh,
                "precipitation_probability": [10.0] * 24,
            },
        }

        response = calculate_disease_risk(weather_data, crop="potato", disease="late blight")

        assert response.risk_level == "Moderate"
        assert 0.5 <= response.risk_score < 0.7
        assert "approaching Smith Period" in response.favorable_conditions_summary

    def test_powdery_mildew_dry_conditions(self):
        """Moderate-to-high humidity (70-85%) with dry leaf canopy and 20-28°C triggers Powdery Mildew."""
        hourly_temps = [24.0] * 12 + [22.0] * 12
        hourly_rh = [78.0] * 12 + [75.0] * 12

        weather_data = {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timezone": "America/New_York",
            "current": {
                "temperature_2m": 24.5,
                "relative_humidity_2m": 76.0,
                "precipitation": 0.0,
                "wind_speed_10m": 6.0,
            },
            "hourly": {
                "temperature_2m": hourly_temps,
                "relative_humidity_2m": hourly_rh,
                "precipitation_probability": [5.0] * 24,
            },
        }

        response = calculate_disease_risk(weather_data, crop="squash", disease="Powdery_mildew")

        assert response.risk_level in ["High", "Moderate"]
        assert response.risk_score >= 0.75
        assert "Powdery Mildew" in response.favorable_conditions_summary
        assert "dry leaf" in response.favorable_conditions_summary

    def test_bacterial_spot_driving_rain(self):
        """Driving rain (>5mm) combined with temperatures > 24°C triggers Critical Bacterial Spot risk."""
        weather_data = {
            "latitude": 30.0,
            "longitude": -90.0,
            "timezone": "America/Chicago",
            "current": {
                "temperature_2m": 26.5,
                "relative_humidity_2m": 92.0,
                "precipitation": 7.5,
                "wind_speed_10m": 18.0,
            },
            "hourly": {
                "temperature_2m": [26.0] * 24,
                "relative_humidity_2m": [90.0] * 24,
                "precipitation_probability": [90.0] * 24,
            },
        }

        response = calculate_disease_risk(weather_data, crop="peach", disease="Bacterial_spot")

        assert response.risk_level == "Critical"
        assert response.risk_score >= 0.90
        assert "Bacterial Spot" in response.favorable_conditions_summary
        assert "Driving rainfall" in response.favorable_conditions_summary

    def test_foliar_rust_risk(self):
        """High humidity (>80%) and temperatures between 16-24°C trigger Foliar Rust."""
        hourly_temps = [20.0] * 12 + [19.0] * 12
        hourly_rh = [86.0] * 12 + [88.0] * 12

        weather_data = {
            "latitude": 42.0,
            "longitude": -93.0,
            "timezone": "America/Chicago",
            "current": {
                "temperature_2m": 19.5,
                "relative_humidity_2m": 87.0,
                "precipitation": 0.0,
                "wind_speed_10m": 7.0,
            },
            "hourly": {
                "temperature_2m": hourly_temps,
                "relative_humidity_2m": hourly_rh,
                "precipitation_probability": [15.0] * 24,
            },
        }

        response = calculate_disease_risk(weather_data, crop="corn", disease="Common_rust")

        assert response.risk_level in ["High", "Moderate"]
        assert response.risk_score >= 0.80
        assert "Foliar Rust" in response.favorable_conditions_summary

    def test_unfavorable_benign_conditions(self):
        """Dry, cool weather yields Low pathogen infection risk."""
        hourly_temps = [12.0] * 24
        hourly_rh = [45.0] * 24

        weather_data = {
            "latitude": 37.0,
            "longitude": -120.0,
            "timezone": "America/Los_Angeles",
            "current": {
                "temperature_2m": 14.0,
                "relative_humidity_2m": 42.0,
                "precipitation": 0.0,
                "wind_speed_10m": 8.0,
            },
            "hourly": {
                "temperature_2m": hourly_temps,
                "relative_humidity_2m": hourly_rh,
                "precipitation_probability": [0.0] * 24,
            },
        }

        response = calculate_disease_risk(weather_data, crop="apple", disease="Apple_scab")

        assert response.risk_level == "Low"
        assert response.risk_score <= 0.35


class TestSprayAdvisory:
    def test_wind_drift_warning_when_wind_exceeds_15kmh(self):
        """Wind speeds > 15 km/h trigger high drift risk advisory."""
        advisory = generate_spray_advisory(
            current_temp=22.0,
            current_rh=60.0,
            current_precip=0.0,
            current_wind=19.4,
            hourly_precip_prob=[10.0, 10.0, 10.0, 10.0],
        )

        assert "High drift risk" in advisory
        assert "15 km/h" in advisory
        assert "19.4 km/h" in advisory

    def test_active_precipitation_spray_warning(self):
        """Active rainfall triggers wash-off spray postponement advisory."""
        advisory = generate_spray_advisory(
            current_temp=20.0,
            current_rh=85.0,
            current_precip=2.4,
            current_wind=8.0,
            hourly_precip_prob=[80.0, 80.0, 80.0, 80.0],
        )

        assert "Unfavorable spray window" in advisory
        assert "Active precipitation" in advisory
        assert "2.4 mm" in advisory

    def test_impending_rain_spray_warning(self):
        """High rain probability in next 4 hours triggers marginal spray window advisory."""
        advisory = generate_spray_advisory(
            current_temp=21.0,
            current_rh=70.0,
            current_precip=0.0,
            current_wind=10.0,
            hourly_precip_prob=[75.0, 80.0, 85.0, 90.0],
        )

        assert "Marginal spray window" in advisory
        assert "rainfast contact time" in advisory

    def test_high_temperature_warning(self):
        """Temperatures > 32°C trigger volatilization warning."""
        advisory = generate_spray_advisory(
            current_temp=34.5,
            current_rh=40.0,
            current_precip=0.0,
            current_wind=5.0,
            hourly_precip_prob=[0.0] * 4,
        )

        assert "High temperature warning" in advisory
        assert "34.5°C" in advisory
        assert "volatilization" in advisory

    def test_optimal_spray_window(self):
        """Calm wind, no rain, and moderate temperature give optimal spray window."""
        advisory = generate_spray_advisory(
            current_temp=21.0,
            current_rh=65.0,
            current_precip=0.0,
            current_wind=7.5,
            hourly_precip_prob=[5.0, 5.0, 10.0, 10.0],
        )

        assert "Optimal spray window" in advisory
        assert "Calm winds" in advisory


# ---------------------------------------------------------------------------
# Unit Tests: WeatherService HTTP Client & Fallback
# ---------------------------------------------------------------------------


class TestWeatherServiceAsync:
    @pytest.mark.asyncio
    async def test_fetch_weather_data_success(self, monkeypatch):
        """Test successful Open-Meteo API query and data parsing."""
        fake_api_data = {
            "latitude": 35.78,
            "longitude": -78.64,
            "timezone": "America/New_York",
            "current": {
                "temperature_2m": 22.1,
                "relative_humidity_2m": 71.0,
                "precipitation": 0.0,
                "wind_speed_10m": 9.2,
            },
            "hourly": {
                "temperature_2m": [22.0, 21.5, 21.0],
                "relative_humidity_2m": [70.0, 75.0, 80.0],
                "precipitation_probability": [10.0, 15.0, 20.0],
            },
        }

        async def _fake_get(*args, **kwargs):
            return _FakeAsyncResponse(fake_api_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

        service = WeatherService()
        result = await service.fetch_weather_data(lat=35.78, lon=-78.64)

        assert result["latitude"] == 35.78
        assert result["current"]["temperature_2m"] == 22.1

    @pytest.mark.asyncio
    async def test_get_weather_risk_fallback_on_network_timeout(self, monkeypatch):
        """Network timeout or connection failure returns graceful fallback response."""

        async def _fake_get_timeout(*args, **kwargs):
            raise httpx.ConnectTimeout("Connection timed out querying Open-Meteo API")

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get_timeout)

        service = WeatherService()
        response = await service.get_weather_risk(lat=35.78, lon=-78.64)

        assert response.risk_level == "Low"
        assert response.risk_score == 0.0
        assert "temporarily unavailable" in response.favorable_conditions_summary
        assert response.location["latitude"] == 35.78
        assert response.location["longitude"] == -78.64


# ---------------------------------------------------------------------------
# Integration Tests: FastAPI Endpoints (/weather/risk, /chat/diagnose)
# ---------------------------------------------------------------------------


class TestWeatherEndpointsIntegration:
    @pytest.fixture(autouse=True)
    def _setup_app(self):
        if not HAS_APP or app is None:
            pytest.skip("FastAPI app or multipart support unavailable.")
        self.client = TestClient(app)

    def test_get_weather_risk_endpoint(self, monkeypatch):
        """GET /weather/risk returns conforming WeatherRiskResponse."""
        fake_api_data = {
            "latitude": 35.78,
            "longitude": -78.64,
            "timezone": "America/New_York",
            "current": {
                "temperature_2m": 18.0,
                "relative_humidity_2m": 92.0,
                "precipitation": 0.0,
                "wind_speed_10m": 8.0,
            },
            "hourly": {
                "temperature_2m": [18.0] * 12 + [24.0] * 12,
                "relative_humidity_2m": [92.0] * 12 + [60.0] * 12,
                "precipitation_probability": [20.0] * 24,
            },
        }

        async def _fake_get(*args, **kwargs):
            return _FakeAsyncResponse(fake_api_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

        response = self.client.get(
            "/weather/risk?lat=35.78&lon=-78.64&crop=tomato&disease=Late_blight",
            headers=VALID_HEADERS,
        )

        assert response.status_code == 200
        data = response.json()
        assert_matches_schema(WeatherRiskResponse, data)
        assert data["risk_level"] in ["High", "Critical"]
        assert data["location"]["latitude"] == pytest.approx(35.78)
        assert data["location"]["longitude"] == pytest.approx(-78.64)
        assert data["current"]["temperature_c"] == pytest.approx(18.0)

    def test_chat_diagnose_with_coordinates_injects_weather_context(self, monkeypatch):
        """POST /chat/diagnose with lat/lon injects weather context into prompt and returns weather_risk."""
        from app.api.v1.routes import query as query_route

        # Mock Vision
        monkeypatch.setattr(
            query_route,
            "validate_image_upload",
            lambda image, contents: None,
        )
        monkeypatch.setattr(
            "app.services.rag_service.diagnose_image",
            lambda *a, **k: VisionPrediction(
                raw_class="Tomato___Late_blight",
                crop="tomato",
                disease="Late blight",
                confidence=0.96,
                low_confidence=False,
            ),
        )

        # Mock Vector store and LLM
        fake_llm = FakeLLMClient("Diagnosis: Late Blight confirmed. High humidity elevates infection pressure.")
        fake_store = FakeVectorStore()
        chat_service = ChatService(fake_store, fake_llm)
        monkeypatch.setattr(query_route, "get_chat_service", lambda: chat_service)

        # Mock Weather API
        fake_api_data = {
            "latitude": 35.78,
            "longitude": -78.64,
            "timezone": "America/New_York",
            "current": {
                "temperature_2m": 19.0,
                "relative_humidity_2m": 94.0,
                "precipitation": 0.0,
                "wind_speed_10m": 11.0,
            },
            "hourly": {
                "temperature_2m": [18.0] * 12 + [24.0] * 12,
                "relative_humidity_2m": [94.0] * 12 + [50.0] * 12,
                "precipitation_probability": [10.0] * 24,
            },
        }

        async def _fake_get(*args, **kwargs):
            return _FakeAsyncResponse(fake_api_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

        response = self.client.post(
            "/chat/diagnose",
            files={"image": ("leaf.jpg", b"fake-jpg-bytes", "image/jpeg")},
            data={
                "latitude": "35.78",
                "longitude": "-78.64",
                "query": "What fungicide should I spray?",
            },
            headers=VALID_HEADERS,
        )

        assert response.status_code == 200
        data = response.json()
        assert_matches_schema(ChatResponse, data)
        assert data["weather_risk"] is not None
        assert data["weather_risk"]["risk_level"] in ["High", "Critical"]
        assert data["weather_risk"]["location"]["latitude"] == pytest.approx(35.78)

        # Verify weather was injected into LLM prompt
        assert len(fake_llm.calls) > 0
        prompt_text = fake_llm.calls[0]
        assert "LOCAL FIELD MICROCLIMATE & WEATHER INTELLIGENCE" in prompt_text
        assert "35.78" in prompt_text

    def test_chat_diagnose_stream_with_coordinates(self, monkeypatch):
        """POST /chat/diagnose/stream yields weather_assessed event and attaches weather_risk to done payload."""
        from app.api.v1.routes import query as query_route

        monkeypatch.setattr(
            query_route,
            "validate_image_upload",
            lambda image, contents: None,
        )
        monkeypatch.setattr(
            "app.services.rag_service.diagnose_image",
            lambda *a, **k: VisionPrediction(
                raw_class="Tomato___Late_blight",
                crop="tomato",
                disease="Late blight",
                confidence=0.95,
                low_confidence=False,
            ),
        )

        fake_llm = FakeLLMClient()
        fake_store = FakeVectorStore()
        chat_service = ChatService(fake_store, fake_llm)
        monkeypatch.setattr(query_route, "get_chat_service", lambda: chat_service)

        fake_api_data = {
            "latitude": 35.78,
            "longitude": -78.64,
            "timezone": "America/New_York",
            "current": {
                "temperature_2m": 19.0,
                "relative_humidity_2m": 93.0,
                "precipitation": 0.0,
                "wind_speed_10m": 6.0,
            },
            "hourly": {
                "temperature_2m": [18.5] * 12 + [23.0] * 12,
                "relative_humidity_2m": [93.0] * 12 + [55.0] * 12,
                "precipitation_probability": [15.0] * 24,
            },
        }

        async def _fake_get(*args, **kwargs):
            return _FakeAsyncResponse(fake_api_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

        response = self.client.post(
            "/chat/diagnose/stream",
            files={"image": ("leaf.jpg", b"fake-jpg-bytes", "image/jpeg")},
            data={
                "latitude": "35.78",
                "longitude": "-78.64",
            },
            headers=VALID_HEADERS,
        )

        assert response.status_code == 200
        text = response.text
        assert "weather_assessed" in text
        assert "Smith Period" in text or "Late Blight" in text
