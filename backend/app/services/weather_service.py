"""Agronomy Microclimate & Weather Intelligence Service.

Provides real-time agro-meteorological forecasting via Open-Meteo API
and pathogen infection risk modeling (Smith Periods, Powdery Mildew,
Bacterial Spot, Foliar Rust) alongside chemical spray window advisories.
"""

import logging
from typing import Any, Literal

import httpx

from app.models.schemas import CurrentWeather, WeatherRiskResponse

logger = logging.getLogger(__name__)

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"


def generate_spray_advisory(
    current_temp: float,
    current_rh: float,
    current_precip: float,
    current_wind: float,
    hourly_precip_prob: list[float] | None = None,
) -> str:
    """Generate an agronomic spray window and chemical drift advisory."""
    hourly_precip_prob = hourly_precip_prob or []

    if current_wind > 15.0:
        return (
            f"High drift risk: wind exceeds 15 km/h (current: {current_wind:.1f} km/h). "
            "Avoid chemical/fungicide spraying to prevent off-target drift and regulatory non-compliance."
        )
    if current_precip > 0.5:
        return (
            f"Unfavorable spray window: Active precipitation ({current_precip:.1f} mm). "
            "Rain will wash off treatments; postpone application until foliage dries."
        )
    if any(p >= 60.0 for p in hourly_precip_prob[:4]):
        return (
            "Marginal spray window: High probability of incoming rain within the next 4 hours. "
            "Ensure adequate rainfast contact time (minimum 2-4 hours) before application."
        )
    if current_temp > 32.0:
        return (
            f"High temperature warning: Ambient temperature is {current_temp:.1f}°C (>32°C). "
            "High risk of chemical volatilization and crop phytotoxicity; delay spraying until cooler evening or morning hours."
        )
    if current_temp < 5.0:
        return (
            f"Low temperature warning: Ambient temperature is {current_temp:.1f}°C (<5°C). "
            "Reduced systemic product uptake and plant metabolism."
        )

    return (
        "Optimal spray window in next 4-6 hours: Calm winds (<15 km/h) and dry conditions forecast. "
        "Favorable for protective or systemic fungicide/bactericide application."
    )


def calculate_disease_risk(
    weather_data: dict[str, Any],
    crop: str | None = None,
    disease: str | None = None,
) -> WeatherRiskResponse:
    """Calculate pathogen infection risk from meteorological data.

    Models:
    - Late Blight / Downy Mildew (Smith Periods): RH >= 90% and Temp 15-22°C for >= 10 consecutive hours.
    - Powdery Mildew: RH 70-85%, dry canopy (no rain), Temp 20-28°C.
    - Bacterial Spot: Driving rain (>5mm) and Temp > 24°C.
    - Foliar Rust: RH > 80% and Temp 16-24°C.
    """
    location_data = {
        "latitude": float(weather_data.get("latitude", 0.0)),
        "longitude": float(weather_data.get("longitude", 0.0)),
        "timezone": str(weather_data.get("timezone", "UTC")),
    }

    current_raw = weather_data.get("current") or {}
    cur_temp = float(current_raw.get("temperature_2m", 20.0))
    cur_rh = float(current_raw.get("relative_humidity_2m", 50.0))
    cur_precip = float(current_raw.get("precipitation", 0.0))
    cur_wind = float(current_raw.get("wind_speed_10m", 5.0))

    hourly_raw = weather_data.get("hourly") or {}
    hourly_temps = [float(t) for t in (hourly_raw.get("temperature_2m") or [])]
    hourly_rh = [float(h) for h in (hourly_raw.get("relative_humidity_2m") or [])]
    hourly_precip_prob = [float(p) for p in (hourly_raw.get("precipitation_probability") or [])]

    current_obj = CurrentWeather(
        temperature_c=cur_temp,
        humidity_pct=cur_rh,
        precipitation_mm=cur_precip,
        wind_kmh=cur_wind,
    )

    # 1. Smith Period Model (Late Blight / Downy Mildew)
    max_consecutive_smith = 0
    current_consecutive_smith = 0
    for t, rh in zip(hourly_temps, hourly_rh, strict=False):
        if rh >= 90.0 and 15.0 <= t <= 22.0:
            current_consecutive_smith += 1
            if current_consecutive_smith > max_consecutive_smith:
                max_consecutive_smith = current_consecutive_smith
        else:
            current_consecutive_smith = 0

    if max_consecutive_smith >= 10:
        smith_score = min(1.0, 0.85 + (max_consecutive_smith - 10) * 0.015)
        smith_level: Literal["Low", "Moderate", "High", "Critical"] = (
            "Critical" if max_consecutive_smith >= 14 else "High"
        )
        smith_summary = (
            f"Smith Period criteria triggered: {max_consecutive_smith} consecutive hours with "
            f"relative humidity >= 90% and temperatures between 15°C and 22°C. "
            "High infection and sporulation pressure for Late Blight / Downy Mildew."
        )
    elif max_consecutive_smith >= 6:
        smith_score = 0.55 + (max_consecutive_smith - 6) * 0.05
        smith_level = "Moderate"
        smith_summary = (
            f"Elevated Late Blight / Downy Mildew risk: {max_consecutive_smith} consecutive hours "
            "of high humidity (>=90%) at 15°C-22°C approaching Smith Period threshold."
        )
    else:
        smith_score = 0.1
        smith_level = "Low"
        smith_summary = (
            "Microclimate is unfavorable for Late Blight / Downy Mildew (no sustained Smith Period detected)."
        )

    # 2. Powdery Mildew Model
    current_powdery = (70.0 <= cur_rh <= 85.0) and (20.0 <= cur_temp <= 28.0) and (cur_precip <= 0.1)
    powdery_hours = sum(
        1 for t, rh in zip(hourly_temps, hourly_rh, strict=False) if 70.0 <= rh <= 85.0 and 20.0 <= t <= 28.0
    )
    if current_powdery or powdery_hours >= 8:
        powdery_score = 0.85 if (current_powdery and powdery_hours >= 6) else 0.75
        powdery_level: Literal["Low", "Moderate", "High", "Critical"] = (
            "High" if powdery_score >= 0.8 else "Moderate"
        )
        powdery_summary = (
            f"Favorable conditions for Powdery Mildew: moderate-to-high humidity (70-85%), "
            f"dry leaf canopy, and warm temperatures (20°C-28°C) observed ({powdery_hours} forecast hours favor conidia development)."
        )
    elif powdery_hours >= 4 or (70.0 <= cur_rh <= 85.0 and 20.0 <= cur_temp <= 28.0):
        powdery_score = 0.55
        powdery_level = "Moderate"
        powdery_summary = (
            f"Moderate Powdery Mildew risk: warm temperatures (20°C-28°C) and moderate humidity recorded for {powdery_hours} hours."
        )
    else:
        powdery_score = 0.1
        powdery_level = "Low"
        powdery_summary = (
            "Conditions unfavorable for Powdery Mildew (humidity or temperature outside 70-85% / 20°C-28°C range)."
        )

    # 3. Bacterial Spot Model
    driving_rain_warm = (cur_precip > 5.0) and (cur_temp > 24.0)
    moderate_rain_warm = (cur_precip > 2.0) and (cur_temp > 24.0)
    warm_heavy_rain_forecast = (cur_temp > 24.0) and any(p >= 70.0 for p in hourly_precip_prob[:12])

    if driving_rain_warm:
        bacterial_score = 0.95
        bacterial_level: Literal["Low", "Moderate", "High", "Critical"] = "Critical"
        bacterial_summary = (
            f"Critical Bacterial Spot risk: Driving rainfall ({cur_precip:.1f} mm > 5mm) combined with "
            f"warm temperatures ({cur_temp:.1f}°C > 24°C) enables rapid bacterial splash dispersal (Xanthomonas/Pseudomonas) and stomatal ingress."
        )
    elif moderate_rain_warm or (cur_precip > 5.0 and cur_temp > 20.0):
        bacterial_score = 0.75
        bacterial_level = "High"
        bacterial_summary = (
            f"High Bacterial Spot risk: Substantial rainfall ({cur_precip:.1f} mm) and warm temperatures ({cur_temp:.1f}°C) create splash-dispersal hazard."
        )
    elif warm_heavy_rain_forecast:
        bacterial_score = 0.55
        bacterial_level = "Moderate"
        bacterial_summary = (
            f"Moderate Bacterial Spot risk: Warm temperature ({cur_temp:.1f}°C) with imminent heavy rainfall forecast."
        )
    else:
        bacterial_score = 0.1
        bacterial_level = "Low"
        bacterial_summary = (
            "Low Bacterial Spot risk: Absence of driving rain (>5mm) and high temperatures (>24°C)."
        )

    # 4. Foliar Rust Model
    current_rust = (cur_rh > 80.0) and (16.0 <= cur_temp <= 24.0)
    rust_hours = sum(1 for t, rh in zip(hourly_temps, hourly_rh, strict=False) if rh > 80.0 and 16.0 <= t <= 24.0)

    if current_rust and rust_hours >= 6:
        rust_score = 0.85
        rust_level: Literal["Low", "Moderate", "High", "Critical"] = "High"
        rust_summary = (
            f"High Foliar Rust risk: High humidity (>80%, current: {cur_rh:.0f}%) and mild temperatures "
            f"(16°C-24°C, current: {cur_temp:.1f}°C) over {rust_hours} hours provide leaf wetness for urediniospore germination."
        )
    elif current_rust or rust_hours >= 4:
        rust_score = 0.65
        rust_level = "Moderate"
        rust_summary = (
            f"Moderate Foliar Rust risk: Sustained humidity >80% and temperatures between 16°C-24°C recorded for {rust_hours} hours."
        )
    else:
        rust_score = 0.1
        rust_level = "Low"
        rust_summary = "Low Foliar Rust risk: Humidity and temperature outside optimal rust infection thresholds."

    # Select target risk based on disease query or overall highest pathogen threat
    models = {
        "smith": (smith_score, smith_level, smith_summary),
        "powdery": (powdery_score, powdery_level, powdery_summary),
        "bacterial": (bacterial_score, bacterial_level, bacterial_summary),
        "rust": (rust_score, rust_level, rust_summary),
    }

    selected_score = 0.1
    selected_level: Literal["Low", "Moderate", "High", "Critical"] = "Low"
    selected_summary = "Overall low pathogen risk. Current microclimate conditions do not favor significant fungal or bacterial outbreak."

    disease_norm = (disease or "").lower().replace("_", " ")

    if any(k in disease_norm for k in ["late blight", "downy mildew", "phytophthora", "late_blight", "downy_mildew", "blight"]):
        selected_score, selected_level, selected_summary = models["smith"]
    elif any(k in disease_norm for k in ["powdery mildew", "powdery_mildew", "oidium", "powdery"]):
        selected_score, selected_level, selected_summary = models["powdery"]
    elif any(k in disease_norm for k in ["bacterial spot", "bacterial_spot", "bacterial", "xanthomonas", "pseudomonas", "spot"]):
        selected_score, selected_level, selected_summary = models["bacterial"]
    elif any(k in disease_norm for k in ["rust", "cedar apple rust", "common rust", "foliar rust", "puccinia"]):
        selected_score, selected_level, selected_summary = models["rust"]
    else:
        # Find maximum risk across all pathogen models
        best_key = max(models, key=lambda k: models[k][0])
        selected_score, selected_level, selected_summary = models[best_key]

    spray_advisory = generate_spray_advisory(
        current_temp=cur_temp,
        current_rh=cur_rh,
        current_precip=cur_precip,
        current_wind=cur_wind,
        hourly_precip_prob=hourly_precip_prob,
    )

    return WeatherRiskResponse(
        location=location_data,
        current=current_obj,
        risk_level=selected_level,
        risk_score=round(float(selected_score), 2),
        favorable_conditions_summary=selected_summary,
        spray_advisory=spray_advisory,
    )


class WeatherService:
    """Async client and intelligence service for Open-Meteo meteorological data."""

    def __init__(self, timeout: float = 10.0, base_url: str = OPEN_METEO_BASE_URL):
        self.timeout = timeout
        self.base_url = base_url

    async def fetch_weather_data(self, lat: float, lon: float) -> dict[str, Any]:
        """Fetch current and 3-day hourly forecast from Open-Meteo."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability",
            "forecast_days": 3,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()

    def calculate_disease_risk(
        self,
        weather_data: dict[str, Any],
        crop: str | None = None,
        disease: str | None = None,
    ) -> WeatherRiskResponse:
        """Instance method wrapper around calculate_disease_risk."""
        return calculate_disease_risk(weather_data, crop=crop, disease=disease)

    def _fallback_response(self, lat: float, lon: float, error_msg: str) -> WeatherRiskResponse:
        """Safe neutral response returned when meteorological query fails."""
        return WeatherRiskResponse(
            location={"latitude": lat, "longitude": lon, "timezone": "UTC"},
            current=CurrentWeather(
                temperature_c=20.0,
                humidity_pct=50.0,
                precipitation_mm=0.0,
                wind_kmh=5.0,
            ),
            risk_level="Low",
            risk_score=0.0,
            favorable_conditions_summary=(
                f"Microclimate data temporarily unavailable ({error_msg}). Defaulting to baseline neutral risk."
            ),
            spray_advisory="Follow standard label safety guidelines and local field observations.",
        )

    async def get_weather_risk(
        self,
        lat: float,
        lon: float,
        crop: str | None = None,
        disease: str | None = None,
    ) -> WeatherRiskResponse:
        """Fetch live meteorological forecast and evaluate pathogen risk with fallback."""
        try:
            weather_data = await self.fetch_weather_data(lat, lon)
            return self.calculate_disease_risk(weather_data, crop=crop, disease=disease)
        except Exception as exc:
            logger.warning("Failed to fetch or compute weather risk for (%s, %s): %s", lat, lon, exc)
            return self._fallback_response(lat, lon, str(exc))
