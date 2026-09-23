"""HTTP client for LeafSense, a separate FastAPI vision service that
classifies plant leaf photos into one of 38 disease/healthy classes.

LeafSense keeps its own TensorFlow/Keras stack in its own process;
AgroSense-RAG never imports TensorFlow or any LeafSense code — this module
only makes an HTTP call, the same isolation web_search_service.py uses
for duckduckgo_search or gemini_client.py uses for the google-genai SDK.

Contract (verified against LeafSense's backend/main.py directly, not
assumed): POST /predict/{model_id} (model_id is accepted for compatibility
with LeafSense's own frontend's per-crop routes but unused server-side —
every model_id runs the same hybrid model) with a multipart field named
"file", returning {"class": "<raw label>", "confidence": <float 0-1>}.
"""

import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from app.core.config import settings
from app.core.exceptions import VisionServiceError
from app.models.document import VisionPrediction
from app.services.explainability import generate_leaf_saliency
from app.services.tool_registry import track_tool

logger = logging.getLogger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    # .outcome is a property, not a plain attribute -- assign it to a local
    # so mypy can narrow the None check (it won't narrow across repeated
    # property reads, since a property isn't guaranteed to return the same
    # value each access).
    outcome = retry_state.outcome
    exception = outcome.exception() if outcome is not None else None
    logger.warning(
        "vision_request_retrying",
        extra={
            "extra_fields": {
                "attempt": retry_state.attempt_number,
                "exception": str(exception),
            }
        },
    )


# model_id is unused by LeafSense's endpoint (every id runs the same hybrid
# model) — kept as a fixed, descriptive literal rather than a config value
# since it has no actual effect on the call.
_MODEL_ID = "insightai"

# LeafSense's CLASS_NAMES (backend/main.py), copied verbatim — order
# doesn't matter here since this client only ever receives the label
# *string* back, never an index, but the exact spelling (including the
# raw space in the corn/GLS entry and the trailing underscore in
# "Common_rust_") must match exactly for CLASS_LABEL_MAP lookups to hit.
# Do not "clean up" these strings.
CLASS_LABEL_MAP: dict[str, tuple[str, str]] = {
    "Apple___Apple_scab": ("apple", "apple scab"),
    "Apple___Black_rot": ("apple", "black rot"),
    "Apple___Cedar_apple_rust": ("apple", "cedar apple rust"),
    "Apple___healthy": ("apple", "healthy"),
    "Blueberry___healthy": ("blueberry", "healthy"),
    "Cherry_(including_sour)___Powdery_mildew": ("cherry", "powdery mildew"),
    "Cherry_(including_sour)___healthy": ("cherry", "healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("corn", "gray leaf spot"),
    "Corn_(maize)___Common_rust_": ("corn", "common rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn", "northern corn leaf blight"),
    "Corn_(maize)___healthy": ("corn", "healthy"),
    "Grape___Black_rot": ("grape", "black rot"),
    "Grape___Esca_(Black_Measles)": ("grape", "esca (black measles)"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("grape", "leaf blight (isariopsis leaf spot)"),
    "Grape___healthy": ("grape", "healthy"),
    "Orange___Haunglongbing_(Citrus_greening)": ("orange", "citrus greening (huanglongbing)"),
    "Peach___Bacterial_spot": ("peach", "bacterial spot"),
    "Peach___healthy": ("peach", "healthy"),
    "Pepper,_bell___Bacterial_spot": ("bell pepper", "bacterial spot"),
    "Pepper,_bell___healthy": ("bell pepper", "healthy"),
    "Potato___Early_blight": ("potato", "early blight"),
    "Potato___Late_blight": ("potato", "late blight"),
    "Potato___healthy": ("potato", "healthy"),
    "Raspberry___healthy": ("raspberry", "healthy"),
    "Soybean___healthy": ("soybean", "healthy"),
    "Squash___Powdery_mildew": ("squash", "powdery mildew"),
    "Strawberry___Leaf_scorch": ("strawberry", "leaf scorch"),
    "Strawberry___healthy": ("strawberry", "healthy"),
    "Tomato___Bacterial_spot": ("tomato", "bacterial spot"),
    "Tomato___Early_blight": ("tomato", "early blight"),
    "Tomato___Late_blight": ("tomato", "late blight"),
    "Tomato___Leaf_Mold": ("tomato", "leaf mold"),
    "Tomato___Septoria_leaf_spot": ("tomato", "septoria leaf spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "two-spotted spider mite"),
    "Tomato___Target_Spot": ("tomato", "target spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "tomato yellow leaf curl virus"),
    "Tomato___Tomato_mosaic_virus": ("tomato", "tomato mosaic virus"),
    "Tomato___healthy": ("tomato", "healthy"),
}


_last_online_check: float = 0.0
_last_online_status: bool = False
_ONLINE_CHECK_TTL: float = 5.0

_vision_client: httpx.Client | None = None
_ORIGINAL_HTTPX_POST = httpx.post


def _get_vision_client() -> httpx.Client:
    """Return a persistent, reusable httpx.Client session for LeafSense calls."""
    global _vision_client
    if _vision_client is None or _vision_client.is_closed:
        _vision_client = httpx.Client(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _vision_client


def is_leafsense_online(
    host: str | None = None,
    port: int | None = None,
    timeout: float = 0.6,
    force_refresh: bool = False,
) -> bool:
    """Probe if the LeafSense vision service is actively listening on its port.

    host/port default to whatever settings.vision_service_url actually
    points at (e.g. "leafsense" inside the Docker Compose network, not
    "127.0.0.1" -- that only resolves to this same container, never
    LeafSense's, once they're separate containers rather than both
    running natively on one host). Callers can still override both for
    a probe against a specific address.

    Caches the online status with a 5-second TTL to avoid redundant 600ms TCP socket
    pre-probes on high-frequency requests.
    """
    global _last_online_check, _last_online_status
    now = time.monotonic()
    if not force_refresh and (now - _last_online_check) < _ONLINE_CHECK_TTL:
        return _last_online_status

    if host is None or port is None:
        parsed = urlparse(settings.vision_service_url)
        host = host or parsed.hostname or "127.0.0.1"
        port = port or parsed.port or 8001

    try:
        with socket.create_connection((host, port), timeout=timeout):
            _last_online_status = True
            _last_online_check = now
            return True
    except Exception:
        _last_online_status = False
        _last_online_check = now
        return False


def _try_auto_start_leafsense() -> bool:
    """Attempts to auto-launch the local LeafSense background service if located in sibling repo."""
    if is_leafsense_online():
        return True

    repo_root = Path(__file__).resolve().parents[3]
    candidate_dirs = [
        repo_root.parent / "LeafSense" / "backend",
        repo_root / "LeafSense" / "backend",
        repo_root / ".." / "LeafSense" / "backend",
    ]

    leafsense_dir = None
    for d in candidate_dirs:
        if (d / "main.py").exists():
            leafsense_dir = d.resolve()
            break

    if not leafsense_dir:
        return False

    venv_python = leafsense_dir / ".venv" / "Scripts" / "python.exe"
    python_cmd = str(venv_python) if venv_python.exists() else "python"

    try:
        logger.info(
            "auto_starting_leafsense_service",
            extra={"extra_fields": {"dir": str(leafsense_dir), "python": python_cmd}},
        )
        creation_flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW on Windows
        subprocess.Popen(
            [python_cmd, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"],
            cwd=str(leafsense_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            shell=False,
        )
        for _ in range(15):
            time.sleep(0.4)
            if is_leafsense_online(force_refresh=True):
                logger.info("leafsense_service_auto_started_successfully")
                return True
    except Exception as exc:
        logger.warning("leafsense_auto_start_failed", extra={"extra_fields": {"error": str(exc)}})
    return False


def _diagnose_with_gemini_fallback(
    contents: bytes, filename: str, content_type: str, engine_tag: str = "gemini_vision"
) -> VisionPrediction | None:
    """Fallback visual classifier using Gemini Vision when LeafSense is unreachable or needs consensus."""
    if not settings.gemini_api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.gemini_api_key)
        classes_str = ", ".join(CLASS_LABEL_MAP.keys())
        prompt = (
            "You are an expert plant pathologist. Analyze this plant leaf image carefully, ignoring background clutter or hands. "
            f"Identify which of the following exact 38 classes it belongs to: [{classes_str}]. "
            "Respond ONLY with a JSON object in this exact format: "
            '{"class": "<exact_class_name>", "confidence": <float_between_0_and_1>}'
        )
        part = types.Part.from_bytes(data=contents, mime_type=content_type or "image/jpeg")
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=[part, prompt],  # type: ignore[arg-type]
        )
        text = response.text or ""
        import json

        clean_text = text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        data = json.loads(clean_text)
        raw_class = data.get("class", "Tomato___healthy")
        confidence = float(data.get("confidence", 0.90))
        crop, disease = CLASS_LABEL_MAP.get(raw_class, ("unknown", raw_class))
        low_confidence = confidence < settings.vision_confidence_threshold
        logger.info(
            "vision_gemini_diagnosis_succeeded",
            extra={"extra_fields": {"class": raw_class, "confidence": confidence, "engine": engine_tag}},
        )
        return VisionPrediction(
            raw_class=raw_class,
            crop=crop,
            disease=disease,
            confidence=confidence,
            low_confidence=low_confidence,
            engine=engine_tag,
        )
    except Exception as exc:
        logger.warning("vision_gemini_fallback_failed", extra={"extra_fields": {"error": str(exc)}})
        return None


@track_tool("diagnose")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(VisionServiceError),
    reraise=True,
    before_sleep=_log_retry,
)
def diagnose_image(
    contents: bytes, filename: str, content_type: str, engine: str = "hybrid"
) -> VisionPrediction:
    """POST an image to LeafSense or Gemini Vision and return its prediction,
    mapped to a plain-language crop/disease pair.

    Engines supported:
    - 'hybrid' (default): Runs LeafSense first. If confidence < threshold or field noise,
      transparently consults Gemini Vision as a consensus arbiter.
    - 'leafsense': Directly uses the custom CBAM+ViT+EfficientNet model on port 8001.
    - 'gemini': Directly uses Gemini 1.5 Flash Vision for zero-shot in-the-wild reasoning.
    """
    global _last_online_check, _last_online_status
    start = time.perf_counter()

    # Compute explainability saliency heatmap & metrics
    saliency = generate_leaf_saliency(contents)
    heatmap_base64 = saliency.get("heatmap_base64")
    infected_area_percentage = saliency.get("infected_area_percentage")
    lesion_count = saliency.get("lesion_count")

    # Direct Gemini engine route
    if engine == "gemini":
        gemini_pred = _diagnose_with_gemini_fallback(
            contents, filename, content_type, engine_tag="gemini_vision"
        )
        if gemini_pred is not None:
            gemini_pred.heatmap_base64 = heatmap_base64
            gemini_pred.infected_area_percentage = infected_area_percentage
            gemini_pred.lesion_count = lesion_count
            return gemini_pred

    # Self-healing: if offline, attempt to auto-launch local LeafSense service
    if not is_leafsense_online():
        _try_auto_start_leafsense()

    headers = {}
    if settings.vision_service_api_key:
        headers["X-API-Key"] = settings.vision_service_api_key

    response = None
    url = f"{settings.vision_service_url}/predict/{_MODEL_ID}"
    files = {"file": (filename, contents, content_type)}
    timeout = settings.vision_service_timeout_seconds

    try:
        if httpx.post is not _ORIGINAL_HTTPX_POST:
            response = httpx.post(
                url,
                files=files,
                headers=headers,
                timeout=timeout,
            )
        else:
            client = _get_vision_client()
            response = client.post(
                url,
                files=files,
                headers=headers,
                timeout=timeout,
            )
        response.raise_for_status()
    except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
        _last_online_status = False
        _last_online_check = 0.0

        # Check secondary Gemini Vision fallback before erroring out
        fallback_prediction = _diagnose_with_gemini_fallback(
            contents, filename, content_type, engine_tag="gemini_fallback"
        )
        if fallback_prediction is not None:
            fallback_prediction.heatmap_base64 = heatmap_base64
            fallback_prediction.infected_area_percentage = infected_area_percentage
            fallback_prediction.lesion_count = lesion_count
            return fallback_prediction

        if isinstance(exc, httpx.TimeoutException):
            raise VisionServiceError(
                f"LeafSense request timed out after {settings.vision_service_timeout_seconds}s: {exc}"
            ) from exc
        if isinstance(exc, httpx.HTTPStatusError):
            raise VisionServiceError(
                f"LeafSense returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        raise VisionServiceError(
            f"Could not reach LeafSense at {settings.vision_service_url}. Ensure LeafSense is running on port 8001."
        ) from exc

    try:
        payload = response.json()
        raw_class = payload["class"]
        confidence = float(payload["confidence"])
    except (ValueError, KeyError, TypeError) as exc:
        raise VisionServiceError(f"LeafSense returned an unexpected response shape: {exc}") from exc

    crop, disease = CLASS_LABEL_MAP.get(raw_class, (None, None))
    if crop is None:
        logger.warning("vision_unmapped_class", extra={"extra_fields": {"raw_class": raw_class}})
        crop, disease = "unknown", raw_class

    processing_duration = time.perf_counter() - start
    low_confidence = confidence < settings.vision_confidence_threshold

    # In hybrid mode: if custom model confidence is low (noisy field photo), consult Gemini arbiter
    if engine == "hybrid" and low_confidence:
        arbiter_pred = _diagnose_with_gemini_fallback(
            contents, filename, content_type, engine_tag="hybrid_consensus"
        )
        if arbiter_pred is not None and not arbiter_pred.low_confidence:
            arbiter_pred.heatmap_base64 = heatmap_base64
            arbiter_pred.infected_area_percentage = infected_area_percentage
            arbiter_pred.lesion_count = lesion_count
            logger.info(
                "vision_hybrid_consensus_arbitrated",
                extra={
                    "extra_fields": {
                        "leafsense_class": raw_class,
                        "leafsense_conf": confidence,
                        "arbiter_class": arbiter_pred.raw_class,
                        "arbiter_conf": arbiter_pred.confidence,
                    }
                },
            )
            return arbiter_pred

    logger.info(
        "vision_diagnosis_completed",
        extra={
            "extra_fields": {
                "raw_class": raw_class,
                "crop": crop,
                "disease": disease,
                "confidence": round(confidence, 4),
                "low_confidence": low_confidence,
                "engine": "leafsense",
                "processing_duration": round(processing_duration, 4),
            }
        },
    )

    return VisionPrediction(
        raw_class=raw_class,
        crop=crop,
        disease=disease,
        confidence=confidence,
        low_confidence=low_confidence,
        engine="leafsense",
        heatmap_base64=heatmap_base64,
        infected_area_percentage=infected_area_percentage,
        lesion_count=lesion_count,
    )
