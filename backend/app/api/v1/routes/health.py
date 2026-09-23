"""Health/readiness endpoint.

Kept deliberately open (no auth dependency) and side-effect-free so load
balancers and uptime monitors can hit it. Reports more than liveness: the
LLM provider's readiness (whether its API key is present — a boolean,
never the key itself) and which multi-modal RAG capabilities this
deployment has enabled, so an operator can tell at a glance why uploads
and captioning behave the way they do.
"""

from typing import Any

from fastapi import APIRouter

from app.core.config import settings
from app.core.database import db_enabled
from app.services.cache_service import cache_service
from app.services.document_service import is_tesseract_available
from app.services.vision_client import is_leafsense_online

router = APIRouter()


def _provider_configured(provider: str | None) -> bool:
    """Whether the given LLM provider has an API key configured.

    /health is unauthenticated, so this returns a boolean only — never a
    key or a partial key — to avoid leaking credentials through a public
    endpoint.
    """
    if provider is None:
        return False
    if provider == "gemini":
        return bool(settings.gemini_api_key)
    if provider == "groq":
        return bool(settings.groq_api_key)
    return False


@router.get("/health", tags=["Health"])
def health_check() -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "ok",
        "llm": {
            "provider": settings.llm_provider,
            "provider_configured": _provider_configured(settings.llm_provider),
            "fallback_provider": settings.fallback_llm_provider,
            "fallback_configured": _provider_configured(settings.fallback_llm_provider),
            "model_routing_enabled": settings.model_routing_enabled,
        },
        "multimodal": {
            "image_extraction_enabled": settings.image_extraction_enabled,
            "image_captioning_enabled": settings.image_captioning_enabled,
            "table_extraction_enabled": settings.table_extraction_enabled,
            "vision_qa_enabled": settings.vision_qa_enabled,
            "ocr_available": is_tesseract_available(),
        },
        "vision_service": {
            "url": settings.vision_service_url,
            "online": is_leafsense_online(),
            "has_gemini_fallback": bool(settings.gemini_api_key),
            "disease_classes_count": 38,
        },
        "vector_store": {
            "backend": "pgvector" if settings.pgvector_enabled else "faiss",
            "model": settings.embedding_model_name,
            "status": "ready",
        },
        "cache": cache_service.get_stats(),
    }
    if db_enabled():
        body["database"] = "connected"
    return body


@router.get("/health/vision", tags=["Health"])
def vision_health_check() -> dict[str, Any]:
    """Dedicated health check for the LeafSense deep vision network."""
    online = is_leafsense_online()
    return {
        "online": online,
        "url": settings.vision_service_url,
        "has_gemini_fallback": bool(settings.gemini_api_key),
        "status": "online" if online else "offline",
    }
