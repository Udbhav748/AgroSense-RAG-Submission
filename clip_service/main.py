"""CLIP embedding microservice for AgroSense-RAG.

A separate FastAPI process that owns a CLIP model and serves L2-normalized
text/image embeddings over HTTP — the LeafSense-shaped pattern from
docs/MULTIUSER_MULTIMODAL_PLAN.md Phase 4: the main backend never imports
torch/transformers, it just makes an HTTP call to this service, and this
service runs its own model in its own process.

Endpoints (the contract backend/app/services/clip_client.py is written
against):

- GET  /health           -> {"status": "ok", "model": ..., "model_loaded": bool}
- POST /embed/text       - body {"text": "..."} -> {"embedding": [...], "dimension": N}
- POST /embed/image      - multipart field "file" -> {"embedding": [...], "dimension": N}

Embeddings are L2-normalized before being returned, so inner product across
text <-> image vectors == cosine similarity — that's what makes a CLIP text
embedding of a query directly comparable to CLIP image embeddings in the
image FAISS index.

Optional auth: set CLIP_API_KEY to require an X-API-Key header (mirroring
AgroSense-RAG's own inbound auth convention). Model is loaded lazily on first
request so importing this app never triggers the torch/transformers import
cost on a process that won't use it.
"""

import io
import logging
import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="AgroSense-RAG CLIP Embedding Service")

_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
_API_KEY = os.environ.get("CLIP_API_KEY", "")

_client = None  # CLIPProcessor + CLIPModel pair, loaded lazily


def _load_model():
    """Load CLIPProcessor + CLIPModel once (thread-safe enough for this
    project's single-model, single-process usage — the same lazy-load +
    cache-once lifecycle embedding_service.py uses for SentenceTransformer)."""
    global _client
    if _client is not None:
        return _client
    from transformers import CLIPModel, CLIPProcessor

    logger.info("loading CLIP model %s", _MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
    model = CLIPModel.from_pretrained(_MODEL_NAME)
    model.eval()
    _client = (processor, model)
    logger.info("CLIP model loaded (dimension=%d)", model.config.projection_dim)
    return _client


def _check_auth(x_api_key: str | None) -> None:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _embedding_or_400(exc) -> None:
    raise HTTPException(status_code=400, detail=f"Failed to build embedding: {exc}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": _MODEL_NAME,
        "model_loaded": _client is not None,
        "dimension": _client[1].config.projection_dim if _client is not None else None,
    }


class EmbedTextRequest(BaseModel):
    text: str


@app.post("/embed/text")
def embed_text(req: EmbedTextRequest, x_api_key: str | None = Header(default=None)) -> dict:
    _check_auth(x_api_key)
    try:
        processor, model = _load_model()
        import torch

        with torch.no_grad():
            inputs = processor(text=[req.text], padding=True, return_tensors="pt")
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        embedding = features.squeeze(0).tolist()
    except Exception as exc:  # noqa: BLE001 — fail loud with a 400, client treats non-2xx as ClipServiceError
        _embedding_or_400(exc)
    return {"embedding": embedding, "dimension": len(embedding)}


@app.post("/embed/image")
def embed_image(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict:
    _check_auth(x_api_key)
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        processor, model = _load_model()
        import torch

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        embedding = features.squeeze(0).tolist()
    except Exception as exc:  # noqa: BLE001
        _embedding_or_400(exc)
    return {"embedding": embedding, "dimension": len(embedding)}