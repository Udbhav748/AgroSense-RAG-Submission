"""Phase 1: vision_node — the explicit graph node handle_diagnose now
delegates to instead of calling diagnose_image/_build_diagnosis_query
inline."""

from __future__ import annotations

import pytest

import app.services.rag_service as rag_service_module
from app.models.document import VisionPrediction
from app.services.agent_graph.nodes import GraphContext, vision_node
from app.services.agent_graph.state import AgentState


def make_prediction(crop="tomato", disease="early blight", confidence=0.92):
    return VisionPrediction(
        raw_class=f"{crop}___{disease}".replace(" ", "_"),
        crop=crop,
        disease=disease,
        confidence=confidence,
        low_confidence=False,
    )


def test_vision_node_populates_diagnosis_and_query(monkeypatch):
    prediction = make_prediction()
    monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: prediction)
    context = GraphContext(
        metadata={
            "image_bytes": b"fake-jpeg-bytes",
            "filename": "leaf.jpg",
            "content_type": "image/jpeg",
            "engine": "hybrid",
        }
    )
    state = AgentState(query="what's wrong with my plant?")
    result = vision_node(state, context)

    assert result.diagnosis is not None
    assert result.diagnosis.crop == "tomato"
    assert result.diagnosis.disease == "early blight"
    assert "early blight" in result.retrieval_query
    assert "tomato" in result.retrieval_query
    assert result.metadata["crop_context"] == "tomato"
    assert result.metadata["disease_context"] == "early blight"
    assert "vision" in result.node_timings


def test_vision_node_healthy_prediction_builds_healthy_query(monkeypatch):
    prediction = make_prediction(disease="healthy")
    monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: prediction)
    context = GraphContext(metadata={"image_bytes": b"x", "filename": "f.jpg", "content_type": "image/jpeg"})
    state = AgentState(query="")
    result = vision_node(state, context)
    assert "healthy" in result.retrieval_query


def test_vision_node_requires_image_bytes():
    with pytest.raises(RuntimeError):
        vision_node(AgentState(query="q"), GraphContext(metadata={}))
    with pytest.raises(RuntimeError):
        vision_node(AgentState(query="q"), None)


def test_vision_node_propagates_vision_service_errors(monkeypatch):
    from app.core.exceptions import VisionServiceError

    def _raise(*a, **k):
        raise VisionServiceError("vision backend unreachable")

    monkeypatch.setattr(rag_service_module, "diagnose_image", _raise)
    context = GraphContext(metadata={"image_bytes": b"x", "filename": "f.jpg", "content_type": "image/jpeg"})
    with pytest.raises(VisionServiceError):
        vision_node(AgentState(query="q"), context)


def test_vision_node_unknown_crop_is_not_stamped_as_context(monkeypatch):
    prediction = make_prediction(crop="unknown", disease="unknown")
    monkeypatch.setattr(rag_service_module, "diagnose_image", lambda *a, **k: prediction)
    context = GraphContext(metadata={"image_bytes": b"x", "filename": "f.jpg", "content_type": "image/jpeg"})
    result = vision_node(AgentState(query=""), context)
    assert result.metadata["crop_context"] is None
    assert result.metadata["disease_context"] is None
