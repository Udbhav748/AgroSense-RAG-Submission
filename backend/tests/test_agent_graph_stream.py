"""Tests for StateGraph agent execution and SSE event streaming."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.document import RetrievedChunk
from app.services.agent_graph import AgentState, END

VALID_HEADERS = {"X-API-Key": settings.api_key}


def test_extract_agent_graph_node_output_planner():
    from app.api.v1.routes.query import _extract_agent_graph_node_output

    state = AgentState(query="test query", plan={"action": "retrieve", "reason": "user asked for facts"})
    out = _extract_agent_graph_node_output("planner", state)
    assert out["action"] == "retrieve"
    assert out["reason"] == "user asked for facts"


def test_extract_agent_graph_node_output_fact_checker():
    from app.api.v1.routes.query import _extract_agent_graph_node_output

    state = AgentState(query="test query", fact_check_result={"passed": True, "score": 0.95})
    out = _extract_agent_graph_node_output("fact_checker", state)
    assert out["grounded"] is True
    assert out["score"] == 0.95


def test_extract_agent_graph_node_output_document_analyst():
    from app.api.v1.routes.query import _extract_agent_graph_node_output

    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="doc1",
        text="Sample text",
        score=0.9,
    )
    state = AgentState(query="test query", retrieved_chunks=[chunk])
    out = _extract_agent_graph_node_output("document_analyst", state)
    assert out["chunks_retrieved"] == 1


def test_agent_graph_stream_endpoint():
    client = TestClient(app)

    mock_chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Tomato late blight fungicide schedule.",
        score=0.95,
        metadata={"source": "tomato.pdf"},
    )

    with (
        patch("app.services.rag_service.retrieve", return_value=[mock_chunk]),
        patch("app.services.retrieval_service.retrieve", return_value=[mock_chunk]),
    ):
        response = client.post(
            "/chat/agent-graph/stream",
            headers=VALID_HEADERS,
            json={"query": "How to treat late blight in tomatoes?"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: "):
                data_str = line[len("data: "):]
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass

        assert len(events) > 0
        event_types = [e.get("type") for e in events]
        assert "node_start" in event_types
        assert "node_complete" in event_types
        assert "graph_done" in event_types
