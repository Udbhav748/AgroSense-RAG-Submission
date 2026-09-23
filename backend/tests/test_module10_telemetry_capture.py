"""Tests for the Module 10 evaluation-only telemetry capture utility
(eval/module10/metrics/telemetry_capture.py). Exercises the capture
mechanism in isolation with synthetic log records — no live LLM/graph
run required — so the instrumentation itself is reproducibly tested.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.module10.metrics.telemetry_capture import (  # noqa: E402
    capture_telemetry,
    extract,
)


def _log(logger_name: str, message: str, **fields):
    logger = logging.getLogger(logger_name)
    logger.info(message, extra={"extra_fields": fields})


def test_capture_extracts_node_sequence():
    with capture_telemetry() as handler:
        _log("app.services.agent_graph.events", "agent_node_trace", node="planner", status="ok")
        _log("app.services.agent_graph.events", "agent_node_trace", node="retrieval", status="ok")
        _log("app.services.agent_graph.events", "agent_node_trace", node="generator", status="ok")

    telemetry = extract(handler)
    assert telemetry.node_sequence == ["planner", "retrieval", "generator"]
    assert telemetry.node_statuses == [
        {"node": "planner", "status": "ok"},
        {"node": "retrieval", "status": "ok"},
        {"node": "generator", "status": "ok"},
    ]


def test_capture_extracts_chat_query_handled_cost_fields():
    with capture_telemetry() as handler:
        _log(
            "app.services.rag_service",
            "chat_query_handled",
            steps_taken=3,
            estimated_cost_usd=0.0021,
            total_tokens=512,
            llm_calls=2,
        )

    telemetry = extract(handler)
    assert telemetry.steps_taken == 3
    assert telemetry.estimated_cost_usd == 0.0021
    assert telemetry.total_tokens == 512
    assert telemetry.llm_calls == 2


def test_capture_ignores_unrelated_log_lines():
    with capture_telemetry() as handler:
        _log("app.services.agent_graph.events", "some_other_event", node="planner")
        logging.getLogger("app.services.rag_service").info("unrelated line with no extra_fields")

    telemetry = extract(handler)
    assert telemetry.node_sequence == []
    assert telemetry.chat_query_handled is None


def test_handler_detached_after_context_exits():
    node_logger = logging.getLogger("app.services.agent_graph.events")
    handlers_before = list(node_logger.handlers)

    with capture_telemetry() as handler:
        assert handler in node_logger.handlers

    assert handler not in node_logger.handlers
    assert node_logger.handlers == handlers_before


def test_handler_detached_even_on_exception():
    node_logger = logging.getLogger("app.services.agent_graph.events")
    handlers_before = list(node_logger.handlers)

    captured_handler = None
    try:
        with capture_telemetry() as handler:
            captured_handler = handler
            raise ValueError("boom")
    except ValueError:
        pass

    assert captured_handler not in node_logger.handlers
    assert node_logger.handlers == handlers_before


def test_no_sensitive_content_leaked_when_query_text_logged_elsewhere():
    """The capture only reads `extra_fields` dicts from the two named
    log lines it targets — even if some unrelated log record on the same
    logger carries free-text (e.g. an exception message with a fragment
    of user input), it must not surface in the extracted telemetry."""
    with capture_telemetry() as handler:
        logging.getLogger("app.services.rag_service").warning(
            "some_debug_line containing raw user query text: what is my SSN 123-45-6789"
        )
        _log("app.services.rag_service", "chat_query_handled", steps_taken=1, estimated_cost_usd=0.001)

    telemetry = extract(handler)
    assert telemetry.chat_query_handled == {"steps_taken": 1, "estimated_cost_usd": 0.001}
    assert "123-45-6789" not in str(telemetry.chat_query_handled)
