"""Phase 1: AgentState field coverage and no-secret-leak checks."""

from __future__ import annotations

from app.services.agent_graph.state import AgentState


def test_state_initializes_with_defaults():
    state = AgentState(query="hello")
    assert state.query == "hello"
    assert state.workflow_status == "pending"
    assert state.approval_status == "not_required"
    assert state.retrieval_grade is None
    assert state.tool_calls == []
    assert state.node_timings == {}
    assert state.termination_reason is None


def test_required_phase1_fields_exist():
    state = AgentState(query="q")
    required = [
        "request_id",
        "trace_id",
        "session_id",
        "tenant_id",
        "query",
        "intent",
        "planned_action",
        "planned_steps",
        "current_node",
        "workflow_status",
        "retrieved_chunks",
        "reranked_chunks",
        "retrieval_grade",
        "web_results",
        "tool_calls",
        "tool_results",
        "history",
        "memory_context",
        "draft_answer",
        "final_answer",
        "structured_output",
        "validation_errors",
        "approval_required",
        "approval_type",
        "approval_status",
        "retry_count",
        "reflection_count_v2",
        "loop_count",
        "steps_taken",
        "node_timings",
        "workflow_start_time",
        "workflow_end_time",
        "token_usage",
        "estimated_cost_usd",
        "error_type",
        "error_message",
        "root_cause",
        "source_type",
        "final_sources",
    ]
    for field_name in required:
        assert hasattr(state, field_name), f"AgentState missing field {field_name}"


def test_state_does_not_define_secret_fields():
    field_names = set(AgentState.model_fields.keys())
    forbidden_substrings = ["api_key", "password", "secret", "authorization", "token_value"]
    for name in field_names:
        lowered = name.lower()
        for bad in forbidden_substrings:
            assert bad not in lowered, f"AgentState field '{name}' looks like it stores a secret"


def test_state_survives_copy_with_across_transitions():
    state = AgentState(query="q", request_id="r1", trace_id="t1")
    s2 = state.copy_with(planned_action="retrieve", steps_taken=1)
    s3 = s2.copy_with(retrieval_grade="good", steps_taken=2)
    s4 = s3.copy_with(draft_answer="answer", steps_taken=3)
    assert s4.request_id == "r1"
    assert s4.trace_id == "t1"
    assert s4.planned_action == "retrieve"
    assert s4.retrieval_grade == "good"
    assert s4.draft_answer == "answer"
    assert s4.steps_taken == 3
    # Original state is untouched (immutability)
    assert state.steps_taken == 0
    assert state.draft_answer == ""
