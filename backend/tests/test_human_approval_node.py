"""Phase 1: human_approval_node — not_required / pending / approved / rejected."""

from __future__ import annotations

from app.services.agent_graph.human_approval import human_approval_node
from app.services.agent_graph.routing import route_after_approval
from app.services.agent_graph.state import AgentState
from app.services.approval_service import get_approval_store


def test_not_required_when_approval_not_required():
    state = AgentState(query="q", approval_required=False)
    result = human_approval_node(state)
    assert result.approval_status == "not_required"


def test_registers_pending_approval_when_required_and_no_reference():
    state = AgentState(query="search this", approval_required=True, approval_type="web_search")
    result = human_approval_node(state)
    assert result.approval_status == "pending"
    assert result.approval_payload_reference is not None

    stored = get_approval_store().get(result.approval_payload_reference)
    assert stored is not None
    assert stored.status == "pending"
    assert stored.action == "web_search"


def test_reflects_approved_status_from_store():
    store = get_approval_store()
    approval = store.register(action="web_search", payload={"query": "x"})
    store.resolve(approval.approval_id, approved=True, resolved_by="operator")

    state = AgentState(
        query="x",
        approval_required=True,
        approval_type="web_search",
        approval_payload_reference=approval.approval_id,
    )
    result = human_approval_node(state)
    assert result.approval_status == "approved"
    assert route_after_approval(result) == "resume"


def test_reflects_rejected_status_from_store():
    store = get_approval_store()
    approval = store.register(action="document_delete", payload={"document_id": "d1"})
    store.resolve(approval.approval_id, approved=False, resolved_by="operator")

    state = AgentState(
        query="delete it",
        approval_required=True,
        approval_type="document_delete",
        approval_payload_reference=approval.approval_id,
    )
    result = human_approval_node(state)
    assert result.approval_status == "rejected"
    assert route_after_approval(result) == "safe_finalizer"


def test_pending_approval_routes_to_safe_finalizer_not_resume():
    state = AgentState(query="q", approval_required=True, approval_type="web_search")
    result = human_approval_node(state)
    assert result.approval_status == "pending"
    # A pending (unresolved) approval must never resolve to "resume".
    assert route_after_approval(result) == "safe_finalizer"


def test_expired_pending_approval_is_observed_as_expired():
    store = get_approval_store()
    approval = store.register(action="web_search", payload={"query": "x"}, ttl_seconds=-1)

    state = AgentState(
        query="x",
        approval_required=True,
        approval_type="web_search",
        approval_payload_reference=approval.approval_id,
    )
    result = human_approval_node(state)
    assert result.approval_status == "expired"
    assert route_after_approval(result) == "safe_finalizer"
