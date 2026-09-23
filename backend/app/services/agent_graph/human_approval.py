"""Generic human-approval node standardizing the two existing approval
gates (web search, document deletion) behind one reusable abstraction.

This module never executes the guarded action itself, and never
auto-approves anything — approval is enforced in code (route_after_approval
only returns "resume" when approval_status == "approved"), not by asking an
LLM to decide. It wraps the existing `approval_service.ApprovalStore`
(already the queue backing GET/POST /api/v1/approvals) rather than
introducing a second approval system.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from app.core.metrics import get_metrics
from app.services.approval_service import (
    STATUS_APPROVED,
    STATUS_EXPIRED,
    STATUS_NOT_REQUIRED,
    STATUS_PENDING,
    STATUS_REJECTED,
    get_approval_store,
)

if TYPE_CHECKING:
    from app.services.agent_graph.nodes import GraphContext
    from app.services.agent_graph.state import AgentState

logger = logging.getLogger(__name__)


class ApprovalType(str, Enum):
    """The only two approval surfaces this phase implements. Extensible
    later (external_write, email_send, bulk_delete, sensitive_action) by
    adding members here — no change needed to the node/routing logic."""

    WEB_SEARCH = "web_search"
    DOCUMENT_DELETE = "document_delete"


@dataclass(frozen=True)
class ApprovalRequest:
    """Mirrors PDF section 21's ApprovalRequest shape. Thin view over the
    underlying `approval_service.Approval` record — not a second store."""

    id: str
    request_id: str | None
    trace_id: str | None
    type: str
    reason: str | None
    created_at: float
    expires_at: float | None
    status: str
    action_reference: str | None


def human_approval_node(state: AgentState, context: GraphContext | None = None) -> AgentState:
    """Resolve state.approval_status from the ApprovalStore.

    Behavior:
    - approval_required is False -> not_required, node is a no-op.
    - A prior approval_payload_reference (approval_id) exists -> look it up
      and mirror its current status (pending/approved/rejected/expired).
    - No prior reference and approval is required -> register a new pending
      approval and stop the workflow there (status stays "pending"); the
      caller (route_after_approval) sends it to safe_finalizer, exactly like
      today's inline "skip and record a pending approval" behavior in
      ChatService._search_web / routes/documents.py. The workflow does not
      block synchronously waiting for a human — a re-submitted request with
      the resolved approval_id (or a client-provided confirm_web_search=true/
      approved=true, which continue to work exactly as before) is what
      resumes it, matching the existing request-response approval model.
    """
    if not state.approval_required:
        return state.copy_with(approval_status=STATUS_NOT_REQUIRED)

    get_metrics().record_agent_approval(event="required", action=state.approval_type or "unknown")
    store = get_approval_store()

    if state.approval_payload_reference:
        approval = store.get(state.approval_payload_reference)
        if approval is None:
            # Reference is stale/unknown — treat as if approval must be
            # re-requested rather than silently proceeding.
            approval = store.register(
                action=state.approval_type or "unknown",
                requested_by=None,
                payload={"query": state.query} if state.approval_type == "web_search" else {},
                note=state.approval_reason,
            )
        logger.info(
            "human_approval_checked",
            extra={
                "extra_fields": {
                    "approval_id": approval.approval_id,
                    "action": approval.action,
                    "status": approval.status,
                }
            },
        )
        if approval.status in (STATUS_APPROVED, STATUS_REJECTED):
            get_metrics().record_agent_approval(
                event="approved" if approval.status == STATUS_APPROVED else "rejected",
                action=approval.action,
            )
        return state.copy_with(
            approval_status=approval.status,
            approval_payload_reference=approval.approval_id,
        )

    # No reference yet: register a new pending approval. Never auto-approve.
    approval = store.register(
        action=state.approval_type or "unknown",
        requested_by=None,
        payload={"query": state.query} if state.approval_type == "web_search" else {},
        note=state.approval_reason,
    )
    logger.info(
        "human_approval_registered",
        extra={
            "extra_fields": {
                "approval_id": approval.approval_id,
                "action": approval.action,
                "requested_at": time.time(),
            }
        },
    )
    return state.copy_with(
        approval_status=STATUS_PENDING,
        approval_payload_reference=approval.approval_id,
    )


__all__ = [
    "STATUS_APPROVED",
    "STATUS_EXPIRED",
    "STATUS_NOT_REQUIRED",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "ApprovalRequest",
    "ApprovalType",
    "human_approval_node",
]
