"""In-memory human approval queue (Feature #5).

Turns this app's two existing human-in-the-loop gates — web search
(web_search_requires_approval) and document deletion
(document_delete_requires_approval) — into a *trackable, resolvable queue*
rather than just an anonymous request-time skip/deny. When a gated action
is requested without the client's approval flag, the request is recorded
here as a pending approval; an operator lists pending approvals (GET
/api/v1/approvals) and resolves them approved/rejected
(POST /api/v1/approvals/{id}/resolve), which records the decision, who
made it, when, and why. Every mutation is also logged as an audit_event.

Scope: this store does NOT execute the gated action itself — callers
(routes/documents.py::delete_document, agent_graph/human_approval.py)
still perform their own tenant/owner/confirmation checks and the actual
delete/search. But as of the Phase 5 gap-closure pass, both guarded
actions now verify a request's approval against THIS store's resolved
state rather than trusting a client-supplied boolean alone:
document_delete requires a caller-supplied `approval_id` that resolves
here to `STATUS_APPROVED` for that exact document_id (see
delete_document's docstring for the full before/after); the chat graph's
web-search escalation routes through `human_approval_node`, which reads
this same store via `state.approval_payload_reference`. The historical
`confirm_web_search=true` client flag remains a separate, still-valid
fast path for callers who don't use the approval queue at all — it is
not a way to bypass a queued approval once one has been registered for
that specific request.

Storage follows the exact precedent of InMemorySessionStore (session_store.py):
in-memory + thread-safe + bounded with LRU eviction. No persistence —
consistent with the ephemeral-filesystem deployment constraint documented
in session_store.py; this is a queue of *recent* pending actions for a
human to review, not durable records.
"""

import functools
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

APPROVAL_ACTION_WEB_SEARCH = "web_search"
APPROVAL_ACTION_DOCUMENT_DELETE = "document_delete"

# Statuses an approval can be in. Pending = awaiting a human; the two
# terminal states record the operator's decision.
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
# Additive (Phase 1 explicit-workflow support): an action that never needed
# a human in the loop, and one whose pending request aged out unresolved.
# Existing callers that only ever construct/compare against PENDING/APPROVED/
# REJECTED are unaffected — these are new states, not renames.
STATUS_NOT_REQUIRED = "not_required"
STATUS_EXPIRED = "expired"


@dataclass
class Approval:
    approval_id: str
    action: str  # APPROVAL_ACTION_* constant
    requested_by: str | None
    payload: dict[str, Any]  # e.g. {"query": ...} for web_search, {"document_id": ...} for delete
    status: str = STATUS_PENDING
    note: str | None = None
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None
    resolved_by: str | None = None
    # Additive: optional expiry. None (the default, and what every existing
    # caller gets) means "never expires" — identical to current behavior.
    expires_at: float | None = None

    def is_expired(self, *, now: float | None = None) -> bool:
        """True if this approval is still PENDING but past its expires_at.

        Does not mutate status itself — callers (e.g. human_approval_node)
        decide when to observe/apply expiry so a read-only check here can't
        race a concurrent resolve().
        """
        if self.status != STATUS_PENDING or self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "action": self.action,
            "requested_by": self.requested_by,
            "payload": self.payload,
            "status": self.status,
            "note": self.note,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "expires_at": self.expires_at,
        }


class ApprovalStore:
    """Thread-safe, LRU-bounded registry of pending+resolved approvals."""

    def __init__(self, max_approvals: int = 500):
        if max_approvals <= 0:
            raise ValueError("max_approvals must be positive")
        self._max_approvals = max_approvals
        self._approvals: dict[str, Approval] = {}
        self._lock = threading.Lock()

    def _evict_if_needed(self) -> None:
        """Drop the oldest-recorded approval if at capacity. Called with
        lock held."""
        if len(self._approvals) >= self._max_approvals:
            oldest_id = min(self._approvals, key=lambda aid: self._approvals[aid].created_at)
            self._approvals.pop(oldest_id, None)

    def register(
        self,
        *,
        action: str,
        requested_by: str | None = None,
        payload: dict[str, Any] | None = None,
        note: str | None = None,
        ttl_seconds: float | None = None,
    ) -> Approval:
        """Record a new pending approval and return it.

        ttl_seconds is additive and optional: omitted (the default), an
        approval behaves exactly as before — pending until a human resolves
        it, never auto-expiring.
        """
        approval = Approval(
            approval_id=str(uuid.uuid4()),
            action=action,
            requested_by=requested_by,
            payload=payload or {},
            note=note,
            expires_at=(time.time() + ttl_seconds) if ttl_seconds is not None else None,
        )
        with self._lock:
            self._evict_if_needed()
            self._approvals[approval.approval_id] = approval
        logger.info(
            "audit_event",
            extra={
                "extra_fields": {
                    "event": "approval_requested",
                    "approval_id": approval.approval_id,
                    "action": action,
                    "requested_by": requested_by or "unknown",
                }
            },
        )
        return approval

    def get(self, approval_id: str) -> Approval | None:
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is not None and approval.is_expired():
                approval.status = STATUS_EXPIRED
                approval.resolved_at = time.time()
        return approval

    def list_approvals(
        self, action: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """All approvals matching optional action/status filters, newest
        first (most-recently-created first)."""
        with self._lock:
            results = [a.to_dict() for a in self._approvals.values()]
        if action is not None:
            results = [a for a in results if a["action"] == action]
        if status is not None:
            results = [a for a in results if a["status"] == status]
        return sorted(results, key=lambda a: a["created_at"], reverse=True)

    def resolve(
        self,
        approval_id: str,
        *,
        approved: bool,
        resolved_by: str,
        note: str | None = None,
    ) -> Approval | None:
        """Mark an approval approved/rejected. Returns None when the
        approval doesn't exist. Refuses to flip a non-pending approval
        (a decision is one-shot)."""
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval.status != STATUS_PENDING:
                return approval
            approval.status = STATUS_APPROVED if approved else STATUS_REJECTED
            approval.resolved_at = time.time()
            approval.resolved_by = resolved_by
            if note:
                approval.note = note
        logger.info(
            "audit_event",
            extra={
                "extra_fields": {
                    "event": "approval_resolved",
                    "approval_id": approval_id,
                    "action": approval.action,
                    "status": approval.status,
                    "resolved_by": resolved_by,
                    "note": note,
                }
            },
        )
        return approval


@functools.lru_cache(maxsize=1)
def get_approval_store() -> ApprovalStore:
    """One process-wide approval queue, shared across routes and the RAG
    service exactly like get_vector_store()/get_llm_client()."""
    return ApprovalStore()
