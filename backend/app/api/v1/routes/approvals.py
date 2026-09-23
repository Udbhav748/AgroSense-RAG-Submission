"""Admin-facing human approval queue (Feature #5).

GET /approvals — list pending (or all, filterable by action/status)
  approval requests recorded when a gated action (web search,
  document deletion) was requested without its approval flag.
POST /approvals/{approval_id}/resolve — grant (approved=true) or deny
  (approved=false) a pending approval, recording who decided and an
  optional note.

Both routes are admin-gated: resolving an approval is a privileged action
analogous to ANALYTICS_VIEW, and view is scoped to admins too (the queue
is an operational surface, not something every member should be able to
browse). Follows the approach of admin.py's usage-summary route exactly —
the router-level require_auth handles authentication, and
permissions.check_permission handles authorization inline.
"""

import logging

from fastapi import APIRouter, Depends, Request

from app.core import permissions
from app.core.auth import require_auth
from app.core.exceptions import ApprovalNotFoundError
from app.models.schemas import (
    ApprovalItem,
    ApprovalListResponse,
    ApprovalResolveRequest,
    ApprovalResolveResponse,
)
from app.services.approval_service import (
    APPROVAL_ACTION_DOCUMENT_DELETE,
    APPROVAL_ACTION_WEB_SEARCH,
    STATUS_PENDING,
    get_approval_store,
)

router = APIRouter(tags=["Approvals"], dependencies=[Depends(require_auth)])
logger = logging.getLogger(__name__)


@router.get("/approvals", response_model=ApprovalListResponse)
def list_approvals(
    request: Request,
    action: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> ApprovalListResponse:
    permissions.check_permission(
        request,
        permissions.APPROVAL_VIEW,
        message="Viewing the approval queue requires the admin role.",
    )
    if action is not None and action not in (
        APPROVAL_ACTION_WEB_SEARCH,
        APPROVAL_ACTION_DOCUMENT_DELETE,
    ):
        raise ApprovalNotFoundError(
            f"Unknown approval action '{action}'. Supported: {APPROVAL_ACTION_WEB_SEARCH}, "
            f"{APPROVAL_ACTION_DOCUMENT_DELETE}."
        )
    if status is not None and status not in (STATUS_PENDING, "approved", "rejected"):
        raise ApprovalNotFoundError(
            f"Unknown approval status '{status}'. Supported: pending, approved, rejected."
        )

    approvals = [
        ApprovalItem(**entry)
        for entry in get_approval_store().list_approvals(action=action, status=status)[
            : max(1, min(limit, 500))
        ]
    ]

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "approvals_listed",
                "path": request.url.path,
                "client": getattr(request.state, "client_name", "unknown"),
                "action": action,
                "status": status,
                "count": len(approvals),
            }
        },
    )
    return ApprovalListResponse(approvals=approvals, count=len(approvals))


@router.post("/approvals/{approval_id}/resolve", response_model=ApprovalResolveResponse)
def resolve_approval(
    approval_id: str, payload: ApprovalResolveRequest, request: Request
) -> ApprovalResolveResponse:
    permissions.check_permission(
        request,
        permissions.APPROVAL_RESOLVE,
        message="Resolving approvals requires the admin role.",
    )
    resolved_by = getattr(request.state, "client_name", "unknown")
    approval = get_approval_store().resolve(
        approval_id,
        approved=payload.approved,
        resolved_by=resolved_by,
        note=payload.note,
    )
    if approval is None:
        raise ApprovalNotFoundError(f"No approval found with id {approval_id}")

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "approval_resolved_via_api",
                "path": request.url.path,
                "approval_id": approval_id,
                "action": approval.action,
                "status": approval.status,
                "approved": payload.approved,
                "resolved_by": resolved_by,
            }
        },
    )
    return ApprovalResolveResponse(**approval.to_dict())
