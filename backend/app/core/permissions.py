"""Central permission registry — replaces ad-hoc `if role != "admin"`
checks scattered across route handlers with one reusable, extensible
mechanism. Deliberately small: two permissions, two roles, proportionate
to what this app's action surface actually needs today (see
docs/DESIGN_REVIEW.md's RBAC section for why /upload, /chat,
/chat/feedback, etc. are NOT gated — gating them would break normal
member usage). Adding a new gated action is a 2-line change here plus
one check_permission() call at the route, not a new copy-pasted block.

Kept as a plain function (check_permission), not a FastAPI Depends() —
a dependency runs before the route body, which would reorder this ahead
of confirm/approval checks that intentionally run first (see
documents.py's delete_document). Call it inline at the exact point the
old inline block used to sit.
"""

import logging
from typing import Any

from fastapi import Request

from app.core.exceptions import ForbiddenError

logger = logging.getLogger(__name__)

DOCUMENT_DELETE = "document_delete"
DOCUMENT_LIST_ALL_TENANTS = "document_list_all_tenants"
ANALYTICS_VIEW = "analytics_view"
APPROVAL_VIEW = "approval_view"
APPROVAL_RESOLVE = "approval_resolve"
CACHE_INVALIDATE = "cache_invalidate"

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "admin": frozenset(
        {
            DOCUMENT_DELETE,
            DOCUMENT_LIST_ALL_TENANTS,
            ANALYTICS_VIEW,
            APPROVAL_VIEW,
            APPROVAL_RESOLVE,
            CACHE_INVALIDATE,
        }
    ),
    "member": frozenset(
        {
            DOCUMENT_DELETE,
        }
    ),
}

# Permissions that fall back to "allowed" when role is None (DB disabled
# — no role concept exists at all, matching that action's pre-RBAC
# behavior). Anything NOT listed here defaults to denied-when-unknown,
# the safer default for a gated action with no pre-RBAC history to fall
# back to.
_DEGRADE_ALLOW_WHEN_NO_ROLE = frozenset({DOCUMENT_DELETE, CACHE_INVALIDATE})


def has_permission(role: str | None, permission: str) -> bool:
    if role is None:
        return permission in _DEGRADE_ALLOW_WHEN_NO_ROLE
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


def check_permission(request: Request, permission: str, *, message: str, **log_fields: Any) -> None:
    """Raise ForbiddenError (and log an audit_event) if the caller's role
    lacks `permission`. log_fields are merged into the audit log's
    extra_fields (e.g. document_id) — callers pass whatever's in scope."""
    role = getattr(request.state, "role", None)
    if has_permission(role, permission):
        return
    logger.warning(
        "audit_event",
        extra={
            "extra_fields": {
                "event": f"{permission}_denied_role",
                "path": request.url.path,
                "client": getattr(request.state, "client_name", "unknown"),
                "role": role,
                **log_fields,
            }
        },
    )
    raise ForbiddenError(message)
