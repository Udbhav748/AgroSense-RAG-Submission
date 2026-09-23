"""Unit tests for app/core/permissions.py — the central permission
registry that replaced two duplicated inline `if role != "admin"` blocks
in documents.py. Most important thing to protect: the role=None
asymmetry (DOCUMENT_DELETE degrades to allowed, DOCUMENT_LIST_ALL_TENANTS
does not) — a future "simplification" that unifies the two would be a
real regression, not a cleanup.
"""

import pytest

from app.core.exceptions import ForbiddenError
from app.core.permissions import (
    DOCUMENT_DELETE,
    DOCUMENT_LIST_ALL_TENANTS,
    check_permission,
    has_permission,
)


class _FakeURL:
    def __init__(self, path: str):
        self.path = path


class _FakeState:
    def __init__(self, role, client_name="test-client"):
        self.role = role
        self.client_name = client_name


class _FakeRequest:
    """Minimal stand-in for fastapi.Request — check_permission only
    reads request.state and request.url.path."""

    def __init__(self, role, client_name="test-client", path="/documents/x"):
        self.state = _FakeState(role, client_name)
        self.url = _FakeURL(path)


class TestHasPermission:
    def test_admin_has_both_permissions(self):
        assert has_permission("admin", DOCUMENT_DELETE) is True
        assert has_permission("admin", DOCUMENT_LIST_ALL_TENANTS) is True

    def test_member_has_document_delete_but_not_list_all_tenants(self):
        assert has_permission("member", DOCUMENT_DELETE) is True
        assert has_permission("member", DOCUMENT_LIST_ALL_TENANTS) is False

    def test_unknown_role_has_no_permissions(self):
        assert has_permission("some-future-role", DOCUMENT_DELETE) is False
        assert has_permission("some-future-role", DOCUMENT_LIST_ALL_TENANTS) is False

    def test_none_role_asymmetry(self):
        # DOCUMENT_DELETE degrades to allowed (pre-RBAC fallback, DB
        # disabled); DOCUMENT_LIST_ALL_TENANTS has no pre-RBAC behavior
        # to fall back to, so it's denied instead. These must NOT be
        # unified into one rule.
        assert has_permission(None, DOCUMENT_DELETE) is True
        assert has_permission(None, DOCUMENT_LIST_ALL_TENANTS) is False


class TestCheckPermission:
    def test_allowed_returns_none(self):
        request = _FakeRequest(role="admin")
        assert check_permission(request, DOCUMENT_DELETE, message="denied") is None

    def test_member_allowed_for_document_delete(self):
        request = _FakeRequest(role="member")
        assert check_permission(request, DOCUMENT_DELETE, message="denied") is None

    def test_member_denied_raises_forbidden_error_for_list_all_tenants(self):
        request = _FakeRequest(role="member")
        with pytest.raises(ForbiddenError) as exc_info:
            check_permission(request, DOCUMENT_LIST_ALL_TENANTS, message="Listing all tenants requires the admin role.")
        assert exc_info.value.detail == "Listing all tenants requires the admin role."
        assert exc_info.value.status_code == 403

    def test_none_role_allowed_for_document_delete(self):
        request = _FakeRequest(role=None)
        assert check_permission(request, DOCUMENT_DELETE, message="denied") is None

    def test_none_role_denied_for_list_all_tenants(self):
        request = _FakeRequest(role=None)
        with pytest.raises(ForbiddenError):
            check_permission(request, DOCUMENT_LIST_ALL_TENANTS, message="denied")

    def test_extra_log_fields_do_not_affect_outcome(self):
        # log_fields (e.g. document_id) are merged into the audit log
        # only — they must not change the allow/deny decision.
        request = _FakeRequest(role="admin")
        assert check_permission(request, DOCUMENT_DELETE, message="denied", document_id="doc-1") is None
