"""Unauthorized Access Rate check for DELETE /documents/{id}'s tenant-
ownership gate (app/api/v1/routes/documents.py), plus a companion
authorized-action check for the member/admin role gate
(app/core/permissions.py).

Rate = Successful Unauthorized Actions / Unauthorized Attempts, desired
value zero — same framing as the checklist's other zero-desired metrics
(false_refusal_rate, data_leak_rate in eval/run_eval.py). Two genuinely
unauthorized attempt shapes, each of which the route is supposed to deny:

1. Cross-tenant delete: requester's tenant_id != the document's owner
   tenant_id (a genuinely different pair of ids from #2, so a
   hardcoded-id bug in the check wouldn't slip through both).
2. Same shape as #1 with a second, unrelated pair of tenant ids.

Deliberately does NOT test the "no tenant info at all" path (DB
disabled, or a legacy document with no owner row) — that's a documented,
*accepted* gap (see documents.py's own comment: the ownership check is
"only enforced when we can actually verify it"), not something this
metric should flag as a failure.

HISTORY / CORRECTION (Module 10 gap-closure pass): this script used to
treat a *same-tenant* delete by a "member"-role client as a third
"unauthorized attempt" that should be denied — folding it into the
Unauthorized Access Rate and reporting a measured rate of 0.3333 (1/3)
when it succeeded. Investigation traced this to a real discrepancy
between this script's own assumption and `app/core/permissions.py`'s
actual, deliberately-documented design: `ROLE_PERMISSIONS["member"]`
explicitly includes `DOCUMENT_DELETE` (see that module's docstring —
"gating [normal actions] would break normal member usage"). A member
deleting a document *owned by their own tenant* is authorized by design;
only the cross-tenant boundary (#1/#2 above) is the actual security
gate this metric is meant to police. The scenario is kept below as
`check_member_can_delete_own_tenant_document` — a separate, correctly-
labeled confirmation that this authorized path still works — rather
than removed, so a future regression in either direction (a member
wrongly denied, or a genuine cross-tenant bypass) is still caught.

Uses FastAPI's TestClient against the real app — real routes, real
auth/role-gate code — with only the vector store, LLM, embedding, and
tenant/ownership resolution mocked, mirroring tests/test_security.py's
`client` fixture exactly (same boundary-mocking style, just two
registered clients instead of one, which no existing fixture in this
repo does — see this script's `_client_context()`). No live LLM, no
network, no DATABASE_URL needed — resolve_tenant/get_document_owner are
mocked directly rather than requiring a real Postgres instance, the same
simplification test_main.py's own cross-tenant delete tests already make.

Usage (from backend/):
    python eval/unauthorized_access_check.py
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.models.document import EmbeddedChunk  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402

FAKE_EMBEDDING_DIM = 8
FAKE_EMBEDDING = [1.0] + [0.0] * (FAKE_EMBEDDING_DIM - 1)
DOCUMENT_ID = "22222222-2222-2222-2222-222222222222"

# Client B's key never needs to map to a real seeded document — only
# client A's key does — but both must be registered so require_api_key
# resolves a real client_name for each, exactly like a live deployment
# with two API_KEYS entries would.
CLIENT_A_KEY = "unauth-check-client-a-key"
CLIENT_B_KEY = "unauth-check-client-b-key"


@contextmanager
def _client_context(tmp_path: Path):
    """Yields a TestClient with one seeded document, two registered API
    keys (client-a/client-b), and vector-store/LLM/embedding mocked —
    the same boundary-mocking style as tests/test_security.py's `client`
    fixture, extended to two clients since resolving *different*
    tenant_ids for different callers is the whole point of this check."""
    store = FAISSVectorStore(
        index_path=tmp_path / "index.faiss",
        metadata_path=tmp_path / "metadata.json",
    )
    store.create_index(dimension=FAKE_EMBEDDING_DIM)
    store.add_embeddings(
        [
            EmbeddedChunk(
                chunk_id="chunk-1",
                document_id=DOCUMENT_ID,
                embedding=FAKE_EMBEDDING,
                metadata={"chunk_index": 0, "total_chunks": 1, "source": "pdf", "text": "irrelevant"},
            )
        ]
    )

    original_api_keys = settings.api_keys
    settings.api_keys = json.dumps({"client-a": CLIENT_A_KEY, "client-b": CLIENT_B_KEY})
    try:
        with (
            patch("app.api.v1.routes.query.get_vector_store", lambda: store),
            patch("app.api.v1.routes.documents.get_vector_store", lambda: store),
            TestClient(app) as test_client,
        ):
            yield test_client
    finally:
        settings.api_keys = original_api_keys


def _attempt_delete(
    tmp_path: Path,
    requester_tenant: tuple[int, str],
    owner_tenant_id: int,
    label: str,
) -> tuple[bool, int]:
    """One unauthorized-attempt trial. Returns (denied, status_code).
    denied=True is the desired outcome (the checklist's "zero" case)."""
    with _client_context(tmp_path) as client:
        with (
            patch("app.core.auth.resolve_tenant", lambda client_name: requester_tenant),
            patch("app.api.v1.routes.documents.get_document_owner", lambda document_id: owner_tenant_id),
        ):
            response = client.delete(
                f"/documents/{DOCUMENT_ID}",
                params={"confirm": "true"},
                headers={"X-API-Key": CLIENT_B_KEY},
            )
    denied = response.status_code in (403, 404)
    print(f"  {label:<45s} status={response.status_code} {'DENIED (ok)' if denied else 'SUCCEEDED (BAD)'}")
    return denied, response.status_code


def check_member_can_delete_own_tenant_document(tmp_path: Path) -> tuple[bool, int]:
    """A same-tenant delete by a 'member'-role client SHOULD succeed —
    this is the authorized path, not an attack (see the module docstring's
    HISTORY note). Returns (authorized_as_expected, status_code); pass
    means status_code == 200, the OPPOSITE convention from
    _attempt_delete's `denied` — never merge this result into the
    Unauthorized Access Rate denominator.
    """
    with _client_context(tmp_path) as client:
        with (
            patch("app.core.auth.resolve_tenant", lambda client_name: (1, "member")),
            patch("app.api.v1.routes.documents.get_document_owner", lambda document_id: 1),
        ):
            response = client.delete(
                f"/documents/{DOCUMENT_ID}",
                params={"confirm": "true"},
                headers={"X-API-Key": CLIENT_B_KEY},
            )
    authorized_as_expected = response.status_code == 200
    label = "same-tenant delete, member role (should SUCCEED — authorized by design)"
    print(f"  {label:<58s} status={response.status_code} {'OK (authorized)' if authorized_as_expected else 'UNEXPECTEDLY DENIED (regression)'}")
    return authorized_as_expected, response.status_code


def main() -> None:
    import tempfile

    print("=== Unauthorized Access Rate (DELETE /documents/{id}) ===\n")
    print("Cross-tenant attempts (should all be DENIED):")

    unauthorized_results: list[bool] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        denied, _ = _attempt_delete(
            tmp_path,
            requester_tenant=(2, "admin"),
            owner_tenant_id=1,
            label="cross-tenant delete (tenant 2 -> tenant 1's doc)",
        )
        unauthorized_results.append(denied)

        denied, _ = _attempt_delete(
            tmp_path,
            requester_tenant=(99, "admin"),
            owner_tenant_id=5,
            label="cross-tenant delete (tenant 99 -> tenant 5's doc)",
        )
        unauthorized_results.append(denied)

        print("\nAuthorized-action confirmation (should SUCCEED — not counted as an unauthorized attempt):")
        authorized_ok, _ = check_member_can_delete_own_tenant_document(tmp_path)

    total_attempts = len(unauthorized_results)
    successful_unauthorized = sum(1 for denied in unauthorized_results if not denied)
    rate = successful_unauthorized / total_attempts if total_attempts else None

    print(f"\nUnauthorized attempts: {total_attempts}")
    print(f"Successful unauthorized actions: {successful_unauthorized}  (desired: 0)")
    print(f"Unauthorized Access Rate: {rate:.4f}" if rate is not None else "Unauthorized Access Rate: n/a")
    print(f"Member-can-delete-own-tenant-document check: {'PASS' if authorized_ok else 'FAIL (regression — members should be able to delete their own documents)'}")

    if successful_unauthorized or not authorized_ok:
        print("\nFAIL.")
        sys.exit(1)
    print("\nPASS -- no unauthorized action succeeded, and the authorized member path still works.")


if __name__ == "__main__":
    main()
