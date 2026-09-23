"""Routes for individual user signup/login (JWT), alongside the
existing API-key auth used by documents.py/query.py.

Unauthenticated by design: POST /signup and POST /login are how a
caller *becomes* authenticated, so neither declares require_auth. Only
GET /me does.
"""

import logging

from fastapi import APIRouter, Depends, Request

from app.core.auth import require_auth
from app.core.security import rate_limit_dependency
from app.models.schemas import (
    AuthTokenResponse,
    CurrentUserResponse,
    LoginRequest,
    SignupRequest,
)
from app.services.user_service import (
    authenticate_user,
    create_access_token,
    create_user,
    revoke_all_tokens,
)

router = APIRouter(prefix="/auth", tags=["Auth"])
logger = logging.getLogger(__name__)


@router.post(
    "/signup",
    response_model=AuthTokenResponse,
    status_code=201,
    dependencies=[Depends(rate_limit_dependency)],
)
def signup(payload: SignupRequest) -> AuthTokenResponse:
    """Create a personal account (and its own private tenant — see
    user_service.create_user) and return a ready-to-use JWT.

    payload.consent is Literal[True] on the schema itself (see
    schemas.py) — a request without real consent 422s before this
    function body ever runs, so there's nothing to check here.
    """
    user_id, tenant_id, _role = create_user(payload.email, payload.password)
    logger.info(
        "audit_event",
        extra={
            "extra_fields": {"event": "user_signed_up", "user_id": user_id, "tenant_id": tenant_id}
        },
    )
    token = create_access_token(user_id, payload.email)
    return AuthTokenResponse(access_token=token)


@router.post(
    "/login", response_model=AuthTokenResponse, dependencies=[Depends(rate_limit_dependency)]
)
def login(payload: LoginRequest) -> AuthTokenResponse:
    user_id, tenant_id, _role = authenticate_user(payload.email, payload.password)
    logger.info(
        "audit_event",
        extra={
            "extra_fields": {"event": "user_logged_in", "user_id": user_id, "tenant_id": tenant_id}
        },
    )
    token = create_access_token(user_id, payload.email)
    return AuthTokenResponse(access_token=token)


@router.post("/logout", status_code=204, dependencies=[Depends(require_auth)])
def logout(request: Request) -> None:
    """Invalidate every JWT previously issued to the caller (see
    user_service.revoke_all_tokens) — real server-side logout, not just
    the client discarding its token. request.state.user_id is only set
    on the JWT auth path (see require_auth); an API-key caller has no
    token to revoke, so this is a no-op for them rather than an error —
    hitting this endpoint with the wrong credential type isn't a
    mistake worth surfacing, just nothing to do.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        revoke_all_tokens(user_id)
        logger.info(
            "audit_event",
            extra={"extra_fields": {"event": "user_logged_out", "user_id": user_id}},
        )


@router.get("/me", response_model=CurrentUserResponse, dependencies=[Depends(require_auth)])
def me(request: Request) -> CurrentUserResponse:
    """Current caller's identity, how the frontend hydrates on load.

    Works for either auth path (JWT or X-API-Key) since both populate
    the same request.state fields — for an API-key caller, "email" is
    actually that key's client_name, not a real email address (API keys
    have no email at all); this endpoint is meant for the web frontend's
    JWT session, not for service clients.
    """
    return CurrentUserResponse(
        email=request.state.client_name,
        tenant_id=request.state.tenant_id,
        role=request.state.role,
    )
