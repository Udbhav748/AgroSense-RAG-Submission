"""Domain-specific exceptions, mapped to HTTP responses in app.core.error_handlers.

Each subclass also carries a taxonomy_category, drawn from a fixed
vocabulary (input, intent, planner, tool, retriever, memory, prompt,
reasoning, output, deployment) describing *where in the pipeline* the
failure originated — independent of its HTTP status code. Logged
alongside every error (see error_handlers.py) and used by
eval/metrics_report.py to break down error rates by category. Not every
category has a corresponding exception today (e.g. "intent"/"planner"/
"memory" have no failure mode yet, since routing and history handling
can't currently raise) — the vocabulary is fixed, its usage isn't.
"""


class AppError(Exception):
    status_code: int = 500
    taxonomy_category: str = "output"
    error_code: str = "APP_ERROR"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class UnauthorizedError(AppError):
    status_code = 401
    taxonomy_category = "input"
    error_code = "UNAUTHORIZED"


class RateLimitExceededError(AppError):
    """The caller exceeded the per-identity (API key / JWT user) or per-IP
    sliding-window request limit. Distinct from UnauthorizedError (401):
    a rate-limited request may be perfectly well authenticated — it's just
    too fast — so it must not look like an auth failure to the caller.
    The error_handlers taxonomy already reserved this category."""

    status_code = 429
    taxonomy_category = "rate_limit"
    error_code = "RATE_LIMIT_EXCEEDED"


class AccountLockedError(AppError):
    """Too many failed login attempts for this email within
    Settings.login_lockout_window_seconds (see user_service.py's
    _is_locked_out) — checked before the password is even compared.

    Same status code as RateLimitExceededError (a caller that's locked
    out is, like a rate-limited one, well-formed but temporarily
    rejected) but a distinct error_code: this is keyed by the *account*
    under attack, not the caller, so it survives an attacker rotating
    IPs the way an IP-keyed limiter alone would not."""

    status_code = 429
    taxonomy_category = "rate_limit"
    error_code = "ACCOUNT_LOCKED"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    taxonomy_category = "input"
    error_code = "UNSUPPORTED_FILE_TYPE"


class EmptyFileError(AppError):
    status_code = 400
    taxonomy_category = "input"
    error_code = "EMPTY_FILE"


class FileTooLargeError(AppError):
    status_code = 413
    taxonomy_category = "input"
    error_code = "FILE_TOO_LARGE"


class DocumentNotFoundError(AppError):
    status_code = 404
    taxonomy_category = "input"
    error_code = "DOCUMENT_NOT_FOUND"


class SessionNotFoundError(AppError):
    """Mirrors DocumentNotFoundError's shape: a non-owner requesting
    another tenant's session sees the same 404 a genuinely missing
    session_id would give, never a 403 that would confirm the session
    exists."""

    status_code = 404
    taxonomy_category = "input"
    error_code = "SESSION_NOT_FOUND"


class ConfirmationRequiredError(AppError):
    status_code = 400
    taxonomy_category = "input"
    error_code = "CONFIRMATION_REQUIRED"


class ApprovalRequiredError(AppError):
    """A deployment policy (Settings.document_delete_requires_approval,
    mirroring web_search_requires_approval) requires the caller to
    explicitly opt into a high-risk action via an extra request flag.
    Distinct from ConfirmationRequiredError (mistake-prevention: did you
    mean to do this at all?) and ForbiddenError (role/permission denial):
    this is a human-approval gate that can be toggled per-deployment."""

    status_code = 400
    taxonomy_category = "input"
    error_code = "APPROVAL_REQUIRED"


class ForbiddenError(AppError):
    """The caller is authenticated (UnauthorizedError doesn't apply) but
    lacks the role/permission the action requires — see auth.py's role
    check and documents.py's delete route."""

    status_code = 403
    taxonomy_category = "input"
    error_code = "FORBIDDEN"


class ApprovalNotFoundError(AppError):
    """The approval_id in a GET /approvals or POST /approvals/{id}/resolve
    request doesn't match any recorded approval (or a filter targeted an
    unknown action/status value)."""

    status_code = 404
    taxonomy_category = "input"
    error_code = "APPROVAL_NOT_FOUND"


class CorruptedPDFError(AppError):
    status_code = 422
    taxonomy_category = "input"
    error_code = "CORRUPTED_PDF"


class TextExtractionError(AppError):
    status_code = 500
    taxonomy_category = "tool"
    error_code = "TEXT_EXTRACTION_ERROR"


class ImageExtractionError(AppError):
    status_code = 500
    taxonomy_category = "tool"
    error_code = "IMAGE_EXTRACTION_ERROR"


class ImageCaptioningError(AppError):
    """A caption generation call failed at the LLM/provider layer. Raised
    per-image by image_captioning_service's client call; the service
    catches it and degrades to "no caption chunk" for that image rather
    than failing the whole upload (see its module docstring)."""

    status_code = 502
    taxonomy_category = "tool"
    error_code = "IMAGE_CAPTIONING_ERROR"


class TableExtractionError(AppError):
    status_code = 500
    taxonomy_category = "tool"
    error_code = "TABLE_EXTRACTION_ERROR"


class EmbeddingModelLoadError(AppError):
    status_code = 500
    taxonomy_category = "deployment"
    error_code = "EMBEDDING_MODEL_LOAD_ERROR"


class EmbeddingGenerationError(AppError):
    status_code = 500
    taxonomy_category = "tool"
    error_code = "EMBEDDING_GENERATION_ERROR"


class VectorStoreNotFoundError(AppError):
    status_code = 404
    taxonomy_category = "retriever"
    error_code = "VECTOR_STORE_NOT_FOUND"


class CorruptedVectorStoreError(AppError):
    status_code = 422
    taxonomy_category = "retriever"
    error_code = "CORRUPTED_VECTOR_STORE"


class EmbeddingDimensionMismatchError(AppError):
    status_code = 400
    taxonomy_category = "retriever"
    error_code = "EMBEDDING_DIMENSION_MISMATCH"


class MetadataSyncError(AppError):
    status_code = 500
    taxonomy_category = "retriever"
    error_code = "METADATA_SYNC_ERROR"


class LLMConfigurationError(AppError):
    status_code = 500
    taxonomy_category = "deployment"
    error_code = "LLM_CONFIGURATION_ERROR"


class LLMAPIError(AppError):
    status_code = 502
    taxonomy_category = "tool"
    error_code = "LLM_API_ERROR"


class LLMTimeoutError(AppError):
    status_code = 504
    taxonomy_category = "tool"
    error_code = "LLM_TIMEOUT"


class LLMEmptyResponseError(AppError):
    status_code = 502
    taxonomy_category = "output"
    error_code = "LLM_EMPTY_RESPONSE"


class WebSearchError(AppError):
    status_code = 502
    taxonomy_category = "tool"
    error_code = "WEB_SEARCH_ERROR"


class RerankingError(AppError):
    status_code = 500
    taxonomy_category = "tool"
    error_code = "RERANKING_ERROR"


class VisionServiceError(AppError):
    status_code = 502
    taxonomy_category = "tool"
    error_code = "VISION_SERVICE_ERROR"


class ClipServiceError(AppError):
    """The CLIP embedding microservice (clip_service/, reached via
    clip_client.py) is unreachable, timed out, returned a non-2xx, or
    returned a payload outside its documented {embedding, dimension}
    contract. Callers degrade cross-modal retrieval on this (see
    hybrid_search.py) rather than failing the request — CLIP is an
    enhancement, the same posturing reranking already uses."""

    status_code = 502
    taxonomy_category = "tool"
    error_code = "CLIP_SERVICE_ERROR"


class PromptGenerationError(AppError):
    status_code = 400
    taxonomy_category = "prompt"
    error_code = "PROMPT_GENERATION_ERROR"


class AuthConfigurationError(AppError):
    """JWT_SECRET_KEY isn't set — same "fail loud" shape as
    LLMConfigurationError's missing-API-key case. Raised by the /auth
    routes, never by require_auth (an unconfigured deployment simply
    never receives a JWT to begin with, since signup/login can't have
    issued one)."""

    status_code = 500
    taxonomy_category = "deployment"
    error_code = "AUTH_CONFIGURATION_ERROR"


class DatabaseNotConfiguredError(AppError):
    """Raised by user_service.py's _session() when an /auth route is hit
    but DATABASE_URL isn't set — individual user login has nowhere to
    persist a User/Tenant. 503, not 500: the server is fine, this one
    action just isn't available in the current deployment (the same
    "not a bug, a config gap" signal AuthConfigurationError gives for a
    missing JWT_SECRET_KEY). Distinct from that error because the fix is
    different: JWT_SECRET_KEY is a value to set, DATABASE_URL is a whole
    dependency that has to exist first."""

    status_code = 503
    taxonomy_category = "deployment"
    error_code = "DATABASE_NOT_CONFIGURED"


class EmailAlreadyRegisteredError(AppError):
    status_code = 409
    taxonomy_category = "input"
    error_code = "EMAIL_ALREADY_REGISTERED"


class ChatServiceError(AppError):
    status_code = 500
    taxonomy_category = "reasoning"
    error_code = "CHAT_SERVICE_ERROR"


class TaskNotFoundError(AppError):
    """The task_id in a GET /documents/tasks/{task_id} request doesn't match
    any recorded task (or belongs to a different tenant)."""

    status_code = 404
    taxonomy_category = "input"
    error_code = "TASK_NOT_FOUND"
