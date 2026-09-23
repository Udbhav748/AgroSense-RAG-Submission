"""Pydantic request/response schemas for API validation and serialization."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.document import RetrievedChunk


class DocumentUploadResponse(BaseModel):
    document_id: str
    original_filename: str
    stored_filename: str
    file_size: int
    upload_timestamp: datetime
    status: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's natural-language question.")
    top_k: int | None = Field(
        None,
        gt=0,
        description="Number of chunks to retrieve. Defaults to Settings.retrieval_top_k.",
    )
    min_score: float | None = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Minimum similarity score to keep a chunk. Defaults to Settings.retrieval_min_score.",
    )
    history: list[dict[str, Any]] | None = Field(
        None,
        description="Prior conversation turns, oldest first, each shaped like "
        "{'role': 'user'|'assistant', 'content': str}. Only the most recent "
        "6 are used. Ignored if session_id is provided (server-side history takes precedence).",
    )
    session_id: str | None = Field(
        None,
        description="Session identifier for server-side chat history. If omitted on first request, "
        "a new session is created and its ID returned in the response. On subsequent requests, "
        "include this ID to continue the same conversation.",
    )
    confirm_web_search: bool = Field(
        False,
        description="Human approval for the web-search tool. When Settings.web_search_requires_approval "
        "is enabled, web search is skipped unless this is true — an explicit human-in-the-loop "
        "gate on the agent's only outbound side-effect.",
    )
    approval_id: str | None = Field(
        None,
        description="Resume a previously-registered web-search approval (see GET/POST /api/v1/approvals). "
        "When Settings.web_search_requires_approval is enabled and confirm_web_search was not set, the "
        "first request registers a pending Approval and its id is returned in the response metadata; "
        "resubmit the same query with that id here once an operator has resolved it via "
        "POST /api/v1/approvals/{id}/resolve. Ignored if the referenced approval is not approved.",
    )
    structured_response: bool = Field(
        False,
        description="Request a JSON-mode structured answer. Only takes effect when "
        "Settings.structured_output_enabled is true; otherwise ignored.",
    )
    persona: str | None = Field(
        None,
        description="Optional tone/style preset for the answer (see prompt_builder.PERSONAS). "
        "Only affects HOW the answer is phrased — never whether it's grounded or cited.",
    )
    document_ids: list[str] | None = Field(
        None,
        description="When set, retrieval is scoped to exactly these document_ids — chat "
        "against a named collection instead of the whole library.",
    )
    language: str = Field(
        "en",
        description="Language code for multilingual localization (e.g. 'en', 'es', 'hi', 'pt', 'fr', 'sw').",
    )


class SourceReference(BaseModel):
    number: int = Field(
        ...,
        description="The [N] bracket marker this source corresponds to in `answer`'s inline "
        "citations — 1-indexed, documents first then web results, matching the order the "
        "model was shown them in (see prompt_builder.build_prompt).",
    )
    document_id: str
    chunk_id: str
    excerpt: str = Field(..., description="First ~200 characters of the cited chunk's text.")
    url: str | None = Field(
        None,
        description="Source URL, present only for web-sourced citations (document_id='web').",
    )
    content_type: str = Field(
        "text",
        description="What kind of content this citation was derived from: 'text' (default, "
        "ordinary chunked document text), 'image_caption' (multi-modal RAG — a Gemini-generated "
        "caption of an extracted figure), or 'table' (a table reduced to markdown text). "
        "Read straight off the underlying chunk's metadata.",
    )
    page_number: int | None = Field(
        None,
        description="1-based source page, when known — always present for image_caption/table "
        "citations, generally absent for ordinary text chunks (which aren't tracked per-page).",
    )


class DiagnosisInfo(BaseModel):
    raw_class: str = Field(..., description="Raw class label as returned by LeafSense.")
    crop: str = Field(..., description="Plain-language crop name, e.g. 'peach'.")
    disease: str = Field(..., description="Plain-language disease name, e.g. 'bacterial spot'.")
    confidence: float = Field(
        ..., description="Model confidence for the predicted class, in [0, 1]."
    )
    low_confidence: bool = Field(
        ..., description="True if confidence is below Settings.vision_confidence_threshold."
    )
    heatmap_base64: str | None = Field(
        None, description="Base64 encoded PNG heatmap lesion overlay."
    )
    infected_area_percentage: float | None = Field(
        None, description="Estimated percentage of leaf surface infected (0.0% to 100.0%)."
    )
    lesion_count: int | None = Field(
        None, description="Estimated count of distinct lesion spots."
    )


class CurrentWeather(BaseModel):
    temperature_c: float = Field(..., description="Current temperature in degrees Celsius.")
    humidity_pct: float = Field(..., description="Current relative humidity percentage (0-100).")
    precipitation_mm: float = Field(..., description="Current precipitation in mm.")
    wind_kmh: float = Field(..., description="Current wind speed in km/h at 10m.")


class WeatherRiskResponse(BaseModel):
    location: dict[str, Any] = Field(..., description="Geographic coordinates and timezone metadata.")
    current: CurrentWeather = Field(..., description="Current atmospheric weather readings.")
    risk_level: Literal["Low", "Moderate", "High", "Critical"] = Field(
        ..., description="Categorical microclimate disease infection risk level."
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized pathogen infection risk score (0.0 to 1.0)."
    )
    favorable_conditions_summary: str = Field(
        ..., description="Detailed explanation of environmental conditions driving pathogen pressure."
    )
    spray_advisory: str = Field(
        ..., description="Actionable agronomical spray window and wind drift guidance."
    )


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    sources: list[SourceReference]
    processing_time: float
    tool_used: str
    steps_taken: int
    answer_source: Literal["documents", "web", "mixed"] = Field(
        "documents",
        description="Whether the answer drew on retrieved document chunks, web search "
        "results, or both. Always 'documents' for non-retrieval actions.",
    )
    diagnosis: DiagnosisInfo | None = Field(
        None,
        description="Present only for image-diagnosis requests (POST /chat/diagnose) — "
        "the LeafSense prediction that produced this response's retrieval query.",
    )
    session_id: str = Field(
        ...,
        description="Session identifier for this conversation. If this is a new session, "
        "the server generates and returns it; otherwise it echoes the incoming session_id. "
        "Client should store this (e.g. in localStorage) and send it on subsequent requests "
        "to continue the same conversation history.",
    )
    retrieval_confidence: Literal["good", "weak", "insufficient"] = Field(
        "good",
        description="How confident retrieval was that it found on-topic context: 'good' "
        "(top score cleared the grade threshold), 'weak' (chunks survived the score floor "
        "but weren't confidently on-topic), or 'insufficient' (nothing survived). Surfaced "
        "so the user can double-check a weak/insufficient-grounded answer against the source.",
    )
    is_clarifying_question: bool = Field(
        False,
        description="True when, instead of the canned 'couldn't find that' fallback, the "
        "answer is one short clarifying question the app asked because retrieval graded "
        "'insufficient' and no grounded answer was possible.",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Up to 3 short, suggested follow-up questions the user might ask next "
        "(the 'related questions' pattern). Empty when the feature is off or the model "
        "produced nothing parseable.",
    )
    hallucination_detected: bool = Field(
        False,
        description="True when the lexical groundedness check (Feature #4) found that the "
        "answer shares almost no vocabulary with the retrieved context it was supposedly "
        "grounded on — a likely hallucination signal. A recommendation to double-check "
        "against the sources, never a blocker: the answer is delivered unchanged.",
    )
    grounding_score: float | None = Field(
        None,
        description="Fraction of the answer's content tokens that also appeared in the "
        "retrieved context (0..1), when the grounding check ran. Higher is better-grounded; "
        "below Settings.hallucination_grounding_threshold is what sets "
        "hallucination_detected.",
    )
    weather_risk: WeatherRiskResponse | None = Field(
        None,
        description="Present when field coordinates (latitude/longitude) are supplied — "
        "the microclimate pathogen infection risk assessment and spray advisory.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (e.g. cached flag, latency breakdown).",
    )


class RubricScores(BaseModel):
    """A reviewer's 1-5 rating on each of the answer-quality checklist's
    seven criteria. All seven are required together — a partial rubric
    (e.g. rating correctness but not safety) isn't a meaningful data
    point for the per-criterion averages or Inter-Annotator Agreement
    computed from it (see eval/metrics_report.py)."""

    correctness: int = Field(
        ..., ge=1, le=5, description="Is the answer factually and logically right?"
    )
    helpfulness: int = Field(
        ..., ge=1, le=5, description="Does it help the user complete their task?"
    )
    completeness: int = Field(..., ge=1, le=5, description="Does it cover every required part?")
    safety: int = Field(
        ..., ge=1, le=5, description="Does it avoid harmful, unauthorized, or risky behavior?"
    )
    tone: int = Field(
        ..., ge=1, le=5, description="Is the communication appropriate for the context?"
    )
    groundedness: int = Field(
        ..., ge=1, le=5, description="Are factual claims supported by evidence?"
    )
    citation_quality: int = Field(
        ..., ge=1, le=5, description="Do citations point to the correct sources?"
    )


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    rating: Literal["up", "down"]
    comment: str | None = None
    rubric: RubricScores | None = Field(
        None,
        description="Optional detailed per-criterion review (see RubricScores). The quick "
        "thumbs-up/down above (`rating`) remains valid on its own — this is an additive, "
        "opt-in deeper review, not a replacement.",
    )


class FeedbackResponse(BaseModel):
    status: str


class StructuredAnswer(BaseModel):
    """Schema the LLM's JSON-mode output is validated against.

    The prompt (see prompt_builder.build_structured_prompt) asks the model
    to emit exactly this shape; anything else is a parse failure and the
    caller falls back to the free-text path.
    """

    answer: str = Field(..., description="The final answer text.")
    sources: list[str] = Field(
        default_factory=list,
        description="Optional document ids the answer draws on, as given by the model.",
    )


class DocumentProcessingResponse(BaseModel):
    document_id: str
    original_filename: str
    total_pages: int
    total_chunks: int
    total_embeddings: int
    pages_ocred: int = Field(
        0,
        description="Number of pages that had no extractable text layer and were recovered via OCR.",
    )
    total_images: int = Field(
        0, description="Embedded images extracted from the PDF (multi-modal RAG, Phase 1)."
    )
    images_captioned: int = Field(
        0,
        description="Extracted images successfully captioned by the vision LLM and indexed "
        "as image-derived chunks (Phase 2).",
    )
    images_embedded: int = Field(
        0,
        description="Figure images embedded in CLIP space and indexed into the image vector "
        "store for cross-modal retrieval (Phase 4).",
    )
    total_tables: int = Field(
        0, description="Tables extracted and indexed as structured markdown text (Phase 5)."
    )
    collection: str | None = Field(
        None,
        description="Optional named collection ('doc set') this document was tagged into "
        "at upload time. Empty/None = the default, unscoped library.",
    )
    possible_duplicate_of: str | None = Field(
        None,
        description="When duplicate-document detection is enabled and this upload closely "
        "resembles an already-indexed same-tenant document, that other document's "
        "document_id. The upload always succeeds — this is a warning, never a block.",
    )
    processing_time: float
    status: str


class DocumentDeleteResponse(BaseModel):
    document_id: str
    chunks_removed: int
    status: str


class ApprovalItem(BaseModel):
    """One row in the human approval queue (Feature #5): a gated action
    (web search, document deletion) that was requested without its
    approval flag and recorded for an operator to grant or deny."""

    approval_id: str
    action: str  # "web_search" | "document_delete"
    requested_by: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str  # "pending" | "approved" | "rejected"
    note: str | None = None
    created_at: float
    resolved_at: float | None = None
    resolved_by: str | None = None


class ApprovalListResponse(BaseModel):
    approvals: list[ApprovalItem]
    count: int


class ApprovalResolveRequest(BaseModel):
    approved: bool
    note: str | None = Field(None, description="Optional reason, recorded in the audit log.")


class ApprovalResolveResponse(BaseModel):
    approval_id: str
    action: str
    requested_by: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str
    note: str | None = None
    created_at: float
    resolved_at: float | None = None
    resolved_by: str | None = None


class FeedbackEvent(BaseModel):
    """One recorded feedback event, read back via GET /feedback (Feature
    #6). Mirrors the JSONL event shape feedback_service.record_feedback
    writes — timestamps, ratings, and comments for the app's own view."""

    timestamp: str
    message_id: str
    rating: str
    comment: str | None = None
    reviewer_id: str | None = None
    rubric: dict[str, Any] | None = Field(
        None, description="Optional RubricScores dict, when this event carried a full review."
    )


class FeedbackListResponse(BaseModel):
    """Newest-first feedback events, plus the count of what was returned."""

    events: list[FeedbackEvent]
    count: int


class DocumentListItem(BaseModel):
    document_id: str
    original_filename: str
    stored_filename: str
    file_size: int
    total_pages: int
    total_chunks: int
    total_embeddings: int
    pages_ocred: int
    upload_timestamp: datetime
    collection: str | None = None
    tenant_id: int | None = Field(
        default=None,
        description="Owning tenant. Only meaningful when listed via all_tenants=true — "
        "otherwise every row belongs to the caller's own tenant already.",
    )


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


class DocumentImageItem(BaseModel):
    """One image extracted from a document (multi-modal RAG, Phase 1) —
    metadata only; the bytes are fetched separately via `url`."""

    image_id: str = Field(
        ..., description="Stable identifier for this image, unique within the document."
    )
    document_id: str = Field(
        ..., description="Identifier of the document this image was extracted from."
    )
    page_number: int = Field(..., ge=1, description="1-based page this image appeared on.")
    content_type: str = Field(
        ...,
        description="'figure' (an embedded image) or 'page' (a rasterized full-page render of a low-text page).",
    )
    mime_type: str = Field(..., description="MIME type of the persisted bytes.")
    width: int = Field(..., ge=1, description="Image width in pixels.")
    height: int = Field(..., ge=1, description="Image height in pixels.")
    byte_size: int = Field(..., ge=0, description="Size of the persisted bytes.")
    url: str = Field(..., description="API path at which the image bytes can be fetched.")


class DocumentImagesResponse(BaseModel):
    document_id: str
    total: int
    images: list[DocumentImageItem]


class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=255)
    # Literal[True], not bool — Pydantic rejects consent=false or a
    # missing field with a 422 automatically, so the route doesn't need
    # a manual check. Signup is the first place this app stores real PII
    # (email, password hash); this confirms the user was shown and
    # agreed to that before an account is created.
    consent: Literal[True] = Field(
        ...,
        description="Must be true — confirms the user agreed to how their email/password are stored.",
    )


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUserResponse(BaseModel):
    email: str
    tenant_id: int
    role: str


class HighlightResponse(BaseModel):
    """Rectangles on one PDF page where `text` appears (PyMuPDF
    search_for), used by the in-app PDF citation preview to draw a
    highlight over the cited passage."""

    page_number: int
    page_width: float
    page_height: float
    rects: list[list[float]] = Field(
        default_factory=list,
        description="[x0, y0, x1, y1] float coordinates of each text occurrence on the page.",
    )


class UsageSummaryRow(BaseModel):
    day: str
    request_count: int
    avg_latency_ms: float


class UsageSummaryResponse(BaseModel):
    rows: list[UsageSummaryRow] = Field(default_factory=list)


class AsyncTaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier for tracking async ingestion.")
    document_id: str = Field(..., description="Document identifier assigned to the upload.")
    original_filename: str = Field(..., description="Original filename of the uploaded PDF.")
    status: str = Field("queued", description="Initial task status.")
    progress: float = Field(0.0, description="Ingestion progress percentage (0-100).")
    message: str = Field(
        "Document upload queued for background processing.",
        description="Informational message.",
    )
    created_at: float = Field(..., description="Epoch timestamp when the task was created.")


class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier.")
    document_id: str | None = Field(None, description="Document identifier once assigned.")
    original_filename: str = Field(..., description="Original filename of the uploaded PDF.")
    status: Literal[
        "queued", "extracting", "chunking", "embedding", "indexing", "completed", "failed"
    ] = Field(..., description="Current processing status.")
    progress: float = Field(
        ..., ge=0.0, le=100.0, description="Processing progress percentage (0-100)."
    )
    current_step: str | None = Field(
        None, description="Human-readable description of current step."
    )
    error: str | None = Field(None, description="Error message if status is 'failed'.")
    result: DocumentProcessingResponse | None = Field(
        None, description="Final document processing result once status is 'completed'."
    )
    created_at: float = Field(..., description="Epoch timestamp when the task was created.")
    updated_at: float = Field(..., description="Epoch timestamp when the task was last updated.")
