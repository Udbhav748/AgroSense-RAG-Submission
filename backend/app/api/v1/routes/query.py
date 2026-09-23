"""Routes for natural language question-answering against uploaded documents."""

import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Iterator
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.core.auth import require_auth
from app.core.config import settings
from app.core.exceptions import SessionNotFoundError, VectorStoreNotFoundError
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackEvent,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    WeatherRiskResponse,
)
from app.services.agent_graph import (
    END,
    AgentState,
    GraphContext,
    create_rag_agent_graph,
)
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.feedback_service import list_feedback as list_feedback_events
from app.services.feedback_service import record_feedback
from app.services.llm_client import LLMClient
from app.services.llm_provider import build_llm_client
from app.services.rag_service import ChatService
from app.services.session_repository import (
    get_session_owner,
    list_sessions,
    set_session_title_if_unset,
)
from app.services.session_store import get_session_store
from app.services.validation_service import validate_image_upload
from app.services.vector_store import VectorStore
from app.services.weather_service import WeatherService

router = APIRouter(tags=["Chat"], dependencies=[Depends(require_auth)])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Load the persisted vector index once and reuse it on every request.

    Shared with the upload route (see documents.py) so both hit the same
    in-memory index — a document processed via /upload is immediately
    visible to /chat without a reload. If no index has been persisted yet,
    this returns an empty, uninitialized store rather than raising:
    DocumentProcessingService creates it on the first embedding batch, and
    VectorStore.search() already raises VectorStoreNotFoundError on an
    uninitialized store, so /chat still 404s correctly until then.

    Backend selection: FAISS by default; when PGVECTOR_ENABLED (and
    DATABASE_URL, which pgvector requires), a PgvectorVectorStore backed by
    the document_embeddings table instead. The swap is config-driven so a
    deployment can migrate between stores without touching routes or
    orchestration services, which depend only on the VectorStore ABC.
    """
    if settings.pgvector_enabled:
        from app.services.pgvector_store import PgvectorVectorStore

        store: VectorStore = PgvectorVectorStore()
        from app.core.metrics import get_metrics

        get_metrics().record_vectors(store.total_vectors())
        return store

    store = FAISSVectorStore()
    with contextlib.suppress(VectorStoreNotFoundError):
        store.load()
    from app.core.metrics import get_metrics

    get_metrics().record_vectors(store.total_vectors())
    return store


@lru_cache(maxsize=1)
def get_image_vector_store() -> VectorStore:
    """Load the persisted CLIP *image* index once and reuse it on every
    request (Phase 4 of multi-modal RAG — cross-modal retrieval).

    A second FAISSVectorStore instance pointed at image_index.faiss /
    image_metadata.json (see Settings.image_vector_index_filename /
    _metadata_filename), so image vectors live in their own space and
    persistence files alongside the text index. Same per-process singleton
    lifecycle as get_vector_store, shared by /upload (ingest), /chat
    (fusion), and DELETE /documents/{id} (cleanup). Uninitialized or
    missing index degrades to an empty store — callers gate on
    Settings.clip_embedding_enabled, not on this store's state.
    """
    data_dir = settings.data_dir(settings.vector_store_dir_name)
    store = FAISSVectorStore(
        index_path=data_dir / settings.image_vector_index_filename,
        metadata_path=data_dir / settings.image_vector_metadata_filename,
    )
    with contextlib.suppress(VectorStoreNotFoundError):
        store.load()
    return store


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Build the configured LLM client (see Settings.llm_provider /
    fallback_llm_provider) once and reuse it on every request."""
    return build_llm_client()


def get_chat_service() -> ChatService:
    # Agent memory is the shared per-process singleton (services/
    # agent_memory.py) when the feature is enabled, else None — ChatService
    # treats None as "memory disabled", so enabling the flag later is the
    # entire wiring surface.
    from app.services.agent_memory import get_agent_memory

    return ChatService(
        get_vector_store(),
        get_llm_client(),
        image_vector_store=get_image_vector_store(),
        agent_memory=get_agent_memory() if settings.agent_memory_enabled else None,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    # Constructed here rather than via Depends(): a Depends() dependency
    # runs before FastAPI validates the request body, so a missing vector
    # store would raise (404) before an invalid body could return its 422.
    # Calling it here, after `payload` is already validated, preserves 422s.
    chat_service = get_chat_service()
    session_store = get_session_store()
    tenant_id = getattr(request.state, "tenant_id", None)

    # Resolve session_id: use provided, or create new if none/not found
    session_id = session_store.get_or_create_session(payload.session_id, tenant_id=tenant_id)
    # Best-effort, no-op after the first call for a given session — see
    # session_repository.py. Gives the history-list UI (GET
    # /chat/sessions) something to show besides a raw UUID.
    set_session_title_if_unset(session_id, payload.query)

    # Get server-side history (takes precedence over client-sent history)
    history = session_store.get_history(session_id)
    if history is None:
        # Fallback to client-provided history if session not found
        history = payload.history

    logger.info(
        "chat_request_received",
        extra={
            "extra_fields": {
                "query_length": len(payload.query),
                "top_k": payload.top_k,
                "min_score": payload.min_score,
                "client": getattr(request.state, "client_name", "unknown"),
                "session_id": session_id,
                "history_turns": len(history) if history else 0,
            }
        },
    )

    response = chat_service.handle_query(
        payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
        history=history,
        session_id=session_id,
        confirm_web_search=payload.confirm_web_search,
        structured_response=payload.structured_response,
        tenant_id=tenant_id,
        persona=payload.persona,
        document_ids=payload.document_ids,
        language=payload.language,
    )

    # Append this turn to server-side history
    session_store.append_turn(session_id, "user", payload.query)
    session_store.append_turn(session_id, "assistant", response.answer)

    logger.info(
        "chat_response_sent",
        extra={
            "extra_fields": {
                "answer_length": len(response.answer),
                "retrieved_chunk_count": len(response.retrieved_chunks),
                "processing_time": response.processing_time,
                "client": getattr(request.state, "client_name", "unknown"),
                "session_id": session_id,
            }
        },
    )

    # Attach session_id to response (Pydantic model allows extra fields via model_dump)
    response_data = response.model_dump()
    response_data["session_id"] = session_id
    return ChatResponse(**response_data)


def _sse_line(event: dict[str, Any]) -> str:
    """Serialize one stream_query() or stream_diagnose() event as an SSE `data:` line.

    The "done" event's payload is a ChatResponse (a Pydantic model, not
    plain JSON yet) — every other event type is already a plain dict.
    model_dump(mode="json") matches exactly what response_model=ChatResponse
    would have serialized on POST /chat, so a client parsing "done" gets
    the identical shape either endpoint would give it. If the caller has
    already converted it to a dict (chat_stream injects session_id into the
    done payload before this is called), leave it as-is.
    """
    payload = event.get("payload")
    if hasattr(payload, "model_dump"):
        event = {**event, "payload": payload.model_dump(mode="json")}
    return f"data: {json.dumps(event)}\n\n"


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    # Same auth (router-level Depends(require_auth)) and request body
    # as POST /chat — this is the same pipeline, just fanning out its
    # progress as Server-Sent Events instead of returning one response at
    # the end. POST /chat itself is untouched; this is an additive route.
    chat_service = get_chat_service()
    session_store = get_session_store()
    tenant_id = getattr(request.state, "tenant_id", None)

    # Resolve session_id: use provided, or create new if none/not found
    session_id = session_store.get_or_create_session(payload.session_id, tenant_id=tenant_id)
    set_session_title_if_unset(session_id, payload.query)

    # Get server-side history (takes precedence over client-sent history)
    history = session_store.get_history(session_id)
    if history is None:
        history = payload.history

    logger.info(
        "chat_stream_request_received",
        extra={
            "extra_fields": {
                "query_length": len(payload.query),
                "top_k": payload.top_k,
                "min_score": payload.min_score,
                "client": getattr(request.state, "client_name", "unknown"),
                "session_id": session_id,
                "history_turns": len(history) if history else 0,
            }
        },
    )

    def event_source() -> Iterator[str]:
        full_answer_parts: list[str] = []

        for event in chat_service.stream_query(
            payload.query,
            top_k=payload.top_k,
            min_score=payload.min_score,
            history=history,
            session_id=session_id,
            confirm_web_search=payload.confirm_web_search,
            tenant_id=tenant_id,
            persona=payload.persona,
            document_ids=payload.document_ids,
            language=payload.language,
        ):
            # Capture answer chunks to append to history after stream completes
            if event.get("type") == "answer_chunk":
                token = (
                    event.get("payload", {}).get("token")
                    if isinstance(event.get("payload"), dict)
                    else None
                )
                text = token or event.get("text") or ""
                full_answer_parts.append(text)
            elif event.get("type") == "done":
                # Stream completed — append this turn to server-side history
                full_answer = "".join(full_answer_parts)
                session_store.append_turn(session_id, "user", payload.query)
                session_store.append_turn(session_id, "assistant", full_answer)

                # Inject session_id into the done payload
                done_payload = event["payload"]
                if hasattr(done_payload, "model_dump"):
                    payload_data = done_payload.model_dump(mode="json")
                else:
                    payload_data = done_payload
                payload_data["session_id"] = session_id
                event = {**event, "payload": payload_data}

            yield _sse_line(event)

    return StreamingResponse(event_source(), media_type="text/event-stream")


def _extract_agent_graph_node_output(node_name: str, state: AgentState) -> dict[str, Any]:
    """Extract node-specific output payload for agent graph streaming event."""
    if node_name == "planner":
        action = "retrieve"
        if isinstance(state.plan, dict):
            action = state.plan.get("action", "retrieve")
        elif isinstance(state.plan, str):
            action = state.plan
        out: dict[str, Any] = {"action": action}
        if isinstance(state.plan, dict):
            if "document_id" in state.plan and state.plan["document_id"]:
                out["document_id"] = state.plan["document_id"]
            if "reason" in state.plan and state.plan["reason"]:
                out["reason"] = state.plan["reason"]
        return out

    if node_name == "document_analyst":
        out = {"chunks_retrieved": len(state.retrieved_chunks)}
        if state.document_id:
            out["document_id"] = state.document_id
        return out

    if node_name == "fact_checker":
        grounded = state.is_fact_check_passed()
        score = 1.0
        if isinstance(state.fact_check_result, dict) and "score" in state.fact_check_result:
            score = float(state.fact_check_result["score"])
        elif isinstance(state.fact_check_result, bool):
            score = 1.0 if state.fact_check_result else 0.0
        out = {
            "grounded": grounded,
            "score": score,
        }
        if isinstance(state.fact_check_result, dict) and "details" in state.fact_check_result:
            out["details"] = state.fact_check_result["details"]
        return out

    if node_name == "web_researcher":
        return {"results_count": len(state.web_results)}

    if node_name == "summarizer":
        out = {"summary_length": len(state.draft_answer) if state.draft_answer else 0}
        if state.document_id:
            out["document_id"] = state.document_id
        return out

    if node_name == "synthesizer":
        return {
            "answer_length": len(state.draft_answer) if state.draft_answer else 0,
            "sources_count": len(state.final_response.sources) if state.final_response else 0,
        }

    return {}


@router.post("/chat/agent-graph/stream")
async def chat_agent_graph_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Execute the StateGraph multi-agent pipeline and stream real-time node transitions and tokens."""
    session_store = get_session_store()
    tenant_id = getattr(request.state, "tenant_id", None)

    session_id = session_store.get_or_create_session(payload.session_id, tenant_id=tenant_id)
    set_session_title_if_unset(session_id, payload.query)

    history = session_store.get_history(session_id)
    if history is None:
        history = payload.history

    logger.info(
        "chat_agent_graph_stream_request_received",
        extra={
            "extra_fields": {
                "query_length": len(payload.query),
                "client": getattr(request.state, "client_name", "unknown"),
                "session_id": session_id,
                "tenant_id": tenant_id,
            }
        },
    )

    async def event_source() -> AsyncIterator[str]:
        context = GraphContext(
            llm_client=get_llm_client(),
            vector_store=get_vector_store(),
            image_vector_store=get_image_vector_store(),
        )
        graph = create_rag_agent_graph(context=context)

        initial_state = AgentState(
            query=payload.query,
            history=history,
            session_id=session_id,
            tenant_id=tenant_id,
            confirm_web_search=payload.confirm_web_search,
            persona=payload.persona,
            document_id=payload.document_ids[0] if payload.document_ids else None,
        )

        current_state = initial_state
        current_node = graph.entry_point
        step_index = 0

        while current_node != END and step_index < graph.max_steps:
            step_index += 1
            start_t = time.perf_counter()
            timestamp = time.time()

            yield _sse_line({
                "type": "node_start",
                "node": current_node,
                "timestamp": timestamp,
            })

            try:
                current_state = await graph._invoke_node(current_node, current_state, context)
            except Exception as exc:
                logger.error(
                    "graph_node_execution_failed",
                    extra={"extra_fields": {"node": current_node, "error": str(exc)}},
                )
                current_state = current_state.copy_with(error=f"{type(exc).__name__}: {exc}")
                duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
                yield _sse_line({
                    "type": "node_complete",
                    "node": current_node,
                    "output": {"error": str(exc)},
                    "duration_ms": duration_ms,
                })
                break

            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

            # Stream token events if synthesizer produced/updated draft answer
            if current_node == "synthesizer" and current_state.draft_answer:
                tokens = re.findall(r"\S+|\s+", current_state.draft_answer)
                for tok in tokens:
                    yield _sse_line({"type": "token", "text": tok})

            output = _extract_agent_graph_node_output(current_node, current_state)
            yield _sse_line({
                "type": "node_complete",
                "node": current_node,
                "output": output,
                "duration_ms": duration_ms,
            })

            current_node = await graph._get_next_node(current_node, current_state, context)

        # Append turn to session store
        final_answer = current_state.draft_answer or (
            current_state.final_response.answer if current_state.final_response else ""
        )
        if final_answer:
            session_store.append_turn(session_id, "user", payload.query)
            session_store.append_turn(session_id, "assistant", final_answer)

        final_state_data = current_state.model_dump(mode="json")
        if session_id and "session_id" in final_state_data:
            final_state_data["session_id"] = session_id

        yield _sse_line({
            "type": "graph_done",
            "final_state": final_state_data,
        })

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.get("/weather/risk", response_model=WeatherRiskResponse)
async def get_weather_risk(
    lat: float,
    lon: float,
    crop: str | None = None,
    disease: str | None = None,
) -> WeatherRiskResponse:
    """Assess field microclimate pathogen infection risk and chemical spray advisory from Open-Meteo data."""
    weather_service = WeatherService()
    return await weather_service.get_weather_risk(lat=lat, lon=lon, crop=crop, disease=disease)


@router.post("/chat/diagnose", response_model=ChatResponse)
async def diagnose(
    request: Request,
    image: UploadFile = File(...),
    query: str | None = Form(None),
    session_id: str | None = Form(None),
    confirm_web_search: bool = Form(False),
    engine: str = Form("hybrid"),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    language: str = Form("en"),
) -> ChatResponse:
    chat_service = get_chat_service()
    session_store = get_session_store()
    tenant_id = getattr(request.state, "tenant_id", None)

    contents = await image.read()
    await image.seek(0)
    validate_image_upload(image, contents)

    session_id = session_store.get_or_create_session(session_id, tenant_id=tenant_id)
    set_session_title_if_unset(session_id, query or "Leaf diagnosis")
    history = session_store.get_history(session_id)

    # Module 10 gap-closure (real parallel execution): weather is no
    # longer pre-fetched and awaited HERE, before handle_diagnose even
    # starts -- that was sequential (weather fully done, then vision
    # starts). latitude/longitude are passed straight through instead,
    # and handle_diagnose now runs the weather lookup CONCURRENTLY with
    # vision classification internally (they're independent), via
    # agent_graph.engine.run_concurrent_branches. See handle_diagnose's
    # own docstring for the concurrency detail.
    logger.info(
        "diagnose_request_received",
        extra={
            "extra_fields": {
                "filename": image.filename,
                "content_type": image.content_type,
                "has_accompanying_query": query is not None,
                "session_id": session_id,
                "history_turns": len(history) if history else 0,
                "engine": engine,
                "has_weather_coords": latitude is not None and longitude is not None,
                "language": language,
            }
        },
    )

    response = await run_in_threadpool(
        chat_service.handle_diagnose,
        contents,
        image.filename or "upload",
        image.content_type or "application/octet-stream",
        query=query,
        history=history,
        session_id=session_id,
        confirm_web_search=confirm_web_search,
        tenant_id=tenant_id,
        engine=engine,
        language=language,
        latitude=latitude,
        longitude=longitude,
    )

    # Append this turn to server-side history, same shape /chat uses. The
    # "user" turn has no raw text to store (the input was an image, not a
    # message) — reconstructed from the diagnosis result instead, so a
    # resumed conversation reads sensibly rather than showing a blank user
    # turn or just the assistant's answer with no visible question.
    if response.diagnosis is not None:
        diagnosis_label = (
            f"{response.diagnosis.disease} on {response.diagnosis.crop}"
            if response.diagnosis.disease != "healthy"
            else f"healthy {response.diagnosis.crop}"
        )
        user_turn = f"[Uploaded a leaf photo — diagnosed: {diagnosis_label}]"
        if query:
            user_turn += f" {query}"
    else:
        user_turn = query or "[Uploaded a leaf photo for diagnosis]"
    session_store.append_turn(session_id, "user", user_turn)
    session_store.append_turn(session_id, "assistant", response.answer)

    logger.info(
        "diagnose_response_sent",
        extra={
            "extra_fields": {
                "answer_length": len(response.answer),
                "retrieved_chunk_count": len(response.retrieved_chunks),
                "processing_time": response.processing_time,
                "client": getattr(request.state, "client_name", "unknown"),
                "session_id": session_id,
            }
        },
    )

    # Attach the resolved session_id, same defensive override /chat uses —
    # handle_diagnose already threads session_id through to _respond() on
    # every return path, so this is redundant today, but keeps the two
    # routes' guarantees identical rather than relying on that staying true.
    response_data = response.model_dump()
    response_data["session_id"] = session_id
    return ChatResponse(**response_data)


@router.post("/chat/diagnose/stream")
async def diagnose_stream(
    request: Request,
    image: UploadFile = File(...),
    query: str | None = Form(None),
    session_id: str | None = Form(None),
    confirm_web_search: bool = Form(False),
    engine: str = Form("hybrid"),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    language: str = Form("en"),
) -> StreamingResponse:
    """Streamed SSE endpoint for plant leaf photo diagnosis and treatment recommendations.

    Streams real-time trace events, immediate vision diagnosis prediction, retrieval trace,
    and LLM token chunks over SSE text/event-stream.
    """
    chat_service = get_chat_service()
    session_store = get_session_store()
    tenant_id = getattr(request.state, "tenant_id", None)

    contents = await image.read()
    await image.seek(0)
    validate_image_upload(image, contents)

    session_id = session_store.get_or_create_session(session_id, tenant_id=tenant_id)
    set_session_title_if_unset(session_id, query or "Leaf diagnosis")
    history = session_store.get_history(session_id)

    weather_risk: WeatherRiskResponse | None = None
    if latitude is not None and longitude is not None:
        try:
            weather_service = WeatherService()
            weather_risk = await weather_service.get_weather_risk(
                lat=latitude, lon=longitude
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch microclimate risk for (%s, %s): %s",
                latitude,
                longitude,
                exc,
            )

    logger.info(
        "diagnose_stream_request_received",
        extra={
            "extra_fields": {
                "filename": image.filename,
                "content_type": image.content_type,
                "has_accompanying_query": query is not None,
                "session_id": session_id,
                "history_turns": len(history) if history else 0,
                "engine": engine,
                "has_weather": weather_risk is not None,
                "language": language,
            }
        },
    )

    filename = image.filename or "upload"
    content_type = image.content_type or "application/octet-stream"

    def event_source() -> Iterator[str]:
        full_answer_parts: list[str] = []

        for event in chat_service.stream_diagnose(
            contents,
            filename,
            content_type,
            query=query,
            history=history,
            session_id=session_id,
            confirm_web_search=confirm_web_search,
            tenant_id=tenant_id,
            engine=engine,
            weather_risk=weather_risk,
            language=language,
        ):
            if event.get("type") == "answer_chunk":
                token = (
                    event.get("payload", {}).get("token")
                    if isinstance(event.get("payload"), dict)
                    else None
                )
                text = token or event.get("text") or ""
                full_answer_parts.append(text)
            elif event.get("type") == "done":
                full_answer = "".join(full_answer_parts)
                done_payload = event["payload"]
                if hasattr(done_payload, "model_dump"):
                    payload_data = done_payload.model_dump(mode="json")
                else:
                    payload_data = done_payload
                payload_data["session_id"] = session_id

                diagnosis_info = payload_data.get("diagnosis")
                if diagnosis_info:
                    crop = diagnosis_info.get("crop", "")
                    disease = diagnosis_info.get("disease", "")
                    if disease and disease != "healthy":
                        diag_label = f"{disease} on {crop}" if crop else disease
                    else:
                        diag_label = f"healthy {crop}" if crop else "healthy"
                    user_turn = f"[Uploaded a leaf photo — diagnosed: {diag_label}]"
                    if query:
                        user_turn += f" {query}"
                else:
                    user_turn = query or "[Uploaded a leaf photo for diagnosis]"

                answer_text = payload_data.get("answer") or full_answer
                session_store.append_turn(session_id, "user", user_turn)
                session_store.append_turn(session_id, "assistant", answer_text)

                event = {**event, "payload": payload_data}

            yield _sse_line(event)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.post("/chat/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest, request: Request) -> FeedbackResponse:
    # reviewer_id comes from the authenticated caller (require_auth,
    # the router-level dependency), never from the request body — a
    # client-supplied reviewer identity would let one API key pose as
    # multiple reviewers and fabricate Inter-Annotator Agreement.
    reviewer_id = getattr(request.state, "client_name", None)
    record_feedback(
        feedback.message_id,
        feedback.rating,
        feedback.comment,
        reviewer_id=reviewer_id,
        rubric=feedback.rubric.model_dump() if feedback.rubric else None,
    )

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "feedback_submitted",
                "path": request.url.path,
                "message_id": feedback.message_id,
                "rating": feedback.rating,
                "client": reviewer_id or "unknown",
                "has_rubric": feedback.rubric is not None,
            }
        },
    )

    return FeedbackResponse(status="recorded")


@router.get("/feedback", response_model=FeedbackListResponse)
def list_feedback(
    request: Request, limit: int = 50, reviewer_id: str | None = None
) -> FeedbackListResponse:
    """Surface recorded feedback events back to the app (Feature #6):
    newest-first, best-effort. An authenticated caller sees feedback from
    their own API key/client by default; admins may pass reviewer_id to
    inspect a specific reviewer's judgments (or null to see all).
    """
    from app.core import permissions

    caller = getattr(request.state, "client_name", None)
    role = getattr(request.state, "role", None)
    if reviewer_id is not None and role != "admin":
        permissions.check_permission(
            request,
            permissions.ANALYTICS_VIEW,
            message="Filtering feedback by reviewer requires the admin role.",
        )
    resolved_reviewer = reviewer_id
    if resolved_reviewer is None and role != "admin":
        resolved_reviewer = caller
    if resolved_reviewer is None and role != "admin":
        # No caller identity (shouldn't happen behind require_auth) — don't
        # leak other reviewers' feedback by returning everything.
        resolved_reviewer = "__none__"

    events = [
        FeedbackEvent(**event)
        for event in list_feedback_events(limit=limit, reviewer_id=resolved_reviewer)
    ]

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "feedback_listed",
                "path": request.url.path,
                "client": caller or "unknown",
                "reviewer_id": resolved_reviewer,
                "count": len(events),
            }
        },
    )
    return FeedbackListResponse(events=events, count=len(events))


@router.get("/chat/sessions")
def list_chat_sessions(request: Request) -> dict[str, Any]:
    """List the caller's own past conversations (title, timestamps),
    newest-accessed first — the history sidebar's data source. []
    when the DB is disabled: session listing needs durable storage that
    the in-memory store, tied to one process, can't provide anyway."""
    tenant_id = getattr(request.state, "tenant_id", None)
    sessions = list_sessions(tenant_id)

    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "sessions_listed",
                "path": request.url.path,
                "count": len(sessions),
                "client": getattr(request.state, "client_name", "unknown"),
            }
        },
    )
    return {"sessions": sessions, "total": len(sessions)}


def _check_session_ownership(request: Request, session_id: str) -> None:
    """Same "unknown isn't a mismatch, but a known mismatch is a 404"
    pattern documents.py's delete route already uses for
    get_document_owner — a non-owner sees the same response a
    genuinely missing session_id would give, never a 403 that would
    confirm the session exists."""
    requester_tenant_id = getattr(request.state, "tenant_id", None)
    if requester_tenant_id is None:
        return
    owner_tenant_id = get_session_owner(session_id)
    if owner_tenant_id is not None and owner_tenant_id != requester_tenant_id:
        raise SessionNotFoundError(f"No session found with id {session_id}")


@router.get("/chat/sessions/{session_id}")
def get_chat_session(session_id: str, request: Request) -> dict[str, Any]:
    """Full turn history for one session — what the frontend calls to
    resume a past conversation from the history list."""
    _check_session_ownership(request, session_id)

    session_store = get_session_store()
    history = session_store.get_history(session_id)
    if history is None:
        raise SessionNotFoundError(f"No session found with id {session_id}")

    return {"session_id": session_id, "turns": history}


@router.delete("/chat/sessions/{session_id}")
def delete_chat_session(session_id: str, request: Request) -> dict[str, Any]:
    """Delete one session by id — replaces the old header-based
    DELETE /chat/session (which read X-Session-ID and had a dead-code
    path that tried, and failed, to also read a JSON body). A path
    param is the natural shape once a history list UI exists: each row
    has its own delete action."""
    _check_session_ownership(request, session_id)

    session_store = get_session_store()
    deleted = session_store.delete_session(session_id)
    if deleted:
        logger.info(
            "audit_event",
            extra={
                "extra_fields": {
                    "event": "session_deleted",
                    "path": request.url.path,
                    "session_id": session_id,
                    "client": getattr(request.state, "client_name", "unknown"),
                }
            },
        )
        return {"status": "deleted", "session_id": session_id}

    return {"status": "not_found", "session_id": session_id}


@router.delete("/cache")
def invalidate_cache(request: Request) -> dict[str, Any]:
    """Invalidate the semantic query cache when new documents are uploaded or indexed."""
    from app.core import permissions
    from app.services.cache_service import cache_service

    permissions.check_permission(
        request,
        permissions.CACHE_INVALIDATE,
        message="Invalidating cache requires the admin role.",
    )
    cache_service.invalidate()
    logger.info(
        "audit_event",
        extra={
            "extra_fields": {
                "event": "cache_invalidated",
                "path": request.url.path,
                "client": getattr(request.state, "client_name", "unknown"),
            }
        },
    )
    return {"status": "invalidated", "message": "Semantic query cache cleared successfully"}


@router.get("/cache/metrics")
def get_cache_metrics(request: Request) -> dict[str, Any]:
    """Return live metrics and statistics for the semantic query cache."""
    from app.services.cache_service import cache_service

    return cache_service.get_metrics()
