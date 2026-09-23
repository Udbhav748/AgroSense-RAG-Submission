"""Orchestrates a small hand-rolled agent: plans which tool a query needs,
executes it, and corrects the result when retrieval or generation comes up
short.

Deliberately plain Python — no LangChain/LangGraph agent runtime. A
planner (_plan) picks one of three actions from a query (and, for
"summarize", which document_id it names); a tool executes it
(conversational canned reply, retrieval + generation, or document
summarization). The `retrieve` action runs a small corrective RAG loop:
_grade_retrieval scores what came back, a weak/insufficient grade can pull
in web search results (services/web_search_service.py) alongside or
instead of document chunks, and _correct catches the specific failure mode
of "there was context but the model didn't use it," regenerating up to
_MAX_LLM_CALLS total generate() calls per request — escalating to web
search once, if not already tried, before giving up.

ChatService still contains no retrieval, prompt, or LLM logic itself — it
coordinates RetrievalService, PromptBuilder, SummarizationService,
WebSearchService, and LLMClient. It depends on the VectorStore and
LLMClient interfaces, never on a concrete implementation (FAISSVectorStore,
GeminiClient); those are constructed elsewhere and handed in.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.core.exceptions import AppError, ChatServiceError, WebSearchError
from app.core.metrics import get_metrics
from app.core.usage_tracking import current_usage, reset_usage
from app.models.schemas import (
    ChatResponse,
    DiagnosisInfo,
    SourceReference,
    WeatherRiskResponse,
)
from app.services.agent_events import log_agent_handoff
from app.services.cache_service import cache_service
from app.services.local_research_agent import LocalResearchAgent
from app.services.prompt_builder import (
    FALLBACK_REPLY,
    GENERATION_ERROR_REPLY,
    PROMPT_VERSION,
    REFLECTION_INSTRUCTION,
    build_prompt,
    build_structured_prompt,
    strip_sources_section,
)
from app.services.prompt_injection_service import detect_possible_injection
from app.services.rag.router import extract_crop_context as _extract_crop_context
from app.services.research_agent import ResearchAgent, ResearchFindings
from app.services.retrieval_service import retrieve
from app.services.router_agent import RouterAgent
from app.services.structured_output import parse_structured_answer
from app.services.summarization_service import summarize_document
from app.services.tools.base import ToolContext
from app.services.tools.factory import build_tool_registry
from app.services.vision_client import diagnose_image
from app.services.vision_qa_service import try_vision_qa
from app.services.web_search_service import search_web, web_search_ready

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from app.models.document import RetrievedChunk, VisionPrediction, WebSearchResult
    from app.services.agent_executor import AgentExecutor
    from app.services.agent_memory import AgentMemory
    from app.services.llm_client import LLMClient
    from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Hard ceiling on generate() calls per /chat request (the initial answer,
# plus the corrective loop's reflection and web-fallback regenerations) —
# the loop-prevention/Loop Rate control. The corrective loop as written
# never needs more than 3 (see _correct), so this is a defensive backstop
# against a future change to the loop rather than something normal traffic
# is expected to hit; hitting it logs "loop_capped".
_MAX_LLM_CALLS = 3


def _capture_prompt(prompt: str, *, variant: str) -> None:
    """Log the exact prompt when Settings.log_prompt_content is on.

    Off by default — prompts embed retrieved document text, so capturing
    them is an explicit data-retention decision (Settings.log_prompt_max_chars
    caps what's logged; the capture is a truncated prefix with a marker,
    never silently cut). This is what closes the "prompt recorded but not
    captured" debugging gap in the checklist.
    """
    if not settings.log_prompt_content:
        return
    max_chars = settings.log_prompt_max_chars
    captured = prompt if len(prompt) <= max_chars else prompt[:max_chars] + "...[truncated]"
    logger.info(
        "prompt_captured",
        extra={
            "extra_fields": {
                "prompt_version": PROMPT_VERSION,
                "variant": variant,
                "prompt_length": len(prompt),
                "captured_length": len(captured),
                "prompt": captured,
            }
        },
    )


# Each entry is (normalized exact phrases, canned response). Checked in
# order; the query must match one of the phrases entirely (after
# normalization) — a real question that merely contains a word like
# "thanks" mid-sentence should still go to the RAG pipeline.
_CONVERSATIONAL_INTENTS = [
    (
        {"hi", "hey", "yo", "hiya"},
        "Hi! I'm AgroSense-RAG. How can I help you with your uploaded documents today?",
    ),
    (
        {"hello", "good morning", "good afternoon", "good evening"},
        "Hello! Upload a document or ask me a question about one you've already uploaded.",
    ),
    (
        {"thanks", "thank you", "thanks a lot", "thank you so much", "many thanks"},
        "You're welcome! Let me know if you need help understanding anything in your documents.",
    ),
    (
        {"bye", "goodbye", "see you", "see ya", "farewell"},
        "Goodbye! Come back anytime you have questions about your documents.",
    ),
    (
        {"who are you", "what are you"},
        "I'm AgroSense-RAG, an AI-powered document assistant. I can analyze your uploaded "
        "documents, answer questions, summarize content, and help you quickly find "
        "information.",
    ),
    (
        {"what can you do", "help", "how do you work", "what do you do"},
        "I can:\n"
        "- Answer questions from uploaded PDFs\n"
        "- Summarize documents\n"
        "- Explain concepts\n"
        "- Find important information\n"
        "- Help you study or review documents",
    ),
    # Meta/status remarks — the user is checking whether the bot is
    # working or venting about it, not asking a document question. Without
    # this, these fall through to retrieve and (depending on min_score)
    # can return an odd, technically-grounded-but-irrelevant answer built
    # from whatever chunk happened to score highest.
    (
        {
            "why not responding",
            "why are you not responding",
            "why aren't you responding",
            "why is this not responding",
            "why isn't this working",
            "why is this not working",
            "this is not working",
            "this isn't working",
            "not working",
            "are you there",
            "are you working",
            "is this working",
            "is this thing working",
            "hello are you there",
            "anyone there",
            "is anyone there",
            "did you get my message",
            "what happened",
            "what happend",
            "did that work",
        },
        "I'm here and working — I just didn't find anything relevant to that in "
        "your uploaded documents. Try asking a specific question about what's in "
        'them, like "what does this document say about...".',
    ),
]

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Matches "summarize"/"summarise"/"summary" anywhere in the query.
_SUMMARIZE_RE = re.compile(r"\bsummar(?:y|ize|ise)\b", re.IGNORECASE)

# A document_id is a uuid4 (see upload_service.save_uploaded_file) — this
# is how a "summarize" query names which document it means. Queries that
# mention "summarize" without one fall back to the normal retrieve action:
# there's no document-name index to resolve a title against.
_DOCUMENT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)

# How many of the most recent conversation turns get sent to the LLM.
_MAX_HISTORY_TURNS = 6

# Length of the excerpt shown per cited chunk in ChatResponse.sources.
_EXCERPT_LENGTH = 200


def _normalize(query: str) -> str:
    stripped = _PUNCTUATION_RE.sub("", query.lower())
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _excerpt(text: str) -> str:
    """First ~200 chars of a chunk's text, trimmed at a clean boundary."""
    stripped = text.strip()
    if len(stripped) <= _EXCERPT_LENGTH:
        return stripped
    return stripped[:_EXCERPT_LENGTH].rstrip() + "…"


def _source_references(chunks: list[RetrievedChunk], start: int = 1) -> list[SourceReference]:
    """One SourceReference per retrieved chunk — chunk-level, not deduped
    by document, so a citation always points at the specific passage the
    answer actually drew from.

    number starts at `start` and counts up in list order — this must
    match the [N] labels prompt_builder.build_prompt() puts on these same
    chunks (same order, same starting point), since that's what the
    model's inline citation markers refer back to.
    """
    return [
        SourceReference(
            number=i,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            excerpt=_excerpt(chunk.text),
            # Defaults ("text"/None) cover ordinary chunks untouched by
            # multi-modal RAG; image-caption/table chunks carry these in
            # their own metadata (see chunking_service.chunk_text) and
            # just pass through here — this file's ownership of the
            # RAG/citation surface, not multi-modal RAG's.
            content_type=chunk.metadata.get("content_type", "text"),
            page_number=chunk.metadata.get("page_number"),
        )
        for i, chunk in enumerate(chunks, start=start)
    ]


def _web_source_references(results: list[WebSearchResult], start: int = 1) -> list[SourceReference]:
    """One SourceReference per web result, using the same shape as document
    citations: document_id='web' marks it as non-document, chunk_id is the
    URL itself (a web result has no chunk id, but the URL is a stable,
    unique-enough identifier), and url carries the link for the frontend to
    render out.

    number continues from `start` (see _source_references) — web results
    are always numbered after document chunks, matching
    prompt_builder.build_prompt()'s context ordering (documents first).
    """
    return [
        SourceReference(
            number=i,
            document_id="web",
            chunk_id=result.url,
            excerpt=_excerpt(result.snippet),
            url=result.url,
        )
        for i, result in enumerate(results, start=start)
    ]


def _answer_source(chunks: list[RetrievedChunk], web_results: list[WebSearchResult]) -> str:
    """ "documents" | "web" | "mixed", based on which context actually made
    it into the final prompt — not on what was attempted. A web search that
    was tried but returned nothing doesn't count as "web"."""
    if not web_results:
        return "documents"
    return "web" if not chunks else "mixed"


# Small English stopword set for the lexical groundedness check (Feature
# #4). Deliberately hand-maintained rather than pulled from a library —
# the project's monitoring/eval philosophy is dependency-free stand-ins,
# and a 60-word set is more than enough to strip the colorless filler
# ("the", "and", "is") that would otherwise dominate a token-overlap
# ratio. Looser (case-insensitive, accent-naive) than the gold standard
# lists; fine for a signal that's a pointer to double-check, not a gate.
_GROUNDEDNESS_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "for",
        "with",
        "without",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "not",
        "no",
        "yes",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "my",
        "your",
        "our",
        "their",
        "about",
        "into",
        "between",
        "over",
        "under",
        "again",
        "also",
        "just",
        "very",
        "too",
        "same",
        "some",
        "such",
        "only",
        "other",
        "any",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "much",
    }
)


def _content_tokens(text: str) -> set[str]:
    """Lowercased, stopword-stripped, alphanumeric content tokens."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _GROUNDEDNESS_STOPWORDS and len(token) > 1
    }


def _grounding_score(
    answer: str, chunks: list[RetrievedChunk], web_results: list[WebSearchResult]
) -> float:
    """Fraction of the answer's content tokens that also appear in the
    retrieved context (chunk text + web snippets). 1.0 if the answer has
    no content tokens to compare. A purely lexical proxy for groundedness —
    deliberately cheap and dependency-free; it can't catch a fluent
    fabrication that happens to reuse source vocabulary, but it reliably
    flags the more common failure of an answer produced from nothing.
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = set()
    for chunk in chunks:
        context_tokens |= _content_tokens(chunk.text)
    for result in web_results:
        context_tokens |= _content_tokens(result.snippet)
    if not context_tokens:
        return 1.0
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)


def _detect_hallucination(
    answer: str,
    chunks: list[RetrievedChunk],
    web_results: list[WebSearchResult],
) -> tuple[bool, float | None]:
    """Run the Feature #4 lexical groundedness check.

    Returns (detected, score). Detected is True when the check has
    something to say (there was context, the answer is long enough to
    score) and the answer's content-token overlap with that context falls
    below Settings.hallucination_grounding_threshold. Never blocks — it's
    a signal surfaced to the user to double-check against the sources.
    """
    if not settings.hallucination_detection_enabled:
        return False, None
    if len(answer.strip()) < settings.hallucination_min_answer_chars:
        return False, None
    if not chunks and not web_results:
        return False, None
    score = _grounding_score(answer, chunks, web_results)
    detected = score < settings.hallucination_grounding_threshold
    return detected, round(score, 4)


# Matches prompt_builder._SOURCES_HEADING's own marker string (not the
# regex itself — this operates incrementally on a live token stream,
# where a full-string regex can't run until the heading has fully
# arrived). See _stream_filtering_sources.
_SOURCES_MARKER = "sources:"
_SOURCES_HOLD_BACK = len(_SOURCES_MARKER) + 4  # margin for surrounding newlines


def _stream_filtering_sources(chunks: Iterator[str]) -> Iterator[str]:
    """Forward pieces of a raw LLM token stream, holding back enough
    trailing text that the model's own "Sources:" heading (see
    prompt_builder.strip_sources_section — the non-streaming path strips
    this from the complete answer before it's ever shown) never partially
    reaches the client before it can be recognized and dropped. Once the
    heading is found, everything from there on is suppressed — matching
    strip_sources_section's behavior of removing "Sources:" through the
    end of the text — but the underlying iterator is still drained so the
    caller (which accumulates the full raw text separately) sees it all.
    """
    pending = ""
    sources_found = False
    for chunk in chunks:
        if sources_found:
            continue
        pending += chunk
        idx = pending.lower().find(_SOURCES_MARKER)
        if idx != -1:
            safe_prefix = pending[:idx].rstrip("\n")
            if safe_prefix:
                yield safe_prefix
            sources_found = True
            pending = ""
            continue
        if len(pending) > _SOURCES_HOLD_BACK:
            flush, pending = pending[:-_SOURCES_HOLD_BACK], pending[-_SOURCES_HOLD_BACK:]
            yield flush
    if not sources_found and pending:
        yield pending


def _trace_event(stage: str, detail: dict[str, Any]) -> dict[str, Any]:
    """Build an SSE trace event dict and log it server-side in the same
    call — the same pipeline progress already logged via plan_decided/
    retrieval_completed/retrieval_graded/etc. elsewhere in this file, this
    just adds one more named line marking exactly what was fanned out to
    the client, for correlating client-observed timing against the
    existing structured logs."""
    logger.info("trace_event_emitted", extra={"extra_fields": {"stage": stage, **detail}})
    return {"type": "trace", "stage": stage, "detail": detail}


def _match_conversational_reply(query: str) -> str | None:
    """Return a canned reply if query is small talk, else None."""
    normalized = _normalize(query)
    for phrases, response in _CONVERSATIONAL_INTENTS:
        if normalized in phrases:
            return response
    return None


def _build_diagnosis_query(
    prediction: VisionPrediction,
    user_query: str | None = None,
    collection: str | None = None,
) -> str:
    """Turn a vision prediction into the query fed to the existing
    retrieval pipeline. Includes crop, not just disease name: several
    LeafSense classes share a disease name across crops (e.g.
    "Bacterial_spot" exists for both peach and tomato, with different
    corpus content), so crop alone disambiguates which document's chunks
    should actually match."""
    crop = collection or prediction.crop or _extract_crop_context(user_query) or "crop"
    disease = prediction.disease
    base = f"{disease} on {crop}" if disease != "healthy" else f"healthy {crop}"
    return f"{base}. {user_query}" if user_query else base


def _build_diagnosis_info(prediction: VisionPrediction) -> DiagnosisInfo:
    """Build the DiagnosisInfo response field from a vision prediction.
    Extracted from handle_diagnose/stream_diagnose, which each constructed
    this identically 3 times (cache-hit stamping, a research-agent direct
    hit, and the final response) — one implementation now, not six."""
    return DiagnosisInfo(
        raw_class=prediction.raw_class,
        crop=prediction.crop,
        disease=prediction.disease,
        confidence=prediction.confidence,
        low_confidence=prediction.low_confidence,
        heatmap_base64=prediction.heatmap_base64,
        infected_area_percentage=prediction.infected_area_percentage,
        lesion_count=prediction.lesion_count,
    )


def _format_weather_instruction(weather_risk: WeatherRiskResponse) -> str:
    """Format structured microclimate intelligence into an agronomy prompt directive."""
    loc = (
        weather_risk.location
        if isinstance(weather_risk.location, dict)
        else weather_risk.location.model_dump()
    )
    lat = loc.get("latitude", loc.get("lat", 0.0))
    lon = loc.get("longitude", loc.get("lon", 0.0))
    return (
        f"[LOCAL FIELD MICROCLIMATE & WEATHER INTELLIGENCE]\n"
        f"- Location Coordinates: Latitude {lat}, Longitude {lon}\n"
        f"- Atmospheric Conditions: {weather_risk.current.temperature_c}°C, {weather_risk.current.humidity_pct}% Relative Humidity, {weather_risk.current.precipitation_mm}mm Precipitation, Wind {weather_risk.current.wind_kmh} km/h\n"
        f"- Pathogen Infection Risk Assessment: {weather_risk.risk_level} (Score: {weather_risk.risk_score})\n"
        f"- Meteorological Analysis: {weather_risk.favorable_conditions_summary}\n"
        f"- Chemical Spray Advisory: {weather_risk.spray_advisory}\n\n"
        f"AGRONOMY DIRECTIVE: Ground your diagnostic advice in this local field weather context. Explicitly advise the grower on pathogen pressure under these conditions and whether current wind/rain conditions permit immediate spraying or require waiting for an optimal window."
    )


@dataclass
class PlanDecision:
    action: str  # "conversational" | "retrieve" | "summarize" | "diagnose"
    document_id: str | None = None
    crop: str | None = None
    collection: str | None = None


@dataclass
class RetrievalAugmentation:
    """Return type of ChatService._augment_weak_retrieval — see its
    docstring. At most one of vision_qa/local_research/research_agent
    produces a direct final_answer; a plain web_search fallback only ever
    populates web_results/web_search_attempted."""

    final_answer: str | None = None
    final_chunks: list[RetrievedChunk] | None = None
    tool_used: str | None = None  # "vision_qa" | "local_research" | "research_agent" | None
    extra_steps: int = 0
    web_results: list[WebSearchResult] = field(default_factory=list)
    web_search_attempted: bool = False


class ChatService:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        image_vector_store: VectorStore | None = None,
        agent_memory: AgentMemory | None = None,
    ):
        self._vector_store = vector_store
        self._llm_client = llm_client
        # Phase 4 cross-modal store: image vectors (CLIP space) belong to
        # their own index, injected from get_image_vector_store(). Optional
        # — None keeps retrieval text-only regardless of the config flag.
        self._image_vector_store = image_vector_store
        # Multi-agent layer (config-gated; construction is cheap — no LLM
        # calls happen until decide()/run() are actually invoked):
        #   - the router agent upgrades the keyword planner with LLM intent
        #     classification ("research" is an action the regex planner
        #     can't express);
        #   - the research agent owns the weak/insufficient-retrieval path
        #     with a plan→search→read→synthesize loop.
        # Both degrade to the existing deterministic paths when disabled or
        # when they fail, so enabling them is strictly additive.
        self._router_agent = RouterAgent(llm_client, fallback_planner=self._plan)
        self._research_agent = ResearchAgent(llm_client)
        self._local_research_agent = LocalResearchAgent(llm_client, vector_store)
        # Dynamic tool registry (services/tools/): the agent-facing
        # invocation surface. Where this service's inline paths call
        # retrieve()/search_web()/etc. directly (each @track_tool'd once),
        # a future Planner/Executor agent calls registry.execute(name, args)
        # instead. Construction is cheap — no I/O until execute() runs.
        self._tool_registry = build_tool_registry()
        self._tool_context = ToolContext(
            vector_store=vector_store,
            llm_client=llm_client,
            agent_memory=agent_memory,
        )
        # AgentExecutor (services/agent_executor.py): the opt-in
        # planner→ReAct-executor orchestration layer behind
        # Settings.agent_executor_enabled. When enabled, eligible queries
        # are handed to an AgentExecutor that asks PlanningAgent for one
        # next tool step at a time — each step sees every earlier step's
        # result before it's chosen — instead of this service's inline
        # corrective loop. Built
        # lazily (construction is cheap — no LLM calls) and only invoked
        # when the flag is on, so the inline path is byte-for-byte
        # unchanged by default.
        self._agent_executor: AgentExecutor | None = None
        # Agent memory (services/agent_memory.py): a bounded per-session
        # working memory of turns + extracted facts, injected into later
        # prompts. Optional — None (the default) disables it entirely; the
        # route wires the shared singleton when Settings.agent_memory_enabled.
        # Every interaction is best-effort and never raises into the
        # pipeline: memory is a quality enhancement, not a new failure mode.
        self._agent_memory = agent_memory
        # Response cache: keyed by (normalized_query, session_id, doc_set_hash)
        # Only caches final responses after corrective loop completes.
        # Max 256 entries, LRU eviction.
        self._response_cache: dict[str, ChatResponse] = {}
        self._cache_keys: list[str] = []
        self._max_cache_size = 256

    def _make_cache_key(
        self,
        query: str,
        session_id: str | None,
        vector_store: VectorStore,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        """Create a cache key from normalized query, session_id, document set
        hash, tenant_id, and the scoped document_ids.

        tenant_id and document_ids must both be part of the key: retrieve()
        filters by both, so two callers asking the same question with
        different tenant or document scoping can get different, correctly-
        scoped chunks — without either here, the second caller could be served
        the first caller's cached answer straight past that filter.
        """
        normalized = _normalize(query)
        doc_hash = str(vector_store.total_vectors())  # simple proxy for document set
        session = session_id or "no-session"
        tenant = str(tenant_id) if tenant_id is not None else "no-tenant"
        # Sorted, stable representation of the scoped document set — None,
        # empty, and "all" must collapse to the same unscoped bucket.
        if document_ids:
            doc_scope = hashlib.sha256(",".join(sorted(document_ids)).encode()).hexdigest()[:16]
        else:
            doc_scope = "no-docs"
        key_str = f"{normalized}|{session}|{doc_hash}|{tenant}|{doc_scope}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _get_cached_response(
        self,
        query: str,
        crop: str | None = None,
        disease: str | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> ChatResponse | None:
        """Get cached response from SemanticQueryCache."""
        cached_data = cache_service.get(
            query=query,
            crop=crop,
            disease=disease,
            tenant_id=tenant_id,
            document_ids=document_ids,
        )
        if cached_data is not None:
            if isinstance(cached_data, ChatResponse):
                return cached_data
            if isinstance(cached_data, dict):
                try:
                    return ChatResponse.model_validate(cached_data)
                except Exception as exc:
                    logger.debug("Failed to deserialize cached ChatResponse: %s", exc)
                    return None
        return None

    def _cache_response(
        self,
        query: str,
        response: ChatResponse,
        crop: str | None = None,
        disease: str | None = None,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> None:
        """Cache response with adaptive TTL into SemanticQueryCache."""
        cache_service.set(
            query=query,
            response=response.model_dump(),
            crop=crop,
            disease=disease,
            tenant_id=tenant_id,
            document_ids=document_ids,
            ttl=3600,
        )

    def _plan(self, query: str, history: list[dict[str, str]] | None = None) -> PlanDecision:
        """Decide which tool this query needs.

        Plain keyword/regex checks — no LLM call, no external planner.
        history is accepted for a future planner that considers context,
        but isn't used by these checks today.
        """
        crop = _extract_crop_context(query)
        collection = crop

        if _match_conversational_reply(query) is not None:
            return PlanDecision(action="conversational", crop=crop, collection=collection)

        if _SUMMARIZE_RE.search(query):
            match = _DOCUMENT_ID_RE.search(query)
            if match:
                return PlanDecision(
                    action="summarize",
                    document_id=match.group(0),
                    crop=crop,
                    collection=collection,
                )

        return PlanDecision(action="retrieve", crop=crop, collection=collection)

    def _agent_executor_instance(self) -> AgentExecutor:
        """Lazily construct the AgentExecutor (services/agent_executor.py)
        the first time it's needed. Cheap — no LLM calls until execute().
        Deliberately not built in __init__: the executor is only ever
        exercised when Settings.agent_executor_enabled, and constructing it
        unconditionally would force every /chat request (and the eval
        harness, which constructs ChatService directly) through its import
        chain even when disabled."""
        if self._agent_executor is None:
            from app.services.agent_executor import AgentExecutor
            from app.services.planning_agent import PlanningAgent

            self._agent_executor = AgentExecutor(
                llm_client=self._llm_client,
                tool_registry=self._tool_registry,
                planning_agent=PlanningAgent(self._llm_client, self._tool_registry),
                agent_memory=self._agent_memory,
            )
        return self._agent_executor

    def _handle_via_executor(
        self,
        query: str,
        history: list[dict[str, str]] | None,
        session_id: str | None,
        tenant_id: int | None,
        top_k: int | None,
        min_score: float | None,
        confirm_web_search: bool,
        persona: str | None,
    ) -> ChatResponse:
        """Run a query through the AgentExecutor path and return a
        ChatResponse. The executor is async; handle_query is sync, so this
        bridges by driving the executor's coroutine on a short-lived event
        loop. Called only when Settings.agent_executor_enabled."""
        import asyncio

        executor = self._agent_executor_instance()
        context = ToolContext(
            vector_store=self._vector_store,
            llm_client=self._llm_client,
            tenant_id=tenant_id,
            session_id=session_id,
            agent_memory=self._agent_memory,
        )

        async def _run() -> ChatResponse:
            result = await executor.execute(
                query,
                context=context,
                session_id=session_id,
                history=history,
                top_k=top_k,
                min_score=min_score,
                confirm_web_search=confirm_web_search,
                persona=persona,
            )
            return ChatResponse(
                answer=result.answer,
                retrieved_chunks=result.retrieved_chunks,
                sources=result.sources,
                processing_time=round(result.processing_time, 4),
                tool_used=result.tool_used,
                steps_taken=result.steps_taken,
                answer_source=result.answer_source,
                hallucination_detected=result.hallucination_detected,
                grounding_score=result.hallucination_score,
                follow_up_questions=result.follow_up_questions,
                session_id=session_id or "",
            )

        try:
            return asyncio.run(_run())
        except RuntimeError:
            # A running event loop (async test / async route) can't be
            # asyncio.run()'d over — drive the coroutine on the existing
            # loop instead.
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())

    def _inject_memory(
        self, history: list[dict[str, str]] | None, session_id: str | None
    ) -> list[dict[str, str]] | None:
        """Append the session's remembered facts to the history passed to
        the router and generation prompts, as a synthetic system turn.

        No-op (returns history unchanged) when agent memory is disabled, no
        session is in play, or nothing is remembered yet — the injected
        block is the durable fact layer, while the live conversation turns
        already flow through history as normal user/assistant turns. A
        memory failure degrades to the un-augmented history, never raises.
        """
        if self._agent_memory is None or not settings.agent_memory_enabled or not session_id:
            return history
        try:
            block = self._agent_memory.build_context(session_id)
        except Exception as exc:
            logger.warning(
                "agent_memory_context_failed",
                extra={"extra_fields": {"session_id": session_id, "error": str(exc)}},
            )
            return history
        if not block:
            return history
        # Drop any earlier synthetic memory turn so a re-route (or a
        # retried request) doesn't stack stale fact blocks, then append the
        # fresh one last — trailing turns survive the history cap slicing.
        turns = [t for t in (history or []) if t.get("role") != "system"]
        return turns + [{"role": "system", "content": block}]

    def _extract_facts(self, query: str, answer: str) -> list[tuple[str, str, str]]:
        """Extract durable factual claims from a Q&A exchange via one
        JSON-mode LLM call. Returns a list of (key, value, confidence)
        tuples; [] on any parse failure or when the model found nothing
        worth remembering. Never raises — callers wrap this in _remember's
        exception guard."""
        prompt = (
            "Extract the durable factual claims from this question-answer "
            "exchange — the concrete facts a later question might want to "
            'refer back to (e.g. "project deadline", "team size"). Exclude '
            "conversational filler, opinions, and transient remarks.\n"
            "Return ONLY a single JSON object, no prose, matching exactly "
            '{"facts": [{"key": "<short lowercase label>", "value": "<the '
            'claim>", "confidence": "high" | "medium" | "low"}]}. '
            'Return {"facts": []} if there are no durable facts.\n\n'
            f"Question: {query}\nAnswer: {answer}"
        )
        raw = self._llm_client.generate(prompt).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        import json

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
        facts = payload.get("facts") if isinstance(payload, dict) else None
        if not isinstance(facts, list):
            return []
        out: list[tuple[str, str, str]] = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip().lower()
            value = str(item.get("value", "")).strip()
            if not key or not value:
                continue
            confidence = str(item.get("confidence", "medium")).strip().lower()
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            out.append((key, value, confidence))
        return out

    def _remember(
        self,
        session_id: str | None,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        query_type: str,
    ) -> None:
        """Record this exchange into agent memory: the two turns always,
        and — when fact extraction is enabled — durable key/value facts
        extracted from the answer. Small talk is skipped for fact
        extraction (nothing worth remembering). Best-effort and fully
        swallowed on failure — memory is a quality enhancement, never a new
        failure mode."""
        if self._agent_memory is None or not settings.agent_memory_enabled or not session_id:
            return
        try:
            self._agent_memory.add_turn(session_id, "user", query)
            self._agent_memory.add_turn(session_id, "assistant", answer)
        except Exception as exc:
            logger.warning(
                "agent_memory_turn_failed",
                extra={"extra_fields": {"session_id": session_id, "error": str(exc)}},
            )
            return

        if not settings.agent_memory_fact_extraction_enabled or query_type == "conversational":
            return
        try:
            facts = self._extract_facts(query, answer)
        except Exception as exc:
            logger.warning(
                "agent_memory_extraction_failed",
                extra={"extra_fields": {"session_id": session_id, "error": str(exc)}},
            )
            return
        if not facts:
            return
        source_chunk_ids = [chunk.chunk_id for chunk in chunks]
        for key, value, confidence in facts:
            self._agent_memory.upsert_fact(
                session_id, key, value, confidence=confidence, source_chunk_ids=source_chunk_ids
            )
        logger.info(
            "agent_memory_facts_stored",
            extra={"extra_fields": {"session_id": session_id, "fact_count": len(facts)}},
        )

    def _route(self, query: str, history: list[dict[str, str]] | None = None) -> PlanDecision:
        """Decide the query's action: the keyword planner, upgraded by the
        LLM router agent when Settings.agent_routing_enabled.

        The router can return "research" — the one action the regex
        planner can't express — and degrades to the planner's decision on
        any failure, so routing is additive, never a new failure mode.
        """
        injection_categories = detect_possible_injection(query)
        if injection_categories:
            logger.warning(
                "possible_injection_detected",
                extra={"extra_fields": {"source": "query", "categories": injection_categories}},
            )

        plan = self._plan(query, history)
        if not settings.agent_routing_enabled:
            return plan
        routed = self._router_agent.decide(query, history)
        crop = getattr(routed, "crop", None) or plan.crop
        collection = getattr(routed, "collection", None) or plan.collection or crop
        if routed.action != plan.action or routed.document_id != plan.document_id:
            logger.info(
                "router_decision",
                extra={
                    "extra_fields": {
                        "planner_action": plan.action,
                        "routed_action": routed.action,
                        "document_id": routed.document_id,
                        "crop": crop,
                        "collection": collection,
                        "query_length": len(query),
                    }
                },
            )
        return PlanDecision(
            action=routed.action,
            document_id=routed.document_id,
            crop=crop,
            collection=collection,
        )

    def _research_handoff(
        self, query: str, confirm_web_search: bool, *, reason: str
    ) -> ResearchFindings:
        """Hand the query to the Research agent (the handoff event is what
        the checklist's Agent Handoff Accuracy is computed from). reason is
        the trigger — "planned" (router chose research) or "weak_grade"
        (retrieval graded weak/insufficient)."""
        log_agent_handoff(
            "router" if reason == "planned" else "retrieval_grader",
            "research",
            query,
            reason=reason,
        )
        return self._research_agent.run(query, confirm_web_search=confirm_web_search)

    def _research_steps(self, findings: ResearchFindings) -> int:
        """How many of ChatResponse.steps_taken a research pass accounts
        for: the handoff plus one per sub-step the agent actually took."""
        return 1 + len(findings.steps)

    def _augment_weak_retrieval(
        self,
        query: str,
        retrieval_query: str,
        chunks: list[RetrievedChunk],
        grade: str,
        plan_action: str,
        tenant_id: int | None,
        confirm_web_search: bool,
    ) -> RetrievalAugmentation:
        """Escalate a weak/insufficient retrieval grade through, in
        precedence order: vision-grounded QA -> local document research
        agent -> research agent (web) -> plain web search fallback.

        Extracted verbatim from handle_query's inline block (each check's
        settings flag, ordering, and early-return condition is unchanged)
        so BOTH handle_query and the graph's context_augmentation_node call
        this one implementation — no duplicated escalation logic between
        the two execution paths, per the "avoid duplicated logic between
        streaming and non-streaming" design constraint.

        A direct hit (vision QA / local research / research agent) sets
        `final_answer` (and, for local research, `final_chunks`) — callers
        should treat that as an immediate response, skipping generation
        entirely, exactly as handle_query's early `return self._respond(...)`
        did before this extraction. No hit at all still returns
        `web_results`/`web_search_attempted` for the generation step that
        follows.
        """
        result = RetrievalAugmentation()

        if settings.vision_qa_enabled and grade != "good":
            vision_answer = try_vision_qa(query, chunks, self._llm_client)
            if vision_answer:
                result.final_answer = vision_answer
                result.tool_used = "vision_qa"
                result.extra_steps = 1
                return result

        if settings.local_research_agent_enabled and grade != "good":
            local_findings = self._local_research_agent.run(retrieval_query, tenant_id=tenant_id)
            if local_findings.answer:
                result.final_answer = local_findings.answer
                result.final_chunks = local_findings.chunks
                result.tool_used = "local_research"
                result.extra_steps = 1
                return result

        research_attempted = False
        if grade != "good" or plan_action == "research":
            if settings.research_agent_enabled and settings.web_search_enabled:
                research_attempted = True
                findings = self._research_handoff(
                    query,
                    confirm_web_search,
                    reason="planned" if plan_action == "research" else "weak_grade",
                )
                if findings.answer:
                    result.final_answer = findings.answer
                    result.tool_used = "research_agent"
                    result.extra_steps = self._research_steps(findings)
                    result.web_results = findings.results
                    return result
            if settings.web_search_enabled and not research_attempted:
                result.web_results = self._search_web(query, confirm_web_search=confirm_web_search)
                result.web_search_attempted = True

        return result

    def _generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None,
        extra_instruction: str | None = None,
        web_results: list[WebSearchResult] | None = None,
        persona: str | None = None,
        language: str | None = None,
    ) -> str:
        prompt = build_prompt(
            query,
            chunks,
            history=history,
            extra_instruction=extra_instruction,
            web_results=web_results,
            persona=persona,
            language=language,
        )
        _capture_prompt(prompt, variant="reflection" if extra_instruction else "standard")
        logger.info(
            "generation_requested",
            extra={
                "extra_fields": {
                    "prompt_version": PROMPT_VERSION,
                    "chunk_count": len(chunks),
                    "web_result_count": len(web_results) if web_results else 0,
                    "is_reflection": extra_instruction is not None,
                    "language": language or "en",
                }
            },
        )
        return strip_sources_section(self._llm_client.generate(prompt))

    def _generate_streamed(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None,
        extra_instruction: str | None = None,
        web_results: list[WebSearchResult] | None = None,
        persona: str | None = None,
        language: str | None = None,
    ) -> Iterator[tuple[bool, str]]:
        """Streamed counterpart to _generate: same prompt-building and
        logging, but yields (False, piece) for each safe-to-show chunk of
        raw model output as it arrives, then exactly one final
        (True, final_answer) once the stream ends — final_answer is
        strip_sources_section()-cleaned the same way _generate()'s return
        value is, computed from the complete raw text so it's identical
        regardless of how the provider happened to chunk it.
        """
        prompt = build_prompt(
            query,
            chunks,
            history=history,
            extra_instruction=extra_instruction,
            web_results=web_results,
            persona=persona,
            language=language,
        )
        _capture_prompt(prompt, variant="streamed")
        logger.info(
            "generation_requested",
            extra={
                "extra_fields": {
                    "prompt_version": PROMPT_VERSION,
                    "chunk_count": len(chunks),
                    "web_result_count": len(web_results) if web_results else 0,
                    "is_reflection": extra_instruction is not None,
                    "streamed": True,
                    "language": language or "en",
                }
            },
        )

        raw_parts: list[str] = []

        def _raw_stream() -> Iterator[str]:
            for piece in self._llm_client.generate_stream(prompt):
                raw_parts.append(piece)
                yield piece

        for filtered_piece in _stream_filtering_sources(_raw_stream()):
            yield (False, filtered_piece)

        yield (True, strip_sources_section("".join(raw_parts)))

    def _generate_structured(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None,
        web_results: list[WebSearchResult] | None = None,
        persona: str | None = None,
        language: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Structured-output counterpart to _generate: same context assembly
        via build_structured_prompt, but the provider is asked for a JSON
        object (response_mime_type / response_format) which is parsed and
        validated against StructuredAnswer. On any parse failure the request
        degrades to the plain free-text path (parse_structured_answer never
        raises) — structured output is a win-when-it-works enhancement,
        never a new failure mode.

        Returns (answer_text, structured_payload). structured_payload is the
        validated {"answer": ..., "sources": [...]} dict when the provider's
        output actually parsed and validated against StructuredAnswer, or
        None when the request degraded to the free-text fallback — callers
        must not treat a fallback answer as if it were a successful
        structured response.
        """
        prompt = build_structured_prompt(
            query, chunks, history=history, web_results=web_results, persona=persona, language=language
        )
        _capture_prompt(prompt, variant="structured")
        logger.info(
            "generation_requested",
            extra={
                "extra_fields": {
                    "prompt_version": PROMPT_VERSION,
                    "chunk_count": len(chunks),
                    "web_result_count": len(web_results) if web_results else 0,
                    "structured": True,
                    "language": language or "en",
                }
            },
        )
        raw = self._llm_client.generate_structured(prompt)
        structured = parse_structured_answer(raw)
        if structured is None:
            logger.info(
                "structured_output_fallback",
                extra={"extra_fields": {"query_length": len(query), "chunk_count": len(chunks)}},
            )
            fallback_answer = self._generate(
                query, chunks, history, web_results=web_results, persona=persona, language=language
            )
            return fallback_answer, None
        logger.info(
            "structured_output_success",
            extra={
                "extra_fields": {
                    "query_length": len(query),
                    "source_count": len(structured.sources),
                }
            },
        )
        return structured.answer, {"answer": structured.answer, "sources": structured.sources}

    def _grade_retrieval(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Cheap heuristic grade of retrieval quality — no LLM call, a
        function of chunk count and top score alone. query is accepted for
        a future semantic grader but isn't used by this heuristic today
        (same pattern as _plan's history parameter).

        - "insufficient": nothing survived retrieval_service's min_score
          floor.
        - "weak": chunks survived that floor, but the top score is still
          below Settings.retrieval_grade_threshold — technically on-topic,
          not confidently so.
        - "good": top score clears the threshold.

        A "weak"/"insufficient" grade is what makes the web search fallback
        eligible to fire (see handle_query), if Settings.web_search_enabled.
        """
        if not chunks:
            grade, top_score = "insufficient", None
        else:
            top_score = max(chunk.score for chunk in chunks)
            rerank_scores = [
                chunk.metadata["rerank_score"]
                for chunk in chunks
                if isinstance(getattr(chunk, "metadata", None), dict)
                and "rerank_score" in chunk.metadata
            ]
            if rerank_scores:
                top_rerank = max(rerank_scores)
                grade = (
                    "good"
                    if (
                        top_rerank >= settings.retrieval_grade_threshold
                        or top_score >= settings.retrieval_grade_threshold
                    )
                    else "weak"
                )
            else:
                effective_thresh = (
                    (settings.retrieval_grade_threshold / (getattr(settings, "hybrid_rrf_k", 60) + 1))
                    if settings.hybrid_search_enabled and top_score < 0.1
                    else settings.retrieval_grade_threshold
                )
                grade = "good" if top_score >= effective_thresh else "weak"

        logger.info(
            "retrieval_graded",
            extra={
                "extra_fields": {"grade": grade, "top_score": top_score, "chunk_count": len(chunks)}
            },
        )
        get_metrics().record_retrieval_grade(grade)
        return grade

    def _contextualize_query(self, query: str, history: list[dict[str, str]] | None) -> str:
        """Rewrite a follow-up question into a standalone one, using
        conversation history, before it's used for retrieval. Only called
        when history is non-empty. Degrades to the raw query on any LLM
        failure — this is a retrieval-quality enhancement, never a
        dependency the request can fail on.
        """
        if not history:
            return query
        prompt = (
            "Conversation history:\n"
            + "\n".join(
                f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history
            )
            + f"\n\nFollow-up question: {query}\n\n"
            "Rewrite the follow-up question as a standalone question that "
            "makes sense without the conversation history. Return ONLY the "
            "rewritten question, nothing else."
        )
        try:
            rewritten = self._llm_client.generate(prompt).strip()
            return rewritten if rewritten else query
        except Exception:
            return query

    def _verify_citations(self, answer: str, chunks: list[RetrievedChunk]) -> bool:
        """True if every [N] citation in `answer` is actually supported by
        its cited chunk's text. True (pass) if there are no citations to
        check, or on any LLM/parse failure — this is a stricter check
        layered on top of _is_ungrounded, never a stricter gate that can
        make an otherwise-fine answer fail closed.
        """
        citation_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        if not citation_numbers:
            return True
        chunk_by_number = {i + 1: chunk for i, chunk in enumerate(chunks)}
        cited_pairs = [
            (n, chunk_by_number[n].text) for n in citation_numbers if n in chunk_by_number
        ]
        if not cited_pairs:
            return True
        prompt = (
            "Answer:\n"
            + answer
            + "\n\n"
            + "\n\n".join(f"Excerpt [{n}]:\n{text}" for n, text in cited_pairs)
            + "\n\nFor each excerpt number above, does the answer's claim "
            "attributed to it actually match what that excerpt says? "
            'Respond with ONLY a JSON object like {"1": true, "2": false}, '
            "one entry per excerpt number shown."
        )
        try:
            import json

            raw = self._llm_client.generate(prompt).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
            return all(result.get(str(n), True) for n, _ in cited_pairs)
        except Exception:
            return True

    def _maybe_ask_clarifying_question(self, query: str, answer: str, grade: str) -> tuple[str, bool]:
        """Agent 1.4 — Ask-instead-of-guess: when retrieval graded
        "insufficient" and the corrective loop still couldn't produce a
        grounded answer (sitting on the literal fallback line), ask one
        short clarifying question instead of shipping the canned "couldn't
        find that" reply, when Settings.clarifying_question_enabled.
        Returns (answer, is_clarifying_question) — degrades to
        (answer, False) unchanged on any failure or when the conditions
        don't apply, exactly matching handle_query's original inline
        behavior (this is a straight extraction, not new logic).
        """
        if not (
            settings.clarifying_question_enabled
            and grade == "insufficient"
            and answer.strip() == FALLBACK_REPLY
        ):
            return answer, False
        try:
            clarifying_prompt = (
                f"The user asked: {query}\n\n"
                "No relevant information was found in their documents, and "
                "the question may be ambiguous or missing detail. Suggest "
                "ONE short clarifying question to ask them. Return ONLY the "
                "question, nothing else."
            )
            clarification = self._llm_client.generate(clarifying_prompt).strip()
            if clarification:
                return clarification, True
        except Exception:
            pass  # falls through, answer stays FALLBACK_REPLY exactly as today
        return answer, False

    def _suggest_follow_ups(self, query: str, answer: str) -> list[str]:
        """Suggest up to 3 short follow-up questions. Degrades to an
        empty list on any LLM/parse failure — never blocks or fails the
        main answer."""
        prompt = (
            f"Question: {query}\nAnswer: {answer}\n\n"
            "Suggest up to 3 short, natural follow-up questions the user "
            "might ask next. Return ONLY a JSON array of strings, e.g. "
            '["question one?", "question two?"]. Return an empty array [] '
            "if you can't think of good ones."
        )
        try:
            import json

            raw = self._llm_client.generate(prompt).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
            return [str(q) for q in result][:3] if isinstance(result, list) else []
        except Exception:
            return []

    def _search_web(self, query: str, confirm_web_search: bool = False) -> list[WebSearchResult]:
        """Best-effort web search. A failure here degrades to an empty
        result list rather than failing the whole chat request — web
        search is an enhancement to the corrective loop, not a dependency
        the request can't survive without.

        The human-approval gate: when Settings.web_search_requires_approval
        is on, web search is skipped unless the client explicitly sent
        confirm_web_search=true (see ChatRequest). A skipped-but-requested
        search is logged distinctly from a failed one, so the two are
        distinguishable in monitoring.
        """
        if settings.web_search_requires_approval and not confirm_web_search:
            logger.info(
                "web_search_skipped_pending_approval",
                extra={"extra_fields": {"query_length": len(query)}},
            )
            # Feature #5 — surface the skipped action as a pending approval
            # so an operator can grant/reject it via the approval queue
            # (GET /approvals, POST /approvals/{id}/resolve) instead of it
            # vanishing into a log line.
            from app.core.request_context import get_client_name
            from app.services.approval_service import get_approval_store

            approval = get_approval_store().register(
                action="web_search",
                requested_by=get_client_name(),
                payload={"query": query},
            )
            logger.info(
                "approval_created_for_skipped_web_search",
                extra={
                    "extra_fields": {
                        "approval_id": approval.approval_id,
                        "query_length": len(query),
                    }
                },
            )
            return []
        if not web_search_ready():
            # Feature #7 — force-disable without key: a provider that
            # isn't usable is treated as disabled (degrade to []) rather
            # than as a real search that merely found nothing. The
            # distinct "web_search_unavailable" event is logged inside
            # web_search_ready(). Callers should retry after fixing config.
            return []
        try:
            # search_web() is itself typed -> list[WebSearchResult], but the
            # @track_tool decorator wrapping it currently erases that to Any
            # (its Callable[..., Any] signature doesn't preserve the wrapped
            # function's exact return type) — annotate explicitly here
            # rather than weaken this function's own return type.
            results: list[WebSearchResult] = search_web(query)
            if results:
                get_metrics().record_web_search_fallback(stage="retrieval")
            return results
        except WebSearchError as exc:
            logger.warning("web_search_failed", extra={"extra_fields": {"error": str(exc)}})
            return []

    def _is_ungrounded(
        self, answer: str, chunks: list[RetrievedChunk], web_results: list[WebSearchResult]
    ) -> bool:
        """True if the answer looks like it ignored context that was
        actually available: empty, or exactly the fixed fallback line,
        while there was at least one chunk or web result it could have
        drawn from. If there's no context at all, the fallback line is the
        correct, expected answer — not a failure to correct.

        Also true for GENERATION_ERROR_REPLY (a distinct sentinel from
        FALLBACK_REPLY as of the Phase 3 faithfulness-regression fix —
        see generator_node): a transient provider failure deserves the
        same corrective-loop retry a "not found" answer gets, since a
        second attempt has a real chance of succeeding where the first
        hit a timeout/rate-limit."""
        if not chunks and not web_results:
            return False
        stripped = answer.strip()
        return not stripped or stripped in (FALLBACK_REPLY, GENERATION_ERROR_REPLY)

    def _correct(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        answer: str,
        history: list[dict[str, str]] | None,
        web_results: list[WebSearchResult],
        web_search_attempted: bool,
        llm_calls: int,
        steps_taken: int,
        confirm_web_search: bool = False,
        persona: str | None = None,
        language: str = "en",
    ) -> tuple[str, int, int, list[WebSearchResult], bool]:
        """Generalizes the old single-shot _reflect into the corrective
        RAG loop's regeneration stage.

        If the answer fails the groundedness check (empty, or the fixed
        fallback line) despite having context, regenerate once. If it's
        still ungrounded and web search is enabled but wasn't already
        pulled in for this request, fetch web results and make one final
        attempt with them added to the context. Capped at _MAX_LLM_CALLS
        total generate() calls (checked before each additional call) —
        hitting the cap logs "loop_capped" and returns whatever answer
        exists rather than looping further.

        Returns (answer, llm_calls, steps_taken, web_results,
        web_search_attempted) — web_results/web_search_attempted come back
        out since this method may fetch them partway through, and the
        caller needs the final values to build sources/answer_source.
        """
        ungrounded = self._is_ungrounded(answer, chunks, web_results) or (
            settings.citation_verification_enabled and not self._verify_citations(answer, chunks)
        )
        if not ungrounded:
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        if llm_calls >= _MAX_LLM_CALLS:
            get_metrics().inc_counter("loop_capped_total", {"stage": "reflection"})
            logger.warning(
                "loop_capped",
                extra={"extra_fields": {"llm_calls": llm_calls, "stage": "reflection"}},
            )
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        logger.info(
            "reflection_triggered",
            extra={"extra_fields": {"query_length": len(query), "chunk_count": len(chunks)}},
        )
        answer = self._generate(
            query,
            chunks,
            history,
            extra_instruction=REFLECTION_INSTRUCTION,
            web_results=web_results,
            persona=persona,
            language=language,
        )
        llm_calls += 1
        steps_taken += 1  # regeneration

        if not self._is_ungrounded(answer, chunks, web_results):
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        if web_search_attempted or not settings.web_search_enabled:
            # Either web context was already in play for this request and
            # didn't help, or the fallback isn't enabled — no further
            # corrective option, return the best answer we have.
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        if llm_calls >= _MAX_LLM_CALLS:
            get_metrics().inc_counter("loop_capped_total", {"stage": "web_fallback"})
            logger.warning(
                "loop_capped",
                extra={"extra_fields": {"llm_calls": llm_calls, "stage": "web_fallback"}},
            )
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        web_results = self._search_web(query, confirm_web_search=confirm_web_search)
        web_search_attempted = True
        steps_taken += 1  # web search

        if not web_results:
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        logger.info("web_fallback_triggered", extra={"extra_fields": {"query_length": len(query)}})
        answer = self._generate(
            query,
            chunks,
            history,
            extra_instruction=REFLECTION_INSTRUCTION,
            web_results=web_results,
            persona=persona,
            language=language,
        )
        llm_calls += 1
        steps_taken += 1  # regeneration with web context

        return answer, llm_calls, steps_taken, web_results, web_search_attempted

    def _correct_streamed(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        answer: str,
        history: list[dict[str, str]] | None,
        web_results: list[WebSearchResult],
        web_search_attempted: bool,
        llm_calls: int,
        steps_taken: int,
        confirm_web_search: bool = False,
        persona: str | None = None,
        language: str | None = None,
    ) -> Generator[dict[str, Any], None, tuple[str, int, int, list[WebSearchResult], bool]]:
        """Streamed counterpart to _correct — mirrors its exact branches,
        conditions, and log lines (the two must be kept in sync; a
        divergence here is a real behavioral difference between /chat and
        /chat/stream, not just a cosmetic one) but yields a "reflecting"
        trace event plus streamed answer_chunk pieces for each
        regeneration instead of blocking on generate(). Ends with `return
        (answer, llm_calls, steps_taken, web_results, web_search_attempted)`
        — the value of `yield from self._correct_streamed(...)` in the
        caller, per normal generator-return semantics.

        A "reflecting" trace event means the answer streamed so far
        belongs to a discarded attempt: a fresh one is about to start from
        scratch, not continue it. See README's /chat/stream section for
        the exact client-side contract.
        """
        ungrounded = self._is_ungrounded(answer, chunks, web_results) or (
            settings.citation_verification_enabled and not self._verify_citations(answer, chunks)
        )
        if not ungrounded:
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        if llm_calls >= _MAX_LLM_CALLS:
            get_metrics().inc_counter("loop_capped_total", {"stage": "reflection"})
            logger.warning(
                "loop_capped",
                extra={"extra_fields": {"llm_calls": llm_calls, "stage": "reflection"}},
            )
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        logger.info(
            "reflection_triggered",
            extra={"extra_fields": {"query_length": len(query), "chunk_count": len(chunks)}},
        )
        yield _trace_event("reflecting", {"reason": "ungrounded_answer"})
        answer = ""
        for is_final, value in self._generate_streamed(
            query,
            chunks,
            history,
            extra_instruction=REFLECTION_INSTRUCTION,
            web_results=web_results,
            persona=persona,
            language=language,
        ):
            if is_final:
                answer = value
            else:
                yield {"type": "answer_chunk", "payload": {"token": value}, "text": value}
        llm_calls += 1
        steps_taken += 1  # regeneration

        if not self._is_ungrounded(answer, chunks, web_results):
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        if web_search_attempted or not settings.web_search_enabled:
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        if llm_calls >= _MAX_LLM_CALLS:
            get_metrics().inc_counter("loop_capped_total", {"stage": "web_fallback"})
            logger.warning(
                "loop_capped",
                extra={"extra_fields": {"llm_calls": llm_calls, "stage": "web_fallback"}},
            )
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        web_results = self._search_web(query, confirm_web_search=confirm_web_search)
        web_search_attempted = True
        steps_taken += 1  # web search
        yield _trace_event("web_search", {"result_count": len(web_results)})

        if not web_results:
            return answer, llm_calls, steps_taken, web_results, web_search_attempted

        logger.info("web_fallback_triggered", extra={"extra_fields": {"query_length": len(query)}})
        yield _trace_event("reflecting", {"reason": "web_fallback"})
        answer = ""
        for is_final, value in self._generate_streamed(
            query,
            chunks,
            history,
            extra_instruction=REFLECTION_INSTRUCTION,
            web_results=web_results,
            persona=persona,
            language=language,
        ):
            if is_final:
                answer = value
            else:
                yield {"type": "answer_chunk", "payload": {"token": value}, "text": value}
        llm_calls += 1
        steps_taken += 1  # regeneration with web context

        return answer, llm_calls, steps_taken, web_results, web_search_attempted

    def handle_query(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        confirm_web_search: bool = False,
        structured_response: bool = False,
        tenant_id: int | None = None,
        persona: str | None = None,
        document_ids: list[str] | None = None,
        language: str = "en",
        approval_id: str | None = None,
    ) -> ChatResponse:
        start = time.perf_counter()
        steps_taken = 1  # planning
        reset_usage()  # per-request LLM token/cost rollup

        # Agent memory: append the session's remembered facts to the
        # history before routing/generation, so the router and the model
        # can draw on what was established earlier in the conversation.
        history = self._inject_memory(history, session_id)

        plan = self._route(query, history)
        logger.info(
            "plan_decided",
            extra={"extra_fields": {"action": plan.action, "query_length": len(query)}},
        )

        if settings.agent_executor_enabled and plan.action in ("retrieve", "research"):
            # AgentExecutor orchestration (Feature: planner → ReAct
            # executor over the dynamic tool registry). Only for the
            # retrieval/research paths — small talk and summarization stay
            # on the cheap inline paths (conversational needs no tools at
            # all, and summarize is a single tool call that the executor
            # would add planning overhead to for no benefit).
            return self._handle_via_executor(
                query,
                history=history,
                session_id=session_id,
                tenant_id=tenant_id,
                top_k=top_k,
                min_score=min_score,
                confirm_web_search=confirm_web_search,
                persona=persona,
            )

        if plan.action == "conversational":
            # plan.action == "conversational" is only ever set (in _plan/
            # _route) when this same match already succeeded, so this is
            # never actually None — the FALLBACK_REPLY default is just a
            # typed, non-crashing guard against the two ever desyncing,
            # not expected behavior.
            return self._respond(
                answer=_match_conversational_reply(query) or FALLBACK_REPLY,
                retrieved_chunks=[],
                query=query,
                query_type="conversational",
                tool_used="none",
                steps_taken=steps_taken,
                start=start,
                session_id=session_id,
            )

        if plan.action == "summarize":
            steps_taken += 1  # fetch document chunks
            try:
                summary, chunks = summarize_document(
                    plan.document_id, self._vector_store, self._llm_client, tenant_id=tenant_id
                )
            except AppError:
                raise
            except Exception as exc:
                raise ChatServiceError(f"Unexpected error while handling chat query: {exc}") from exc
            steps_taken += 1  # generation
            return self._respond(
                answer=summary,
                retrieved_chunks=chunks,
                query=query,
                query_type="summarize",
                tool_used="summarization",
                steps_taken=steps_taken,
                start=start,
                session_id=session_id,
            )

        # plan.action in ("retrieve", "research") — run the explicit chat
        # workflow graph (backend/app/services/agent_graph/graph.py):
        # cache -> retrieval -> grading -> weak-retrieval augmentation
        # (vision QA / local research / research agent / web search) ->
        # generation -> corrective reflection -> validation -> finalize.
        # The graph's nodes delegate to this same ChatService's methods
        # (_get_cached_response, _grade_retrieval, _generate, _correct,
        # _respond, _cache_response, ...) — nothing below is reimplemented,
        # only re-sequenced through explicit, traced nodes.
        try:
            return self._run_chat_graph(
                query=query,
                plan=plan,
                history=history,
                session_id=session_id,
                confirm_web_search=confirm_web_search,
                structured_response=structured_response,
                tenant_id=tenant_id,
                persona=persona,
                document_ids=document_ids,
                language=language,
                top_k=top_k,
                min_score=min_score,
                perf_start=start,
                approval_id=approval_id,
            )
        except AppError:
            # Already a well-formed domain exception from retrieval, prompt
            # building, summarization, or the LLM client — propagate it
            # unchanged.
            raise
        except Exception as exc:
            raise ChatServiceError(f"Unexpected error while handling chat query: {exc}") from exc

    def _run_chat_graph(
        self,
        *,
        query: str,
        plan: PlanDecision,
        history: list[dict[str, str]] | None,
        session_id: str | None,
        confirm_web_search: bool,
        structured_response: bool,
        tenant_id: int | None,
        persona: str | None,
        document_ids: list[str] | None,
        language: str,
        top_k: int | None,
        min_score: float | None,
        perf_start: float,
        approval_id: str | None = None,
    ) -> ChatResponse:
        """Builds the initial AgentState from an already-decided plan (this
        method never re-plans — `plan` is handle_query's own `_route()`
        result, including any RouterAgent upgrade) and runs it through
        `build_chat_graph()`. Deferred imports: `agent_graph` imports this
        module at top level (its nodes delegate back into ChatService), so
        importing it back here at module level would be circular — the
        same pattern `_agent_executor_instance` already uses for
        AgentExecutor/PlanningAgent.
        """
        import asyncio

        from app.services.agent_graph.graph import build_chat_graph
        from app.services.agent_graph.nodes import GraphContext
        from app.services.agent_graph.state import AgentState

        initial_state = AgentState(
            query=query,
            history=history,
            session_id=session_id,
            tenant_id=tenant_id,
            confirm_web_search=confirm_web_search,
            persona=persona,
            document_ids=document_ids,
            document_id=plan.document_id,
            retrieval_top_k=top_k,
            retrieval_min_score=min_score,
            plan={
                "action": plan.action,
                "document_id": plan.document_id,
                "crop": plan.crop,
                "collection": plan.collection,
            },
            planned_action=plan.action,
            intent=plan.action,
            metadata={"structured_response": structured_response, "language": language},
            perf_start=perf_start,
            approval_payload_reference=approval_id,
        )
        context = GraphContext(
            chat_service=self,
            vector_store=self._vector_store,
            image_vector_store=self._image_vector_store,
        )
        graph = build_chat_graph()
        result_state = asyncio.run(graph.run(initial_state, context))
        if result_state.final_response is not None:
            return result_state.final_response
        # Defensive fallback — every real path through the graph sets
        # final_response (cache hit, conversational, summarize, or
        # finalizer_node); this only triggers if a node failed before
        # reaching any of them (e.g. no chat_service, which can't happen
        # here since `self` is always provided).
        raise ChatServiceError(
            result_state.error_message or "Chat workflow graph did not produce a response."
        )

    def stream_query(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        confirm_web_search: bool = False,
        structured_response: bool = False,
        tenant_id: int | None = None,
        persona: str | None = None,
        document_ids: list[str] | None = None,
        language: str = "en",
    ) -> Iterator[dict[str, Any]]:
        """Streamed counterpart to handle_query, for POST /chat/stream.

        Same planner and pipeline (conversational / summarize / retrieve
        with its corrective loop) as handle_query, but yields progress as
        it happens instead of returning one ChatResponse at the end. Each
        yielded dict is one of:

        - {"type": "trace", "stage": ..., "detail": {...}} before/after a
          pipeline step. A "reflecting" stage means any answer_chunk text
          streamed before it belongs to a discarded attempt — a fresh one
          is starting over, not continuing it.
        - {"type": "answer_chunk", "text": "..."} — a piece of generated
          text, in order, already filtered so a "Sources:" heading never
          reaches the client (see _stream_filtering_sources).
        - {"type": "error", "detail": {"error_type", "message",
          "status_code"}} — emitted in place of "done" if the pipeline
          fails. SSE responses commit to a 200 status as soon as the
          stream starts, so a mid-stream AppError can't become an HTTP
          error status the way it would on POST /chat; this is how it's
          surfaced instead.
        - exactly one final {"type": "done", "payload": ChatResponse} (on
          success) — the same response shape POST /chat returns.

        routes/query.py's /chat/stream serializes these directly as SSE
        `data:` lines.
        """
        start = time.perf_counter()
        steps_taken = 1  # planning
        reset_usage()  # per-request LLM token/cost rollup

        history = self._inject_memory(history, session_id)

        plan = self._route(query, history)
        logger.info(
            "plan_decided",
            extra={"extra_fields": {"action": plan.action, "query_length": len(query)}},
        )
        yield _trace_event("planning", {"action": plan.action})

        if plan.action == "conversational":
            # Same non-None invariant as handle_query's conversational
            # branch — see the comment there.
            answer = _match_conversational_reply(query) or FALLBACK_REPLY
            yield {"type": "answer_chunk", "text": answer}
            response = self._respond(
                answer=answer,
                retrieved_chunks=[],
                query=query,
                query_type="conversational",
                tool_used="none",
                steps_taken=steps_taken,
                start=start,
                session_id=session_id,
            )
            yield {"type": "done", "payload": response}
            return

        recent_history = history[-_MAX_HISTORY_TURNS:] if history else None

        try:
            if plan.action == "summarize":
                steps_taken += 1  # fetch document chunks
                yield _trace_event("retrieval", {"document_id": plan.document_id})
                summary, chunks = summarize_document(
                    plan.document_id, self._vector_store, self._llm_client, tenant_id=tenant_id
                )
                steps_taken += 1  # generation
                yield _trace_event("generating", {})
                # summarize_document() calls the LLM's blocking generate(),
                # not generate_stream() — summarization isn't part of this
                # task's streaming scope, so the whole summary is emitted
                # as one chunk rather than faked as a token stream.
                yield {"type": "answer_chunk", "text": summary}
                response = self._respond(
                    answer=summary,
                    retrieved_chunks=chunks,
                    query=query,
                    query_type="summarize",
                    tool_used="summarization",
                    steps_taken=steps_taken,
                    start=start,
                    session_id=session_id,
                )
                yield {"type": "done", "payload": response}
                return

            # plan.action == "retrieve" — the corrective RAG loop, streamed.
            # Cache check, retrieval, and grading below delegate to the
            # same node functions handle_query's graph uses (cache_lookup_
            # node / retrieval_node / retrieval_grader_node) — called
            # directly here rather than through the async graph engine,
            # since streaming needs to interleave SSE yields between them
            # and (for generation/reflection, below) needs token-level
            # granularity the engine's per-node streaming can't provide.
            # This still eliminates what would otherwise be duplicated
            # retrieve_kwargs-building/cache-shaping logic between the two
            # execution paths — only the escalation cascade right below
            # stays as its own inline implementation, to preserve its
            # existing fine-grained per-strategy SSE progress events
            # (local_research / research_handoff / research_<stage> /
            # web_search), which a single context_augmentation_node call
            # would otherwise collapse into one opaque completion event.
            from app.services.agent_graph.cache_node import cache_lookup_node
            from app.services.agent_graph.nodes import (
                GraphContext,
                retrieval_grader_node,
                retrieval_node,
            )
            from app.services.agent_graph.state import AgentState

            plan_crop = getattr(plan, "crop", None)
            plan_disease = getattr(plan, "disease", None)
            graph_context = GraphContext(
                chat_service=self, vector_store=self._vector_store, image_vector_store=self._image_vector_store
            )
            node_state = AgentState(
                query=query,
                history=history,
                session_id=session_id,
                tenant_id=tenant_id,
                document_ids=document_ids,
                document_id=plan.document_id,
                retrieval_top_k=top_k,
                retrieval_min_score=min_score,
                plan={
                    "action": plan.action,
                    "document_id": plan.document_id,
                    "crop": plan.crop,
                    "collection": plan.collection,
                },
                planned_action=plan.action,
            )

            node_state = cache_lookup_node(node_state, graph_context)
            if node_state.metadata.get("cache_hit"):
                logger.info(
                    "cache_hit",
                    extra={
                        "extra_fields": {
                            "query": query,
                            "crop": plan_crop,
                            "disease": plan_disease,
                            "session_id": session_id or "none",
                            "cached": True,
                        }
                    },
                )
                yield {
                    "type": "trace",
                    "event": "cache_hit",
                    "stage": "cache_hit",
                    "payload": {"cached": True},
                    "detail": {"cached": True},
                }
                yield {"type": "done", "payload": node_state.final_response}
                return

            steps_taken += 1  # retrieval
            node_state = retrieval_node(node_state, graph_context)
            chunks = node_state.retrieved_chunks
            retrieval_query = node_state.retrieval_query or query
            yield _trace_event("retrieval", {"chunk_count": len(chunks)})

            steps_taken += 1  # grading
            node_state = retrieval_grader_node(node_state, graph_context)
            grade = node_state.retrieval_grade
            yield _trace_event("grading", {"grade": grade})

            # See handle_query for why plan.action == "research" forces
            # this block regardless of grade.
            web_results: list[WebSearchResult] = []
            web_search_attempted = False
            research_attempted = False
            # Local Document Research Agent (Agent 2.3) — same reasoning and
            # ordering as handle_query: weak/insufficient retrieval gets a
            # plan-decompose-search-synthesize pass over the app's OWN
            # documents before the web research handoff.
            if settings.local_research_agent_enabled and grade != "good":
                yield _trace_event("local_research", {"query": retrieval_query})
                local_findings = self._local_research_agent.run(
                    retrieval_query, tenant_id=tenant_id
                )
                if local_findings.answer:
                    steps_taken += 1  # local research pass
                    yield {"type": "answer_chunk", "text": local_findings.answer}
                    response = self._respond(
                        answer=local_findings.answer,
                        retrieved_chunks=local_findings.chunks,
                        query=query,
                        query_type="document_query",
                        tool_used="local_research",
                        steps_taken=steps_taken,
                        start=start,
                        session_id=session_id,
                        retrieval_confidence=grade,
                    )
                    yield {"type": "done", "payload": response}
                    return
            if grade != "good" or plan.action == "research":
                if settings.research_agent_enabled and settings.web_search_enabled:
                    research_attempted = True
                    yield _trace_event(
                        "research_handoff",
                        {"reason": "planned" if plan.action == "research" else "weak_grade"},
                    )
                    findings = self._research_handoff(
                        query,
                        confirm_web_search,
                        reason="planned" if plan.action == "research" else "weak_grade",
                    )
                    for step in findings.steps:
                        yield _trace_event(f"research_{step['stage']}", step)
                    if findings.answer:
                        steps_taken += self._research_steps(findings)
                        yield {"type": "answer_chunk", "text": findings.answer}
                        response = self._respond(
                            answer=findings.answer,
                            retrieved_chunks=chunks,
                            query=query,
                            query_type="document_query",
                            tool_used="research_agent",
                            steps_taken=steps_taken,
                            start=start,
                            web_results=findings.results,
                            session_id=session_id,
                        )
                        yield {"type": "done", "payload": response}
                        return
                if settings.web_search_enabled and not research_attempted:
                    web_results = self._search_web(query, confirm_web_search=confirm_web_search)
                    web_search_attempted = True
                    steps_taken += 1  # web search
                    yield _trace_event("web_search", {"result_count": len(web_results)})

            yield _trace_event("generating", {})
            answer = ""
            for is_final, value in self._generate_streamed(
                query, chunks, recent_history, web_results=web_results, persona=persona, language=language
            ):
                if is_final:
                    answer = value
                else:
                    yield {"type": "answer_chunk", "payload": {"token": value}, "text": value}
            llm_calls = 1
            steps_taken += 1  # generation

            (
                answer,
                llm_calls,
                steps_taken,
                web_results,
                web_search_attempted,
            ) = yield from self._correct_streamed(
                query,
                chunks,
                answer,
                recent_history,
                web_results,
                web_search_attempted,
                llm_calls,
                steps_taken,
                confirm_web_search=confirm_web_search,
                persona=persona,
                language=language,
            )

            # Same Agent 1.4 ask-instead-of-guess + Agent 2.2 follow-up
            # logic as handle_query — both now call the same extracted
            # ChatService methods, so there's exactly one implementation of
            # each, not two. The chunks streamed above are provisional —
            # the done payload's answer is authoritative and the UI
            # renders it.
            answer, is_clarifying_question = self._maybe_ask_clarifying_question(query, answer, grade)

            follow_up_questions = []
            if settings.follow_up_questions_enabled:
                follow_up_questions = self._suggest_follow_ups(query, answer)
        except AppError as exc:
            logger.info(
                "chat_stream_error",
                extra={
                    "extra_fields": {
                        "error_type": type(exc).__name__,
                        "status_code": exc.status_code,
                    }
                },
            )
            yield {
                "type": "error",
                "detail": {
                    "error_type": type(exc).__name__,
                    "message": exc.detail,
                    "status_code": exc.status_code,
                },
            }
            return
        except Exception as exc:
            chat_error = ChatServiceError(f"Unexpected error while handling chat query: {exc}")
            logger.info(
                "chat_stream_error",
                extra={
                    "extra_fields": {
                        "error_type": type(chat_error).__name__,
                        "status_code": chat_error.status_code,
                    }
                },
            )
            yield {
                "type": "error",
                "detail": {
                    "error_type": type(chat_error).__name__,
                    "message": chat_error.detail,
                    "status_code": chat_error.status_code,
                },
            }
            return

        tool_used = "web_search" if web_results else "retrieval"
        response = self._respond(
            answer=answer,
            retrieved_chunks=chunks,
            query=query,
            query_type="document_query",
            tool_used=tool_used,
            steps_taken=steps_taken,
            start=start,
            web_results=web_results,
            session_id=session_id,
            retrieval_confidence=grade,
            is_clarifying_question=is_clarifying_question,
            follow_up_questions=follow_up_questions,
        )
        # Cache the final response (retrieve action only — a "research"
        if plan.action == "retrieve":
            self._cache_response(
                query=query,
                response=response,
                crop=getattr(plan, "crop", None),
                disease=getattr(plan, "disease", None),
                tenant_id=tenant_id,
                document_ids=document_ids,
            )
        yield {"type": "done", "payload": response}

    def handle_diagnose(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        query: str | None = None,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        confirm_web_search: bool = False,
        tenant_id: int | None = None,
        persona: str | None = "agronomist",
        engine: str = "hybrid",
        weather_risk: WeatherRiskResponse | None = None,
        language: str = "en",
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> ChatResponse:
        """Diagnose a plant photo via LeafSense or Gemini, then run the predicted
        disease through the same corrective RAG loop handle_query uses.

        `latitude`/`longitude` (Module 10 gap-closure -- real parallel
        execution): when given and `weather_risk` wasn't already
        pre-fetched by the caller, the microclimate lookup
        (`WeatherService.get_weather_risk`, a real Open-Meteo HTTP call)
        runs CONCURRENTLY with the vision classification step below via
        `run_concurrent_branches` -- the two are genuinely independent
        (weather depends only on lat/lon, vision only on the image
        bytes; neither needs the other's result until the prompt-
        building step much later) and previously ran sequentially
        (the caller awaited weather fully before this method even
        started). `weather_risk` stays accepted as-is for full backward
        compatibility with any caller that already has it precomputed.
        """
        start = time.perf_counter()
        steps_taken = 1  # planning

        plan = PlanDecision(action="diagnose")
        reset_usage()  # per-request LLM token/cost rollup
        logger.info(
            "plan_decided",
            extra={
                "extra_fields": {"action": plan.action, "query_length": len(query) if query else 0, "engine": engine}
            },
        )

        history = self._inject_memory(history, session_id)

        recent_history = history[-_MAX_HISTORY_TURNS:] if history else None

        try:
            steps_taken += 1  # vision inference
            # Delegates to the explicit vision_node (agent_graph/nodes.py),
            # which wraps diagnose_image + _build_diagnosis_query/_build_
            # diagnosis_info — the same functions this method used to call
            # inline. Image bytes travel via GraphContext.metadata, not
            # AgentState, so they're never deep-copied into the graph
            # engine's per-step snapshot history (see vision_node's
            # docstring).
            from app.services.agent_graph.engine import run_concurrent_branches
            from app.services.agent_graph.nodes import GraphContext, vision_node
            from app.services.agent_graph.state import AgentState

            vision_state: AgentState

            def _run_vision() -> AgentState:
                return vision_node(
                    AgentState(query=query or "", history=history),
                    GraphContext(
                        metadata={
                            "image_bytes": image_bytes,
                            "filename": filename,
                            "content_type": content_type,
                            "engine": engine,
                        }
                    ),
                )

            if weather_risk is None and latitude is not None and longitude is not None:
                # Real parallel execution (Module 10 gap-closure): vision
                # classification (blocking HTTP to LeafSense/Gemini) and
                # the weather lookup (async HTTP to Open-Meteo) are
                # genuinely independent -- run them concurrently instead
                # of sequentially, via the graph engine's reusable
                # concurrency primitive. asyncio.run() is safe here:
                # handle_diagnose always executes inside FastAPI's
                # run_in_threadpool, a fresh worker thread with no
                # existing event loop.
                async def _fetch_weather() -> WeatherRiskResponse:
                    from app.services.weather_service import WeatherService

                    return await WeatherService().get_weather_risk(lat=latitude, lon=longitude)

                branch_results = asyncio.run(
                    run_concurrent_branches(
                        {"vision": _run_vision, "weather": _fetch_weather},
                        request_id=session_id,
                    )
                )
                vision_branch = branch_results["vision"]
                if not vision_branch.success:
                    # Vision failing is a real, existing failure mode for this
                    # method (see the except AppError/Exception blocks below)
                    # -- re-raise the ORIGINAL exception object (not a new
                    # RuntimeError) so a VisionServiceError (an AppError
                    # subclass) still hits `except AppError: raise` exactly as
                    # it would have in the old sequential call, preserving
                    # its status code/taxonomy instead of being silently
                    # downgraded to a generic ChatServiceError.
                    raise vision_branch.exception  # noqa: RSE102 -- re-raising a captured exception object, not a bare `raise`
                vision_state = vision_branch.value

                weather_branch = branch_results["weather"]
                if weather_branch.success:
                    weather_risk = weather_branch.value
                else:
                    # Matches the route layer's existing behavior for a
                    # failed weather lookup: log and continue without it,
                    # never fail the whole diagnosis over an optional
                    # enrichment call.
                    logger.warning(
                        "Failed to fetch microclimate risk for (%s, %s): %s",
                        latitude,
                        longitude,
                        weather_branch.error,
                    )
            else:
                vision_state = _run_vision()

            diagnosis_info = vision_state.diagnosis
            diagnosis_query = vision_state.retrieval_query
            crop_context = vision_state.metadata.get("crop_context")
            disease_context = vision_state.metadata.get("disease_context")

            # Check semantic cache for instant sub-50ms diagnosis response
            cached = self._get_cached_response(
                query=diagnosis_query,
                crop=crop_context,
                disease=disease_context,
                tenant_id=tenant_id,
            )
            if cached is not None:
                logger.info(
                    "cache_hit",
                    extra={
                        "extra_fields": {
                            "query": diagnosis_query,
                            "crop": crop_context,
                            "disease": disease_context,
                            "session_id": session_id or "none",
                            "cached": True,
                        }
                    },
                )
                cached_dict = cached.model_dump()
                cached_dict["session_id"] = session_id or cached_dict.get("session_id", "")
                cached_dict["diagnosis"] = diagnosis_info.model_dump()
                if weather_risk is not None:
                    cached_dict["weather_risk"] = weather_risk.model_dump()
                if "metadata" not in cached_dict or not isinstance(cached_dict["metadata"], dict):
                    cached_dict["metadata"] = {}
                cached_dict["metadata"]["cached"] = True
                return ChatResponse.model_validate(cached_dict)

            steps_taken += 1  # retrieval
            chunks = retrieve(
                diagnosis_query,
                self._vector_store,
                tenant_id=tenant_id,
                image_vector_store=self._image_vector_store,
                collection=crop_context,
                rerank=True,
            )

            steps_taken += 1  # grading
            grade = self._grade_retrieval(diagnosis_query, chunks)

            web_results: list[WebSearchResult] = []
            web_search_attempted = False
            research_attempted = False
            if grade != "good":
                if settings.research_agent_enabled and settings.web_search_enabled:
                    research_attempted = True
                    findings = self._research_handoff(
                        diagnosis_query, confirm_web_search, reason="diagnose_weak_grade"
                    )
                    if findings.answer:
                        steps_taken += self._research_steps(findings)
                        return self._respond(
                            answer=findings.answer,
                            retrieved_chunks=chunks,
                            query=diagnosis_query,
                            query_type="diagnose",
                            tool_used="research_agent",
                            steps_taken=steps_taken,
                            start=start,
                            web_results=findings.results,
                            diagnosis=diagnosis_info,
                            session_id=session_id,
                            weather_risk=weather_risk,
                        )
                if settings.web_search_enabled and not research_attempted:
                    web_results = self._search_web(
                        diagnosis_query, confirm_web_search=confirm_web_search
                    )
                    web_search_attempted = True
                    steps_taken += 1  # web search

            extra_instruction = _format_weather_instruction(weather_risk) if weather_risk else None
            answer = self._generate(
                diagnosis_query,
                chunks,
                recent_history,
                extra_instruction=extra_instruction,
                web_results=web_results,
                persona=persona,
                language=language,
            )
            llm_calls = 1
            steps_taken += 1  # generation

            answer, llm_calls, steps_taken, web_results, web_search_attempted = self._correct(
                diagnosis_query,
                chunks,
                answer,
                recent_history,
                web_results,
                web_search_attempted,
                llm_calls,
                steps_taken,
                confirm_web_search=confirm_web_search,
                persona=persona,
                language=language,
            )
        except AppError:
            # Includes VisionServiceError from diagnose_image, alongside the
            # same retrieval/prompt/LLM exceptions handle_query can raise.
            raise
        except Exception as exc:
            raise ChatServiceError(
                f"Unexpected error while handling image diagnosis: {exc}"
            ) from exc

        tool_used = "web_search" if web_results else "diagnose"
        response = self._respond(
            answer=answer,
            retrieved_chunks=chunks,
            query=diagnosis_query,
            query_type="diagnose",
            tool_used=tool_used,
            steps_taken=steps_taken,
            start=start,
            web_results=web_results,
            diagnosis=diagnosis_info,
            session_id=session_id,
            weather_risk=weather_risk,
        )
        self._cache_response(
            query=diagnosis_query,
            response=response,
            crop=crop_context,
            disease=disease_context,
            tenant_id=tenant_id,
        )
        return response

    def stream_diagnose(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        query: str | None = None,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        confirm_web_search: bool = False,
        tenant_id: int | None = None,
        persona: str | None = "agronomist",
        engine: str = "hybrid",
        weather_risk: WeatherRiskResponse | None = None,
        language: str = "en",
    ) -> Iterator[dict[str, Any]]:
        """Streamed counterpart to handle_diagnose, for POST /chat/diagnose/stream."""
        start = time.perf_counter()
        steps_taken = 1  # planning

        plan = PlanDecision(action="diagnose")
        reset_usage()  # per-request LLM token/cost rollup
        logger.info(
            "plan_decided",
            extra={
                "extra_fields": {"action": plan.action, "query_length": len(query) if query else 0, "engine": engine}
            },
        )

        history = self._inject_memory(history, session_id)
        recent_history = history[-_MAX_HISTORY_TURNS:] if history else None

        yield {
            "type": "trace",
            "event": "vision_analyzing",
            "stage": "vision_analyzing",
            "payload": {"filename": filename, "engine": engine},
            "detail": {"filename": filename, "engine": engine},
        }

        if weather_risk is not None:
            yield {
                "type": "trace",
                "event": "weather_assessed",
                "stage": "weather_assessed",
                "payload": (
                    weather_risk.model_dump()
                    if hasattr(weather_risk, "model_dump")
                    else weather_risk
                ),
                "detail": {
                    "risk_level": weather_risk.risk_level,
                    "risk_score": weather_risk.risk_score,
                },
            }

        try:
            steps_taken += 1  # vision inference
            prediction = diagnose_image(image_bytes, filename, content_type, engine=engine)

            # Emitted immediately once vision classifier returns
            yield {
                "type": "diagnosis",
                "payload": prediction.model_dump(),
            }

            crop_context = (
                prediction.crop if prediction.crop and prediction.crop != "unknown" else None
            )
            disease_context = (
                prediction.disease if prediction.disease and prediction.disease != "unknown" else None
            )
            diagnosis_query = _build_diagnosis_query(prediction, query, collection=crop_context)

            # Check semantic cache for instant response
            cached = self._get_cached_response(
                query=diagnosis_query,
                crop=crop_context,
                disease=disease_context,
                tenant_id=tenant_id,
            )
            if cached is not None:
                logger.info(
                    "cache_hit",
                    extra={
                        "extra_fields": {
                            "query": diagnosis_query,
                            "crop": crop_context,
                            "disease": disease_context,
                            "session_id": session_id or "none",
                            "cached": True,
                        }
                    },
                )
                cached_dict = cached.model_dump()
                cached_dict["session_id"] = session_id or cached_dict.get("session_id", "")
                cached_dict["diagnosis"] = _build_diagnosis_info(prediction).model_dump()
                if weather_risk is not None:
                    cached_dict["weather_risk"] = weather_risk.model_dump()
                if "metadata" not in cached_dict or not isinstance(cached_dict["metadata"], dict):
                    cached_dict["metadata"] = {}
                cached_dict["metadata"]["cached"] = True
                cached_resp = ChatResponse.model_validate(cached_dict)
                yield {
                    "type": "trace",
                    "event": "cache_hit",
                    "stage": "cache_hit",
                    "payload": {"cached": True},
                    "detail": {"cached": True},
                }
                yield {"type": "done", "payload": cached_resp}
                return

            steps_taken += 1  # retrieval
            chunks = retrieve(
                diagnosis_query,
                self._vector_store,
                tenant_id=tenant_id,
                image_vector_store=self._image_vector_store,
                collection=crop_context,
                rerank=True,
            )

            yield {
                "type": "trace",
                "event": "retrieval_completed",
                "stage": "retrieval_completed",
                "payload": {"chunks_count": len(chunks)},
                "detail": {"chunks_count": len(chunks)},
            }

            steps_taken += 1  # grading
            grade = self._grade_retrieval(diagnosis_query, chunks)
            yield _trace_event("grading", {"grade": grade})

            web_results: list[WebSearchResult] = []
            web_search_attempted = False
            research_attempted = False
            if grade != "good":
                if settings.research_agent_enabled and settings.web_search_enabled:
                    research_attempted = True
                    yield _trace_event("research_handoff", {"reason": "diagnose_weak_grade"})
                    findings = self._research_handoff(
                        diagnosis_query, confirm_web_search, reason="diagnose_weak_grade"
                    )
                    for step in findings.steps:
                        yield _trace_event(f"research_{step['stage']}", step)
                    if findings.answer:
                        steps_taken += self._research_steps(findings)
                        yield {
                            "type": "answer_chunk",
                            "payload": {"token": findings.answer},
                            "text": findings.answer,
                        }
                        response = self._respond(
                            answer=findings.answer,
                            retrieved_chunks=chunks,
                            query=diagnosis_query,
                            query_type="diagnose",
                            tool_used="research_agent",
                            steps_taken=steps_taken,
                            start=start,
                            web_results=findings.results,
                            diagnosis=_build_diagnosis_info(prediction),
                            session_id=session_id,
                            weather_risk=weather_risk,
                        )
                        yield {"type": "done", "payload": response}
                        return
                if settings.web_search_enabled and not research_attempted:
                    web_results = self._search_web(
                        diagnosis_query, confirm_web_search=confirm_web_search
                    )
                    web_search_attempted = True
                    steps_taken += 1  # web search
                    yield _trace_event("web_search", {"result_count": len(web_results)})

            yield _trace_event("generating", {})
            answer = ""
            extra_instruction = _format_weather_instruction(weather_risk) if weather_risk else None
            for is_final, value in self._generate_streamed(
                diagnosis_query,
                chunks,
                recent_history,
                extra_instruction=extra_instruction,
                web_results=web_results,
                persona=persona,
                language=language,
            ):
                if is_final:
                    answer = value
                else:
                    yield {
                        "type": "answer_chunk",
                        "payload": {"token": value},
                        "text": value,
                    }
            llm_calls = 1
            steps_taken += 1  # generation

            (
                answer,
                llm_calls,
                steps_taken,
                web_results,
                web_search_attempted,
            ) = yield from self._correct_streamed(
                diagnosis_query,
                chunks,
                answer,
                recent_history,
                web_results,
                web_search_attempted,
                llm_calls,
                steps_taken,
                confirm_web_search=confirm_web_search,
                persona=persona,
                language=language,
            )

            follow_up_questions = []
            if settings.follow_up_questions_enabled:
                follow_up_questions = self._suggest_follow_ups(diagnosis_query, answer)

        except AppError as exc:
            logger.info(
                "diagnose_stream_error",
                extra={
                    "extra_fields": {
                        "error_type": type(exc).__name__,
                        "status_code": exc.status_code,
                    }
                },
            )
            yield {
                "type": "error",
                "detail": {
                    "error_type": type(exc).__name__,
                    "message": exc.detail,
                    "status_code": exc.status_code,
                },
                "payload": {
                    "error_type": type(exc).__name__,
                    "message": exc.detail,
                    "status_code": exc.status_code,
                },
            }
            return
        except Exception as exc:
            chat_error = ChatServiceError(f"Unexpected error while handling image diagnosis: {exc}")
            logger.info(
                "diagnose_stream_error",
                extra={
                    "extra_fields": {
                        "error_type": type(chat_error).__name__,
                        "status_code": chat_error.status_code,
                    }
                },
            )
            yield {
                "type": "error",
                "detail": {
                    "error_type": type(chat_error).__name__,
                    "message": chat_error.detail,
                    "status_code": chat_error.status_code,
                },
                "payload": {
                    "error_type": type(chat_error).__name__,
                    "message": chat_error.detail,
                    "status_code": chat_error.status_code,
                },
            }
            return

        tool_used = "web_search" if web_results else "diagnose"
        response = self._respond(
            answer=answer,
            retrieved_chunks=chunks,
            query=diagnosis_query,
            query_type="diagnose",
            tool_used=tool_used,
            steps_taken=steps_taken,
            start=start,
            web_results=web_results,
            diagnosis=_build_diagnosis_info(prediction),
            session_id=session_id,
            retrieval_confidence=grade,
            follow_up_questions=follow_up_questions,
            weather_risk=weather_risk,
        )
        self._cache_response(
            query=diagnosis_query,
            response=response,
            crop=crop_context,
            disease=disease_context,
            tenant_id=tenant_id,
        )
        yield {"type": "done", "payload": response}

    def _respond(
        self,
        *,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        query: str,
        query_type: str,
        tool_used: str,
        steps_taken: int,
        start: float,
        web_results: list[WebSearchResult] | None = None,
        diagnosis: DiagnosisInfo | None = None,
        session_id: str | None = None,
        retrieval_confidence: str = "good",
        is_clarifying_question: bool = False,
        follow_up_questions: list[str] | None = None,
        weather_risk: WeatherRiskResponse | None = None,
    ) -> ChatResponse:
        processing_duration = time.perf_counter() - start
        web_results = web_results or []
        chunk_sources = _source_references(retrieved_chunks)
        web_sources = _web_source_references(web_results, start=len(chunk_sources) + 1)
        sources = chunk_sources + web_sources
        answer_source = _answer_source(retrieved_chunks, web_results)

        # Per-request LLM usage rollup (accumulated via usage_tracking
        # ContextVar by the clients on each generate/generate_stream call).
        usage = current_usage()

        # Feature #4 — lexical groundedness check. Signal-only: the answer
        # is delivered unchanged; a low score just sets the
        # hallucination_detected flag and logs/metrics it for the operator
        # to inspect (and the user to double-check against the sources).
        hallucination_detected, grounding_score = _detect_hallucination(
            answer, retrieved_chunks, web_results
        )
        if hallucination_detected:
            get_metrics().inc_counter(
                "hallucinations_detected_total", {"answer_source": answer_source}
            )
            logger.warning(
                "hallucination_detected",
                extra={
                    "extra_fields": {
                        "grounding_score": grounding_score,
                        "answer_source": answer_source,
                        "answer_length": len(answer),
                    }
                },
            )

        log_fields = {
            "query_length": len(query),
            "query_type": query_type,
            "tool_used": tool_used,
            "steps_taken": steps_taken,
            "retrieved_chunk_count": len(retrieved_chunks),
            "web_result_count": len(web_results),
            "answer_source": answer_source,
            "processing_duration": round(processing_duration, 4),
            "llm_calls": usage["llm_calls"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": usage["estimated_cost_usd"],
            "grounding_score": grounding_score,
            "hallucination_detected": hallucination_detected,
        }
        if diagnosis is not None:
            log_fields.update(
                {
                    "diagnosis_crop": diagnosis.crop,
                    "diagnosis_disease": diagnosis.disease,
                    "diagnosis_confidence": diagnosis.confidence,
                    "diagnosis_low_confidence": diagnosis.low_confidence,
                }
            )
        if weather_risk is not None:
            log_fields.update(
                {
                    "weather_risk_level": weather_risk.risk_level,
                    "weather_risk_score": weather_risk.risk_score,
                }
            )
        logger.info("chat_query_handled", extra={"extra_fields": log_fields})

        # Agent memory: record this exchange (turns, and — when fact
        # extraction is enabled — durable facts) so later questions in the
        # session can draw on it. Best-effort; never affects the response.
        self._remember(session_id, query, answer, retrieved_chunks, query_type)

        return ChatResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            sources=sources,
            processing_time=round(processing_duration, 4),
            tool_used=tool_used,
            steps_taken=steps_taken,
            answer_source=answer_source,
            diagnosis=diagnosis,
            session_id=session_id or "",
            retrieval_confidence=retrieval_confidence,
            is_clarifying_question=is_clarifying_question,
            follow_up_questions=follow_up_questions or [],
            hallucination_detected=hallucination_detected,
            grounding_score=grounding_score,
            weather_risk=weather_risk,
        )
