"""Phase 1: the new production chat graph (build_chat_graph) — built and
unit-tested in isolation. Not yet wired into ChatService.handle_query/
stream_query (that lands in a later commit), so these tests drive the graph
directly with a fake ChatService double rather than going through the HTTP
layer.
"""

from __future__ import annotations

import pytest

from app.models.document import RetrievedChunk
from app.models.schemas import ChatResponse
from app.services.agent_graph.graph import build_chat_graph
from app.services.agent_graph.nodes import GraphContext, generator_node
from app.services.agent_graph.state import AgentState
from app.services.prompt_builder import FALLBACK_REPLY, GENERATION_ERROR_REPLY
from app.services.rag_service import PlanDecision, RetrievalAugmentation


class FakeChatService:
    """Duck-typed stand-in for ChatService exposing only the private
    methods the new nodes delegate to — no LLM/network calls."""

    def __init__(
        self,
        plan_action="retrieve",
        grade="good",
        ungrounded=False,
        augmentation: RetrievalAugmentation | None = None,
        cached_response: ChatResponse | None = None,
        corrected_answer: str | None = None,
    ):
        self._plan_action = plan_action
        self._grade = grade
        self._ungrounded = ungrounded
        self._augmentation = augmentation or RetrievalAugmentation()
        self._cached_response = cached_response
        self._corrected_answer = corrected_answer
        self.generate_calls: list[str] = []
        self.cache_write_calls: list[dict] = []
        self.cache_lookup_calls = 0
        self.correct_calls = 0
        self.augment_calls: list[bool | None] = []

    def _plan(self, query, history=None):
        return PlanDecision(action=self._plan_action)

    def _route(self, query, history=None):
        return self._plan(query, history)

    def _grade_retrieval(self, query, chunks):
        return self._grade

    def _generate(
        self, query, chunks, history, extra_instruction=None, web_results=None, persona=None, language=None
    ):
        self.generate_calls.append(extra_instruction or "initial")
        return "grounded answer [1]"

    def _generate_structured(self, *args, **kwargs):
        answer = self._generate(*args, **kwargs)
        return answer, {"answer": answer, "sources": []}

    def _is_ungrounded(self, answer, chunks, web_results):
        return self._ungrounded

    def _correct(
        self,
        query,
        chunks,
        answer,
        history,
        web_results,
        web_search_attempted,
        llm_calls,
        steps_taken,
        confirm_web_search=False,
        persona=None,
        language=None,
    ):
        self.correct_calls += 1
        final_answer = self._corrected_answer if self._corrected_answer is not None else answer
        return final_answer, llm_calls + 1, steps_taken + 1, web_results, web_search_attempted

    def _augment_weak_retrieval(self, *args, **kwargs):
        # args[-1] is confirm_web_search per ChatService's real positional
        # signature -- recorded so Phase 5's approval-wiring tests can
        # assert whether the guarded web-search escalation actually ran.
        self.augment_calls.append(args[-1] if args else None)
        return self._augmentation

    def _get_cached_response(self, query, crop=None, disease=None, tenant_id=None, document_ids=None):
        self.cache_lookup_calls += 1
        return self._cached_response

    def _cache_response(self, query, response, crop=None, disease=None, tenant_id=None, document_ids=None):
        self.cache_write_calls.append({"query": query, "response": response})

    def _maybe_ask_clarifying_question(self, query, answer, grade):
        return answer, False

    def _suggest_follow_ups(self, query, answer):
        return []

    def _respond(
        self,
        *,
        answer,
        retrieved_chunks,
        query,
        query_type,
        tool_used,
        steps_taken,
        start,
        web_results=None,
        session_id=None,
        retrieval_confidence="good",
        is_clarifying_question=False,
        follow_up_questions=None,
        **_ignored,
    ):
        return ChatResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            sources=[],
            processing_time=0.0,
            tool_used=tool_used,
            steps_taken=steps_taken,
            answer_source="documents" if retrieved_chunks else ("web" if web_results else "documents"),
            session_id=session_id or "",
            retrieval_confidence=retrieval_confidence,
            is_clarifying_question=is_clarifying_question,
            follow_up_questions=follow_up_questions or [],
        )


def make_chunk(score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c1", document_id="d1", text="content", score=score)


class FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, *args, **kwargs):
        return self.chunks

    def search_bm25(self, *args, **kwargs):
        return self.chunks


@pytest.mark.asyncio
async def test_conversational_path_finalizes_without_retrieval():
    graph = build_chat_graph()
    ctx = GraphContext(chat_service=FakeChatService(plan_action="conversational"))
    state = AgentState(query="hi")
    result = await graph.run(state, ctx)
    assert result.workflow_status == "completed"
    assert result.final_response is not None
    assert result.termination_reason == "success"


@pytest.mark.asyncio
async def test_good_retrieval_path_skips_augmentation():
    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="good", ungrounded=False)
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="what is the scope?")
    result = await graph.run(state, ctx)
    assert result.retrieval_grade == "good"
    assert result.tool_calls == []  # context_augmentation never ran
    assert result.workflow_status == "completed"
    assert result.final_response is not None


@pytest.mark.asyncio
async def test_weak_retrieval_path_falls_back_to_web_search():
    graph = build_chat_graph()
    augmentation = RetrievalAugmentation(web_results=[], web_search_attempted=True)
    fake = FakeChatService(plan_action="retrieve", grade="weak", ungrounded=False, augmentation=augmentation)
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk(score=0.2)]))
    state = AgentState(query="what happened today?")
    result = await graph.run(state, ctx)
    assert result.retrieval_grade == "weak"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool_name"] == "web_search"
    assert result.workflow_status == "completed"


@pytest.mark.asyncio
async def test_weak_retrieval_direct_hit_skips_generation_entirely():
    """vision QA / local research / research agent producing a direct
    answer (via _augment_weak_retrieval) must skip generator/output_
    validation/reflection entirely and go straight to the finalizer —
    exactly like handle_query's early `return self._respond(...)`."""
    graph = build_chat_graph()
    augmentation = RetrievalAugmentation(final_answer="vision says X", tool_used="vision_qa", extra_steps=1)
    fake = FakeChatService(plan_action="retrieve", grade="insufficient", augmentation=augmentation)
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([]))
    state = AgentState(query="what disease is this?")
    result = await graph.run(state, ctx)
    assert result.final_response.answer == "vision says X"
    assert fake.generate_calls == []  # generator never invoked
    assert result.workflow_status == "completed"


@pytest.mark.asyncio
async def test_reflection_delegates_to_correct_exactly_once():
    """reflection_node calls ChatService._correct wholesale (which owns its
    own bounded internal regenerate/web-fallback escalation) rather than
    the graph looping generically — so from the graph's perspective this is
    always exactly one call, regardless of whether _correct internally
    regenerated 0, 1, or 2 extra times."""
    graph = build_chat_graph()
    fake = FakeChatService(
        plan_action="retrieve", grade="good", ungrounded=True, corrected_answer="corrected [1]"
    )
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q")
    result = await graph.run(state, ctx)
    assert fake.correct_calls == 1
    assert result.final_response.answer == "corrected [1]"
    assert result.workflow_status == "completed"
    assert result.termination_reason == "success"
    assert len(fake.generate_calls) == 1  # generator itself only ran once


@pytest.mark.asyncio
async def test_node_timings_and_trace_ids_recorded():
    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="good", ungrounded=False)
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q")
    result = await graph.run(state, ctx)
    assert result.request_id is not None
    assert result.trace_id is not None
    assert "planner" in result.node_timings
    assert "cache_lookup" in result.node_timings
    assert "retrieval" in result.node_timings
    assert "generator" in result.node_timings
    assert "finalizer" in result.node_timings


@pytest.mark.asyncio
async def test_retrieval_without_vector_store_degrades_safely():
    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="insufficient", ungrounded=False)
    ctx = GraphContext(chat_service=fake, vector_store=None)
    state = AgentState(query="q")
    result = await graph.run(state, ctx)
    assert result.error_type == "retriever"
    # Workflow still reaches a final response rather than crashing.
    assert result.final_response is not None


@pytest.mark.asyncio
async def test_cache_hit_short_circuits_retrieval_and_generation():
    graph = build_chat_graph()
    cached = ChatResponse(
        answer="cached answer",
        retrieved_chunks=[],
        sources=[],
        processing_time=0.01,
        tool_used="retrieval",
        steps_taken=3,
        answer_source="documents",
        session_id="",
    )
    fake = FakeChatService(plan_action="retrieve", cached_response=cached)
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q")
    result = await graph.run(state, ctx)
    assert result.final_response.answer == "cached answer"
    assert result.final_response.metadata.get("cached") is True
    assert fake.generate_calls == []  # generation never ran
    assert "retrieval" not in result.node_timings  # retrieval never ran either


@pytest.mark.asyncio
async def test_cache_write_happens_only_for_plain_retrieve():
    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="good")
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q")
    await graph.run(state, ctx)
    assert len(fake.cache_write_calls) == 1


class _RaisingChatService(FakeChatService):
    """_generate raises every call -- simulates an LLM provider failure
    (timeout/rate-limit/API error) that survives all of groq_client.py's/
    gemini_client.py's tenacity retries and propagates out of
    ChatService._generate, exactly as it does in production."""

    def _generate(self, *args, **kwargs):
        self.generate_calls.append("raised")
        raise RuntimeError("simulated LLM provider failure (e.g. Groq 429 after 3 retries)")


def test_generator_exception_uses_generation_error_reply_not_fallback():
    """PHASE 3 FAITHFULNESS-REGRESSION FIX regression test: before this fix,
    generator_node's except block set draft_answer=FALLBACK_REPLY on ANY
    exception from ChatService._generate, making a provider failure
    indistinguishable from a legitimate 'not in the documents' answer.
    This reproduces the failure path directly (bypassing the graph's own
    retry/reflection wiring) and asserts the node now emits the distinct
    GENERATION_ERROR_REPLY sentinel instead, with error_type/root_cause
    still recorded for tracing."""
    fake = _RaisingChatService(plan_action="retrieve")
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q", retrieved_chunks=[make_chunk()])

    result = generator_node(state, ctx)

    assert result.draft_answer == GENERATION_ERROR_REPLY
    assert result.draft_answer != FALLBACK_REPLY
    assert result.error_type is not None
    assert "simulated LLM provider failure" in result.error_message


def test_generation_error_reply_triggers_reflection_retry():
    """A GENERATION_ERROR_REPLY draft answer must still be treated as
    ungrounded by ChatService._is_ungrounded (the real implementation, not
    the FakeChatService stub) so the corrective loop gives a transient
    provider failure a second, real chance to succeed -- the same
    treatment FALLBACK_REPLY already got before this fix."""
    from app.models.document import RetrievedChunk
    from app.services.rag_service import ChatService

    chunks = [RetrievedChunk(chunk_id="c1", document_id="d1", text="content", score=0.9)]
    service = ChatService.__new__(ChatService)  # no LLM/vector-store deps needed for _is_ungrounded
    assert service._is_ungrounded(GENERATION_ERROR_REPLY, chunks, []) is True
    assert service._is_ungrounded(FALLBACK_REPLY, chunks, []) is True
    assert service._is_ungrounded("a real grounded answer [1]", chunks, []) is False
    # No context at all: GENERATION_ERROR_REPLY would never legitimately be
    # produced here (there's nothing to call the LLM with), but confirm the
    # "no context -> not ungrounded" short-circuit still holds regardless.
    assert service._is_ungrounded(GENERATION_ERROR_REPLY, [], []) is False


# ---------------------------------------------------------------------------
# PHASE 5: human_approval_node wired into the live production graph for the
# web-search escalation. Before this, human_approval_node was registered in
# build_chat_graph() but had no inbound edge -- dead code (see
# docs/MODULE10_FINAL_AUDIT.md Section 8, Finding 1). These tests exercise
# the graph's real routing, not human_approval_node in isolation (already
# covered by test_human_approval_node.py).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weak_retrieval_approval_required_blocks_web_search_but_still_generates(monkeypatch):
    """No confirm_web_search, no approval reference, gate on: the request
    must be routed through human_approval_node (registering a pending
    Approval), never reach context_augmentation (so no web search happens),
    but still reach generator with the chunks retrieval already found --
    web search specifically is guarded, not generation from existing
    context."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "web_search_requires_approval", True)
    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="weak")
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q", confirm_web_search=False)

    result = await graph.run(state, ctx)

    assert fake.augment_calls == []  # context_augmentation never ran -- web search never happened
    assert len(fake.generate_calls) == 1  # generation from existing chunks still happened
    assert result.approval_status == "pending"
    assert result.final_response.metadata.get("approval_status") == "pending"
    assert result.final_response.metadata.get("approval_id") is not None


@pytest.mark.asyncio
async def test_weak_retrieval_genuine_approval_allows_web_search(monkeypatch):
    """A request carrying a reference to a genuinely APPROVED approval must
    reach context_augmentation (the real guarded action) via human_approval_node's
    'resume' route, with the effective confirm flag set."""
    from app.core.config import settings
    from app.services.approval_service import get_approval_store

    monkeypatch.setattr(settings, "web_search_requires_approval", True)
    approval = get_approval_store().register(action="web_search", payload={"query": "q"})
    get_approval_store().resolve(approval.approval_id, approved=True, resolved_by="operator")

    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="weak")
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q", confirm_web_search=False, approval_payload_reference=approval.approval_id)

    result = await graph.run(state, ctx)

    assert fake.augment_calls == [True]  # context_augmentation ran with the approval honored
    assert result.approval_status == "approved"


@pytest.mark.asyncio
async def test_weak_retrieval_rejected_approval_blocks_web_search(monkeypatch):
    from app.core.config import settings
    from app.services.approval_service import get_approval_store

    monkeypatch.setattr(settings, "web_search_requires_approval", True)
    approval = get_approval_store().register(action="web_search", payload={"query": "q"})
    get_approval_store().resolve(approval.approval_id, approved=False, resolved_by="operator")

    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="weak")
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q", confirm_web_search=False, approval_payload_reference=approval.approval_id)

    result = await graph.run(state, ctx)

    assert fake.augment_calls == []  # rejected -- web search must not run
    assert result.approval_status == "rejected"
    assert result.final_response.metadata.get("approval_status") == "rejected"


@pytest.mark.asyncio
async def test_confirm_web_search_fast_path_still_bypasses_approval_queue(monkeypatch):
    """Backward compatibility: a caller that already sets confirm_web_search=true
    directly (the pre-existing self-service fast path) must skip
    human_approval_node entirely, exactly as before this phase's wiring."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "web_search_requires_approval", True)
    graph = build_chat_graph()
    fake = FakeChatService(plan_action="retrieve", grade="weak")
    ctx = GraphContext(chat_service=fake, vector_store=FakeVectorStore([make_chunk()]))
    state = AgentState(query="q", confirm_web_search=True)

    result = await graph.run(state, ctx)

    assert fake.augment_calls == [True]
    assert result.approval_status == "not_required"  # human_approval_node never ran


# ---------------------------------------------------------------------------
# MODULE 10 FAITHFULNESS ROOT-CAUSE FIX (P2, 2026-09-20): FallbackLLMClient
# (app/services/fallback_llm_client.py) already implements and unit-tests a
# second-provider fallback when the primary provider's own internal retries
# are exhausted -- it was simply never wired in (Settings.fallback_llm_provider
# was unset). These tests reproduce the exact regression at the
# generator_node/ChatService integration level (not just FallbackLLMClient in
# isolation, which was already covered by tests/test_fallback_llm_client.py)
# and confirm the real fix: a real, grounded fallback answer, never a
# fabricated one, and never a silent GENERATION_ERROR_REPLY when a second
# provider was actually available and capable of answering.
# ---------------------------------------------------------------------------


class _RaisingPrimaryLLM:
    """Simulates the exact production failure: a Groq call that has
    already exhausted its own internal tenacity retries and raises
    LLMAPIError -- the trigger FallbackLLMClient watches for."""

    def generate(self, prompt: str) -> str:
        from app.core.exceptions import LLMAPIError

        raise LLMAPIError("simulated Groq 429 after 3 retries (reproduces eval-orange-01/pepper-01)")


class _GroundedFallbackLLM:
    """Simulates the secondary provider succeeding with a real,
    context-grounded answer -- not fabricated content, just a different
    provider answering the same evidence-bearing prompt."""

    def generate(self, prompt: str) -> str:
        assert "Apple scab" in prompt or "scab" in prompt.lower(), "fallback must see the same grounded prompt, not a stripped-down retry"
        return "Apple scab is treated with sulfur or captan fungicide, applied per label [1]."


def test_generator_node_recovers_via_fallback_provider_instead_of_generation_error_reply():
    """Reproduces the exact regression (a provider failure surviving
    retries) and proves the fix: with FallbackLLMClient wired in (as
    .env now configures via FALLBACK_LLM_PROVIDER), the final answer is
    the fallback provider's real, grounded text -- never
    GENERATION_ERROR_REPLY, and never a fabricated answer with no
    supporting evidence."""
    from app.services.fallback_llm_client import FallbackLLMClient
    from app.services.rag_service import ChatService

    fallback_client = FallbackLLMClient(
        primary=_RaisingPrimaryLLM(),
        primary_name="groq",
        fallback=_GroundedFallbackLLM(),
        fallback_name="gemini",
    )
    chat_service = ChatService.__new__(ChatService)
    chat_service._llm_client = fallback_client

    chunks = [RetrievedChunk(chunk_id="c1", document_id="d1", text="Apple scab treatment: Sulfur 80% WDG or Captan 50% WP.", score=0.9)]
    ctx = GraphContext(chat_service=chat_service)
    state = AgentState(query="What's the treatment for apple scab?", retrieved_chunks=chunks)

    result = generator_node(state, ctx)

    assert result.draft_answer == "Apple scab is treated with sulfur or captan fungicide, applied per label [1]."
    assert result.draft_answer != GENERATION_ERROR_REPLY
    assert result.draft_answer != FALLBACK_REPLY
    assert result.error_type is None  # recovered successfully -- not a failure state


def test_generator_node_still_returns_generation_error_reply_when_both_providers_fail():
    """The fix must not paper over a genuine total outage: when BOTH
    providers fail, the honest GENERATION_ERROR_REPLY (not a fabricated
    answer) is still exactly what's returned -- proving the fallback
    fix doesn't weaken the existing, correct no-fabrication guarantee."""
    from app.core.exceptions import LLMAPIError
    from app.services.fallback_llm_client import FallbackLLMClient
    from app.services.rag_service import ChatService

    class _AlsoRaisingFallback:
        def generate(self, prompt: str) -> str:
            raise LLMAPIError("simulated total outage: fallback provider also down")

    fallback_client = FallbackLLMClient(
        primary=_RaisingPrimaryLLM(),
        primary_name="groq",
        fallback=_AlsoRaisingFallback(),
        fallback_name="gemini",
    )
    chat_service = ChatService.__new__(ChatService)
    chat_service._llm_client = fallback_client

    chunks = [RetrievedChunk(chunk_id="c1", document_id="d1", text="Apple scab treatment: Sulfur.", score=0.9)]
    ctx = GraphContext(chat_service=chat_service)
    state = AgentState(query="What's the treatment for apple scab?", retrieved_chunks=chunks)

    result = generator_node(state, ctx)

    assert result.draft_answer == GENERATION_ERROR_REPLY  # honest, not fabricated
    assert result.error_type is not None
