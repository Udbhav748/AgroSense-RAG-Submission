"""AgentExecutor: runs a genuine ReAct loop - Plan -> Act -> Observe ->
Repeat, one step at a time.

Each iteration asks PlanningAgent.plan_next() for exactly one next step,
given every Observation gathered so far this run, executes it against the
ToolRegistry, and folds the result back in as the next Observation before
asking again. This is what lets a later step react to what an earlier one
actually returned (skip web_search once retrieval alone answered the
question; stop early if a tool came back empty) — a fixed upfront plan
can't do that, since every step's args would already be committed before
any tool had run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.core.exceptions import ChatServiceError
from app.models.schemas import ChatResponse, SourceReference
from app.services.planning_agent import Observation
from app.services.prompt_builder import build_prompt, strip_sources_section
from app.services.tool_registry import _summarize_output
from app.services.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.models.document import RetrievedChunk, WebSearchResult
    from app.services.agent_memory import AgentMemory
    from app.services.llm_client import LLMClient
    from app.services.planning_agent import PlanningAgent, PlanStep
    from app.services.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 5
_MAX_SYNTHESIS_LLM_CALLS = 3


class AgentExecutorError(ChatServiceError):
    status_code = 500
    error_code = "AGENT_EXECUTOR_ERROR"


@dataclass
class AgentResult:
    answer: str
    sources: list[SourceReference]
    retrieved_chunks: list[RetrievedChunk]
    steps_taken: int
    tool_used: str
    answer_source: str
    processing_time: float
    hallucination_detected: bool = False
    hallucination_score: float | None = None
    follow_up_questions: list[str] = field(default_factory=list)


@dataclass
class _StepResult:
    step: PlanStep
    result: ToolResult
    latency_ms: float


class AgentExecutor:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        planning_agent: PlanningAgent,
        agent_memory: AgentMemory | None = None,
    ):
        self._llm = llm_client
        self._tools = tool_registry
        self._planner = planning_agent
        self._memory = agent_memory

    async def execute(
        self,
        query: str,
        context: ToolContext,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        confirm_web_search: bool = False,
        persona: str | None = None,
    ) -> AgentResult:
        start_time = time.perf_counter()

        # Thread the client's web-search confirmation into the tool context
        # so WebSearchTool can enforce the human-approval gate
        # (Settings.web_search_requires_approval) the same way the inline
        # path does — the executor must not be a bypass for it.
        context.confirm_web_search = confirm_web_search

        if self._memory and settings.agent_memory_enabled and session_id:
            history = self._inject_memory(history, session_id)

        step_results: list[_StepResult] = []
        observations: list[Observation] = []
        all_chunks: list[RetrievedChunk] = []
        all_web_results: list[WebSearchResult] = []
        tool_used = "retrieval"
        answer_source = "documents"
        llm_calls = 0

        for _ in range(_MAX_ITERATIONS):
            step = await self._planner.plan_next(query, history, observations)
            llm_calls += 1

            if step.tool == "synthesize":
                break

            step_start = time.perf_counter()
            try:
                result = await self._tools.execute(step.tool, step.args, context)
            except Exception as exc:
                logger.error(
                    "agent_tool_execution_failed",
                    extra={"extra_fields": {"tool": step.tool, "error": str(exc)}},
                )
                result = ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

            step_latency = (time.perf_counter() - step_start) * 1000
            step_results.append(_StepResult(step=step, result=result, latency_ms=step_latency))
            observations.append(
                Observation(
                    step=step,
                    success=result.success,
                    summary=_summarize_output(result.data) if result.success else result.error,
                )
            )

            if step.tool == "retrieval" and result.success and result.data:
                all_chunks.extend(result.data)
                tool_used = "retrieval"
            elif step.tool == "web_search" and result.success and result.data:
                all_web_results.extend(result.data)
                tool_used = "web_search"
                answer_source = "web" if not all_chunks else "mixed"
            elif step.tool == "summarization" and result.success and result.data:
                summary, chunks = result.data
                all_chunks.extend(chunks)
                tool_used = "summarization"
            elif step.tool == "diagnose" and result.success and result.data:
                tool_used = "diagnose"
            elif step.tool == "vision_qa" and result.success and result.data:
                tool_used = "vision_qa"
        else:
            # Loop exhausted _MAX_ITERATIONS without the planner ever
            # choosing "synthesize" — synthesize from whatever was
            # gathered rather than looping forever.
            logger.warning(
                "agent_max_iterations_reached", extra={"extra_fields": {"max": _MAX_ITERATIONS}}
            )

        answer, used_web = await self._synthesize(
            query, all_chunks, all_web_results, history, persona, step_results, llm_calls
        )
        llm_calls += 1

        sources = self._build_sources(all_chunks, all_web_results)
        steps_taken = len(step_results) + 1

        hallucination_detected = False
        hallucination_score = None
        if settings.hallucination_detection_enabled:
            hallucination_detected, hallucination_score = self._detect_hallucination(
                answer, all_chunks, all_web_results
            )

        follow_up_questions = []
        if settings.follow_up_questions_enabled:
            follow_up_questions = self._suggest_follow_ups(query, answer)

        if self._memory and settings.agent_memory_enabled and session_id:
            self._remember(session_id, query, answer, all_chunks, tool_used)

        processing_time = time.perf_counter() - start_time

        return AgentResult(
            answer=answer,
            sources=sources,
            retrieved_chunks=all_chunks,
            steps_taken=steps_taken,
            tool_used=tool_used,
            answer_source=answer_source if used_web else "documents",
            processing_time=processing_time,
            hallucination_detected=hallucination_detected,
            hallucination_score=hallucination_score,
            follow_up_questions=follow_up_questions,
        )

    async def _synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        web_results: list[WebSearchResult],
        history: list[dict[str, str]] | None,
        persona: str | None,
        step_results: list[_StepResult],
        llm_calls: int,
    ) -> tuple[str, bool]:
        used_web = bool(web_results)

        if not chunks and not web_results:
            from app.services.prompt_builder import FALLBACK_REPLY

            return FALLBACK_REPLY, used_web

        extra_instruction = (
            "Use the following context from tool executions to answer the user's question. "
            "Cite sources inline with [N] markers. If the context is insufficient, say so."
        )

        prompt = build_prompt(
            query=query,
            chunks=chunks,
            history=history,
            extra_instruction=extra_instruction,
            web_results=web_results,
            persona=persona,
        )

        answer = strip_sources_section(self._llm.generate(prompt))
        return answer, used_web

    def _build_sources(
        self, chunks: list[RetrievedChunk], web_results: list[WebSearchResult]
    ) -> list[SourceReference]:
        from app.services.rag_service import _source_references, _web_source_references

        sources: list[SourceReference] = []
        if chunks:
            sources.extend(_source_references(chunks, start=1))
        if web_results:
            sources.extend(_web_source_references(web_results, start=len(sources) + 1))
        return sources

    def _detect_hallucination(
        self, answer: str, chunks: list[RetrievedChunk], web_results: list[WebSearchResult]
    ) -> tuple[bool, float | None]:
        from app.services.rag_service import _detect_hallucination

        return _detect_hallucination(answer, chunks, web_results)

    def _suggest_follow_ups(self, query: str, answer: str) -> list[str]:
        """Suggest up to 3 short follow-up questions. Degrades to an empty
        list on any LLM/parse failure — never blocks or fails the main
        answer. Mirrors ChatService._suggest_follow_ups in rag_service.py
        (duplicated rather than imported: that's an instance method bound
        to ChatService's own LLM client, not a module-level function)."""
        prompt = (
            f"Question: {query}\nAnswer: {answer}\n\n"
            "Suggest up to 3 short, natural follow-up questions the user "
            "might ask next. Return ONLY a JSON array of strings, e.g. "
            '["question one?", "question two?"]. Return an empty array [] '
            "if you can't think of good ones."
        )
        try:
            import json

            raw = self._llm.generate(prompt).strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
            return [str(q) for q in result][:3] if isinstance(result, list) else []
        except Exception:
            return []

    def _inject_memory(
        self, history: list[dict[str, str]] | None, session_id: str
    ) -> list[dict[str, str]] | None:
        if self._memory is None or not settings.agent_memory_enabled or not session_id:
            return history
        try:
            block = self._memory.build_context(session_id)
        except Exception as exc:
            logger.warning(
                "agent_memory_context_failed",
                extra={"extra_fields": {"session_id": session_id, "error": str(exc)}},
            )
            return history
        if not block:
            return history
        turns = [t for t in (history or []) if t.get("role") != "system"]
        return turns + [{"role": "system", "content": block}]

    def _remember(
        self,
        session_id: str,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        query_type: str,
    ) -> None:
        if self._memory is None or not settings.agent_memory_enabled or not session_id:
            return
        try:
            self._memory.add_turn(session_id, "user", query)
            self._memory.add_turn(session_id, "assistant", answer)
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
        source_chunk_ids = [getattr(chunk, "chunk_id", str(chunk)) for chunk in chunks]
        for key, value, confidence in facts:
            self._memory.upsert_fact(
                session_id, key, value, confidence=confidence, source_chunk_ids=source_chunk_ids
            )

    def _extract_facts(self, query: str, answer: str) -> list[tuple[str, str, str]]:
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
        raw = self._llm.generate(prompt)
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        import json

        try:
            payload = json.loads(raw)
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

    async def stream_execute(
        self,
        query: str,
        context: ToolContext,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        confirm_web_search: bool = False,
        persona: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        start_time = time.perf_counter()

        # Same approval-gate threading as execute() — the streamed path
        # must enforce the gate identically, not become a second bypass.
        context.confirm_web_search = confirm_web_search

        if self._memory and settings.agent_memory_enabled and session_id:
            history = self._inject_memory(history, session_id)

        step_results: list[_StepResult] = []
        observations: list[Observation] = []
        all_chunks: list[RetrievedChunk] = []
        all_web_results: list[WebSearchResult] = []
        tool_used = "retrieval"
        answer_source = "documents"
        llm_calls = 0

        for _ in range(_MAX_ITERATIONS):
            yield {
                "type": "trace",
                "stage": "planning",
                "detail": {"step_index": len(observations)},
            }
            step = await self._planner.plan_next(query, history, observations)
            llm_calls += 1

            if step.tool == "synthesize":
                break

            yield {
                "type": "trace",
                "stage": "tool_execution",
                "detail": {"tool": step.tool, "reason": step.reason},
            }

            step_start = time.perf_counter()
            try:
                result = await self._tools.execute(step.tool, step.args, context)
            except Exception as exc:
                logger.error(
                    "agent_tool_execution_failed",
                    extra={"extra_fields": {"tool": step.tool, "error": str(exc)}},
                )
                result = ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

            step_latency = (time.perf_counter() - step_start) * 1000
            step_results.append(_StepResult(step=step, result=result, latency_ms=step_latency))
            observations.append(
                Observation(
                    step=step,
                    success=result.success,
                    summary=_summarize_output(result.data) if result.success else result.error,
                )
            )

            if step.tool == "retrieval" and result.success and result.data:
                all_chunks.extend(result.data)
                tool_used = "retrieval"
                yield {
                    "type": "trace",
                    "stage": "retrieval",
                    "detail": {"chunk_count": len(result.data)},
                }
            elif step.tool == "web_search" and result.success and result.data:
                all_web_results.extend(result.data)
                tool_used = "web_search"
                answer_source = "web" if not all_chunks else "mixed"
                yield {
                    "type": "trace",
                    "stage": "web_search",
                    "detail": {"result_count": len(result.data)},
                }
            elif step.tool == "summarization" and result.success and result.data:
                summary, chunks = result.data
                all_chunks.extend(chunks)
                tool_used = "summarization"
            elif step.tool == "diagnose" and result.success and result.data:
                tool_used = "diagnose"
            elif step.tool == "vision_qa" and result.success and result.data:
                tool_used = "vision_qa"

            if not result.success:
                yield {
                    "type": "trace",
                    "stage": "tool_error",
                    "detail": {"tool": step.tool, "error": result.error},
                }
        else:
            # Loop exhausted _MAX_ITERATIONS without the planner ever
            # choosing "synthesize" — synthesize from whatever was
            # gathered rather than looping forever.
            yield {
                "type": "trace",
                "stage": "warning",
                "detail": {"message": "max iterations reached"},
            }

        yield {
            "type": "trace",
            "stage": "generating",
            "detail": {"chunk_count": len(all_chunks), "web_result_count": len(all_web_results)},
        }

        if not all_chunks and not all_web_results:
            from app.services.prompt_builder import FALLBACK_REPLY

            answer = FALLBACK_REPLY
        else:
            extra_instruction = (
                "Use the following context from tool executions to answer the user's question. "
                "Cite sources inline with [N] markers. If the context is insufficient, say so."
            )
            prompt = build_prompt(
                query=query,
                chunks=all_chunks,
                history=history,
                extra_instruction=extra_instruction,
                web_results=all_web_results,
                persona=persona,
            )
            from app.services.rag_service import _stream_filtering_sources

            # generate_stream is a sync iterator (LLMClient interface) —
            # _stream_filtering_sources consumes it synchronously and drops
            # the model's own "Sources:" heading before it reaches the
            # client, same as the non-streaming path's strip_sources_section.
            raw_stream = self._llm.generate_stream(prompt)
            answer_parts = []
            for filtered in _stream_filtering_sources(raw_stream):
                answer_parts.append(filtered)
                yield {"type": "answer_chunk", "text": filtered}

            answer = "".join(answer_parts)
            llm_calls += 1

        sources = self._build_sources(all_chunks, all_web_results)
        steps_taken = len(step_results) + 1

        hallucination_detected = False
        hallucination_score = None
        if settings.hallucination_detection_enabled:
            hallucination_detected, hallucination_score = self._detect_hallucination(
                answer, all_chunks, all_web_results
            )

        follow_up_questions = []
        if settings.follow_up_questions_enabled:
            follow_up_questions = self._suggest_follow_ups(query, answer)

        if self._memory and settings.agent_memory_enabled and session_id:
            self._remember(session_id, query, answer, all_chunks, tool_used)

        processing_time = time.perf_counter() - start_time

        payload = ChatResponse(
            answer=answer,
            retrieved_chunks=all_chunks,
            sources=sources,
            processing_time=processing_time,
            tool_used=tool_used,
            steps_taken=steps_taken,
            answer_source=answer_source if all_web_results else "documents",
            hallucination_detected=hallucination_detected,
            grounding_score=hallucination_score,
            follow_up_questions=follow_up_questions,
            session_id=session_id or "",
        )

        yield {"type": "done", "payload": payload}
