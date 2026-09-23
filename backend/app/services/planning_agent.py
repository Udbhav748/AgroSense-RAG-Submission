"""PlanningAgent: the "Plan" half of AgentExecutor's ReAct (Plan -> Act ->
Observe -> Repeat) loop.

plan_next() decides exactly ONE next step at a time — a tool call, or
"synthesize" to stop — given every observation this run has accumulated so
far (see Observation below). It never commits to a multi-step plan
upfront: a step chosen now can react to what an earlier step in the same
run actually returned (e.g. skip web_search once retrieval alone already
answered the question, or fall back to synthesizing early if a tool came
back empty), which a fixed upfront plan structurally cannot do.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.llm_client import LLMClient
    from app.services.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_STEP_SCHEMA = (
    '{"tool": "retrieval|web_search|summarization|diagnose|vision_qa|synthesize", '
    '"args": {...}, "reason": "..."}'
)

_NEXT_STEP_PROMPT_TEMPLATE = """You are the planner for a multi-agent document assistant, working in a Plan -> Act -> Observe loop. Decide the SINGLE next action needed to answer the user's query, given what has already been done this run.

Available tools:
{tool_descriptions}

Rules:
- Choose exactly ONE next step.
- Use retrieval for document questions.
- Use web_search for current/external info (only if it's in the tool list above).
- Use summarization for "summarize <doc_id>" requests.
- Use diagnose for image classification queries.
- Use vision_qa for questions about figures/charts in documents.
- Choose "synthesize" as soon as you have enough to answer — never call more tools than the question needs.
- Also choose "synthesize" if the last step failed or came back empty and repeating it wouldn't help.
- Only use tools available in the tool list above.

{observations}

Output ONLY valid JSON matching:
{step_schema}"""

_SYNTHESIZE_TOOL_DESC = (
    "synthesize: Combine everything gathered so far into a final grounded "
    "answer. Takes no args — choose this once no further tool call would help."
)


@dataclass
class PlanStep:
    tool: str
    args: dict[str, Any]
    reason: str


@dataclass
class Observation:
    """One completed step from this run's ReAct loop, fed back into the
    next plan_next() call so later steps can react to what earlier ones
    returned. summary is tool_registry._summarize_output(result.data) on
    success (the same shape-not-content trimming the tool_invocation log
    event already uses — bounded size, no raw chunk text bloating the
    prompt) or the tool's error string on failure."""

    step: PlanStep
    success: bool
    summary: Any


class PlanningAgent:
    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry):
        self._llm = llm_client
        self._tools = tool_registry

    async def plan_next(
        self,
        query: str,
        history: list[dict[str, Any]] | None,
        observations: list[Observation],
    ) -> PlanStep:
        """Decide the next single step. Never raises: an LLM failure or an
        unparseable/invalid response degrades to a safe fallback step (see
        _fallback_step) — one bad planning call must not abort the whole
        agent run."""
        prompt = self._prompt_for(query, history, observations)
        try:
            raw = self._llm.generate_structured(prompt)
        except Exception as exc:
            logger.warning(
                "planner_llm_failed",
                extra={
                    "extra_fields": {
                        "query_length": len(query),
                        "error": str(exc),
                        "step_index": len(observations),
                    }
                },
            )
            return self._fallback_step(query, observations)
        return self._parse_step(raw, query, observations)

    def _prompt_for(
        self, query: str, history: list[dict[str, Any]] | None, observations: list[Observation]
    ) -> str:
        prompt = _NEXT_STEP_PROMPT_TEMPLATE.format(
            tool_descriptions=self._format_tool_descriptions(),
            observations=self._format_observations(observations),
            step_schema=_STEP_SCHEMA,
        )
        if history:
            hist = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:])
            prompt += f"\n\nConversation history:\n{hist}"
        prompt += f"\n\nQuery: {query}"
        return prompt

    def _format_tool_descriptions(self) -> str:
        specs = self._tools.list()
        lines = []
        for spec in specs:
            args_json = json.dumps(spec.args_schema, indent=2)
            lines.append(f"- {spec.name}: {spec.description}\n  Args schema: {args_json}")
        lines.append(f"\n- synthesize: {_SYNTHESIZE_TOOL_DESC}")
        return "\n".join(lines)

    def _format_observations(self, observations: list[Observation]) -> str:
        if not observations:
            return "Nothing has been done yet — this is the first step."
        lines = [
            f"- Step {i}: {obs.step.tool}({obs.step.args}) -> "
            f"{'succeeded' if obs.success else 'failed'}: {obs.summary}"
            for i, obs in enumerate(observations, start=1)
        ]
        return "Steps already taken this run:\n" + "\n".join(lines)

    def _parse_step(self, raw: str, query: str, observations: list[Observation]) -> PlanStep:
        if not raw or not raw.strip():
            logger.warning(
                "planner_empty_response", extra={"extra_fields": {"query_length": len(query)}}
            )
            return self._fallback_step(query, observations)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                logger.warning(
                    "planner_no_json_object", extra={"extra_fields": {"query_length": len(query)}}
                )
                return self._fallback_step(query, observations)
            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                logger.warning(
                    "planner_json_parse_failed",
                    extra={"extra_fields": {"query_length": len(query)}},
                )
                return self._fallback_step(query, observations)

        if not isinstance(payload, dict):
            logger.warning("planner_not_dict", extra={"extra_fields": {"query_length": len(query)}})
            return self._fallback_step(query, observations)

        tool = str(payload.get("tool", "")).strip()
        valid_tools = {t.name for t in self._tools.list()}
        valid_tools.add("synthesize")
        if tool not in valid_tools:
            logger.warning(
                "planner_unknown_tool",
                extra={"extra_fields": {"tool": tool, "query_length": len(query)}},
            )
            return self._fallback_step(query, observations)

        args = payload.get("args", {})
        if not isinstance(args, dict):
            args = {}
        reason = str(payload.get("reason", "")).strip()

        logger.info(
            "planner_step_decided",
            extra={
                "extra_fields": {
                    "tool": tool,
                    "step_index": len(observations),
                    "query_length": len(query),
                }
            },
        )
        return PlanStep(tool=tool, args=args, reason=reason)

    def _fallback_step(self, query: str, observations: list[Observation]) -> PlanStep:
        """Safe degrade when the planner itself fails or returns garbage:
        retrieval on the very first step (the same default the old
        deterministic keyword planner used), synthesize on any later step
        — never retry a broken planning call indefinitely, and never
        repeat a tool call blindly without the LLM actually choosing to."""
        if not observations:
            logger.info(
                "planner_using_fallback_retrieval",
                extra={"extra_fields": {"query_length": len(query)}},
            )
            return PlanStep(
                tool="retrieval", args={"query": query}, reason="fallback: direct retrieval"
            )
        logger.info(
            "planner_using_fallback_synthesize",
            extra={"extra_fields": {"query_length": len(query), "steps_so_far": len(observations)}},
        )
        return PlanStep(
            tool="synthesize", args={}, reason="fallback: synthesizing from what's gathered so far"
        )
