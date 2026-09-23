"""The dynamic ToolRegistry: register tools by name, look them up, and
execute them with validated arguments under one instrumentation point.

This is the agent-facing invocation surface. Where the inline ChatService
path calls service functions directly (each already wrapped by
tool_registry.py's @track_tool), an agent needs to call *by name* with a
dict of args it got from the LLM planner — no import, no hardcoded
signature. The registry is that layer:

    registry = build_tool_registry()
    result = await registry.execute("retrieval", {"query": "..."}, context)

Contract handling mirrors tool_registry.py's @track_tool philosophy:
- args are validated against the tool's Pydantic args_schema up front;
  a violation raises ToolInputError (422, caller's fault).
- a returned ToolResult whose `data` fails the tool's declared output
  type raises ToolOutputError (500, the tool itself is broken).
- every call emits one structured "tool_invocation" event and bumps the
  tool metrics (success/error + latency) — the single source for Tool
  Success Rate on the agent path.

Expected tool failures (web search disabled, empty result set, page
unreadable) are NOT raised — the tool returns ToolResult(success=False,
error=...) and the agent decides what to do with it, so a flaky external
dependency never takes down a multi-step agent run.

ToolInputError/ToolOutputError are reused from tool_registry.py rather
than redefined — one source of truth for the tool contract errors.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.core.exceptions import ChatServiceError
from app.core.metrics import get_metrics
from app.services.tool_registry import (
    ToolInputError,
    ToolOutputError,
    _summarize_input,
    _summarize_output,
)
from app.services.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ToolNotFoundError(ChatServiceError):
    """Raised when registry.execute() names a tool that isn't registered."""

    status_code = 404
    error_code = "TOOL_NOT_FOUND"

    def __init__(self, tool: str):
        super().__init__(f"Tool '{tool}' not found in registry")


class ToolRegistry:
    """Name-keyed collection of BaseTool implementations.

    Thread-safety: register() is only called at construction time
    (factory.py), execute() only reads self._tools — no locking needed.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        self._output_adapters: dict[str, TypeAdapter[Any]] = {}

    def register(self, tool: BaseTool[Any]) -> None:
        """Register a tool, building its output TypeAdapter once so
        execute()'s output validation is a cheap schema check, not a
        per-call schema build."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        self._output_adapters[tool.name] = TypeAdapter(tool.output_schema)

    def get(self, name: str) -> BaseTool[Any]:
        """Return the tool registered under `name`. Raises
        ToolNotFoundError if none."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list(self) -> list[ToolSpec]:
        """Registry-visible metadata for every tool, ready to hand to a
        model-facing tool list (prompt_builder-style) or the eval
        harness. Sorted by name for stable output."""
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema.model_json_schema(),
                output_schema=self._output_adapters[name].json_schema(),
            )
            for name, tool in sorted(self._tools.items())
        ]

    async def execute(self, name: str, args: dict[str, Any], context: ToolContext) -> ToolResult:
        """Validate args, run the tool, validate the result, and log one
        tool_invocation event. Returns a ToolResult on success; raises
        ToolNotFoundError / ToolInputError / ToolOutputError on contract
        violations. Expected tool failures come back as
        ToolResult(success=False, error=...)."""
        tool = self.get(name)
        start = time.perf_counter()

        # Validate input against the tool's args schema before running.
        try:
            validated_args = tool.args_schema.model_validate(args)
        except ValidationError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            get_metrics().record_tool_invocation(tool=name, success=False, latency_ms=latency_ms)
            logger.warning(
                "tool_invocation",
                extra={
                    "extra_fields": {
                        "tool": name,
                        "success": False,
                        "error_type": "ToolInputError",
                        "error_message": str(exc)[:200],
                        "input": _summarize_input(args),
                        "latency_ms": round(latency_ms, 2),
                    }
                },
            )
            raise ToolInputError(name, str(exc)) from exc

        try:
            result = await tool.execute(validated_args, context)
        except Exception as exc:
            # An unexpected tool crash — surface as a failed ToolResult so
            # the agent can decide, and log it distinctly (a bare non-AppError
            # escaping a tool is a bug worth seeing in logs).
            latency_ms = (time.perf_counter() - start) * 1000
            get_metrics().record_tool_invocation(tool=name, success=False, latency_ms=latency_ms)
            logger.warning(
                "tool_invocation",
                extra={
                    "extra_fields": {
                        "tool": name,
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:200],
                        "input": _summarize_input(validated_args.model_dump()),
                        "latency_ms": round(latency_ms, 2),
                    }
                },
            )
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

        # Validate a successful result against the tool's declared output
        # type; a violated contract is the tool's bug (500), not the
        # agent's.
        if result.success:
            try:
                self._output_adapters[name].validate_python(result.data)
            except ValidationError as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                get_metrics().record_tool_invocation(
                    tool=name, success=False, latency_ms=latency_ms
                )
                logger.warning(
                    "tool_invocation",
                    extra={
                        "extra_fields": {
                            "tool": name,
                            "success": False,
                            "error_type": "ToolOutputError",
                            "error_message": str(exc)[:200],
                            "input": _summarize_input(validated_args.model_dump()),
                            "latency_ms": round(latency_ms, 2),
                        }
                    },
                )
                raise ToolOutputError(name, str(exc)) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        get_metrics().record_tool_invocation(
            tool=name, success=result.success, latency_ms=latency_ms
        )
        logger.info(
            "tool_invocation",
            extra={
                "extra_fields": {
                    "tool": name,
                    "success": result.success,
                    "error": result.error,
                    "input": _summarize_input(validated_args.model_dump()),
                    "output": _summarize_output(result.data),
                    "latency_ms": round(latency_ms, 2),
                }
            },
        )

        result.latency_ms = latency_ms
        return result
