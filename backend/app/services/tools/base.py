"""Core abstractions for the dynamic tool registry.

The registry (tools/registry.py) is the agent-facing way to invoke a tool
by name with validated arguments — the piece the Router/Planner/Executor
agents call instead of importing a service function directly. These are
the shared types everything in tools/ builds on:

- BaseTool:    the interface every tool implements (name, description,
               Pydantic args schema, declared output type, execute()).
- ToolContext: injected collaborators a tool needs at run time
               (vector_store, llm_client) plus auth/session context —
               the registry hands this in, tools never construct it.
- ToolResult:  the uniform envelope a tool returns; success/data/error so
               an agent can branch on outcome instead of catching.
- ToolSpec:    registry metadata (name, description, JSON schemas) used by
               prompt_builder-style model-facing tool lists and the eval
               harness.

Deliberately thin — no logic here, just contracts. Tools live in
tools/implementations.py, constructed in tools/factory.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.core.config import settings as default_settings

if TYPE_CHECKING:
    from app.services.llm_client import LLMClient
    from app.services.vector_store import VectorStore


@dataclass
class ToolContext:
    """Dependencies injected into every tool execution by the registry.

    vector_store/llm_client are the same singletons ChatService holds
    (get_vector_store / get_llm_client in query.py). tenant_id and
    session_id are auth/session context threaded from the request so a
    tool can scope its work the same way the rest of the pipeline does.
    """

    vector_store: VectorStore
    llm_client: LLMClient
    tenant_id: int | None = None
    session_id: str | None = None
    settings: Any = field(default_factory=lambda: default_settings)
    # Whether the client explicitly confirmed this request's web search
    # (confirm_web_search=true on /chat). The human-approval gate —
    # Settings.web_search_requires_approval — is enforced by tools that
    # reach the network (WebSearchTool), which read this field the same
    # way the inline path reads ChatRequest.confirm_web_search. Threaded
    # from the request into the tool context by the executor (see
    # agent_executor.py).
    confirm_web_search: bool = False
    # Optional session-scoped agent memory (services/agent_memory.py) —
    # carried here so tools/agent context can draw on remembered facts
    # without tools/ depending on that module. Typed loosely: memory is
    # an in-progress, failure-safe enhancement; a concrete type import
    # would couple the registry to it.
    agent_memory: Any = None


class ToolResult(BaseModel):
    """Uniform envelope every tool returns.

    An agent branches on `success` rather than catching — expected
    failures (no vector store, web search disabled, page unreadable) are
    ToolResult(success=False, error=...) from the tool, and the registry
    only raises for contract violations (bad args, bad output shape).
    """

    success: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class ToolSpec(BaseModel):
    """Registry-visible metadata for one tool: what it's called, what it
    does, and the JSON schemas for its arguments and return value."""

    name: str
    description: str
    args_schema: dict[str, Any]
    output_schema: dict[str, Any]


# Bound to BaseModel and shared by every tool's declared args_schema/execute
# pair. Generic on T_Args (rather than a fixed `args: BaseModel` parameter)
# so each concrete tool can narrow to its own Pydantic input type without
# violating Liskov substitution: RetrievalTool.execute(args: RetrievalInput,
# ...) satisfies BaseTool[RetrievalInput], and callers that only hold a
# BaseTool[Any] (the registry's storage type) still type-check because Any
# is compatible in both variance directions.
T_Args = TypeVar("T_Args", bound=BaseModel)


@runtime_checkable
class BaseTool(Protocol[T_Args]):
    """The tool interface. A tool is a named capability with a declared
    Pydantic args schema and a declared output type; execute() validates
    its work against those and returns a ToolResult.

    runtime_checkable so isinstance checks work against structurally
    conforming classes without a shared base.
    """

    name: str
    description: str
    args_schema: type[T_Args]
    # Not `type`: some tools declare a generic alias here (e.g.
    # tuple[str, list[RetrievedChunk]]), which isn't itself an instance of
    # `type` — Any is the accurate annotation for "arbitrary type
    # expression used for introspection only", same rationale as
    # ToolContext.settings/agent_memory above.
    output_schema: Any

    async def execute(self, args: T_Args, context: ToolContext) -> ToolResult: ...
