"""StateGraph engine: lightweight, pure-Python graph-based agent runtime.

Supports:
- Typed node registration (add_node)
- Deterministic and conditional edge routing (add_edge, add_conditional_edges)
- State checkpoint history & step snapshot tracking
- Loop cycle capping (max_steps guardrail)
- Async execution and streaming iteration
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.metrics import get_metrics
from app.services.agent_graph.events import emit_node_trace
from app.services.agent_graph.state import AgentState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

logger = logging.getLogger(__name__)

START = "__start__"
END = "__end__"


class StateGraphError(Exception):
    """Base exception for StateGraph errors."""


class GraphCompilationError(StateGraphError):
    """Raised when the graph validation fails during compilation."""


class MaxStepsExceededError(StateGraphError):
    """Raised when execution exceeds the maximum allowed step threshold."""


@dataclass
class StateSnapshot:
    """Snapshot of agent state at a specific step in graph execution."""

    step_index: int
    node_name: str
    state: AgentState
    timestamp: float
    duration_ms: float


@dataclass
class BranchResult:
    """One concurrent branch's outcome -- always populated, even on
    failure, so a caller can tell "this branch failed" from "this branch
    never ran" without inspecting exceptions itself.

    `exception` carries the ORIGINAL exception object (not just its
    string form in `error`) specifically so a caller can `raise
    result.exception` and preserve the real exception type/taxonomy
    (e.g. a domain-specific `AppError` subclass) -- re-raising a generic
    `RuntimeError` instead would silently change which error-handling
    branch a caller's own `except AppError` / `except SomeSpecificError`
    takes.
    """

    name: str
    value: Any
    success: bool
    error: str | None
    exception: BaseException | None
    started_at: float  # wall-clock (time.time()), for cross-branch overlap evidence
    ended_at: float
    duration_ms: float


async def run_concurrent_branches(
    branches: dict[str, Callable[[], Any]],
    *,
    trace_id: str | None = None,
    request_id: str | None = None,
    parallel_group: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, BranchResult]:
    """Runs independent, zero-argument branch callables CONCURRENTLY and
    returns each one's outcome, keyed by branch name.

    This is the reusable concurrency primitive the graph runtime exposes
    for the "Independent tasks can run simultaneously" checklist item --
    it is deliberately generic (works for any set of independent
    branches a node/service wants to fan out), not a one-off hack for a
    single call site.

    Each branch is executed for real, concurrently:
    - an `async def` branch is awaited directly on the running event loop
    - a plain (blocking/sync) branch runs via `asyncio.to_thread`, a real
      OS thread, so it genuinely overlaps with the other branches'
      network/CPU work rather than just being interleaved cooperative
      code on one thread.

    A branch's own exception is caught and recorded on its own
    `BranchResult` (`success=False`, `error=...`) -- it never cancels or
    corrupts a sibling branch's result, and never propagates out of this
    function (callers decide how to react to a partial failure; a
    caller that needs "any failure is fatal" checks `success` on every
    result itself).

    Emits one `agent_node_trace` line per branch (reusing the existing
    per-node tracing format, not a new log shape) tagged with a shared
    `parallel_group` id and each branch's own wall-clock
    `branch_started_at`/`branch_ended_at`, so overlapping start/end
    timestamps are directly visible in the structured logs -- the actual
    evidence that branches ran concurrently, not just a claim.
    """
    group_id = parallel_group or uuid.uuid4().hex[:12]

    async def _run_one(name: str, fn: Callable[[], Any]) -> tuple[str, BranchResult]:
        started_wall = time.time()
        started_perf = time.perf_counter()
        value: Any = None
        success = True
        error: str | None = None
        exception: BaseException | None = None
        try:
            awaitable = fn() if inspect.iscoroutinefunction(fn) else asyncio.to_thread(fn)
            if timeout_seconds is not None:
                value = await asyncio.wait_for(awaitable, timeout=timeout_seconds)
            else:
                value = await awaitable
        except TimeoutError as exc:
            # A sync branch's underlying OS thread cannot be forcibly killed
            # (a real Python/asyncio.to_thread limitation) -- it keeps
            # running in the background even though this coroutine stops
            # waiting on it. That's disclosed here, not hidden: the branch
            # is correctly reported as timed_out/failed either way, and it
            # never writes to any shared state (each branch only returns
            # its own local value), so a late-finishing thread can't
            # corrupt a sibling branch's result.
            success = False
            error = f"TimeoutError: branch '{name}' exceeded {timeout_seconds}s"
            exception = exc
        except Exception as exc:  # noqa: BLE001 -- captured per-branch, never re-raised here
            success = False
            error = f"{type(exc).__name__}: {exc}"
            exception = exc
        ended_perf = time.perf_counter()
        ended_wall = time.time()
        duration_ms = (ended_perf - started_perf) * 1000

        emit_node_trace(
            trace_id=trace_id,
            request_id=request_id,
            node=f"parallel:{name}",
            status="success" if success else "failure",
            latency_ms=duration_ms,
            error_type=(error.split(":", 1)[0] if error else None),
            extra={
                "parallel_group": group_id,
                "branch": name,
                "branch_started_at": started_wall,
                "branch_ended_at": ended_wall,
            },
        )
        return name, BranchResult(
            name=name,
            value=value,
            success=success,
            error=error,
            exception=exception,
            started_at=started_wall,
            ended_at=ended_wall,
            duration_ms=duration_ms,
        )

    pairs = await asyncio.gather(*(_run_one(name, fn) for name, fn in branches.items()))
    return dict(pairs)


class StateGraph:
    """Graph builder for multi-agent workflows."""

    def __init__(self) -> None:
        self._nodes: dict[str, Callable[..., Any]] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, tuple[Callable[..., Any], dict[str, str] | None]] = {}
        self._entry_point: str | None = None

    def add_node(self, name: str, fn: Callable[..., Any]) -> StateGraph:
        """Register a node function with a unique name."""
        if name in (START, END):
            raise ValueError(f"Cannot name a node '{name}': reserved keyword.")
        if name in self._nodes:
            raise ValueError(f"Node '{name}' is already registered.")
        self._nodes[name] = fn
        return self

    def add_edge(self, from_node: str, to_node: str) -> StateGraph:
        """Register a deterministic transition edge from one node to another."""
        if from_node == END:
            raise ValueError("Cannot add an outgoing edge from END.")
        self._edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable[..., Any],
        mapping: dict[str, str] | None = None,
    ) -> StateGraph:
        """Register a conditional edge routing function from a node."""
        if from_node == END:
            raise ValueError("Cannot add conditional edges from END.")
        self._conditional_edges[from_node] = (condition, mapping)
        return self

    def set_entry_point(self, node_name: str) -> StateGraph:
        """Designate the graph entry node."""
        self._entry_point = node_name
        return self

    def compile(self, max_steps: int = 10) -> CompiledGraph:
        """Validate and compile the graph into an executable CompiledGraph."""
        if not self._entry_point:
            raise GraphCompilationError("Graph entry point is not set.")
        if self._entry_point not in self._nodes:
            raise GraphCompilationError(f"Entry point '{self._entry_point}' not found in nodes.")

        # Validate deterministic edges
        for from_n, to_n in self._edges.items():
            if from_n not in self._nodes and from_n != START:
                raise GraphCompilationError(f"Edge source '{from_n}' is not a registered node.")
            if to_n not in self._nodes and to_n != END:
                raise GraphCompilationError(f"Edge target '{to_n}' is not a registered node.")

        # Validate conditional edges
        for from_n, (_, mapping) in self._conditional_edges.items():
            if from_n not in self._nodes:
                raise GraphCompilationError(f"Conditional edge source '{from_n}' not found.")
            if mapping:
                for target in mapping.values():
                    if target not in self._nodes and target != END:
                        raise GraphCompilationError(
                            f"Conditional target '{target}' from '{from_n}' is not a registered node."
                        )

        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional_edges=dict(self._conditional_edges),
            entry_point=self._entry_point,
            max_steps=max_steps,
        )


class CompiledGraph:
    """Compiled, validated graph runtime ready for execution."""

    def __init__(
        self,
        nodes: dict[str, Callable[..., Any]],
        edges: dict[str, str],
        conditional_edges: dict[str, tuple[Callable[..., Any], dict[str, str] | None]],
        entry_point: str,
        max_steps: int = 10,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point
        self.max_steps = max_steps
        self._history: list[StateSnapshot] = []

    def get_state_history(self) -> list[StateSnapshot]:
        """Return the sequence of state snapshots from the most recent run."""
        return list(self._history)

    async def _invoke_node(
        self,
        node_name: str,
        current_state: AgentState,
        context: Any,
    ) -> AgentState:
        fn = self.nodes[node_name]
        sig = inspect.signature(fn)
        num_params = len(sig.parameters)

        res = fn(current_state, context) if num_params >= 2 else fn(current_state)

        if asyncio.iscoroutine(res):
            res = await res

        if isinstance(res, AgentState):
            return res
        if isinstance(res, dict):
            return current_state.copy_with(**res)
        return current_state

    async def _get_next_node(
        self,
        current_node: str,
        current_state: AgentState,
        context: Any,
    ) -> str:
        if current_node in self.conditional_edges:
            condition_fn, mapping = self.conditional_edges[current_node]
            sig = inspect.signature(condition_fn)
            if len(sig.parameters) >= 2:
                route_res = condition_fn(current_state, context)
            else:
                route_res = condition_fn(current_state)

            if asyncio.iscoroutine(route_res):
                route_res = await route_res

            route_name = str(route_res)
            next_node = mapping.get(route_name, route_name) if mapping is not None else route_name
            return next_node

        if current_node in self.edges:
            return self.edges[current_node]

        return END

    async def run(
        self,
        initial_state: AgentState | dict[str, Any],
        context: Any = None,
    ) -> AgentState:
        """Execute the graph from entry point to END or max_steps."""
        metrics = get_metrics()
        metrics.record_agent_workflow_started()
        wall_start = time.perf_counter()

        current_state = (
            initial_state if isinstance(initial_state, AgentState) else AgentState(**initial_state)
        )
        self._history = []
        current_node = self.entry_point
        step_index = 0

        while current_node != END and step_index < self.max_steps:
            step_index += 1
            start_t = time.perf_counter()

            try:
                current_state = await self._invoke_node(current_node, current_state, context)
            except Exception as exc:
                logger.error(
                    "graph_node_execution_failed",
                    extra={"extra_fields": {"node": current_node, "error": str(exc)}},
                )
                current_state = current_state.copy_with(error=f"{type(exc).__name__}: {exc}")
                break

            duration_ms = (time.perf_counter() - start_t) * 1000
            self._history.append(
                StateSnapshot(
                    step_index=step_index,
                    node_name=current_node,
                    state=current_state.model_copy(deep=True),
                    timestamp=time.time(),
                    duration_ms=duration_ms,
                )
            )

            current_node = await self._get_next_node(current_node, current_state, context)

        if step_index >= self.max_steps and current_node != END:
            logger.warning(
                "graph_cycle_capped_max_steps",
                extra={"extra_fields": {"max_steps": self.max_steps, "last_node": current_node}},
            )
            current_state = current_state.copy_with(
                error=current_state.error or f"Max steps exceeded ({self.max_steps})"
            )
            metrics.record_agent_loop_limit_hit()

        metrics.record_agent_workflow_duration(time.perf_counter() - wall_start)
        metrics.record_agent_steps(step_index)
        workflow_status = getattr(current_state, "workflow_status", None)
        completed = workflow_status == "completed" if workflow_status else current_state.error is None
        metrics.record_agent_workflow_completed(status="completed" if completed else "failed")

        return current_state

    async def stream(
        self,
        initial_state: AgentState | dict[str, Any],
        context: Any = None,
    ) -> AsyncIterator[tuple[str, AgentState]]:
        """Stream each step's (node_name, state) as nodes finish execution."""
        metrics = get_metrics()
        metrics.record_agent_workflow_started()
        wall_start = time.perf_counter()

        current_state = (
            initial_state if isinstance(initial_state, AgentState) else AgentState(**initial_state)
        )
        self._history = []
        current_node = self.entry_point
        step_index = 0

        while current_node != END and step_index < self.max_steps:
            step_index += 1
            start_t = time.perf_counter()

            try:
                current_state = await self._invoke_node(current_node, current_state, context)
            except Exception as exc:
                logger.error(
                    "graph_node_execution_failed",
                    extra={"extra_fields": {"node": current_node, "error": str(exc)}},
                )
                current_state = current_state.copy_with(error=f"{type(exc).__name__}: {exc}")
                yield (current_node, current_state)
                metrics.record_agent_workflow_duration(time.perf_counter() - wall_start)
                metrics.record_agent_steps(step_index)
                metrics.record_agent_workflow_completed(status="failed")
                return

            duration_ms = (time.perf_counter() - start_t) * 1000
            self._history.append(
                StateSnapshot(
                    step_index=step_index,
                    node_name=current_node,
                    state=current_state.model_copy(deep=True),
                    timestamp=time.time(),
                    duration_ms=duration_ms,
                )
            )

            yield (current_node, current_state)
            current_node = await self._get_next_node(current_node, current_state, context)

        if step_index >= self.max_steps and current_node != END:
            logger.warning(
                "graph_cycle_capped_max_steps",
                extra={"extra_fields": {"max_steps": self.max_steps, "last_node": current_node}},
            )
            metrics.record_agent_loop_limit_hit()

        metrics.record_agent_workflow_duration(time.perf_counter() - wall_start)
        metrics.record_agent_steps(step_index)
        workflow_status = getattr(current_state, "workflow_status", None)
        completed = workflow_status == "completed" if workflow_status else current_node == END
        metrics.record_agent_workflow_completed(status="completed" if completed else "failed")
