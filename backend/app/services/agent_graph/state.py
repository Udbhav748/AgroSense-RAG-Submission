"""State definitions for graph-based multi-agent workflows.

Provides typed, serializable AgentState capturing all data flowing through
the StateGraph runtime: query, plan, retrieval chunks, web results,
intermediate drafts, fact-check outcomes, reflection counts, and final responses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Real (not TYPE_CHECKING-only) imports: these are Pydantic field types below,
# and Pydantic must resolve them to actual classes at model-build time, not
# just at static-analysis time. ruff's TC001 can't tell that apart from an
# ordinary type-hint-only import, so it's suppressed per-line below.
from app.models.document import RetrievedChunk, WebSearchResult  # noqa: TC001
from app.models.schemas import ChatResponse, DiagnosisInfo  # noqa: TC001


class AgentState(BaseModel):
    """The central state container passed between nodes in the StateGraph runtime.

    Pure Python, typed Pydantic model supporting state immutability,
    step snapshotting, and serialized checkpointing.
    """

    # Core user input and intent plan
    query: str = Field(..., description="The user's natural-language query or prompt.")
    plan: dict[str, Any] | str | None = Field(
        default=None,
        description="Current workflow plan or routed action decided by the planner node.",
    )

    # Retrieval and research artifacts
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Document chunks retrieved from local vector store / BM25 index.",
    )
    web_results: list[WebSearchResult] = Field(
        default_factory=list,
        description="External search results retrieved from web research passes.",
    )
    diagnosis: DiagnosisInfo | None = Field(
        default=None,
        description="Optional image diagnosis information for multimodal queries.",
    )

    # Generation and verification artifacts
    draft_answer: str = Field(
        default="",
        description="Intermediate synthesized answer before verification/reflection.",
    )
    fact_check_result: dict[str, Any] | bool | None = Field(
        default=None,
        description="Outcome of citation and claim verification against ground-truth chunks.",
    )

    # Execution telemetry and control
    steps_taken: int = Field(
        default=0,
        description="Number of node executions completed in this graph run.",
    )
    reflection_count: int = Field(
        default=0,
        description="Number of corrective reflection loops executed so far.",
    )
    history: list[dict[str, Any]] | None = Field(
        default=None,
        description="Recent conversation history turns for multi-turn context.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if a node failed during execution.",
    )
    final_response: ChatResponse | None = Field(
        default=None,
        description="The final validated ChatResponse payload delivered to the client.",
    )

    # Contextual metadata and configuration flags
    document_id: str | None = Field(
        default=None,
        description="Specific document UUID for document-targeted actions (e.g. summarization).",
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier for state persistence and memory integration.",
    )
    tenant_id: int | None = Field(
        default=None,
        description="Tenant ID for multi-tenant data isolation.",
    )
    confirm_web_search: bool = Field(
        default=False,
        description="Human approval flag for external web search side-effects.",
    )
    persona: str | None = Field(
        default=None,
        description="Optional tone/style preset (e.g. 'concise', 'eli5').",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional node-specific metadata or intermediate flags.",
    )

    # ------------------------------------------------------------------
    # Phase 1 additions: explicit tracing/workflow/approval fields required
    # by the production workflow spec. All optional/defaulted so existing
    # construction sites (old agent_graph/nodes.py, its tests) are
    # unaffected — this is a superset of the field set above, not a
    # replacement of it.
    # ------------------------------------------------------------------

    # Identity / correlation
    request_id: str | None = Field(
        default=None, description="Unique ID for this single workflow execution."
    )
    trace_id: str | None = Field(
        default=None, description="Correlation ID shared across all node trace log lines."
    )
    tenant_id_str: str | None = Field(
        default=None,
        description="String-form tenant identity where the numeric tenant_id above isn't set "
        "(e.g. auth-derived tenant slugs).",
    )

    # Planner / intent (explicit mirrors of the existing `plan` blob, so
    # routing functions can pattern-match on typed fields instead of
    # re-parsing `plan`)
    intent: str | None = Field(
        default=None, description="Planner-decided intent, e.g. 'retrieve', 'diagnose'."
    )
    planned_action: str | None = Field(
        default=None, description="The single action the planner committed to for this request."
    )
    planned_steps: list[str] = Field(
        default_factory=list, description="Ordered list of node names the planner expects to run."
    )
    current_node: str | None = Field(
        default=None, description="Name of the node currently/most-recently executing."
    )
    workflow_status: str = Field(
        default="pending",
        description="pending | running | completed | failed — coarse workflow lifecycle status.",
    )

    # Retrieval request parameters (mirrors handle_query's own parameters,
    # so retrieval_node can build the exact same retrieve() kwargs handle_
    # query did)
    document_ids: list[str] | None = Field(
        default=None, description="Explicit document-id filter for retrieval/caching."
    )
    retrieval_top_k: int | None = Field(default=None)
    retrieval_min_score: float | None = Field(default=None)
    retrieval_query: str | None = Field(
        default=None,
        description="The (possibly contextualized) query actually sent to retrieve() — may "
        "differ from `query` when query_contextualization rewrote a follow-up into a "
        "standalone question. Every other use of the request (generation, citations, caching, "
        "logging) stays on the original `query` field, matching handle_query's own split.",
    )

    # Retrieval / grading detail
    reranked_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks after cross-encoder reranking, when retrieve() performed it "
        "(same list as retrieved_chunks when reranking is disabled/unavailable — this field "
        "does not imply a second, independent reranking pass; see retrieval_node).",
    )
    retrieval_grade: str | None = Field(
        default=None, description="'good' | 'weak' | 'insufficient', from grade_retrieval()."
    )
    retrieval_grade_reason: str | None = Field(
        default=None, description="Why this grade was chosen (top score, chunk count, threshold)."
    )

    # Tool bookkeeping (bounded summaries only — never raw sensitive I/O)
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One bounded entry per tool invocation: tool_name, input_summary, "
        "output_summary, success, error_type, latency_ms, retry_count, timestamp.",
    )
    tool_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Bounded tool result summaries, paired with tool_calls."
    )

    # Memory (explicit lifetime split from the request-scoped fields above)
    conversation_history: list[dict[str, Any]] | None = Field(
        default=None,
        description="Bounded prior-turn history for this session (mirrors `history`).",
    )
    memory_context: str | None = Field(
        default=None,
        description="Rendered fact/summary context from AgentMemory.build_context(), if any.",
    )

    # Generation / validation
    final_answer: str | None = Field(
        default=None, description="The answer text after output validation has passed."
    )
    structured_output: dict[str, Any] | None = Field(
        default=None, description="Parsed structured-answer payload, when structured mode is used."
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Reasons output_validation_node rejected a draft answer."
    )

    # Approval
    approval_required: bool = Field(default=False)
    approval_type: str | None = Field(
        default=None, description="'web_search' | 'document_delete' (see human_approval.py)."
    )
    approval_reason: str | None = Field(default=None)
    approval_payload_reference: str | None = Field(
        default=None, description="approval_id in the ApprovalStore, not the raw payload."
    )
    approval_status: str = Field(
        default="not_required",
        description="not_required | pending | approved | rejected | expired.",
    )

    # Bounded-loop counters
    retry_count: int = Field(default=0)
    reflection_count_v2: int = Field(
        default=0,
        description="Phase 1 alias tracked alongside the existing reflection_count field above "
        "(kept separate rather than repurposing reflection_count, whose semantics the existing "
        "agent_graph/nodes.py fact_checker_node already depends on).",
    )
    loop_count: int = Field(default=0)

    # Telemetry
    node_timings: dict[str, float] = Field(
        default_factory=dict, description="node_name -> latency_ms for the most recent run."
    )
    workflow_start_time: float | None = Field(default=None)
    workflow_end_time: float | None = Field(default=None)
    perf_start: float | None = Field(
        default=None,
        description="time.perf_counter() value at validate_request_node — kept separate from "
        "workflow_start_time (a wall-clock time.time() value used for logging/display) because "
        "ChatService._respond computes processing_time from a perf_counter delta, and the two "
        "clocks aren't comparable.",
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict, description="e.g. {'prompt_tokens': .., 'completion_tokens': ..}."
    )
    estimated_cost_usd: float | None = Field(default=None)

    # Error / termination
    error_type: str | None = Field(
        default=None, description="Existing core/exceptions.py taxonomy category, if any."
    )
    error_message: str | None = Field(default=None)
    root_cause: str | None = Field(
        default=None, description="'unknown' when the system genuinely can't determine one."
    )
    termination_reason: str | None = Field(
        default=None,
        description="success | validation_failure | approval_rejected | retry_exhausted | "
        "loop_limit_reached | tool_failure | model_failure | safe_fallback.",
    )

    # Sources
    source_type: str | None = Field(
        default=None, description="'documents' | 'web' | 'mixed' | 'vision'."
    )
    final_sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Rendered SourceReference-shaped dicts for the response."
    )

    def copy_with(self, **kwargs: Any) -> AgentState:
        """Return a new copy of AgentState with updated fields."""
        return self.model_copy(update=kwargs)

    def is_fact_check_passed(self) -> bool:
        """Check whether fact verification passed."""
        if self.fact_check_result is None:
            return True
        if isinstance(self.fact_check_result, bool):
            return self.fact_check_result
        if isinstance(self.fact_check_result, dict):
            return bool(self.fact_check_result.get("verified", True))
        return True
