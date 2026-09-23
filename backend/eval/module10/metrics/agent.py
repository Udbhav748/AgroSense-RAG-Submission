"""Agent-specific metrics: task success, planning, workflow/node health.

Reuses:
- run_eval.py's step_efficiency()/EXPECTED_MIN_STEPS for Step Efficiency.
- app.core.metrics.get_metrics().agent_workflow_summary() for Workflow
  Completion Rate, Node Success Rate, Average Node Latency, Loop Rate,
  Average Steps — the real, live Phase-1 registry, not re-derived.

New in this module: task_success_rate() with an explicit, pluggable
success-criteria function (rather than run_eval.py's fixed keyword-only
check) so agent_eval.json's richer cases (fact-checks, action-checks) can
define success per PDF section 21's examples; generic
tool_argument_accuracy() generalizing run_eval.py's summarize-only
document_id check to retrieve/summarize/web_research per PDF section 18
(diagnose is explicitly out of scope for this text-only harness — see
agent_eval.json's _schema_note); planning_success_rate() for multi-step
cases (PDF section 19); completion_time() derived from ChatResponse's own
processing_time field (a real, already-tracked value — not a new metric
invented from nothing).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.metrics import get_metrics
from eval.run_eval import EXPECTED_MIN_STEPS, step_efficiency

__all__ = [
    "CostCase",
    "TaskCase",
    "ToolArgCase",
    "agent_workflow_metrics",
    "completion_time_stats",
    "cost_per_successful_task",
    "planning_success_rate",
    "step_efficiency",
    "task_success_rate",
    "tool_argument_accuracy",
]


@dataclass
class TaskCase:
    case_id: str
    success: bool
    reason: str = ""


def task_success_rate(cases: list[TaskCase]) -> dict:
    if not cases:
        return {"task_success_rate": None, "n": 0}
    successes = sum(1 for c in cases if c.success)
    return {
        "task_success_rate": round(successes / len(cases), 4),
        "n": len(cases),
        "failures": [{"case_id": c.case_id, "reason": c.reason} for c in cases if not c.success],
    }


@dataclass
class ToolArgCase:
    case_id: str
    tool: str
    expected: object
    actual: object
    applicable: bool = True
    note: str = ""


def tool_argument_accuracy(cases: list[ToolArgCase]) -> dict:
    """Generic tool-argument accuracy across tools, not just summarize's
    document_id (the one existing, narrow check in run_eval.py). Cases
    marked applicable=False (e.g. diagnose, or retrieve's caller-supplied
    top_k — see agent_eval.json's tool_argument_cases) are excluded from
    the accuracy denominator and reported separately with their reason,
    per PDF section 18's "document why instead of fabricating a result."
    """
    applicable = [c for c in cases if c.applicable]
    not_applicable = [c for c in cases if not c.applicable]
    per_tool: dict[str, list[ToolArgCase]] = {}
    for c in applicable:
        per_tool.setdefault(c.tool, []).append(c)

    per_tool_accuracy = {
        tool: round(sum(1 for c in group if c.actual == c.expected) / len(group), 4)
        for tool, group in per_tool.items()
    }
    overall = (
        round(sum(1 for c in applicable if c.actual == c.expected) / len(applicable), 4)
        if applicable
        else None
    )
    return {
        "tool_argument_accuracy": overall,
        "n_applicable": len(applicable),
        "per_tool": per_tool_accuracy,
        "not_applicable": [
            {"case_id": c.case_id, "tool": c.tool, "reason": c.note} for c in not_applicable
        ],
        "mismatches": [
            {"case_id": c.case_id, "tool": c.tool, "expected": c.expected, "actual": c.actual}
            for c in applicable
            if c.actual != c.expected
        ],
    }


@dataclass
class PlanCase:
    case_id: str
    expected_steps: list[str]
    actual_steps: list[str]


def planning_success_rate(cases: list[PlanCase]) -> dict:
    """A plan 'succeeds' if the actual node sequence contains every
    expected node in the expected relative order (subsequence match, not
    exact-list-equality) — a workflow that also runs e.g. cache_lookup or
    an extra reflection pass shouldn't be marked wrong for including a
    real, correct extra step; it should be marked wrong for MISSING or
    MIS-ORDERING an expected one.
    """
    if not cases:
        return {"planning_success_rate": None, "n": 0}

    def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
        it = iter(actual)
        return all(step in it for step in expected)

    results = [
        {"case_id": c.case_id, "success": _is_subsequence(c.expected_steps, c.actual_steps), "actual_steps": c.actual_steps}
        for c in cases
    ]
    successes = sum(1 for r in results if r["success"])
    return {
        "planning_success_rate": round(successes / len(cases), 4),
        "n": len(cases),
        "per_case": results,
    }


def completion_time_stats(processing_times_seconds: list[float]) -> dict:
    """Average/P50/P95 completion time from ChatResponse.processing_time
    values collected during an eval run — the PDF's 'Completion Time'
    metric, which this project previously only had as raw per-request log
    lines (processing_duration), never aggregated. Not fabricated: computed
    only over samples actually observed in this run."""
    if not processing_times_seconds:
        return {"average_completion_time_s": None, "p50_s": None, "p95_s": None, "n": 0}
    ordered = sorted(processing_times_seconds)
    n = len(ordered)

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
        return ordered[idx]

    return {
        "average_completion_time_s": round(sum(ordered) / n, 4),
        "p50_s": round(_pct(50), 4),
        "p95_s": round(_pct(95), 4),
        "n": n,
    }


@dataclass
class CostCase:
    case_id: str
    success: bool
    cost_usd: float | None  # None = cost genuinely unavailable for this case — never coerce to 0.0


def cost_per_successful_task(cases: list[CostCase]) -> dict:
    """Cost Per Successful Task = total cost of successful cases / count of
    successful cases. A case with cost_usd=None (e.g. it errored before any
    LLM call, or the run used a non-metered path) is excluded from both the
    numerator and denominator and reported separately under
    `cases_with_unavailable_cost` — never silently treated as $0, which
    would understate true cost per task.
    """
    if not cases:
        return {"cost_per_successful_task_usd": None, "n_successful": 0, "n_total": 0}

    successful = [c for c in cases if c.success]
    successful_with_cost = [c for c in successful if c.cost_usd is not None]
    successful_unavailable = [c.case_id for c in successful if c.cost_usd is None]

    total_cost = sum(c.cost_usd for c in successful_with_cost)
    cost_per_task = round(total_cost / len(successful_with_cost), 6) if successful_with_cost else None

    return {
        "cost_per_successful_task_usd": cost_per_task,
        "n_successful": len(successful),
        "n_successful_with_measured_cost": len(successful_with_cost),
        "n_total": len(cases),
        "total_cost_usd_over_measured_successes": round(total_cost, 6) if successful_with_cost else None,
        "cases_with_unavailable_cost": successful_unavailable,
        "note": "Only successful cases with a real, measured estimated_cost_usd (captured via "
        "telemetry_capture.py from the chat_query_handled log line) contribute to the average. "
        "A case whose cost could not be measured is listed, never assumed to cost $0.",
    }


def agent_workflow_metrics() -> dict:
    """Workflow Completion Rate / Node Success Rate / Average Node Latency
    / Loop Rate / Average Steps — read live from the real Phase-1
    Prometheus-style registry (core/metrics.py), not recomputed. Caller is
    responsible for reset_metrics() before the run it wants isolated
    numbers for (see run_agent_eval.py)."""
    return get_metrics().agent_workflow_summary()
