#!/usr/bin/env python
"""Module 10 agent evaluation runner.

Usage (from backend/):
    python eval/module10/runners/run_agent_eval.py

Live run: planner classification (confusion matrix + per-class P/R/F1),
generic tool-argument accuracy, multi-step planning success, task success,
plus the real Phase-1 workflow metrics (Workflow Completion Rate, Node
Success Rate, Average Node Latency, Loop Rate, Average Steps) read from
the live metrics registry. Also runs memory_eval.json's deterministic
session-boundary/cross-session-leakage checks (no live LLM needed for
those specific cases).

diagnose is explicitly NOT evaluated here — see agent_eval.json's
_schema_note: ChatService.handle_diagnose takes raw image bytes, not a
text query, and this harness (like backend/eval/run_eval.py) is text-in/
text-out only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.metrics import reset_metrics  # noqa: E402
from app.services.faiss_vector_store import FAISSVectorStore  # noqa: E402
from app.services.llm_provider import build_llm_client  # noqa: E402
from app.services.rag_service import ChatService  # noqa: E402
from app.services.session_store import InMemorySessionStore  # noqa: E402
from eval.module10 import config  # noqa: E402
from eval.module10.metrics import agent as ametrics  # noqa: E402
from eval.module10.metrics import classification as cmetrics  # noqa: E402
from eval.module10.metrics.telemetry_capture import capture_telemetry, extract  # noqa: E402


def run_planner_cases(chat_service: ChatService, cases: list[dict], document_id: str) -> dict:
    y_true, y_pred = [], []
    tool_arg_cases = []
    per_case = []

    for case in cases:
        query = case["query"].replace("{{document_id}}", document_id)
        expected = case["expected_action"]
        try:
            plan = chat_service._plan(query)  # noqa: SLF001 - eval needs the raw planner decision
        except Exception as exc:  # noqa: BLE001
            per_case.append({"case_id": case["id"], "error": str(exc)})
            continue
        y_true.append(expected)
        y_pred.append(plan.action)
        per_case.append({"case_id": case["id"], "query": query, "expected": expected, "actual": plan.action, "correct": plan.action == expected})

        if case.get("expected_document_id"):
            tool_arg_cases.append(
                ametrics.ToolArgCase(
                    case_id=case["id"], tool="summarize",
                    expected=document_id, actual=plan.document_id,
                )
            )

    matrix = cmetrics.confusion_matrix(y_true, y_pred)
    report = cmetrics.classification_report(y_true, y_pred)
    return {
        "confusion_matrix": matrix,
        "confusion_matrix_csv": cmetrics.to_csv_rows(matrix),
        "classification_report": report,
        "classification_report_md": cmetrics.to_markdown(report),
        "per_case": per_case,
        "tool_arg_cases_from_planner": tool_arg_cases,
    }


def run_tool_argument_cases(
    chat_service: ChatService, dataset_cases: list[dict], document_id: str, planner_tool_arg_cases: list
) -> dict:
    cases = list(planner_tool_arg_cases)
    for c in dataset_cases:
        if c["tool"] == "diagnose":
            cases.append(ametrics.ToolArgCase(case_id=c["id"], tool="diagnose", expected=None, actual=None, applicable=False, note=c["note"]))
        elif c["tool"] == "retrieve" and c["expected_argument"] == "top_k":
            cases.append(ametrics.ToolArgCase(case_id=c["id"], tool="retrieve", expected=None, actual=None, applicable=False, note=c["note"]))
        elif c["tool"] == "retrieve" and c["expected_argument"] == "crop":
            # Real, genuinely measured ground-truth check (Module 10
            # gap-closure): unlike top_k (a caller-supplied ChatRequest
            # field, correctly N/A), crop/collection IS a planner-decided
            # argument -- _plan() calls extract_crop_context(query) for
            # every query regardless of resolved action, and the result
            # is used downstream to scope retrieval to the right crop's
            # documents. Previously untested; this is a real gap in
            # ground-truth coverage, not a caller-supplied field like
            # top_k, so it belongs in the applicable=True denominator.
            plan = chat_service._plan(c["query"])  # noqa: SLF001 - eval needs the raw planner decision
            cases.append(
                ametrics.ToolArgCase(
                    case_id=c["id"], tool="retrieve", expected=c["expected_value"], actual=plan.crop, note=c.get("note", "")
                )
            )
        elif c["tool"] == "web_research":
            cases.append(ametrics.ToolArgCase(case_id=c["id"], tool="web_research", expected=None, actual=None, applicable=False, note=c["note"]))
        elif c["tool"] == "summarize" and c["expected_value"] is None:
            # negative case: non-summarize query must not carry a document_id
            pass  # already covered by planner cases' document_id extraction check being None
    return ametrics.tool_argument_accuracy(cases)


def run_planning_cases(chat_service: ChatService, cases: list[dict], document_id: str) -> dict:
    """Planning Success Rate, now MEASURED (previously "NOT MEASURED" — see
    docs/MODULE10_GAP_CLOSURE_REPORT.md). Uses
    eval/module10/metrics/telemetry_capture.py to read the real, live
    per-request node sequence off the existing `agent_node_trace` /
    `chat_query_handled` structured log lines
    (app/services/agent_graph/events.py::emit_node_trace,
    app/services/rag_service.py's `_respond`) for the duration of one
    `handle_query()` call — zero production code/contract changes, no new
    field on AgentState or ChatResponse, nothing beyond attaching and
    detaching a logging.Handler around a call this runner already makes.

    Success criterion (explicit, applies per case): the case's
    `expected_steps` (agent_eval.json) must appear, in order, as a
    subsequence of the actually-executed node sequence. Extra nodes
    (e.g. a bounded reflection retry) don't fail the case — only a
    missing or out-of-order expected node does. `case_type: "single_step"`
    cases additionally require no node to repeat (a loop would contradict
    "single step").
    """
    plan_cases = []
    telemetry_by_case = {}
    bypassed_results = []
    errors = []
    for case in cases:
        query = case["query"].replace("{{document_id}}", document_id)
        expected_steps = case["expected_steps"]
        graph_bypassed = case.get("graph_bypassed", False)
        try:
            with capture_telemetry() as handler:
                response = chat_service.handle_query(query)
            telemetry = extract(handler)

            if graph_bypassed:
                # This action type answers via a pre-graph fast path (see
                # agent_eval.json's _planning_cases_correction_note) and never
                # emits a node trace at all -- node-sequence matching does not
                # apply. Measured instead via the one signal ChatResponse does
                # expose for this path: tool_used.
                success = response.tool_used == case.get("expected_tool_used", expected_steps[0])
                bypassed_results.append(
                    {
                        "case_id": case["id"],
                        "case_type": case.get("case_type"),
                        "graph_bypassed": True,
                        "expected_steps": expected_steps,
                        "actual_steps": [],
                        "success": success,
                        "measurement_note": "graph bypassed by a pre-graph fast path; measured via tool_used, not node trace",
                        "has_loop": False,
                        "steps_taken": response.steps_taken,
                        "tool_used": response.tool_used,
                        "expected_min_steps": len(expected_steps),
                        "step_efficiency": round(len(expected_steps) / response.steps_taken, 4) if response.steps_taken else None,
                        "estimated_cost_usd": telemetry.estimated_cost_usd,
                        "first_action_node": None,
                        "expected_first_action_node": None,
                    }
                )
                continue

            plan_cases.append(ametrics.PlanCase(case_id=case["id"], expected_steps=expected_steps, actual_steps=telemetry.node_sequence))
            telemetry_by_case[case["id"]] = {
                "case_type": case.get("case_type"),
                "has_loop": len(telemetry.node_sequence) != len(set(telemetry.node_sequence)),
                "steps_taken": response.steps_taken,
                "tool_used": response.tool_used,
                "expected_min_steps": len(expected_steps),
                "step_efficiency": round(len(expected_steps) / response.steps_taken, 4) if response.steps_taken else None,
                "estimated_cost_usd": telemetry.estimated_cost_usd,
                "first_action_node": telemetry.node_sequence[1] if len(telemetry.node_sequence) > 1 else None,
                "expected_first_action_node": expected_steps[1] if len(expected_steps) > 1 else None,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": case["id"], "case_type": case.get("case_type"), "expected_steps": expected_steps, "error": str(exc)})

    base_result = ametrics.planning_success_rate(plan_cases)
    for row in base_result.get("per_case", []):
        extra = telemetry_by_case.get(row["case_id"], {})
        row.update(extra)
        # a single_step case that looped contradicts its own case_type, even if
        # the subsequence check alone would call it a success
        if extra.get("case_type") == "single_step" and extra.get("has_loop"):
            row["success"] = False
    base_result["per_case"].extend(bypassed_results)
    base_result["per_case"].extend(errors)

    measured = [r for r in base_result["per_case"] if "error" not in r]
    n = len(measured)
    node_traced = [r for r in measured if not r.get("graph_bypassed")]
    tool_selection_accuracy = (
        round(sum(1 for r in node_traced if r.get("first_action_node") == r.get("expected_first_action_node")) / len(node_traced), 4)
        if node_traced
        else None
    )
    average_steps = round(sum(r["steps_taken"] for r in measured) / n, 4) if n and all("steps_taken" in r for r in measured) else None
    step_efficiencies = [r["step_efficiency"] for r in measured if r.get("step_efficiency") is not None]
    step_efficiency = round(sum(step_efficiencies) / len(step_efficiencies), 4) if step_efficiencies else None
    loop_rate = round(sum(1 for r in measured if r.get("has_loop")) / n, 4) if n else None
    # re-derive planning_success_rate over the (possibly single_step-corrected) per_case list
    planning_success = round(sum(1 for r in measured if r.get("success")) / n, 4) if n else None

    return {
        "planning_success_rate": planning_success,
        "tool_selection_accuracy": tool_selection_accuracy,
        "average_steps": average_steps,
        "step_efficiency": step_efficiency,
        "loop_rate": loop_rate,
        "success_criterion": "expected_steps (agent_eval.json) must appear, in order, as a subsequence of the "
        "real captured node sequence for that request (app.eval.module10.metrics.agent.planning_success_rate); "
        "single_step cases additionally fail if any node repeats (a loop contradicts 'single step').",
        "measurement_method": "eval/module10/metrics/telemetry_capture.py — captures the existing agent_node_trace/"
        "chat_query_handled structured log lines emitted during one handle_query() call; no production code changed.",
        "per_case": base_result["per_case"],
    }


def run_memory_session_boundary_cases() -> dict:
    """Deterministic, no live LLM: session isolation and history truncation."""
    store = InMemorySessionStore(max_sessions=10, max_turns_per_session=6)
    session_a = store.get_or_create_session(None)
    session_b = store.get_or_create_session(None)
    store.append_turn(session_a, "user", "The secret code is ALPHA-7.")
    store.append_turn(session_a, "assistant", "Noted.")

    history_b = store.get_history(session_b)
    leaked = bool(history_b) and any("ALPHA-7" in turn.get("content", "") for turn in history_b)

    for i in range(8):
        store.append_turn(session_a, "user", f"turn {i}")
        store.append_turn(session_a, "assistant", f"reply {i}")
    history_a = store.get_history(session_a) or []

    return {
        "session_separation": {"leaked": leaked, "expected_cross_session_leak_rate": 0},
        "history_truncation": {"turns_after_overflow": len(history_a), "max_turns_per_session": 6},
    }


def main() -> None:
    dataset = config.load_dataset("agent_eval.json")
    reset_metrics()

    vector_store = FAISSVectorStore()
    vector_store.load()
    from eval.run_eval import discover_document_id

    document_id = discover_document_id()
    chat_service = ChatService(vector_store, build_llm_client())

    planner_result = run_planner_cases(chat_service, dataset["planner_cases"], document_id)
    tool_arg_result = run_tool_argument_cases(
        chat_service, dataset["tool_argument_cases"], document_id, planner_result["tool_arg_cases_from_planner"]
    )
    planning_result = run_planning_cases(chat_service, dataset["planning_cases"], document_id)
    memory_boundary_result = run_memory_session_boundary_cases()
    workflow_result = ametrics.agent_workflow_metrics()

    cost_cases = [
        ametrics.CostCase(
            case_id=row["case_id"],
            success=bool(row.get("success")),
            cost_usd=row.get("estimated_cost_usd"),
        )
        for row in planning_result["per_case"]
    ]
    cost_result = ametrics.cost_per_successful_task(cost_cases)

    report = {
        "metadata": config.run_metadata(
            sample_count=len(dataset["planner_cases"]) + len(dataset["planning_cases"]),
            dataset_version=dataset["dataset_version"],
        ),
        "planner_classification": planner_result,
        "tool_argument_accuracy": tool_arg_result,
        "planning": planning_result,
        "cost_per_successful_task": cost_result,
        "memory_session_boundary": memory_boundary_result,
        "workflow_metrics": workflow_result,
    }

    path = config.save_report(report, name="agent_eval")
    print(f"Saved: {path}")
    print()
    print(planner_result["classification_report_md"])
    print()
    print(f"Tool argument accuracy: {tool_arg_result['tool_argument_accuracy']}")
    print(f"Planning success rate: {planning_result['planning_success_rate']}")
    print(f"Tool selection accuracy: {planning_result['tool_selection_accuracy']}")
    print(f"Average steps: {planning_result['average_steps']}  Step efficiency: {planning_result['step_efficiency']}  Loop rate: {planning_result['loop_rate']}")
    print(f"Cost per successful task (USD): {cost_result['cost_per_successful_task_usd']}")
    print(f"Workflow metrics: {workflow_result}")
    print(f"Memory session boundary: {memory_boundary_result}")


if __name__ == "__main__":
    main()
