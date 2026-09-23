"""Combines every module10 runner's output into the full_audit_<ts>.json
schema. Pure aggregation — no metric math lives here, only assembly.
"""

from __future__ import annotations

from eval.module10.config import run_metadata


def build_full_audit_report(
    *,
    rag_report: dict | None,
    agent_report: dict | None,
    security_report: dict | None,
    memory_report: dict | None,
    failure_report: dict | None,
    regression_report: dict | None = None,
) -> dict:
    sections = {
        "rag": rag_report,
        "agent": agent_report,
        "security": security_report,
        "memory": memory_report,
        "failure": failure_report,
        "regression": regression_report,
    }
    total_samples = sum(
        (section or {}).get("metadata", {}).get("sample_count", 0) for section in sections.values()
    )
    return {
        "metadata": run_metadata(sample_count=total_samples),
        **sections,
    }
