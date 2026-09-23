"""Planner/tool-selection classification metrics.

Deliberately a thin re-export of backend/eval/run_eval.py's existing,
already-tested confusion-matrix/precision/recall/F1 implementation
(hand-rolled, no scikit-learn dependency in this project — see
run_eval.py:401-440) rather than a second, independently-computed version.
Module 10's requirement (PDF section 16/17: confusion matrix + per-class
P/R/F1 + macro/weighted F1) is already satisfied by that code; this module
exists so module10/runners/run_agent_eval.py has a stable, documented
import path without depending on eval/run_eval.py's module-level script
structure (argparse, live LLM calls, etc.) for pure math.
"""

from __future__ import annotations

from eval.run_eval import ACTIONS, classification_report, confusion_matrix

__all__ = ["ACTIONS", "classification_report", "confusion_matrix", "to_csv_rows", "to_markdown"]


def to_csv_rows(matrix: dict[str, dict[str, int]]) -> list[list[str]]:
    """Confusion matrix as CSV-ready rows (header + one row per actual
    class), for confusion_matrix.csv (PDF section 17)."""
    header = ["actual\\predicted", *ACTIONS]
    rows = [header]
    for actual in ACTIONS:
        rows.append([actual, *[str(matrix[actual][predicted]) for predicted in ACTIONS]])
    return rows


def to_markdown(report: dict) -> str:
    """classification_report.md (PDF section 17): per-class P/R/F1/support
    plus macro/weighted F1 and accuracy, rendered as a markdown table."""
    lines = [
        "| class | precision | recall | f1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in report["per_class"].items():
        lines.append(
            f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['support']} |"
        )
    lines.append("")
    lines.append(f"**Accuracy:** {report['accuracy']:.4f}")
    lines.append(f"**Macro F1:** {report['macro_f1']:.4f}")
    lines.append(f"**Weighted F1:** {report['weighted_f1']:.4f}")
    return "\n".join(lines)
