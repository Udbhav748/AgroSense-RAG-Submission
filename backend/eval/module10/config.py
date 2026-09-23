"""Shared configuration for the Module 10 evaluation package.

Every module10 runner imports from here rather than hardcoding paths,
K values, or version strings, so the recorded metadata in every result
JSON (dataset_version, evaluation_version, K, ...) is guaranteed
consistent across runners instead of drifting per-script.

This package deliberately does NOT reimplement backend/eval/run_eval.py's
metric math (confusion_matrix, classification_report, precision_at_k,
recall_at_k, reciprocal_rank, hit_at_k, citation_supported, is_grounded,
judge_groundedness, check_response_schema) or eval/pii_recall_check.py /
eval/unauthorized_access_check.py's core checks — module10/metrics/*.py
and module10/runners/*.py import and call those directly. See
eval/module10/README.md for the full reuse map.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MODULE10_DIR = Path(__file__).resolve().parent
EVAL_DIR = MODULE10_DIR.parent
BACKEND_DIR = EVAL_DIR.parent

DATASETS_DIR = MODULE10_DIR / "datasets"
REPORTS_DIR = MODULE10_DIR / "reports"
EVIDENCE_DIR = MODULE10_DIR / "evidence"

DATASET_VERSION = "module10_v1"
EVALUATION_VERSION = "module10_eval_v1"

# Fixed K for every retrieval metric in this package (P@K/Recall@K/Hit@K),
# matching backend/eval/run_eval.py's own existing K=5 convention — not
# re-justified per-metric, kept identical so the two evaluators' numbers
# are directly comparable.
DEFAULT_K = 5


def git_commit() -> str:
    """Short commit hash of the current checkout, or 'unknown' outside a
    git repo / on any git failure — never fabricated, never blank."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def utc_timestamp() -> str:
    """UTC timestamp matching run_eval.py's own filename convention
    (YYYYMMDDTHHMMSSZ), for a result file name and for the metadata
    block inside it."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_metadata(*, sample_count: int, dataset_version: str | None = None) -> dict:
    """The reproducibility metadata block every module10 result JSON must
    carry (PDF section 6/32): timestamp, git commit, dataset/evaluation
    version, python version, and model/provider/feature-flag configuration
    — filled in from the live app.core.config.settings singleton, not
    hardcoded, so it reflects whatever .env this run actually used.
    """
    from app.core.config import settings

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "dataset_version": dataset_version or DATASET_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "python_version": platform.python_version(),
        "model_provider": settings.llm_provider,
        "model_name": (
            settings.gemini_model_name
            if settings.llm_provider == "gemini"
            else settings.groq_model_name
        ),
        "fallback_llm_provider": settings.fallback_llm_provider,
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranking_model_name,
        "configuration": {
            "hybrid_search_enabled": settings.hybrid_search_enabled,
            "reranking_enabled": settings.reranking_enabled,
            "web_search_enabled": settings.web_search_enabled,
            "vision_qa_enabled": settings.vision_qa_enabled,
            "structured_output_enabled": settings.structured_output_enabled,
            "retrieval_grade_threshold": settings.retrieval_grade_threshold,
            "top_k": DEFAULT_K,
        },
        "sample_count": sample_count,
    }


def save_report(report: dict, *, name: str) -> Path:
    """Write a timestamped result artifact to reports/ — never overwrites
    a prior result (PDF section 6)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = utc_timestamp()
    path = REPORTS_DIR / f"{name}_{ts}.json"
    import json

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def load_dataset(filename: str) -> list[dict]:
    import json

    path = DATASETS_DIR / filename
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
