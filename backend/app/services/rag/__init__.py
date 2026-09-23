"""Modular RAG package for AgroSense-RAG.

Provides:
- router: extract_crop_context — crop-name extraction shared with
  rag_service.py (imported there as _extract_crop_context). This package
  used to also hold a full duplicate query-planning/reflection/streaming
  implementation (retrieval_grader.py, reflection_engine.py,
  stream_adapter.py, and most of router.py) that was never wired into any
  production code path — rag_service.py always used its own equivalents.
  Removed once Phase 1's agent-graph audit confirmed zero production
  callers (see git history).
"""

from app.services.rag.router import extract_crop_context

__all__ = [
    "extract_crop_context",
]
