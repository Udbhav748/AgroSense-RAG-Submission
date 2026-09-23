"""LocalResearchAgent: the same plan-decompose-search-synthesize shape
as ResearchAgent (research_agent.py), pointed at this app's own
document retrieval instead of the web. Used when a query is complex
enough that one retrieve() call may only surface part of the answer
(e.g. a question comparing two sections of a document)."""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.core.config import settings
from app.models.document import RetrievedChunk
from app.services.llm_client import LLMClient
from app.services.prompt_builder import build_prompt, strip_sources_section
from app.services.query_planning import plan_subqueries
from app.services.retrieval_service import retrieve
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class LocalResearchFindings:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    answer: str = ""


class LocalResearchAgent:
    def __init__(self, llm_client: LLMClient, vector_store: VectorStore):
        self._llm_client = llm_client
        self._vector_store = vector_store

    def run(self, query: str, tenant_id: int | None = None) -> LocalResearchFindings:
        queries = plan_subqueries(self._llm_client, query, settings.local_research_max_subqueries)
        chunks_by_id: dict[str, RetrievedChunk] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
            futures = [
                pool.submit(retrieve, q, self._vector_store, tenant_id=tenant_id) for q in queries
            ]
            for future in futures:
                try:
                    for chunk in future.result():
                        chunks_by_id[chunk.chunk_id] = chunk
                except Exception:
                    logger.warning("local_research_subquery_failed", extra={"extra_fields": {}})
        merged_chunks = list(chunks_by_id.values())
        if not merged_chunks:
            return LocalResearchFindings(queries=queries)
        prompt = build_prompt(query, merged_chunks)
        try:
            answer = strip_sources_section(self._llm_client.generate(prompt)).strip()
        except Exception:
            answer = ""
        return LocalResearchFindings(chunks=merged_chunks, queries=queries, answer=answer)
