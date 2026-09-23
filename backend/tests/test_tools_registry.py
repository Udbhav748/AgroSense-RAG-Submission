"""Tests for the dynamic tool registry (services/tools/): BaseTool
registration, name-keyed lookup, args/output validation, the
tool_invocation log event, and that the concrete tools actually execute
against the same fakes the rest of the suite uses.

Uses asyncio.run (Python 3.8+ test runner compatibility) since the suite
doesn't configure an async plugin.
"""

import asyncio

import pytest

from app.core.config import settings
from app.models.document import RetrievedChunk, WebSearchResult
from app.services.tool_registry import ToolInputError, ToolOutputError
from app.services.tools.base import ToolContext
from app.services.tools.factory import build_tool_registry
from app.services.tools.registry import ToolNotFoundError


def run(coro):
    return asyncio.run(coro)


class FakeLLMClient:
    def __init__(self, response="summary text"):
        self.response = response
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


class FakeVectorStore:
    def __init__(self, chunks=None):
        self._chunks = chunks or []

    def search(self, query_vector, top_k, tenant_id=None):
        return self._chunks[:top_k]

    def get_chunks_by_document(self, document_id, tenant_id=None):
        return self._chunks


def make_chunk(chunk_id="c1", document_id="doc-1", score=0.9):
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=document_id, text="chunk text", score=score, metadata={}
    )


def make_context(vector_store=None, llm_client=None):
    return ToolContext(
        vector_store=vector_store or FakeVectorStore(),
        llm_client=llm_client or FakeLLMClient(),
    )


class TestRegistryConstruction:
    def test_lists_all_builtin_tools(self):
        registry = build_tool_registry()
        names = {spec.name for spec in registry.list()}
        assert names == {"retrieval", "web_search", "summarization", "diagnose", "vision_qa"}

    def test_list_specs_carry_json_schemas(self):
        registry = build_tool_registry()
        specs = {spec.name: spec for spec in registry.list()}
        retrieval = specs["retrieval"]
        assert "query" in retrieval.args_schema["properties"]
        assert retrieval.output_schema  # JSON schema for list[RetrievedChunk]

    def test_get_unknown_tool_raises_not_found(self):
        registry = build_tool_registry()
        with pytest.raises(ToolNotFoundError) as exc:
            registry.get("nope")
        assert exc.value.status_code == 404

    def test_duplicate_registration_rejected(self):
        from app.services.tools.implementations import RetrievalTool

        registry = build_tool_registry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(RetrievalTool())


class TestRetrievalTool:
    def _retrieval_context(self, chunks):
        return make_context(vector_store=FakeVectorStore(chunks=chunks))

    def test_executes_against_vector_store(self, monkeypatch):
        registry = build_tool_registry()
        # _retrieve_core falls through to plain semantic search for a
        # non-FAISS store, which would load the real sentence-transformers
        # model — monkeypatch embed_query like test_retrieval_service does.
        monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda q: [0.1, 0.2])
        context = self._retrieval_context([make_chunk("c1", score=0.9)])
        result = run(registry.execute("retrieval", {"query": "question"}, context))
        assert result.success
        assert [chunk.chunk_id for chunk in result.data] == ["c1"]

    def test_returns_empty_on_no_match(self, monkeypatch):
        registry = build_tool_registry()
        monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda q: [0.1, 0.2])
        context = self._retrieval_context([])
        result = run(registry.execute("retrieval", {"query": "nothing here"}, context))
        assert result.success
        assert result.data == []

    def test_blank_query_raises_tool_input_error(self):
        registry = build_tool_registry()
        context = make_context()
        with pytest.raises(ToolInputError) as exc:
            run(registry.execute("retrieval", {"query": ""}, context))
        assert exc.value.status_code == 422


class TestWebSearchTool:
    def test_master_switch_disabled_returns_empty_without_calling_provider(self, monkeypatch):
        """Regression test: WEB_SEARCH_ENABLED=false must hold on the
        AgentExecutor/tool-registry path exactly like it does on the
        inline ChatService path (rag_service._search_web), not just be a
        suggestion in PlanningAgent's prompt. web_search_ready() and
        _search_web_core are mocked to prove real results even so —
        the master switch must short-circuit before either is reached."""
        monkeypatch.setattr(settings, "web_search_enabled", False)
        ready_calls = []
        search_calls = []
        monkeypatch.setattr(
            "app.services.tools.implementations.web_search_ready",
            lambda: ready_calls.append(1) or True,
        )
        monkeypatch.setattr(
            "app.services.tools.implementations._search_web_core",
            lambda q, mr=None: (
                search_calls.append(1)
                or [WebSearchResult(title="T", url="https://e.com", snippet="s")]
            ),
        )

        registry = build_tool_registry()
        result = run(registry.execute("web_search", {"query": "hello"}, make_context()))

        assert result.success
        assert result.data == []
        assert ready_calls == []
        assert search_calls == []

    def test_disabled_provider_returns_empty_success(self, monkeypatch):
        registry = build_tool_registry()
        context = make_context()
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr("app.services.tools.implementations.web_search_ready", lambda: False)
        result = run(registry.execute("web_search", {"query": "hello"}, context))
        assert result.success
        assert result.data == []

    def test_enabled_provider_returns_results(self, monkeypatch):
        registry = build_tool_registry()
        context = make_context()
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr("app.services.tools.implementations.web_search_ready", lambda: True)
        monkeypatch.setattr(
            "app.services.tools.implementations._search_web_core",
            lambda q, mr=None: [WebSearchResult(title="T", url="https://e.com", snippet="s")],
        )
        result = run(registry.execute("web_search", {"query": "hello"}, context))
        assert result.success
        assert len(result.data) == 1


class TestSummarizationTool:
    def test_executes_against_fake_store_and_llm(self):
        registry = build_tool_registry()
        context = make_context(
            vector_store=FakeVectorStore(chunks=[make_chunk("c1")]),
            llm_client=FakeLLMClient(response="the summary"),
        )
        result = run(registry.execute("summarization", {"document_id": "doc-1"}, context))
        assert result.success
        summary, chunks = result.data
        assert summary == "the summary"
        assert len(chunks) == 1


class TestOutputValidation:
    def test_tool_violating_output_schema_raises_tool_output_error(self):
        from app.services.tools.base import ToolResult
        from app.services.tools.implementations import RetrievalInput
        from app.services.tools.registry import ToolRegistry

        class BadRetrievalTool:
            name = "retrieval"
            description = "bad"
            args_schema = RetrievalInput
            output_schema = list[RetrievedChunk]

            async def execute(self, args, context):
                return ToolResult(success=True, data=["not", "chunks"])

        registry = ToolRegistry()
        registry.register(BadRetrievalTool())
        with pytest.raises(ToolOutputError):
            run(registry.execute("retrieval", {"query": "q"}, make_context()))


class TestInstrumentation:
    def test_success_logs_tool_invocation(self, monkeypatch, caplog):
        registry = build_tool_registry()
        monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda q: [0.1, 0.2])
        context = make_context(vector_store=FakeVectorStore(chunks=[make_chunk("c1")]))
        with caplog.at_level("INFO", logger="app.services.tools.registry"):
            run(registry.execute("retrieval", {"query": "question"}, context))
        events = [r for r in caplog.records if r.message == "tool_invocation"]
        assert len(events) == 1
        fields = events[0].extra_fields
        assert fields["tool"] == "retrieval"
        assert fields["success"] is True
        assert fields["output"] == {"count": 1, "ids": ["c1"]}
        assert "latency_ms" in fields

    def test_missing_vector_store_returns_failed_result(self):
        from app.services.tools.base import ToolContext

        registry = build_tool_registry()
        # vector_store and llm_client default to None on a bare context.
        context = ToolContext(vector_store=None, llm_client=None)
        result = run(registry.execute("retrieval", {"query": "q"}, context))
        assert result.success is False
        assert "VectorStore" in result.error
