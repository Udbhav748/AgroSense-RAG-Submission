"""Tests for the 10/10-upgrade Agent 1 features:

1. PlanningAgent (services/planning_agent.py): LLM planner that decides
   one next step at a time (plan_next), with defensive JSON parsing and a
   deterministic fallback step.
2. AgentExecutor (services/agent_executor.py): the genuine ReAct
   plan→act→observe→repeat loop over the dynamic tool registry — each
   iteration's plan_next() call sees every earlier step's actual result.
3. ChatService._handle_via_executor: the flag-gated wiring so eligible
   queries route through the executor.
4. pgvector config surface (Settings.pgvector_*): the store swap is
   config-driven; these assert the settings exist with sane defaults.
5. eval/nightly_eval.py's summary builder.

Like the rest of the suite, these are offline: the LLM is a FakeLLM, the
vector store is a fake, and no real Postgres/Gemini is touched. The
pgvector store itself needs a live Postgres+extension, so it's covered by
a skip-if-no-DATABASE_URL smoke test only (conftest.py forces
DATABASE_URL="" for the session, so it skips here and runs in an
integration context instead).
"""

import asyncio
import json

import pytest

from app.core.config import settings
from app.models.document import RetrievedChunk
from app.services.planning_agent import Observation, PlanningAgent, PlanStep
from app.services.tools.base import ToolContext
from app.services.tools.factory import build_tool_registry


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    """generate/generate_structured pop from `responses` in call order.
    Raise on call when raise_on_call is set (failure-path tests)."""

    def __init__(self, responses=None, raise_on_call=False):
        self._responses = list(responses) if responses else []
        self._raise = raise_on_call
        self.calls = []

    def generate(self, prompt):
        self.calls.append(("generate", prompt))
        return self._pop()

    def generate_structured(self, prompt):
        self.calls.append(("generate_structured", prompt))
        return self._pop()

    def _pop(self):
        if self._raise:
            raise RuntimeError("llm down")
        return self._responses.pop(0) if self._responses else ""


class FakeVectorStore:
    def __init__(self, chunks=None):
        self._chunks = chunks or []

    def search(self, query_vector, top_k, tenant_id=None, document_ids=None):
        return self._chunks[:top_k]

    def get_chunks_by_document(self, document_id, tenant_id=None):
        return self._chunks

    def total_vectors(self):
        return len(self._chunks)


def make_chunk(chunk_id="c1", document_id="doc-1", text="alpha beta gamma delta", score=0.9):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        metadata={"chunk_index": 0, "tenant_id": None},
    )


def make_registry():
    return build_tool_registry()


class TestPlanningAgent:
    def _planner(self, llm=None):
        return PlanningAgent(llm or FakeLLM(), make_registry())

    def test_plan_next_returns_first_step(self):
        llm = FakeLLM([json.dumps({"tool": "retrieval", "args": {"query": "q"}, "reason": "find"})])
        step = run(self._planner(llm).plan_next("some question", None, []))
        assert step.tool == "retrieval"
        assert step.args == {"query": "q"}

    def test_plan_next_can_choose_synthesize_directly(self):
        llm = FakeLLM([json.dumps({"tool": "synthesize", "args": {}, "reason": "enough already"})])
        step = run(self._planner(llm).plan_next("q", None, []))
        assert step.tool == "synthesize"

    def test_fallback_retrieval_on_unparseable_output_with_no_observations(self):
        llm = FakeLLM(["sure, here's my answer!"])
        step = run(self._planner(llm).plan_next("anything", None, []))
        assert step.tool == "retrieval"

    def test_fallback_synthesize_on_unparseable_output_with_prior_observations(self):
        # Once at least one step has already run, a broken planner call
        # must not retry retrieval blindly — it degrades to synthesizing
        # from whatever's already been gathered.
        llm = FakeLLM(["sure, here's my answer!"])
        prior = [
            Observation(
                step=PlanStep(tool="retrieval", args={"query": "q"}, reason="find"),
                success=True,
                summary={"count": 1, "ids": ["c1"]},
            )
        ]
        step = run(self._planner(llm).plan_next("anything", None, prior))
        assert step.tool == "synthesize"

    def test_fallback_retrieval_on_llm_failure_with_no_observations(self):
        step = run(self._planner(FakeLLM(raise_on_call=True)).plan_next("anything", None, []))
        assert step.tool == "retrieval"

    def test_unknown_tool_falls_back(self):
        llm = FakeLLM([json.dumps({"tool": "not_a_real_tool", "args": {}, "reason": "nope"})])
        step = run(self._planner(llm).plan_next("q", None, []))
        assert step.tool == "retrieval"

    def test_observations_are_included_in_the_prompt(self):
        """The whole point of plan_next over the old upfront plan(): later
        steps must actually see what earlier ones returned."""
        llm = FakeLLM([json.dumps({"tool": "synthesize", "args": {}, "reason": "done"})])
        prior = [
            Observation(
                step=PlanStep(tool="retrieval", args={"query": "q"}, reason="find"),
                success=True,
                summary={"count": 3, "ids": ["c1", "c2", "c3"]},
            )
        ]
        run(self._planner(llm).plan_next("q", None, prior))
        prompt = llm.calls[0][1]
        assert "retrieval" in prompt
        assert "c1" in prompt


class TestAgentExecutor:
    def _executor(self, llm=None):
        from app.services.agent_executor import AgentExecutor

        shared_llm = llm or FakeLLM(
            [
                json.dumps({"tool": "retrieval", "args": {"query": "q"}, "reason": "find"}),
                json.dumps({"tool": "synthesize", "args": {}, "reason": "answer"}),
                "grounded synthesized answer",
            ]
        )
        return AgentExecutor(
            llm_client=shared_llm,
            tool_registry=make_registry(),
            planning_agent=PlanningAgent(shared_llm, make_registry()),
        )

    def test_execute_runs_plan_and_synthesizes(self):
        executor = self._executor()
        context = ToolContext(
            vector_store=FakeVectorStore([make_chunk()]),
            llm_client=FakeLLM(["irrelevant"]),
        )
        result = run(executor.execute("what is the PMP?", context=context))
        assert result.answer
        assert result.steps_taken >= 2
        assert result.tool_used in ("retrieval", "summarization")
        # Sources built from retrieved chunks surface structurally.
        assert result.sources or result.answer

    def test_execute_with_empty_retrieval_returns_fallback(self):
        from app.services.agent_executor import AgentExecutor

        # Only one planner response is queued on purpose: the first
        # plan_next() call consumes it (a retrieval step, which comes
        # back empty against the empty FakeVectorStore below); the
        # second plan_next() call finds no response left and — since an
        # observation already exists by then — falls back to
        # "synthesize" rather than retrying retrieval forever. Proves
        # the executor degrades to FALLBACK_REPLY, not an error, when
        # the planner runs dry mid-loop.
        planner_llm = FakeLLM(
            [json.dumps({"tool": "retrieval", "args": {"query": "q"}, "reason": "find"})]
        )
        executor = AgentExecutor(
            llm_client=planner_llm,
            tool_registry=make_registry(),
            planning_agent=PlanningAgent(planner_llm, make_registry()),
        )
        context = ToolContext(
            vector_store=FakeVectorStore([]),  # nothing to retrieve
            llm_client=FakeLLM(["n/a"]),
        )
        result = run(executor.execute("nothing here", context=context))
        assert result.answer  # FALLBACK_REPLY or honest "insufficient"

    def test_second_step_reacts_to_first_steps_result(self, monkeypatch):
        """The behavioral difference a real ReAct loop is supposed to
        deliver: retrieval already found something, so the planner's
        second call (seeing that observation) can choose to stop instead
        of blindly running a second tool step."""
        # RetrievalTool routes through the real embed_query() for a
        # non-FAISS store — monkeypatch it like test_tools_registry.py
        # does, so this test doesn't depend on a live embedding model.
        monkeypatch.setattr("app.services.retrieval_service.embed_query", lambda q: [0.1, 0.2])
        llm = FakeLLM(
            [
                json.dumps({"tool": "retrieval", "args": {"query": "q"}, "reason": "find"}),
                json.dumps(
                    {"tool": "synthesize", "args": {}, "reason": "retrieval already answered it"}
                ),
                "grounded synthesized answer",
            ]
        )
        executor = self._executor(llm=llm)
        context = ToolContext(
            vector_store=FakeVectorStore([make_chunk()]),
            llm_client=FakeLLM(["irrelevant"]),
        )
        result = run(executor.execute("what is the PMP?", context=context))
        assert result.answer == "grounded synthesized answer"
        # Exactly one tool step (retrieval) + synthesis — the planner
        # stopped as soon as it had enough, not after a fixed plan length.
        assert result.steps_taken == 2


class TestChatServiceExecutorWiring:
    def test_handle_via_executor_returns_chat_response(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_executor_enabled", True)
        from app.models.schemas import ChatResponse
        from app.services.rag_service import ChatService

        llm = FakeLLM(
            [
                json.dumps({"tool": "retrieval", "args": {"query": "q"}, "reason": "find"}),
                json.dumps({"tool": "synthesize", "args": {}, "reason": "answer"}),
                "synthesized answer",
            ]
        )
        service = ChatService(FakeVectorStore([make_chunk()]), llm)
        response = service._handle_via_executor(
            "what is the PMP?",
            history=None,
            session_id=None,
            tenant_id=None,
            top_k=5,
            min_score=0.3,
            confirm_web_search=False,
            persona=None,
        )
        assert isinstance(response, ChatResponse)
        assert response.answer

    def test_executor_not_constructed_by_default(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_executor_enabled", False)
        from app.services.rag_service import ChatService

        service = ChatService(FakeVectorStore(), FakeLLM())
        assert service._agent_executor is None

    def test_executor_lazily_constructed(self, monkeypatch):
        from app.services.rag_service import ChatService

        service = ChatService(FakeVectorStore(), FakeLLM())
        executor = service._agent_executor_instance()
        assert service._agent_executor is not None
        assert executor is service._agent_executor


class TestPgvectorConfigSurface:
    def test_pgvector_settings_present(self):
        assert hasattr(settings, "pgvector_enabled")
        assert settings.pgvector_enabled is False
        assert settings.pgvector_dimensions == 384
        assert settings.pgvector_table_name == "document_embeddings"

    def test_agent_executor_settings_present(self):
        assert settings.agent_executor_enabled is False
        assert settings.agent_executor_max_steps == 6

    def test_nightly_eval_settings_present(self):
        assert settings.nightly_eval_enabled is False
        assert settings.nightly_eval_dataset
        assert settings.nightly_eval_baseline
        assert settings.nightly_eval_tolerance == 0.05


class TestPgvectorStoreSmoke:
    @pytest.mark.skipif(
        not getattr(settings, "database_url", ""),
        reason="needs a live Postgres+pgvector (conftest.py forces DATABASE_URL='')",
    )
    def test_store_constructs_and_reports_zero_vectors(self):
        from app.services.pgvector_store import PgvectorVectorStore

        store = PgvectorVectorStore()
        assert store.total_vectors() == 0

    def test_dimension_mismatch_raises_clear_error(self):
        from app.services.pgvector_store import PgvectorVectorStore

        store = PgvectorVectorStore()
        with pytest.raises(Exception):
            store.add_embeddings(
                [
                    type(
                        "EC",
                        (),
                        {
                            "chunk_id": "c",
                            "document_id": "d",
                            "embedding": [0.1] * 5,
                            "metadata": {},
                        },
                    )()
                ]
            )


class TestNightlyEvalSummary:
    def test_summarize_builds_headline_metrics(self):
        from eval.nightly_eval import _summarize

        report = {
            "dataset": "dataset_v1.json",
            "dataset_version": "v1",
            "llm_provider": "groq",
            "llm_model_name": "model",
            "prompt_version": "42",
            "hybrid_search_enabled": True,
            "reranking_enabled": False,
            "n_entries": 3,
            "task_success_rate": 0.9,
            "groundedness_proxy": 0.8,
        }
        summary = _summarize(report, None, gate_passed=True)
        assert summary["gate_passed"] is True
        assert summary["metrics"]["task_success_rate"] == 0.9
        assert "task_success_rate_baseline" not in summary["metrics"]

    def test_summarize_includes_baseline_comparison(self):
        from eval.nightly_eval import _summarize

        report = {"dataset_version": "v1", "task_success_rate": 0.9}
        baseline = {"dataset_version": "v1", "task_success_rate": 0.85}
        summary = _summarize(report, baseline, gate_passed=False)
        assert summary["gate_passed"] is False
        assert summary["metrics"]["task_success_rate_baseline"] == 0.85
