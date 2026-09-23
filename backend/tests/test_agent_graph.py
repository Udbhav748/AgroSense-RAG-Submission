"""Comprehensive test suite for the Graph-Based Agent State Engine (StateGraph).

Covers:
- StateGraph core compilation, validation, and node execution.
- Deterministic transitions and conditional branching.
- Checkpoint history and step snapshotting.
- Cycle capping and loop guardrails (max_steps).
- Node-level functions (planner, document_analyst, fact_checker, web_researcher, summarizer, synthesizer).
- End-to-end multi-agent RAG workflow with reflection loops and recovery.
"""


import pytest

from app.core.config import settings
from app.models.document import RetrievedChunk, WebSearchResult
from app.services.agent_graph import (
    END,
    AgentState,
    GraphCompilationError,
    GraphContext,
    StateGraph,
    create_rag_agent_graph,
    document_analyst_node,
    fact_checker_node,
    planner_node,
    synthesizer_node,
    web_researcher_node,
)


class MockLLM:
    """Mock LLM popping responses in FIFO order."""

    def __init__(self, responses: list[str] | None = None, raise_on_call: bool = False):
        self.responses = list(responses) if responses else []
        self.raise_on_call = raise_on_call
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_on_call:
            raise RuntimeError("LLM service unavailable")
        return self.responses.pop(0) if self.responses else "Mock response."

    def generate_structured(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_on_call:
            raise RuntimeError("LLM service unavailable")
        return self.responses.pop(0) if self.responses else "{}"


class MockVectorStore:
    """Mock VectorStore for document retrieval tests."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        self.chunks = chunks or []

    def search(self, *args, **kwargs) -> list[RetrievedChunk]:
        return self.chunks

    def search_bm25(self, *args, **kwargs) -> list[RetrievedChunk]:
        return self.chunks


def make_chunk(chunk_id: str = "c1", doc_id: str = "doc1", text: str = "Project scope content", score: float = 0.9):
    return RetrievedChunk(chunk_id=chunk_id, document_id=doc_id, text=text, score=score)


# ==============================================================================
# Core Engine Tests
# ==============================================================================

class TestStateGraphRuntime:
    @pytest.mark.asyncio
    async def test_linear_graph_execution(self):
        graph = StateGraph()

        async def step_a(state: AgentState) -> dict:
            return {"metadata": {"step_a": True}, "steps_taken": state.steps_taken + 1}

        async def step_b(state: AgentState) -> dict:
            return {"metadata": {**state.metadata, "step_b": True}, "steps_taken": state.steps_taken + 1}

        graph.add_node("node_a", step_a)
        graph.add_node("node_b", step_b)
        graph.set_entry_point("node_a")
        graph.add_edge("node_a", "node_b")
        graph.add_edge("node_b", END)

        compiled = graph.compile()
        initial = AgentState(query="test query")
        final_state = await compiled.run(initial)

        assert final_state.metadata == {"step_a": True, "step_b": True}
        assert final_state.steps_taken == 2
        history = compiled.get_state_history()
        assert len(history) == 2
        assert history[0].node_name == "node_a"
        assert history[1].node_name == "node_b"

    @pytest.mark.asyncio
    async def test_conditional_branching(self):
        graph = StateGraph()

        graph.add_node("classifier", lambda s: s.copy_with(plan="branch_left"))
        graph.add_node("left_node", lambda s: s.copy_with(draft_answer="left"))
        graph.add_node("right_node", lambda s: s.copy_with(draft_answer="right"))
        graph.set_entry_point("classifier")

        def route(state: AgentState) -> str:
            return "left_node" if state.plan == "branch_left" else "right_node"

        graph.add_conditional_edges("classifier", route)
        graph.add_edge("left_node", END)
        graph.add_edge("right_node", END)

        compiled = graph.compile()
        final_state = await compiled.run(AgentState(query="branching test"))
        assert final_state.draft_answer == "left"

    def test_compilation_validation_missing_entry(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: s)
        with pytest.raises(GraphCompilationError, match="entry point"):
            graph.compile()

    def test_compilation_validation_invalid_edge_target(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: s)
        graph.set_entry_point("a")
        graph.add_edge("a", "non_existent_node")
        with pytest.raises(GraphCompilationError, match="not a registered node"):
            graph.compile()

    @pytest.mark.asyncio
    async def test_cycle_capping_max_steps(self):
        graph = StateGraph()

        # Cyclic graph: A -> B -> A
        graph.add_node("a", lambda s: s.copy_with(steps_taken=s.steps_taken + 1))
        graph.add_node("b", lambda s: s.copy_with(steps_taken=s.steps_taken + 1))
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")

        compiled = graph.compile(max_steps=6)
        final_state = await compiled.run(AgentState(query="loop test"))

        assert final_state.steps_taken == 6
        assert "Max steps exceeded" in (final_state.error or "")
        assert len(compiled.get_state_history()) == 6

    @pytest.mark.asyncio
    async def test_streaming_execution(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: s.copy_with(draft_answer="step a"))
        graph.add_node("b", lambda s: s.copy_with(draft_answer="step b"))
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        compiled = graph.compile()
        steps = []
        async for node_name, state in compiled.stream(AgentState(query="streaming")):
            steps.append((node_name, state.draft_answer))

        assert steps == [("a", "step a"), ("b", "step b")]


# ==============================================================================
# Node Tests
# ==============================================================================

class TestAgentGraphNodes:
    @pytest.mark.asyncio
    async def test_planner_conversational(self):
        state = AgentState(query="hello")
        out = await planner_node(state)
        assert isinstance(out.plan, dict)
        assert out.plan["action"] == "conversational"
        assert "Upload a document" in out.draft_answer

    @pytest.mark.asyncio
    async def test_planner_summarize_uuid(self):
        state = AgentState(query="summarize 12345678-1234-5678-1234-567812345678")
        out = await planner_node(state)
        assert isinstance(out.plan, dict)
        assert out.plan["action"] == "summarize"
        assert out.document_id == "12345678-1234-5678-1234-567812345678"

    @pytest.mark.asyncio
    async def test_document_analyst_retrieval(self, monkeypatch):
        chunk = make_chunk(text="Project scope is defined in chapter 1.")
        monkeypatch.setattr("app.services.agent_graph.nodes.retrieve", lambda **kwargs: [chunk])
        context = GraphContext(vector_store=MockVectorStore())

        state = AgentState(query="What is project scope?")
        out = await document_analyst_node(state, context)

        assert len(out.retrieved_chunks) == 1
        assert out.retrieved_chunks[0].text == "Project scope is defined in chapter 1."

    @pytest.mark.asyncio
    async def test_web_researcher_execution(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr(settings, "web_search_requires_approval", False)
        web_res = WebSearchResult(title="Example", url="https://example.com", snippet="Current news info.")
        monkeypatch.setattr("app.services.agent_graph.nodes.search_web", lambda q, max_results=3: [web_res])

        state = AgentState(query="latest stock prices")
        out = await web_researcher_node(state)
        assert len(out.web_results) == 1
        assert out.web_results[0].url == "https://example.com"

    @pytest.mark.asyncio
    async def test_synthesizer_generation_and_delimiters(self):
        llm = MockLLM(["Project scope defines the boundaries [1]."])
        chunk = make_chunk(text="Scope definition details.")
        context = GraphContext(llm_client=llm)

        state = AgentState(query="What is scope?", retrieved_chunks=[chunk])
        out = await synthesizer_node(state, context)

        assert "Project scope defines the boundaries [1]" in out.draft_answer
        assert out.final_response is not None
        assert len(out.final_response.sources) == 1

    @pytest.mark.asyncio
    async def test_fact_checker_valid_citations(self):
        llm = MockLLM(['{"1": true}'])
        chunk = make_chunk(text="Scope details here.")
        context = GraphContext(llm_client=llm)

        state = AgentState(
            query="question",
            draft_answer="Answer claim supported [1].",
            retrieved_chunks=[chunk],
        )
        out = await fact_checker_node(state, context)

        assert out.is_fact_check_passed() is True
        assert out.reflection_count == 0

    @pytest.mark.asyncio
    async def test_fact_checker_invalid_citations(self):
        chunk = make_chunk(text="Scope details.")
        # Citation [2] does not exist (only chunk 1 exists)
        state = AgentState(
            query="question",
            draft_answer="Hallucinated citation [2].",
            retrieved_chunks=[chunk],
        )
        out = await fact_checker_node(state)

        assert out.is_fact_check_passed() is False
        assert out.reflection_count == 1


# ==============================================================================
# End-to-End RAG Graph Tests
# ==============================================================================

class TestRAGAgentGraphWorkflow:
    @pytest.mark.asyncio
    async def test_e2e_document_qa_workflow(self, monkeypatch):
        chunk = make_chunk(text="Grounded factual evidence.")
        monkeypatch.setattr("app.services.agent_graph.nodes.retrieve", lambda **kwargs: [chunk])
        monkeypatch.setattr(settings, "citation_verification_enabled", False)

        llm = MockLLM(["Direct answer according to facts [1]."])
        context = GraphContext(llm_client=llm, vector_store=MockVectorStore())

        graph = create_rag_agent_graph(context)
        state = await graph.run(AgentState(query="What are the facts?"), context=context)

        assert "Direct answer according to facts [1]" in state.draft_answer
        assert state.final_response is not None
        assert state.final_response.tool_used == "agent_graph"
        assert len(state.final_response.sources) == 1

    @pytest.mark.asyncio
    async def test_e2e_fact_check_reflection_loop(self, monkeypatch):
        chunk = make_chunk(text="True fact.")
        monkeypatch.setattr("app.services.agent_graph.nodes.retrieve", lambda **kwargs: [chunk])
        monkeypatch.setattr(settings, "citation_verification_enabled", True)

        # 1. Synthesizer draft 1
        # 2. Fact checker verifies draft 1 -> fails {"1": false}
        # 3. Synthesizer reflection draft 2
        # 4. Fact checker verifies draft 2 -> passes {"1": true}
        llm = MockLLM([
            "First draft [1].",
            '{"1": false}',
            "Corrected reflected draft [1].",
            '{"1": true}',
        ])
        context = GraphContext(llm_client=llm, vector_store=MockVectorStore())

        graph = create_rag_agent_graph(context, max_reflections=2)
        state = await graph.run(AgentState(query="Verify this"), context=context)

        assert "Corrected reflected draft [1]" in state.draft_answer
        assert state.reflection_count == 1
        assert state.is_fact_check_passed() is True

    @pytest.mark.asyncio
    async def test_e2e_conversational_fast_path(self):
        graph = create_rag_agent_graph()
        state = await graph.run(AgentState(query="hello"))

        assert "Upload a document" in state.draft_answer
        assert state.final_response is not None
        assert state.final_response.tool_used == "conversational"
        assert state.steps_taken >= 1
