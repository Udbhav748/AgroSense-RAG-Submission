"""Tests for claim-level inline citation: prompt_builder numbers each
context excerpt with a [N] bracket label and instructs the model to cite
inline, and rag_service builds ChatResponse.sources with matching
`number` fields — chunks first (1..len(chunks)), web results continuing
from there. The two numbering schemes must agree, or a [N] in the answer
text would point at the wrong source.
"""

import time

from app.services.prompt_builder import _INSTRUCTIONS, build_prompt
from app.services.rag_service import ChatService, _source_references, _web_source_references
from tests.test_rag_service import FakeLLMClient, FakeVectorStore, make_chunk, make_web_result


class TestPromptNumbersExcerpts:
    def test_chunks_numbered_from_one_in_order(self):
        chunks = [make_chunk(document_id="doc-a"), make_chunk(document_id="doc-b")]
        prompt = build_prompt("question", chunks)
        assert "[1] [Document doc-a]" in prompt
        assert "[2] [Document doc-b]" in prompt

    def test_web_results_numbered_after_chunks(self):
        chunks = [make_chunk(document_id="doc-a")]
        web_results = [make_web_result(url="https://a.test"), make_web_result(url="https://b.test")]
        prompt = build_prompt("question", chunks, web_results=web_results)
        assert "[1] [Document doc-a]" in prompt
        assert "[2] [Web result: Example]" in prompt
        assert "[3] [Web result: Example]" in prompt

    def test_web_results_start_at_one_when_no_chunks(self):
        web_results = [make_web_result()]
        prompt = build_prompt("question", [], web_results=web_results)
        assert "[1] [Web result: Example]" in prompt

    def test_no_context_has_no_numbered_excerpt_blocks(self):
        # _INSTRUCTIONS itself mentions "[1]" as an example, so check for
        # an actual numbered excerpt block rather than the bare string.
        prompt = build_prompt("question", [])
        assert "[1] [Document" not in prompt
        assert "[1] [Web result" not in prompt


class TestInstructionsRequestInlineCitation:
    def test_instructs_inline_bracket_citation(self):
        lower = _INSTRUCTIONS.lower()
        assert "cite" in lower
        assert "inline" in lower

    def test_no_longer_requests_a_trailing_sources_list(self):
        # Replaced by inline citation — a trailing list would just
        # duplicate the structured ChatResponse.sources the frontend
        # already renders.
        assert "list the document source" not in _INSTRUCTIONS.lower()


class TestSourceReferenceNumbering:
    def test_chunk_sources_numbered_from_one(self):
        chunks = [make_chunk(document_id="doc-a"), make_chunk(document_id="doc-b")]
        sources = _source_references(chunks)
        assert [s.number for s in sources] == [1, 2]

    def test_web_sources_continue_numbering_after_chunks(self):
        chunks = [make_chunk()]
        web_results = [make_web_result(url="https://a.test"), make_web_result(url="https://b.test")]
        chunk_sources = _source_references(chunks)
        web_sources = _web_source_references(web_results, start=len(chunk_sources) + 1)
        assert [s.number for s in chunk_sources + web_sources] == [1, 2, 3]

    def test_web_sources_start_at_one_when_no_chunks(self):
        web_sources = _web_source_references([make_web_result()])
        assert [s.number for s in web_sources] == [1]


class TestRespondBuildsNumberedSources:
    def test_sources_numbered_end_to_end_documents_then_web(self):
        service = ChatService(FakeVectorStore(), FakeLLMClient())
        chunks = [make_chunk(document_id="doc-a"), make_chunk(document_id="doc-b")]
        web_results = [make_web_result(url="https://a.test")]

        response = service._respond(
            answer="answer text [1][2][3]",
            retrieved_chunks=chunks,
            query="question",
            query_type="document_query",
            tool_used="retrieval",
            steps_taken=1,
            start=time.perf_counter(),
            web_results=web_results,
        )

        assert [s.number for s in response.sources] == [1, 2, 3]
        assert response.sources[2].document_id == "web"
