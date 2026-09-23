"""Unit tests for ChatService.stream_query (the SSE streaming pipeline)
and its supporting helpers (_stream_filtering_sources).

retrieve/search_web are monkeypatched at the rag_service module level,
the same pattern test_rag_service.py and test_rag_service_diagnose.py
use — these tests cover stream_query's event sequencing and its parity
with handle_query's blocking behavior, not retrieval internals.
"""

import app.services.rag_service as rag_service_module
from app.core.config import settings
from app.core.exceptions import VectorStoreNotFoundError
from app.models.document import RetrievedChunk
from app.services.prompt_builder import FALLBACK_REPLY
from app.services.rag_service import ChatService, _stream_filtering_sources


class FakeLLMClient:
    """generate_stream yields `response` split into a few pieces (not one
    single chunk) so tests actually exercise multi-piece reassembly; pops
    sequentially through `responses` when given, mirroring
    test_rag_service.py's FakeLLMClient."""

    def __init__(self, response="grounded streamed answer", responses=None):
        self.response = response
        self._responses = list(responses) if responses is not None else None
        self.calls = []

    def _next_response(self):
        if self._responses:
            return self._responses.pop(0)
        return self.response

    def generate(self, prompt):
        self.calls.append(prompt)
        return self._next_response()

    def generate_stream(self, prompt):
        self.calls.append(prompt)
        text = self._next_response()
        third = max(len(text) // 3, 1)
        for i in range(0, len(text), third) or [0]:
            piece = text[i : i + third]
            if piece:
                yield piece


class FakeVectorStore:
    """stream_query never touches the vector store directly — retrieve()
    is monkeypatched at the module level — except for the response-cache
    key, which calls total_vectors() as a proxy for the document set. That
    must exist on the fake for stream_query to reach the patched
    retrieve(); the value itself is irrelevant to these tests."""

    def total_vectors(self):
        return 1


def make_service(llm_client=None):
    return ChatService(FakeVectorStore(), llm_client or FakeLLMClient())


def make_chunk(text="chunk text", score=0.9):
    return RetrievedChunk(chunk_id="chunk-1", document_id="doc-1", text=text, score=score, metadata={})


def collect(events):
    return list(events)


class TestStreamFilteringSources:
    def test_passes_through_text_with_no_sources_heading(self):
        chunks = ["Hello ", "world, ", "this is fine."]
        assert "".join(_stream_filtering_sources(iter(chunks))) == "Hello world, this is fine."

    def test_suppresses_sources_heading_and_everything_after(self):
        chunks = ["The answer is X.\n\n", "Sources:\n", "- doc-123\n", "- doc-456"]
        result = "".join(_stream_filtering_sources(iter(chunks)))
        assert result == "The answer is X."
        assert "Sources" not in result
        assert "doc-123" not in result

    def test_heading_split_across_chunk_boundaries_still_caught(self):
        # "Sources:" split mid-word across two chunks — the held-back
        # margin must be wide enough to catch this, not just a heading
        # that happens to land whole in one chunk.
        chunks = ["The answer.\n\nSour", "ces:\n- doc-1"]
        result = "".join(_stream_filtering_sources(iter(chunks)))
        assert result == "The answer."

    def test_case_insensitive_heading_match(self):
        chunks = ["Answer text.\n\n", "SOURCES:\n- doc-1"]
        result = "".join(_stream_filtering_sources(iter(chunks)))
        assert result == "Answer text."


class TestStreamQueryConversational:
    def test_conversational_yields_planning_then_answer_then_done(self):
        service = make_service()
        events = collect(service.stream_query("hi"))

        assert events[0] == {"type": "trace", "stage": "planning", "detail": {"action": "conversational"}}
        assert events[1]["type"] == "answer_chunk"
        assert events[-1]["type"] == "done"
        assert events[-1]["payload"].tool_used == "none"
        assert events[-1]["payload"].answer == events[1]["text"]


class TestStreamQueryRetrieveHappyPath:
    def test_trace_sequence_and_streamed_answer_matches_done_payload(self, monkeypatch):
        monkeypatch.setattr(
            rag_service_module, "retrieve", lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()]
        )
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)

        llm_client = FakeLLMClient(response="This is the grounded answer.")
        service = make_service(llm_client)

        events = collect(service.stream_query("what does the document say?"))

        trace_stages = [e["stage"] for e in events if e["type"] == "trace"]
        assert trace_stages == ["planning", "retrieval", "grading", "generating"]

        answer_chunks = [e["text"] for e in events if e["type"] == "answer_chunk"]
        assert len(answer_chunks) > 1  # actually multiple pieces, not one blob
        assert "".join(answer_chunks) == "This is the grounded answer."

        assert events[-1]["type"] == "done"
        done_payload = events[-1]["payload"]
        assert done_payload.answer == "This is the grounded answer."
        assert done_payload.tool_used == "retrieval"
        assert len(done_payload.retrieved_chunks) == 1

    def test_matches_handle_query_steps_taken_for_equivalent_run(self, monkeypatch):
        # Same fixture run through both handle_query and stream_query
        # should produce identical steps_taken — a divergence here would
        # mean the two pipelines aren't actually equivalent.
        monkeypatch.setattr(
            rag_service_module, "retrieve", lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()]
        )
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)

        blocking_service = make_service(FakeLLMClient(response="answer text"))
        streaming_service = make_service(FakeLLMClient(response="answer text"))

        blocking_response = blocking_service.handle_query("a question")
        streaming_events = collect(streaming_service.stream_query("a question"))
        streaming_response = streaming_events[-1]["payload"]

        assert streaming_response.steps_taken == blocking_response.steps_taken
        assert streaming_response.tool_used == blocking_response.tool_used


class TestStreamQueryReflection:
    def test_ungrounded_first_answer_triggers_reflecting_trace_and_regenerates(self, monkeypatch):
        monkeypatch.setattr(
            rag_service_module, "retrieve", lambda query, vector_store, top_k=None, min_score=None, tenant_id=None, document_ids=None, image_vector_store=None: [make_chunk()]
        )
        monkeypatch.setattr(settings, "retrieval_grade_threshold", 0.5)

        llm_client = FakeLLMClient(responses=[FALLBACK_REPLY, "the corrected, grounded answer"])
        service = make_service(llm_client)

        events = collect(service.stream_query("a question"))

        trace_stages = [e["stage"] for e in events if e["type"] == "trace"]
        assert trace_stages == ["planning", "retrieval", "grading", "generating", "reflecting"]

        done_payload = events[-1]["payload"]
        assert done_payload.answer == "the corrected, grounded answer"
        # Two generate_stream calls: the ungrounded first attempt, then
        # the reflection regen.
        assert len(llm_client.calls) == 2

        # The reassembled answer_chunk stream includes text from BOTH
        # attempts (the discarded FALLBACK_REPLY, then the correction) —
        # a "reflecting" trace event is the client's signal to reset its
        # accumulator, not a guarantee only the final text was streamed.
        answer_chunks_text = "".join(e["text"] for e in events if e["type"] == "answer_chunk")
        assert FALLBACK_REPLY in answer_chunks_text
        assert "the corrected, grounded answer" in answer_chunks_text


class TestStreamQuerySummarize:
    def test_summarize_action_yields_retrieval_and_generating_stages(self, monkeypatch):
        chunks = [make_chunk()]
        monkeypatch.setattr(
            rag_service_module,
            "summarize_document",
            lambda document_id, vector_store, llm_client, tenant_id=None: ("a concise summary", chunks),
        )

        service = make_service()
        document_id = "123e4567-e89b-12d3-a456-426614174000"
        events = collect(service.stream_query(f"please summarize {document_id}"))

        trace_stages = [e["stage"] for e in events if e["type"] == "trace"]
        assert trace_stages == ["planning", "retrieval", "generating"]

        answer_chunks = [e["text"] for e in events if e["type"] == "answer_chunk"]
        assert answer_chunks == ["a concise summary"]

        done_payload = events[-1]["payload"]
        assert done_payload.answer == "a concise summary"
        assert done_payload.tool_used == "summarization"


class TestStreamQueryErrorHandling:
    def test_retrieval_failure_yields_error_event_not_an_exception(self, monkeypatch):
        def _raise_not_found(query, vector_store, top_k=None, min_score=None, tenant_id=None, **kwargs):
            raise VectorStoreNotFoundError("No vector store found.")

        monkeypatch.setattr(rag_service_module, "retrieve", _raise_not_found)

        service = make_service()
        events = collect(service.stream_query("a question"))

        assert events[-1]["type"] == "error"
        assert events[-1]["detail"]["error_type"] == "VectorStoreNotFoundError"
        assert events[-1]["detail"]["status_code"] == 404
        # No "done" event when the pipeline fails.
        assert all(e["type"] != "done" for e in events)
