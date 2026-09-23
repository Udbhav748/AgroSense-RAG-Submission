"""Tests for embedding_service.py's token-budget safety net
(_split_oversized_chunk / generate_embeddings).

The real SentenceTransformer model is never loaded here (no test in this
suite does — see test_main.py's "never touches ... the real
sentence-transformers model"). A fake whitespace tokenizer stands in for
the real BERT wordpiece tokenizer: deterministic, dependency-free, and
enough to exercise the splitting logic itself, which only depends on
`.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)`
returning `input_ids`/`offset_mapping` and `.max_seq_length`.
"""

import numpy as np

from app.models.document import DocumentChunk
from app.services.embedding_service import (
    _SPECIAL_TOKEN_BUDGET,
    _split_oversized_chunk,
    generate_embeddings,
)


class _FakeTokenizer:
    """One "token" per whitespace-separated word — offsets computed from
    the actual text, so reconstructing pieces from them is a real,
    verifiable round trip, not a stand-in value."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        words = text.split(" ")
        offsets = []
        input_ids = []
        pos = 0
        for word in words:
            start = text.index(word, pos)
            end = start + len(word)
            offsets.append((start, end))
            input_ids.append(len(input_ids))
            pos = end
        return {"input_ids": input_ids, "offset_mapping": offsets}


class _FakeModel:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = _FakeTokenizer()

    def encode(self, texts, normalize_embeddings=True, batch_size=8):
        # One fixed-size vector per input text — content doesn't matter,
        # only that the count matches len(texts).
        return np.array([[0.1, 0.2, 0.3] for _ in texts])


def make_chunk(text, chunk_id="c1", document_id="doc-1", chunk_index=0, metadata=None):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        metadata=metadata or {"tenant_id": 1, "source": "pdf"},
    )


class TestSplitOversizedChunk:
    def test_chunk_within_budget_returned_unchanged(self):
        chunk = make_chunk("one two three")
        model = _FakeModel(max_seq_length=10)  # budget = 8, well above 3 words
        result = _split_oversized_chunk(chunk, model)
        assert result == [chunk]

    def test_oversized_chunk_is_split_into_token_safe_pieces(self):
        text = "one two three four five six seven eight nine ten"
        chunk = make_chunk(text)
        # budget = max_seq_length - _SPECIAL_TOKEN_BUDGET = 5 - 2 = 3 tokens/piece
        model = _FakeModel(max_seq_length=3 + _SPECIAL_TOKEN_BUDGET)

        pieces = _split_oversized_chunk(chunk, model)

        assert len(pieces) == 4  # ceil(10 / 3)
        # Every piece must itself be within budget once re-tokenized.
        for piece in pieces:
            token_count = len(model.tokenizer(piece.text, add_special_tokens=False)["input_ids"])
            assert token_count <= 3
        # Reconstructing the pieces must cover the original text exactly
        # (offsets are contiguous, nothing dropped or duplicated).
        assert " ".join(piece.text for piece in pieces) == text

    def test_split_pieces_get_unique_ids_and_split_metadata(self):
        text = " ".join(f"word{i}" for i in range(10))
        chunk = make_chunk(text, chunk_id="original-id")
        model = _FakeModel(max_seq_length=3 + _SPECIAL_TOKEN_BUDGET)

        pieces = _split_oversized_chunk(chunk, model)

        assert len({p.chunk_id for p in pieces}) == len(pieces)  # all unique
        assert all(p.chunk_id != "original-id" for p in pieces)
        assert [p.metadata["split_part"] for p in pieces] == list(range(len(pieces)))
        assert all(p.metadata["split_of"] == len(pieces) for p in pieces)

    def test_split_pieces_preserve_lineage_and_other_metadata(self):
        text = " ".join(f"word{i}" for i in range(10))
        chunk = make_chunk(text, document_id="doc-42", chunk_index=3, metadata={"tenant_id": 7, "source": "pdf"})
        model = _FakeModel(max_seq_length=3 + _SPECIAL_TOKEN_BUDGET)

        pieces = _split_oversized_chunk(chunk, model)

        assert all(p.document_id == "doc-42" for p in pieces)
        assert all(p.chunk_index == 3 for p in pieces)
        assert all(p.metadata["tenant_id"] == 7 for p in pieces)
        assert all(p.metadata["source"] == "pdf" for p in pieces)


class TestGenerateEmbeddingsSplitsOversizedChunks:
    def test_oversized_chunk_yields_more_embeddings_than_input_chunks(self, monkeypatch):
        text = " ".join(f"word{i}" for i in range(10))
        chunks = [make_chunk(text, chunk_id="c1"), make_chunk("short chunk", chunk_id="c2")]
        model = _FakeModel(max_seq_length=3 + _SPECIAL_TOKEN_BUDGET)
        monkeypatch.setattr("app.services.embedding_service.get_embedding_model", lambda: model)

        embedded = generate_embeddings(chunks)

        # c1 (10 words, budget 3/piece) splits into 4; c2 stays 1.
        assert len(embedded) == 5
        assert len(chunks) == 2  # input untouched
        # Every embedded chunk still carries its (possibly split) text —
        # concatenating c1's pieces back together must equal the original.
        reconstructed = " ".join(e.metadata["text"] for e in embedded if e.chunk_id != "c2")
        assert reconstructed == text

    def test_chunks_within_budget_are_unaffected(self, monkeypatch):
        chunks = [make_chunk("short chunk one", chunk_id="c1"), make_chunk("short chunk two", chunk_id="c2")]
        model = _FakeModel(max_seq_length=256)
        monkeypatch.setattr("app.services.embedding_service.get_embedding_model", lambda: model)

        embedded = generate_embeddings(chunks)

        assert len(embedded) == 2
        assert {e.chunk_id for e in embedded} == {"c1", "c2"}
