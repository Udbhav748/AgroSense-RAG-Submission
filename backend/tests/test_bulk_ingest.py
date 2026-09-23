"""Unit tests for high-throughput bulk ingestion script (backend/scripts/bulk_ingest.py).

Verifies file scanning, collection auto-tagging, markdown/CSV/PDF parsing,
batch embedding generation, resume checkpointing, FAISS indexing, and error resilience.
"""

import sys
from pathlib import Path

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from unittest.mock import MagicMock, patch

import pytest

from app.models.document import DocumentChunk, EmbeddedChunk
from app.services.faiss_vector_store import FAISSVectorStore, faiss
from scripts.bulk_ingest import (
    BulkIngester,
    CheckpointManager,
    detect_collection,
    generate_batch_embeddings,
    parse_csv_file,
    parse_document_file,
    parse_markdown_file,
    scan_documents,
    split_oversized_chunks,
)

FAKE_DIM = 4


class MockSentenceTransformer:
    """Lightweight mock of SentenceTransformer for fast unit tests."""

    def __init__(self, dim: int = FAKE_DIM):
        self.dim = dim
        self.max_seq_length = 256
        self.tokenizer = None

    def encode(
        self,
        sentences: list[str],
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ):
        import numpy as np

        count = len(sentences)
        vecs = np.ones((count, self.dim), dtype="float32")
        # Normalize vectors
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).astype("float32")


class TestFileScanner:
    """Tests for document directory scanning and format filtering."""

    def test_scan_documents_finds_md_csv_pdf(self, tmp_path: Path):
        (tmp_path / "tomato").mkdir()
        (tmp_path / "potato").mkdir()

        f1 = tmp_path / "tomato" / "early_blight.md"
        f2 = tmp_path / "potato" / "late_blight.markdown"
        f3 = tmp_path / "matrix.csv"
        f4 = tmp_path / "guide.pdf"

        f1.write_text("# Early Blight", encoding="utf-8")
        f2.write_text("# Late Blight", encoding="utf-8")
        f3.write_text("crop,disease\ntomato,blight", encoding="utf-8")
        f4.write_bytes(b"%PDF-1.4 mock pdf content")

        found = scan_documents(tmp_path, recursive=True)
        found_names = {p.name for p in found}

        assert found_names == {
            "early_blight.md",
            "late_blight.markdown",
            "matrix.csv",
            "guide.pdf",
        }

    def test_scan_documents_skips_hidden_and_unsupported_files(self, tmp_path: Path):
        (tmp_path / ".hidden_folder").mkdir()
        (tmp_path / ".hidden_folder" / "secret.md").write_text("secret", encoding="utf-8")
        (tmp_path / ".checkpoint.json").write_text("{}", encoding="utf-8")
        (tmp_path / "_temp.md").write_text("temp", encoding="utf-8")
        (tmp_path / "script.py").write_text("print('hi')", encoding="utf-8")
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "valid.md").write_text("# Valid", encoding="utf-8")

        found = scan_documents(tmp_path, recursive=True)
        found_names = {p.name for p in found}

        assert found_names == {"valid.md"}

    def test_scan_documents_nonexistent_directory_raises(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            scan_documents(non_existent)

    def test_scan_documents_non_recursive(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "top.md").write_text("top", encoding="utf-8")
        (tmp_path / "sub" / "nested.md").write_text("nested", encoding="utf-8")

        found_recursive = scan_documents(tmp_path, recursive=True)
        assert len(found_recursive) == 2

        found_top_only = scan_documents(tmp_path, recursive=False)
        assert len(found_top_only) == 1
        assert found_top_only[0].name == "top.md"


class TestCollectionDetection:
    """Tests for auto-tagging collection from folder hierarchy or overrides."""

    def test_detect_collection_from_subfolder(self, tmp_path: Path):
        base = tmp_path / "plant_disease_docs"
        base.mkdir()
        tomato_dir = base / "tomato"
        tomato_dir.mkdir()
        file_path = tomato_dir / "early_blight_guide.md"
        file_path.write_text("content", encoding="utf-8")

        col = detect_collection(file_path, base)
        assert col == "tomato"

    def test_detect_collection_nested_subfolder(self, tmp_path: Path):
        base = tmp_path / "plant_disease_docs"
        base.mkdir()
        pepper_dir = base / "bell_pepper"
        pepper_dir.mkdir()
        file_path = pepper_dir / "bacterial_spot.md"
        file_path.write_text("content", encoding="utf-8")

        col = detect_collection(file_path, base)
        assert col == "bell_pepper"

    def test_detect_collection_at_root(self, tmp_path: Path):
        base = tmp_path / "plant_disease_docs"
        base.mkdir()
        file_path = base / "treatment_matrix.csv"
        file_path.write_text("content", encoding="utf-8")

        col = detect_collection(file_path, base)
        assert col == "plant_disease_docs"

    def test_detect_collection_explicit_override(self, tmp_path: Path):
        base = tmp_path / "docs"
        base.mkdir()
        sub = base / "tomato"
        sub.mkdir()
        file_path = sub / "guide.md"
        file_path.write_text("content", encoding="utf-8")

        col = detect_collection(file_path, base, explicit_collection="custom_crop")
        assert col == "custom_crop"


class TestDocumentParsers:
    """Tests for Markdown, CSV, and general document parsing."""

    def test_parse_markdown_file(self, tmp_path: Path):
        md_file = tmp_path / "early_blight.md"
        md_content = """# Tomato Early Blight Guide
## Symptoms
Concentric rings develop on lower leaves.
## Treatment
Apply Chlorothalonil 75% WP at 2.0g/L or Copper Hydroxide.
"""
        md_file.write_text(md_content, encoding="utf-8")

        chunks = parse_markdown_file(
            file_path=md_file,
            document_id="doc-123",
            collection="tomato",
            tenant_id=42,
            chunk_size=500,
            chunk_overlap=50,
        )

        assert len(chunks) >= 1
        first = chunks[0]
        assert first.document_id == "doc-123"
        assert first.metadata["source"] == "markdown"
        assert first.metadata["collection"] == "tomato"
        assert first.metadata["tenant_id"] == 42
        assert first.metadata["file_name"] == "early_blight.md"
        assert "Concentric rings" in first.text or "Tomato Early Blight" in first.text

    def test_parse_markdown_empty_file(self, tmp_path: Path):
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding="utf-8")

        chunks = parse_markdown_file(empty_file, "doc-0", "tomato")
        assert chunks == []

    def test_parse_csv_file_generates_fact_cards_and_tables(self, tmp_path: Path):
        csv_file = tmp_path / "treatment_matrix.csv"
        csv_content = """crop,disease,pathogen_type,organic_remedy,chemical_active_ingredient,dosage_per_liter,spray_interval_days,pre_harvest_interval_days,safety_precautions
tomato,early blight,Fungal,Copper octanoate / Bacillus subtilis,Chlorothalonil 75% WP,2.0g/L,7-10,7,Wear gloves and goggles.
potato,late blight,Oomycete,Copper hydroxide,Mandipropamid 23.4% SC,0.8ml/L,5-7,14,Destroy infected volunteer plants.
"""
        csv_file.write_text(csv_content, encoding="utf-8")

        chunks = parse_csv_file(
            file_path=csv_file,
            document_id="matrix-doc",
            collection="matrix",
            tenant_id=1,
        )

        assert len(chunks) >= 3  # 2 record cards + 1 table chunk

        # Check semantic card chunk
        record_chunks = [c for c in chunks if c.metadata.get("content_type") == "text/csv-record"]
        assert len(record_chunks) == 2
        assert "Tomato" in record_chunks[0].text
        assert "Early Blight" in record_chunks[0].text
        assert "Chlorothalonil" in record_chunks[0].text
        assert record_chunks[0].metadata["crop"] == "tomato"
        assert record_chunks[0].metadata["disease"] == "early blight"

        # Check table chunk
        table_chunks = [c for c in chunks if c.metadata.get("content_type") == "text/markdown-table"]
        assert len(table_chunks) >= 1
        assert "| crop | disease |" in table_chunks[0].text
        assert "| tomato | early blight |" in table_chunks[0].text

    def test_parse_document_file_routing(self, tmp_path: Path):
        md_file = tmp_path / "note.md"
        md_file.write_text("# Note\nContent", encoding="utf-8")

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2", encoding="utf-8")

        other_file = tmp_path / "image.png"
        other_file.write_bytes(b"\x89PNG")

        assert len(parse_document_file(md_file, "id1", "col")) >= 1
        assert len(parse_document_file(csv_file, "id2", "col")) >= 1
        assert parse_document_file(other_file, "id3", "col") == []


class TestBatchEmbedding:
    """Tests for batch embedding generation and tokenizer splitting."""

    def test_generate_batch_embeddings_with_mock_model(self):
        mock_model = MockSentenceTransformer(dim=4)
        chunks = [
            DocumentChunk(
                chunk_id=f"chunk-{i}",
                document_id="doc-1",
                chunk_index=i,
                text=f"This is chunk content number {i}",
                metadata={"source": "test", "index": i},
            )
            for i in range(10)
        ]

        embedded = generate_batch_embeddings(chunks, batch_size=4, model=mock_model)

        assert len(embedded) == 10
        for ec in embedded:
            assert isinstance(ec, EmbeddedChunk)
            assert len(ec.embedding) == 4
            assert "text" in ec.metadata
            assert ec.document_id == "doc-1"

    def test_split_oversized_chunks_with_tokenizer_mock(self):
        mock_model = MagicMock()
        mock_model.max_seq_length = 10
        # Mock tokenizer that splits long strings into 30 tokens
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": list(range(24)),
            "offset_mapping": [(i * 5, (i + 1) * 5) for i in range(24)],
        }
        mock_model.tokenizer = mock_tokenizer

        chunk = DocumentChunk(
            chunk_id="c-long",
            document_id="d-1",
            chunk_index=0,
            text="A" * 120,
            metadata={"source": "test"},
        )

        safe = split_oversized_chunks([chunk], mock_model)
        assert len(safe) >= 2


class TestCheckpointing:
    """Tests for resume checkpointing and file change detection."""

    def test_checkpoint_manager_lifecycle(self, tmp_path: Path):
        chk_file = tmp_path / ".checkpoint.json"
        base_dir = tmp_path / "data"
        base_dir.mkdir()

        doc_file = base_dir / "test.md"
        doc_file.write_text("Initial text", encoding="utf-8")

        mgr = CheckpointManager(chk_file)
        assert not mgr.is_file_processed(doc_file, base_dir)

        mgr.record_file(doc_file, base_dir, "tomato", 5)
        assert mgr.is_file_processed(doc_file, base_dir)

        # Reload from disk
        mgr2 = CheckpointManager(chk_file)
        assert mgr2.is_file_processed(doc_file, base_dir)

        # Clear
        mgr2.clear()
        assert not mgr2.is_file_processed(doc_file, base_dir)

    def test_checkpoint_detects_file_modification(self, tmp_path: Path):
        chk_file = tmp_path / ".checkpoint.json"
        base_dir = tmp_path / "data"
        base_dir.mkdir()

        doc_file = base_dir / "test.md"
        doc_file.write_text("Version 1 content", encoding="utf-8")

        mgr = CheckpointManager(chk_file)
        mgr.record_file(doc_file, base_dir, "tomato", 3)
        assert mgr.is_file_processed(doc_file, base_dir)

        # Modify file content
        doc_file.write_text("Version 2 changed content is different", encoding="utf-8")
        assert not mgr.is_file_processed(doc_file, base_dir)


class TestBulkIngestionFlow:
    """Integration-style tests for BulkIngester with mock and real vector stores."""

    def test_bulk_ingest_directory_with_faiss(self, tmp_path: Path):
        if faiss is None:
            pytest.skip("faiss not installed in host environment")

        # Create mock knowledge base folder
        kb_dir = tmp_path / "plant_disease_docs"
        (kb_dir / "tomato").mkdir(parents=True)
        (kb_dir / "apple").mkdir(parents=True)

        f1 = kb_dir / "tomato" / "early_blight.md"
        f1.write_text(
            "# Tomato Early Blight\nSymptoms include concentric rings on leaves.\nTreatment is Chlorothalonil 2g/L.",
            encoding="utf-8",
        )

        f2 = kb_dir / "apple" / "apple_scab.md"
        f2.write_text(
            "# Apple Scab Guide\nOlive green velvety spots on foliage.\nApply Captan 50WP at 2.5g/L.",
            encoding="utf-8",
        )

        f3 = kb_dir / "treatment_dosage_matrix.csv"
        f3.write_text(
            "crop,disease,pathogen_type,organic_remedy,chemical_active_ingredient,dosage_per_liter,spray_interval_days,pre_harvest_interval_days,safety_precautions\n"
            "tomato,early blight,Fungal,Bacillus subtilis,Chlorothalonil 75% WP,2.0g/L,7-10,7,Wear PPE.\n",
            encoding="utf-8",
        )

        index_path = tmp_path / "index.faiss"
        meta_path = tmp_path / "metadata.json"
        chk_path = tmp_path / "checkpoint.json"

        store = FAISSVectorStore(index_path=index_path, metadata_path=meta_path)

        mock_model = MockSentenceTransformer(dim=4)

        with patch("scripts.bulk_ingest.get_embedding_model", return_value=mock_model):
            ingester = BulkIngester(
                vector_store=store,
                batch_size=2,
                chunk_size=500,
                checkpoint_path=chk_path,
            )

            stats = ingester.ingest_directory(kb_dir)

            assert stats.total_files_scanned == 3
            assert stats.files_processed == 3
            assert stats.files_skipped == 0
            assert stats.total_vectors_indexed > 0
            assert store.total_vectors() > 0

            # Verify FAISS store can search
            query_vec = [1.0, 0.0, 0.0, 0.0]
            results = store.search(query_vec, top_k=5)
            assert len(results) > 0

            # Verify collection tags are present in metadata
            collections = {r.metadata.get("collection") for r in results if "collection" in r.metadata}
            assert len(collections) > 0

            # Run a second time: files should be SKIPPED via checkpoint cache
            stats2 = ingester.ingest_directory(kb_dir)
            assert stats2.files_skipped == 3
            assert stats2.files_processed == 0

            # Run with force=True: all files should be re-ingested
            ingester.force = True
            stats3 = ingester.ingest_directory(kb_dir)
            assert stats3.files_processed == 3
            assert stats3.files_skipped == 0

    def test_bulk_ingest_error_resilience_skips_bad_file(self, tmp_path: Path):
        if faiss is None:
            pytest.skip("faiss not installed in host environment")

        kb_dir = tmp_path / "docs"
        kb_dir.mkdir()

        good_file = kb_dir / "good.md"
        good_file.write_text("# Good Document\nValid content", encoding="utf-8")

        bad_file = kb_dir / "bad.md"
        bad_file.write_text("# Bad Document", encoding="utf-8")

        store = FAISSVectorStore(
            index_path=tmp_path / "idx.faiss", metadata_path=tmp_path / "meta.json"
        )
        mock_model = MockSentenceTransformer(dim=4)

        # Mock parse_document_file to raise an exception for bad.md
        original_parse = parse_document_file

        def failing_parse(file_path: Path, *args, **kwargs):
            if file_path.name == "bad.md":
                raise ValueError("Simulated parsing crash on corrupt file")
            return original_parse(file_path, *args, **kwargs)

        with patch("scripts.bulk_ingest.get_embedding_model", return_value=mock_model):
            with patch("scripts.bulk_ingest.parse_document_file", side_effect=failing_parse):
                ingester = BulkIngester(
                    vector_store=store,
                    batch_size=2,
                    checkpoint_path=tmp_path / "chk.json",
                )
                stats = ingester.ingest_directory(kb_dir)

                assert stats.total_files_scanned == 2
                assert stats.files_processed == 1
                assert stats.files_failed == 1
                assert stats.total_vectors_indexed > 0

    def test_live_treatment_dosage_matrix_parsing(self):
        # Verify the actual project treatment_dosage_matrix.csv parses all 38 classes
        matrix_path = Path(__file__).resolve().parents[2] / "data" / "plant_disease_docs" / "treatment_dosage_matrix.csv"
        if not matrix_path.exists():
            pytest.skip("treatment_dosage_matrix.csv not yet at data/plant_disease_docs/")

        chunks = parse_csv_file(
            file_path=matrix_path,
            document_id="matrix-test-id",
            collection="treatment_matrix",
        )

        record_chunks = [c for c in chunks if c.metadata.get("content_type") == "text/csv-record"]
        # Exactly 38 LeafSense classes!
        assert len(record_chunks) == 38

        crops = {c.metadata["crop"] for c in record_chunks}
        assert "tomato" in crops
        assert "apple" in crops
        assert "corn" in crops
        assert "potato" in crops
        assert "grape" in crops
        assert "bell pepper" in crops
        assert "orange" in crops
        assert "squash" in crops
        assert "cherry" in crops
        assert "peach" in crops
        assert "strawberry" in crops
