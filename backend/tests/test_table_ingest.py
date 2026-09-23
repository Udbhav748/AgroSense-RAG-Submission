"""Unit and integration tests for layout-aware tabular ingestion and parsing.

Verifies:
1. Parsing of CSV files and Markdown tables into atomic semantic units:
   `[TABLE ROW: Crop={crop} | Disease={disease} | Active Ingredient={chemical} | Rate={dosage} | Spray Interval={interval} | PHI={phi_days}]`
2. Structured metadata extraction (`table_type="dosage_matrix"`, `crop`, `disease`, `active_ingredient`, `phi_days`, etc.).
3. Preservation of sliding-window boundaries (table rows are never broken across arbitrary character windows).
4. Full integration with BulkIngester and FAISS vector storage.
5. Live project `treatment_dosage_matrix.csv` parsing across all 38 disease classes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.models.document import DocumentChunk, EmbeddedChunk
from app.services.document_parser import (
    LayoutAwareDocumentParser,
    detect_context_from_text,
    extract_markdown_tables,
    extract_phi_days,
    format_table_row_unit,
    normalize_column_name,
    parse_csv_content,
    parse_layout_aware_markdown,
    parse_tabular_row_dict,
)
from app.services.faiss_vector_store import FAISSVectorStore, faiss
from scripts.bulk_ingest import BulkIngester, parse_csv_file, parse_markdown_file

FAKE_DIM = 4


class MockSentenceTransformer:
    """Lightweight mock for SentenceTransformer embeddings."""

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
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).astype("float32")


class TestTabularRowParsing:
    """Tests verifying atomic semantic unit generation and normalization."""

    def test_format_table_row_unit_exact_format(self):
        unit = format_table_row_unit(
            crop="apple",
            disease="apple scab",
            active_ingredient="Captan 50% WP or Difenoconazole 25% EC",
            rate="2.5g/L (Captan)",
            spray_interval="7-10",
            phi_days=14,
        )
        expected = (
            "[TABLE ROW: Crop=apple | Disease=apple scab | "
            "Active Ingredient=Captan 50% WP or Difenoconazole 25% EC | "
            "Rate=2.5g/L (Captan) | Spray Interval=7-10 | PHI=14]"
        )
        assert unit == expected

    def test_format_table_row_unit_with_extra_fields(self):
        unit = format_table_row_unit(
            crop="tomato",
            disease="early blight",
            active_ingredient="Chlorothalonil 75% WP",
            rate="2.0g/L",
            spray_interval="7-10",
            phi_days=7,
            extra_fields={"safety_precautions": "Wear PPE", "pathogen_type": "Fungal"},
        )
        assert unit.startswith("[TABLE ROW: Crop=tomato | Disease=early blight")
        assert "Active Ingredient=Chlorothalonil 75% WP" in unit
        assert "Rate=2.0g/L" in unit
        assert "Spray Interval=7-10" in unit
        assert "PHI=7" in unit
        assert "Safety Precautions=Wear PPE" in unit
        assert "Pathogen Type=Fungal" in unit

    def test_parse_tabular_row_dict_aliases(self):
        # Test varied header names
        row = {
            "Host Crop": "Grape",
            "Target Disease": "Black Rot",
            "Chemical": "Myclobutanil 20% WP",
            "Dosage Rate / Liter": "0.5ml/L",
            "Application Interval": "10-14 days",
            "PHI (Days)": "66",
            "Mode of Action": "FRAC 3",
        }
        parsed = parse_tabular_row_dict(row, table_type="dosage_matrix")

        assert parsed.crop == "Grape"
        assert parsed.disease == "Black Rot"
        assert parsed.active_ingredient == "Myclobutanil 20% WP"
        assert parsed.rate == "0.5ml/L"
        assert parsed.spray_interval == "10-14 days"
        assert parsed.phi_days == 66
        assert parsed.table_type == "dosage_matrix"
        assert parsed.metadata["crop"] == "grape"
        assert parsed.metadata["disease"] == "black rot"
        assert parsed.metadata["active_ingredient"] == "Myclobutanil 20% WP"
        assert parsed.metadata["phi_days"] == 66
        assert parsed.metadata["is_table"] is True
        assert "[TABLE ROW: Crop=Grape | Disease=Black Rot" in parsed.formatted_text

    def test_extract_phi_days(self):
        assert extract_phi_days("14") == 14
        assert extract_phi_days(7) == 7
        assert extract_phi_days("14 days") == 14
        assert extract_phi_days("0") == 0
        assert extract_phi_days(None) == 0
        assert extract_phi_days("N/A") == "N/A"

    def test_normalize_column_name(self):
        assert normalize_column_name("Chemical Active Ingredient") == "chemical_active_ingredient"
        assert normalize_column_name("PHI (Days)") == "phi_days"
        assert normalize_column_name("Dosage Rate / Liter") == "dosage_rate_/_liter"
        assert normalize_column_name("  **Crop**  ") == "crop"


class TestMarkdownTableDetectionAndParsing:
    """Tests for extracting and parsing Markdown tables."""

    def test_extract_markdown_tables_multitable(self):
        md_text = """# Comprehensive Guide: Tomato Early Blight
Some introductory text.

## 4. Organic IPM
| Control Measure | Specification / Agent | Application Method & Timing |
| :--- | :--- | :--- |
| Bio-fungicide | Bacillus subtilis | Foliar spray at 4ml/L |
| Copper Fungicide | Copper Octanoate | 2.0g/L every 7 days |

## 5. Chemical Fungicide Rotation
| FRAC Group | Active Ingredient | Trade Formulation | Dosage Rate / Liter | Application Interval | PHI (Days) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FRAC M05 | Chlorothalonil 75% WP | Bravo Weather Stik | 2.0 g/L | 7 to 10 days | 7 days |
| FRAC 11 | Azoxystrobin 23% SC | Quadris | 0.8 ml/L | 10 to 14 days | 0 days |

## 6. Prevention Checklist
- Rotate crops
"""
        tables = extract_markdown_tables(md_text)
        assert len(tables) == 2

        # First table (IPM)
        t1 = tables[0]
        assert "Organic IPM" in t1.section_heading
        assert len(t1.rows) == 2
        assert t1.rows[0]["Specification / Agent"] == "Bacillus subtilis"

        # Second table (Chemical Rotation)
        t2 = tables[1]
        assert "Chemical Fungicide Rotation" in t2.section_heading
        assert len(t2.rows) == 2
        assert t2.rows[0]["Active Ingredient"] == "Chlorothalonil 75% WP"
        assert t2.rows[0]["PHI (Days)"] == "7 days"

    def test_context_detection_from_markdown(self):
        md_text = """# Comprehensive Diagnostic & Management Guide: Tomato Early Blight
- **Crop**: Tomato (*Solanum lycopersicum*)
- **Disease**: Early Blight
"""
        ctx = detect_context_from_text(md_text)
        assert ctx["crop"] == "tomato"
        assert ctx["disease"] == "early blight"


class TestSlidingWindowBoundaryPreservation:
    """Tests verifying that table rows are never split across sliding window boundaries."""

    def test_sliding_window_preserves_table_rows(self, tmp_path: Path):
        md_file = tmp_path / "guide_with_table.md"
        # Create a document where a naive 100-character sliding window would cut table rows in half
        prose_lead = "Tomato early blight causes bullseye concentric rings on the lower leaves of solanaceous crops. " * 3
        table_content = """| FRAC Group | Active Ingredient | Dosage Rate / Liter | Application Interval | PHI (Days) |
| :--- | :--- | :--- | :--- | :--- |
| FRAC M05 | Chlorothalonil 75% WP | 2.0 g/L | 7 to 10 days | 7 days |
| FRAC 11 | Azoxystrobin 23% SC | 0.8 to 1.0 ml/L | 10 to 14 days | 0 days |
| FRAC 3 | Difenoconazole 25% EC | 0.5 ml/L | 10 to 14 days | 7 days |
"""
        prose_tail = "Always observe recommended rotation schedules to prevent fungicide resistance buildup. " * 3

        full_doc = f"# Tomato Early Blight Diagnostic Guide\n\n{prose_lead}\n\n## Chemical Management\n{table_content}\n\n## Conclusion\n{prose_tail}"
        md_file.write_text(full_doc, encoding="utf-8")

        # Parse with a small chunk size (120 characters)
        chunks = parse_markdown_file(
            file_path=md_file,
            document_id="tbl-preserve-test",
            collection="tomato",
            chunk_size=120,
            chunk_overlap=20,
        )

        table_row_chunks = [c for c in chunks if c.metadata.get("content_type") == "text/table-row"]
        assert len(table_row_chunks) == 3

        # Verify each table row is complete, intact, and holds all semantic associations
        for chunk in table_row_chunks:
            assert "[TABLE ROW:" in chunk.text
            assert "Crop=tomato" in chunk.text
            assert "Disease=early blight" in chunk.text
            assert "Active Ingredient=" in chunk.text
            assert "Rate=" in chunk.text
            assert "Spray Interval=" in chunk.text
            assert "PHI=" in chunk.text
            # Verify metadata
            assert chunk.metadata["table_type"] == "dosage_matrix"
            assert chunk.metadata["crop"] == "tomato"
            assert "early blight" in chunk.metadata["disease"]
            assert chunk.metadata["is_table"] is True
            assert "phi_days" in chunk.metadata

        # Verify specific chemicals
        active_ingredients = [c.metadata["active_ingredient"] for c in table_row_chunks]
        assert "Chlorothalonil 75% WP" in active_ingredients
        assert "Azoxystrobin 23% SC" in active_ingredients
        assert "Difenoconazole 25% EC" in active_ingredients

        # Verify chunk indices are strictly monotonic and sequential
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestCSVTabularParsing:
    """Tests for CSV parsing into atomic units and slices."""

    def test_parse_csv_file_atomic_units(self, tmp_path: Path):
        csv_file = tmp_path / "dosage.csv"
        csv_text = """crop,disease,pathogen_type,organic_remedy,chemical_active_ingredient,dosage_per_liter,spray_interval_days,pre_harvest_interval_days,safety_precautions
apple,apple scab,Fungal,Sulfur 80% WDG,Captan 50% WP or Difenoconazole 25% EC,2.5g/L,7-10,14,Wear full PPE.
orange,citrus greening,Bacterial,Horticultural mineral oil 1%,Imidacloprid 17.8% SL,0.5ml/L,21-30,15,Protect bees.
"""
        csv_file.write_text(csv_text, encoding="utf-8")

        chunks = parse_csv_file(
            file_path=csv_file,
            document_id="csv-test-doc",
            collection="general",
        )

        record_chunks = [c for c in chunks if c.metadata.get("content_type") == "text/csv-record"]
        assert len(record_chunks) == 2

        # Apple scab row
        c1 = record_chunks[0]
        assert "[TABLE ROW: Crop=apple | Disease=apple scab" in c1.text
        assert "Active Ingredient=Captan 50% WP or Difenoconazole 25% EC" in c1.text
        assert "Rate=2.5g/L" in c1.text
        assert "Spray Interval=7-10" in c1.text
        assert "PHI=14" in c1.text
        assert c1.metadata["table_type"] == "dosage_matrix"
        assert c1.metadata["crop"] == "apple"
        assert c1.metadata["disease"] == "apple scab"
        assert c1.metadata["active_ingredient"] == "Captan 50% WP or Difenoconazole 25% EC"
        assert c1.metadata["phi_days"] == 14
        assert c1.metadata["is_table"] is True
        assert c1.metadata["pathogen_type"] == "Fungal"
        assert c1.metadata["safety_precautions"] == "Wear full PPE."

        # Orange greening row
        c2 = record_chunks[1]
        assert "[TABLE ROW: Crop=orange | Disease=citrus greening" in c2.text
        assert c2.metadata["crop"] == "orange"
        assert c2.metadata["phi_days"] == 15
        assert c2.metadata["safety_precautions"] == "Protect bees."

    def test_parse_csv_content_empty(self):
        chunks = parse_csv_content("", "empty-doc")
        assert chunks == []

    def test_layout_aware_parser_class(self, tmp_path: Path):
        parser = LayoutAwareDocumentParser(chunk_size=300, chunk_overlap=30)
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("crop,disease,dosage_per_liter\ntomato,blight,2g/L", encoding="utf-8")

        chunks = parser.parse_file(csv_file, "doc-parser-1")
        assert len(chunks) >= 1
        assert chunks[0].metadata["crop"] == "tomato"


class TestBulkIngestionTabularIntegration:
    """Integration tests for BulkIngester with tabular documents and vector storage."""

    def test_bulk_ingest_dosage_matrix_and_markdown_tables(self, tmp_path: Path):
        if faiss is None:
            pytest.skip("faiss not installed in host environment")

        kb_dir = tmp_path / "plant_disease_docs"
        (kb_dir / "tomato").mkdir(parents=True)

        guide_file = kb_dir / "tomato" / "guide.md"
        guide_file.write_text(
            "# Tomato Disease Guide\n\n## Chemical Treatments\n"
            "| Active Ingredient | Dosage Rate / Liter | PHI (Days) |\n"
            "| :--- | :--- | :--- |\n"
            "| Chlorothalonil 75% WP | 2.0 g/L | 7 |\n",
            encoding="utf-8",
        )

        matrix_file = kb_dir / "treatment_dosage_matrix.csv"
        matrix_file.write_text(
            "crop,disease,pathogen_type,organic_remedy,chemical_active_ingredient,dosage_per_liter,spray_interval_days,pre_harvest_interval_days,safety_precautions\n"
            "tomato,early blight,Fungal,Bacillus subtilis,Chlorothalonil 75% WP,2.0g/L,7-10,7,Wear PPE.\n"
            "apple,apple scab,Fungal,Sulfur 80%,Captan 50% WP,2.5g/L,7-10,14,Wear PPE.\n",
            encoding="utf-8",
        )

        index_path = tmp_path / "tbl_index.faiss"
        meta_path = tmp_path / "tbl_meta.json"
        chk_path = tmp_path / "tbl_chk.json"

        store = FAISSVectorStore(index_path=index_path, metadata_path=meta_path)
        mock_model = MockSentenceTransformer(dim=4)

        with patch("scripts.bulk_ingest.get_embedding_model", return_value=mock_model):
            ingester = BulkIngester(
                vector_store=store,
                batch_size=4,
                chunk_size=500,
                checkpoint_path=chk_path,
            )

            stats = ingester.ingest_directory(kb_dir)

            assert stats.total_files_scanned == 2
            assert stats.files_processed == 2
            assert stats.files_failed == 0
            assert stats.total_vectors_indexed > 0
            assert store.total_vectors() > 0

            # Query vector store and inspect retrieved metadata
            query_vec = [1.0, 0.0, 0.0, 0.0]
            results = store.search(query_vec, top_k=10)
            assert len(results) > 0

            # Verify table metadata tags are present in indexed vectors
            table_results = [r for r in results if r.metadata.get("is_table") is True]
            assert len(table_results) > 0

            crops_indexed = {r.metadata.get("crop") for r in table_results if "crop" in r.metadata}
            assert "tomato" in crops_indexed
            assert "apple" in crops_indexed

            table_types = {r.metadata.get("table_type") for r in table_results if "table_type" in r.metadata}
            assert "dosage_matrix" in table_types

    def test_live_treatment_dosage_matrix_parsing_full_catalog(self):
        # Verify the actual project treatment_dosage_matrix.csv parses all 38 classes
        matrix_path = Path(__file__).resolve().parents[2] / "data" / "plant_disease_docs" / "treatment_dosage_matrix.csv"
        if not matrix_path.exists():
            pytest.skip("treatment_dosage_matrix.csv not located at data/plant_disease_docs/")

        chunks = parse_csv_file(
            file_path=matrix_path,
            document_id="matrix-live-catalog",
            collection="treatment_matrix",
        )

        record_chunks = [c for c in chunks if c.metadata.get("content_type") == "text/csv-record"]
        # Exactly 38 LeafSense disease/crop classes
        assert len(record_chunks) == 38

        for chunk in record_chunks:
            # Check atomic unit format
            assert chunk.text.startswith("Agricultural Treatment & Dosage Reference:")
            assert "[TABLE ROW: Crop=" in chunk.text
            assert "| Disease=" in chunk.text
            assert "| Active Ingredient=" in chunk.text
            assert "| Rate=" in chunk.text
            assert "| Spray Interval=" in chunk.text
            assert "| PHI=" in chunk.text

            # Check metadata fields
            meta = chunk.metadata
            assert meta["table_type"] == "dosage_matrix"
            assert meta["is_table"] is True
            assert meta["crop"] != ""
            assert meta["disease"] != ""
            assert "phi_days" in meta
            assert "active_ingredient" in meta
            assert "rate" in meta
            assert "spray_interval" in meta

        # Verify key crops are all represented
        crops = {c.metadata["crop"] for c in record_chunks}
        for expected_crop in [
            "apple",
            "cherry",
            "corn",
            "grape",
            "orange",
            "peach",
            "bell pepper",
            "potato",
            "squash",
            "strawberry",
            "tomato",
        ]:
            assert expected_crop in crops
