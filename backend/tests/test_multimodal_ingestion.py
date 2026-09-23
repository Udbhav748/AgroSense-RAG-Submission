"""Tests for Track 3 multi-modal RAG: image extraction (Phase 1),
Gemini captioning (Phase 2), vision-grounded QA (Phase 3), table
extraction (Phase 5).

All tests are offline — real PDFs are built in-process with PyMuPDF
(already installed), captioning/QA use a fake LLM client, and embedding
is stubbed so the sentence-transformers model is never loaded. OCR is
stubbed unavailable so the tesseract binary is never invoked.
"""

import asyncio
import io

import fitz
import pytest
from PIL import Image

from app.core.config import settings
from app.core.exceptions import LLMConfigurationError
from app.models.document import (
    EmbeddedChunk,
    ExtractedImage,
    RetrievedChunk,
)
from app.services import document_service
from app.services.chunking_service import chunk_text
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import extract_images_from_pdf
from app.services.gemini_client import GeminiClient
from app.services.image_captioning_service import (
    caption_images,
    load_image_manifest,
    write_image_manifest,
)
from app.services.table_extraction_service import extract_tables_from_pdf
from app.services.vision_qa_service import try_vision_qa


def _png_bytes(size=(300, 200), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _make_text_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(page.rect, text)
    doc.save(path)
    doc.close()


class FakeVisionLLM:
    """Stands in for GeminiClient: captions without a network call, and can
    be configured to fail like an unsupported/non-vision client."""

    def __init__(self, caption="A red square on a white background."):
        self.caption = caption
        self.vision_calls = []
        self.fail_with = None

    def generate(self, prompt: str) -> str:
        return "stub"

    def generate_stream(self, prompt: str):
        yield "stub"

    def generate_with_image(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png"):
        self.vision_calls.append({"prompt": prompt, "image_bytes": image_bytes, "mime_type": mime_type})
        if self.fail_with is not None:
            raise self.fail_with
        return self.caption


class FakeVectorStore:
    """In-memory VectorStore stand-in: records everything added, returns
    nothing on search (retrieval behavior is stubbed per-test)."""

    def __init__(self):
        self.added = []

    def create_index(self, dimension: int) -> None:
        pass

    def add_embeddings(self, embedded_chunks) -> None:
        self.added.extend(embedded_chunks)

    def search(self, query_vector, top_k, tenant_id=None):
        return []

    def delete_document(self, document_id: str) -> int:
        return 0

    def get_chunks_by_document(self, document_id: str, tenant_id=None):
        return []

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass

    def total_vectors(self) -> int:
        return len(self.added)


@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    """Stub tesseract unavailable so OCR never runs — image-only pages
    keep their thin/empty native text and the low_text_pages decision
    still fires, deterministically."""
    monkeypatch.setattr(document_service, "_tesseract_available", lambda: False)


class TestImageExtraction:
    def test_pdf_with_embedded_image_yields_image_record(self, tmp_path):
        path = tmp_path / "with_image.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(page.rect, "A page of text." * 20)
        image_page = doc.new_page()
        image_page.insert_image(fitz.Rect(50, 50, 350, 250), stream=_png_bytes())
        doc.save(path)
        doc.close()

        records = extract_images_from_pdf("doc-1", path)

        figures = [r for r in records if r["content_type"] == "figure"]
        assert figures, "expected at least one embedded-figure record"
        fig = figures[0]
        assert fig["document_id"] == "doc-1"
        assert fig["page_number"] == 2
        assert fig["mime_type"] == "image/png"
        assert fig["width"] >= 1 and fig["height"] >= 1
        assert fig["byte_size"] > 0
        assert fig["data"].startswith(b"\x89PNG")

    def test_tiny_image_below_min_side_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "image_min_side_px", 100)
        path = tmp_path / "tiny.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_image(fitz.Rect(10, 10, 40, 40), stream=_png_bytes(size=(30, 30)))
        doc.save(path)
        doc.close()

        records = extract_images_from_pdf("doc-1", path)

        assert all(r["content_type"] != "figure" for r in records)

    def test_low_text_page_produces_full_page_raster(self, tmp_path):
        path = tmp_path / "scanned.pdf"
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_image(page.rect, stream=_png_bytes(size=(400, 300)))
        doc.save(path)
        doc.close()

        records = extract_images_from_pdf("doc-1", path)

        page_rasters = [r for r in records if r["content_type"] == "page"]
        assert page_rasters, "expected a page raster for the image-only page"
        assert page_rasters[0]["page_number"] == 1
        assert page_rasters[0]["mime_type"] == "image/png"

    def test_image_count_is_capped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "image_max_count_per_document", 1)
        path = tmp_path / "many.pdf"
        doc = fitz.open()
        page = doc.new_page()
        for i in range(3):
            page.insert_image(
                fitz.Rect(20 + i * 100, 50, 100 + i * 100, 150),
                stream=_png_bytes(size=(80, 100)),
            )
        doc.save(path)
        doc.close()

        records = extract_images_from_pdf("doc-1", path)

        assert len(records) <= 1


class TestTextChunking:
    def test_chunk_text_sets_modality_metadata(self):
        chunks = chunk_text(
            text="A caption of a figure.",
            document_id="doc-1",
            tenant_id=7,
            source="image_caption",
            content_type="image_caption",
            page_number=3,
            image_id="doc-1_img_5",
        )

        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta["source"] == "image_caption"
        assert meta["content_type"] == "image_caption"
        assert meta["page_number"] == 3
        assert meta["tenant_id"] == 7
        assert meta["image_id"] == "doc-1_img_5"
        assert meta["document_id"] == "doc-1"
        assert chunks[0].chunk_index == 0
        assert chunks[0].text == "A caption of a figure."


class TestImageCaptioning:
    def _persisted_image(self, tmp_path, monkeypatch, **overrides):
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))
        monkeypatch.setattr(settings, "image_storage_dir_name", "images")
        image_dir = tmp_path / "images"
        image_dir.mkdir(exist_ok=True)
        data = _png_bytes()
        storage_path = "doc-1_img_5.png"
        (image_dir / storage_path).write_bytes(data)
        defaults = dict(
            image_id="doc-1_img_5",
            document_id="doc-1",
            page_number=2,
            content_type="figure",
            storage_path=storage_path,
            mime_type="image/png",
            width=300,
            height=200,
            byte_size=len(data),
        )
        defaults.update(overrides)
        return ExtractedImage(**defaults)

    def test_caption_produces_image_derived_chunk(self, tmp_path, monkeypatch):
        image = self._persisted_image(tmp_path, monkeypatch)
        llm = FakeVisionLLM(caption="A bar chart showing revenue growth.")

        chunks = caption_images([image], llm, document_id="doc-1", tenant_id=7)

        assert len(chunks) == 1
        assert chunks[0].text == "A bar chart showing revenue growth."
        meta = chunks[0].metadata
        assert meta["source"] == "image_caption"
        assert meta["content_type"] == "image_caption"
        assert meta["image_id"] == "doc-1_img_5"
        assert meta["page_number"] == 2
        assert len(llm.vision_calls) == 1
        assert llm.vision_calls[0]["mime_type"] == "image/png"
        assert "page 2" in llm.vision_calls[0]["prompt"].lower()

    def test_unsupported_client_degrades_to_no_chunks(self, tmp_path, monkeypatch):
        image = self._persisted_image(tmp_path, monkeypatch)
        llm = FakeVisionLLM()
        llm.fail_with = LLMConfigurationError("no vision support")

        chunks = caption_images([image], llm, document_id="doc-1", tenant_id=None)

        assert chunks == []

    def test_caption_failure_degrades_to_no_chunks(self, tmp_path, monkeypatch):
        image = self._persisted_image(tmp_path, monkeypatch)
        llm = FakeVisionLLM()
        llm.fail_with = RuntimeError("provider error")

        chunks = caption_images([image], llm, document_id="doc-1", tenant_id=None)

        assert chunks == []

    def test_page_rasters_are_not_captioned(self, tmp_path, monkeypatch):
        image = self._persisted_image(tmp_path, monkeypatch, content_type="page")
        llm = FakeVisionLLM()

        chunks = caption_images([image], llm, document_id="doc-1", tenant_id=None)

        assert chunks == []
        assert llm.vision_calls == []

    def test_caption_is_truncated_to_config_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "image_caption_max_chars", 10)
        image = self._persisted_image(tmp_path, monkeypatch)
        llm = FakeVisionLLM(caption="a" * 200)

        chunks = caption_images([image], llm, document_id="doc-1", tenant_id=None)

        assert len(chunks) == 1
        assert chunks[0].text == "a" * 10


class TestTableExtraction:
    def _make_grid_pdf(self, path):
        doc = fitz.open()
        page = doc.new_page(width=400, height=400)
        y = 50
        for _ in range(4):
            page.draw_line((20, y), (380, y))
            y += 40
        x = 20
        for _ in range(4):
            page.draw_line((x, 50), (x, y))
            x += 90
        page.insert_text((30, 70), "Name")
        page.insert_text((120, 70), "Qty")
        page.insert_text((30, 110), "Widget")
        page.insert_text((120, 110), "5")
        doc.save(path)
        doc.close()

    def test_extracts_table_from_drawn_grid(self, tmp_path):
        path = tmp_path / "table.pdf"
        self._make_grid_pdf(path)

        tables = extract_tables_from_pdf("doc-1", path)

        assert len(tables) >= 1
        table = tables[0]
        assert table.document_id == "doc-1"
        assert table.page_number == 1
        assert table.column_count >= 1
        assert "Name" in table.markdown
        assert "| ---" in table.markdown

    def test_table_cap_is_enforced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "table_max_count_per_document", 1)
        path = tmp_path / "table.pdf"
        # two pages each with a detected grid table
        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page(width=300, height=300)
            page.draw_line((20, 50), (280, 50))
            page.draw_line((20, 90), (280, 90))
            page.draw_line((20, 50), (20, 90))
            page.draw_line((150, 50), (150, 90))
            page.insert_text((30, 75), "H1")
            page.insert_text((160, 75), "V1")
        doc.save(path)
        doc.close()

        tables = extract_tables_from_pdf("doc-1", path)

        assert len(tables) <= 1


class TestGeminiImageInput:
    def test_generate_with_image_sends_inline_blob(self, monkeypatch):
        monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
        client = GeminiClient()

        captured = {}

        class _FakeResponse:
            text = "The image shows a red square."
            usage_metadata = None

        def _fake_generate_content(**kwargs):
            captured["kwargs"] = kwargs
            return _FakeResponse()

        monkeypatch.setattr(client._client.models, "generate_content", _fake_generate_content)

        image_bytes = b"\x89PNG fake bytes"
        result = client.generate_with_image("What is in this image?", image_bytes, "image/png")

        assert result == "The image shows a red square."
        contents = captured["kwargs"]["contents"]
        part = contents[0].parts[0]
        assert part.inline_data.data == image_bytes
        assert part.inline_data.mime_type == "image/png"
        assert contents[0].parts[1].text == "What is in this image?"

    def test_empty_response_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
        client = GeminiClient()

        class _FakeEmpty:
            text = None
            usage_metadata = None

        monkeypatch.setattr(
            client._client.models,
            "generate_content",
            lambda **kwargs: _FakeEmpty(),
        )

        from app.core.exceptions import LLMEmptyResponseError

        with pytest.raises(LLMEmptyResponseError):
            client.generate_with_image("prompt", b"bytes")


class TestVisionQA:
    def _setup_images(self, tmp_path, monkeypatch, count=2):
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))
        monkeypatch.setattr(settings, "image_storage_dir_name", "images")
        image_dir = tmp_path / "images"
        image_dir.mkdir(exist_ok=True)
        for n in range(1, count + 1):
            (image_dir / f"doc-1_page_{n}.png").write_bytes(_png_bytes())

    def test_returns_answer_from_page_raster(self, tmp_path, monkeypatch):
        self._setup_images(tmp_path, monkeypatch, count=2)
        chunks = [
            RetrievedChunk(
                chunk_id="c1", document_id="doc-1", text="thin", score=0.45, metadata={}
            )
        ]
        llm = FakeVisionLLM(caption="The plant shows leaf spot disease.")

        answer = try_vision_qa("What disease does the plant have?", chunks, llm)

        assert answer == "The plant shows leaf spot disease."
        assert len(llm.vision_calls) == 1
        assert llm.vision_calls[0]["mime_type"] == "image/png"

    def test_returns_none_without_page_rasters(self, tmp_path, monkeypatch):
        chunks = [
            RetrievedChunk(
                chunk_id="c1", document_id="doc-1", text="thin", score=0.45, metadata={}
            )
        ]
        llm = FakeVisionLLM()

        answer = try_vision_qa("question", chunks, llm)

        assert answer is None
        assert llm.vision_calls == []

    def test_empty_chunks_returns_none(self, tmp_path, monkeypatch):
        llm = FakeVisionLLM()

        assert try_vision_qa("question", [], llm) is None

    def test_max_pages_caps_vision_calls(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "vision_qa_max_pages", 3)
        self._setup_images(tmp_path, monkeypatch, count=5)
        chunks = [
            RetrievedChunk(
                chunk_id="c1", document_id="doc-1", text="thin", score=0.45, metadata={}
            )
        ]
        llm = FakeVisionLLM()
        llm.fail_with = RuntimeError("vision unavailable")

        answer = try_vision_qa("question", chunks, llm)

        assert answer is None
        assert len(llm.vision_calls) == 3


class TestImageManifest:
    """The per-document manifest backing GET /documents/{id}/images: what
    ingestion writes must be exactly what the listing endpoint reads."""

    def _persisted_image(self, tmp_path, monkeypatch, *, image_id="doc-1_img_1", page_number=2):
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))
        monkeypatch.setattr(settings, "image_storage_dir_name", "images")
        data = _png_bytes()
        image_dir = tmp_path / "images"
        image_dir.mkdir(exist_ok=True)
        (image_dir / f"{image_id}.png").write_bytes(data)
        return ExtractedImage(
            image_id=image_id,
            document_id="doc-1",
            page_number=page_number,
            content_type="figure",
            storage_path=f"{image_id}.png",
            mime_type="image/png",
            width=300,
            height=200,
            byte_size=len(data),
        )

    def test_round_trip(self, tmp_path, monkeypatch):
        image = self._persisted_image(tmp_path, monkeypatch)

        write_image_manifest([image])
        loaded = load_image_manifest("doc-1")

        assert len(loaded) == 1
        assert loaded[0].image_id == "doc-1_img_1"
        assert loaded[0].page_number == 2
        assert loaded[0].storage_path == "doc-1_img_1.png"

    def test_missing_manifest_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))
        monkeypatch.setattr(settings, "image_storage_dir_name", "images")

        assert load_image_manifest("no-such-doc") == []

    def test_corrupt_manifest_degrades_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))
        monkeypatch.setattr(settings, "image_storage_dir_name", "images")
        image_dir = tmp_path / "images"
        image_dir.mkdir(exist_ok=True)
        (image_dir / "doc-1_images.json").write_text("not json {{{")

        assert load_image_manifest("doc-1") == []

    def test_empty_write_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))
        monkeypatch.setattr(settings, "image_storage_dir_name", "images")

        write_image_manifest([])

        assert load_image_manifest("doc-1") == []


class TestProcessingPipeline:
    """End-to-end ingestion with the multi-modal flags enabled: image
    extraction + captioning + tables all flow through the same
    chunking/embedding/store pipeline, and the response reports counts.
    Embedding is stubbed so no model loads."""

    def _make_rich_pdf(self, path):
        doc = fitz.open()
        text_page = doc.new_page()
        text_page.insert_textbox(text_page.rect, "Substantial body text. " * 30)
        image_page = doc.new_page()
        image_page.insert_image(fitz.Rect(50, 50, 350, 250), stream=_png_bytes(size=(300, 200)))
        table_page = doc.new_page(width=400, height=400)
        y = 50
        for _ in range(4):
            table_page.draw_line((20, y), (380, y))
            y += 40
        x = 20
        for _ in range(4):
            table_page.draw_line((x, 50), (x, y))
            x += 90
        table_page.insert_text((30, 70), "Name")
        table_page.insert_text((120, 70), "Qty")
        table_page.insert_text((30, 110), "Widget")
        table_page.insert_text((120, 110), "5")
        doc.save(path)
        doc.close()

    def _run_pipeline(self, tmp_path, monkeypatch, *, extract=True, captions=True, tables=True):
        monkeypatch.setattr(settings, "image_extraction_enabled", extract)
        monkeypatch.setattr(settings, "image_captioning_enabled", captions)
        monkeypatch.setattr(settings, "table_extraction_enabled", tables)
        monkeypatch.setattr(settings, "data_dir_override", str(tmp_path))

        pdf_path = tmp_path / "rich.pdf"
        self._make_rich_pdf(pdf_path)
        pdf_bytes = pdf_path.read_bytes()

        from fastapi import UploadFile

        upload = UploadFile(file=io.BytesIO(pdf_bytes), filename="rich.pdf")

        fake_saved_path = tmp_path / "uploaded" / "doc-1.pdf"
        fake_saved_path.parent.mkdir(parents=True, exist_ok=True)
        fake_saved_path.write_bytes(pdf_bytes)

        async def _fake_save_uploaded_file(file):
            return {
                "document_id": "doc-1",
                "original_filename": file.filename,
                "stored_filename": "doc-1.pdf",
                "file_size": len(pdf_bytes),
            }

        def _fake_generate_embeddings(chunks):
            return [
                EmbeddedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    embedding=[1.0] + [0.0] * 7,
                    metadata={**c.metadata, "text": c.text},
                )
                for c in chunks
            ]

        monkeypatch.setattr(
            "app.services.document_processing_service.save_uploaded_file", _fake_save_uploaded_file
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.UPLOAD_DIR", fake_saved_path.parent
        )
        monkeypatch.setattr(
            "app.services.document_processing_service.generate_embeddings",
            _fake_generate_embeddings,
        )

        store = FakeVectorStore()
        llm = FakeVisionLLM(caption="A red square figure on the page.")
        service = DocumentProcessingService(store, llm_client=llm)
        return service, store, llm

    def test_pipeline_indexes_caption_and_table_chunks(self, tmp_path, monkeypatch):
        service, store, llm = self._run_pipeline(tmp_path, monkeypatch)

        response = asyncio.run(service.process(_make_upload(tmp_path), tenant_id=7))

        assert response.total_images >= 1
        assert response.images_captioned >= 1
        assert response.total_tables >= 1

        assert len(load_image_manifest("doc-1")) >= 1

        sources = {c.metadata.get("source") for c in store.added}
        assert "image_caption" in sources
        assert "table" in sources
        assert "pdf" in sources

        caption_chunk = next(c for c in store.added if c.metadata.get("source") == "image_caption")
        assert caption_chunk.metadata["page_number"] == 2

    def test_pipeline_is_unchanged_when_flags_disabled(self, tmp_path, monkeypatch):
        service, store, llm = self._run_pipeline(
            tmp_path, monkeypatch, extract=False, captions=False, tables=False
        )

        response = asyncio.run(service.process(_make_upload(tmp_path), tenant_id=7))

        assert response.total_images == 0
        assert response.images_captioned == 0
        assert response.total_tables == 0
        sources = {c.metadata.get("source") for c in store.added}
        assert sources == {"pdf"}


def _make_upload(tmp_path):
    from fastapi import UploadFile

    pdf_path = tmp_path / "rich.pdf"
    return UploadFile(file=io.BytesIO(pdf_path.read_bytes()), filename="rich.pdf")
