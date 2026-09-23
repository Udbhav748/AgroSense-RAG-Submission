"""Reusable domain models for the document processing pipeline.

These represent a document as it moves through upload, text extraction, and
chunking. They are consumed and produced by services (upload, extraction,
future chunking/embedding/retrieval) rather than being API wire formats
themselves.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UploadedDocument(BaseModel):
    document_id: str = Field(
        ..., description="Unique identifier assigned to the uploaded document."
    )
    original_filename: str = Field(
        ..., description="Filename as provided by the client at upload time."
    )
    stored_filename: str = Field(
        ..., description="UUID-based filename the document is stored under on disk."
    )
    file_size: int = Field(..., description="Size of the uploaded file, in bytes.")
    upload_timestamp: datetime = Field(
        ..., description="UTC timestamp of when the document was uploaded."
    )


class ExtractedDocument(BaseModel):
    document_id: str = Field(
        ..., description="Identifier of the document this extraction was produced from."
    )
    extracted_text: str = Field(
        ..., description="Full text extracted from the document, in page order."
    )
    total_pages: int = Field(..., description="Total number of pages in the source document.")
    extracted_characters: int = Field(..., description="Character count of extracted_text.")
    pages_ocred: int = Field(
        ...,
        description="Number of pages that had no extractable text layer and were recovered via OCR.",
    )


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for this chunk.")
    document_id: str = Field(
        ..., description="Identifier of the document this chunk was derived from."
    )
    chunk_index: int = Field(
        ..., description="Zero-based position of this chunk within the document."
    )
    text: str = Field(..., description="Text content of the chunk.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata about the chunk (e.g. page number, section).",
    )


class EmbeddedChunk(BaseModel):
    chunk_id: str = Field(
        ..., description="Identifier of the chunk this embedding was generated from."
    )
    document_id: str = Field(
        ..., description="Identifier of the document this chunk was derived from."
    )
    embedding: list[float] = Field(
        ..., description="Embedding vector generated from the chunk's text."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata carried over unchanged from the source DocumentChunk.",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(..., description="Identifier of the retrieved chunk.")
    document_id: str = Field(
        ..., description="Identifier of the document this chunk was derived from."
    )
    text: str = Field(..., description="Text content of the retrieved chunk.")
    score: float = Field(..., description="Similarity score between the query and this chunk.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with the chunk (e.g. chunk_index, total_chunks, source).",
    )


class WebSearchResult(BaseModel):
    title: str = Field(..., description="Title of the web search result.")
    url: str = Field(..., description="URL of the web search result.")
    snippet: str = Field(..., description="Snippet/summary text of the web search result.")


class VisionPrediction(BaseModel):
    raw_class: str = Field(
        ..., description="Raw class label as returned by LeafSense (e.g. 'Apple___Apple_scab')."
    )
    crop: str = Field(
        ..., description="Plain-language crop name mapped from raw_class (e.g. 'apple')."
    )
    disease: str = Field(
        ..., description="Plain-language disease name mapped from raw_class (e.g. 'apple scab')."
    )
    confidence: float = Field(
        ..., description="Model confidence for the predicted class, in [0, 1]."
    )
    low_confidence: bool = Field(
        ..., description="True if confidence is below Settings.vision_confidence_threshold."
    )
    engine: str = Field(
        default="leafsense",
        description="Inference engine that produced the prediction ('leafsense', 'gemini_vision', 'hybrid_consensus').",
    )
    heatmap_base64: str | None = Field(
        default=None,
        description="Base64-encoded PNG image of the leaf saliency / lesion heatmap overlay.",
    )
    infected_area_percentage: float | None = Field(
        default=None,
        description="Estimated percentage of leaf surface infected (0.0 to 100.0).",
    )
    lesion_count: int | None = Field(
        default=None,
        description="Estimated count of distinct lesion spots.",
    )


class ExtractedImage(BaseModel):
    """One image extracted from an uploaded PDF (Phase 1 of multi-modal
    RAG). Carries enough provenance to be cited and re-located on disk:
    which page it came from, its original dimensions, and a storage
    path that survives restarts (delete_document's cleanups know how to
    remove it alongside the upload's own files)."""

    image_id: str = Field(
        ..., description="Stable identifier for this image, unique within the document."
    )
    document_id: str = Field(
        ..., description="Identifier of the document this image was extracted from."
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-based page this image appeared on — the page number the OCR/vision paths report.",
    )
    content_type: str = Field(
        "figure",
        description="What this image is: 'figure' (an embedded image), 'page' (a rasterized "
        "full-page render of a low-text page, produced for vision QA).",
    )
    storage_path: str = Field(
        ...,
        description="Filesystem path (relative to Settings.image_storage_dir_name) where the bytes are persisted.",
    )
    mime_type: str = Field("image/png", description="MIME type of the persisted bytes.")
    width: int = Field(..., ge=1, description="Image width in pixels.")
    height: int = Field(..., ge=1, description="Image height in pixels.")
    byte_size: int = Field(..., ge=0, description="Size of the persisted bytes.")


class ClipEmbedding(BaseModel):
    """A vector produced by the CLIP embedding microservice (clip_service/),
    along with its declared dimension. The embedding is L2-normalized by
    the service, so inner product == cosine similarity across text and
    image vectors — the property the image-index search relies on."""

    embedding: list[float]
    dimension: int


class ExtractedTable(BaseModel):
    """One table extracted from a PDF page, already reduced to structured
    text (markdown) so it can flow through the existing text chunking/
    embedding pipeline exactly like body text — the vector store never
    needs to know it was ever a table."""

    table_id: str = Field(
        ..., description="Stable identifier for this table, unique within the document."
    )
    document_id: str = Field(
        ..., description="Identifier of the document this table was extracted from."
    )
    page_number: int = Field(..., ge=1, description="1-based page this table appeared on.")
    markdown: str = Field(
        ..., description="The table as a markdown grid (header row, separator, then data rows)."
    )
    row_count: int = Field(
        ..., ge=0, description="Number of data rows in the table (header excluded)."
    )
    column_count: int = Field(..., ge=0, description="Number of columns in the table.")
