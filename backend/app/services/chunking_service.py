"""Splits an ExtractedDocument into DocumentChunks using LangChain's
RecursiveCharacterTextSplitter.

Independent of embeddings and vector storage: this service only turns text
into sized, overlapping chunks with traceable metadata.

The langchain_text_splitters package's __init__ eagerly imports its
sentence_transformers submodule (which pulls in torch/transformers — a
30-45s cold import). Importing the splitter inside the functions below
keeps that heavy work off the startup path entirely: chunking only ever
happens during an upload, never when uvicorn first binds the port.
"""

import logging
import time
import uuid
from typing import Any

from app.core.config import settings
from app.models.document import DocumentChunk, ExtractedDocument

logger = logging.getLogger(__name__)


def chunk_document(
    document: ExtractedDocument, tenant_id: int | None = None
) -> list[DocumentChunk]:
    """Split document.extracted_text into DocumentChunks.

    Chunk size and overlap come from Settings (chunk_size, chunk_overlap).

    tenant_id (None when the DB/multi-tenancy is disabled) rides along in
    each chunk's metadata so the vector store can later filter retrieval
    by it — see retrieval_service.retrieve() and
    FAISSVectorStore.search()/search_bm25(). Without this, every chunk is
    just as searchable to every tenant regardless of who uploaded it.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    start = time.perf_counter()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    texts = splitter.split_text(document.extracted_text)
    total_chunks = len(texts)

    chunks = [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document.document_id,
            chunk_index=index,
            text=text,
            metadata={
                "document_id": document.document_id,
                "chunk_index": index,
                "total_chunks": total_chunks,
                "source": "pdf",
                "tenant_id": tenant_id,
            },
        )
        for index, text in enumerate(texts)
    ]

    processing_duration = time.perf_counter() - start

    logger.info(
        "document_chunked",
        extra={
            "extra_fields": {
                "document_id": document.document_id,
                "total_chunks": total_chunks,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "processing_duration": round(processing_duration, 4),
            }
        },
    )

    return chunks


def chunk_text(
    *,
    text: str,
    document_id: str,
    tenant_id: int | None,
    chunk_index_start: int = 0,
    source: str,
    content_type: str,
    page_number: int | None = None,
    **extra_metadata: Any,
) -> list[DocumentChunk]:
    """Split arbitrary text into DocumentChunks — the shared helper for
    multi-modal ingestion (image captions, table text), which reuses the
    exact same splitter and metadata shape as chunk_document.

    The metadata carries {document_id, chunk_index, total_chunks, source,
    content_type, tenant_id} plus whatever extra_metadata the caller
    passes (image_id, table_id, ...). source/content_type identify the
    modality — from the vector store's point of view an image-caption
    chunk or a table chunk is just another chunk, distinguishable only by
    these fields. chunk_index_start lets the caller lay these chunks after
    the body-text chunks so chunk_index ordering (used by
    get_chunks_by_document / summarization) stays stable.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    start = time.perf_counter()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    texts = splitter.split_text(text)
    total_chunks = len(texts)

    chunks = [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=chunk_index_start + index,
            text=piece,
            metadata={
                "document_id": document_id,
                "chunk_index": chunk_index_start + index,
                "total_chunks": total_chunks,
                "source": source,
                "content_type": content_type,
                "tenant_id": tenant_id,
                **({"page_number": page_number} if page_number is not None else {}),
                **extra_metadata,
            },
        )
        for index, piece in enumerate(texts)
    ]

    logger.info(
        "multimodal_text_chunked",
        extra={
            "extra_fields": {
                "document_id": document_id,
                "source": source,
                "content_type": content_type,
                "total_chunks": total_chunks,
                "processing_duration": round(time.perf_counter() - start, 4),
            }
        },
    )

    return chunks
