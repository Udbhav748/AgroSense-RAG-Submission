"""High-throughput bulk ingestion script for AgroSense-RAG.

Scans local document directories (Markdown .md, CSV .csv, PDF .pdf),
extracts text and tables, auto-detects collection names (e.g. crop subfolder),
generates embeddings in configurable batches (64/128), and indexes vectors
into FAISS (or Pgvector) with console progress reporting and resume checkpointing.

Usage:
    python backend/scripts/bulk_ingest.py data/plant_disease_docs
    python backend/scripts/bulk_ingest.py --dir data/plant_disease_docs --batch-size 128
    python backend/scripts/bulk_ingest.py data/plant_disease_docs --collection tomato --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.core.config import settings
from app.core.exceptions import AppError, VectorStoreNotFoundError
from app.core.logging import configure_logging
from app.models.document import DocumentChunk, EmbeddedChunk
from app.services.document_parser import (
    LayoutAwareDocumentParser,
    parse_csv_content,
    parse_layout_aware_markdown,
)
from app.services.embedding_service import get_embedding_model
from app.services.faiss_vector_store import (
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    FAISSVectorStore,
)
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".csv", ".pdf", ".txt"}
DEFAULT_CHECKPOINT_NAME = ".bulk_ingest_checkpoint.json"
_SPECIAL_TOKEN_BUDGET = 2


@dataclass
class IngestionStats:
    total_files_scanned: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_chunks_created: int = 0
    total_vectors_indexed: int = 0
    elapsed_seconds: float = 0.0
    collections_tagged: dict[str, int] = field(default_factory=dict)


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for change detection and checkpointing."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_documents(
    directory: Path,
    pattern: str = "*",
    recursive: bool = True,
) -> list[Path]:
    """Scan directory for supported document files (.md, .csv, .pdf, .txt)."""
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {directory}")

    candidates = directory.rglob(pattern) if recursive else directory.glob(pattern)
    matched_files: list[Path] = []

    for path in sorted(candidates):
        if not path.is_file():
            continue
        # Skip hidden files, system files, and files inside hidden/private directories
        if any(part.startswith(".") or part.startswith("_") for part in path.relative_to(directory).parts):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            matched_files.append(path)

    return matched_files


def detect_collection(
    file_path: Path,
    base_dir: Path,
    explicit_collection: str | None = None,
) -> str:
    """Auto-detect collection tag from relative folder structure or explicit override."""
    if explicit_collection and explicit_collection.strip():
        return explicit_collection.strip().lower()

    try:
        rel_path = file_path.resolve().relative_to(base_dir.resolve())
        # If inside a subdirectory (e.g. data/plant_disease_docs/tomato/guide.md -> tomato)
        parts = rel_path.parts
        if len(parts) > 1:
            return parts[0].lower().replace(" ", "_")
    except ValueError:
        pass

    # If at root of base_dir, use base_dir name or fallback to 'general'
    dir_name = base_dir.resolve().name.lower().replace(" ", "_")
    return dir_name if dir_name not in {"", ".", ".."} else "general"


def _fallback_split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pure-Python fallback splitter when langchain_text_splitters is unavailable."""
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])
    return chunks


def parse_markdown_file(
    file_path: Path,
    document_id: str,
    collection: str,
    tenant_id: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Parse Markdown file with layout awareness, preserving tabular row structures."""
    content = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []

    return parse_layout_aware_markdown(
        content=content,
        document_id=document_id,
        collection=collection,
        tenant_id=tenant_id,
        file_name=file_path.name,
        file_path=str(file_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def parse_csv_file(
    file_path: Path,
    document_id: str,
    collection: str,
    tenant_id: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Parse CSV into layout-aware atomic table row chunks and structured table slices."""
    content = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []

    return parse_csv_content(
        content=content,
        document_id=document_id,
        collection=collection,
        tenant_id=tenant_id,
        file_name=file_path.name,
        file_path=str(file_path),
        table_type="dosage_matrix",
        include_table_slices=True,
    )


def parse_pdf_file(
    file_path: Path,
    document_id: str,
    collection: str,
    tenant_id: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Parse PDF file into DocumentChunks using PyMuPDF (fitz)."""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF (fitz) is not installed; skipping PDF %s", file_path.name)
        return []

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise AppError(f"Failed to open PDF {file_path.name}: {exc}") from exc

    page_texts: list[str] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_texts.append(page.get_text("text"))

    full_text = "\n\n".join(page_texts).strip()
    if not full_text:
        return []

    c_size = chunk_size or settings.chunk_size
    c_overlap = chunk_overlap or settings.chunk_overlap

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
    )
    texts = splitter.split_text(full_text)

    return [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=i,
            text=t,
            metadata={
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(texts),
                "source": "pdf",
                "content_type": "application/pdf",
                "file_name": file_path.name,
                "file_path": str(file_path),
                "collection": collection,
                "page_count": len(doc),
                "tenant_id": tenant_id,
            },
        )
        for i, t in enumerate(texts)
    ]


def parse_document_file(
    file_path: Path,
    document_id: str,
    collection: str,
    tenant_id: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Route document to appropriate parser based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown_file(
            file_path,
            document_id=document_id,
            collection=collection,
            tenant_id=tenant_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif suffix == ".csv":
        return parse_csv_file(
            file_path,
            document_id=document_id,
            collection=collection,
            tenant_id=tenant_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif suffix == ".pdf":
        return parse_pdf_file(
            file_path,
            document_id=document_id,
            collection=collection,
            tenant_id=tenant_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        logger.warning("Unsupported file format: %s", file_path)
        return []


def split_oversized_chunks(
    chunks: list[DocumentChunk],
    model: Any,
) -> list[DocumentChunk]:
    """Ensure chunk tokens do not exceed the SentenceTransformer max_seq_length."""
    budget = getattr(model, "max_seq_length", 256) - _SPECIAL_TOKEN_BUDGET
    tokenizer = getattr(model, "tokenizer", None)

    if tokenizer is None:
        return chunks

    safe_chunks: list[DocumentChunk] = []
    for chunk in chunks:
        try:
            encoding = tokenizer(
                chunk.text, add_special_tokens=False, return_offsets_mapping=True
            )
            token_count = len(encoding["input_ids"])
            if token_count <= budget:
                safe_chunks.append(chunk)
                continue

            offsets = encoding["offset_mapping"]
            for start_idx in range(0, token_count, budget):
                token_slice = offsets[start_idx : start_idx + budget]
                char_start, char_end = token_slice[0][0], token_slice[-1][1]
                sub_text = chunk.text[char_start:char_end]
                safe_chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        text=sub_text,
                        metadata={**chunk.metadata, "split_subpart": True},
                    )
                )
        except Exception:
            safe_chunks.append(chunk)

    return safe_chunks


def generate_batch_embeddings(
    chunks: list[DocumentChunk],
    batch_size: int = 64,
    model: Any | None = None,
) -> list[EmbeddedChunk]:
    """Generate normalized embeddings for DocumentChunks in batches."""
    if not chunks:
        return []

    embedding_model = model or get_embedding_model()
    safe_chunks = split_oversized_chunks(chunks, embedding_model)

    texts = [chunk.text for chunk in safe_chunks]
    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    embedded: list[EmbeddedChunk] = []
    for chunk, vector in zip(safe_chunks, vectors, strict=False):
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        embedded.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                embedding=vec_list,
                metadata={**chunk.metadata, "text": chunk.text},
            )
        )

    return embedded


class CheckpointManager:
    """Manages ingestion resume checkpoints to avoid duplicate processing."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.data: dict[str, Any] = {"version": "1.0", "files": {}}
        self.load()

    def load(self) -> None:
        if self.checkpoint_path.exists():
            try:
                self.data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Corrupted checkpoint file, resetting: %s", exc)
                self.data = {"version": "1.0", "files": {}}

    def save(self) -> None:
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.checkpoint_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            temp_path.replace(self.checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)

    def is_file_processed(self, file_path: Path, base_dir: Path) -> bool:
        rel_key = str(file_path.resolve().relative_to(base_dir.resolve()))
        record = self.data.get("files", {}).get(rel_key)
        if not record:
            return False

        current_size = file_path.stat().st_size
        current_mtime = file_path.stat().st_mtime

        if record.get("size") == current_size and record.get("mtime") == current_mtime:
            return True

        # If mtime or size changed, check SHA256
        current_hash = compute_file_hash(file_path)
        return record.get("sha256") == current_hash

    def record_file(
        self,
        file_path: Path,
        base_dir: Path,
        collection: str,
        chunks_count: int,
    ) -> None:
        rel_key = str(file_path.resolve().relative_to(base_dir.resolve()))
        stat = file_path.stat()
        self.data.setdefault("files", {})[rel_key] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": compute_file_hash(file_path),
            "collection": collection,
            "chunks_count": chunks_count,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.save()

    def clear(self) -> None:
        self.data = {"version": "1.0", "files": {}}
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
            except OSError:
                pass


class BulkIngester:
    """High-throughput bulk ingestion engine for plant knowledge bases."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        batch_size: int = 64,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        checkpoint_path: Path | None = None,
        force: bool = False,
    ):
        self.batch_size = batch_size
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.force = force

        # Initialize vector store
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = FAISSVectorStore()
            try:
                self.vector_store.load()
            except VectorStoreNotFoundError:
                pass

        # Checkpoint path
        chk_path = (
            checkpoint_path
            or settings.data_dir(settings.vector_store_dir_name) / DEFAULT_CHECKPOINT_NAME
        )
        self.checkpoint = CheckpointManager(chk_path)
        if force:
            self.checkpoint.clear()

    def ingest_directory(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
        collection: str | None = None,
        tenant_id: int | None = None,
    ) -> IngestionStats:
        """Scan directory and bulk ingest all documents."""
        start_time = time.perf_counter()
        stats = IngestionStats()

        files = scan_documents(directory, pattern=pattern, recursive=recursive)
        stats.total_files_scanned = len(files)

        if not files:
            print(f"[BulkIngest] No supported documents found in {directory}")
            return stats

        print("\n=======================================================")
        print(f" Bulk Ingestion: {len(files)} document(s) in {directory}")
        print(f" Batch Size: {self.batch_size} | Chunk Size: {self.chunk_size}")
        print(f" Force Re-index: {self.force}")
        print("=======================================================\n")

        pending_chunks: list[DocumentChunk] = []
        files_in_current_batch: list[tuple[Path, str, int]] = []

        for idx, file_path in enumerate(files, start=1):
            file_col = detect_collection(file_path, directory, explicit_collection=collection)
            stats.collections_tagged[file_col] = stats.collections_tagged.get(file_col, 0) + 1

            if not self.force and self.checkpoint.is_file_processed(file_path, directory):
                stats.files_skipped += 1
                print(f"[{idx}/{len(files)}] SKIP (cached) -> {file_path.name} [{file_col}]")
                continue

            try:
                # Generate deterministic document ID from relative path
                rel_str = str(file_path.resolve().relative_to(directory.resolve()))
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"file://{rel_str}"))

                chunks = parse_document_file(
                    file_path,
                    document_id=doc_id,
                    collection=file_col,
                    tenant_id=tenant_id,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )

                if not chunks:
                    print(f"[{idx}/{len(files)}] EMPTY -> {file_path.name}")
                    continue

                stats.total_chunks_created += len(chunks)
                pending_chunks.extend(chunks)
                files_in_current_batch.append((file_path, file_col, len(chunks)))
                stats.files_processed += 1

                print(
                    f"[{idx}/{len(files)}] PARSED -> {file_path.name} ({len(chunks)} chunks) [{file_col}]"
                )

                # When pending chunk count reaches or exceeds batch_size, flush embeddings to vector store
                if len(pending_chunks) >= self.batch_size:
                    self._flush_batch(pending_chunks, files_in_current_batch, directory, stats)
                    pending_chunks = []
                    files_in_current_batch = []

            except Exception as exc:
                stats.files_failed += 1
                logger.exception("Error processing file %s: %s", file_path, exc)
                print(f"[{idx}/{len(files)}] ERROR -> {file_path.name}: {exc}")

        # Flush any remaining chunks
        if pending_chunks:
            self._flush_batch(pending_chunks, files_in_current_batch, directory, stats)

        stats.elapsed_seconds = round(time.perf_counter() - start_time, 2)
        self._print_summary(stats)
        return stats

    def _flush_batch(
        self,
        chunks: list[DocumentChunk],
        file_info_list: list[tuple[Path, str, int]],
        base_dir: Path,
        stats: IngestionStats,
    ) -> None:
        """Embed and index a batch of chunks into the vector store."""
        if not chunks:
            return

        batch_start = time.perf_counter()
        embedded_chunks = generate_batch_embeddings(chunks, batch_size=self.batch_size)

        if not embedded_chunks:
            return

        dim = len(embedded_chunks[0].embedding)
        try:
            self.vector_store.add_embeddings(embedded_chunks)
        except VectorStoreNotFoundError:
            self.vector_store.create_index(dimension=dim)
            self.vector_store.add_embeddings(embedded_chunks)

        self.vector_store.save()
        stats.total_vectors_indexed += len(embedded_chunks)

        # Update checkpoint for all files in this flushed batch
        for fpath, col, c_count in file_info_list:
            self.checkpoint.record_file(fpath, base_dir, col, c_count)

        duration = time.perf_counter() - batch_start
        rate = len(embedded_chunks) / duration if duration > 0 else 0
        print(
            f"   >> INDEXED BATCH: {len(embedded_chunks)} vectors in {duration:.2f}s ({rate:.1f} vec/s) | Total in index: {self.vector_store.total_vectors()}"
        )

    def _print_summary(self, stats: IngestionStats) -> None:
        """Print ingestion summary report."""
        print("\n=======================================================")
        print(f" Ingestion Completed in {stats.elapsed_seconds:.2f}s")
        print(f" Files Scanned:   {stats.total_files_scanned}")
        print(f" Files Indexed:   {stats.files_processed}")
        print(f" Files Skipped:   {stats.files_skipped} (from checkpoint)")
        print(f" Files Failed:    {stats.files_failed}")
        print(f" Chunks Created:  {stats.total_chunks_created}")
        print(f" Vectors Indexed: {stats.total_vectors_indexed}")
        print(f" Index Total:     {self.vector_store.total_vectors()} vectors")
        print(" Collections:")
        for col, count in stats.collections_tagged.items():
            print(f"   - {col}: {count} file(s)")
        print("=======================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=None,
        help="Directory of documents to ingest (defaults to data/plant_disease_docs).",
    )
    parser.add_argument(
        "--dir",
        dest="dir_flag",
        type=Path,
        default=None,
        help="Explicit directory path of documents to ingest.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Force a specific collection name instead of auto-detecting from subfolder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size for SentenceTransformer (default: 64).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Text chunk size in characters (default from settings: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Text chunk overlap in characters (default from settings: 200).",
    )
    parser.add_argument(
        "--store",
        choices=["faiss", "pgvector"],
        default="faiss",
        help="Vector store backend to target (default: faiss).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Custom FAISS index path.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Custom FAISS metadata JSON path.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="Custom checkpoint JSON file path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion, ignoring existing checkpoint.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for files to ingest (default: *).",
    )
    parser.add_argument(
        "--tenant-id",
        type=int,
        default=None,
        help="Optional tenant ID to tag all ingested chunks with.",
    )
    args = parser.parse_args()

    configure_logging()
    logging.getLogger().setLevel(logging.INFO)

    target_dir = args.dir_flag or args.directory
    if target_dir is None:
        # Default to data/plant_disease_docs at workspace root
        default_p = _BACKEND_ROOT.parent / "data" / "plant_disease_docs"
        if not default_p.exists():
            default_p = _BACKEND_ROOT / "data" / "plant_disease_docs"
        target_dir = default_p

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[BulkIngest Error] Target directory does not exist: {target_dir}")
        sys.exit(1)

    # Initialize requested vector store
    if args.store == "pgvector":
        from app.services.pgvector_store import PgvectorVectorStore

        store: VectorStore = PgvectorVectorStore()
    else:
        idx_p = args.index_path or DEFAULT_INDEX_PATH
        meta_p = args.metadata_path or DEFAULT_METADATA_PATH
        store = FAISSVectorStore(index_path=idx_p, metadata_path=meta_p)
        try:
            store.load()
        except VectorStoreNotFoundError:
            pass

    ingester = BulkIngester(
        vector_store=store,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        checkpoint_path=args.checkpoint_file,
        force=args.force,
    )

    ingester.ingest_directory(
        directory=target_dir,
        pattern=args.pattern,
        recursive=True,
        collection=args.collection,
        tenant_id=args.tenant_id,
    )


if __name__ == "__main__":
    main()
