"""Extracts tables from PDF pages using PyMuPDF's built-in table
detection (page.find_tables, present since pymupdf 1.24) and reduces
each table to structured markdown text (Phase 5 of multi-modal RAG) so
it flows through the existing text chunking/embedding pipeline exactly
like body text — the vector store never needs to know it was ever a
table.

pdfplumber was the plan's preferred library (pure-Python, no native
deps), but PyMuPDF's find_tables already ships with this project's
existing pymupdf dependency and adds no new package, keeping table
extraction offline-testable here. If table fidelity ever needs to
improve, swap the backend of extract_tables_from_pdf for
pdfplumber.extract_tables without touching the chunking pipeline.
"""

import logging
from pathlib import Path
from typing import Any

import fitz

from app.core.config import settings
from app.core.exceptions import CorruptedPDFError, DocumentNotFoundError
from app.models.document import ExtractedTable

logger = logging.getLogger(__name__)


def _table_to_markdown(table: fitz.table.Table) -> str:
    """Reduce a fitz Table to a markdown grid: header row, separator row,
    then data rows. Missing/empty cells become empty strings. Returns ""
    when there's no detectable header row."""
    rows = table.extract()
    if not rows:
        return ""
    column_count = max(len(row) for row in rows)
    if column_count == 0:
        return ""

    def _row_md(row: list[Any]) -> str:
        cells = [str(cell or "").strip() for cell in row]
        cells += [""] * (column_count - len(cells))
        return "| " + " | ".join(cells) + " |"

    lines = [
        _row_md(rows[0]),
        "| " + " | ".join(["---"] * column_count) + " |",
    ]
    lines.extend(_row_md(row) for row in rows[1:])
    return "\n".join(lines)


def extract_tables_from_pdf(document_id: str, file_path: Path) -> list[ExtractedTable]:
    """Extract the tables of a PDF, in page order, as structured markdown.

    Gated behind Settings.table_extraction_enabled by the caller.
    Per-page detection failures are logged and skipped (degrade, never
    fail); extraction is capped at Settings.table_max_count_per_document
    so a pathological PDF can't balloon ingestion.
    """
    if not file_path.is_file():
        raise DocumentNotFoundError(f"No such file: {file_path}")

    try:
        document = fitz.open(file_path)
    except fitz.FileDataError as exc:
        raise CorruptedPDFError(f"File is not a valid PDF: {file_path.name}") from exc

    tables: list[ExtractedTable] = []
    try:
        for page in document:
            if len(tables) >= settings.table_max_count_per_document:
                break
            try:
                finder = page.find_tables()
            except Exception as exc:
                logger.warning(
                    "table_extraction_page_failed",
                    extra={
                        "extra_fields": {
                            "document_id": document_id,
                            "page_number": page.number + 1,
                            "error": str(exc),
                        }
                    },
                )
                continue

            for table in finder.tables:
                if len(tables) >= settings.table_max_count_per_document:
                    break
                markdown = _table_to_markdown(table)
                if not markdown:
                    continue
                tables.append(
                    ExtractedTable(
                        table_id=f"{document_id}_tbl_{page.number + 1}_{len(tables) + 1}",
                        document_id=document_id,
                        page_number=page.number + 1,
                        markdown=markdown,
                        row_count=max(0, table.row_count - 1),  # header excluded
                        column_count=table.col_count,
                    )
                )
    finally:
        document.close()

    logger.info(
        "tables_extracted",
        extra={"extra_fields": {"document_id": document_id, "table_count": len(tables)}},
    )
    return tables
