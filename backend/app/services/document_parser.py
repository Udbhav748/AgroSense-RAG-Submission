"""Layout-Aware Ingestion & Tabular Parsing Service for AgroSense-RAG.

Specialized parser for agricultural documents, treatment dosage matrices,
and complex multi-column structured tables (CSV and Markdown tables).

Key capabilities:
1. Detects CSV files, Markdown tables, and multi-column tables in agricultural documents.
2. Formats tabular rows into atomic semantic units:
   `[TABLE ROW: Crop={crop} | Disease={disease} | Active Ingredient={chemical} | Rate={dosage} | Spray Interval={interval} | PHI={phi_days}]`
3. Extracts and attaches rich structured metadata (`table_type="dosage_matrix"`, `crop`,
   `disease`, `active_ingredient`, `phi_days`, `rate`, `spray_interval`, `is_table=True`).
4. Preserves layout and semantic row-column associations: tabular rows are never
   arbitrarily split or destroyed across character sliding-window chunk boundaries.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.document import DocumentChunk

logger = logging.getLogger(__name__)

# Column name aliases for standard agricultural dosage matrices and treatment tables
_CROP_ALIASES = {
    "crop",
    "crop_name",
    "plant",
    "plant_species",
    "host",
    "host_crop",
    "target_crop",
    "commodity",
}

_DISEASE_ALIASES = {
    "disease",
    "disease_name",
    "target_disease",
    "pathogen_disease",
    "condition",
    "disorder",
    "pest",
    "target_pest",
    "infection",
}

_ACTIVE_INGREDIENT_ALIASES = {
    "chemical_active_ingredient",
    "active_ingredient",
    "active_ingredients",
    "chemical",
    "active",
    "fungicide",
    "pesticide",
    "insecticide",
    "bactericide",
    "chemical_name",
    "agent",
    "specification_/_agent",
    "specification_agent",
    "treatment",
}

_RATE_ALIASES = {
    "dosage_per_liter",
    "dosage_rate_/_liter",
    "dosage_rate",
    "dosage",
    "rate",
    "rate_per_liter",
    "application_rate",
    "dose",
    "concentration",
    "rate_/_liter",
    "dosage_rate_per_liter",
}

_INTERVAL_ALIASES = {
    "spray_interval_days",
    "spray_interval",
    "application_interval",
    "interval_days",
    "interval",
    "retreatment_interval",
    "frequency",
    "spray_frequency",
}

_PHI_ALIASES = {
    "pre_harvest_interval_days",
    "pre_harvest_interval",
    "phi_(days)",
    "phi_days",
    "phi",
    "harvest_interval",
    "withholding_period",
    "preharvest_interval",
}

_SAFETY_ALIASES = {
    "safety_precautions",
    "precautions",
    "safety_notes",
    "ppe",
    "warning",
    "hazards",
    "safety",
}

_PATHOGEN_TYPE_ALIASES = {
    "pathogen_type",
    "type",
    "organism_type",
    "causal_agent",
    "pathogen_classification",
}

_ORGANIC_REMEDY_ALIASES = {
    "organic_remedy",
    "organic_treatment",
    "biological_control",
    "bio_fungicide",
    "organic",
}

_TRADE_NAME_ALIASES = {
    "trade_formulation",
    "trade_name",
    "product_name",
    "brand_name",
    "formulation",
}

_REI_ALIASES = {
    "rei_(hours)",
    "rei_hours",
    "rei",
    "re_entry_interval",
}

_MOA_ALIASES = {
    "mode_of_action",
    "moa",
    "frac_group",
    "frac_code",
    "frac",
    "irac_group",
}


@dataclass
class ParsedTableRow:
    """Atomic representation of a single parsed table row."""

    crop: str
    disease: str
    active_ingredient: str
    rate: str
    spray_interval: str
    phi_days: int | str
    table_type: str = "dosage_matrix"
    raw_row: dict[str, str] = field(default_factory=dict)
    extra_fields: dict[str, str] = field(default_factory=dict)
    formatted_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedTable:
    """Structured representation of an extracted table from Markdown or text."""

    headers: list[str]
    rows: list[dict[str, str]]
    table_type: str = "dosage_matrix"
    title: str = ""
    section_heading: str = ""
    raw_markdown: str = ""
    start_char: int = -1
    end_char: int = -1


def normalize_column_name(col: str) -> str:
    """Normalize column name for fuzzy alias lookup (lowercase, underscores)."""
    clean = re.sub(r"[^\w\s/]", "", col.strip().lower())
    clean = re.sub(r"[\s\-]+", "_", clean)
    return clean


def extract_phi_days(value: Any) -> int | str:
    """Extract numeric PHI in days if possible, preserving string if non-numeric."""
    if value is None:
        return 0
    val_str = str(value).strip()
    if not val_str:
        return 0

    # Match simple integer
    if val_str.isdigit():
        return int(val_str)

    # Match patterns like "7 days", "14 days", "7-10 days" -> return first number or string
    m = re.search(r"(\d+)\s*(?:days?|d)?", val_str, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    return val_str


def detect_context_from_text(text: str) -> dict[str, str]:
    """Detect crop and disease context from document headings, titles, or body text."""
    context: dict[str, str] = {}

    # Match Title/H1 like: "# Comprehensive Diagnostic & Management Guide: Tomato Early Blight"
    title_match = re.search(
        r"#\s+(?:Comprehensive\s+Diagnostic\s+&\s+Management\s+Guide:\s*)?([A-Za-z\s]+?)\s+([A-Za-z\s]+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if title_match:
        cand_crop = title_match.group(1).strip().lower()
        cand_disease = title_match.group(2).strip().lower()
        if cand_crop and cand_disease:
            context["crop"] = cand_crop
            context["disease"] = cand_disease

    # Match explicit "- **Crop**: Tomato (*Solanum lycopersicum*)"
    crop_match = re.search(r"-\s*\*\*Crop\*\*:\s*([A-Za-z\s]+?)(?:\s*\(|$|\n)", text, re.IGNORECASE)
    if crop_match:
        context["crop"] = crop_match.group(1).strip().lower()

    # Match explicit "- **Disease**: Early Blight"
    disease_match = re.search(r"-\s*\*\*Disease\*\*:\s*([A-Za-z0-9\s]+?)(?:\s*\(|$|\n)", text, re.IGNORECASE)
    if disease_match:
        context["disease"] = disease_match.group(1).strip().lower()

    return context


def format_table_row_unit(
    crop: str,
    disease: str,
    active_ingredient: str,
    rate: str,
    spray_interval: str,
    phi_days: Any,
    extra_fields: dict[str, str] | None = None,
    include_extra_in_unit: bool = True,
) -> str:
    """Format row into the required atomic semantic unit string.

    Example output:
    `[TABLE ROW: Crop=apple | Disease=apple scab | Active Ingredient=Captan 50% WP | Rate=2.5g/L | Spray Interval=7-10 | PHI=14]`
    """
    phi_str = str(phi_days) if phi_days is not None else "0"

    parts = [
        f"Crop={crop.strip() if crop else 'Unknown'}",
        f"Disease={disease.strip() if disease else 'Unknown'}",
        f"Active Ingredient={active_ingredient.strip() if active_ingredient else 'None'}",
        f"Rate={rate.strip() if rate else 'N/A'}",
        f"Spray Interval={spray_interval.strip() if spray_interval else 'N/A'}",
        f"PHI={phi_str.strip()}",
    ]

    if include_extra_in_unit and extra_fields:
        for k, v in extra_fields.items():
            if v and str(v).strip():
                clean_k = k.replace("_", " ").title()
                clean_v = str(v).strip().replace("\n", " ")
                parts.append(f"{clean_k}={clean_v}")

    return f"[TABLE ROW: {' | '.join(parts)}]"


def parse_tabular_row_dict(
    row: dict[str, Any],
    doc_context: dict[str, str] | None = None,
    collection: str | None = None,
    table_type: str = "dosage_matrix",
) -> ParsedTableRow:
    """Parse a single tabular row dictionary into a ParsedTableRow atomic unit."""
    context = doc_context or {}
    norm_row = {normalize_column_name(k): str(v or "").strip() for k, v in row.items() if k is not None}

    # Extract Crop
    crop = ""
    for alias in _CROP_ALIASES:
        if alias in norm_row and norm_row[alias]:
            crop = norm_row[alias]
            break
    if not crop:
        crop = context.get("crop", collection or "general")

    # Extract Disease
    disease = ""
    for alias in _DISEASE_ALIASES:
        if alias in norm_row and norm_row[alias]:
            disease = norm_row[alias]
            break
    if not disease:
        disease = context.get("disease", "")

    # Extract Active Ingredient / Chemical / Agent
    active_ingredient = ""
    for alias in _ACTIVE_INGREDIENT_ALIASES:
        if alias in norm_row and norm_row[alias]:
            active_ingredient = norm_row[alias]
            break

    # Extract Rate / Dosage
    rate = ""
    for alias in _RATE_ALIASES:
        if alias in norm_row and norm_row[alias]:
            rate = norm_row[alias]
            break

    # Extract Spray Interval
    spray_interval = ""
    for alias in _INTERVAL_ALIASES:
        if alias in norm_row and norm_row[alias]:
            spray_interval = norm_row[alias]
            break

    # Extract PHI (Pre-Harvest Interval)
    phi_val = ""
    for alias in _PHI_ALIASES:
        if alias in norm_row and norm_row[alias]:
            phi_val = norm_row[alias]
            break
    phi_days = extract_phi_days(phi_val) if phi_val else 0

    # Collect other standard / extra fields
    extra_fields: dict[str, str] = {}
    known_matched = (
        _CROP_ALIASES
        | _DISEASE_ALIASES
        | _ACTIVE_INGREDIENT_ALIASES
        | _RATE_ALIASES
        | _INTERVAL_ALIASES
        | _PHI_ALIASES
    )

    for k, v in row.items():
        if k is None:
            continue
        norm_k = normalize_column_name(k)
        if norm_k not in known_matched and str(v).strip():
            extra_fields[str(k).strip()] = str(v).strip()

    # If active_ingredient is still empty, look in extra fields for specification / agent
    if not active_ingredient:
        for k in list(extra_fields.keys()):
            norm_k = normalize_column_name(k)
            if any(term in norm_k for term in ("agent", "ingredient", "chemical", "control_measure", "specification")):
                active_ingredient = extra_fields[k]
                break

    # Construct formatted unit text
    formatted_text = format_table_row_unit(
        crop=crop,
        disease=disease,
        active_ingredient=active_ingredient,
        rate=rate,
        spray_interval=spray_interval,
        phi_days=phi_days,
        extra_fields=extra_fields,
    )

    # Build metadata
    metadata: dict[str, Any] = {
        "table_type": table_type,
        "is_table": True,
        "crop": crop.lower().strip(),
        "disease": disease.lower().strip(),
        "active_ingredient": active_ingredient.strip(),
        "rate": rate.strip(),
        "dosage": rate.strip(),
        "spray_interval": str(spray_interval).strip(),
        "phi_days": phi_days,
        "collection": crop.lower().replace(" ", "_").strip() if crop else (collection or "general"),
    }

    # Add extra agricultural metadata tags when available
    for alias_set, meta_key in [
        (_PATHOGEN_TYPE_ALIASES, "pathogen_type"),
        (_ORGANIC_REMEDY_ALIASES, "organic_remedy"),
        (_SAFETY_ALIASES, "safety_precautions"),
        (_TRADE_NAME_ALIASES, "trade_formulation"),
        (_REI_ALIASES, "rei_hours"),
        (_MOA_ALIASES, "mode_of_action"),
    ]:
        for alias in alias_set:
            if alias in norm_row and norm_row[alias]:
                metadata[meta_key] = norm_row[alias]
                break

    return ParsedTableRow(
        crop=crop,
        disease=disease,
        active_ingredient=active_ingredient,
        rate=rate,
        spray_interval=spray_interval,
        phi_days=phi_days,
        table_type=table_type,
        raw_row={str(k): str(v) for k, v in row.items() if k is not None},
        extra_fields=extra_fields,
        formatted_text=formatted_text,
        metadata=metadata,
    )


def extract_markdown_tables(text: str) -> list[ParsedTable]:
    """Detect and extract all Markdown tables with their header context and positions.

    Supports GFM markdown tables with header, delimiter row (`|---|---|`), and data rows.
    """
    tables: list[ParsedTable] = []
    lines = text.splitlines()
    i = 0

    current_section = ""

    while i < len(lines):
        line = lines[i]
        trimmed = line.strip()

        # Track section headings
        if trimmed.startswith("#"):
            current_section = trimmed.lstrip("#").strip()

        # Detect markdown table header and delimiter:
        # A markdown table row has at least 2 pipes: e.g. `| Col1 | Col2 |`
        if trimmed.startswith("|") and trimmed.endswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Delimiter row consists of |, -, :, and whitespace
            if re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", next_line):
                # Header row found!
                header_line = trimmed
                raw_headers = [c.strip() for c in header_line.split("|")[1:-1]]
                headers = [h for h in raw_headers if h]

                # Parse data rows
                table_lines = [line, lines[i + 1]]
                table_rows: list[dict[str, str]] = []
                row_idx = i + 2

                while row_idx < len(lines):
                    r_line = lines[row_idx].strip()
                    if r_line.startswith("|") and r_line.endswith("|"):
                        table_lines.append(lines[row_idx])
                        cells = [c.strip() for c in r_line.split("|")[1:-1]]
                        # Map to headers
                        row_dict: dict[str, str] = {}
                        for h_idx, h in enumerate(headers):
                            row_dict[h] = cells[h_idx] if h_idx < len(cells) else ""
                        table_rows.append(row_dict)
                        row_idx += 1
                    else:
                        break

                raw_table_md = "\n".join(table_lines)
                table_type = "dosage_matrix"
                # Determine table type from headers or section
                norm_h_str = " ".join(normalize_column_name(h) for h in headers)
                if any(k in norm_h_str for k in ("dosage", "rate", "phi", "chemical", "active_ingredient", "spray", "frac")):
                    table_type = "dosage_matrix"
                elif any(k in norm_h_str for k in ("control_measure", "remedy", "ipm", "organic")):
                    table_type = "ipm_matrix"
                else:
                    table_type = "structured_table"

                tables.append(
                    ParsedTable(
                        headers=headers,
                        rows=table_rows,
                        table_type=table_type,
                        title=current_section,
                        section_heading=current_section,
                        raw_markdown=raw_table_md,
                    )
                )
                i = row_idx
                continue
        i += 1

    return tables


def parse_csv_content(
    content: str,
    document_id: str,
    collection: str = "general",
    tenant_id: int | None = None,
    chunk_index_start: int = 0,
    file_name: str | None = None,
    file_path: str | None = None,
    table_type: str = "dosage_matrix",
    include_table_slices: bool = True,
    slice_size: int = 8,
) -> list[DocumentChunk]:
    """Parse CSV text into atomic semantic row chunks and structured table slices.

    Row-column semantic associations are preserved atomically per chunk.
    """
    clean_content = content.strip()
    if not clean_content:
        return []

    reader = list(csv.DictReader(io.StringIO(clean_content)))
    if not reader:
        return []

    headers = [h for h in (reader[0].keys() if reader else []) if h is not None]
    chunks: list[DocumentChunk] = []

    # 1. Atomic table row chunks: [TABLE ROW: Crop=... | Disease=... | ...]
    for i, row in enumerate(reader):
        parsed = parse_tabular_row_dict(
            row=row,
            collection=collection,
            table_type=table_type,
        )

        chunk_text = parsed.formatted_text
        # Prepend a readable title for dense semantic retrieval
        crop_title = parsed.crop.title() if parsed.crop else "General"
        disease_title = parsed.disease.title() if parsed.disease else "Treatment"
        full_text = f"Agricultural Treatment & Dosage Reference: {crop_title} - {disease_title}\n{chunk_text}"

        meta = {
            "document_id": document_id,
            "chunk_index": chunk_index_start + len(chunks),
            "source": "csv",
            "content_type": "text/csv-record",
            "file_name": file_name or f"{document_id}.csv",
            "file_path": file_path or f"{document_id}.csv",
            "tenant_id": tenant_id,
            "row_index": i,
            **parsed.metadata,
        }

        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=chunk_index_start + len(chunks),
                text=full_text,
                metadata=meta,
            )
        )

    # 2. Structured markdown table slices for holistic table querying
    if include_table_slices and headers:
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

        for start_idx in range(0, len(reader), slice_size):
            slice_rows = reader[start_idx : start_idx + slice_size]
            table_lines = [
                f"# Treatment & Dosage Matrix Table (Records {start_idx + 1}-{start_idx + len(slice_rows)})",
                header_line,
                separator_line,
            ]
            for r in slice_rows:
                table_lines.append(
                    "| " + " | ".join(str(r.get(h, "")).strip().replace("|", "/") for h in headers) + " |"
                )
            slice_text = "\n".join(table_lines)

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=chunk_index_start + len(chunks),
                    text=slice_text,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": chunk_index_start + len(chunks),
                        "source": "csv",
                        "content_type": "text/markdown-table",
                        "table_type": table_type,
                        "is_table": True,
                        "file_name": file_name or f"{document_id}.csv",
                        "file_path": file_path or f"{document_id}.csv",
                        "collection": collection,
                        "tenant_id": tenant_id,
                        "table_slice_start": start_idx,
                        "table_slice_count": len(slice_rows),
                    },
                )
            )

    # Sync total_chunks in metadata
    total_count = len(chunks)
    for c in chunks:
        c.metadata["total_chunks"] = total_count

    return chunks


def _fallback_split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pure-Python fallback character sliding window splitter for prose text."""
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - chunk_overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def parse_layout_aware_markdown(
    content: str,
    document_id: str,
    collection: str = "general",
    tenant_id: int | None = None,
    file_name: str | None = None,
    file_path: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Parse Markdown with layout awareness: separates prose from tables.

    Prose sections are chunked via paragraph/sliding window splitters, while
    Markdown tables are parsed into atomic row units preserving row-column
    semantic associations without being fragmented across chunk boundaries.
    """
    clean_text = content.strip()
    if not clean_text:
        return []

    c_size = chunk_size or settings.chunk_size
    c_overlap = chunk_overlap or settings.chunk_overlap

    doc_context = detect_context_from_text(clean_text)
    detected_crop = doc_context.get("crop", collection)

    tables = extract_markdown_tables(clean_text)
    chunks: list[DocumentChunk] = []

    # If no tables exist, use standard text splitting
    if not tables:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
            )
            texts = splitter.split_text(clean_text)
        except Exception:
            texts = _fallback_split_text(clean_text, c_size, c_overlap)

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
                    "source": "markdown",
                    "content_type": "text/markdown",
                    "file_name": file_name or f"{document_id}.md",
                    "file_path": file_path or f"{document_id}.md",
                    "collection": detected_crop.lower().replace(" ", "_"),
                    "tenant_id": tenant_id,
                    **doc_context,
                },
            )
            for i, t in enumerate(texts)
        ]

    # When tables exist: segment into prose blocks and table blocks
    # Replace each table with a placeholder token to preserve prose flow
    table_placeholder_map: dict[str, ParsedTable] = {}
    modified_text = clean_text

    for idx, tbl in enumerate(tables):
        token = f"__TABLE_BLOCK_PLACEHOLDER_{idx}__"
        table_placeholder_map[token] = tbl
        # Replace the table raw markdown with the placeholder token
        modified_text = modified_text.replace(tbl.raw_markdown, f"\n\n{token}\n\n", 1)

    # Split document by double newlines into segments
    segments = modified_text.split("\n\n")
    current_prose_buffer: list[str] = []

    def _flush_prose_buffer(buf: list[str]) -> list[DocumentChunk]:
        if not buf:
            return []
        p_text = "\n\n".join(buf).strip()
        if not p_text:
            return []
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
            )
            sub_texts = splitter.split_text(p_text)
        except Exception:
            sub_texts = _fallback_split_text(p_text, c_size, c_overlap)

        prose_chunks = []
        for st in sub_texts:
            if not st.strip():
                continue
            prose_chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=len(chunks) + len(prose_chunks),
                    text=st,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": len(chunks) + len(prose_chunks),
                        "source": "markdown",
                        "content_type": "text/markdown",
                        "file_name": file_name or f"{document_id}.md",
                        "file_path": file_path or f"{document_id}.md",
                        "collection": detected_crop.lower().replace(" ", "_"),
                        "tenant_id": tenant_id,
                        "is_table": False,
                        **doc_context,
                    },
                )
            )
        return prose_chunks

    for seg in segments:
        seg_trimmed = seg.strip()
        if not seg_trimmed:
            continue

        if seg_trimmed in table_placeholder_map:
            # Flush accumulated prose before the table
            if current_prose_buffer:
                chunks.extend(_flush_prose_buffer(current_prose_buffer))
                current_prose_buffer = []

            tbl = table_placeholder_map[seg_trimmed]
            # 1. Parse each table row as an atomic semantic chunk
            for r_idx, row in enumerate(tbl.rows):
                parsed = parse_tabular_row_dict(
                    row=row,
                    doc_context=doc_context,
                    collection=collection,
                    table_type=tbl.table_type,
                )
                title_prefix = f"[{tbl.section_heading or 'Agricultural Reference Table'}]\n"
                full_row_text = f"{title_prefix}{parsed.formatted_text}"

                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        chunk_index=len(chunks),
                        text=full_row_text,
                        metadata={
                            "document_id": document_id,
                            "chunk_index": len(chunks),
                            "source": "markdown_table",
                            "content_type": "text/table-row",
                            "file_name": file_name or f"{document_id}.md",
                            "file_path": file_path or f"{document_id}.md",
                            "table_title": tbl.title,
                            "section": tbl.section_heading,
                            "row_index": r_idx,
                            "tenant_id": tenant_id,
                            **parsed.metadata,
                        },
                    )
                )

            # 2. Also keep the intact Markdown table chunk for full-table lookup queries
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=f"# {tbl.section_heading or 'Table'}\n{tbl.raw_markdown}",
                    metadata={
                        "document_id": document_id,
                        "chunk_index": len(chunks),
                        "source": "markdown_table",
                        "content_type": "text/markdown-table",
                        "table_type": tbl.table_type,
                        "is_table": True,
                        "table_title": tbl.title,
                        "file_name": file_name or f"{document_id}.md",
                        "file_path": file_path or f"{document_id}.md",
                        "collection": detected_crop.lower().replace(" ", "_"),
                        "tenant_id": tenant_id,
                        **doc_context,
                    },
                )
            )
        else:
            current_prose_buffer.append(seg)

    # Flush remaining prose after all tables
    if current_prose_buffer:
        chunks.extend(_flush_prose_buffer(current_prose_buffer))

    # Re-index chunks to ensure strict consecutive indexing
    for idx, c in enumerate(chunks):
        c.chunk_index = idx
        c.metadata["chunk_index"] = idx
        c.metadata["total_chunks"] = len(chunks)

    return chunks


class LayoutAwareDocumentParser:
    """Unified layout-aware document parser engine."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def parse_csv(
        self,
        file_path_or_content: str | Path,
        document_id: str,
        collection: str = "general",
        tenant_id: int | None = None,
        chunk_index_start: int = 0,
        table_type: str = "dosage_matrix",
    ) -> list[DocumentChunk]:
        """Parse CSV file or raw CSV string."""
        if isinstance(file_path_or_content, Path):
            path = file_path_or_content
            text = path.read_text(encoding="utf-8", errors="replace")
            return parse_csv_content(
                content=text,
                document_id=document_id,
                collection=collection,
                tenant_id=tenant_id,
                chunk_index_start=chunk_index_start,
                file_name=path.name,
                file_path=str(path),
                table_type=table_type,
            )
        elif isinstance(file_path_or_content, str):
            # Check if it's an existing file path string
            path_cand = Path(file_path_or_content)
            if path_cand.is_file():
                text = path_cand.read_text(encoding="utf-8", errors="replace")
                return parse_csv_content(
                    content=text,
                    document_id=document_id,
                    collection=collection,
                    tenant_id=tenant_id,
                    chunk_index_start=chunk_index_start,
                    file_name=path_cand.name,
                    file_path=str(path_cand),
                    table_type=table_type,
                )
            else:
                return parse_csv_content(
                    content=file_path_or_content,
                    document_id=document_id,
                    collection=collection,
                    tenant_id=tenant_id,
                    chunk_index_start=chunk_index_start,
                    table_type=table_type,
                )
        return []

    def parse_markdown(
        self,
        file_path_or_content: str | Path,
        document_id: str,
        collection: str = "general",
        tenant_id: int | None = None,
    ) -> list[DocumentChunk]:
        """Parse Markdown file or raw string with layout awareness."""
        if isinstance(file_path_or_content, Path):
            path = file_path_or_content
            text = path.read_text(encoding="utf-8", errors="replace")
            return parse_layout_aware_markdown(
                content=text,
                document_id=document_id,
                collection=collection,
                tenant_id=tenant_id,
                file_name=path.name,
                file_path=str(path),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif isinstance(file_path_or_content, str):
            path_cand = Path(file_path_or_content)
            if path_cand.is_file():
                text = path_cand.read_text(encoding="utf-8", errors="replace")
                return parse_layout_aware_markdown(
                    content=text,
                    document_id=document_id,
                    collection=collection,
                    tenant_id=tenant_id,
                    file_name=path_cand.name,
                    file_path=str(path_cand),
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            else:
                return parse_layout_aware_markdown(
                    content=file_path_or_content,
                    document_id=document_id,
                    collection=collection,
                    tenant_id=tenant_id,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
        return []

    def parse_file(
        self,
        file_path: Path,
        document_id: str,
        collection: str = "general",
        tenant_id: int | None = None,
    ) -> list[DocumentChunk]:
        """Route to appropriate parser based on file extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return self.parse_csv(
                file_path_or_content=file_path,
                document_id=document_id,
                collection=collection,
                tenant_id=tenant_id,
            )
        elif suffix in {".md", ".markdown", ".txt"}:
            return self.parse_markdown(
                file_path_or_content=file_path,
                document_id=document_id,
                collection=collection,
                tenant_id=tenant_id,
            )
        else:
            logger.warning("Unsupported layout-aware file format: %s", file_path)
            return []
