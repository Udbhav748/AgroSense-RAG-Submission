"""Shared utility functions (e.g., file validation, text cleaning)."""

# Intrinsic to the PDF file format, not a deployment-configurable value.
PDF_MAGIC_BYTES = b"%PDF-"


def is_pdf_mime_type(content_type: str | None, allowed_mime_types: list[str]) -> bool:
    return content_type in allowed_mime_types


def has_pdf_magic_bytes(contents: bytes) -> bool:
    return contents.startswith(PDF_MAGIC_BYTES)


def is_empty(contents: bytes) -> bool:
    return len(contents) == 0


def exceeds_size_limit(contents: bytes, max_size_bytes: int) -> bool:
    return len(contents) > max_size_bytes
