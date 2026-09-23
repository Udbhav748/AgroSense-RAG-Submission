"""Syncs uploads/, vector_store/, and feedback/ to/from S3.

Exists for the Lambda deployment: Lambda's filesystem is read-only outside
/tmp, and /tmp itself is wiped whenever Lambda tears down an idle execution
environment — a stricter, more frequent version of the same "wipes on
spin-down" problem demo_seed_service.py already papers over for Render.
download_all() restores whatever was last synced before demo_seed_service
ever runs, so its own "already has vectors" check sees real data and stays
a no-op.

A no-op everywhere Settings.s3_sync_enabled is False — local dev,
docker-compose, and Render never touch S3. boto3 is imported lazily inside
_client() so importing this module has no AWS SDK side effect when disabled.

Every function here is best-effort: a failed sync must never fail a
request that already succeeded locally — only durability across the next
cold start is at risk.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> Any:
    # boto3 ships no inline type stubs (would need the separate
    # boto3-stubs/mypy-boto3-s3 package, not a project dependency), so
    # boto3.client("s3") resolves to Any regardless of annotation here.
    import boto3

    return boto3.client("s3")


def download_all() -> None:
    """Restore uploads/, vector_store/, and feedback/ from S3. Called once
    from main.py's lifespan startup, before the vector store is first
    loaded."""
    if not settings.s3_sync_enabled:
        return

    from app.services.faiss_vector_store import DEFAULT_INDEX_PATH
    from app.services.feedback_service import FEEDBACK_DIR
    from app.services.upload_service import UPLOAD_DIR

    for prefix, local_dir in (
        (settings.upload_dir_name, UPLOAD_DIR),
        (settings.vector_store_dir_name, DEFAULT_INDEX_PATH.parent),
        (settings.feedback_dir_name, FEEDBACK_DIR),
    ):
        try:
            _download_prefix(prefix, local_dir)
        except Exception:
            logger.warning(
                "s3_sync_download_failed",
                extra={"extra_fields": {"prefix": prefix}},
                exc_info=True,
            )


def _download_prefix(prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    client = _client()
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_sync_bucket_name, Prefix=f"{prefix}/"
    ):
        for obj in page.get("Contents", []):
            filename = obj["Key"].rsplit("/", 1)[-1]
            if not filename:
                continue
            client.download_file(
                settings.s3_sync_bucket_name, obj["Key"], str(local_dir / filename)
            )
            count += 1
    logger.info(
        "s3_sync_downloaded", extra={"extra_fields": {"prefix": prefix, "file_count": count}}
    )


def upload_file(local_path: Path, prefix: str) -> None:
    if not settings.s3_sync_enabled:
        return
    try:
        _client().upload_file(
            str(local_path), settings.s3_sync_bucket_name, f"{prefix}/{local_path.name}"
        )
    except Exception:
        logger.warning(
            "s3_sync_upload_failed",
            extra={"extra_fields": {"prefix": prefix, "file": local_path.name}},
            exc_info=True,
        )


def upload_dir(local_dir: Path, prefix: str) -> None:
    """Non-recursive — every synced directory here is flat."""
    if not settings.s3_sync_enabled:
        return
    for path in local_dir.iterdir():
        if path.is_file():
            upload_file(path, prefix)


def delete_file(filename: str, prefix: str) -> None:
    """So a deleted document's original file doesn't resurrect on the next
    cold start's download_all()."""
    if not settings.s3_sync_enabled:
        return
    try:
        _client().delete_object(Bucket=settings.s3_sync_bucket_name, Key=f"{prefix}/{filename}")
    except Exception:
        logger.warning(
            "s3_sync_delete_failed",
            extra={"extra_fields": {"prefix": prefix, "file": filename}},
            exc_info=True,
        )
