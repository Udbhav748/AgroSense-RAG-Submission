"""Background task manager for asynchronous document ingestion (Recommendation 2).

Tracks long-running document ingestion through granular pipeline stages:
- queued: File validated and queued for background worker
- extracting: Extracting text, metadata, images, and tables from PDF
- chunking: Partitioning extracted text into token-bounded chunks
- embedding: Generating dense vector embeddings for chunks
- indexing: Writing embeddings to vector store and persisting metadata
- completed: All stages finished; DocumentProcessingResponse available
- failed: An error occurred; error message and failure stage recorded

Enforces tenant isolation and bounded LRU storage for task tracking.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.models.schemas import DocumentProcessingResponse, TaskStatusResponse

logger = logging.getLogger(__name__)

# Task status constants
TASK_STATUS_QUEUED = "queued"
TASK_STATUS_EXTRACTING = "extracting"
TASK_STATUS_CHUNKING = "chunking"
TASK_STATUS_EMBEDDING = "embedding"
TASK_STATUS_INDEXING = "indexing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

VALID_TASK_STATUSES = {
    TASK_STATUS_QUEUED,
    TASK_STATUS_EXTRACTING,
    TASK_STATUS_CHUNKING,
    TASK_STATUS_EMBEDDING,
    TASK_STATUS_INDEXING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
}


@dataclass
class IngestionTask:
    """Represents a background document ingestion job."""

    task_id: str
    original_filename: str
    status: str = TASK_STATUS_QUEUED
    progress: float = 0.0
    current_step: str | None = "Queued for processing"
    document_id: str | None = None
    stored_filename: str | None = None
    file_size: int = 0
    tenant_id: int | None = None
    collection: str | None = None
    error: str | None = None
    result: DocumentProcessingResponse | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_response(self) -> TaskStatusResponse:
        """Serialize into API response schema."""
        return TaskStatusResponse(
            task_id=self.task_id,
            document_id=self.document_id,
            original_filename=self.original_filename,
            status=self.status,
            progress=round(self.progress, 1),
            current_step=self.current_step,
            error=self.error,
            result=self.result,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class TaskService:
    """Thread-safe, LRU-bounded registry for asynchronous ingestion tasks."""

    def __init__(self, max_tasks: int = 1000):
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        self._max_tasks = max_tasks
        self._tasks: dict[str, IngestionTask] = {}
        self._lock = threading.Lock()

    def _evict_if_needed(self) -> None:
        """Drop the oldest recorded task if at capacity. Called with lock held."""
        if len(self._tasks) >= self._max_tasks:
            oldest_id = min(self._tasks, key=lambda tid: self._tasks[tid].created_at)
            self._tasks.pop(oldest_id, None)

    def create_task(
        self,
        original_filename: str,
        document_id: str | None = None,
        stored_filename: str | None = None,
        file_size: int = 0,
        tenant_id: int | None = None,
        collection: str | None = None,
        task_id: str | None = None,
    ) -> IngestionTask:
        """Register a new ingestion task in 'queued' state."""
        tid = task_id or str(uuid.uuid4())
        task = IngestionTask(
            task_id=tid,
            original_filename=original_filename,
            document_id=document_id,
            stored_filename=stored_filename,
            file_size=file_size,
            tenant_id=tenant_id,
            collection=collection,
            status=TASK_STATUS_QUEUED,
            progress=0.0,
            current_step="Queued for background processing",
        )
        with self._lock:
            self._evict_if_needed()
            self._tasks[tid] = task

        logger.info(
            "audit_event",
            extra={
                "extra_fields": {
                    "event": "ingestion_task_created",
                    "task_id": tid,
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "filename": original_filename,
                }
            },
        )
        return task

    def get_task(self, task_id: str, tenant_id: int | None = None) -> IngestionTask | None:
        """Fetch a task by ID.

        Enforces tenant isolation: if tenant_id is provided, tasks belonging
        to another tenant return None (404 pattern).
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if tenant_id is not None and task.tenant_id is not None and task.tenant_id != tenant_id:
                return None
            return task

    def update_status(
        self,
        task_id: str,
        status: str,
        progress: float,
        current_step: str | None = None,
        document_id: str | None = None,
    ) -> None:
        """Update the status, progress percentage, and step description of a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = status
            task.progress = min(max(progress, 0.0), 100.0)
            if current_step:
                task.current_step = current_step
            if document_id:
                task.document_id = document_id
            task.updated_at = time.time()

    def complete_task(
        self,
        task_id: str,
        result: DocumentProcessingResponse,
    ) -> None:
        """Mark a task as completed with its DocumentProcessingResponse result."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = TASK_STATUS_COMPLETED
            task.progress = 100.0
            task.current_step = "Processing completed successfully"
            task.result = result
            task.document_id = result.document_id
            task.updated_at = time.time()

        logger.info(
            "audit_event",
            extra={
                "extra_fields": {
                    "event": "ingestion_task_completed",
                    "task_id": task_id,
                    "document_id": result.document_id,
                    "duration": result.processing_time,
                }
            },
        )

    def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> None:
        """Mark a task as failed with an error message."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = TASK_STATUS_FAILED
            task.error = error
            task.current_step = f"Failed: {error}"
            task.updated_at = time.time()

        logger.warning(
            "audit_event",
            extra={
                "extra_fields": {
                    "event": "ingestion_task_failed",
                    "task_id": task_id,
                    "error": error,
                }
            },
        )

    def list_tasks(
        self,
        tenant_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[IngestionTask]:
        """List tasks matching optional tenant_id and status filters, newest first."""
        with self._lock:
            tasks = list(self._tasks.values())
        if tenant_id is not None:
            tasks = [t for t in tasks if t.tenant_id == tenant_id]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def clear(self) -> None:
        """Reset the task registry (for test teardown)."""
        with self._lock:
            self._tasks.clear()


@functools.lru_cache(maxsize=1)
def get_task_service() -> TaskService:
    """Process-wide singleton TaskService instance."""
    return TaskService()


async def execute_background_ingestion(
    task_id: str,
    document_id: str,
    original_filename: str,
    stored_filename: str,
    file_size: int,
    tenant_id: int | None = None,
    collection: str | None = None,
    service: Any | None = None,
    task_service: TaskService | None = None,
) -> None:
    """Execute the end-to-end document processing pipeline asynchronously.

    Invoked by FastAPI BackgroundTasks. Reports stage transitions:
    - extracting (20%)
    - chunking (40%)
    - embedding (70%)
    - indexing (90%)
    - completed (100%)
    Or sets failed status on any exception.
    """
    ts = task_service or get_task_service()
    try:
        if service is None:
            from app.api.v1.routes.query import (
                get_image_vector_store,
                get_llm_client,
                get_vector_store,
            )
            from app.services.document_processing_service import DocumentProcessingService

            llm_client = get_llm_client() if settings.image_captioning_enabled else None
            image_store = get_image_vector_store() if settings.clip_embedding_enabled else None
            service = DocumentProcessingService(
                get_vector_store(),
                llm_client=llm_client,
                image_vector_store=image_store,
            )

        def on_progress(stage: str, progress: float, step_description: str | None = None) -> None:
            ts.update_status(
                task_id=task_id,
                status=stage,
                progress=progress,
                current_step=step_description,
                document_id=document_id,
            )

        result = await service.process_saved_file(
            document_id=document_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_size=file_size,
            tenant_id=tenant_id,
            collection=collection,
            progress_callback=on_progress,
        )
        ts.complete_task(task_id, result)
        from app.services.cache_service import cache_service

        cache_service.invalidate()
    except Exception as exc:
        logger.exception(
            "background_ingestion_failed",
            extra={
                "extra_fields": {
                    "task_id": task_id,
                    "document_id": document_id,
                    "error": str(exc),
                }
            },
        )
        ts.fail_task(task_id, str(exc))
