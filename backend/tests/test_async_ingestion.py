"""Unit tests for asynchronous document ingestion and task management (Recommendation 2 & 3)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.schemas import DocumentProcessingResponse
from app.services.task_service import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_EXTRACTING,
    TASK_STATUS_FAILED,
    TASK_STATUS_QUEUED,
    TaskService,
    execute_background_ingestion,
    get_task_service,
)


@pytest.fixture(autouse=True)
def reset_task_service():
    service = get_task_service()
    service.clear()
    yield
    service.clear()


class TestTaskServiceUnit:
    def test_create_and_get_task(self):
        service = TaskService()
        task = service.create_task(original_filename="sample.pdf", tenant_id=1)
        assert task.status == TASK_STATUS_QUEUED
        assert task.progress == 0.0
        assert task.original_filename == "sample.pdf"
        assert task.tenant_id == 1

        fetched = service.get_task(task.task_id, tenant_id=1)
        assert fetched is not None
        assert fetched.task_id == task.task_id

    def test_tenant_isolation_in_task_lookup(self):
        service = TaskService()
        task = service.create_task(original_filename="secret.pdf", tenant_id=1)

        # Cross-tenant request returns None (404 pattern)
        assert service.get_task(task.task_id, tenant_id=2) is None
        # Same tenant succeeds
        assert service.get_task(task.task_id, tenant_id=1) is not None

    def test_status_transitions(self):
        service = TaskService()
        task = service.create_task(original_filename="doc.pdf")

        service.update_status(task.task_id, TASK_STATUS_EXTRACTING, progress=25.0, current_step="Extracting text")
        t = service.get_task(task.task_id)
        assert t.status == TASK_STATUS_EXTRACTING
        assert t.progress == 25.0
        assert t.current_step == "Extracting text"

        mock_resp = DocumentProcessingResponse(
            document_id="doc-123",
            original_filename="doc.pdf",
            total_pages=5,
            total_chunks=10,
            total_embeddings=10,
            processing_time=1.23,
            status="processed",
        )
        service.complete_task(task.task_id, mock_resp)
        t = service.get_task(task.task_id)
        assert t.status == TASK_STATUS_COMPLETED
        assert t.progress == 100.0
        assert t.result == mock_resp

    def test_failure_transition(self):
        service = TaskService()
        task = service.create_task(original_filename="corrupted.pdf")
        service.fail_task(task.task_id, error="PDF is corrupted")

        t = service.get_task(task.task_id)
        assert t.status == TASK_STATUS_FAILED
        assert t.error == "PDF is corrupted"

    def test_lru_eviction(self):
        service = TaskService(max_tasks=2)
        t1 = service.create_task("doc1.pdf")
        t2 = service.create_task("doc2.pdf")
        t3 = service.create_task("doc3.pdf")

        assert service.get_task(t1.task_id) is None
        assert service.get_task(t2.task_id) is not None
        assert service.get_task(t3.task_id) is not None

    @pytest.mark.asyncio
    async def test_execute_background_ingestion_success(self):
        service = TaskService()
        task = service.create_task("report.pdf", document_id="doc-99")

        mock_doc_service = MagicMock()
        mock_resp = DocumentProcessingResponse(
            document_id="doc-99",
            original_filename="report.pdf",
            total_pages=2,
            total_chunks=4,
            total_embeddings=4,
            processing_time=0.45,
            status="processed",
        )
        mock_doc_service.process_saved_file = AsyncMock(return_value=mock_resp)

        await execute_background_ingestion(
            task_id=task.task_id,
            document_id="doc-99",
            original_filename="report.pdf",
            stored_filename="doc-99.pdf",
            file_size=1024,
            tenant_id=1,
            collection="Finance",
            service=mock_doc_service,
            task_service=service,
        )

        completed_task = service.get_task(task.task_id)
        assert completed_task is not None
        assert completed_task.status == TASK_STATUS_COMPLETED
        assert completed_task.progress == 100.0
        assert completed_task.result == mock_resp

    @pytest.mark.asyncio
    async def test_execute_background_ingestion_error(self):
        service = TaskService()
        task = service.create_task("bad.pdf", document_id="doc-err")

        mock_doc_service = MagicMock()
        mock_doc_service.process_saved_file = AsyncMock(side_effect=RuntimeError("OCR engine crashed"))

        await execute_background_ingestion(
            task_id=task.task_id,
            document_id="doc-err",
            original_filename="bad.pdf",
            stored_filename="doc-err.pdf",
            file_size=512,
            service=mock_doc_service,
            task_service=service,
        )

        failed_task = service.get_task(task.task_id)
        assert failed_task is not None
        assert failed_task.status == TASK_STATUS_FAILED
        assert "OCR engine crashed" in (failed_task.error or "")

