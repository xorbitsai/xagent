"""Unit tests for the document ingestion pipeline."""

from __future__ import annotations

from typing import Dict, List, Union

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DocumentValidationError,
    EmbeddingAdapterError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkForEmbedding,
    ChunkStrategy,
    DocumentProcessingStatus,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
    IngestionConfig,
    IngestionResult,
    ParseDocumentResponse,
    ParsedParagraph,
    ParseMethod,
)
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion
from xagent.core.tools.core.RAG_tools.progress.manager import ProgressManager


class _StubEmbeddingAdapter(BaseEmbedding):
    """Deterministic embedding adapter for tests."""

    def __init__(self, prefix: str = "vec") -> None:
        self.prefix = prefix

    def encode(  # type: ignore[override]
        self,
        text: Union[str, List[str]],
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


STATUS_EVENTS: List[Dict[str, str]] = []


def _capture_status_record(
    collection: str,
    doc_id: str,
    *,
    status: str,
    message: str,
    parse_hash: str,
) -> None:
    STATUS_EVENTS.append(
        {
            "collection": collection,
            "doc_id": doc_id,
            "status": status,
            "message": message,
            "parse_hash": parse_hash,
        }
    )


def _capture_record_ingestion_status(
    collection: str,
    doc_id: str | None,
    *,
    status: DocumentProcessingStatus,
    message: str,
    parse_hash: str | None,
    user_id: int | None = None,
) -> None:
    """Wrapper to capture _record_ingestion_status calls."""
    if doc_id:
        _capture_status_record(
            collection=collection,
            doc_id=doc_id,
            status=status.value,
            message=message,
            parse_hash=parse_hash or "",
        )


def _patch_embedding_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub embedding adapter resolution to avoid touching model hub."""

    stub_config = EmbeddingModelConfig(
        id="embedding-default",
        model_name="text-embedding-v3",
        model_provider="dashscope",
        dimension=2,
    )
    stub_adapter = _StubEmbeddingAdapter()

    monkeypatch.setattr(
        document_ingestion,
        "_resolve_embedding_adapter",
        lambda _cfg: (stub_config, stub_adapter),
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real sleep calls in retry loops."""

    monkeypatch.setattr(document_ingestion.time, "sleep", lambda _seconds: None)
    # Mock asyncio.sleep for async retry loops
    import asyncio

    async def _no_async_sleep(seconds: float) -> None:
        """Mock async sleep to avoid delays in tests."""
        pass

    monkeypatch.setattr(asyncio, "sleep", _no_async_sleep)


@pytest.fixture(autouse=True)
def _clear_status_events() -> None:
    """Ensure captured status events are reset between tests."""

    STATUS_EVENTS.clear()


def _patch_pipeline_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install happy-path doubles for all pipeline dependencies."""

    _patch_embedding_adapter(monkeypatch)

    # Mock collection to exist
    from unittest.mock import AsyncMock

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    mock_collection = CollectionInfo(
        name="demo",
        embedding_model_id="embedding-default",
        embedding_dimension=2,
    )
    # Mock the async method that validate_document_processing_sync calls
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.get_collection",
        AsyncMock(return_value=mock_collection),
    )

    # Mock initialize_collection_embedding to return the initialized collection
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.initialize_collection_embedding",
        AsyncMock(return_value=mock_collection),
    )

    monkeypatch.setattr(
        document_ingestion,
        "register_document",
        lambda **_: {"doc_id": "doc-1", "created": True, "user_id": 1},
    )

    parse_response = ParseDocumentResponse(
        doc_id="doc-1",
        parse_hash="hash-1",
        paragraphs=[ParsedParagraph(text="para", metadata={})],
        written=True,
    ).model_dump()

    monkeypatch.setattr(
        document_ingestion,
        "parse_document",
        lambda **_: parse_response,
    )

    monkeypatch.setattr(
        document_ingestion,
        "chunk_document",
        lambda **_: {"chunk_count": 2, "created": True},
    )

    chunks = [
        ChunkForEmbedding(
            doc_id="doc-1",
            chunk_id="chunk-1",
            parse_hash="hash-1",
            text="text-1",
            chunk_hash="chunk-hash-1",
            index=0,
        ),
        ChunkForEmbedding(
            doc_id="doc-1",
            chunk_id="chunk-2",
            parse_hash="hash-1",
            text="text-2",
            chunk_hash="chunk-hash-2",
            index=1,
        ),
    ]
    read_response = EmbeddingReadResponse(
        chunks=chunks,
        total_count=2,
        pending_count=2,
    ).model_dump()

    monkeypatch.setattr(
        document_ingestion,
        "read_chunks_for_embedding",
        lambda **_: read_response,
    )

    def _mock_write_vectors(**kwargs: object) -> dict:
        """Mock write_vectors_to_db that supports multiple calls for batch processing."""
        embeddings = kwargs.get("embeddings", [])
        return EmbeddingWriteResponse(
            upsert_count=len(embeddings),
            deleted_stale_count=0,
            index_status="created" if kwargs.get("create_index", False) else "skipped",
        ).model_dump()

    monkeypatch.setattr(
        document_ingestion,
        "write_vectors_to_db",
        _mock_write_vectors,
    )

    # Mock status recording to avoid database operations in tests
    monkeypatch.setattr(
        document_ingestion,
        "_record_ingestion_status",
        _capture_record_ingestion_status,
    )

    # Mock collection manager functions
    monkeypatch.setattr(
        document_ingestion,
        "validate_document_processing_sync",
        lambda **_: None,  # No-op for tests
    )


def test_process_document_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline returns success status with deterministic provider."""

    _patch_pipeline_dependencies(monkeypatch)

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=IngestionConfig(),
    )

    assert isinstance(result, IngestionResult)
    assert result.status == "success"
    assert result.embedding_count == 2
    assert result.vector_count == 2


def test_process_document_applies_spreadsheet_safeguards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spreadsheet ingestion should force safer token-based chunking and batching."""

    _patch_pipeline_dependencies(monkeypatch)

    captured_chunk_kwargs: Dict[str, object] = {}
    encode_batch_sizes: List[int] = []

    stub_config = EmbeddingModelConfig(
        id="embedding-default",
        model_name="text-embedding-v3",
        model_provider="dashscope",
        dimension=2,
    )

    class _RecordingEmbeddingAdapter(_StubEmbeddingAdapter):
        def encode(  # type: ignore[override]
            self,
            text: Union[str, List[str]],
            dimension: int | None = None,
            instruct: str | None = None,
        ) -> Union[List[float], List[List[float]]]:
            encode_batch_sizes.append(1 if isinstance(text, str) else len(text))
            return super().encode(text, dimension, instruct)

    monkeypatch.setattr(
        document_ingestion,
        "_resolve_embedding_adapter",
        lambda _cfg: (stub_config, _RecordingEmbeddingAdapter()),
    )

    def _capture_chunk_document(**kwargs: object) -> dict:
        captured_chunk_kwargs.update(kwargs)
        return {"chunk_count": 2, "created": True}

    monkeypatch.setattr(document_ingestion, "chunk_document", _capture_chunk_document)

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/table.xlsx",
        config=IngestionConfig(
            parse_method=ParseMethod.DEFAULT,
            chunk_strategy=ChunkStrategy.FIXED_SIZE,
            chunk_size=1600,
            chunk_overlap=200,
            use_token_count=False,
            embedding_batch_size=10,
        ),
    )

    assert result.status == "success"
    assert captured_chunk_kwargs["chunk_strategy"] == ChunkStrategy.RECURSIVE
    assert captured_chunk_kwargs["chunk_size"] == 512
    assert captured_chunk_kwargs["chunk_overlap"] == 64
    assert captured_chunk_kwargs["use_token_count"] is True
    compute_step = next(
        step for step in result.completed_steps if step.name == "compute_embeddings"
    )
    assert compute_step.metadata.get("use_async") is True
    assert encode_batch_sizes == [1, 1]


def test_process_document_preserves_stricter_spreadsheet_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spreadsheet safeguards should not enlarge already strict caller limits."""

    _patch_pipeline_dependencies(monkeypatch)

    captured_chunk_kwargs: Dict[str, object] = {}
    captured_parse_method: Dict[str, object] = {}

    def _capture_parse_document(**kwargs: object) -> dict:
        captured_parse_method["parse_method"] = kwargs["parse_method"]
        return ParseDocumentResponse(
            doc_id="doc-1",
            parse_hash="hash-1",
            paragraphs=[ParsedParagraph(text="para", metadata={})],
            written=True,
        ).model_dump()

    def _capture_chunk_document(**kwargs: object) -> dict:
        captured_chunk_kwargs.update(kwargs)
        return {"chunk_count": 2, "created": True}

    monkeypatch.setattr(document_ingestion, "parse_document", _capture_parse_document)
    monkeypatch.setattr(document_ingestion, "chunk_document", _capture_chunk_document)

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/small.csv",
        config=IngestionConfig(
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=128,
            chunk_overlap=16,
            use_token_count=False,
            embedding_batch_size=1,
        ),
    )

    assert result.status == "success"
    assert captured_parse_method["parse_method"] == ParseMethod.DEFAULT
    assert captured_chunk_kwargs["chunk_size"] == 128
    assert captured_chunk_kwargs["chunk_overlap"] == 16
    assert captured_chunk_kwargs["use_token_count"] is True


def test_process_document_uses_async_writes_for_spreadsheets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spreadsheet safeguards should keep provider calls single-row while allowing batched DB writes."""

    _patch_pipeline_dependencies(monkeypatch)

    write_calls: List[int] = []

    def _record_write_vectors_to_db(**kwargs: object) -> EmbeddingWriteResponse:
        embeddings = kwargs.get("embeddings", [])
        write_calls.append(len(embeddings))
        return EmbeddingWriteResponse(
            upsert_count=len(embeddings),
            deleted_stale_count=0,
            index_status="created",
        )

    monkeypatch.setattr(
        document_ingestion,
        "write_vectors_to_db",
        _record_write_vectors_to_db,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/table.xlsx",
        config=IngestionConfig(
            embedding_use_async=False,
            embedding_batch_size=8,
            embedding_concurrent=3,
        ),
    )

    assert result.status == "success"
    compute_step = next(
        step for step in result.completed_steps if step.name == "compute_embeddings"
    )
    assert compute_step.metadata.get("use_async") is True
    assert compute_step.metadata.get("concurrent") == 3
    assert write_calls == [2]


def test_process_document_reports_embedding_progress_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress manager should expose completed/remaining counts for embedding and vector writes."""

    _patch_pipeline_dependencies(monkeypatch)
    manager = ProgressManager()
    manager._active_tasks.clear()

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/table.xlsx",
        config=IngestionConfig(
            embedding_use_async=False,
            embedding_batch_size=8,
            embedding_concurrent=2,
        ),
        progress_manager=manager,
        user_id=1,
    )

    assert result.status == "success"
    task_id = "ingest_demo__tmp_table_xlsx"
    task = manager.get_task_progress(task_id)
    assert task is not None
    assert task.metadata.get("steps") is not None

    compute_step = task.metadata["steps"]["compute_embeddings"]
    assert compute_step["current_count"] == 2
    assert compute_step["total_count"] == 2
    assert compute_step["message"] == "Completed"

    write_step = task.metadata["steps"]["write_vectors_to_db"]
    assert write_step["current_count"] == 2
    assert write_step["total_count"] == 2
    assert write_step["message"] == "Completed"


def test_process_document_rejects_oversized_spreadsheet_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spreadsheet rows larger than the token budget should fail before embedding."""

    _patch_pipeline_dependencies(monkeypatch)

    monkeypatch.setattr(
        document_ingestion,
        "read_chunks_for_embedding",
        lambda **_: EmbeddingReadResponse(
            chunks=[
                ChunkForEmbedding(
                    doc_id="doc-1",
                    chunk_id="chunk-1",
                    index=0,
                    parse_hash="hash-1",
                    text="x " * 2000,
                    chunk_hash="chunk-hash-1",
                    metadata={"file_type": "xlsx", "row_type": "data"},
                )
            ],
            pending_count=1,
            total_count=1,
        ),
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/table.xlsx",
        config=IngestionConfig(),
    )

    assert result.status == "partial"
    assert result.failed_step == "read_chunks_for_embedding"
    assert "token budget" in result.message.lower()
    assert {step.name for step in result.completed_steps} == {
        "initialize_collection",
        "resolve_embedding_adapter",
        "register_document",
        "parse_document",
        "chunk_document",
    }
    assert STATUS_EVENTS == [
        {
            "collection": "demo",
            "doc_id": "doc-1",
            "status": DocumentProcessingStatus.RUNNING.value,
            "message": "Document ingestion started.",
            "parse_hash": "",
        },
        {
            "collection": "demo",
            "doc_id": "doc-1",
            "status": DocumentProcessingStatus.FAILED.value,
            "message": result.message,
            "parse_hash": "hash-1",
        },
    ]


def test_process_document_passes_file_id_to_register_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline should forward file_id to document registration."""
    _patch_pipeline_dependencies(monkeypatch)
    captured_kwargs: Dict[str, object] = {}

    def _capture_register_document(**kwargs: object) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {"doc_id": "doc-1", "created": True}

    monkeypatch.setattr(
        document_ingestion, "register_document", _capture_register_document
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=IngestionConfig(),
        file_id="file-123",
    )

    assert result.status == "success"
    assert captured_kwargs["file_id"] == "file-123"


def test_process_document_skips_embedding_when_no_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline exits early when there are no pending chunks to embed."""

    _patch_pipeline_dependencies(monkeypatch)

    empty_read = EmbeddingReadResponse(
        chunks=[],
        total_count=0,
        pending_count=0,
    ).model_dump()

    monkeypatch.setattr(
        document_ingestion,
        "read_chunks_for_embedding",
        lambda **_: empty_read,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=IngestionConfig(),
    )

    assert result.status == "success"
    assert result.embedding_count == 0
    assert result.vector_count == 0
    assert result.completed_steps[-1].name == "read_chunks_for_embedding"
    assert STATUS_EVENTS[-1]["status"] == DocumentProcessingStatus.SUCCESS.value
    assert "no pending embeddings" in STATUS_EVENTS[-1]["message"].lower()


def test_process_document_partial_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline reports partial status when write step fails."""

    _patch_pipeline_dependencies(monkeypatch)

    def _failing_write_vectors_to_db(**_: object) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        document_ingestion,
        "write_vectors_to_db",
        _failing_write_vectors_to_db,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=IngestionConfig(),
    )

    assert result.status == "partial"
    # Writing is now integrated into the compute_embeddings step loop
    assert result.failed_step in ("write_vectors_to_db", "compute_embeddings")
    assert result.embedding_count == 2
    assert result.vector_count == 0
    assert result.message.startswith("Failed to write embedding batch")
    assert STATUS_EVENTS[-1]["status"] == DocumentProcessingStatus.FAILED.value


def test_process_document_failed_before_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization errors should surface with failed_step='initialize'."""

    # Mock validate_document_processing_sync to pass validation
    from unittest.mock import AsyncMock

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    mock_collection = CollectionInfo(name="demo")
    # Mock the async method that validate_document_processing_sync calls
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.get_collection",
        AsyncMock(return_value=mock_collection),
    )

    monkeypatch.setattr(
        document_ingestion,
        "_resolve_embedding_adapter",
        _raise_adapter_error,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=IngestionConfig(parse_method="deepdoc"),
    )

    assert result.status == "partial"
    # Failure can occur in initialize_collection (if resolve_embedding_adapter fails there)
    # or in resolve_embedding_adapter step
    assert result.failed_step in ("initialize_collection", "resolve_embedding_adapter")
    assert "adapter missing" in result.message
    assert STATUS_EVENTS == []


def test_process_document_with_async_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline works correctly with async concurrent embedding mode."""

    _patch_pipeline_dependencies(monkeypatch)

    # Test with async mode enabled
    config = IngestionConfig(
        parse_method="deepdoc",
        embedding_model_id="embedding-default",  # Ensure embedding_model_id is set
        embedding_use_async=True,
        embedding_concurrent=5,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=config,
    )

    assert isinstance(result, IngestionResult)
    assert result.status == "success"
    assert result.embedding_count == 2
    assert result.vector_count == 2
    assert {step.name for step in result.completed_steps} == {
        "initialize_collection",
        "resolve_embedding_adapter",
        "register_document",
        "parse_document",
        "chunk_document",
        "read_chunks_for_embedding",
        "compute_embeddings",
        "write_vectors_to_db",
    }

    # Verify async mode metadata is recorded
    compute_step = next(
        step for step in result.completed_steps if step.name == "compute_embeddings"
    )
    assert compute_step.metadata.get("use_async") is True
    assert compute_step.metadata.get("concurrent") == 5


def test_process_document_with_batch_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline works correctly with batch embedding mode (default)."""

    _patch_pipeline_dependencies(monkeypatch)

    # Test with batch mode (default, embedding_use_async=False)
    config = IngestionConfig(
        embedding_use_async=False,
        embedding_batch_size=2,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=config,
    )

    assert isinstance(result, IngestionResult)
    assert result.status == "success"
    assert result.embedding_count == 2
    assert result.vector_count == 2

    # Verify batch mode metadata is recorded
    compute_step = next(
        step for step in result.completed_steps if step.name == "compute_embeddings"
    )
    assert compute_step.metadata.get("use_async") is False
    assert compute_step.metadata.get("batch_size") == 2


def test_process_document_streams_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embeddings should be written batch-by-batch when batch size is small."""

    _patch_pipeline_dependencies(monkeypatch)

    calls: List[dict] = []

    def _record_write_vectors_to_db(**kwargs: object) -> EmbeddingWriteResponse:
        calls.append(
            {
                "create_index": kwargs.get("create_index", False),
                "size": len(kwargs.get("embeddings", [])),
            }
        )
        return EmbeddingWriteResponse(
            upsert_count=len(kwargs.get("embeddings", [])),
            deleted_stale_count=0,
            index_status="created",
        )

    monkeypatch.setattr(
        document_ingestion,
        "write_vectors_to_db",
        _record_write_vectors_to_db,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path="/tmp/doc.pdf",
        config=IngestionConfig(embedding_batch_size=1),
    )

    assert result.embedding_count == 2
    assert result.vector_count == 2
    assert len(calls) == 2
    assert calls[0]["create_index"] is False
    assert calls[1]["create_index"] is True
    assert sum(call["size"] for call in calls) == 2
    assert STATUS_EVENTS[-1]["status"] == DocumentProcessingStatus.SUCCESS.value


def _raise_adapter_error(*_: object) -> None:
    raise EmbeddingAdapterError("adapter missing")


def test_process_document_rejects_path_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that process_document rejects path traversal attempts in source_path."""

    _patch_embedding_adapter(monkeypatch)

    # Mock collection manager functions to prevent LanceDB connection issues
    from unittest.mock import AsyncMock

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    mock_collection = CollectionInfo(
        name="demo",
        embedding_model_id="embedding-default",
        embedding_dimension=2,
    )

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.get_collection",
        AsyncMock(return_value=mock_collection),
    )

    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.initialize_collection_embedding",
        AsyncMock(return_value=mock_collection),
    )

    monkeypatch.setattr(
        document_ingestion,
        "validate_document_processing_sync",
        lambda **_: None,  # No-op for tests
    )

    # Test various path traversal attack patterns
    malicious_paths = [
        "../../../etc/passwd",  # Unix path traversal
        "..\\..\\..\\windows\\system32\\config\\sam",  # Windows path traversal
        "/etc/passwd",  # Absolute path (if should be restricted)
        "..%2F..%2F..%2Fetc%2Fpasswd",  # URL-encoded path traversal
        "....//....//....//etc/passwd",  # Double-dot encoding
        "..%252F..%252F..%252Fetc%252Fpasswd",  # Double URL encoding
    ]

    for malicious_path in malicious_paths:
        result = document_ingestion.process_document(
            collection="demo",
            source_path=malicious_path,
            config=IngestionConfig(),
        )

        # Should fail early with validation error or file not found
        assert result.status in ("error", "partial")
        # The failed_step should indicate where it failed (likely register_document or initialize_collection)
        assert result.failed_step in (
            "register_document",
            "initialize_collection",
            "resolve_embedding_adapter",
        )
        # Should contain error message about invalid path, file not found, or unsupported file type
        # Note: Some paths may pass existence check but fail file type validation
        message_lower = result.message.lower()
        assert (
            "path" in message_lower
            or "not found" in message_lower
            or "invalid" in message_lower
            or "does not exist" in message_lower
            or "file type" in message_lower
            or "unsupported" in message_lower
        )


def test_process_document_rejects_path_traversal_with_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that process_document should reject path traversal even if the target file exists.

    This test mocks register_document to simulate path traversal detection.
    In a properly secured system, paths containing '..' or absolute paths outside
    allowed directories should be rejected even if the file exists.
    """

    _patch_embedding_adapter(monkeypatch)

    # Mock collection to exist
    from unittest.mock import AsyncMock

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    mock_collection = CollectionInfo(
        name="demo",
        embedding_model_id="embedding-default",
        embedding_dimension=2,
    )
    # Mock the async method that validate_document_processing_sync calls
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.get_collection",
        AsyncMock(return_value=mock_collection),
    )

    # Mock initialize_collection_embedding to return the initialized collection
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.initialize_collection_embedding",
        AsyncMock(return_value=mock_collection),
    )

    monkeypatch.setattr(
        document_ingestion,
        "validate_document_processing_sync",
        lambda **_: None,  # No-op for tests
    )

    # Create a temporary file outside the allowed directory
    outside_file = tmp_path.parent / "sensitive_file.txt"
    outside_file.write_text("sensitive content")

    # Try to access it using path traversal
    malicious_path = f"../{outside_file.name}"

    # Mock register_document to simulate path traversal detection
    # This represents the expected behavior: reject paths with traversal patterns
    def _mock_register_with_traversal(**kwargs: object) -> dict:
        source_path = kwargs.get("source_path", "")
        # Check for path traversal patterns
        if ".." in source_path or (
            source_path.startswith("/") and source_path != str(outside_file)
        ):
            raise DocumentValidationError(
                f"Path traversal detected or path outside allowed directory: {source_path}"
            )
        return {"doc_id": "doc-1", "created": True}

    monkeypatch.setattr(
        document_ingestion,
        "register_document",
        _mock_register_with_traversal,
    )

    result = document_ingestion.process_document(
        collection="demo",
        source_path=malicious_path,
        config=IngestionConfig(parse_method="default"),
    )

    # Should fail with validation error
    assert result.status == "partial"
    assert result.failed_step == "register_document"
    assert (
        "traversal" in result.message.lower()
        or "invalid" in result.message.lower()
        or "outside" in result.message.lower()
    )


def test_process_document_rejects_absolute_paths_outside_allowed_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that process_document rejects absolute paths to sensitive system files.

    Absolute paths to system files should be rejected even if they exist,
    as they represent a security risk.
    """

    _patch_embedding_adapter(monkeypatch)

    # Mock collection to exist
    from unittest.mock import AsyncMock

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    mock_collection = CollectionInfo(name="demo")
    # Mock the async method that validate_document_processing_sync calls
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.collection_manager.get_collection",
        AsyncMock(return_value=mock_collection),
    )

    # Test absolute paths that should be restricted
    # These files typically don't exist in test environments, so they'll fail with "not found"
    # But if they did exist, they should still be rejected
    absolute_paths = [
        "/etc/passwd",
        "/etc/shadow",
        "/root/.ssh/id_rsa",
        "C:\\Windows\\System32\\config\\SAM",  # Windows absolute path
    ]

    for abs_path in absolute_paths:
        result = document_ingestion.process_document(
            collection="demo",
            source_path=abs_path,
            config=IngestionConfig(),
        )

        # Should fail with validation error (file not found, path validation, or file type)
        assert result.status in ("error", "partial")
        assert result.failed_step in (
            "register_document",
            "initialize_collection",
            "resolve_embedding_adapter",
        )
        # Should contain error message about path, file not found, permission, or unsupported file type
        # Note: Absolute paths to system files may exist but fail file type validation
        # In CI environments with sandboxing, permission denied errors are also valid rejections
        message_lower = result.message.lower()
        assert (
            "path" in message_lower
            or "not found" in message_lower
            or "does not exist" in message_lower
            or "file type" in message_lower
            or "unsupported" in message_lower
            or "permission" in message_lower
            or "denied" in message_lower
        ), f"Unexpected error message for {abs_path}: {result.message}"
