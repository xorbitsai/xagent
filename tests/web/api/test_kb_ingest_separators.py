"""Tests for /api/kb/ingest and /api/kb/ingest-web separators parameter parsing and passthrough."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from xagent.config import WEB_CRAWL_TLS_IMPERSONATE
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionOperationResult,
    IngestionConfig,
    IngestionResult,
    WebIngestionResult,
)
from xagent.web.api.kb import kb_router
from xagent.web.models.background_job import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobType,
)
from xagent.web.models.database import get_db


def _ingest_test_get_upload_path_side_effect(tmpdir: str):
    """Match ``get_upload_path`` behavior for ingest tests.

    File uploads use a non-empty filename; collection lock uses ``filename == ""``.
    """

    base = Path(tmpdir)

    def _side_effect(
        filename: str,
        user_id=None,
        collection=None,
        **kwargs,
    ):
        if not filename and user_id is not None and collection is not None:
            return base / f"user_{user_id}" / collection
        name = Path(filename).name if filename else "file.txt"
        return base / name

    return _side_effect


@pytest.fixture
def mock_user():
    """Minimal user-like object for ingest dependency."""
    u = type("User", (), {"id": 1, "is_admin": False})()
    return u


def _make_mock_db():
    """Create a minimal DB session mock used by ingest tests.

    The tests explicitly configure only `query(...).filter(...).first()`; other session
    methods (e.g. add/flush/commit) are left as MagicMock defaults.
    """
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return db


@pytest.fixture
def app_with_kb(mock_user):
    """FastAPI app with kb_router and mocked auth + ingestion."""
    from xagent.web.api.kb import get_current_user

    def override_get_current_user():
        return mock_user

    def override_get_db():
        yield _make_mock_db()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.fixture
def admin_user():
    """Minimal admin user-like object for delete dependency."""
    u = type("User", (), {"id": 1, "is_admin": True})()
    return u


@pytest.fixture
def app_with_kb_admin(admin_user):
    """FastAPI app with kb_router and mocked auth as admin."""
    from xagent.web.api.kb import get_current_user

    def override_get_current_user():
        return admin_user

    def override_get_db():
        yield _make_mock_db()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    return app


def test_ingest_separators_valid_json_passed_to_config(app_with_kb, mock_user):
    """POST /api/kb/ingest with valid separators JSON passes list to IngestionConfig."""
    captured_config: list[IngestionConfig] = []

    def capture_ingestion(
        collection,
        source_path,
        *,
        ingestion_config,
        file_id=None,
        user_id,
        progress_manager=None,
        is_admin=False,
    ):
        captured_config.append(ingestion_config)
        return IngestionResult(
            status="success",
            doc_id="test-doc",
            chunk_count=1,
            embedding_count=1,
            message="ok",
            completed_steps=[],
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(
                "xagent.web.api.kb.run_document_ingestion",
                side_effect=capture_ingestion,
            ),
            patch("xagent.web.api.kb.get_upload_path") as mock_path,
        ):
            mock_path.side_effect = _ingest_test_get_upload_path_side_effect(tmpdir)

            payload = {
                "file": ("test.txt", io.BytesIO(b"hello world"), "text/plain"),
                "collection": "test_coll",
                "chunk_strategy": "recursive",
                "chunk_size": "1000",
                "chunk_overlap": "200",
                "separators": json.dumps(["\n\n", "\n", "。"]),
            }

            client = TestClient(app_with_kb)
            response = client.post(
                "/api/kb/ingest",
                data={
                    "collection": payload["collection"],
                    "chunk_strategy": payload["chunk_strategy"],
                    "chunk_size": payload["chunk_size"],
                    "chunk_overlap": payload["chunk_overlap"],
                    "separators": payload["separators"],
                },
                files={"file": payload["file"]},
            )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].separators == ["\n\n", "\n", "。"]


def test_delete_collection_removes_only_current_user_docs_when_name_is_shared(
    app_with_kb, mock_user
):
    """Non-admin deletes own documents even when other users share the collection name."""
    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch("xagent.web.api.kb.delete_collection_physical_dir") as mock_physical,
        patch("xagent.web.api.kb.delete_collection_uploaded_files", return_value=0),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        # Simulate total_count=5 and own_count=3 for the same collection.
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 5},
            {"test_collection": 3},
            {"test_collection": 5},
            {"test_collection": 3},
        ]
        mock_store.list_document_records.side_effect = [[], []]
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="test_collection",
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )
        mock_physical.return_value = MagicMock(
            status="not_found", error=None, collection_dir=None
        )

        client = TestClient(app_with_kb)
        resp = client.delete("/api/kb/collections/test_collection")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    mock_delete_collection.assert_called_once_with(
        "test_collection", mock_user.id, False
    )


def test_delete_collection_rechecks_permission_before_full_delete(
    app_with_kb, mock_user
):
    """Single delete rechecks live state but mixed ownership still deletes only own docs."""
    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch("xagent.web.api.kb.delete_collection_physical_dir") as mock_physical,
        patch("xagent.web.api.kb.delete_collection_uploaded_files", return_value=0),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 2},
            {"test_collection": 1},
            {"test_collection": 2},
            {"test_collection": 1},
        ]
        mock_store.list_document_records.side_effect = [[], []]
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="test_collection",
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )
        mock_physical.return_value = MagicMock(
            status="not_found", error=None, collection_dir=None
        )

        client = TestClient(app_with_kb)
        resp = client.delete("/api/kb/collections/test_collection")

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    mock_delete_collection.assert_called_once_with(
        "test_collection", mock_user.id, False
    )
    mock_physical.assert_called_once()


def test_delete_collection_removes_stale_config_without_deleting_other_users_docs(
    app_with_kb, mock_user
):
    """Config-only stale entries should be removable without touching documents."""
    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(return_value=None)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch("xagent.web.api.kb.delete_collection_physical_dir") as mock_physical,
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync",
            return_value={"config_rows": 1, "metadata_rows": 0},
        ) as mock_delete_config,
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=mock_metadata_store,
        ),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 0},
            {"test_collection": 1},
            {"test_collection": 0},
        ]

        client = TestClient(app_with_kb)
        resp = client.delete("/api/kb/collections/test_collection")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "knowledge base list" in data["message"]
    assert data["deleted_counts"] == {"config_rows": 1, "metadata_rows": 0}
    mock_delete_config.assert_called_once_with(
        collection_name="test_collection",
        user_id=mock_user.id,
        is_admin=False,
        delete_orphaned_metadata=False,
    )
    mock_delete_collection.assert_not_called()
    mock_physical.assert_not_called()
    # config_only path must not delete global metadata used by other users.
    mock_metadata_store.delete_collection.assert_not_called()


def test_batch_delete_allows_config_only_stale_collection(app_with_kb, mock_user):
    """Batch preflight should pass own_count=0 stale config entries to cleanup."""
    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(return_value=None)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync",
            return_value={"config_rows": 1, "metadata_rows": 0},
        ),
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=mock_metadata_store,
        ),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        # Preflight makes a single batched call per filter (totals/owns), then
        # the delete path rechecks live mode before mutating config.
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 0},
            {"test_collection": 1},
            {"test_collection": 0},
        ]

        client = TestClient(app_with_kb)
        resp = client.post(
            "/api/kb/collections/batch-delete",
            json={"collection_names": ["test_collection"]},
        )

    assert resp.status_code == 200
    assert resp.json()["deleted"] == ["test_collection"]
    assert resp.json()["failed"] == []
    mock_delete_collection.assert_not_called()
    mock_metadata_store.delete_collection.assert_not_called()


def test_batch_config_only_preflight_can_become_full_delete(app_with_kb, mock_user):
    """A stale config-only preflight must not hide a newly owned collection."""
    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(return_value=None)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch("xagent.web.api.kb.delete_collection_physical_dir") as mock_physical,
        patch("xagent.web.api.kb.delete_collection_uploaded_files", return_value=0),
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync"
        ) as mock_delete_config,
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=mock_metadata_store,
        ),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 0},
            {"test_collection": 1},
            {"test_collection": 1},
        ]
        mock_store.list_document_records.side_effect = [[], []]
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="test_collection",
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )
        mock_physical.return_value = MagicMock(
            status="not_found", error=None, collection_dir=None
        )

        client = TestClient(app_with_kb)
        resp = client.post(
            "/api/kb/collections/batch-delete",
            json={"collection_names": ["test_collection"]},
        )

    assert resp.status_code == 200
    assert resp.json()["deleted"] == ["test_collection"]
    mock_delete_config.assert_not_called()
    mock_delete_collection.assert_called_once_with(
        "test_collection", mock_user.id, False
    )
    mock_metadata_store.delete_collection.assert_not_called()


def test_batch_delete_rechecks_and_deletes_only_current_user_docs_for_shared_name(
    app_with_kb, mock_user
):
    """A stale batch preflight rechecks, then tenant-scoped delete handles mixed ownership."""
    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(return_value=None)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch("xagent.web.api.kb.delete_collection_physical_dir") as mock_physical,
        patch("xagent.web.api.kb.delete_collection_uploaded_files", return_value=0),
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=mock_metadata_store,
        ),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 1},
            {"test_collection": 2},
            {"test_collection": 1},
        ]
        mock_store.list_document_records.side_effect = [[], []]
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="test_collection",
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )
        mock_physical.return_value = MagicMock(
            status="not_found", error=None, collection_dir=None
        )

        client = TestClient(app_with_kb)
        resp = client.post(
            "/api/kb/collections/batch-delete",
            json={"collection_names": ["test_collection"]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == ["test_collection"]
    assert body["failed"] == []
    mock_delete_collection.assert_called_once_with(
        "test_collection", mock_user.id, False
    )
    mock_physical.assert_called_once()
    mock_metadata_store.delete_collection.assert_not_called()


def test_delete_collection_returns_not_found_when_user_config_already_cleaned(
    app_with_kb, mock_user
):
    """Config-only cleanup must require a real user config row."""
    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(return_value=None)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch("xagent.web.api.kb.delete_collection_physical_dir") as mock_physical,
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync",
            return_value={"config_rows": 0, "metadata_rows": 0},
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=mock_metadata_store,
        ),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 0},
            {"test_collection": 1},
            {"test_collection": 0},
        ]

        client = TestClient(app_with_kb)
        resp = client.delete("/api/kb/collections/test_collection")

    assert resp.status_code == 404
    assert "not in your knowledge base list" in resp.json()["detail"]
    mock_delete_collection.assert_not_called()
    mock_physical.assert_not_called()
    mock_metadata_store.delete_collection.assert_not_called()


def test_batch_delete_fails_when_config_only_row_is_absent(app_with_kb, mock_user):
    """Batch config-only cleanup should not report success without user config."""
    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(return_value=None)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync",
            return_value={"config_rows": 0, "metadata_rows": 0},
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=mock_metadata_store,
        ),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 0},
            {"test_collection": 1},
            {"test_collection": 0},
        ]

        client = TestClient(app_with_kb)
        resp = client.post(
            "/api/kb/collections/batch-delete",
            json={"collection_names": ["test_collection"]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == []
    assert len(body["failed"]) == 1
    assert body["failed"][0]["name"] == "test_collection"
    assert "not in your knowledge base list" in body["failed"][0]["error"]
    mock_delete_collection.assert_not_called()
    mock_metadata_store.delete_collection.assert_not_called()


def test_batch_delete_marks_config_cleanup_failure_as_failed(app_with_kb, mock_user):
    """When _remove_user_collection_config raises in config_only path, item should fail."""
    mock_metadata_store = MagicMock()
    mock_metadata_store.delete_collection = AsyncMock(return_value=None)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync",
            side_effect=RuntimeError("metadata store down"),
        ),
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=mock_metadata_store,
        ),
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 0},
            {"test_collection": 1},
            {"test_collection": 0},
        ]

        client = TestClient(app_with_kb)
        resp = client.post(
            "/api/kb/collections/batch-delete",
            json={"collection_names": ["test_collection"]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == []
    assert len(body["failed"]) == 1
    failed_item = body["failed"][0]
    assert failed_item["name"] == "test_collection"
    assert "Failed to delete collection configuration" in failed_item["error"]
    mock_delete_collection.assert_not_called()
    mock_metadata_store.delete_collection.assert_not_called()


def test_delete_collection_allowed_for_admin_with_other_users_docs(
    app_with_kb_admin, admin_user
):
    """Admin user can delete collections even when they contain other users' docs."""
    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.list_document_records.return_value = []
        # Admin path bypasses permission pre-check, keep a safe default.
        mock_store.count_documents_grouped_by_collection.return_value = {
            "test_collection": 5
        }

        # Simulate successful delete_collection
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="test_collection",
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        client = TestClient(app_with_kb_admin)
        resp = client.delete("/api/kb/collections/test_collection")

    assert resp.status_code == 200
    mock_delete_collection.assert_called_once()


def test_document_delete_cleanup_removes_config_after_last_owned_doc():
    """Last owned document deletion should clear only the current user's config."""
    from xagent.web.api import kb

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync",
            return_value={"config_rows": 1, "metadata_rows": 0},
        ) as mock_delete_config,
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 1},
            {"test_collection": 0},
        ]

        result = kb._cleanup_collection_config_if_no_owned_documents(
            "test_collection",
            user_id=(mock_user_id := 1),
        )

    assert result == {"config_rows": 1, "metadata_rows": 0}
    mock_delete_config.assert_called_once_with(
        collection_name="test_collection",
        user_id=mock_user_id,
        is_admin=False,
        delete_orphaned_metadata=False,
    )


def test_document_delete_cleanup_deletes_orphaned_metadata_when_collection_empty():
    """If no documents remain globally, orphan metadata can be removed too."""
    from xagent.web.api import kb

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch(
            "xagent.web.api.kb.delete_collection_metadata_sync",
            return_value={"config_rows": 1, "metadata_rows": 1},
        ) as mock_delete_config,
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 0},
            {"test_collection": 0},
        ]

        result = kb._cleanup_collection_config_if_no_owned_documents(
            "test_collection",
            user_id=1,
        )

    assert result == {"config_rows": 1, "metadata_rows": 1}
    mock_delete_config.assert_called_once_with(
        collection_name="test_collection",
        user_id=1,
        is_admin=False,
        delete_orphaned_metadata=True,
    )


def test_delete_document_forbidden_for_non_admin_other_users_doc(
    app_with_kb, mock_user
):
    """Non-admin user should not be able to delete documents they don't own."""
    with (
        patch(
            "xagent.providers.vector_store.lancedb.get_connection_from_env"
        ) as mock_get_conn,
        patch(
            "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.ensure_documents_table"
        ) as mock_ensure_docs,
        patch(
            "xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils.query_to_list"
        ) as mock_query_to_list,
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document"
        ) as mock_delete_document,
    ):
        mock_ensure_docs.return_value = None

        # We don't care about the actual connection, open_table, or filter expression here,
        # because query_to_list receives the already-filtered search object.
        mock_conn = MagicMock()
        mock_table = MagicMock()
        mock_conn.open_table.return_value = mock_table
        mock_get_conn.return_value = mock_conn

        # Simulate that, after applying user filter, there are no matching records
        mock_query_to_list.return_value = []

        client = TestClient(app_with_kb)
        resp = client.delete(
            "/api/kb/collections/test_collection/documents/doc.txt",
        )

    # No accessible document -> 403 from delete_document_api, and delete_document must not be called
    assert resp.status_code == 403
    body = resp.json()
    assert "Access denied for collection" in body.get("detail", "")
    assert "test_collection" in body.get("detail", "")
    mock_delete_document.assert_not_called()


def test_delete_document_keeps_success_when_config_cleanup_fails(
    app_with_kb_admin, admin_user
):
    """Cleanup failures must not downgrade overall status from success."""
    from fastapi import HTTPException as _HTTPException

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document"
        ) as mock_delete_document,
        patch(
            "xagent.web.api.kb._cleanup_collection_config_if_no_owned_documents",
            side_effect=_HTTPException(
                status_code=500,
                detail="Failed to delete collection configuration: boom",
            ),
        ) as mock_cleanup,
    ):
        mock_record = MagicMock()
        mock_record.doc_id = "doc_123"
        mock_record.source_path = "/tmp/doc.txt"
        mock_record.metadata = {}
        mock_get_vector_store.return_value.list_document_records.return_value = [
            mock_record
        ]
        mock_delete_document.return_value = type(
            "DeleteResult",
            (),
            {"status": "success", "message": "ok"},
        )()

        client = TestClient(app_with_kb_admin)
        resp = client.delete(
            "/api/kb/collections/test_collection/documents/doc.txt",
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["deleted_doc_ids"] == ["doc_123"]
    assert "collection_config_cleanup_error" in body
    assert "boom" in body["collection_config_cleanup_error"]
    assert "errors" not in body
    mock_cleanup.assert_called_once()


def test_delete_document_allowed_for_admin_any_doc(app_with_kb_admin, admin_user):
    """Admin user can delete documents regardless of owner."""
    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document"
        ) as mock_delete_document,
    ):
        # New API flow resolves by vector_store.list_document_records first.
        mock_record = MagicMock()
        mock_record.doc_id = "doc_123"
        mock_record.source_path = "/tmp/doc.txt"
        mock_record.metadata = {}
        mock_get_vector_store.return_value.list_document_records.return_value = [
            mock_record
        ]
        mock_delete_document.return_value = type(
            "DeleteResult",
            (),
            {"status": "success", "message": "ok"},
        )()

        client = TestClient(app_with_kb_admin)
        resp = client.delete(
            "/api/kb/collections/test_collection/documents/doc.txt",
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["deleted_doc_ids"] == ["doc_123"]
    # delete_document should be invoked once with the resolved doc_id
    mock_delete_document.assert_called_once()


def test_ingest_separators_missing_uses_none(app_with_kb, mock_user):
    """POST without separators field leaves config.separators as None."""
    captured_config: list[IngestionConfig] = []

    def capture_ingestion(
        collection,
        source_path,
        *,
        ingestion_config,
        file_id=None,
        user_id,
        progress_manager=None,
        is_admin=False,
    ):
        captured_config.append(ingestion_config)
        return IngestionResult(
            status="success",
            doc_id="test-doc",
            chunk_count=1,
            embedding_count=1,
            message="ok",
            completed_steps=[],
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(
                "xagent.web.api.kb.run_document_ingestion",
                side_effect=capture_ingestion,
            ),
            patch("xagent.web.api.kb.get_upload_path") as mock_path,
        ):
            mock_path.side_effect = _ingest_test_get_upload_path_side_effect(tmpdir)

            client = TestClient(app_with_kb)
            response = client.post(
                "/api/kb/ingest",
                data={
                    "collection": "test_coll",
                    "chunk_strategy": "recursive",
                    "chunk_size": "1000",
                    "chunk_overlap": "200",
                },
                files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].separators is None


def test_ingest_returns_413_when_file_exceeds_limit(app_with_kb, monkeypatch):
    """KB ingest should return 413 when the uploaded file exceeds the configured limit."""
    import xagent.web.api.kb

    monkeypatch.setattr(xagent.web.api.kb, "MAX_FILE_SIZE", 4)

    client = TestClient(app_with_kb)
    response = client.post(
        "/api/kb/ingest",
        data={"collection": "test_coll"},
        files={"file": ("big.txt", io.BytesIO(b"12345"), "text/plain")},
    )

    assert response.status_code == 413
    assert "maximum limit" in response.json()["detail"].lower()


def test_ingest_job_uses_staged_snapshot_in_payload(app_with_kb):
    captured_payload = {}

    def fake_create_background_job(db, *, payload, **kwargs):
        captured_payload.update(payload)
        return BackgroundJob(
            id="job-1",
            user_id=1,
            job_type=BackgroundJobType.KB_INGEST_DOCUMENT.value,
            queue="kb_ingest",
            status=BackgroundJobStatus.ENQUEUED.value,
            payload=payload,
            progress={"message": "Queued", "completed": 0, "total": 1},
            attempts=0,
            max_attempts=3,
        )

    async def fake_enqueue(db, job):
        return job

    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("xagent.web.api.kb.get_upload_path") as mock_path,
            patch(
                "xagent.web.api.kb._ensure_background_job_queue_available_async",
                new=AsyncMock(),
            ),
            patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError),
            patch("xagent.web.api.kb.get_metadata_store", return_value=metadata_store),
            patch(
                "xagent.web.api.kb.get_non_terminal_background_job_by_idempotency_key",
                return_value=None,
            ),
            patch(
                "xagent.web.api.kb.create_background_job",
                side_effect=fake_create_background_job,
            ),
            patch("xagent.web.api.kb.admit_kb_ingest_target"),
            patch(
                "xagent.web.api.kb._enqueue_background_job_or_503_async",
                side_effect=fake_enqueue,
            ),
        ):
            mock_path.side_effect = _ingest_test_get_upload_path_side_effect(tmpdir)
            client = TestClient(app_with_kb)
            response = client.post(
                "/api/kb/ingest/jobs",
                data={"collection": "test_coll"},
                files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            )

        assert response.status_code == 202
        source_path = Path(captured_payload["source_path"])
        target_path = Path(captured_payload["target_path"])
        assert source_path != target_path
        assert ".background-ingest" in source_path.parts
        assert source_path.read_bytes() == b"hello"
        assert not target_path.exists()
        assert captured_payload["file_size"] == 5
        assert captured_payload["file_id"]
        assert captured_payload["generation_id"]
        assert (
            captured_payload["file_sha256"]
            == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        metadata_store.save_collection_config.assert_not_awaited()


def test_ingest_web_job_payload_records_new_collection_state(app_with_kb):
    captured_payload = {}

    def fake_create_background_job(db, *, payload, **kwargs):
        captured_payload.update(payload)
        return BackgroundJob(
            id="job-1",
            user_id=1,
            job_type=BackgroundJobType.KB_INGEST_WEB.value,
            queue="kb_ingest",
            status=BackgroundJobStatus.ENQUEUED.value,
            payload=payload,
            progress={"message": "Queued", "completed": 0, "total": 1},
            attempts=0,
            max_attempts=3,
        )

    async def fake_enqueue(db, job):
        return job

    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()

    with (
        patch(
            "xagent.web.api.kb._ensure_background_job_queue_available_async",
            new=AsyncMock(),
        ),
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError),
        patch("xagent.web.api.kb.get_metadata_store", return_value=metadata_store),
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_non_terminal_background_job_by_idempotency_key",
            return_value=None,
        ),
        patch(
            "xagent.web.api.kb.create_background_job",
            side_effect=fake_create_background_job,
        ),
        patch(
            "xagent.web.api.kb._enqueue_background_job_or_503_async",
            side_effect=fake_enqueue,
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web/jobs",
            data={
                "collection": "web_jobs_new_collection",
                "start_url": "https://example.com",
            },
        )

    assert response.status_code == 202
    assert captured_payload["collection_existed_before"] is False


def test_ingest_web_job_does_not_write_collection_config_when_enqueue_fails(
    app_with_kb,
):
    def fake_create_background_job(db, *, payload, **kwargs):
        return BackgroundJob(
            id="job-1",
            user_id=1,
            job_type=BackgroundJobType.KB_INGEST_WEB.value,
            queue="kb_ingest",
            status=BackgroundJobStatus.PENDING.value,
            payload=payload,
            progress={"message": "Queued", "completed": 0, "total": 1},
            attempts=0,
            max_attempts=3,
        )

    async def fake_enqueue(db, job):
        raise HTTPException(
            status_code=503, detail="Background job queue is unavailable"
        )

    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock(
        return_value={"metadata_rows": 0, "config_rows": 1}
    )

    with (
        patch(
            "xagent.web.api.kb._ensure_background_job_queue_available_async",
            new=AsyncMock(),
        ),
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError),
        patch("xagent.web.api.kb.get_metadata_store", return_value=metadata_store),
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_non_terminal_background_job_by_idempotency_key",
            return_value=None,
        ),
        patch(
            "xagent.web.api.kb.create_background_job",
            side_effect=fake_create_background_job,
        ),
        patch(
            "xagent.web.api.kb._enqueue_background_job_or_503_async",
            side_effect=fake_enqueue,
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web/jobs",
            data={
                "collection": "web_jobs_enqueue_failure",
                "start_url": "https://example.com",
            },
        )

    assert response.status_code == 503
    metadata_store.save_collection_config.assert_not_awaited()
    metadata_store.delete_collection_metadata.assert_not_awaited()


def test_ingest_separators_invalid_json_request_succeeds_uses_default(
    app_with_kb, mock_user
):
    """POST with invalid separators JSON still returns 200; config uses default (None)."""
    captured_config: list[IngestionConfig] = []

    def capture_ingestion(
        collection,
        source_path,
        *,
        ingestion_config,
        file_id=None,
        user_id,
        progress_manager=None,
        is_admin=False,
    ):
        captured_config.append(ingestion_config)
        return IngestionResult(
            status="success",
            doc_id="test-doc",
            chunk_count=1,
            embedding_count=1,
            message="ok",
            completed_steps=[],
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(
                "xagent.web.api.kb.run_document_ingestion",
                side_effect=capture_ingestion,
            ),
            patch("xagent.web.api.kb.get_upload_path") as mock_path,
        ):
            mock_path.side_effect = _ingest_test_get_upload_path_side_effect(tmpdir)

            client = TestClient(app_with_kb)
            response = client.post(
                "/api/kb/ingest",
                data={
                    "collection": "test_coll",
                    "chunk_strategy": "recursive",
                    "chunk_size": "1000",
                    "chunk_overlap": "200",
                    "separators": "not json",
                },
                files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].separators is None


def test_ingest_separators_empty_array_uses_none(app_with_kb, mock_user):
    """POST with separators='[]' results in config.separators being empty list []."""
    captured_config: list[IngestionConfig] = []

    def capture_ingestion(
        collection,
        source_path,
        *,
        ingestion_config,
        file_id=None,
        user_id,
        progress_manager=None,
        is_admin=False,
    ):
        captured_config.append(ingestion_config)
        return IngestionResult(
            status="success",
            doc_id="test-doc",
            chunk_count=1,
            embedding_count=1,
            message="ok",
            completed_steps=[],
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch(
                "xagent.web.api.kb.run_document_ingestion",
                side_effect=capture_ingestion,
            ),
            patch("xagent.web.api.kb.get_upload_path") as mock_path,
        ):
            mock_path.side_effect = _ingest_test_get_upload_path_side_effect(tmpdir)

            client = TestClient(app_with_kb)
            response = client.post(
                "/api/kb/ingest",
                data={
                    "collection": "test_coll",
                    "chunk_strategy": "recursive",
                    "chunk_size": "1000",
                    "chunk_overlap": "200",
                    "separators": "[]",
                },
                files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].separators == []


def test_ingest_returns_403_when_file_save_fails(app_with_kb, mock_user):
    """File system save errors should be normalized to HTTP 403 by decorator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("xagent.web.api.kb.get_upload_path") as mock_path,
            patch("builtins.open", side_effect=PermissionError("disk denied")),
        ):
            mock_path.return_value = str(Path(tmpdir) / "test.txt")

            client = TestClient(app_with_kb)
            response = client.post(
                "/api/kb/ingest",
                data={"collection": "test_coll"},
                files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            )

    assert response.status_code == 403
    assert "File system error:" in str(response.json().get("detail", ""))


async def _fake_run_web_ingestion(
    collection,
    crawl_config,
    *,
    ingestion_config,
    user_id,
    is_admin=False,
    file_handler=None,
):
    """Async fake that captures ingestion_config and returns WebIngestionResult."""
    captured_config: list = _fake_run_web_ingestion.captured  # type: ignore[attr-defined]
    captured_config.append(ingestion_config)
    captured_crawl_config = getattr(_fake_run_web_ingestion, "captured_crawl", None)
    if captured_crawl_config is not None:
        captured_crawl_config.append(crawl_config)
    return WebIngestionResult(
        status="success",
        collection=collection,
        total_urls_found=0,
        pages_crawled=0,
        pages_failed=0,
        documents_created=0,
        chunks_created=0,
        embeddings_created=0,
        message="ok",
        elapsed_time_ms=0,
    )


def test_ingest_web_separators_valid_json_passed_to_config(app_with_kb, monkeypatch):
    """POST /api/kb/ingest-web with valid separators passes list to IngestionConfig."""
    monkeypatch.setenv(WEB_CRAWL_TLS_IMPERSONATE, "auto")
    captured_config: list[IngestionConfig] = []
    _fake_run_web_ingestion.captured = captured_config  # type: ignore[attr-defined]
    captured_crawl_config: list = []
    _fake_run_web_ingestion.captured_crawl = captured_crawl_config  # type: ignore[attr-defined]

    with patch(
        "xagent.web.api.kb.run_web_ingestion", side_effect=_fake_run_web_ingestion
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_coll",
                "start_url": "https://example.com",
                "chunk_strategy": "recursive",
                "chunk_size": "1000",
                "chunk_overlap": "200",
                "separators": json.dumps(["\n", " "]),
            },
        )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].separators == ["\n", " "]
    assert len(captured_crawl_config) == 1
    assert captured_crawl_config[0].tls_impersonate == "auto"


def test_ingest_web_separators_invalid_json_request_succeeds(app_with_kb):
    """POST ingest-web with invalid separators JSON still returns 200; config has None."""
    captured_config: list[IngestionConfig] = []
    _fake_run_web_ingestion.captured = captured_config  # type: ignore[attr-defined]

    with patch(
        "xagent.web.api.kb.run_web_ingestion", side_effect=_fake_run_web_ingestion
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_coll",
                "start_url": "https://example.com",
                "chunk_strategy": "recursive",
                "chunk_size": "1000",
                "chunk_overlap": "200",
                "separators": "[1,2,3]",
            },
        )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].separators is None


def test_ingest_web_missing_protocol_returns_422(app_with_kb):
    """POST ingest-web rejects start_url values without an explicit HTTP(S) scheme."""
    with patch("xagent.web.api.kb.run_web_ingestion") as mock_run_web_ingestion:
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_coll",
                "start_url": "www.example.com",
            },
        )

    assert response.status_code == 422
    assert (
        response.json()["detail"]
        == "Invalid start_url: URL must start with http:// or https://"
    )
    mock_run_web_ingestion.assert_not_called()


def test_ingest_web_uppercase_scheme_is_normalized(app_with_kb):
    """POST ingest-web should accept uppercase schemes via shared URL normalization."""
    captured_config = []

    async def capture_web_ingestion(*, crawl_config, **kwargs):
        captured_config.append(crawl_config)
        return WebIngestionResult(
            status="success",
            collection=kwargs["collection"],
            total_urls_found=1,
            pages_crawled=1,
            pages_failed=0,
            documents_created=1,
            chunks_created=1,
            embeddings_created=1,
            crawled_urls=[crawl_config.start_url],
            failed_urls={},
            message="ok",
            warnings=[],
            elapsed_time_ms=1,
        )

    with patch(
        "xagent.web.api.kb.run_web_ingestion",
        new=AsyncMock(side_effect=capture_web_ingestion),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_coll",
                "start_url": "HTTP://Example.com/docs#intro",
            },
        )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].start_url == "http://example.com/docs"


@pytest.mark.parametrize("start_url", ["http://@", "http://:80"])
def test_ingest_web_hostless_url_returns_422(app_with_kb, start_url):
    """POST ingest-web should reject hostless HTTP(S) URLs at validation time."""
    with patch("xagent.web.api.kb.run_web_ingestion") as mock_run_web_ingestion:
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={"collection": "web_coll", "start_url": start_url},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid start_url: URL must include a hostname"
    mock_run_web_ingestion.assert_not_called()


def test_ingest_web_error_cleans_new_collection_config(app_with_kb):
    """POST ingest-web should clean saved config when a new collection fails before any docs are created."""
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock(
        return_value={"metadata_rows": 0, "config_rows": 1}
    )

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_collection_sync", side_effect=ValueError("missing")
        ),
        patch(
            "xagent.web.api.kb.run_web_ingestion",
            return_value=WebIngestionResult(
                status="error",
                collection="web_new_collection",
                total_urls_found=1,
                pages_crawled=0,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[],
                failed_urls={"https://example.com": "crawl failed"},
                message="crawl failed",
                warnings=[],
                elapsed_time_ms=0,
            ),
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_new_collection",
                "start_url": "https://example.com",
            },
        )

    assert response.status_code == 500
    metadata_store.delete_collection_metadata.assert_awaited_once_with(
        collection_name="web_new_collection",
        user_id=1,
        is_admin=False,
        delete_orphaned_metadata=True,
    )


def test_ingest_web_error_with_successful_docs_keeps_new_collection_config(
    app_with_kb,
):
    """Rollback failures surface as error without deleting config used by successful pages."""
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()
    metadata_store.delete_collection = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_collection_sync", side_effect=ValueError("missing")
        ),
        patch(
            "xagent.web.api.kb.run_web_ingestion",
            return_value=WebIngestionResult(
                status="error",
                collection="web_new_collection",
                total_urls_found=2,
                pages_crawled=2,
                pages_failed=1,
                documents_created=1,
                chunks_created=1,
                embeddings_created=1,
                crawled_urls=["https://example.com/ok"],
                failed_urls={"https://example.com/bad": "rollback failed"},
                message="rollback failed",
                warnings=[],
                elapsed_time_ms=0,
            ),
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_new_collection",
                "start_url": "https://example.com",
            },
        )

    assert response.status_code == 500
    metadata_store.delete_collection_metadata.assert_not_awaited()
    metadata_store.save_collection_config.assert_awaited_once()
    metadata_store.delete_collection.assert_not_awaited()


def test_ingest_web_error_with_rollback_side_effects_keeps_new_collection_config(
    app_with_kb,
):
    """A single-page rollback failure should not orphan rows by deleting config."""
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_collection_sync", side_effect=ValueError("missing")
        ),
        patch(
            "xagent.web.api.kb.run_web_ingestion",
            return_value=WebIngestionResult(
                status="error",
                collection="web_new_collection",
                total_urls_found=1,
                pages_crawled=1,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=["https://example.com/page"],
                failed_urls={"https://example.com/page": "rollback failed"},
                message="rollback failed",
                warnings=[],
                elapsed_time_ms=0,
                side_effects_may_remain=True,
            ),
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_new_collection",
                "start_url": "https://example.com",
            },
        )

    assert response.status_code == 500
    assert response.json()["side_effects_may_remain"] is True
    metadata_store.delete_collection_metadata.assert_not_awaited()
    metadata_store.save_collection_config.assert_awaited_once()


def test_ingest_web_existing_collection_error_restores_previous_config(app_with_kb):
    """POST ingest-web should restore old config when an existing collection fails."""
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":333}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()
    metadata_store.delete_collection = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_collection_sync", return_value=MagicMock()),
        patch(
            "xagent.web.api.kb.run_web_ingestion",
            return_value=WebIngestionResult(
                status="error",
                collection="web_existing_collection",
                total_urls_found=1,
                pages_crawled=0,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[],
                failed_urls={"https://example.com": "crawl failed"},
                message="crawl failed",
                warnings=[],
                elapsed_time_ms=0,
            ),
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_existing_collection",
                "start_url": "https://example.com",
                "chunk_size": "2048",
            },
        )

    assert response.status_code == 500
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "web_existing_collection",
        "config_json": '{"chunk_size":333}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()


def test_ingest_web_config_only_collection_error_restores_previous_config(
    app_with_kb,
):
    """A config-only web collection should restore old config on failed ingest."""
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":333}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()
    metadata_store.delete_collection = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb.get_collection_sync", side_effect=ValueError("missing")
        ),
        patch(
            "xagent.web.api.kb.run_web_ingestion",
            return_value=WebIngestionResult(
                status="error",
                collection="web_config_only_collection",
                total_urls_found=1,
                pages_crawled=0,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[],
                failed_urls={"https://example.com": "crawl failed"},
                message="crawl failed",
                warnings=[],
                elapsed_time_ms=0,
            ),
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_config_only_collection",
                "start_url": "https://example.com",
                "chunk_size": "2048",
            },
        )

    assert response.status_code == 500
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "web_config_only_collection",
        "config_json": '{"chunk_size":333}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()
    metadata_store.delete_collection.assert_awaited_once_with(
        "web_config_only_collection"
    )


def test_ingest_web_existing_collection_partial_restores_previous_config(app_with_kb):
    """POST ingest-web should restore old config when an existing collection is partial."""
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":555}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_collection_sync", return_value=MagicMock()),
        patch(
            "xagent.web.api.kb.run_web_ingestion",
            return_value=WebIngestionResult(
                status="partial",
                collection="web_existing_collection",
                total_urls_found=2,
                pages_crawled=1,
                pages_failed=1,
                documents_created=1,
                chunks_created=1,
                embeddings_created=1,
                crawled_urls=["https://example.com/ok"],
                failed_urls={"https://example.com/bad": "embedding failed"},
                message="partial failure",
                warnings=[],
                elapsed_time_ms=0,
            ),
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_existing_collection",
                "start_url": "https://example.com",
                "chunk_size": "2048",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "web_existing_collection",
        "config_json": '{"chunk_size":555}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()


def test_ingest_web_surfaces_embedding_configuration_fix_guidance(app_with_kb):
    """POST ingest-web should explain how to fix embedding configuration errors."""
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock(
        return_value={"metadata_rows": 0, "config_rows": 1}
    )

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_collection_sync", side_effect=ValueError("missing")
        ),
        patch(
            "xagent.web.api.kb.run_web_ingestion",
            return_value=WebIngestionResult(
                status="error",
                collection="web_embedding_error",
                total_urls_found=1,
                pages_crawled=0,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[],
                failed_urls={"https://example.com": "embedding failed"},
                message=(
                    "Model 'text-embedding-v4' not found in hub and no "
                    "environment configuration available for embedding."
                ),
                warnings=[],
                elapsed_time_ms=0,
            ),
        ),
    ):
        client = TestClient(app_with_kb)
        response = client.post(
            "/api/kb/ingest-web",
            data={
                "collection": "web_embedding_error",
                "start_url": "https://example.com",
            },
        )

    assert response.status_code == 500
    message = response.json()["message"]
    assert (
        "Cause: knowledge-base ingestion requires a resolvable embedding model"
        in message
    )
    assert "Current embedding_model_id: 'text-embedding-v4'." in message
    assert "How to fix:" in message
