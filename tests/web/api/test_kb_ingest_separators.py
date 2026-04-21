"""Tests for /api/kb/ingest and /api/kb/ingest-web separators parameter parsing and passthrough."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionConfig,
    IngestionResult,
    WebIngestionResult,
)
from xagent.web.api.kb import kb_router
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


def test_delete_collection_forbidden_for_non_admin_with_other_users_docs(
    app_with_kb, mock_user
):
    """Non-admin is rejected by _check_can_delete_collection before delete_collection."""
    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
    ):
        mock_store = MagicMock()
        mock_get_vector_store.return_value = mock_store
        mock_store.list_document_records.return_value = []
        # Simulate total_count=5 and own_count=3 for the same collection.
        mock_store.count_documents_grouped_by_collection.side_effect = [
            {"test_collection": 5},
            {"test_collection": 3},
        ]

        client = TestClient(app_with_kb)
        resp = client.delete("/api/kb/collections/test_collection")

    assert resp.status_code == 403
    assert "admin users" in resp.json()["detail"]
    mock_delete_collection.assert_not_called()


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

    # No accessible document -> 404 from delete_document_api, and delete_document must not be called
    assert resp.status_code == 404
    body = resp.json()
    assert "Document not found" in body.get("detail", "")
    mock_delete_document.assert_not_called()


def test_delete_document_allowed_for_admin_any_doc(app_with_kb_admin, admin_user):
    """Admin user can delete documents regardless of owner."""
    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user"
        ) as mock_list_documents_for_user,
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document"
        ) as mock_delete_document,
    ):
        # For admin, list_documents path should return all matching records.
        mock_list_documents_for_user.return_value = [
            {
                "collection": "test_collection",
                "doc_id": "doc_123",
                "source_path": "/tmp/doc.txt",
            }
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


def test_ingest_web_separators_valid_json_passed_to_config(app_with_kb):
    """POST /api/kb/ingest-web with valid separators passes list to IngestionConfig."""
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
                "separators": json.dumps(["\n", " "]),
            },
        )

    assert response.status_code == 200
    assert len(captured_config) == 1
    assert captured_config[0].separators == ["\n", " "]


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
