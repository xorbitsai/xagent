"""Tests for GET /api/kb/collections/{collection}/documents per-document status."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.kb import kb_router
from xagent.web.models.database import get_db


@pytest.fixture
def admin_user():
    return type("User", (), {"id": 1, "is_admin": True})()


@pytest.fixture
def app_with_kb_admin(admin_user):
    """Admin bypasses collection-access checks, keeping the test focused on the join."""
    from xagent.web.api.kb import get_current_user

    def override_get_current_user():
        return admin_user

    def override_get_db():
        yield MagicMock()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    return app


def _records():
    return [
        {"doc_id": "d1", "file_id": "f1", "source_path": "/x/a.pdf"},
        {"doc_id": "d2", "file_id": "f2", "source_path": "/x/b.pdf"},
        {"doc_id": "d3", "file_id": "f3", "source_path": "/x/legacy.txt"},
    ]


def test_document_status_joins_records_with_ingestion_status(app_with_kb_admin):
    status_rows = [
        {"doc_id": "d1", "status": "running", "message": "parsing", "updated_at": "2026-06-30T00:00:00Z"},
        {"doc_id": "d2", "status": "failed", "message": "boom", "updated_at": "2026-06-30T01:00:00Z"},
    ]

    with (
        patch("xagent.web.api.kb.list_document_records", return_value=_records()),
        patch("xagent.web.api.kb.load_ingestion_status", return_value=status_rows),
        patch("xagent.web.api.kb._build_uploaded_filename_map", return_value={}),
        patch("xagent.web.api.kb._get_document_record_file_id", side_effect=lambda r: r["file_id"]),
        patch(
            "xagent.web.api.kb._resolve_document_filename",
            side_effect=lambda r, _m: r["source_path"].rsplit("/", 1)[-1],
        ),
    ):
        client = TestClient(app_with_kb_admin)
        response = client.get("/api/kb/collections/demo/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["collection"] == "demo"
    docs = {d["filename"]: d for d in body["documents"]}

    # Running document: not deletable mid-pipeline, carries message.
    assert docs["a.pdf"]["status"] == "running"
    assert docs["a.pdf"]["can_delete"] is False
    assert docs["a.pdf"]["message"] == "parsing"

    # Failed document: terminal, deletable.
    assert docs["b.pdf"]["status"] == "failed"
    assert docs["b.pdf"]["can_delete"] is True

    # Legacy indexed document with no status record defaults to success.
    assert docs["legacy.txt"]["status"] == "success"
    assert docs["legacy.txt"]["can_delete"] is True

    # Sorted by filename.
    assert [d["filename"] for d in body["documents"]] == ["a.pdf", "b.pdf", "legacy.txt"]


def test_document_status_degrades_when_status_store_unavailable(app_with_kb_admin):
    """A failing status store must not break the listing; rows fall back to success."""
    with (
        patch("xagent.web.api.kb.list_document_records", return_value=_records()[:1]),
        patch("xagent.web.api.kb.load_ingestion_status", side_effect=RuntimeError("db down")),
        patch("xagent.web.api.kb._build_uploaded_filename_map", return_value={}),
        patch("xagent.web.api.kb._get_document_record_file_id", side_effect=lambda r: r["file_id"]),
        patch(
            "xagent.web.api.kb._resolve_document_filename",
            side_effect=lambda r, _m: r["source_path"].rsplit("/", 1)[-1],
        ),
    ):
        client = TestClient(app_with_kb_admin)
        response = client.get("/api/kb/collections/demo/documents")

    assert response.status_code == 200
    body = response.json()
    assert len(body["documents"]) == 1
    assert body["documents"][0]["status"] == "success"


def test_document_status_rejects_invalid_collection_name(app_with_kb_admin):
    client = TestClient(app_with_kb_admin)
    response = client.get("/api/kb/collections/..%2F..%2Fetc/documents")
    assert response.status_code in (404, 422)


def test_delete_blocks_in_progress_document(app_with_kb_admin):
    """A document still being processed (running) is rejected with 409."""
    record = type("Rec", (), {"doc_id": "d1", "source_path": "/x/a.pdf"})()

    with (
        patch(
            "xagent.web.api.kb._ensure_collection_access_for_document_delete",
            return_value=None,
        ),
        patch("xagent.web.api.kb.get_vector_index_store") as store,
        patch("xagent.web.api.kb._build_uploaded_filename_map", return_value={}),
        patch(
            "xagent.web.api.kb._get_document_record_file_id",
            side_effect=lambda r: None,
        ),
        patch(
            "xagent.web.api.kb._resolve_document_filename",
            side_effect=lambda r, _m: "a.pdf",
        ),
        patch(
            "xagent.web.api.kb.load_ingestion_status",
            return_value=[{"doc_id": "d1", "status": "running"}],
        ),
        patch("xagent.web.api.kb.delete_document") as delete_document,
    ):
        store.return_value.list_document_records.return_value = [record]
        client = TestClient(app_with_kb_admin)
        response = client.delete(
            "/api/kb/collections/demo/documents/a.pdf?doc_id=d1"
        )

    assert response.status_code == 409
    # The guard must run before any actual deletion is attempted.
    delete_document.assert_not_called()


def test_delete_force_bypasses_in_progress_guard(app_with_kb_admin):
    """force=true lets a stranded in-progress row be deleted anyway."""
    record = type("Rec", (), {"doc_id": "d1", "source_path": "/x/a.pdf"})()

    with (
        patch(
            "xagent.web.api.kb._ensure_collection_access_for_document_delete",
            return_value=None,
        ),
        patch("xagent.web.api.kb.get_vector_index_store") as store,
        patch("xagent.web.api.kb._build_uploaded_filename_map", return_value={}),
        patch(
            "xagent.web.api.kb._get_document_record_file_id",
            side_effect=lambda r: None,
        ),
        patch(
            "xagent.web.api.kb._resolve_document_filename",
            side_effect=lambda r, _m: "a.pdf",
        ),
        patch(
            "xagent.web.api.kb.load_ingestion_status",
            return_value=[{"doc_id": "d1", "status": "running"}],
        ),
        patch("xagent.web.api.kb.delete_document") as delete_document,
    ):
        store.return_value.list_document_records.return_value = [record]
        delete_document.return_value = type(
            "Res", (), {"status": "success", "message": ""}
        )()
        client = TestClient(app_with_kb_admin)
        response = client.delete(
            "/api/kb/collections/demo/documents/a.pdf?doc_id=d1&force=true"
        )

    # Guard bypassed: the delete was attempted for the resolved doc_id.
    assert response.status_code == 200
    delete_document.assert_called_once()
    assert delete_document.call_args.args[1] == "d1"
