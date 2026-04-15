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
            mock_path.return_value = str(Path(tmpdir) / "test.txt")

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
            mock_path.return_value = str(Path(tmpdir) / "test.txt")

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
            mock_path.return_value = str(Path(tmpdir) / "test.txt")

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
            mock_path.return_value = str(Path(tmpdir) / "test.txt")

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
