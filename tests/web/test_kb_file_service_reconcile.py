"""Tests for uploaded files reconciliation helpers."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lancedb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.management.status import write_ingestion_status
from xagent.providers.vector_store.lancedb import get_connection_from_env
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.kb_file_service import (
    aggregate_uploaded_file_statuses,
    reconcile_uploaded_files,
)


@pytest.fixture
def reconcile_env(monkeypatch: pytest.MonkeyPatch):
    with (
        tempfile.TemporaryDirectory() as lancedb_dir,
        tempfile.TemporaryDirectory() as uploads_dir,
    ):
        monkeypatch.setenv("LANCEDB_DIR", lancedb_dir)
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", uploads_dir)

        conn = lancedb.connect(lancedb_dir)
        ensure_documents_table(conn)
        docs_table = conn.open_table("documents")

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        yield docs_table, SessionLocal, Path(uploads_dir)


def _create_user(session_local: sessionmaker, user_id: int = 1) -> None:
    db = session_local()
    db.add(
        User(
            id=user_id,
            username=f"user_{user_id}",
            password_hash="hash",
            is_admin=False,
        )
    )
    db.commit()
    db.close()


def test_aggregate_uploaded_file_statuses_returns_expected_priority(reconcile_env):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)

    file_path = uploads_dir / "user_1" / "kb" / "a.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content", encoding="utf-8")

    db = session_local()
    file_record = UploadedFile(
        user_id=1,
        filename="a.md",
        storage_path=str(file_path),
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )
    db.add(file_record)
    db.commit()
    db.refresh(file_record)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-agg",
                "file_id": file_record.file_id,
                "source_path": str(file_path),
                "user_id": 1,
            }
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-agg",
        status="success",
        message="done",
        parse_hash="",
        user_id=1,
    )

    status_map = aggregate_uploaded_file_statuses(
        file_ids=[file_record.file_id],
        user_id=1,
        is_admin=False,
    )
    assert status_map[file_record.file_id] == "SUCCESS"
    db.close()


def test_reconcile_uploaded_files_deletes_stale_failed_and_unknown(reconcile_env):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)
    db = session_local()

    failed_path = uploads_dir / "user_1" / "kb" / "failed.md"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("failed", encoding="utf-8")

    unknown_path = uploads_dir / "user_1" / "kb" / "unknown.md"
    unknown_path.write_text("unknown", encoding="utf-8")

    success_path = uploads_dir / "user_1" / "kb" / "success.md"
    success_path.write_text("success", encoding="utf-8")

    old_time = datetime.now(timezone.utc) - timedelta(days=10)

    failed_file = UploadedFile(
        user_id=1,
        filename="failed.md",
        storage_path=str(failed_path),
        mime_type="text/markdown",
        file_size=failed_path.stat().st_size,
        created_at=old_time,
    )
    unknown_file = UploadedFile(
        user_id=1,
        filename="unknown.md",
        storage_path=str(unknown_path),
        mime_type="text/markdown",
        file_size=unknown_path.stat().st_size,
        created_at=old_time,
    )
    success_file = UploadedFile(
        user_id=1,
        filename="success.md",
        storage_path=str(success_path),
        mime_type="text/markdown",
        file_size=success_path.stat().st_size,
        created_at=old_time,
    )
    db.add_all([failed_file, unknown_file, success_file])
    db.commit()
    db.refresh(failed_file)
    db.refresh(unknown_file)
    db.refresh(success_file)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-failed",
                "file_id": failed_file.file_id,
                "source_path": str(failed_path),
                "user_id": 1,
            },
            {
                "collection": "kb",
                "doc_id": "doc-success",
                "file_id": success_file.file_id,
                "source_path": str(success_path),
                "user_id": 1,
            },
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-failed",
        status="failed",
        message="failed",
        parse_hash="",
        user_id=1,
    )
    write_ingestion_status(
        "kb",
        "doc-success",
        status="success",
        message="ok",
        parse_hash="",
        user_id=1,
    )

    result = reconcile_uploaded_files(
        db,
        user_id=1,
        is_admin=False,
        stale_ttl_hours=24,
        delete_stale=True,
    )

    assert result["stale_candidates"] == 2
    assert result["deleted"] == 2
    assert result["cleanup_errors"] == 0
    assert not failed_path.exists()
    assert not unknown_path.exists()
    assert success_path.exists()

    remaining = db.query(UploadedFile).all()
    assert len(remaining) == 1
    assert remaining[0].file_id == success_file.file_id

    refreshed_docs_table = get_connection_from_env().open_table("documents")
    rows = refreshed_docs_table.search().where("collection == 'kb'").to_list()
    row_by_doc_id = {str(row.get("doc_id")): row for row in rows}
    assert "doc-failed" not in row_by_doc_id
    assert "doc-success" in row_by_doc_id
    db.close()


def test_reconcile_uploaded_files_records_cleanup_error_when_documents_delete_fails(
    reconcile_env, monkeypatch: pytest.MonkeyPatch
):
    docs_table, session_local, uploads_dir = reconcile_env
    _create_user(session_local, user_id=1)
    db = session_local()

    failed_path = uploads_dir / "user_1" / "kb" / "failed-delete.md"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("failed", encoding="utf-8")

    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    failed_file = UploadedFile(
        user_id=1,
        filename="failed-delete.md",
        storage_path=str(failed_path),
        mime_type="text/markdown",
        file_size=failed_path.stat().st_size,
        created_at=old_time,
    )
    db.add(failed_file)
    db.commit()
    db.refresh(failed_file)

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-delete-failed",
                "file_id": failed_file.file_id,
                "source_path": str(failed_path),
                "user_id": 1,
            }
        ]
    )
    write_ingestion_status(
        "kb",
        "doc-delete-failed",
        status="failed",
        message="failed",
        parse_hash="",
        user_id=1,
    )

    real_conn = get_connection_from_env()
    real_table = real_conn.open_table("documents")

    class _DeleteFailingTable:
        def delete(self, where: str) -> None:
            if failed_file.file_id in where:
                raise RuntimeError("delete failed")
            real_table.delete(where)

        def __getattr__(self, item: str):
            return getattr(real_table, item)

    class _DeleteFailingConn:
        def open_table(self, name: str):
            if name == "documents":
                return _DeleteFailingTable()
            return real_conn.open_table(name)

    # Mock kb_file_service's connection (used for querying documents)
    monkeypatch.setattr(
        "xagent.web.services.kb_file_service.get_connection_from_env",
        lambda: _DeleteFailingConn(),
    )
    monkeypatch.setattr(
        "xagent.web.services.kb_file_service.ensure_documents_table",
        lambda _conn: None,
    )
    # Mock cascade_cleaner's connection (used by cascade_delete)
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.version_management.cascade_cleaner.get_vector_store_raw_connection",
        lambda: _DeleteFailingConn(),
    )

    result = reconcile_uploaded_files(
        db,
        user_id=1,
        is_admin=False,
        stale_ttl_hours=24,
        delete_stale=True,
    )

    assert result["cleanup_errors"] == 1
    assert result["deleted"] == 0
    assert failed_path.exists()
    still_exists = (
        db.query(UploadedFile)
        .filter(UploadedFile.file_id == failed_file.file_id)
        .first()
    )
    assert still_exists is not None
    db.close()
