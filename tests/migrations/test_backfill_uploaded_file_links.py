"""Tests for documents.file_id backfill migration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import lancedb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.migrations.lancedb import backfill_uploaded_file_links
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


@pytest.fixture
def migration_env(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        conn = lancedb.connect(temp_dir)
        ensure_documents_table(conn)
        docs_table = conn.open_table("documents")

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)

        monkeypatch.setattr(
            backfill_uploaded_file_links,
            "get_session_local",
            lambda: SessionLocal,
        )
        yield conn, docs_table, SessionLocal, Path(temp_dir)


def _create_user(db: Session, user_id: int = 1) -> None:
    user = User(
        id=user_id,
        username=f"user_{user_id}",
        password_hash="hash",
        is_admin=False,
    )
    db.add(user)
    db.commit()


def test_backfill_documents_file_links_matches_existing_uploaded_file(migration_env):
    conn, docs_table, SessionLocal, base_dir = migration_env
    source_file = base_dir / "user_1" / "kb" / "page.md"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("content", encoding="utf-8")

    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-1",
                "file_id": None,
                "source_path": str(source_file),
                "user_id": 1,
            }
        ]
    )

    db = SessionLocal()
    _create_user(db, user_id=1)
    uploaded = UploadedFile(
        user_id=1,
        filename=source_file.name,
        storage_path=str(source_file),
        mime_type="text/markdown",
        file_size=source_file.stat().st_size,
    )
    db.add(uploaded)
    db.commit()
    db.close()

    result = backfill_uploaded_file_links.backfill_documents_file_links(
        dry_run=False,
        conn=conn,
    )

    assert result["backfilled_by_match"] == 1
    assert result["failures"] == 0


def test_backfill_documents_file_links_creates_uploaded_file_when_missing(
    migration_env,
):
    conn, docs_table, SessionLocal, base_dir = migration_env
    source_file = base_dir / "user_1" / "kb" / "new_page.md"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("new-content", encoding="utf-8")
    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-2",
                "file_id": None,
                "source_path": str(source_file),
                "user_id": 1,
            }
        ]
    )

    db = SessionLocal()
    _create_user(db, user_id=1)
    db.close()

    result = backfill_uploaded_file_links.backfill_documents_file_links(
        dry_run=False,
        conn=conn,
    )

    assert result["backfilled_by_create"] == 1
    assert result["failures"] == 0

    db = SessionLocal()
    record = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == str(source_file))
        .first()
    )
    assert record is not None
    db.close()


def test_backfill_documents_file_links_dry_run_keeps_documents_unchanged(migration_env):
    conn, docs_table, SessionLocal, base_dir = migration_env
    source_file = base_dir / "user_1" / "kb" / "dry_run.md"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("dry-run", encoding="utf-8")
    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-3",
                "file_id": None,
                "source_path": str(source_file),
                "user_id": 1,
            }
        ]
    )

    db = SessionLocal()
    _create_user(db, user_id=1)
    db.close()

    result = backfill_uploaded_file_links.backfill_documents_file_links(
        dry_run=True,
        conn=conn,
    )

    assert result["backfilled_by_create"] == 1
    assert result["failures"] == 0


def test_backfill_documents_file_links_marks_unbackfillable(migration_env):
    conn, docs_table, SessionLocal, _ = migration_env
    docs_table.add(
        [
            {
                "collection": "kb",
                "doc_id": "doc-4",
                "file_id": None,
                "source_path": "/tmp/not-exists-file.md",
                "user_id": 1,
            }
        ]
    )
    db = SessionLocal()
    _create_user(db, user_id=1)
    db.close()

    result = backfill_uploaded_file_links.backfill_documents_file_links(
        dry_run=False,
        conn=conn,
    )

    assert result["unbackfillable"] == 1
    assert result["unbackfillable_samples"][0]["reason"] == "missing_file_on_disk"


def test_backfill_all_releases_locks_when_backfill_raises(
    migration_env, monkeypatch: pytest.MonkeyPatch
):
    conn, _, _, _ = migration_env

    class _DummyLockFile:
        def fileno(self) -> int:
            return 0

        def close(self) -> None:
            return None

    def _fake_backfill(*, dry_run: bool = False, batch_size: int = 500, conn=None):
        raise RuntimeError("boom")

    lock_file = _DummyLockFile()
    release_calls: list[_DummyLockFile] = []

    monkeypatch.setattr(
        backfill_uploaded_file_links, "_acquire_file_lock", lambda: lock_file
    )
    monkeypatch.setattr(
        backfill_uploaded_file_links,
        "_release_file_lock",
        lambda file_obj: release_calls.append(file_obj),
    )
    monkeypatch.setattr(
        backfill_uploaded_file_links, "backfill_documents_file_links", _fake_backfill
    )

    with pytest.raises(RuntimeError, match="boom"):
        backfill_uploaded_file_links.backfill_all(dry_run=False, batch_size=10)

    assert release_calls == [lock_file]
    assert backfill_uploaded_file_links._migration_lock.acquire(blocking=False)
    backfill_uploaded_file_links._migration_lock.release()
