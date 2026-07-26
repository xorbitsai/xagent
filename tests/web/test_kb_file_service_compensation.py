"""Tests for KB file compensation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from xagent.core.file_storage import get_unscoped_file_storage
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import kb_file_service
from xagent.web.services.kb_file_service import (
    capture_uploaded_file_refresh_snapshot,
    compensate_new_uploaded_file,
    restore_uploaded_file_refresh_snapshot,
    upsert_uploaded_file_record,
)


@pytest.fixture
def compensation_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    uploads_dir = tmp_path / "uploads"
    objects_dir = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", objects_dir.as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    get_unscoped_file_storage.cache_clear()

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    db.add(User(id=1, username="user_1", password_hash="hash", is_admin=False))
    db.commit()
    try:
        yield db, uploads_dir
    finally:
        db.close()
        get_unscoped_file_storage.cache_clear()


def test_compensate_new_uploaded_file_removes_row_local_and_durable(
    compensation_env,
) -> None:
    db, uploads_dir = compensation_env
    file_path = uploads_dir / "user_1" / "kb" / "page.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("content", encoding="utf-8")

    file_record = upsert_uploaded_file_record(
        db,
        user_id=1,
        filename="page.md",
        storage_path=file_path,
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )
    file_id = str(file_record.file_id)
    storage_key = str(file_record.storage_key)
    assert get_unscoped_file_storage().exists(storage_key)

    result = compensate_new_uploaded_file(db, file_id=file_id, user_id=1)
    db.commit()

    assert result.complete
    assert (
        db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first() is None
    )
    assert not file_path.exists()
    assert not get_unscoped_file_storage().exists(storage_key)

    second = compensate_new_uploaded_file(db, file_id=file_id, user_id=1)
    assert second.complete


def test_compensate_new_uploaded_file_reports_incomplete_on_cleanup_failure(
    compensation_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, uploads_dir = compensation_env
    file_path = uploads_dir / "user_1" / "kb" / "page.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("content", encoding="utf-8")
    file_record = upsert_uploaded_file_record(
        db,
        user_id=1,
        filename="page.md",
        storage_path=file_path,
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )

    def fail_delete(*_args, **_kwargs) -> None:
        raise RuntimeError("durable delete failed")

    monkeypatch.setattr(kb_file_service.UploadedFileStore, "delete", fail_delete)

    result = compensate_new_uploaded_file(
        db,
        file_id=str(file_record.file_id),
        user_id=1,
    )

    assert result.status == "incomplete"
    assert result.side_effects_may_remain is True
    assert "durable delete failed" in result.errors[0]


def test_restore_uploaded_file_refresh_snapshot_restores_row_local_and_durable(
    compensation_env,
) -> None:
    db, uploads_dir = compensation_env
    file_path = uploads_dir / "user_1" / "kb" / "page.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("old content", encoding="utf-8")
    file_record = upsert_uploaded_file_record(
        db,
        user_id=1,
        filename="page.md",
        storage_path=file_path,
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )
    storage_key = str(file_record.storage_key)

    backup_path = uploads_dir / "page.md.backup"
    backup_path.write_text("old content", encoding="utf-8")
    snapshot = capture_uploaded_file_refresh_snapshot(
        file_record,
        backup_path=backup_path,
        reindex_marker_applied=True,
    )

    file_path.write_text("new content", encoding="utf-8")
    file_record.filename = "renamed.md"
    file_record.file_size = file_path.stat().st_size
    db.flush()

    result = restore_uploaded_file_refresh_snapshot(db, snapshot)
    db.commit()

    assert result.complete
    assert file_path.read_text(encoding="utf-8") == "old content"
    assert file_record.filename == "page.md"
    with get_unscoped_file_storage().open_read(storage_key) as handle:
        assert handle.read() == b"old content"


def test_restore_uploaded_file_refresh_snapshot_rolls_back_failed_session(
    compensation_env,
) -> None:
    db, uploads_dir = compensation_env
    file_path = uploads_dir / "user_1" / "kb" / "page.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("old content", encoding="utf-8")
    file_record = upsert_uploaded_file_record(
        db,
        user_id=1,
        filename="page.md",
        storage_path=file_path,
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )

    backup_path = uploads_dir / "page.md.backup"
    backup_path.write_text("old content", encoding="utf-8")
    snapshot = capture_uploaded_file_refresh_snapshot(
        file_record,
        backup_path=backup_path,
    )

    file_path.write_text("new content", encoding="utf-8")
    file_record.filename = "renamed.md"
    db.flush()
    db.add(User(username="user_1", password_hash="dupe", is_admin=False))
    with pytest.raises(IntegrityError):
        db.commit()

    result = restore_uploaded_file_refresh_snapshot(db, snapshot)
    db.commit()

    assert result.complete
    assert file_path.read_text(encoding="utf-8") == "old content"
    restored_record = (
        db.query(UploadedFile).filter(UploadedFile.file_id == file_record.file_id).one()
    )
    assert restored_record.filename == "page.md"


def test_restore_uploaded_file_refresh_snapshot_never_revives_compensating_row(
    compensation_env,
) -> None:
    db, uploads_dir = compensation_env
    file_path = uploads_dir / "user_1" / "kb" / "page.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("old content", encoding="utf-8")
    file_record = upsert_uploaded_file_record(
        db,
        user_id=1,
        filename="page.md",
        storage_path=file_path,
        mime_type="text/markdown",
        file_size=file_path.stat().st_size,
    )
    backup_path = uploads_dir / "page.md.backup"
    backup_path.write_text("old content", encoding="utf-8")
    snapshot = capture_uploaded_file_refresh_snapshot(
        file_record,
        backup_path=backup_path,
    )

    file_path.write_text("new content", encoding="utf-8")
    file_record.storage_status = "compensating"
    db.commit()

    result = restore_uploaded_file_refresh_snapshot(db, snapshot)

    db.expire_all()
    persisted = (
        db.query(UploadedFile).filter(UploadedFile.file_id == file_record.file_id).one()
    )
    assert result.side_effects_may_remain
    assert file_path.read_text(encoding="utf-8") == "old content"
    assert persisted.storage_status == "compensating"


def test_restore_uploaded_file_refresh_snapshot_restores_local_before_db_query(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "page.md"
    file_path.write_text("new content", encoding="utf-8")
    backup_path = tmp_path / "page.md.backup"
    backup_path.write_text("old content", encoding="utf-8")
    snapshot = kb_file_service.UploadedFileRefreshSnapshot(
        file_id="file-1",
        user_id=1,
        row_fields={},
        previous_path=file_path,
        backup_path=backup_path,
        had_local_file=True,
    )

    class BrokenSession:
        def rollback(self) -> None:
            pass

        def query(self, *_args):
            raise RuntimeError("database unavailable")

    result = restore_uploaded_file_refresh_snapshot(BrokenSession(), snapshot)  # type: ignore[arg-type]

    assert result.side_effects_may_remain
    assert "local_file_restored" in result.effects
    assert file_path.read_text(encoding="utf-8") == "old content"
