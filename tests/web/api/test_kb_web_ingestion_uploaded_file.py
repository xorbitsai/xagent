"""Tests for web-ingestion uploaded-file persistence helpers.

This module focuses on storage-level behavior around
``xagent.web.api.kb._upsert_uploaded_file_record`` and related URL-hash
dedup semantics used by web ingestion:
- UploadedFile record creation/update
- URL-hash based dedup behavior
- Cross-collection isolation at persistence level
- Failure cleanup expectations

Additionally, this module contains API-level tests that exercise the
web-ingestion route's internal file handler (the nested ``_handle_web_file``)
via a stubbed ``run_web_ingestion`` implementation. This ensures the
end-to-end handler behavior stays covered even though ``_handle_web_file``
is not importable directly (nested function).
"""

import hashlib
import io
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.file_storage.factory import get_file_storage
from xagent.web.api.kb import (
    _WEB_FILE_LOCKS,
    _atomic_replace_file,
    _build_ingest_backup_path,
    _compensate_new_web_ingest_files,
    _copy_upload_file_to_path,
    _create_web_uploaded_file_record,
    _delete_web_rag_side_effects_for_file_id,
    _get_file_sha256,
    _mark_uploaded_file_for_reindex,
    _normalize_web_title_for_filename,
    _RagDocumentSnapshot,
    _recreate_missing_existing_file,
    _refresh_existing_file_if_changed,
    _restore_rag_snapshot_rows,
    _rollback_failed_web_document_ingestion,
    _upsert_uploaded_file_record,
    _WebFileLock,
    kb_router,
)
from xagent.web.models.database import Base, get_db
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


class TestNormalizeWebTitleForFilename:
    """Unit tests for web-title filename normalization."""

    def test_truncates_multibyte_titles_to_safe_filename_budget(self) -> None:
        title = "你" * 120

        normalized = _normalize_web_title_for_filename(title)
        filename = f"{'0' * 16}_{normalized}.md"

        assert normalized != "untitled"
        assert len(normalized.encode("utf-8")) <= 235
        assert len(filename.encode("utf-8")) <= 255


class TestIngestFileHelpers:
    def test_build_ingest_backup_path_bounds_long_filenames(
        self, tmp_path: Path
    ) -> None:
        source_path = tmp_path / f"{'你' * 100}.txt"

        backup_path = _build_ingest_backup_path(source_path)

        assert backup_path.parent == source_path.parent
        backup_name, rollback_suffix = backup_path.name.rsplit(".rollback-", 1)
        assert backup_name
        assert len(rollback_suffix) == 32
        assert len(backup_path.name.encode("utf-8")) <= 255

    def test_copy_upload_file_to_path_enforces_size_limit(self, tmp_path: Path) -> None:
        upload = UploadFile(file=io.BytesIO(b"abcdef"), filename="sample.txt")
        target_path = tmp_path / "sample.txt"

        with pytest.raises(HTTPException) as exc_info:
            _copy_upload_file_to_path(upload, target_path, max_size=3)

        assert exc_info.value.status_code == 413

    def test_copy_upload_file_to_path_returns_written_size(
        self, tmp_path: Path
    ) -> None:
        upload = UploadFile(file=io.BytesIO(b"abcdef"), filename="sample.txt")
        target_path = tmp_path / "sample.txt"

        result = _copy_upload_file_to_path(upload, target_path, max_size=10)

        assert result.total_size == 6
        assert result.sha256 == hashlib.sha256(b"abcdef").hexdigest()
        assert target_path.read_bytes() == b"abcdef"


@pytest.fixture
def db_session():
    """Create a test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session: Session):
    """Create a test user."""
    user = User(
        username="test_user",
        password_hash="hash",
        is_admin=False,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def mock_user():
    """Create a mock user object (simulates FastAPI Depends(get_user))."""
    mock = MagicMock()
    mock.id = 1
    mock.username = "test_user"
    mock.is_admin = False
    return mock


@pytest.fixture(scope="function")
def web_test_env(tmp_path: Path):
    """Create an app+DB env for ingest-web route tests."""
    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    TestingSessionLocal = sessionmaker(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_db] = override_get_db

    Base.metadata.create_all(bind=test_engine)

    session = TestingSessionLocal()
    user = User(
        username="testuser",
        password_hash="hash",
        is_admin=False,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # Mock JWT token (must include type="access" for get_current_user)
    from datetime import datetime, timedelta, timezone

    import jwt

    from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

    payload = {
        "sub": user.username,
        "user_id": user.id,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}

    return app, headers, user, TestingSessionLocal


class TestWebIngestionUploadedFilePersistence:
    """Test uploaded-file persistence behavior used by web ingestion."""

    def test_new_file_creation(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """Test creating a new file when no cache or DB record exists."""
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        get_file_storage.cache_clear()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary markdown file
            temp_file = Path(temp_dir) / "temp.md"
            temp_file.write_text("# Test Page\n\nContent here")

            # Mock get_upload_path to return a path in temp_dir
            persistent_path = Path(temp_dir) / "uploads" / "user_1" / "test_collection"
            persistent_path.mkdir(parents=True, exist_ok=True)

            with patch("xagent.web.api.kb.get_upload_path") as mock_get_path:
                mock_get_path.return_value = persistent_path / "file.md"

                # Simulate ingest-web persistence context
                _processed_urls = {}

                # Mock sanitize_path_component
                with patch(
                    "xagent.web.api.kb.sanitize_path_component"
                ) as mock_sanitize:
                    mock_sanitize.return_value = "test_page"

                    # Simulate storage-layer steps used by the ingest-web handler
                    url = "https://example.com/test"
                    collection = "test_collection"

                    # Simulate URL hash generation
                    import hashlib

                    url_hash = hashlib.sha256(
                        f"{collection}:{url}".encode()
                    ).hexdigest()[:16]
                    filename = f"{url_hash}_test_page.md"

                    # Generate persistent file path
                    persistent_file = mock_get_path(
                        filename,
                        user_id=int(mock_user.id),
                        collection=collection,
                        collection_is_sanitized=True,
                    )

                    # Copy file
                    import shutil

                    shutil.copy2(temp_file, persistent_file)

                    # Create DB record
                    file_record = _upsert_uploaded_file_record(
                        db_session,
                        user_id=int(mock_user.id),
                        filename=filename,
                        storage_path=persistent_file,
                        mime_type="text/markdown",
                        file_size=persistent_file.stat().st_size,
                    )

                    # Verify
                    assert file_record.file_id is not None
                    assert file_record.filename == filename
                    assert file_record.storage_path == str(persistent_file)
                    assert file_record.storage_status == "available"
                    assert file_record.storage_key
                    assert file_record.storage_uri
                    assert persistent_file.exists()
                    with get_file_storage().open_read(
                        str(file_record.storage_key)
                    ) as handle:
                        assert handle.read() == persistent_file.read_bytes()

                    # Verify DB record exists
                    db_record = (
                        db_session.query(UploadedFile)
                        .filter(UploadedFile.file_id == file_record.file_id)
                        .first()
                    )
                    assert db_record is not None
                    assert db_record.filename == filename

    def test_cross_collection_isolation(
        self, db_session: Session, test_user: User, mock_user: MagicMock
    ):
        """Test that same URL in different collections creates separate files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "temp.md"
            temp_file.write_text("# Test\n\nContent")

            persistent_path = Path(temp_dir) / "uploads" / "user_1"
            persistent_path.mkdir(parents=True, exist_ok=True)

            with patch("xagent.web.api.kb.get_upload_path") as mock_get_path:
                # Simulate URL hash generation for different collections
                url = "https://example.com/test"
                collection1 = "collection1"
                collection2 = "collection2"

                import hashlib

                hash1 = hashlib.sha256(f"{collection1}:{url}".encode()).hexdigest()[:16]
                hash2 = hashlib.sha256(f"{collection2}:{url}".encode()).hexdigest()[:16]

                # Verify hashes are different
                assert hash1 != hash2

                # Create files for each collection
                file1_path = persistent_path / collection1 / f"{hash1}_test.md"
                file2_path = persistent_path / collection2 / f"{hash2}_test.md"

                mock_get_path.side_effect = [file1_path, file2_path]

                # Create first record
                file1_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(temp_file, file1_path)
                record1 = _upsert_uploaded_file_record(
                    db_session,
                    user_id=int(mock_user.id),
                    filename=f"{hash1}_test.md",
                    storage_path=file1_path,
                    mime_type="text/markdown",
                    file_size=file1_path.stat().st_size,
                )

                # Create second record
                file2_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(temp_file, file2_path)
                record2 = _upsert_uploaded_file_record(
                    db_session,
                    user_id=int(mock_user.id),
                    filename=f"{hash2}_test.md",
                    storage_path=file2_path,
                    mime_type="text/markdown",
                    file_size=file2_path.stat().st_size,
                )

                # Verify both records exist with different file_ids and filenames
                assert record1.file_id != record2.file_id
                assert record1.filename != record2.filename
                assert hash1 in record1.filename
                assert hash2 in record2.filename

    def test_database_deduplication_reuses_existing_file(
        self, db_session: Session, test_user: User, mock_user: MagicMock
    ):
        """Test that existing DB record is reused for same URL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistent_path = Path(temp_dir) / "uploads" / "user_1" / "test_collection"
            persistent_path.mkdir(parents=True, exist_ok=True)

            with patch("xagent.web.api.kb.get_upload_path") as mock_get_path:
                url = "https://example.com/test"
                collection = "test_collection"

                import hashlib

                url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[
                    :16
                ]
                filename = f"{url_hash}_test.md"

                # Create existing file and record
                existing_path = persistent_path / filename
                existing_path.write_text("# Test\n\nContent")
                mock_get_path.return_value = existing_path

                existing_record = _upsert_uploaded_file_record(
                    db_session,
                    user_id=int(mock_user.id),
                    filename=filename,
                    storage_path=existing_path,
                    mime_type="text/markdown",
                    file_size=existing_path.stat().st_size,
                )

                # Query for existing record
                found_record = (
                    db_session.query(UploadedFile)
                    .filter(
                        UploadedFile.user_id == int(mock_user.id),
                        UploadedFile.filename == filename,
                    )
                    .first()
                )

                # Verify existing record is found
                assert found_record is not None
                assert found_record.file_id == existing_record.file_id
                assert found_record.filename == filename

    def test_upsert_failure_cleans_up_persistent_file(
        self, db_session: Session, test_user: User, mock_user: MagicMock
    ):
        """Test that persistent file is cleaned up if upsert fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "temp.md"
            temp_file.write_text("# Test\n\nContent")

            persistent_path = Path(temp_dir) / "uploads" / "user_1" / "test_collection"
            persistent_path.mkdir(parents=True, exist_ok=True)
            persistent_file = persistent_path / "test.md"

            with patch("xagent.web.api.kb.get_upload_path") as mock_get_path:
                mock_get_path.return_value = persistent_file

                # Copy file
                import shutil

                shutil.copy2(temp_file, persistent_file)
                assert persistent_file.exists()

                # Simulate upsert failure by closing the db session
                # This will cause any db operation to fail
                db_session.close()

                try:
                    # Try to create record with closed session
                    _upsert_uploaded_file_record(
                        db_session,
                        user_id=int(mock_user.id),
                        filename="test.md",
                        storage_path=persistent_file,
                        mime_type="text/markdown",
                        file_size=persistent_file.stat().st_size,
                    )
                except Exception:
                    # Expected to fail due to closed session
                    pass
                finally:
                    # Manual cleanup (simulating the except block in _handle_web_file)
                    if persistent_file.exists():
                        persistent_file.unlink()
                    # Verify cleanup happened
                    assert not persistent_file.exists()

    def test_refresh_existing_file_restores_local_file_when_durable_sync_fails(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        monkeypatch.setenv(
            "XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized")
        )
        get_file_storage.cache_clear()

        temp_dir = tmp_path / "ingest"
        temp_dir.mkdir()
        existing_path = temp_dir / "existing.md"
        existing_path.write_text("old content", encoding="utf-8")
        temp_file_path = temp_dir / "incoming.md"
        temp_file_path.write_text("new content", encoding="utf-8")

        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=int(mock_user.id),
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )

        def failing_upsert(*_args, **_kwargs):
            from xagent.web.services.uploaded_file_store import UploadedFileStore

            UploadedFileStore(db_session).sync_existing(
                existing_record,
                storage_key=str(existing_record.storage_key),
                mime_type="text/markdown",
            )
            db_session.commit()
            raise RuntimeError("durable sync failed")

        ingestion_runs_snapshot = object()
        with (
            patch(
                "xagent.web.api.kb._mark_uploaded_file_for_reindex", return_value=True
            ),
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=ingestion_runs_snapshot,
            ) as mock_snapshot_runs,
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._restore_ingestion_runs_snapshot"
            ) as mock_restore_runs,
            patch(
                "xagent.web.api.kb._atomic_replace_file",
                wraps=_atomic_replace_file,
            ) as atomic_replace,
            patch(
                "xagent.web.api.kb._upsert_uploaded_file_record",
                side_effect=failing_upsert,
            ),
        ):
            with pytest.raises(RuntimeError, match="durable sync failed"):
                _refresh_existing_file_if_changed(
                    existing_record=existing_record,
                    temp_file_path=temp_file_path,
                    db_session=db_session,
                    user_id=int(mock_user.id),
                    is_admin=False,
                    collection_name="test_collection",
                    url="https://example.com/page",
                    filename="existing.md",
                    url_hash="hash",
                    processed_urls={},
                    context="cross-session",
                )

        assert atomic_replace.called
        assert existing_path.read_text(encoding="utf-8") == "old content"
        with get_file_storage().open_read(str(existing_record.storage_key)) as handle:
            assert handle.read() == b"old content"
        mock_snapshot_runs.assert_called_once_with(str(existing_record.file_id))
        mock_restore_runs.assert_called_once_with(ingestion_runs_snapshot)

    def test_refresh_existing_file_restores_ingestion_runs_when_backup_fails(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
    ) -> None:
        existing_path = tmp_path / "existing.md"
        existing_path.write_text("old content", encoding="utf-8")
        temp_file_path = tmp_path / "incoming.md"
        temp_file_path.write_text("new content", encoding="utf-8")

        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=int(mock_user.id),
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )

        ingestion_runs_snapshot = object()
        with (
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=ingestion_runs_snapshot,
            ),
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._mark_uploaded_file_for_reindex", return_value=True
            ),
            patch("xagent.web.api.kb.shutil.copy2", side_effect=OSError("disk full")),
            patch(
                "xagent.web.api.kb._restore_ingestion_runs_snapshot"
            ) as mock_restore_runs,
        ):
            with pytest.raises(OSError, match="disk full"):
                _refresh_existing_file_if_changed(
                    existing_record=existing_record,
                    temp_file_path=temp_file_path,
                    db_session=db_session,
                    user_id=int(mock_user.id),
                    is_admin=False,
                    collection_name="test_collection",
                    url="https://example.com/page",
                    filename="existing.md",
                    url_hash="hash",
                    processed_urls={},
                    context="cross-session",
                )

        assert existing_path.read_text(encoding="utf-8") == "old content"
        mock_restore_runs.assert_called_once_with(ingestion_runs_snapshot)

    def test_create_web_uploaded_file_record_cleans_durable_when_commit_fails(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        get_file_storage.cache_clear()

        persistent_file = tmp_path / "uploads" / "web.md"
        persistent_file.parent.mkdir(parents=True)
        persistent_file.write_text("new content", encoding="utf-8")

        with patch.object(db_session, "commit", side_effect=RuntimeError("db down")):
            with pytest.raises(RuntimeError, match="db down"):
                _create_web_uploaded_file_record(
                    db_session,
                    user_id=int(mock_user.id),
                    filename="web.md",
                    storage_path=persistent_file,
                    mime_type="text/markdown",
                )

        rows = db_session.query(UploadedFile).all()
        assert rows == []
        assert get_file_storage().list("users") == []

    def test_refresh_existing_file_rolls_back_failed_upsert_before_restore(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        monkeypatch.setenv(
            "XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized")
        )
        get_file_storage.cache_clear()

        temp_dir = tmp_path / "ingest"
        temp_dir.mkdir()
        existing_path = temp_dir / "existing.md"
        existing_path.write_text("old content", encoding="utf-8")
        temp_file_path = temp_dir / "incoming.md"
        temp_file_path.write_text("new content", encoding="utf-8")

        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=int(mock_user.id),
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
        file_id = str(existing_record.file_id)

        def failing_upsert(db: Session, *_args, **_kwargs):
            db.add(
                User(
                    username=str(test_user.username),
                    password_hash="duplicate",
                    is_admin=False,
                )
            )
            db.commit()

        with patch(
            "xagent.web.api.kb._mark_uploaded_file_for_reindex", return_value=True
        ):
            with patch(
                "xagent.web.api.kb._upsert_uploaded_file_record",
                side_effect=failing_upsert,
            ):
                with pytest.raises(IntegrityError):
                    _refresh_existing_file_if_changed(
                        existing_record=existing_record,
                        temp_file_path=temp_file_path,
                        db_session=db_session,
                        user_id=int(mock_user.id),
                        is_admin=False,
                        collection_name="test_collection",
                        url="https://example.com/page",
                        filename="existing.md",
                        url_hash="hash",
                        processed_urls={},
                        context="cross-session",
                    )

        assert existing_path.read_text(encoding="utf-8") == "old content"
        restored_record = (
            db_session.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        )
        assert restored_record.filename == "existing.md"
        assert list(temp_dir.glob("existing.md.rollback-*")) == []

    def test_refresh_existing_file_restores_missing_local_from_durable_before_compare(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        monkeypatch.setenv(
            "XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized")
        )
        get_file_storage.cache_clear()

        existing_path = tmp_path / "uploads" / "existing.md"
        existing_path.parent.mkdir()
        existing_path.write_text("old content", encoding="utf-8")
        temp_file_path = tmp_path / "incoming.md"
        temp_file_path.write_text("old content", encoding="utf-8")

        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=int(mock_user.id),
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
        existing_path.unlink()

        processed_urls: dict[str, str] = {}
        with (
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ),
        ):
            result = _refresh_existing_file_if_changed(
                existing_record=existing_record,
                temp_file_path=temp_file_path,
                db_session=db_session,
                user_id=int(mock_user.id),
                is_admin=False,
                collection_name="test_collection",
                url="https://example.com/page",
                filename="existing.md",
                url_hash="hash",
                processed_urls=processed_urls,
                context="cross-session",
            )

        assert result is not None
        assert result["file_id"] == str(existing_record.file_id)
        assert callable(result["rollback_on_failure"])
        assert existing_path.read_text(encoding="utf-8") == "old content"
        assert processed_urls == {}

    def test_refresh_existing_file_refreshes_durable_restored_local_when_changed(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        monkeypatch.setenv(
            "XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized")
        )
        get_file_storage.cache_clear()

        existing_path = tmp_path / "uploads" / "existing.md"
        existing_path.parent.mkdir()
        existing_path.write_text("old content", encoding="utf-8")
        temp_file_path = tmp_path / "incoming.md"
        temp_file_path.write_text("new content", encoding="utf-8")

        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=int(mock_user.id),
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
        existing_path.unlink()

        processed_urls: dict[str, str] = {}
        with (
            patch(
                "xagent.web.api.kb._mark_uploaded_file_for_reindex", return_value=True
            ),
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ),
        ):
            result = _refresh_existing_file_if_changed(
                existing_record=existing_record,
                temp_file_path=temp_file_path,
                db_session=db_session,
                user_id=int(mock_user.id),
                is_admin=False,
                collection_name="test_collection",
                url="https://example.com/page",
                filename="existing.md",
                url_hash="hash",
                processed_urls=processed_urls,
                context="cross-session",
            )

        assert result is not None
        assert existing_path.read_text(encoding="utf-8") == "new content"
        assert processed_urls == {"hash": str(existing_record.file_id)}
        with get_file_storage().open_read(str(existing_record.storage_key)) as handle:
            assert handle.read() == b"new content"

    def test_recreate_missing_existing_file_removes_local_when_upsert_fails(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
    ) -> None:
        existing_path = tmp_path / "uploads" / "existing.md"
        temp_file_path = tmp_path / "incoming.md"
        temp_file_path.write_text("new content", encoding="utf-8")
        existing_record = UploadedFile(
            file_id=str(uuid4()),
            user_id=int(mock_user.id),
            filename="existing.md",
            storage_path=str(existing_path),
            mime_type="text/markdown",
            file_size=0,
        )

        def failing_upsert(*_args, **_kwargs):
            raise RuntimeError("durable sync failed")

        with (
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ),
            patch("xagent.web.api.kb._restore_ingestion_runs_snapshot"),
            patch(
                "xagent.web.api.kb._upsert_uploaded_file_record",
                side_effect=failing_upsert,
            ),
        ):
            with pytest.raises(RuntimeError, match="durable sync failed"):
                _recreate_missing_existing_file(
                    existing_record=existing_record,
                    temp_file_path=temp_file_path,
                    db_session=db_session,
                    user_id=int(mock_user.id),
                    is_admin=False,
                    collection_name="test_collection",
                    filename="existing.md",
                    url_hash="hash",
                    processed_urls={},
                )

        assert not existing_path.exists()

    def test_recreate_missing_existing_file_restores_after_post_commit_upsert_failure(
        self,
        db_session: Session,
        test_user: User,
        mock_user: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        get_file_storage.cache_clear()

        existing_path = tmp_path / "uploads" / "existing.md"
        existing_path.parent.mkdir(parents=True)
        existing_path.write_text("old content", encoding="utf-8")
        temp_file_path = tmp_path / "incoming.md"
        temp_file_path.write_text("new content", encoding="utf-8")

        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=int(mock_user.id),
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
        old_file_id = str(existing_record.file_id)
        old_storage_key = str(existing_record.storage_key)
        from xagent.web.services.managed_file_ref import ManagedFileRef
        from xagent.web.services.uploaded_file_store import UploadedFileStore

        ManagedFileRef(existing_record).delete_durable()
        existing_path.unlink()

        def failing_upsert(
            db_session_arg: Session,
            *,
            user_id: int,
            filename: str,
            storage_path: Path,
            mime_type: str,
            file_size: int,
        ) -> UploadedFile:
            refreshed_record = (
                db_session_arg.query(UploadedFile)
                .filter(UploadedFile.file_id == old_file_id)
                .first()
            )
            assert refreshed_record is not None
            refreshed_record.filename = filename
            refreshed_record.file_size = file_size
            UploadedFileStore(db_session_arg).sync_existing(
                refreshed_record,
                storage_key=old_storage_key,
                mime_type=mime_type,
            )
            db_session_arg.commit()
            raise RuntimeError("post-commit failure")

        ingestion_runs_snapshot = object()
        with (
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=ingestion_runs_snapshot,
            ),
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._restore_ingestion_runs_snapshot"
            ) as mock_restore_runs,
            patch(
                "xagent.web.api.kb._upsert_uploaded_file_record",
                side_effect=failing_upsert,
            ),
        ):
            with pytest.raises(RuntimeError, match="post-commit failure"):
                _recreate_missing_existing_file(
                    existing_record=existing_record,
                    temp_file_path=temp_file_path,
                    db_session=db_session,
                    user_id=int(mock_user.id),
                    is_admin=False,
                    collection_name="test_collection",
                    filename="existing.md",
                    url_hash="hash",
                    processed_urls={},
                )

        assert not existing_path.exists()
        assert get_file_storage().list("users") == []
        row = (
            db_session.query(UploadedFile)
            .filter(UploadedFile.file_id == old_file_id)
            .first()
        )
        assert row is not None
        assert row.storage_key == old_storage_key
        mock_restore_runs.assert_called_once_with(ingestion_runs_snapshot)

    def test_in_memory_cache_deduplication(
        self, db_session: Session, test_user: User, mock_user: MagicMock
    ):
        """Test that in-memory cache prevents duplicate DB queries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistent_path = Path(temp_dir) / "uploads" / "user_1" / "test_collection"
            persistent_path.mkdir(parents=True, exist_ok=True)

            with patch("xagent.web.api.kb.get_upload_path") as mock_get_path:
                url = "https://example.com/test"
                collection = "test_collection"

                import hashlib

                url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[
                    :16
                ]
                filename = f"{url_hash}_test.md"

                # Create file and record
                file_path = persistent_path / filename
                file_path.write_text("# Test\n\nContent")
                mock_get_path.return_value = file_path

                record = _upsert_uploaded_file_record(
                    db_session,
                    user_id=int(mock_user.id),
                    filename=filename,
                    storage_path=file_path,
                    mime_type="text/markdown",
                    file_size=file_path.stat().st_size,
                )

                # Simulate in-memory cache
                _processed_urls = {url_hash: str(record.file_id)}

                # Check cache hit
                assert url_hash in _processed_urls
                cached_file_id = _processed_urls[url_hash]

                # Query DB with cached file_id
                found_record = (
                    db_session.query(UploadedFile)
                    .filter(UploadedFile.file_id == cached_file_id)
                    .first()
                )

                # Verify cache hit returns correct record
                assert found_record is not None
                assert found_record.file_id == record.file_id

    def test_cache_hit_with_deleted_db_record_falls_through(
        self, db_session: Session, test_user: User, mock_user: MagicMock
    ):
        """Test that cache hit with deleted DB record falls through to recreate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistent_path = Path(temp_dir) / "uploads" / "user_1" / "test_collection"
            persistent_path.mkdir(parents=True, exist_ok=True)

            with patch("xagent.web.api.kb.get_upload_path") as mock_get_path:
                url = "https://example.com/test"
                collection = "test_collection"

                import hashlib

                url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[
                    :16
                ]
                filename = f"{url_hash}_test.md"

                # Simulate cache with non-existent file_id
                nonexistent_file_id = str(uuid4())
                _processed_urls = {url_hash: nonexistent_file_id}

                # Query DB with cached file_id
                found_record = (
                    db_session.query(UploadedFile)
                    .filter(UploadedFile.file_id == nonexistent_file_id)
                    .first()
                )

                # Verify record not found (cache miss due to deletion)
                assert found_record is None

                # Should fall through to create new record
                file_path = persistent_path / filename
                file_path.write_text("# Test\n\nContent")
                mock_get_path.return_value = file_path

                new_record = _upsert_uploaded_file_record(
                    db_session,
                    user_id=int(mock_user.id),
                    filename=filename,
                    storage_path=file_path,
                    mime_type="text/markdown",
                    file_size=file_path.stat().st_size,
                )

                # Verify new record was created
                assert new_record.file_id != nonexistent_file_id
                assert new_record.filename == filename


class TestHandleWebFileUserIsolation:
    """Test user isolation in KB file operations."""

    def test_database_dedup_isolated_by_user_id(
        self, db_session: Session, test_user: User, mock_user: MagicMock
    ):
        """Same filename from different users should not reuse UploadedFile rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user1_path = Path(temp_dir) / "uploads" / "user_1" / "c1"
            user2_path = Path(temp_dir) / "uploads" / "user_2" / "c1"
            user1_path.mkdir(parents=True, exist_ok=True)
            user2_path.mkdir(parents=True, exist_ok=True)

            same_filename = "same_hash_page.md"
            path1 = user1_path / same_filename
            path2 = user2_path / same_filename
            path1.write_text("user1")
            path2.write_text("user2")

            record1 = _upsert_uploaded_file_record(
                db_session,
                user_id=1,
                filename=same_filename,
                storage_path=path1,
                mime_type="text/markdown",
                file_size=path1.stat().st_size,
            )
            record2 = _upsert_uploaded_file_record(
                db_session,
                user_id=2,
                filename=same_filename,
                storage_path=path2,
                mime_type="text/markdown",
                file_size=path2.stat().st_size,
            )

            assert record1.file_id != record2.file_id


class TestIngestWebHandleWebFile:
    """API-level coverage for ingest-web file handling and dedup semantics."""

    def test_ingest_web_dedups_same_url_within_request(
        self, web_test_env, tmp_path: Path
    ) -> None:
        """Same URL processed twice in one request should create only one UploadedFile row."""
        app, headers, user, TestingSessionLocal = web_test_env
        client = TestClient(app)

        uploads_root = tmp_path / "uploads"

        def patched_get_upload_path(
            filename: str,
            *,
            user_id: int,
            collection: str,
            collection_is_sanitized: bool,
        ) -> Path:
            assert collection_is_sanitized is True
            return uploads_root / f"user_{user_id}" / collection / filename

        # Stub out the web ingestion runner, but exercise the provided file_handler.
        async def stub_run_web_ingestion(
            *,
            collection: str,
            crawl_config,
            ingestion_config,
            user_id: int,
            is_admin: bool,
            file_handler,
        ):
            temp_md = tmp_path / "temp.md"
            temp_md.write_text("# Title\n\nBody", encoding="utf-8")
            url = "https://example.com/page"
            title = "How to edit a completed job?"

            # Call file handler twice with the same URL; second call must dedup.
            r1 = file_handler(temp_md, title, collection, url)
            r2 = file_handler(temp_md, title, collection, url)
            assert r1["file_id"] == r2["file_id"]
            assert r1["file_path"] == r2["file_path"]

            from xagent.core.tools.core.RAG_tools.core.schemas import WebIngestionResult

            return WebIngestionResult(
                status="success",
                collection=collection,
                total_urls_found=1,
                pages_crawled=1,
                pages_failed=0,
                documents_created=1,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[url],
                failed_urls={},
                message="ok",
                warnings=[],
                elapsed_time_ms=1,
            )

        with (
            patch(
                "xagent.web.api.kb.get_upload_path", side_effect=patched_get_upload_path
            ),
            patch(
                "xagent.web.api.kb.get_session_local", return_value=TestingSessionLocal
            ),
            patch(
                "xagent.web.api.kb.run_web_ingestion",
                side_effect=stub_run_web_ingestion,
            ),
        ):
            response = client.post(
                "/api/kb/ingest-web",
                data={"collection": "c1", "start_url": "https://example.com"},
                headers=headers,
            )

        assert response.status_code == 200

        session = TestingSessionLocal()
        try:
            rows = (
                session.query(UploadedFile)
                .filter(UploadedFile.user_id == user.id)
                .all()
            )
            assert len(rows) == 1
            assert Path(rows[0].storage_path).exists()
        finally:
            session.close()

    def test_ingest_web_upsert_failure_cleans_up_persistent_file(
        self, web_test_env, tmp_path: Path
    ) -> None:
        """If UploadedFile upsert fails, the handler should remove the orphaned persistent file."""
        app, headers, user, TestingSessionLocal = web_test_env
        client = TestClient(app)

        uploads_root = tmp_path / "uploads"
        # We'll compute the deterministic filename the handler will use.
        import hashlib

        url = "https://example.com/page"
        collection = "c1"
        title = "How to edit a completed job?"
        url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[:16]
        filename = f"{url_hash}_{_normalize_web_title_for_filename(title)}.md"
        expected_persistent = uploads_root / f"user_{user.id}" / collection / filename

        def patched_get_upload_path(
            filename_arg: str,
            *,
            user_id: int,
            collection: str,
            collection_is_sanitized: bool,
        ) -> Path:
            assert filename_arg == filename
            return uploads_root / f"user_{user_id}" / collection / filename_arg

        async def stub_run_web_ingestion(
            *,
            collection: str,
            crawl_config,
            ingestion_config,
            user_id: int,
            is_admin: bool,
            file_handler,
        ):
            temp_md = tmp_path / "temp.md"
            temp_md.write_text("# Title\n\nBody", encoding="utf-8")
            file_handler(temp_md, title, collection, url)

            from xagent.core.tools.core.RAG_tools.core.schemas import WebIngestionResult

            return WebIngestionResult(status="success", message="ok", warnings=[])

        def boom_upsert(*args, **kwargs):
            raise RuntimeError("boom")

        with (
            patch(
                "xagent.web.api.kb.get_upload_path", side_effect=patched_get_upload_path
            ),
            patch(
                "xagent.web.api.kb.get_session_local", return_value=TestingSessionLocal
            ),
            patch(
                "xagent.web.api.kb._create_web_uploaded_file_record",
                side_effect=boom_upsert,
            ),
            patch(
                "xagent.web.api.kb.run_web_ingestion",
                side_effect=stub_run_web_ingestion,
            ),
        ):
            response = client.post(
                "/api/kb/ingest-web",
                data={"collection": collection, "start_url": "https://example.com"},
                headers=headers,
            )

        assert response.status_code == 500
        assert not expected_persistent.exists()

    def test_ingest_web_zero_success_failure_compensates_new_uploaded_file(
        self, web_test_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero-success web ingest failure should remove the staged upload row and artifacts."""
        app, headers, user, TestingSessionLocal = web_test_env
        client = TestClient(app)
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        monkeypatch.setenv(
            "XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized")
        )
        get_file_storage.cache_clear()

        uploads_root = tmp_path / "uploads"
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_root))
        url = "https://example.com/page"
        collection = "c1"
        title = "Failed page"
        url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[:16]
        filename = f"{url_hash}_{_normalize_web_title_for_filename(title)}.md"
        expected_persistent = uploads_root / f"user_{user.id}" / collection / filename
        captured: dict[str, str] = {}

        def patched_get_upload_path(
            filename_arg: str,
            *,
            user_id: int,
            collection: str,
            collection_is_sanitized: bool,
        ) -> Path:
            assert collection_is_sanitized is True
            return uploads_root / f"user_{user_id}" / collection / filename_arg

        async def stub_run_web_ingestion(
            *,
            collection: str,
            crawl_config,
            ingestion_config,
            user_id: int,
            is_admin: bool,
            file_handler,
        ):
            temp_md = tmp_path / "temp.md"
            temp_md.write_text("# Title\n\nBody", encoding="utf-8")
            file_info = file_handler(temp_md, title, collection, url)
            captured["file_id"] = str(file_info["file_id"])

            from xagent.core.tools.core.RAG_tools.core.schemas import WebIngestionResult
            from xagent.web.services.managed_file_ref import build_upload_storage_key

            result = WebIngestionResult(
                status="error",
                collection=collection,
                total_urls_found=1,
                pages_crawled=1,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[url],
                failed_urls={url: "embedding failed"},
                message="failed",
                warnings=[],
                elapsed_time_ms=1,
            )
            captured["storage_key"] = build_upload_storage_key(
                user.id,
                captured["file_id"],
                filename,
            )
            assert expected_persistent.exists()
            assert get_file_storage().exists(captured["storage_key"])

            file_info["rollback_on_failure"](result)
            return result

        with (
            patch(
                "xagent.web.api.kb.get_upload_path", side_effect=patched_get_upload_path
            ),
            patch(
                "xagent.web.api.kb.get_session_local", return_value=TestingSessionLocal
            ),
            patch(
                "xagent.web.api.kb.run_web_ingestion",
                side_effect=stub_run_web_ingestion,
            ),
        ):
            response = client.post(
                "/api/kb/ingest-web",
                data={"collection": collection, "start_url": "https://example.com"},
                headers=headers,
            )

        assert response.status_code == 500
        assert captured["file_id"]
        assert not expected_persistent.exists()
        assert not get_file_storage().exists(captured["storage_key"])

        session = TestingSessionLocal()
        try:
            assert (
                session.query(UploadedFile)
                .filter(UploadedFile.file_id == captured["file_id"])
                .first()
                is None
            )
        finally:
            session.close()

    def test_ingest_web_failure_callback_removes_new_uploaded_file(
        self, web_test_env, tmp_path: Path
    ) -> None:
        """The route-provided file handler should compensate new file records on failure."""
        app, headers, user, TestingSessionLocal = web_test_env
        client = TestClient(app)

        uploads_root = tmp_path / "uploads"
        import hashlib

        url = "https://example.com/page"
        collection = "c1"
        title = "How to edit a completed job?"
        url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[:16]
        filename = f"{url_hash}_{_normalize_web_title_for_filename(title)}.md"
        expected_persistent = uploads_root / f"user_{user.id}" / collection / filename

        def patched_get_upload_path(
            filename_arg: str,
            *,
            user_id: int,
            collection: str,
            collection_is_sanitized: bool,
        ) -> Path:
            assert filename_arg == filename
            return uploads_root / f"user_{user_id}" / collection / filename_arg

        async def stub_run_web_ingestion(
            *,
            collection: str,
            crawl_config,
            ingestion_config,
            user_id: int,
            is_admin: bool,
            file_handler,
        ):
            temp_md = tmp_path / "temp.md"
            temp_md.write_text("# Title\n\nBody", encoding="utf-8")
            file_info = file_handler(temp_md, title, collection, url)
            assert callable(file_info["rollback_on_failure"])

            from xagent.core.tools.core.RAG_tools.core.schemas import (
                IngestionResult,
                IngestionStepResult,
                WebIngestionResult,
            )

            file_info["rollback_on_failure"](
                IngestionResult(
                    status="partial",
                    doc_id="doc-1",
                    parse_hash="hash",
                    completed_steps=[
                        IngestionStepResult(
                            name="register_document",
                            metadata={"doc_id": "doc-1", "created": True},
                        )
                    ],
                    failed_step="embed_chunks",
                    message="embedding failed",
                )
            )

            return WebIngestionResult(
                status="error",
                collection=collection,
                total_urls_found=1,
                pages_crawled=1,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[url],
                failed_urls={url: "ingestion failed"},
                message="ingestion failed",
                warnings=[],
                elapsed_time_ms=1,
            )

        with (
            patch(
                "xagent.web.api.kb.get_upload_path", side_effect=patched_get_upload_path
            ),
            patch(
                "xagent.web.api.kb.get_session_local", return_value=TestingSessionLocal
            ),
            patch(
                "xagent.web.api.kb.run_web_ingestion",
                side_effect=stub_run_web_ingestion,
            ),
            patch("xagent.web.api.kb.delete_document") as mock_delete_document,
        ):
            mock_delete_document.return_value = MagicMock(status="success")
            response = client.post(
                "/api/kb/ingest-web",
                data={"collection": collection, "start_url": "https://example.com"},
                headers=headers,
            )

        assert response.status_code == 500
        assert not expected_persistent.exists()
        mock_delete_document.assert_called_once_with(
            collection, "doc-1", user.id, False
        )

        session = TestingSessionLocal()
        try:
            rows = (
                session.query(UploadedFile)
                .filter(UploadedFile.user_id == user.id)
                .all()
            )
            assert rows == []
        finally:
            session.close()

    def test_ingest_web_failure_callback_restores_refreshed_existing_file(
        self, web_test_env, tmp_path: Path
    ) -> None:
        """A refreshed existing web file should be restored when ingestion fails."""
        app, headers, user, TestingSessionLocal = web_test_env
        client = TestClient(app)

        uploads_root = tmp_path / "uploads"
        import hashlib

        url = "https://example.com/page"
        collection = "c1"
        title = "How to edit a completed job?"
        url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[:16]
        filename = f"{url_hash}_{_normalize_web_title_for_filename(title)}.md"
        persistent_file = uploads_root / f"user_{user.id}" / collection / filename
        persistent_file.parent.mkdir(parents=True)
        persistent_file.write_text("old content", encoding="utf-8")

        session = TestingSessionLocal()
        try:
            existing_record = _upsert_uploaded_file_record(
                session,
                user_id=user.id,
                filename=filename,
                storage_path=persistent_file,
                mime_type="text/markdown",
                file_size=persistent_file.stat().st_size,
            )
            existing_file_id = str(existing_record.file_id)
        finally:
            session.close()

        def patched_get_upload_path(
            filename_arg: str,
            *,
            user_id: int,
            collection: str,
            collection_is_sanitized: bool,
        ) -> Path:
            assert filename_arg == filename
            return uploads_root / f"user_{user_id}" / collection / filename_arg

        async def stub_run_web_ingestion(
            *,
            collection: str,
            crawl_config,
            ingestion_config,
            user_id: int,
            is_admin: bool,
            file_handler,
        ):
            temp_md = tmp_path / "temp.md"
            temp_md.write_text("new content", encoding="utf-8")
            file_info = file_handler(temp_md, title, collection, url)
            assert file_info["file_id"] == existing_file_id
            assert persistent_file.read_text(encoding="utf-8") == "new content"
            assert callable(file_info["rollback_on_failure"])

            from xagent.core.tools.core.RAG_tools.core.schemas import WebIngestionResult

            file_info["rollback_on_failure"]()

            return WebIngestionResult(
                status="error",
                collection=collection,
                total_urls_found=1,
                pages_crawled=1,
                pages_failed=1,
                documents_created=0,
                chunks_created=0,
                embeddings_created=0,
                crawled_urls=[url],
                failed_urls={url: "ingestion failed"},
                message="ingestion failed",
                warnings=[],
                elapsed_time_ms=1,
            )

        with (
            patch(
                "xagent.web.api.kb.get_upload_path", side_effect=patched_get_upload_path
            ),
            patch(
                "xagent.web.api.kb.get_session_local", return_value=TestingSessionLocal
            ),
            patch(
                "xagent.web.api.kb._mark_uploaded_file_for_reindex", return_value=True
            ),
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=object(),
            ) as mock_snapshot_runs,
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ) as mock_snapshot_rag,
            patch(
                "xagent.web.api.kb._restore_ingestion_runs_snapshot"
            ) as mock_restore_runs,
            patch(
                "xagent.web.api.kb._restore_rag_document_snapshot"
            ) as mock_restore_rag,
            patch(
                "xagent.web.api.kb.run_web_ingestion",
                side_effect=stub_run_web_ingestion,
            ),
        ):
            response = client.post(
                "/api/kb/ingest-web",
                data={"collection": collection, "start_url": "https://example.com"},
                headers=headers,
            )

        assert response.status_code == 500
        assert persistent_file.read_text(encoding="utf-8") == "old content"
        mock_snapshot_runs.assert_called_once_with(existing_file_id)
        mock_snapshot_rag.assert_called_once_with(
            existing_file_id,
            user_id=user.id,
            is_admin=False,
        )
        mock_restore_runs.assert_called_once()
        mock_restore_rag.assert_called_once_with(
            mock_snapshot_rag.return_value,
            user_id=user.id,
            is_admin=False,
        )

        session = TestingSessionLocal()
        try:
            rows = (
                session.query(UploadedFile)
                .filter(UploadedFile.user_id == user.id)
                .all()
            )
            assert len(rows) == 1
            assert rows[0].file_id == existing_file_id
        finally:
            session.close()

    def test_ingest_web_refresh_restores_file_when_rag_restore_fails(
        self, web_test_env, tmp_path: Path
    ) -> None:
        """File and UploadedFile rollback should continue even if RAG restore fails."""
        app, headers, user, TestingSessionLocal = web_test_env
        client = TestClient(app)

        uploads_root = tmp_path / "uploads"
        import hashlib

        url = "https://example.com/page"
        collection = "c1"
        title = "How to edit a completed job?"
        url_hash = hashlib.sha256(f"{collection}:{url}".encode()).hexdigest()[:16]
        filename = f"{url_hash}_{_normalize_web_title_for_filename(title)}.md"
        persistent_file = uploads_root / f"user_{user.id}" / collection / filename
        persistent_file.parent.mkdir(parents=True)
        persistent_file.write_text("old content", encoding="utf-8")

        session = TestingSessionLocal()
        try:
            existing_record = _upsert_uploaded_file_record(
                session,
                user_id=user.id,
                filename=filename,
                storage_path=persistent_file,
                mime_type="text/markdown",
                file_size=persistent_file.stat().st_size,
            )
            existing_file_id = str(existing_record.file_id)
        finally:
            session.close()

        def patched_get_upload_path(
            filename_arg: str,
            *,
            user_id: int,
            collection: str,
            collection_is_sanitized: bool,
        ) -> Path:
            assert filename_arg == filename
            return uploads_root / f"user_{user_id}" / collection / filename_arg

        async def stub_run_web_ingestion(
            *,
            collection: str,
            crawl_config,
            ingestion_config,
            user_id: int,
            is_admin: bool,
            file_handler,
        ):
            temp_md = tmp_path / "temp.md"
            temp_md.write_text("new content", encoding="utf-8")
            file_info = file_handler(temp_md, title, collection, url)
            assert file_info["file_id"] == existing_file_id
            assert persistent_file.read_text(encoding="utf-8") == "new content"

            file_info["rollback_on_failure"]()

        with (
            patch(
                "xagent.web.api.kb.get_upload_path", side_effect=patched_get_upload_path
            ),
            patch(
                "xagent.web.api.kb.get_session_local", return_value=TestingSessionLocal
            ),
            patch(
                "xagent.web.api.kb._mark_uploaded_file_for_reindex", return_value=True
            ),
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=object(),
            ),
            patch(
                "xagent.web.api.kb._restore_ingestion_runs_snapshot"
            ) as mock_restore_runs,
            patch(
                "xagent.web.api.kb._restore_rag_document_snapshot",
                side_effect=RuntimeError("rag restore failed"),
            ),
            patch(
                "xagent.web.api.kb.run_web_ingestion",
                side_effect=stub_run_web_ingestion,
            ),
        ):
            response = client.post(
                "/api/kb/ingest-web",
                data={"collection": collection, "start_url": "https://example.com"},
                headers=headers,
            )

        assert response.status_code == 500
        assert persistent_file.read_text(encoding="utf-8") == "old content"
        mock_restore_runs.assert_called_once()

        session = TestingSessionLocal()
        try:
            rows = (
                session.query(UploadedFile)
                .filter(UploadedFile.user_id == user.id)
                .all()
            )
            assert len(rows) == 1
            assert rows[0].file_id == existing_file_id
        finally:
            session.close()


class TestWebFileRefreshHelpers:
    """Test helper functions for stale content refresh."""

    def test_web_rollback_exception_path_deletes_docs_for_file_id(self) -> None:
        with (
            patch(
                "xagent.web.api.kb._list_document_refs_for_uploaded_file",
                return_value=[("test_collection", "doc-1")],
            ) as mock_list_refs,
            patch("xagent.web.api.kb.delete_document") as mock_delete_document,
        ):
            mock_delete_document.return_value = MagicMock(status="success")

            _rollback_failed_web_document_ingestion(
                collection_name="test_collection",
                result=None,
                user_id=1,
                is_admin=False,
                file_id="file-1",
            )

        mock_list_refs.assert_called_once_with("file-1")
        mock_delete_document.assert_called_once_with(
            "test_collection",
            "doc-1",
            1,
            False,
        )

    def test_web_rag_side_effect_cleanup_continues_after_delete_failure(
        self,
    ) -> None:
        with (
            patch(
                "xagent.web.api.kb._list_document_refs_for_uploaded_file",
                return_value=[
                    ("test_collection", "doc-1"),
                    ("test_collection", "doc-2"),
                ],
            ),
            patch("xagent.web.api.kb.delete_document") as mock_delete_document,
        ):
            mock_delete_document.side_effect = [
                MagicMock(status="error", message="delete failed"),
                MagicMock(status="success"),
            ]

            with pytest.raises(RuntimeError, match="Best-effort RAG cleanup failed"):
                _delete_web_rag_side_effects_for_file_id(
                    collection_name="test_collection",
                    file_id="file-1",
                    user_id=1,
                    is_admin=False,
                )

        assert mock_delete_document.call_count == 2
        assert mock_delete_document.call_args_list[0].args == (
            "test_collection",
            "doc-1",
            1,
            False,
        )
        assert mock_delete_document.call_args_list[1].args == (
            "test_collection",
            "doc-2",
            1,
            False,
        )

    def test_web_rollback_exception_path_uses_file_id_after_empty_snapshot(
        self,
    ) -> None:
        snapshot = _RagDocumentSnapshot(doc_refs=[], rows_by_table={})
        with (
            patch(
                "xagent.web.api.kb._restore_rag_document_snapshot"
            ) as mock_restore_snapshot,
            patch(
                "xagent.web.api.kb._delete_web_rag_side_effects_for_file_id"
            ) as mock_delete_for_file,
        ):
            _rollback_failed_web_document_ingestion(
                collection_name="test_collection",
                result=None,
                user_id=1,
                is_admin=False,
                rag_snapshot=snapshot,
                file_id="file-1",
            )

        mock_restore_snapshot.assert_called_once_with(
            snapshot,
            user_id=1,
            is_admin=False,
        )
        mock_delete_for_file.assert_called_once_with(
            collection_name="test_collection",
            file_id="file-1",
            user_id=1,
            is_admin=False,
        )

    def test_restore_rag_snapshot_rows_batches_unknown_table_delete(self) -> None:
        table = MagicMock()
        table.schema.names = []

        _restore_rag_snapshot_rows(
            table,
            table_name="custom_table",
            snapshot_rows=[{"collection": "c1", "doc_id": "doc-old"}],
            current_rows=[
                {"collection": "c1", "doc_id": "doc-1"},
                {"collection": "c1", "doc_id": "doc-2"},
            ],
            user_id=1,
            is_admin=False,
        )

        table.delete.assert_called_once()
        delete_filter = table.delete.call_args.args[0]
        assert "(collection = 'c1' and doc_id = 'doc-1')" in delete_filter
        assert "(collection = 'c1' and doc_id = 'doc-2')" in delete_filter
        assert " or " in delete_filter
        table.add.assert_called_once_with([{"collection": "c1", "doc_id": "doc-old"}])

    def test_restore_rag_snapshot_rows_batches_stale_row_delete(self) -> None:
        table = MagicMock()
        table.schema.names = []

        _restore_rag_snapshot_rows(
            table,
            table_name="chunks",
            snapshot_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-old",
                }
            ],
            current_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-old",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-stale",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "parse_hash": "hash",
                    "chunk_id": "chunk-stale-2",
                },
            ],
            user_id=1,
            is_admin=False,
        )

        table.merge_insert.assert_called_once_with(
            ["collection", "doc_id", "parse_hash", "chunk_id"]
        )
        table.delete.assert_called_once()
        delete_filter = table.delete.call_args.args[0]
        assert "chunk_id = 'chunk-old'" not in delete_filter
        assert "(collection = 'c1'" in delete_filter
        assert "chunk_id = 'chunk-stale'" in delete_filter
        assert "chunk_id = 'chunk-stale-2'" in delete_filter
        assert " or " in delete_filter

    def test_restore_rag_snapshot_rows_keys_embeddings_by_parse_and_model(
        self,
    ) -> None:
        table = MagicMock()
        table.schema.names = []

        _restore_rag_snapshot_rows(
            table,
            table_name="embeddings_text_embedding_v4",
            snapshot_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-old",
                    "model": "model-a",
                }
            ],
            current_rows=[
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-old",
                    "model": "model-a",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-new",
                    "model": "model-a",
                },
                {
                    "collection": "c1",
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "parse_hash": "parse-old",
                    "model": "model-b",
                },
            ],
            user_id=1,
            is_admin=False,
        )

        table.merge_insert.assert_called_once_with(
            ["collection", "doc_id", "chunk_id", "parse_hash", "model"]
        )
        table.delete.assert_called_once()
        delete_filter = table.delete.call_args.args[0]
        assert "parse_hash = 'parse-new'" in delete_filter
        assert "model = 'model-b'" in delete_filter
        assert "parse_hash = 'parse-old' and model = 'model-a'" not in delete_filter
        assert " or " in delete_filter

    def test_get_file_sha256_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.md"
            file_path.write_text("old-content", encoding="utf-8")
            old_hash = _get_file_sha256(file_path)

            file_path.write_text("new-content", encoding="utf-8")
            new_hash = _get_file_sha256(file_path)

            assert old_hash != new_hash

    def test_atomic_replace_file_overwrites_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.md"
            target_path = Path(temp_dir) / "target.md"
            source_path.write_text("new-value", encoding="utf-8")
            target_path.write_text("old-value", encoding="utf-8")

            _atomic_replace_file(source_path, target_path)

            assert target_path.read_text(encoding="utf-8") == "new-value"

    def test_web_file_lock_serializes_same_key(self) -> None:
        lock_key = "1:same-url-hash"
        active_count = 0
        peak_active_count = 0
        guard = threading.Lock()

        def _worker() -> None:
            nonlocal active_count, peak_active_count
            with _WebFileLock(lock_key):
                with guard:
                    active_count += 1
                    peak_active_count = max(peak_active_count, active_count)
                time.sleep(0.05)
                with guard:
                    active_count -= 1

        threads = [threading.Thread(target=_worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert peak_active_count == 1

    def test_web_file_lock_registry_entry_is_released_after_use(self) -> None:
        lock_key = "1:transient-url-hash"
        _WEB_FILE_LOCKS.pop(lock_key, None)

        with _WebFileLock(lock_key):
            assert lock_key in _WEB_FILE_LOCKS

        assert lock_key not in _WEB_FILE_LOCKS

    def test_cache_updates_with_upsert_returned_file_id(self) -> None:
        processed_urls: dict[str, str] = {"hash-key": "old-file-id"}

        class _Record:
            def __init__(self, file_id: str) -> None:
                self.file_id = file_id

        file_record = _Record("new-file-id")
        processed_urls["hash-key"] = str(file_record.file_id)

        assert processed_urls["hash-key"] == "new-file-id"

    def test_compensate_new_web_ingest_files_continues_after_commit_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        compensated_file_ids: list[str] = []

        class _CleanupResult:
            side_effects_may_remain = False
            errors: tuple[str, ...] = ()

        def fake_compensate(_db, *, file_id: str, user_id: int):
            assert user_id == 1
            compensated_file_ids.append(file_id)
            return _CleanupResult()

        class _DB:
            def __init__(self) -> None:
                self.commits = 0
                self.rollbacks = 0

            def commit(self) -> None:
                self.commits += 1
                if self.commits == 1:
                    raise RuntimeError("commit unavailable")

            def rollback(self) -> None:
                self.rollbacks += 1

        monkeypatch.setattr(
            "xagent.web.api.kb._compensate_new_uploaded_file",
            fake_compensate,
        )
        db = _DB()

        cleanup_incomplete, cleanup_errors = _compensate_new_web_ingest_files(
            db,  # type: ignore[arg-type]
            file_ids={"file-b", "file-a"},
            user_id=1,
        )

        assert compensated_file_ids == ["file-a", "file-b"]
        assert db.commits == 2
        assert db.rollbacks == 1
        assert cleanup_incomplete is True
        assert cleanup_errors == [
            "Database commit failed for file file-a: commit unavailable"
        ]

    def test_mark_uploaded_file_for_reindex_clears_ingestion_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deleted_filters: list[str] = []

        class _FakeTable:
            def search(self):
                return self

            def where(self, _expr: str):
                return self

            def select(self, _fields: list[str]):
                return self

            def limit(self, _value: int):
                return self

            def delete(self, expr: str) -> None:
                deleted_filters.append(expr)

        class _FakeConn:
            def open_table(self, _name: str):
                return _FakeTable()

        monkeypatch.setattr(
            "xagent.providers.vector_store.lancedb.get_connection_from_env",
            lambda: _FakeConn(),
        )
        monkeypatch.setattr(
            "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.ensure_documents_table",
            lambda _conn: None,
        )
        monkeypatch.setattr(
            "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.ensure_ingestion_runs_table",
            lambda _conn: None,
        )
        monkeypatch.setattr(
            "xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils.query_to_list",
            lambda _query: [{"collection": "kb", "doc_id": "doc-1"}],
        )

        marked = _mark_uploaded_file_for_reindex("file-123")

        assert marked is True
        assert len(deleted_filters) == 1
        assert "collection = 'kb'" in deleted_filters[0]
        assert "doc_id = 'doc-1'" in deleted_filters[0]

    def test_refresh_fails_when_ingestion_run_snapshot_fails(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
    ) -> None:
        """Existing files should not be re-ingested without a rollback snapshot."""
        existing_path = tmp_path / "existing.md"
        existing_path.write_text("old content", encoding="utf-8")
        temp_path = tmp_path / "new.md"
        temp_path.write_text("new content", encoding="utf-8")

        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=test_user.id,
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
        file_id = str(existing_record.file_id)
        processed_urls: dict[str, str] = {}

        with (
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=None,
            ) as mock_snapshot_runs,
            patch("xagent.web.api.kb._mark_uploaded_file_for_reindex") as mock_mark,
        ):
            with pytest.raises(RuntimeError, match="snapshot ingestion status"):
                _refresh_existing_file_if_changed(
                    existing_record=existing_record,
                    temp_file_path=temp_path,
                    db_session=db_session,
                    user_id=test_user.id,
                    is_admin=False,
                    collection_name="test_collection",
                    url="https://example.com/page",
                    filename="existing.md",
                    url_hash="url-hash",
                    processed_urls=processed_urls,
                    context="unit-test",
                )

        assert existing_path.read_text(encoding="utf-8") == "old content"
        assert processed_urls == {}
        mock_snapshot_runs.assert_called_once_with(file_id)
        mock_mark.assert_not_called()

    def test_unchanged_existing_file_result_has_rollback_callback(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
    ) -> None:
        existing_path = tmp_path / "existing.md"
        existing_path.write_text("same content", encoding="utf-8")
        temp_path = tmp_path / "incoming.md"
        temp_path.write_text("same content", encoding="utf-8")
        existing_record = _upsert_uploaded_file_record(
            db_session,
            user_id=test_user.id,
            filename="existing.md",
            storage_path=existing_path,
            mime_type="text/markdown",
            file_size=existing_path.stat().st_size,
        )
        run_snapshot = object()
        rag_snapshot = object()

        with (
            patch(
                "xagent.web.api.kb._snapshot_ingestion_runs_for_uploaded_file",
                return_value=run_snapshot,
            ),
            patch(
                "xagent.web.api.kb._snapshot_rag_documents_for_uploaded_file",
                return_value=rag_snapshot,
            ),
            patch(
                "xagent.web.api.kb._rollback_failed_web_document_ingestion"
            ) as mock_rollback_rag,
            patch(
                "xagent.web.api.kb._restore_ingestion_runs_snapshot"
            ) as mock_restore_runs,
        ):
            result = _refresh_existing_file_if_changed(
                existing_record=existing_record,
                temp_file_path=temp_path,
                db_session=db_session,
                user_id=test_user.id,
                is_admin=False,
                collection_name="test_collection",
                url="https://example.com/page",
                filename="existing.md",
                url_hash="url-hash",
                processed_urls={},
                context="unit-test",
            )
            result["rollback_on_failure"](None)

        mock_rollback_rag.assert_called_once_with(
            collection_name="test_collection",
            result=None,
            user_id=test_user.id,
            is_admin=False,
            rag_snapshot=rag_snapshot,
            file_id=str(existing_record.file_id),
        )
        mock_restore_runs.assert_called_once_with(run_snapshot)
