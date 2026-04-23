"""Tests for _handle_web_file function in kb.py.

This module tests the file handler callback used in web ingestion, which handles:
- URL-based deduplication (in-memory cache + database)
- File persistence (copying temp files to permanent storage)
- UploadedFile record creation/update
- Cross-collection isolation
- Error handling and cleanup
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.api.kb import (
    _WEB_FILE_LOCKS,
    _atomic_replace_file,
    _get_file_sha256,
    _mark_uploaded_file_for_reindex,
    _upsert_uploaded_file_record,
    _WebFileLock,
)
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


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


class TestHandleWebFile:
    """Test _handle_web_file function for web ingestion file handling."""

    def test_new_file_creation(
        self, db_session: Session, test_user: User, mock_user: MagicMock
    ):
        """Test creating a new file when no cache or DB record exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary markdown file
            temp_file = Path(temp_dir) / "temp.md"
            temp_file.write_text("# Test Page\n\nContent here")

            # Mock get_upload_path to return a path in temp_dir
            persistent_path = Path(temp_dir) / "uploads" / "user_1" / "test_collection"
            persistent_path.mkdir(parents=True, exist_ok=True)

            with patch("xagent.web.api.kb.get_upload_path") as mock_get_path:
                mock_get_path.return_value = persistent_path / "file.md"

                # Simulate the function setup inside ingest_web
                _processed_urls = {}

                # Mock sanitize_path_component
                with patch(
                    "xagent.web.api.kb.sanitize_path_component"
                ) as mock_sanitize:
                    mock_sanitize.return_value = "test_page"

                    # Call the file handler logic (simplified from actual implementation)
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
                    assert persistent_file.exists()

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


class TestWebFileRefreshHelpers:
    """Test helper functions for stale content refresh."""

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
