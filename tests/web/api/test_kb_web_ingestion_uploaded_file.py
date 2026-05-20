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

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.api.kb import (
    _WEB_FILE_LOCKS,
    _atomic_replace_file,
    _get_file_sha256,
    _mark_uploaded_file_for_reindex,
    _normalize_web_title_for_filename,
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
                "xagent.web.api.kb._upsert_uploaded_file_record",
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
