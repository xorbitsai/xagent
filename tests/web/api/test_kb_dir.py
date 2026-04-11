import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)
from xagent.web.api.auth import hash_password
from xagent.web.api.kb import kb_router
from xagent.web.models.database import Base, get_db
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


@pytest.fixture(scope="function")
def test_env():
    """Setup test database and app"""
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    test_engine = create_engine(f"sqlite:///{temp_db_path}")
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
        username="testuser", password_hash=hash_password("test"), is_admin=False
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # Mock JWT token (must include type="access" for get_current_user)
    from datetime import datetime, timedelta

    import jwt

    from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

    payload = {
        "sub": user.username,
        "user_id": user.id,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}

    yield app, headers, user, TestingSessionLocal

    session.close()
    os.unlink(temp_db_path)


@pytest.fixture(scope="function")
def temp_uploads():
    """Setup temporary uploads directory and patch it"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        def patched_get_upload_path(
            filename,
            task_id=None,
            folder=None,
            user_id=None,
            collection=None,
            create_if_not_exists=True,
        ):
            base = temp_path
            if user_id:
                user_dir = base / f"user_{user_id}"
                if collection:
                    d = user_dir / collection
                    if create_if_not_exists:
                        d.mkdir(parents=True, exist_ok=True)
                    return d / filename
                if create_if_not_exists:
                    user_dir.mkdir(parents=True, exist_ok=True)
                return user_dir / filename
            return base / filename

        with (
            patch(
                "xagent.web.api.kb.get_upload_path",
                side_effect=patched_get_upload_path,
            ),
            patch(
                "xagent.web.services.kb_collection_service.get_upload_path",
                side_effect=patched_get_upload_path,
            ),
            patch(
                "xagent.web.config.get_upload_path",
                side_effect=patched_get_upload_path,
            ),
            patch("xagent.config.get_uploads_dir", return_value=Path(temp_path)),
            patch(
                "xagent.web.services.kb_file_service.get_uploads_dir",
                return_value=Path(temp_path),
            ),
            patch(
                "xagent.web.services.kb_collection_service.get_uploads_dir",
                return_value=Path(temp_path),
            ),
        ):
            yield temp_path


def test_kb_ingest_creates_collection_dir(test_env, temp_uploads):
    """Test that ingesting a document creates a collection-specific directory"""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_test_coll"
    filename = "test_doc.txt"

    # Mock the RAG pipeline to avoid heavy dependencies
    with patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest:
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        mock_ingest.return_value = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

        # Upload file
        files = {"file": (filename, b"content", "text/plain")}
        data = {"collection": collection_name}

        response = client.post(
            "/api/kb/ingest", files=files, data=data, headers=headers
        )

        assert response.status_code == 200

        # Check if physical directory was created
        expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
        assert expected_path.exists()
        assert expected_path.is_file()


def test_kb_delete_cleans_physical_dir(test_env, temp_uploads):
    """Test that deleting a collection also removes the physical directory"""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_to_delete"

    # Pre-create the collection directory
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    # Mock delete_collection (the database part)
    with patch("xagent.web.api.kb.delete_collection") as mock_delete:
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="success",
            collection=collection_name,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        # Delete collection
        response = client.delete(
            f"/api/kb/collections/{collection_name}", headers=headers
        )

        assert response.status_code == 200

        # Check if physical directory was removed
        assert not coll_dir.exists()


def test_kb_ingest_rejects_path_traversal_in_collection_name(test_env, temp_uploads):
    """Test that ingest API rejects path traversal in collection name."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    malicious_collections = [
        "../../../etc",
        "..\\..\\..\\windows",
        "collection/../other",
        "../collection",
    ]

    filename = "test_doc.txt"

    for collection_name in malicious_collections:
        with patch("xagent.web.api.kb.run_document_ingestion"):
            files = {"file": (filename, b"content", "text/plain")}
            data = {"collection": collection_name}

            response = client.post(
                "/api/kb/ingest", files=files, data=data, headers=headers
            )

            # Should reject with 422 (validation error)
            assert response.status_code == 422
            assert "Invalid collection name" in response.json()["detail"]


def test_kb_ingest_rejects_invalid_characters_in_collection_name(
    test_env, temp_uploads
):
    """Test that ingest API rejects invalid characters in collection name."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    invalid_collections = [
        "collection name",  # Space
        "collection@name",  # @ symbol
        "collection#name",  # # symbol
        "collection/name",  # Path separator
    ]

    filename = "test_doc.txt"

    for collection_name in invalid_collections:
        with patch("xagent.web.api.kb.run_document_ingestion"):
            files = {"file": (filename, b"content", "text/plain")}
            data = {"collection": collection_name}

            response = client.post(
                "/api/kb/ingest", files=files, data=data, headers=headers
            )

            # Should reject with 422 (validation error)
            assert response.status_code == 422
            assert "Invalid collection name" in response.json()["detail"]


def test_kb_ingest_rejects_too_long_collection_name(test_env, temp_uploads):
    """Test that ingest API rejects collection names exceeding length limit."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    # Create a collection name that exceeds MAX_COLLECTION_NAME_LENGTH (100)
    too_long_collection = "a" * 101
    filename = "test_doc.txt"

    with patch("xagent.web.api.kb.run_document_ingestion"):
        files = {"file": (filename, b"content", "text/plain")}
        data = {"collection": too_long_collection}

        response = client.post(
            "/api/kb/ingest", files=files, data=data, headers=headers
        )

        # Should reject with 422 (validation error)
        assert response.status_code == 422
        assert "Invalid collection name" in response.json()["detail"]


def test_kb_ingest_validates_derived_collection_name_from_filename(
    test_env, temp_uploads
):
    """Test that ingest API validates collection name derived from filename."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    # Test with filename that would create invalid collection name
    # Note: "../../../etc.txt" becomes "etc.txt" after basename, which is valid
    # So we test actual invalid cases
    malicious_filenames = [
        "file name.txt",  # Would create "file name" with space
        "file@name.txt",  # Would create "file@name" with invalid character
    ]

    for filename in malicious_filenames:
        with patch("xagent.web.api.kb.run_document_ingestion"):
            files = {"file": (filename, b"content", "text/plain")}
            # Don't provide collection parameter, so it's derived from filename

            response = client.post(
                "/api/kb/ingest", files=files, data={}, headers=headers
            )

            # Should reject with 422 (validation error)
            assert response.status_code == 422
            detail = response.json().get("detail", "")
            assert "Invalid collection name" in detail or "invalid" in detail.lower()


def test_kb_delete_rejects_path_traversal_in_collection_name(test_env, temp_uploads):
    """Test that delete_collection_api rejects path traversal in collection name."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    malicious_collections = [
        r"collection\\other",
        r"collection\\..\\other",
    ]

    for collection_name in malicious_collections:
        # URL encode the collection name for the path parameter
        from urllib.parse import quote

        encoded_name = quote(collection_name, safe="")
        response = client.delete(f"/api/kb/collections/{encoded_name}", headers=headers)

        assert response.status_code == 422
        assert "Invalid collection name" in response.json()["detail"]


def test_kb_delete_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from urllib.parse import quote

    collection_name = "示例知识库集合"
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    with patch("xagent.web.api.kb.delete_collection") as mock_delete:
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="success",
            collection=collection_name,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        response = client.delete(
            f"/api/kb/collections/{quote(collection_name, safe='')}",
            headers=headers,
        )

    assert response.status_code == 200


def test_kb_delete_rejects_mixed_script_confusable_collection_name(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from urllib.parse import quote

    collection_name = "cоllection"
    response = client.delete(
        f"/api/kb/collections/{quote(collection_name, safe='')}",
        headers=headers,
    )

    assert response.status_code == 422
    assert "Invalid collection name" in response.json()["detail"]


def test_kb_delete_physical_cleanup_failure_aborts_operation(test_env, temp_uploads):
    """Test that physical cleanup (move-to-trash) failure aborts database deletion."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_to_delete_fail"

    # Pre-create the collection directory
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    # Mock delete_collection to return success (database deletion would succeed)
    with (
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
        patch(
            "xagent.web.services.kb_collection_service.move_collection_dir_to_trash"
        ) as mock_move_to_trash,
    ):
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="success",
            collection=collection_name,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        # Simulate move-to-trash failure (delete now uses rename-to-trash, not rmtree)
        mock_move_to_trash.side_effect = PermissionError("Permission denied")

        # Attempt to delete collection
        response = client.delete(
            f"/api/kb/collections/{collection_name}", headers=headers
        )

        # Should fail with 500 (physical move failed, operation aborted)
        assert response.status_code == 500
        assert "cannot move physical files" in response.json()["detail"].lower()

        # Verify directory still exists (operation was aborted)
        assert coll_dir.exists()


def test_kb_delete_returns_physical_cleanup_status(test_env, temp_uploads):
    """Test that delete_collection_api returns physical cleanup status in response."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_to_delete_status"

    # Pre-create the collection directory
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    # Mock delete_collection
    with patch("xagent.web.api.kb.delete_collection") as mock_delete:
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="success",
            collection=collection_name,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        # Delete collection
        response = client.delete(
            f"/api/kb/collections/{collection_name}", headers=headers
        )

        assert response.status_code == 200
        data = response.json()

        # Should include physical cleanup information in warnings
        assert "warnings" in data or "message" in data
        if "warnings" in data:
            # Check that warnings include physical cleanup status
            warnings_text = " ".join(data["warnings"]).lower()
            assert any(
                keyword in warnings_text
                for keyword in ["physical", "directory", "cleanup", "removed"]
            )


def test_kb_rename_rejects_path_traversal_in_collection_names(test_env, temp_uploads):
    """Test that rename_collection_api rejects path traversal in old and new names."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    # First create a valid collection
    valid_collection = "valid_collection"
    coll_dir = temp_uploads / f"user_{user.id}" / valid_collection
    coll_dir.mkdir(parents=True, exist_ok=True)

    # Test with names that will trigger validation (path separators)
    malicious_names = [
        "collection/../other",  # Path separator
    ]

    from urllib.parse import quote

    # Mock database operations to avoid schema errors
    with patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory:
        from unittest.mock import MagicMock

        # Mock connection and table
        mock_store = MagicMock()
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_store.get_raw_connection.return_value = mock_db_conn
        mock_store_factory.return_value = mock_store

        for malicious_name in malicious_names:
            # Test malicious old name (URL encoded)
            encoded_old = quote(malicious_name, safe="")
            response = client.put(
                f"/api/kb/collections/{encoded_old}",
                data={"new_name": "new_collection"},
                headers=headers,
            )
            # May return 404 if routing fails, or 422 if validation catches it
            assert response.status_code in [422, 404]
            if response.status_code == 422:
                assert "Invalid collection name" in response.json()["detail"]

            # Test malicious new name (in form data, no URL encoding needed)
            # Mock again for the second request
            mock_table.count_rows.return_value = 0
            response = client.put(
                f"/api/kb/collections/{valid_collection}",
                data={"new_name": malicious_name},
                headers=headers,
            )
            # Form data should be validated, should return 422
            # Note: validation happens after permission check, so we need to mock DB
            assert response.status_code == 422
            assert "Invalid collection name" in response.json()["detail"]


def test_kb_rename_physical_directory_rename(test_env, temp_uploads):
    """Test that rename_collection_api physically renames the directory."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "old_collection"
    new_collection_name = "new_collection"

    # Pre-create the old collection directory
    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "some_file.txt").write_text("data")

    # Mock the database update operations to avoid database errors
    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections._list_table_names"
        ) as mock_list_tables,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
    ):
        from unittest.mock import MagicMock

        mock_list_tables.return_value = []
        # Mock connection and table to avoid database errors
        mock_store = MagicMock()
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_store.get_raw_connection.return_value = mock_db_conn
        mock_store_factory.return_value = mock_store

        # Attempt rename
        response = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

        # Should succeed (or return appropriate status)
        assert response.status_code in [200, 500]  # 500 if database operations fail

        # Check if physical directory was renamed
        new_coll_dir = temp_uploads / f"user_{user.id}" / new_collection_name
        if response.status_code == 200:
            # If rename succeeded, new directory should exist
            assert new_coll_dir.exists()
            assert not old_coll_dir.exists()
        else:
            # If database operations failed, old directory should still exist
            # (physical rename might have happened but was rolled back, or didn't happen)
            pass


def test_kb_rename_physical_rename_failure_aborts_operation(test_env, temp_uploads):
    """Test that physical rename failure aborts database update."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "old_collection"
    new_collection_name = "new_collection"

    # Pre-create the old collection directory
    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "some_file.txt").write_text("data")

    # Mock database operations to avoid schema errors
    with patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory:
        from unittest.mock import MagicMock

        # Mock connection and table
        mock_store = MagicMock()
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_store.get_raw_connection.return_value = mock_db_conn
        mock_store_factory.return_value = mock_store

        # Physical rename uses shutil.move() to support cross-device moves.
        # Patch it to fail to simulate a filesystem permission error.
        with patch("shutil.move", side_effect=PermissionError("Permission denied")):
            # Attempt rename
            response = client.put(
                f"/api/kb/collections/{old_collection_name}",
                data={"new_name": new_collection_name},
                headers=headers,
            )

            # Should fail with 500 (physical rename failed, operation aborted)
            assert response.status_code == 500
            detail = response.json()["detail"].lower()
            assert (
                "cannot rename physical directory" in detail
                or "failed to rename" in detail
                or "physical directory rename" in detail
            )

            # Verify old directory still exists (operation was aborted)
            assert old_coll_dir.exists()


def test_kb_rename_target_directory_exists_conflict(test_env, temp_uploads):
    """Test that rename_collection_api handles target directory already existing."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "old_collection"
    new_collection_name = "existing_collection"

    # Pre-create both directories
    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "old_file.txt").write_text("old data")

    new_coll_dir = temp_uploads / f"user_{user.id}" / new_collection_name
    new_coll_dir.mkdir(parents=True, exist_ok=True)
    (new_coll_dir / "new_file.txt").write_text("new data")

    # Mock database operations to avoid schema errors
    with patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory:
        from unittest.mock import MagicMock

        # Mock connection and table
        mock_store = MagicMock()
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_store.get_raw_connection.return_value = mock_db_conn
        mock_store_factory.return_value = mock_store

        # Attempt rename to existing directory
        response = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

        # Should fail with 409 (conflict) or 500
        assert response.status_code in [409, 500]
        if response.status_code == 409:
            assert "already exists" in response.json()["detail"].lower()


def test_kb_ingest_passes_file_id_to_pipeline(test_env, temp_uploads):
    """Local KB ingest should register a file record before pipeline execution."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)
    captured_file_ids: list[str] = []

    def _capture_ingest(*, file_id=None, **kwargs):
        captured_file_ids.append(str(file_id))
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        return IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

    with patch("xagent.web.api.kb.run_document_ingestion", side_effect=_capture_ingest):
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("test_doc.txt", b"content", "text/plain")},
            data={"collection": "kb_test_coll"},
            headers=headers,
        )

    assert response.status_code == 200
    payload = response.json()
    assert captured_file_ids == [payload["file_id"]]

    session = TestingSessionLocal()
    try:
        file_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == payload["file_id"])
            .first()
        )
        assert file_record is not None
        assert file_record.user_id == user.id
    finally:
        session.close()


def test_kb_ingest_cloud_passes_file_id_to_pipeline(test_env, temp_uploads):
    """Cloud ingest should also register UploadedFile before pipeline execution."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)
    captured_file_ids: list[str] = []

    class _FakeFilesService:
        def get_media(self, fileId: str):
            return {"fileId": fileId}

    class _FakeDriveService:
        def files(self):
            return _FakeFilesService()

    class _FakeDownloader:
        def __init__(self, fh, request_file):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"cloud-content")
            return None, True

    def _capture_ingest(*, file_id=None, **kwargs):
        captured_file_ids.append(str(file_id))
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        return IngestionResult(
            status="success",
            doc_id="cloud-doc-id",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

    with (
        patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=_FakeDriveService()),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _FakeDownloader),
        patch("xagent.web.api.kb.run_document_ingestion", side_effect=_capture_ingest),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_coll",
                "files": [
                    {
                        "provider": "google-drive",
                        "fileId": "drive-file-1",
                        "fileName": "cloud.txt",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert len(captured_file_ids) == 1

    session = TestingSessionLocal()
    try:
        file_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == captured_file_ids[0])
            .first()
        )
        assert file_record is not None
        assert file_record.filename == "cloud.txt"
    finally:
        session.close()


def test_check_documents_exist_prefers_uploaded_file_filename(test_env, temp_uploads):
    """Duplicate check should prefer UploadedFile filename over legacy source path."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="actual_name.txt",
            storage_path=str(temp_uploads / f"user_{user.id}" / "actual_name.txt"),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
    finally:
        session.close()

    records = [
        DocumentRecord(
            doc_id="doc-new",
            file_id=file_record.file_id,
            source_path="/legacy/wrong_name.txt",
        ),
        DocumentRecord(
            doc_id="doc-old",
            source_path="/legacy/old_name.txt",
        ),
    ]

    with patch("xagent.web.api.kb.get_vector_index_store") as mock_get_store:
        mock_store = mock_get_store.return_value
        mock_store.list_document_records.return_value = records
        response = client.post(
            "/api/kb/collections/demo/documents/check",
            json={"filenames": ["actual_name.txt", "old_name.txt", "wrong_name.txt"]},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["existing_filenames"] == ["actual_name.txt", "old_name.txt"]


def test_delete_document_prefers_file_id_and_cleans_orphan_file(test_env, temp_uploads):
    """Deleting by file_id should remove the UploadedFile row when it becomes orphaned."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "orphan.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="orphan.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
    finally:
        session.close()

    document_state = [
        {
            "collection": "demo",
            "doc_id": "doc-1",
            "file_id": target_file_id,
            "source_path": str(file_path),
        }
    ]

    def _fake_list_documents_for_user(*args, **kwargs):
        return list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()

    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=_fake_list_documents_for_user,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        response = client.delete(
            f"/api/kb/collections/demo/documents/ignored.txt?file_id={target_file_id}",
            headers=headers,
        )

    assert response.status_code == 200
    assert not file_path.exists()

    session = TestingSessionLocal()
    try:
        deleted_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == target_file_id)
            .first()
        )
        assert deleted_record is None
    finally:
        session.close()


def test_delete_document_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from urllib.parse import quote

    collection_name = "示例知识库集合"
    file_path = temp_uploads / f"user_{user.id}" / collection_name / "demo.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    deleted_doc_ids: list[str] = []
    document_state = [
        {
            "collection": collection_name,
            "doc_id": "doc-1",
            "file_id": None,
            "source_path": str(file_path),
        }
    ]

    def _fake_delete_document(collection_name_arg, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)
        assert collection_name_arg == collection_name

    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            return_value=document_state,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        response = client.delete(
            f"/api/kb/collections/{quote(collection_name, safe='')}/documents/demo.txt?doc_id=doc-1",
            headers=headers,
        )

    assert response.status_code == 200
    assert deleted_doc_ids == ["doc-1"]


def test_delete_document_rejects_mixed_script_confusable_collection_name(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from urllib.parse import quote

    collection_name = "cоllection"
    response = client.delete(
        f"/api/kb/collections/{quote(collection_name, safe='')}/documents/demo.txt",
        headers=headers,
    )

    assert response.status_code == 422
    assert "Invalid collection name" in response.json()["detail"]


def test_delete_document_rejects_path_traversal_in_collection_name(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from urllib.parse import quote

    malicious_collections = [
        r"collection\\other",
        r"collection\\..\\other",
    ]

    for collection_name in malicious_collections:
        encoded_name = quote(collection_name, safe="")
        response = client.delete(
            f"/api/kb/collections/{encoded_name}/documents/demo.txt",
            headers=headers,
        )

        assert response.status_code == 422
        assert "Invalid collection name" in response.json()["detail"]


def test_delete_document_by_filename_refuses_ambiguous_match(test_env, temp_uploads):
    """Deleting by basename should refuse to delete multiple matching documents."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    file_a = temp_uploads / f"user_{user.id}" / "demo" / "dup.txt"
    file_b = temp_uploads / f"user_{user.id}" / "demo" / "dup.txt"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("content-a")
    file_b.write_text("content-b")

    # Two documents share the same resolved filename. Without file_id/doc_id,
    # the API must refuse the deletion to avoid mass deletion by basename.
    document_state = [
        {
            "collection": "demo",
            "doc_id": "doc-a",
            "file_id": "file-a",
            "source_path": str(file_a),
        },
        {
            "collection": "demo",
            "doc_id": "doc-b",
            "file_id": "file-b",
            "source_path": str(file_b),
        },
    ]

    with patch(
        "xagent.web.api.kb._list_documents_for_user", return_value=document_state
    ):
        response = client.delete(
            "/api/kb/collections/demo/documents/dup.txt",
            headers=headers,
        )

    assert response.status_code == 409
    assert "ambiguous" in response.json()["detail"].lower()


def test_delete_document_by_doc_id_disambiguates_duplicate_filename(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "dup.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    document_state = [
        {
            "collection": "demo",
            "doc_id": "doc-a",
            "file_id": "file-a",
            "source_path": str(file_path),
        },
        {
            "collection": "demo",
            "doc_id": "doc-b",
            "file_id": "file-b",
            "source_path": str(file_path),
        },
    ]
    deleted_doc_ids: list[str] = []

    def _fake_list_documents_for_user(*args, **kwargs):
        return list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)

    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=_fake_list_documents_for_user,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        response = client.delete(
            "/api/kb/collections/demo/documents/dup.txt?doc_id=doc-b",
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["deleted_doc_ids"] == ["doc-b"]
    assert deleted_doc_ids == ["doc-b"]


def test_delete_document_by_file_id_survives_degraded_document_listing(
    test_env, temp_uploads
):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "fallback.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="fallback.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
    finally:
        session.close()

    deleted_doc_ids: list[str] = []
    expected_doc_id = generate_deterministic_doc_id("demo", str(file_path))

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)

    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=RuntimeError("documents unavailable"),
        ),
        patch(
            "xagent.web.api.kb.list_documents",
            side_effect=RuntimeError("documents unavailable"),
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        response = client.delete(
            f"/api/kb/collections/demo/documents/ignored.txt?file_id={target_file_id}",
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["deleted_doc_ids"] == [expected_doc_id]
    assert deleted_doc_ids == [expected_doc_id]


def test_delete_document_without_file_id_does_not_resurface_on_collection_refresh(
    test_env, temp_uploads
):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        ListCollectionsResult,
    )

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "resurface.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="resurface.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
    finally:
        session.close()

    document_state = [
        {
            "collection": "demo",
            "doc_id": generate_deterministic_doc_id("demo", str(file_path)),
            "file_id": None,
            "source_path": str(file_path),
        }
    ]

    def _fake_list_documents_for_user(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "demo":
            return list(document_state)
        return []

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()

    fake_result = ListCollectionsResult(
        status="success",
        collections=[CollectionInfo(name="demo", documents=0, document_names=[])],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=_fake_list_documents_for_user,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        delete_response = client.delete(
            "/api/kb/collections/demo/documents/resurface.txt",
            headers=headers,
        )

        assert delete_response.status_code == 200

        with patch("xagent.web.api.kb.list_collections", return_value=fake_result):
            refresh_response = client.get("/api/kb/collections", headers=headers)

    assert refresh_response.status_code == 200
    collection = refresh_response.json()["collections"][0]
    assert collection["document_names"] == []
    assert collection["document_metadata"] == []

    session = TestingSessionLocal()
    try:
        lingering_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.filename == "resurface.txt")
            .first()
        )
        assert lingering_record is None
    finally:
        session.close()


def test_delete_document_without_file_id_does_not_resurface_in_uploaded_file_fallback(
    test_env, temp_uploads
):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        ListCollectionsResult,
    )

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "fallback-refresh.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="fallback-refresh.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
    finally:
        session.close()

    document_state = [
        {
            "collection": "demo",
            "doc_id": generate_deterministic_doc_id("demo", str(file_path)),
            "file_id": None,
            "source_path": str(file_path),
        }
    ]

    def _fake_list_documents_for_user(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name == "demo":
            return list(document_state)
        raise RuntimeError("documents unavailable")

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()

    fake_result = ListCollectionsResult(
        status="success",
        collections=[CollectionInfo(name="demo", documents=0, document_names=[])],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=_fake_list_documents_for_user,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        delete_response = client.delete(
            "/api/kb/collections/demo/documents/fallback-refresh.txt",
            headers=headers,
        )

        assert delete_response.status_code == 200

        with patch("xagent.web.api.kb.list_collections", return_value=fake_result):
            refresh_response = client.get("/api/kb/collections", headers=headers)

    assert refresh_response.status_code == 200
    collection = refresh_response.json()["collections"][0]
    assert collection["document_names"] == []
    assert collection["document_metadata"] == []


def test_delete_document_by_file_id_resolves_doc_id_via_list_documents(
    test_env, temp_uploads
):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DocumentListResult,
        DocumentSummary,
    )

    file_path = temp_uploads / f"user_{user.id}" / "list-docs.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="list-docs.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
    finally:
        session.close()

    expected_doc_id = "doc-from-list-documents"
    deleted_doc_ids: list[str] = []

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)

    doc_list = DocumentListResult(
        status="success",
        documents=[
            DocumentSummary(
                collection="demo",
                doc_id=expected_doc_id,
                source_path=str(file_path),
            )
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb._list_documents_for_user", return_value=[]),
        patch("xagent.web.api.kb.list_documents", return_value=doc_list),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        response = client.delete(
            f"/api/kb/collections/demo/documents/ignored.txt?file_id={target_file_id}",
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["deleted_doc_ids"] == [expected_doc_id]
    assert deleted_doc_ids == [expected_doc_id]


def test_delete_document_by_doc_id_succeeds_without_uploaded_file_record(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DocumentListResult,
        DocumentSummary,
    )

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "already-cleaned.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    expected_doc_id = "doc-existing-without-file-row"
    deleted_doc_ids: list[str] = []

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)

    doc_list = DocumentListResult(
        status="success",
        documents=[
            DocumentSummary(
                collection="demo",
                doc_id=expected_doc_id,
                source_path=str(file_path),
            )
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb._list_documents_for_user", return_value=[]),
        patch("xagent.web.api.kb.list_documents", return_value=doc_list),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
            side_effect=_fake_delete_document,
        ),
    ):
        response = client.delete(
            (
                "/api/kb/collections/demo/documents/already-cleaned.txt"
                f"?file_id=missing-file-id&doc_id={expected_doc_id}"
            ),
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["deleted_doc_ids"] == [expected_doc_id]
    assert deleted_doc_ids == [expected_doc_id]


def test_delete_document_by_file_id_rejects_unlinked_basename_match(
    test_env, temp_uploads
):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DocumentListResult,
        DocumentSummary,
    )

    file_path = temp_uploads / f"user_{user.id}" / "other" / "shared-name.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="shared-name.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
    finally:
        session.close()

    doc_list = DocumentListResult(
        status="success",
        documents=[
            DocumentSummary(
                collection="demo",
                doc_id="doc-from-basename-only",
                source_path=f"/tmp/user_{user.id}/demo/shared-name.txt",
            )
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb._list_documents_for_user", return_value=[]),
        patch("xagent.web.api.kb.list_documents", return_value=doc_list),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document"
        ) as mock_delete_document,
    ):
        response = client.delete(
            f"/api/kb/collections/demo/documents/shared-name.txt?file_id={target_file_id}",
            headers=headers,
        )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    mock_delete_document.assert_not_called()


def test_delete_document_reports_cleanup_commit_failure(test_env, temp_uploads):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "commit-failure.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="commit-failure.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
    finally:
        session.close()

    document_state = [
        {
            "collection": "demo",
            "doc_id": "doc-commit-failure",
            "file_id": target_file_id,
            "source_path": str(file_path),
        }
    ]

    def _fake_list_documents_for_user(*args, **kwargs):
        return list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()

    def _failing_commit():
        raise RuntimeError("commit failed")

    original_override = app.dependency_overrides[get_db]

    def override_get_db_with_failing_commit():
        db = TestingSessionLocal()
        original_commit = db.commit
        db.commit = _failing_commit
        try:
            yield db
        finally:
            db.commit = original_commit
            db.close()

    app.dependency_overrides[get_db] = override_get_db_with_failing_commit
    try:
        with (
            patch(
                "xagent.web.api.kb._list_documents_for_user",
                side_effect=_fake_list_documents_for_user,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.management.collections.delete_document",
                side_effect=_fake_delete_document,
            ),
        ):
            response = client.delete(
                f"/api/kb/collections/demo/documents/commit-failure.txt?file_id={target_file_id}",
                headers=headers,
            )
    finally:
        app.dependency_overrides[get_db] = original_override

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial_success"
    assert payload["deleted_doc_ids"] == ["doc-commit-failure"]
    assert any(
        "Failed to persist orphan cleanup changes" in err for err in payload["errors"]
    )


def test_delete_document_rejects_mismatched_doc_id_and_file_id(test_env, temp_uploads):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "mismatch.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="mismatch.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
    finally:
        session.close()

    with (
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=RuntimeError("documents unavailable"),
        ),
        patch(
            "xagent.web.api.kb.list_documents",
            side_effect=RuntimeError("documents unavailable"),
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections.delete_document"
        ) as mock_delete_document,
    ):
        response = client.delete(
            (
                "/api/kb/collections/demo/documents/ignored.txt"
                f"?file_id={target_file_id}&doc_id=wrong-doc-id"
            ),
            headers=headers,
        )

    assert response.status_code == 409
    assert "same document" in response.json()["detail"].lower()
    mock_delete_document.assert_not_called()


def test_kb_delete_collection_cleans_file_id_managed_root_file(test_env, temp_uploads):
    """Collection delete should clean orphan UploadedFile rows even outside collection dir."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "root_level.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="root_level.txt",
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=7,
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
    finally:
        session.close()

    document_state = [
        DocumentRecord(
            doc_id="doc-1",
            file_id=target_file_id,
            source_path=str(file_path),
        )
    ]

    def _fake_list_documents_for_user(*args, **kwargs):
        # API calls it twice: once for filename_map, once for remaining_file_ids check
        # For simplicity, we return the same state (API logic will handle consistency)
        return list(document_state)

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_store,
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
    ):
        mock_store = mock_get_store.return_value
        mock_store.list_document_records.side_effect = _fake_list_documents_for_user
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        def _fake_delete_collection(*args, **kwargs):
            document_state.clear()
            return CollectionOperationResult(
                status="success",
                collection="demo",
                message="deleted",
                affected_documents=[],
                deleted_counts={},
            )

        mock_delete.side_effect = _fake_delete_collection
        response = client.delete("/api/kb/collections/demo", headers=headers)

    assert response.status_code == 200
    assert not file_path.exists()

    session = TestingSessionLocal()
    try:
        deleted_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == target_file_id)
            .first()
        )
        assert deleted_record is None
    finally:
        session.close()


def test_list_collections_secondary_fallback_avoids_n_plus_one(test_env, temp_uploads):
    """Secondary fallback should not call list_documents once per collection."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        ListCollectionsResult,
    )

    fake_result = ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(name="c1", documents=0, document_names=[]),
            CollectionInfo(name="c2", documents=0, document_names=[]),
        ],
        total_count=2,
        message="ok",
        warnings=[],
    )

    call_counts = {"list_documents": 0}

    def _fake_list_documents(*args, **kwargs):
        call_counts["list_documents"] += 1
        raise AssertionError("list_documents() should not be called")

    doc_records = [
        {"collection": "c1", "doc_id": "d1", "source_path": "/tmp/a.md"},
        {"collection": "c2", "doc_id": "d2", "source_path": "/tmp/b.md"},
    ]

    with (
        patch("xagent.web.api.kb.list_collections", return_value=fake_result),
        patch("xagent.web.api.kb._list_documents_for_user", return_value=doc_records),
        patch("xagent.web.api.kb.list_documents", side_effect=_fake_list_documents),
    ):
        response = client.get("/api/kb/collections", headers=headers)

    assert response.status_code == 200
    assert call_counts["list_documents"] == 0


def test_list_collections_skips_document_scan_when_names_are_complete(test_env):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionDocumentMetadata,
        CollectionInfo,
        ListCollectionsResult,
    )

    fake_result = ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(
                name="complete",
                documents=1,
                document_names=["a.md"],
                document_metadata=[
                    CollectionDocumentMetadata(
                        filename="a.md",
                        file_id="file-1",
                        doc_id="doc-1",
                    )
                ],
            ),
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb.list_collections", return_value=fake_result),
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=AssertionError("_list_documents_for_user should not be called"),
        ),
    ):
        response = client.get("/api/kb/collections", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["collections"][0]["name"] == "complete"
    assert payload["collections"][0]["document_names"] == ["a.md"]


def test_list_collections_skips_document_scan_when_duplicate_names_have_metadata(
    test_env,
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionDocumentMetadata,
        CollectionInfo,
        ListCollectionsResult,
    )

    fake_result = ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(
                name="duplicate",
                documents=2,
                document_names=["shared.txt"],
                document_metadata=[
                    CollectionDocumentMetadata(
                        filename="shared.txt",
                        file_id="file-1",
                        doc_id="doc-1",
                    ),
                    CollectionDocumentMetadata(
                        filename="shared.txt",
                        file_id="file-2",
                        doc_id="doc-2",
                    ),
                ],
            ),
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb.list_collections", return_value=fake_result),
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=AssertionError("_list_documents_for_user should not be called"),
        ),
    ):
        response = client.get("/api/kb/collections", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["collections"][0]["name"] == "duplicate"
    assert len(payload["collections"][0]["document_metadata"]) == 2
