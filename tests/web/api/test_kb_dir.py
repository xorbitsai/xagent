import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.file_storage.factory import get_file_storage
from xagent.core.tools.core.RAG_tools.core.config import DEFAULT_VECTOR_STORE_SCAN_LIMIT
from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    generate_deterministic_doc_id,
)
from xagent.web.api.auth import hash_password
from xagent.web.api.kb import kb_router
from xagent.web.models.database import Base, get_db
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import ManagedFileRef


@pytest.fixture(scope="function")
def test_env():
    """Setup test database and app"""
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)
    temp_lancedb_dir = tempfile.mkdtemp()

    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    previous_lancedb_dir = os.environ.get("LANCEDB_DIR")
    os.environ["LANCEDB_DIR"] = temp_lancedb_dir
    StorageFactory.get_factory().reset_all()

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

    yield app, headers, user, TestingSessionLocal

    session.close()
    test_engine.dispose()
    StorageFactory.get_factory().reset_all()
    if previous_lancedb_dir is None:
        os.environ.pop("LANCEDB_DIR", None)
    else:
        os.environ["LANCEDB_DIR"] = previous_lancedb_dir
    import shutil

    shutil.rmtree(temp_lancedb_dir, ignore_errors=True)
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
            collection_is_sanitized=False,
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


def _successful_delete_result(collection_name: str, doc_id: str):
    from xagent.core.tools.core.RAG_tools.core.schemas import DocumentOperationResult

    return DocumentOperationResult(
        status="success",
        collection=collection_name,
        doc_id=doc_id,
        new_status="failed",
        message="Document deleted successfully.",
        warnings=[],
        details={},
    )


def _make_delete_tracker():
    """Return (deleted_doc_ids, fake_delete_document) for delete assertions."""
    deleted_doc_ids: list[str] = []

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)
        return _successful_delete_result(collection_name, doc_id)

    return deleted_doc_ids, _fake_delete_document


def test_list_collection_uploaded_file_owner_ids_uses_strict_path_containment(
    test_env,
    temp_uploads,
):
    """UploadedFile owner discovery should not confuse similar collection paths."""
    _, _, user, TestingSessionLocal = test_env
    from xagent.web.services.kb_collection_service import (
        list_collection_uploaded_file_owner_ids,
    )

    session = TestingSessionLocal()
    try:
        matching_path = temp_uploads / f"user_{user.id}" / "FAQ" / "a.pdf"
        similar_path = temp_uploads / f"user_{user.id}" / "FAQ-old" / "b.pdf"
        unrelated_path = temp_uploads / f"user_{user.id}" / "Other" / "c.pdf"
        for path in (matching_path, similar_path, unrelated_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data")

        session.add_all(
            [
                UploadedFile(
                    file_id="match",
                    user_id=user.id,
                    filename="a.pdf",
                    storage_path=str(matching_path),
                    file_size=1,
                ),
                UploadedFile(
                    file_id="similar",
                    user_id=user.id,
                    filename="b.pdf",
                    storage_path=str(similar_path),
                    file_size=1,
                ),
                UploadedFile(
                    file_id="unrelated",
                    user_id=user.id,
                    filename="c.pdf",
                    storage_path=str(unrelated_path),
                    file_size=1,
                ),
            ]
        )
        session.commit()

        assert list_collection_uploaded_file_owner_ids(
            session, collection_name="FAQ"
        ) == {user.id}
        assert (
            list_collection_uploaded_file_owner_ids(session, collection_name="Missing")
            == set()
        )
    finally:
        session.close()


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


def test_kb_ingest_rolls_back_new_collection_on_partial_failure(test_env, temp_uploads):
    """Partial ingest failures should not leave a new KB file or UploadedFile row behind."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    collection_name = "rollback_new_collection"
    filename = "failed.xlsx"

    with patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest:
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        mock_ingest.return_value = IngestionResult(
            status="error",
            doc_id=None,
            parse_hash=None,
            completed_steps=[],
            failed_step="resolve_embedding_adapter",
            message=(
                "Model 'text-embedding-v4' not found in hub and no "
                "environment configuration available for embedding."
            ),
        )

        response = client.post(
            "/api/kb/ingest",
            files={"file": (filename, b"new content", "application/vnd.ms-excel")},
            data={"collection": collection_name},
            headers=headers,
        )

    assert response.status_code == 500
    expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
    assert not expected_path.exists()

    session = TestingSessionLocal()
    try:
        uploaded = session.query(UploadedFile).all()
        assert uploaded == []
    finally:
        session.close()


def test_kb_ingest_rolls_back_existing_file_content_on_partial_failure(
    test_env, temp_uploads
):
    """Partial ingest failures should restore overwritten files in existing KBs."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    collection_name = "existing_collection"
    filename = "existing.xlsx"
    expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text("old content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename=filename,
            storage_path=str(expected_path),
            mime_type="application/vnd.ms-excel",
            file_size=len("old content"),
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
    finally:
        session.close()
    with (
        patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
        patch("xagent.web.api.kb.clear_ingestion_status", return_value=None),
        patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest,
    ):
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            IngestionResult,
            IngestionStepResult,
        )

        mock_ingest.return_value = IngestionResult(
            status="partial",
            doc_id="doc-existing",
            parse_hash="parse-existing",
            completed_steps=[
                IngestionStepResult(name="initialize_collection"),
                IngestionStepResult(name="resolve_embedding_adapter"),
                IngestionStepResult(
                    name="register_document",
                    metadata={"doc_id": "doc-existing", "created": False},
                ),
            ],
            failed_step="compute_embeddings",
            message="embedding failed",
        )

        response = client.post(
            "/api/kb/ingest",
            files={"file": (filename, b"new content", "application/vnd.ms-excel")},
            data={"collection": collection_name},
            headers=headers,
        )

    assert response.status_code == 500
    assert expected_path.read_text() == "old content"

    session = TestingSessionLocal()
    try:
        uploaded = session.query(UploadedFile).all()
        assert len(uploaded) == 1
        assert uploaded[0].storage_path == str(expected_path)
    finally:
        session.close()


def test_kb_ingest_returns_explicit_error_when_rollback_fails(test_env, temp_uploads):
    """Direct ingest should surface rollback failures instead of hiding them in logs."""

    app, headers, user, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionOperationResult,
        IngestionResult,
    )

    with (
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest,
    ):
        mock_delete_collection.return_value = CollectionOperationResult(
            status="error",
            collection="rollback_new_collection",
            message="parse cleanup failed",
            affected_documents=[],
            deleted_counts={},
        )
        mock_ingest.return_value = IngestionResult(
            status="error",
            doc_id=None,
            parse_hash=None,
            completed_steps=[],
            failed_step="resolve_embedding_adapter",
            message=(
                "Model 'text-embedding-v4' not found in hub and no "
                "environment configuration available for embedding."
            ),
        )

        response = client.post(
            "/api/kb/ingest",
            files={"file": ("failed.xlsx", b"new content", "application/vnd.ms-excel")},
            data={"collection": "rollback_new_collection"},
            headers=headers,
        )

    assert response.status_code == 500
    assert "Failed to fully roll back ingest" in response.json()["detail"]
    assert "Original ingestion error:" in response.json()["detail"]
    assert "How to fix:" in response.json()["detail"]


def test_kb_ingest_returns_explicit_error_when_physical_rollback_fails(
    test_env, temp_uploads
):
    """Direct ingest should surface collection-directory rollback failures."""

    app, headers, user, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionOperationResult,
        IngestionResult,
        IngestionStepResult,
    )
    from xagent.web.services.kb_collection_service import CollectionPhysicalDeleteResult

    with (
        patch("xagent.web.api.kb.delete_collection") as mock_delete_collection,
        patch(
            "xagent.web.api.kb.delete_collection_physical_dir"
        ) as mock_physical_delete,
        patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest,
    ):
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="rollback_new_collection",
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )
        mock_physical_delete.return_value = CollectionPhysicalDeleteResult(
            status="failed",
            error="trash lock busy",
            collection_dir=temp_uploads / f"user_{user.id}" / "rollback_new_collection",
        )
        mock_ingest.return_value = IngestionResult(
            status="partial",
            doc_id="doc-failed",
            parse_hash="parse-failed",
            completed_steps=[
                IngestionStepResult(name="initialize_collection"),
                IngestionStepResult(name="resolve_embedding_adapter"),
                IngestionStepResult(
                    name="register_document",
                    metadata={"doc_id": "doc-failed", "created": True},
                ),
            ],
            failed_step="compute_embeddings",
            message="embedding failed",
        )

        response = client.post(
            "/api/kb/ingest",
            files={"file": ("failed.xlsx", b"new content", "application/vnd.ms-excel")},
            data={"collection": "rollback_new_collection"},
            headers=headers,
        )

    assert response.status_code == 500
    assert (
        "delete collection physical directory during rollback failed"
        in response.json()["detail"]
    )


def test_kb_ingest_surfaces_embedding_configuration_fix_guidance(
    test_env, temp_uploads
):
    """Direct ingest should explain why embedding resolution failed and how to fix it."""

    app, headers, _user, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

    with (
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            return_value=IngestionResult(
                status="error",
                doc_id=None,
                parse_hash=None,
                completed_steps=[],
                failed_step="resolve_embedding_adapter",
                message=(
                    "Model 'text-embedding-v4' not found in hub and no "
                    "environment configuration available for embedding."
                ),
            ),
        ),
        patch(
            "xagent.web.api.kb._rollback_failed_ingestion",
            new=AsyncMock(return_value=None),
        ),
    ):
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("failed.xlsx", b"new content", "application/vnd.ms-excel")},
            data={"collection": "embedding_config_missing"},
            headers=headers,
        )

    assert response.status_code == 500
    message = response.json()["message"]
    assert (
        "Cause: knowledge-base ingestion requires a resolvable embedding model"
        in message
    )
    assert "Current embedding_model_id: 'text-embedding-v4'." in message
    assert "How to fix:" in message
    assert "DASHSCOPE_EMBEDDING_MODEL" in message


@pytest.mark.asyncio
async def test_rollback_failed_ingestion_uses_cached_file_id_after_row_delete(
    test_env, temp_uploads
) -> None:
    """Rollback should not touch ORM attributes after UploadedFile cleanup deletes the row."""

    _, _, user, TestingSessionLocal = test_env

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionOperationResult,
        IngestionResult,
        IngestionStepResult,
    )
    from xagent.web.api import kb as kb_module
    from xagent.web.services.kb_collection_service import CollectionPhysicalDeleteResult

    collection_name = "rollback_deleted_uploaded_file"
    file_path = temp_uploads / f"user_{user.id}" / collection_name / "failed.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("new content")

    db = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            file_id="file-rollback-1",
            user_id=user.id,
            filename=file_path.name,
            storage_path=str(file_path),
            mime_type="text/plain",
            file_size=file_path.stat().st_size,
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)
        file_id = str(file_record.file_id)

        result = IngestionResult(
            status="partial",
            doc_id="doc-failed",
            parse_hash="parse-failed",
            completed_steps=[
                IngestionStepResult(
                    name="register_document",
                    metadata={"doc_id": "doc-failed", "created": True},
                )
            ],
            failed_step="compute_embeddings",
            message=(
                "Model 'text-embedding-v4' not found in hub and no environment "
                "configuration available for embedding."
            ),
        )
        mock_store = MagicMock()
        mock_store.list_document_records.side_effect = [
            [
                DocumentRecord(
                    doc_id="doc-failed",
                    file_id=file_id,
                    source_path=str(file_path),
                )
            ],
            [],
        ]
        mock_cleanup_metadata = AsyncMock()

        with (
            patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
            patch(
                "xagent.web.api.kb.delete_collection",
                return_value=CollectionOperationResult(
                    status="success",
                    collection=collection_name,
                    message="deleted",
                    affected_documents=[],
                    deleted_counts={},
                ),
            ),
            patch(
                "xagent.web.api.kb.delete_collection_physical_dir",
                return_value=CollectionPhysicalDeleteResult(
                    status="success",
                    collection_dir=file_path.parent,
                ),
            ),
            patch(
                "xagent.web.api.kb._cleanup_failed_new_collection_metadata",
                new=mock_cleanup_metadata,
            ),
        ):
            await kb_module._rollback_failed_ingestion(
                db=db,
                user=user,
                collection_name=collection_name,
                result=result,
                file_path=file_path,
                file_record=file_record,
                collection_existed_before=False,
                uploaded_file_existed_before=False,
                file_backup_path=None,
                had_existing_file=False,
            )

        remaining = (
            db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        )
        assert remaining is None
        assert not file_path.exists()
        mock_cleanup_metadata.assert_awaited_once_with(
            collection_name=collection_name,
            user=user,
        )
    finally:
        db.close()


def test_kb_ingest_surfaces_restore_failure_on_upload_abort(test_env, temp_uploads):
    """Upload aborts should return rollback failure if backup restore fails."""

    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "existing_collection"
    filename = "too-large.xlsx"
    expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text("old content")

    oversized = b"x" * (11 * 1024 * 1024)

    with patch(
        "xagent.web.api.kb._restore_ingest_file_backup",
        side_effect=OSError("restore exploded"),
    ):
        response = client.post(
            "/api/kb/ingest",
            files={
                "file": (
                    filename,
                    oversized,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            data={"collection": collection_name},
            headers=headers,
        )

    assert response.status_code == 500
    assert "Failed to fully roll back ingest" in response.json()["detail"]


def test_kb_ingest_restores_existing_file_on_upload_io_failure(test_env, temp_uploads):
    """Non-HTTP upload I/O failures should restore the previous file contents."""

    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "existing_collection"
    filename = "stream-failure.xlsx"
    expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text("old content")

    original_open = open

    def _failing_open(path, mode="r", *args, **kwargs):
        if Path(path) == expected_path and mode == "wb":
            raise OSError("stream exploded")
        return original_open(path, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=_failing_open):
        response = client.post(
            "/api/kb/ingest",
            files={
                "file": (
                    filename,
                    b"new content",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            data={"collection": collection_name},
            headers=headers,
        )

    assert response.status_code == 403
    assert "stream exploded" in response.json()["detail"]
    assert expected_path.read_text() == "old content"


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
    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
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


def test_kb_ingest_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "示例知识库集合"
    filename = "test_doc.txt"

    with patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest:
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        mock_ingest.return_value = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

        response = client.post(
            "/api/kb/ingest",
            files={"file": (filename, b"content", "text/plain")},
            data={"collection": collection_name},
            headers=headers,
        )

        assert response.status_code == 200
        expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
        assert expected_path.exists()


def test_kb_ingest_accepts_space_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "team notes"
    filename = "test_doc.txt"

    with patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest:
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        mock_ingest.return_value = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

        response = client.post(
            "/api/kb/ingest",
            files={"file": (filename, b"content", "text/plain")},
            data={"collection": collection_name},
            headers=headers,
        )

        assert response.status_code == 200
        expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
        assert expected_path.exists()


def test_kb_ingest_normalizes_padded_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "  team notes  "
    filename = "test_doc.txt"
    captured_collections: list[str] = []

    def _capture_ingest(*, collection=None, **kwargs):
        captured_collections.append(str(collection))
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
            files={"file": (filename, b"content", "text/plain")},
            data={"collection": collection_name},
            headers=headers,
        )

    assert response.status_code == 200
    assert captured_collections == ["team notes"]
    expected_path = temp_uploads / f"user_{user.id}" / "team notes" / filename
    assert expected_path.exists()


def test_kb_ingest_falls_back_from_pdf_only_parser_for_xlsx(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    captured_parse_methods: list[str] = []

    def _capture_ingest(**kwargs):
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        ingestion_config = kwargs["ingestion_config"]
        captured_parse_methods.append(str(ingestion_config.parse_method))
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
            files={
                "file": (
                    "table.xlsx",
                    b"content",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            data={"collection": "xlsx-demo", "parse_method": "pypdf"},
            headers=headers,
        )

    assert response.status_code == 200
    assert captured_parse_methods == ["default"]


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


def test_kb_ingest_accepts_derived_collection_name_with_spaces(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    filename = "file name.txt"

    with patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest:
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        mock_ingest.return_value = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

        response = client.post(
            "/api/kb/ingest",
            files={"file": (filename, b"content", "text/plain")},
            data={},
            headers=headers,
        )

        assert response.status_code == 200
        expected_path = temp_uploads / f"user_{user.id}" / "file name" / filename
        assert expected_path.exists()


def test_kb_delete_accepts_space_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "team notes"
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
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

        response = client.delete(
            f"/api/kb/collections/{quote(collection_name, safe='')}",
            headers=headers,
        )

    assert response.status_code == 200


def test_kb_delete_normalizes_padded_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "team notes"
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    deleted_collections: list[str] = []

    def _capture_delete(collection, user_id, is_admin):
        deleted_collections.append(str(collection))
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        return CollectionOperationResult(
            status="success",
            collection=collection,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.delete_collection", side_effect=_capture_delete),
    ):
        response = client.delete(
            f"/api/kb/collections/{quote('  team notes  ', safe='')}",
            headers=headers,
        )

    assert response.status_code == 200
    assert deleted_collections == [collection_name]


def test_kb_delete_rejects_path_traversal_in_collection_name(test_env, temp_uploads):
    """Test that delete_collection_api rejects path traversal in collection name."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    malicious_collections = [
        r"collection\\other",
        r"collection\\..\\other",
    ]

    for collection_name in malicious_collections:
        encoded_name = quote(collection_name, safe="")
        response = client.delete(f"/api/kb/collections/{encoded_name}", headers=headers)

        assert response.status_code == 422
        assert "Invalid collection name" in response.json()["detail"]


def test_kb_delete_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "示例知识库集合"
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
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

        response = client.delete(
            f"/api/kb/collections/{quote(collection_name, safe='')}",
            headers=headers,
        )

    assert response.status_code == 200


def test_kb_delete_removes_empty_metadata_only_collection_from_listing(
    test_env, temp_uploads
):
    """Deleting a collection should remove empty metadata-only collections too."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    import asyncio
    import uuid

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    collection_name = f"empty_collection_{uuid.uuid4().hex[:8]}"

    asyncio.run(
        get_metadata_store().save_collection(CollectionInfo(name=collection_name))
    )
    asyncio.run(
        get_metadata_store().save_collection_config(
            collection_name,
            "{}",
            user_id=int(user.id),
        )
    )

    list_before = client.get("/api/kb/collections", headers=headers)
    assert list_before.status_code == 200
    assert any(
        item["name"] == collection_name for item in list_before.json()["collections"]
    )

    delete_response = client.delete(
        f"/api/kb/collections/{collection_name}",
        headers=headers,
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "success"


def test_kb_delete_rejects_mixed_script_confusable_collection_name(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "cоllection"
    response = client.delete(
        f"/api/kb/collections/{quote(collection_name, safe='')}",
        headers=headers,
    )

    assert response.status_code == 422
    assert "Invalid collection name" in response.json()["detail"]


def test_kb_ingest_rejects_utf8_byte_overflow_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "知" * 86
    filename = "test_doc.txt"

    with patch("xagent.web.api.kb.run_document_ingestion"):
        response = client.post(
            "/api/kb/ingest",
            files={"file": (filename, b"content", "text/plain")},
            data={"collection": collection_name},
            headers=headers,
        )

    assert response.status_code == 422
    assert "maximum byte length" in response.json()["detail"]


def test_save_collection_config_normalizes_padded_collection_name(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    from unittest.mock import AsyncMock, MagicMock

    mock_store = MagicMock()
    mock_store.save_collection_config = AsyncMock()

    with patch(
        "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
        return_value=mock_store,
    ):
        response = client.post(
            "/api/kb/collections/%20%20team%20notes%20%20/config",
            json={},
            headers=headers,
        )

    assert response.status_code == 200
    mock_store.save_collection_config.assert_awaited_once_with(
        collection="team notes",
        config_json="{}",
        user_id=int(user.id),
    )
    assert response.json()["collection"] == "team notes"


def test_kb_search_normalizes_padded_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    captured_collections: list[str] = []
    captured_access_check_collections: list[str] = []

    def _capture_search(*, collection=None, **kwargs):
        captured_collections.append(str(collection))
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            SearchPipelineResult,
            SearchType,
        )

        return SearchPipelineResult(
            status="success",
            search_type=SearchType.HYBRID,
            results=[],
            result_count=0,
            warnings=[],
            message="ok",
            used_rerank=False,
        )

    async def _capture_access_check(collection, _user, hide_missing=True):
        # Fail immediately if the unsanitized collection name is passed
        assert collection == "team notes", (
            f"Expected sanitized 'team notes', got '{collection}'"
        )
        captured_access_check_collections.append(str(collection))
        return

    with patch("xagent.web.api.kb.run_document_search", side_effect=_capture_search):
        with patch(
            "xagent.web.api.kb._ensure_collection_access",
            side_effect=_capture_access_check,
        ):
            response = client.post(
                "/api/kb/search",
                data={
                    "collection": "  team notes  ",
                    "query_text": "hello",
                    "embedding_model_id": "text-embedding-v4",
                },
                headers=headers,
            )

    assert response.status_code == 200
    # Verify the sanitized name was passed to search
    assert captured_collections == ["team notes"]
    # Verify the sanitized name (not raw input) was passed to access control
    assert captured_access_check_collections == ["team notes"]


@pytest.mark.asyncio
async def test_ensure_collection_access_hide_missing_uses_single_visible_listing():
    """hide_missing=True should not trigger a second global list_collections call."""
    from fastapi import HTTPException

    from xagent.core.tools.core.RAG_tools.core.schemas import ListCollectionsResult
    from xagent.web.api.kb import _ensure_collection_access

    user = MagicMock()
    user.id = 1
    user.is_admin = False

    visible_only = ListCollectionsResult(
        status="success",
        collections=[],
        total_count=0,
        message="ok",
        warnings=[],
    )

    with patch(
        "xagent.web.api.kb._list_collections_with_retry",
        new_callable=AsyncMock,
        return_value=visible_only,
    ) as mock_list:
        with pytest.raises(HTTPException) as exc_info:
            await _ensure_collection_access(
                "missing-or-forbidden", user, hide_missing=True
            )

    assert exc_info.value.status_code == 403
    assert "Access denied for collection" in str(exc_info.value.detail)
    mock_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_collection_access_returns_404_when_collection_absent_globally():
    """When collection metadata does not exist globally, access check should return 404."""
    from types import SimpleNamespace

    from fastapi import HTTPException

    from xagent.web.api.kb import _ensure_collection_access

    user = MagicMock()
    user.id = 1
    user.is_admin = False

    visible = SimpleNamespace(collections=[])
    all_collections = SimpleNamespace(collections=[])

    with patch(
        "xagent.web.api.kb._list_collections_with_retry",
        new_callable=AsyncMock,
        side_effect=[visible, all_collections],
    ) as mock_list:
        with pytest.raises(HTTPException) as exc_info:
            await _ensure_collection_access("team-a", user, hide_missing=False)

    assert exc_info.value.status_code == 404
    assert "Collection not found" in str(exc_info.value.detail)
    assert mock_list.await_count == 2


def test_kb_delete_physical_cleanup_failure_preserves_uploaded_file_records(
    test_env, temp_uploads
):
    """Physical cleanup failure should preserve UploadedFile rows for retry."""
    app, headers, user, session_local = test_env
    client = TestClient(app)
    from xagent.web.services.kb_collection_service import CollectionPhysicalDeleteResult

    collection_name = "kb_to_delete_fail"

    # Pre-create the collection directory
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    file_path = coll_dir / "some_file.txt"
    file_path.write_text("data")

    db = session_local()
    upload = UploadedFile(
        user_id=user.id,
        filename=file_path.name,
        storage_path=str(file_path),
        mime_type="text/plain",
        file_size=file_path.stat().st_size,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    file_id = upload.file_id
    db.close()

    with (
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_vector_store,
        patch(
            "xagent.web.api.kb.delete_collection_physical_dir"
        ) as mock_physical_delete,
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
        mock_get_vector_store.return_value.list_document_records.side_effect = [
            [
                DocumentRecord(
                    doc_id="doc-1", file_id=file_id, source_path=str(file_path)
                )
            ],
            [],
        ]
        mock_physical_delete.return_value = CollectionPhysicalDeleteResult(
            status="failed",
            error="Permission denied",
            collection_dir=coll_dir,
        )
        mock_move_to_trash.side_effect = PermissionError("Permission denied")

        response = client.delete(
            f"/api/kb/collections/{collection_name}", headers=headers
        )

    assert response.status_code == 200
    assert response.json()["status"] == "partial_success"

    # Physical file remains, so SQL metadata must remain for retry cleanup.
    db = session_local()
    remaining = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
    db.close()
    assert remaining is not None
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
    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
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


def test_kb_delete_skips_uploaded_file_cleanup_when_logical_delete_fails(
    test_env, temp_uploads
):
    """API should not delete UploadedFile rows/files when collection delete returns error."""

    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_delete_error"
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    with (
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
        patch("xagent.web.api.kb.delete_collection_uploaded_files") as mock_cleanup,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_get_store,
    ):
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="error",
            collection=collection_name,
            message="vector cleanup failed",
            affected_documents=[],
            deleted_counts={},
        )
        mock_get_store.return_value.list_document_records.return_value = []

        response = client.delete(
            f"/api/kb/collections/{collection_name}",
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    mock_cleanup.assert_not_called()


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

    # Mock database operations to avoid schema errors
    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
    ):
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
            # Malicious names in path cause routing to fail with 404
            assert response.status_code == 404

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

    # Mock the database and vector store operations
    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections._list_table_names"
        ) as mock_list_tables,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
        patch("xagent.web.api.kb.rename_collection_storage") as mock_rename_storage,
        patch("xagent.web.api.kb.get_ingestion_status_store") as mock_get_status_store,
    ):
        mock_list_tables.return_value = []

        # Mock vector store operations
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = []  # No documents
        mock_store.rename_collection_data.return_value = []  # No warnings
        mock_store_factory.return_value = mock_store

        # Mock rename_collection_storage to simulate success
        mock_rename_result = MagicMock()
        mock_rename_result.status = "not_found"  # No physical dir found
        mock_rename_result.error = None
        mock_rename_result.old_collection_dir = None
        mock_rename_result.new_collection_dir = None
        mock_rename_storage.return_value = mock_rename_result

        # Mock ingestion status operations
        mock_get_status_store.return_value.rename_collection_status.return_value = []

        # Attempt rename
        response = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

        # Should succeed
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )


def test_kb_rename_normalizes_padded_collection_names(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "team notes"
    new_collection_name = "  project archive  "

    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "some_file.txt").write_text("data")

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections._list_table_names"
        ) as mock_list_tables,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
        patch("xagent.web.api.kb.rename_collection_storage") as mock_rename_storage,
        patch("xagent.web.api.kb.get_ingestion_status_store") as mock_get_status_store,
    ):
        mock_list_tables.return_value = []

        # Mock vector store operations
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = []
        mock_store.rename_collection_data.return_value = []
        mock_store_factory.return_value = mock_store

        # Mock rename_collection_storage
        mock_rename_result = MagicMock()
        mock_rename_result.status = "not_found"
        mock_rename_result.error = None
        mock_rename_result.old_collection_dir = None
        mock_rename_result.new_collection_dir = None
        mock_rename_storage.return_value = mock_rename_result

        mock_get_status_store.return_value.rename_collection_status.return_value = []

        response = client.put(
            "/api/kb/collections/%20%20team%20notes%20%20",
            data={"new_name": new_collection_name},
            headers=headers,
        )

    # Should succeed and normalize the padded names
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    result = response.json()
    # When physical dir is not_found, API returns partial_success
    assert result["status"] == "partial_success", (
        f"Expected partial_success, got {result['status']}"
    )


def test_kb_rename_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "示例知识库集合"
    new_collection_name = "知识库归档"

    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "some_file.txt").write_text("data")

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections._list_table_names"
        ) as mock_list_tables,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
        patch("xagent.web.api.kb.rename_collection_storage") as mock_rename_storage,
        patch("xagent.web.api.kb.get_ingestion_status_store") as mock_get_status_store,
    ):
        mock_list_tables.return_value = []

        # Mock vector store operations
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = []
        mock_store.rename_collection_data.return_value = []
        mock_store_factory.return_value = mock_store

        # Mock rename_collection_storage
        mock_rename_result = MagicMock()
        mock_rename_result.status = "not_found"
        mock_rename_result.error = None
        mock_rename_result.old_collection_dir = None
        mock_rename_result.new_collection_dir = None
        mock_rename_storage.return_value = mock_rename_result

        mock_get_status_store.return_value.rename_collection_status.return_value = []

        response = client.put(
            f"/api/kb/collections/{quote(old_collection_name, safe='')}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

    # Should succeed with unicode collection names
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    result = response.json()
    assert result["status"] == "partial_success", (
        f"Expected partial_success, got {result['status']}"
    )


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
    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
    ):
        # Mock vector store operations
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = []
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
            assert response.status_code == 500, (
                f"Expected 500, got {response.status_code}: {response.text}"
            )
            detail = response.json()["detail"].lower()
            assert (
                "cannot rename physical directory" in detail
                or "failed to rename" in detail
                or "physical directory rename" in detail
            ), f"Expected physical rename error message, got: {detail}"

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
    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
        patch("xagent.web.api.kb.rename_collection_storage") as mock_rename_storage,
    ):
        # Mock vector store operations
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = []
        mock_store_factory.return_value = mock_store

        # Mock rename_collection_storage to simulate target directory conflict
        mock_rename_result = MagicMock()
        mock_rename_result.status = "failed"
        mock_rename_result.error = (
            "Another operation is in progress; please try again later."
        )
        mock_rename_storage.return_value = mock_rename_result

        # Attempt rename to existing directory
        response = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

        # Should fail with 409 (conflict)
        assert response.status_code == 409, (
            f"Expected 409, got {response.status_code}: {response.text}"
        )
        detail = response.json()["detail"]
        assert "already exists" in detail or "in progress" in detail, (
            f"Expected conflict error, got: {detail}"
        )


def test_kb_rename_rejects_existing_visible_target_collection(test_env):
    """Rename must reject when target collection already exists and is visible."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "rename_source"
    new_collection_name = "rename_target_existing"

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        ListCollectionsResult,
    )

    visible_for_user = ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(name=old_collection_name, documents=1, document_names=[]),
            CollectionInfo(name=new_collection_name, documents=1, document_names=[]),
        ],
        total_count=2,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb._list_collections_with_retry",
            new_callable=AsyncMock,
            return_value=visible_for_user,
        ),
        patch("xagent.web.api.kb.rename_collection_storage") as mock_rename_storage,
    ):
        response = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

    assert response.status_code == 409, (
        f"Expected 409 when target exists, got {response.status_code}: {response.text}"
    )
    mock_rename_storage.assert_not_called()


def test_delete_after_rename_not_denied_by_stale_list_collections(test_env):
    """When access passes, stale ``list_collections`` alone does not block delete.

    The API still enforces ``_ensure_collection_access`` (403 for true cross-tenant).
    This scenario simulates resolved access while ``list_collections`` remains stale;
    the gate is patched for the delete phase only.
    """
    app, headers, _, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        DocumentOperationResult,
        DocumentProcessingStatus,
        ListCollectionsResult,
    )

    old_collection_name = "rename_access_old"
    new_collection_name = "rename_access_new"

    stale_old_only = ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(name=old_collection_name, documents=1, document_names=[])
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb._list_collections_with_retry") as mock_retry,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
        patch("xagent.web.api.kb.rename_collection_storage") as mock_rename_storage,
        patch("xagent.web.api.kb.get_ingestion_status_store") as mock_get_status_store,
    ):
        mock_retry.side_effect = [stale_old_only, stale_old_only]
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = []
        mock_store.rename_collection_data.return_value = []
        mock_store_factory.return_value = mock_store
        mock_rename_result = MagicMock()
        mock_rename_result.status = "not_found"
        mock_rename_result.error = None
        mock_rename_result.old_collection_dir = None
        mock_rename_result.new_collection_dir = None
        mock_rename_storage.return_value = mock_rename_result
        mock_get_status_store.return_value.rename_collection_status.return_value = []

        rename_resp = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

    assert rename_resp.status_code == 200, rename_resp.text

    doc_match = DocumentRecord(
        doc_id="doc-stale",
        file_id="fake-file-id",
        source_path="/tmp/demo.txt",
    )
    delete_ok = DocumentOperationResult(
        status="success",
        collection=new_collection_name,
        doc_id="doc-stale",
        new_status=DocumentProcessingStatus.FAILED,
        message="Document deleted successfully.",
    )

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.list_collections", return_value=stale_old_only),
        patch("xagent.web.api.kb.get_vector_index_store") as mock_vs,
        patch("xagent.web.api.kb.delete_document", return_value=delete_ok),
    ):
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = [doc_match]
        mock_vs.return_value = mock_store
        delete_resp = client.delete(
            f"/api/kb/collections/{new_collection_name}/documents/demo.txt?file_id=fake-file-id",
            headers=headers,
        )

    assert delete_resp.status_code != 403
    assert (
        f"Access denied for collection: {new_collection_name}" not in delete_resp.text
    )


def test_delete_after_rename_not_blocked_when_new_collection_is_visible(test_env):
    """Diagnostic control: delete is not blocked by access-check when list_collections shows the renamed collection."""
    app, headers, _, _ = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        ListCollectionsResult,
    )

    old_collection_name = "rename_access_old2"
    new_collection_name = "rename_access_new2"

    visible_old = ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(name=old_collection_name, documents=1, document_names=[])
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )
    visible_new = ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(name=new_collection_name, documents=1, document_names=[])
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb._list_collections_with_retry") as mock_retry,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_store_factory,
        patch("xagent.web.api.kb.rename_collection_storage") as mock_rename_storage,
        patch("xagent.web.api.kb.get_ingestion_status_store") as mock_get_status_store,
    ):
        mock_retry.side_effect = [visible_old, visible_old]
        mock_store = MagicMock()
        mock_store.list_document_records.return_value = []
        mock_store.rename_collection_data.return_value = []
        mock_store_factory.return_value = mock_store
        mock_rename_result = MagicMock()
        mock_rename_result.status = "not_found"
        mock_rename_result.error = None
        mock_rename_result.old_collection_dir = None
        mock_rename_result.new_collection_dir = None
        mock_rename_storage.return_value = mock_rename_result
        mock_get_status_store.return_value.rename_collection_status.return_value = []

        rename_resp = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

    assert rename_resp.status_code == 200, rename_resp.text

    # 对照：可见性切换为新集合后，不应再是 access-check 的 403
    with patch("xagent.web.api.kb.list_collections", return_value=visible_new):
        delete_resp = client.delete(
            f"/api/kb/collections/{new_collection_name}/documents/demo.txt?file_id=fake-file-id",
            headers=headers,
        )

    assert delete_resp.status_code != 403


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


def test_kb_ingest_setup_failure_cleans_new_collection_config(test_env, temp_uploads):
    """Setup failures after config save should not leave config rows for new collections."""
    app, headers, _user, _ = test_env
    client = TestClient(app, raise_server_exceptions=False)
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock(
        return_value={"metadata_rows": 0, "config_rows": 1}
    )

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb._upsert_uploaded_file_record",
            side_effect=RuntimeError("boom"),
        ),
    ):
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("test_doc.txt", b"content", "text/plain")},
            data={"collection": "new_collection"},
            headers=headers,
        )

    assert response.status_code == 500
    metadata_store.delete_collection_metadata.assert_awaited_once_with(
        collection_name="new_collection",
        user_id=1,
        is_admin=False,
        delete_orphaned_metadata=True,
    )


def test_kb_ingest_existing_collection_failure_restores_previous_config(
    test_env, temp_uploads
) -> None:
    """Failed direct ingest should not replace an existing collection config."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

    app, headers, _user, _ = test_env
    client = TestClient(app, raise_server_exceptions=False)
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":111}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()
    metadata_store.delete_collection = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_collection_sync", return_value=MagicMock()),
        patch(
            "xagent.web.api.kb._upsert_uploaded_file_record",
            return_value=MagicMock(file_id="file-1"),
        ),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            return_value=IngestionResult(
                status="error",
                doc_id="doc-1",
                parse_hash="",
                failed_step="parse_document",
                message="ingestion failed",
            ),
        ),
        patch("xagent.web.api.kb._rollback_failed_ingestion", new_callable=AsyncMock),
    ):
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("test_doc.txt", b"content", "text/plain")},
            data={"collection": "existing_collection", "chunk_size": "2048"},
            headers=headers,
        )

    assert response.status_code == 500
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "existing_collection",
        "config_json": '{"chunk_size":111}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()


def test_kb_ingest_config_only_collection_failure_restores_previous_config(
    test_env, temp_uploads
) -> None:
    """A saved config without collection metadata is still pre-existing state."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

    app, headers, _user, _ = test_env
    client = TestClient(app, raise_server_exceptions=False)
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":111}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()
    metadata_store.delete_collection = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb._upsert_uploaded_file_record",
            return_value=MagicMock(file_id="file-1"),
        ),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            return_value=IngestionResult(
                status="error",
                doc_id="doc-1",
                parse_hash="",
                failed_step="parse_document",
                message="ingestion failed",
            ),
        ),
        patch("xagent.web.api.kb._rollback_failed_ingestion", new_callable=AsyncMock),
    ):
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("test_doc.txt", b"content", "text/plain")},
            data={"collection": "config_only_collection", "chunk_size": "2048"},
            headers=headers,
        )

    assert response.status_code == 500
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "config_only_collection",
        "config_json": '{"chunk_size":111}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()
    metadata_store.delete_collection.assert_awaited_once_with("config_only_collection")


@pytest.mark.asyncio
async def test_failed_ingest_config_restore_skips_delete_when_snapshot_unknown() -> (
    None
):
    """Unknown prior config state should not be treated as an empty old config."""
    from xagent.web.api.kb import (
        _CollectionConfigSnapshot,
        _restore_collection_config_after_failed_ingest,
    )

    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()

    await _restore_collection_config_after_failed_ingest(
        snapshot=_CollectionConfigSnapshot(
            metadata_store=metadata_store,
            collection="existing_collection",
            user_id=1,
            previous_config_json=None,
            previous_config_known=False,
            saved=True,
        ),
        collection_existed_before=True,
        context="unit-test",
    )

    metadata_store.save_collection_config.assert_not_awaited()
    metadata_store.delete_collection_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_collection_config_snapshot_failure_does_not_save_new_config() -> None:
    """If the old config cannot be read, failed ingests must not leave a new config."""
    from xagent.web.api.kb import _save_collection_config_with_snapshot

    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(side_effect=RuntimeError("boom"))
    metadata_store.save_collection_config = AsyncMock()
    user = MagicMock(id=1)

    with patch(
        "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
        return_value=metadata_store,
    ):
        snapshot = await _save_collection_config_with_snapshot(
            collection="existing_collection",
            config_json='{"chunk_size":2048}',
            user=user,
            context="unit-test",
        )

    assert snapshot.saved is False
    assert snapshot.previous_config_known is False
    metadata_store.save_collection_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_kb_ingest_cloud_rollback_passes_admin_scope() -> None:
    """Local rollback should clear ingestion status with the caller's admin scope."""

    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
    from xagent.web.api import kb as kb_module

    result = IngestionResult(
        status="partial",
        doc_id="doc-1",
        parse_hash="",
        failed_step="parse_document",
        message="partial failure",
    )
    user = MagicMock(id=7, is_admin=True)
    db = MagicMock()

    with (
        patch("xagent.web.api.kb.get_vector_index_store", return_value=MagicMock()),
        patch("xagent.web.api.kb.clear_ingestion_status") as mock_clear_status,
        patch("xagent.web.api.kb._restore_ingest_file_backup"),
    ):
        await kb_module._rollback_failed_ingestion(
            db=db,
            user=user,
            collection_name="cloud-coll",
            result=result,
            file_path=Path("cloud.txt"),
            file_record=MagicMock(file_id="file-1"),
            collection_existed_before=True,
            uploaded_file_existed_before=True,
            file_backup_path=None,
            had_existing_file=False,
        )

    mock_clear_status.assert_called_once_with(
        "cloud-coll",
        "doc-1",
        user_id=7,
        is_admin=True,
    )


def test_kb_ingest_cloud_all_failures_clean_new_collection_config(test_env) -> None:
    """Cloud ingest should remove saved config when a brand-new collection never ingests anything."""
    app, headers, _user, _ = test_env
    client = TestClient(app)
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value=None)
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock(
        return_value={"metadata_rows": 0, "config_rows": 1}
    )

    with patch(
        "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
        return_value=metadata_store,
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_new_collection",
                "files": [
                    {
                        "provider": "unsupported",
                        "fileId": "file-1",
                        "fileName": "doc.txt",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()[0]["status"] == "error"
    metadata_store.delete_collection_metadata.assert_awaited_once_with(
        collection_name="cloud_new_collection",
        user_id=1,
        is_admin=False,
        delete_orphaned_metadata=True,
    )


def test_kb_ingest_cloud_existing_collection_all_failures_restores_config(
    test_env,
) -> None:
    """All-failed cloud ingest should restore config for existing collections."""
    app, headers, _user, _ = test_env
    client = TestClient(app)
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":222}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_collection_sync", return_value=MagicMock()),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_existing_collection",
                "chunk_size": 2048,
                "files": [
                    {
                        "provider": "unsupported",
                        "fileId": "file-1",
                        "fileName": "doc.txt",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()[0]["status"] == "error"
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "cloud_existing_collection",
        "config_json": '{"chunk_size":222}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()


def test_kb_ingest_cloud_config_only_collection_all_failures_restores_config(
    test_env,
) -> None:
    """Cloud ingest must restore config-only collection state on failure."""
    app, headers, _user, _ = test_env
    client = TestClient(app)
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":333}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()
    metadata_store.delete_collection = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_config_only_collection",
                "chunk_size": 2048,
                "files": [
                    {
                        "provider": "unsupported",
                        "fileId": "file-1",
                        "fileName": "doc.txt",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()[0]["status"] == "error"
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "cloud_config_only_collection",
        "config_json": '{"chunk_size":333}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()
    metadata_store.delete_collection.assert_awaited_once_with(
        "cloud_config_only_collection"
    )


def test_kb_ingest_cloud_existing_collection_mixed_failure_restores_config(
    test_env, temp_uploads
) -> None:
    """A mixed cloud ingest should not commit new config for an existing collection."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

    app, headers, _user, _ = test_env
    client = TestClient(app)
    metadata_store = MagicMock()
    metadata_store.get_collection_config = AsyncMock(return_value='{"chunk_size":444}')
    metadata_store.save_collection_config = AsyncMock()
    metadata_store.delete_collection_metadata = AsyncMock()

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

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_collection_sync", return_value=MagicMock()),
        patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=_FakeDriveService()),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _FakeDownloader),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            return_value=IngestionResult(
                status="success",
                doc_id="cloud-doc-id",
                parse_hash="hash",
                message="success",
            ),
        ),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_existing_collection",
                "chunk_size": 2048,
                "files": [
                    {
                        "provider": "google-drive",
                        "fileId": "drive-file-1",
                        "fileName": "doc.txt",
                    },
                    {
                        "provider": "unsupported",
                        "fileId": "file-2",
                        "fileName": "bad.txt",
                    },
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert [item["status"] for item in response.json()] == ["success", "error"]
    assert metadata_store.save_collection_config.await_count == 2
    restore_call = metadata_store.save_collection_config.await_args_list[-1]
    assert restore_call.kwargs == {
        "collection": "cloud_existing_collection",
        "config_json": '{"chunk_size":444}',
        "user_id": 1,
    }
    metadata_store.delete_collection_metadata.assert_not_awaited()


def test_kb_ingest_cloud_denied_request_does_not_persist_collection_config(
    test_env,
) -> None:
    from fastapi import HTTPException

    app, headers, _user, _ = test_env
    client = TestClient(app, raise_server_exceptions=False)
    metadata_store = MagicMock()
    metadata_store.save_collection_config = AsyncMock()

    with (
        patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store",
            return_value=metadata_store,
        ),
        patch(
            "xagent.web.api.kb._ensure_collection_access",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=403, detail="Forbidden"),
        ),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "shared_collection",
                "files": [
                    {
                        "provider": "unsupported",
                        "fileId": "file-1",
                        "fileName": "doc.txt",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 403
    metadata_store.save_collection_config.assert_not_awaited()


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


def test_kb_ingest_cloud_normalizes_parser_and_rolls_back_partial_failure(
    test_env, temp_uploads
):
    """Cloud ingest should normalize parser choice and invoke rollback on partial failures."""

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        IngestionResult,
        IngestionStepResult,
        ParseMethod,
    )

    app, headers, user, _ = test_env
    client = TestClient(app)
    captured_parse_methods = []

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

    def _capture_ingest(*, ingestion_config=None, **kwargs):
        assert ingestion_config is not None
        captured_parse_methods.append(ingestion_config.parse_method)
        return IngestionResult(
            status="partial",
            doc_id="cloud-doc-id",
            parse_hash="hash",
            completed_steps=[
                IngestionStepResult(
                    name="register_document",
                    metadata={"doc_id": "cloud-doc-id", "created": True},
                )
            ],
            failed_step="parse_document",
            message="partial failure",
        )

    with (
        patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=_FakeDriveService()),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _FakeDownloader),
        patch(
            "xagent.web.api.kb.get_collection_sync",
            side_effect=ValueError("missing collection"),
        ),
        patch("xagent.web.api.kb.run_document_ingestion", side_effect=_capture_ingest),
        patch("xagent.web.api.kb._rollback_failed_cloud_ingestion") as mock_rollback,
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_coll",
                "parse_method": ParseMethod.PYPDF.value,
                "files": [
                    {
                        "provider": "google-drive",
                        "fileId": "drive-file-1",
                        "fileName": "cloud.csv",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert captured_parse_methods == [ParseMethod.DEFAULT]
    mock_rollback.assert_called_once()


def test_kb_ingest_cloud_returns_rollback_failure_message(test_env, temp_uploads):
    """Cloud ingest should return explicit rollback failure messages per file."""

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        IngestionResult,
        IngestionStepResult,
    )
    from xagent.web.api.kb import RollbackFailureError

    app, headers, user, _ = test_env
    client = TestClient(app)

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

    with (
        patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=_FakeDriveService()),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _FakeDownloader),
        patch(
            "xagent.web.api.kb.get_collection_sync",
            side_effect=ValueError("missing collection"),
        ),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            return_value=IngestionResult(
                status="partial",
                doc_id="cloud-doc-id",
                parse_hash="hash",
                completed_steps=[
                    IngestionStepResult(
                        name="register_document",
                        metadata={"doc_id": "cloud-doc-id", "created": True},
                    )
                ],
                failed_step="parse_document",
                message="partial failure",
            ),
        ),
        patch(
            "xagent.web.api.kb._rollback_failed_cloud_ingestion",
            side_effect=RollbackFailureError("cloud rollback failed"),
        ),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_coll",
                "files": [
                    {
                        "provider": "google-drive",
                        "fileId": "drive-file-1",
                        "fileName": "cloud.csv",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data[0]["status"] == "error"
    assert "cloud rollback failed" in data[0]["message"]


def test_kb_ingest_cloud_surfaces_restore_failure_on_download_error(
    test_env, temp_uploads
):
    """Cloud download failures should surface backup-restore errors per file."""

    app, headers, user, _ = test_env
    client = TestClient(app)

    class _FakeFilesService:
        def get_media(self, fileId: str):
            return {"fileId": fileId}

    class _FakeDriveService:
        def files(self):
            return _FakeFilesService()

    class _FailingDownloader:
        def __init__(self, fh, request_file):
            self._fh = fh

        def next_chunk(self):
            raise RuntimeError("download blew up")

    with (
        patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=_FakeDriveService()),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _FailingDownloader),
        patch(
            "xagent.web.api.kb._restore_ingest_file_backup",
            side_effect=OSError("restore exploded"),
        ),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_coll",
                "files": [
                    {
                        "provider": "google-drive",
                        "fileId": "drive-file-1",
                        "fileName": "cloud.csv",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data[0]["status"] == "error"
    assert "Failed to fully roll back cloud ingest" in data[0]["message"]


def test_kb_ingest_cloud_surfaces_restore_failure_on_unexpected_error(
    test_env, temp_uploads
):
    """Cloud unexpected errors should also surface restore failures per file."""

    app, headers, user, _ = test_env
    client = TestClient(app)

    class _FakeFilesService:
        def get_media(self, fileId: str):
            return {"fileId": fileId}

    class _FakeDriveService:
        def files(self):
            return _FakeFilesService()

    class _DownloaderWithSuccess:
        def __init__(self, fh, request_file):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"partial")
            return None, True

    with (
        patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=_FakeDriveService()),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _DownloaderWithSuccess),
        patch(
            "xagent.web.api.kb._upsert_uploaded_file_record",
            side_effect=ValueError("unexpected post-download failure"),
        ),
        patch(
            "xagent.web.api.kb._restore_ingest_file_backup",
            side_effect=OSError("restore exploded"),
        ),
    ):
        response = client.post(
            "/api/kb/ingest-cloud",
            json={
                "collection": "cloud_coll",
                "files": [
                    {
                        "provider": "google-drive",
                        "fileId": "drive-file-2",
                        "fileName": "cloud.csv",
                    }
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data[0]["status"] == "error"
    assert "Failed to fully roll back cloud ingest" in data[0]["message"]


def test_restore_ingest_file_backup_raises_when_existing_backup_is_missing(
    temp_uploads,
):
    """Rollback should fail loudly when an overwritten file's backup is missing."""

    from xagent.web.api.kb import _restore_ingest_file_backup

    file_path = temp_uploads / "orphaned.xlsx"
    file_path.write_text("new content")

    with pytest.raises(FileNotFoundError, match="Missing ingest backup"):
        _restore_ingest_file_backup(
            file_path=file_path,
            backup_path=None,
            had_existing_file=True,
        )


def test_kb_ingest_cloud_uses_unique_storage_paths_for_duplicate_filenames(
    test_env, temp_uploads
):
    """Same-name cloud files in one batch should not collide on storage_path."""

    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)
    captured_paths: list[str] = []

    class _FakeFilesService:
        def get_media(self, fileId: str):
            return {"fileId": fileId}

    class _FakeDriveService:
        def files(self):
            return _FakeFilesService()

    class _FakeDownloader:
        def __init__(self, fh, request_file):
            self._fh = fh
            self._request_file = request_file

        def next_chunk(self):
            self._fh.write(str(self._request_file["fileId"]).encode("utf-8"))
            return None, True

    def _capture_ingest(*, source_path=None, **kwargs):
        assert source_path is not None
        captured_paths.append(str(source_path))
        return IngestionResult(
            status="success",
            doc_id=f"doc-{len(captured_paths)}",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

    with (
        patch("xagent.web.api.kb.get_google_credentials", return_value=object()),
        patch("xagent.web.api.kb.build", return_value=_FakeDriveService()),
        patch("xagent.web.api.kb.MediaIoBaseDownload", _FakeDownloader),
        patch(
            "xagent.web.api.kb.get_collection_sync",
            side_effect=ValueError("missing collection"),
        ),
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
                        "fileName": "same-name.csv",
                    },
                    {
                        "provider": "google-drive",
                        "fileId": "drive-file-2",
                        "fileName": "same-name.csv",
                    },
                ],
            },
            headers=headers,
        )

    assert response.status_code == 200
    assert len(captured_paths) == 2
    assert len(set(captured_paths)) == 2

    session = TestingSessionLocal()
    try:
        records = (
            session.query(UploadedFile).order_by(UploadedFile.storage_path.asc()).all()
        )
        assert len(records) == 2
        assert len({record.storage_path for record in records}) == 2
        assert {record.filename for record in records} == {"same-name.csv"}
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


def test_check_documents_exist_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "示例知识库集合"

    with patch("xagent.web.api.kb.get_vector_index_store") as mock_get_store:
        mock_store = mock_get_store.return_value
        mock_store.list_document_records.return_value = []

        response = client.post(
            f"/api/kb/collections/{quote(collection_name, safe='')}/documents/check",
            json={"filenames": ["demo.txt"]},
            headers=headers,
        )

    assert response.status_code == 200
    mock_store.list_document_records.assert_called_once_with(
        collection_name=collection_name,
        user_id=int(user.id),
        is_admin=False,
        max_results=DEFAULT_VECTOR_STORE_SCAN_LIMIT,
    )
    assert response.json()["existing_filenames"] == []


def test_check_documents_exist_rejects_path_traversal_in_collection_name(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    malicious_collections = [
        r"collection\\other",
        r"collection\\..\\other",
    ]

    for collection_name in malicious_collections:
        response = client.post(
            f"/api/kb/collections/{quote(collection_name, safe='')}/documents/check",
            json={"filenames": ["demo.txt"]},
            headers=headers,
        )

        assert response.status_code == 422
        assert "Invalid collection name" in response.json()["detail"]


def test_delete_document_prefers_file_id_and_cleans_orphan_file(
    test_env, temp_uploads, monkeypatch, tmp_path
):
    """Deleting by file_id should remove the UploadedFile row when it becomes orphaned."""
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

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
        session.flush()
        ManagedFileRef(file_record).sync_to_durable()
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
        storage_key = str(file_record.storage_key)
    finally:
        session.close()

    assert get_file_storage().exists(storage_key)

    document_state = [
        DocumentRecord(
            doc_id="doc-1",
            file_id=target_file_id,
            source_path=str(file_path),
        )
    ]

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()
        return _successful_delete_result(collection_name, doc_id)

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = list(document_state)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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
    assert not get_file_storage().exists(storage_key)

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


def test_delete_document_keeps_uploaded_file_when_other_docs_still_reference_it(
    test_env, temp_uploads
):
    """Deleting one collection's doc should not orphan-clean a file still referenced elsewhere."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "shared.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="shared.txt",
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
            "doc_id": "doc-demo",
            "file_id": target_file_id,
            "source_path": str(file_path),
        },
        {
            "collection": "other",
            "doc_id": "doc-other",
            "file_id": target_file_id,
            "source_path": str(file_path),
        },
    ]

    def _fake_list_documents_for_user(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name:
            return [
                record
                for record in document_state
                if record["collection"] == collection_name
            ]
        return list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state[:] = [
            record for record in document_state if record["doc_id"] != doc_id
        ]
        return _successful_delete_result(collection_name, doc_id)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
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
            f"/api/kb/collections/demo/documents/shared.txt?file_id={target_file_id}",
            headers=headers,
        )

    assert response.status_code == 200
    assert file_path.exists()

    session = TestingSessionLocal()
    try:
        remaining_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == target_file_id)
            .first()
        )
        assert remaining_record is not None
    finally:
        session.close()


def test_delete_document_skips_orphan_cleanup_when_remaining_doc_refresh_fails(
    test_env, temp_uploads
):
    """Refresh failures after delete should not guess orphan state for shared files."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "shared-refresh-failure.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="shared-refresh-failure.txt",
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
            "doc_id": "doc-demo",
            "file_id": target_file_id,
            "source_path": str(file_path),
        },
        {
            "collection": "other",
            "doc_id": "doc-other",
            "file_id": target_file_id,
            "source_path": str(file_path),
        },
    ]
    call_count = {"value": 0}

    def _fake_list_documents_for_user(*args, **kwargs):
        call_count["value"] += 1
        collection_name = kwargs.get("collection_name")
        if collection_name:
            return [
                record
                for record in document_state
                if record["collection"] == collection_name
            ]
        if call_count["value"] >= 2:
            raise RuntimeError("refresh failed")
        return list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state[:] = [
            record for record in document_state if record["doc_id"] != doc_id
        ]
        return _successful_delete_result(collection_name, doc_id)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
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
            f"/api/kb/collections/demo/documents/shared-refresh-failure.txt?file_id={target_file_id}",
            headers=headers,
        )

    assert response.status_code == 200
    assert file_path.exists()

    session = TestingSessionLocal()
    try:
        remaining_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == target_file_id)
            .first()
        )
        assert remaining_record is not None
    finally:
        session.close()


def test_delete_document_does_not_cleanup_uploaded_file_when_delete_fails(
    test_env, temp_uploads
):
    """Uploaded file cleanup must only happen after a successful KB delete."""
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    from xagent.core.tools.core.RAG_tools.core.schemas import DocumentOperationResult

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "delete-failure.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="delete-failure.txt",
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
            "doc_id": "doc-failed",
            "file_id": target_file_id,
            "source_path": str(file_path),
        }
    ]

    def _fake_list_documents_for_user(*args, **kwargs):
        collection_name = kwargs.get("collection_name")
        if collection_name:
            return [
                record
                for record in document_state
                if record["collection"] == collection_name
            ]
        return list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        return DocumentOperationResult(
            status="error",
            collection=collection_name,
            doc_id=doc_id,
            new_status="failed",
            message="simulated delete failure",
            warnings=[],
            details={},
        )

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
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
            f"/api/kb/collections/demo/documents/delete-failure.txt?file_id={target_file_id}",
            headers=headers,
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["deleted_doc_ids"] == []
    assert "simulated delete failure" in payload["errors"][0]
    assert file_path.exists()

    session = TestingSessionLocal()
    try:
        remaining_record = (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == target_file_id)
            .first()
        )
        assert remaining_record is not None
    finally:
        session.close()


def test_delete_document_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "示例知识库集合"
    file_path = temp_uploads / f"user_{user.id}" / collection_name / "demo.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    deleted_doc_ids: list[str] = []
    document_state = [
        DocumentRecord(
            doc_id="doc-1",
            file_id=None,
            source_path=str(file_path),
        )
    ]

    def _fake_delete_document(collection_name_arg, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)
        assert collection_name_arg == collection_name
        return _successful_delete_result(collection_name_arg, doc_id)

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = list(document_state)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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
        DocumentRecord(
            doc_id="doc-a",
            file_id="file-a",
            source_path=str(file_a),
        ),
        DocumentRecord(
            doc_id="doc-b",
            file_id="file-b",
            source_path=str(file_b),
        ),
    ]

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = list(document_state)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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
        DocumentRecord(
            doc_id="doc-a",
            file_id="file-a",
            source_path=str(file_path),
        ),
        DocumentRecord(
            doc_id="doc-b",
            file_id="file-b",
            source_path=str(file_path),
        ),
    ]
    deleted_doc_ids: list[str] = []

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        deleted_doc_ids.append(doc_id)
        return _successful_delete_result(collection_name, doc_id)

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = list(document_state)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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

    deleted_doc_ids, _fake_delete_document = _make_delete_tracker()
    expected_doc_id = generate_deterministic_doc_id("demo", target_file_id)
    mock_store = MagicMock()
    mock_store.iter_batches.return_value = []

    mock_store = MagicMock()
    mock_store.list_document_records.side_effect = RuntimeError("documents unavailable")

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
        patch(
            "xagent.web.api.kb.list_documents",
            side_effect=RuntimeError("documents unavailable"),
        ),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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


def test_delete_document_by_file_id_prefers_documents_table_doc_id(
    test_env, temp_uploads
):
    app, headers, user, TestingSessionLocal = test_env
    client = TestClient(app)

    file_path = temp_uploads / f"user_{user.id}" / "demo" / "resolved-from-docs.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")

    session = TestingSessionLocal()
    try:
        file_record = UploadedFile(
            user_id=int(user.id),
            filename="resolved-from-docs.txt",
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

    deleted_doc_ids, _fake_delete_document = _make_delete_tracker()
    expected_doc_id = "doc-from-documents-table"

    mock_batch = MagicMock()
    mock_batch.to_pylist.return_value = [
        {"doc_id": expected_doc_id, "source_path": str(file_path)}
    ]
    mock_store = MagicMock()
    mock_store.iter_batches.return_value = [mock_batch]

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb._list_documents_for_user",
            side_effect=RuntimeError("documents unavailable"),
        ),
        patch(
            "xagent.web.api.kb.list_documents",
            side_effect=RuntimeError("documents unavailable"),
        ),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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
    mock_store.iter_batches.assert_called_once_with(
        table_name="documents",
        columns=["doc_id", "source_path"],
        batch_size=10,
        filters={"collection": "demo", "file_id": target_file_id},
        user_id=None,
        is_admin=True,
    )


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
        session.refresh(file_record)
    finally:
        session.close()

    document_state = [
        DocumentRecord(
            doc_id=generate_deterministic_doc_id("demo", str(file_path)),
            file_id=None,
            source_path=str(file_path),
        )
    ]

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()
        return _successful_delete_result(collection_name, doc_id)

    fake_result = ListCollectionsResult(
        status="success",
        collections=[CollectionInfo(name="demo", documents=0, document_names=[])],
        total_count=1,
        message="ok",
        warnings=[],
    )

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = list(document_state)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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


def test_delete_document_without_file_id_preserves_uploaded_file_on_cleanup_refresh_failure(
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
        session.refresh(file_record)
    finally:
        session.close()

    document_state = [
        DocumentRecord(
            doc_id=generate_deterministic_doc_id("demo", str(file_path)),
            file_id=None,
            source_path=str(file_path),
        )
    ]

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()
        return _successful_delete_result(collection_name, doc_id)

    fake_result = ListCollectionsResult(
        status="success",
        collections=[CollectionInfo(name="demo", documents=0, document_names=[])],
        total_count=1,
        message="ok",
        warnings=[],
    )

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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
    deleted_doc_ids, _fake_delete_document = _make_delete_tracker()

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

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = []

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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
    deleted_doc_ids, _fake_delete_document = _make_delete_tracker()

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

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = []

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = []

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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
        DocumentRecord(
            doc_id="doc-commit-failure",
            file_id=target_file_id,
            source_path=str(file_path),
        )
    ]

    mock_store = MagicMock()
    mock_store.list_document_records.return_value = list(document_state)

    def _fake_delete_document(collection_name, doc_id, user_id, is_admin):
        document_state.clear()
        return _successful_delete_result(collection_name, doc_id)

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
                "xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock
            ),
            patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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

    mock_store = MagicMock()
    mock_store.list_document_records.side_effect = RuntimeError("documents unavailable")

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.get_vector_index_store", return_value=mock_store),
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


def test_kb_delete_collection_cleans_file_id_managed_root_file(
    test_env, temp_uploads, monkeypatch, tmp_path
):
    """Collection delete should clean orphan UploadedFile rows even outside collection dir."""
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

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
        session.flush()
        ManagedFileRef(file_record).sync_to_durable()
        session.commit()
        session.refresh(file_record)
        target_file_id = str(file_record.file_id)
        storage_key = str(file_record.storage_key)
    finally:
        session.close()

    assert get_file_storage().exists(storage_key)

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
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
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
    assert not get_file_storage().exists(storage_key)

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


def test_get_parse_result_accepts_unicode_collection_name(test_env, temp_uploads):
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "示例知识库集合"
    elements = [{"type": "text", "text": "hello", "metadata": {}}]
    pagination = {"page": 1, "page_size": 20, "total_count": 1, "total_pages": 1}

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb.reconstruct_parse_result_from_db",
            return_value=(elements, "hash-1"),
        ) as mock_reconstruct,
        patch(
            "xagent.web.api.kb.paginate_parse_results",
            return_value=(elements, pagination),
        ),
    ):
        response = client.get(
            f"/api/kb/collections/{quote(collection_name, safe='')}/parses/doc-1/parse_result",
            headers=headers,
        )

    assert response.status_code == 200
    mock_reconstruct.assert_called_once_with(
        collection_name,
        "doc-1",
        None,
        user_id=int(user.id),
        is_admin=False,
    )


def test_get_parse_result_rejects_path_traversal_in_collection_name(
    test_env, temp_uploads
):
    app, headers, user, _ = test_env
    client = TestClient(app)

    malicious_collections = [
        r"collection\\other",
        r"collection\\..\\other",
    ]

    for collection_name in malicious_collections:
        response = client.get(
            f"/api/kb/collections/{quote(collection_name, safe='')}/parses/doc-1/parse_result",
            headers=headers,
        )

        assert response.status_code == 422
        assert "Invalid collection name" in response.json()["detail"]


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
