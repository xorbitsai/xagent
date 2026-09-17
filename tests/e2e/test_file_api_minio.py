from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.e2e.app_harness import (
    SeededLocalFile,
    build_access_token,
    configure_e2e_app_environment,
    create_e2e_user,
    disable_external_app_services,
    init_e2e_db,
    run_e2e_app_client,
)
from tests.e2e.minio_harness import MinioStorage, run_minio_storage
from xagent.core.tools.adapters.vibe.file_ingestion_tool import (
    CreateKnowledgeBaseFromFileTool,
)
from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionResult,
    IngestionStepResult,
)
from xagent.core.workspace import TaskWorkspace
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.kb_file_service import reconcile_uploaded_files

pytestmark = [pytest.mark.e2e, pytest.mark.docker]


@pytest.fixture
def minio_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[MinioStorage]:
    yield from run_minio_storage(monkeypatch)


def _upload_text_file(client: TestClient, headers: dict[str, str]) -> dict[str, str]:
    response = client.post(
        "/api/files/upload",
        files={"file": ("source.txt", b"source from minio\n", "text/plain")},
        data={"task_type": "general"},
        headers=headers,
    )
    assert response.status_code == 200
    return response.json()


def _record(session_factory: sessionmaker[Session], file_id: str) -> UploadedFile:
    db = session_factory()
    try:
        return db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
    finally:
        db.close()


def _seed_durable_uploaded_file(
    session_factory: sessionmaker[Session],
    minio_storage: MinioStorage,
    *,
    user_id: int,
    filename: str,
    content: bytes,
    local_path: Path,
    task_id: int | None = None,
    file_id: str | None = None,
    mime_type: str = "text/plain",
    workspace_relative_path: str | None = None,
    workspace_category: str | None = None,
) -> SeededLocalFile:
    resolved_file_id = file_id or str(uuid4())
    storage_key = f"users/{user_id}/uploads/{resolved_file_id}/{filename}"
    minio_storage.put_object(storage_key, content, mime_type)
    db = session_factory()
    try:
        record = UploadedFile(
            file_id=resolved_file_id,
            user_id=user_id,
            task_id=task_id,
            filename=filename,
            storage_path=str(local_path),
            storage_backend="s3",
            storage_key=storage_key,
            storage_uri=f"s3://{minio_storage.bucket}/{minio_storage.prefix}/{storage_key}",
            storage_status="available",
            mime_type=mime_type,
            file_size=len(content),
            workspace_relative_path=workspace_relative_path,
            workspace_category=workspace_category,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()
    return SeededLocalFile(
        file_id=resolved_file_id,
        path=local_path,
        filename=filename,
    )


def test_download_and_preview_materialize_uploaded_file_from_minio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    uploads_dir = configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    del uploads_dir
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="file-api-user")
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        uploaded = _upload_text_file(app.client, app.headers)
        file_id = uploaded["file_id"]
        record = _record(app.session_factory, file_id)
        local_path = Path(str(record.storage_path))
        storage_key = str(record.storage_key)
        durable_sha256 = str(record.checksum)
        assert minio_storage.object_info(storage_key)["Metadata"] == {
            "xagent-sha256": durable_sha256
        }
        assert local_path.exists()
        local_path.unlink()

        download = app.client.get(
            f"/api/files/download/{file_id}",
            headers=app.headers,
        )
        assert download.status_code == 200
        assert download.content == b"source from minio\n"
        assert local_path.read_bytes() == b"source from minio\n"

        local_path.unlink()
        preview = app.client.get(
            f"/api/files/preview/{file_id}",
            headers=app.headers,
        )
        assert preview.status_code == 200
        assert preview.content == b"source from minio\n"
        assert not local_path.exists()
        expected_materialized = (
            tmp_path
            / "materialized"
            / hashlib.sha256(storage_key.encode("utf-8")).hexdigest()[:16]
            / durable_sha256
            / "source.txt"
        )
        assert expected_materialized.read_bytes() == b"source from minio\n"
        assert not list((tmp_path / "materialized").rglob("*.metadata.json"))


def test_public_preview_relative_asset_restores_durable_only_file_from_minio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="public-preview-asset-user")
        task = Task(id=2811, user_id=user.id, title="Public preview asset")
        db.add(task)
        db.commit()
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        html_upload = app.client.post(
            "/api/files/upload",
            files={
                "file": (
                    "index.html",
                    b"<script src='assets/app.js'></script>",
                    "text/html",
                )
            },
            data={"task_type": "general", "task_id": "2811"},
            headers=app.headers,
        )
        assert html_upload.status_code == 200
        html_file_id = html_upload.json()["file_id"]
        html_record = _record(app.session_factory, html_file_id)
        asset_path = Path(str(html_record.storage_path)).parent / "assets" / "app.js"
        seeded_asset = _seed_durable_uploaded_file(
            app.session_factory,
            minio_storage,
            user_id=user.id,
            task_id=2811,
            filename="app.js",
            content=b"console.log('from minio');",
            local_path=asset_path,
            file_id="44444444-4444-4444-8444-444444444444",
            mime_type="application/javascript",
            workspace_relative_path="output/assets/app.js",
            workspace_category="output",
        )
        assert not seeded_asset.path.exists()

        preview_asset = app.client.get(
            f"/api/files/public/preview/{html_file_id}",
            params={"relative_path": "assets/app.js"},
        )

        assert preview_asset.status_code == 200
        assert preview_asset.content == b"console.log('from minio');"
        assert seeded_asset.path.read_bytes() == b"console.log('from minio');"


def test_workspace_user_file_listing_includes_durable_only_minio_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="workspace-list-minio-user")
        task = Task(id=2711, user_id=user.id, title="Workspace durable listing")
        db.add(task)
        db.commit()
    finally:
        db.close()

    missing_local_path = (
        tmp_path
        / "uploads"
        / f"user_{user.id}"
        / "web_task_2711"
        / "input"
        / "durable-only.txt"
    )
    _seed_durable_uploaded_file(
        SessionLocal,
        minio_storage,
        user_id=user.id,
        task_id=2711,
        filename="durable-only.txt",
        content=b"durable only listing\n",
        local_path=missing_local_path,
        file_id="11111111-1111-4111-8111-111111111111",
    )

    workspace = TaskWorkspace(id="web_task_2711", base_dir=str(tmp_path / "uploads"))
    workspace.db_session = SessionLocal()
    try:
        result = workspace.list_all_user_files(include_workspace_files=False)
    finally:
        workspace.db_session.close()

    assert result["success"] is True
    files_by_id = {item["file_id"]: item for item in result["files"]}
    assert "11111111-1111-4111-8111-111111111111" in files_by_id
    listed = files_by_id["11111111-1111-4111-8111-111111111111"]
    assert listed["filename"] == "durable-only.txt"
    assert listed["size"] == len(b"durable only listing\n")
    assert listed["in_current_workspace"] is False


def test_upload_returns_503_and_rolls_back_when_minio_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    del minio_storage
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="file-upload-outage-user")
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        from xagent.core.file_storage.storage import FsspecFileStorage

        def fail_put_file(
            self: FsspecFileStorage,
            source: Path,
            key: str,
            content_type: str | None = None,
        ) -> None:
            del source, key, content_type
            raise RuntimeError("simulated MinIO write outage")

        monkeypatch.setattr(FsspecFileStorage, "put_file", fail_put_file)

        upload = app.client.post(
            "/api/files/upload",
            files={"file": ("outage.txt", b"outage content\n", "text/plain")},
            data={"task_type": "general"},
            headers=app.headers,
        )

        assert upload.status_code == 503
        assert "durable storage" in upload.json()["detail"].lower()
        assert not list((tmp_path / "uploads").rglob("outage.txt"))

        db = app.session_factory()
        try:
            assert (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.user_id == user.id,
                    UploadedFile.filename == "outage.txt",
                )
                .first()
                is None
            )
        finally:
            db.close()


def test_download_and_preview_return_503_when_minio_read_fails_without_local_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    del minio_storage
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="file-read-outage-user")
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        uploaded = _upload_text_file(app.client, app.headers)
        file_id = uploaded["file_id"]
        record = _record(app.session_factory, file_id)
        local_path = Path(str(record.storage_path))
        assert local_path.exists()
        local_path.unlink()

        from xagent.core.file_storage.storage import FsspecFileStorage

        def fail_open_read(self: FsspecFileStorage, key: str) -> None:
            del key
            raise RuntimeError("simulated MinIO read outage")

        monkeypatch.setattr(FsspecFileStorage, "open_read", fail_open_read)

        download = app.client.get(
            f"/api/files/download/{file_id}",
            headers=app.headers,
        )
        preview = app.client.get(
            f"/api/files/preview/{file_id}",
            headers=app.headers,
        )

        assert download.status_code == 503
        assert "durable storage" in download.json()["detail"].lower()
        assert preview.status_code == 503
        assert "durable storage" in preview.json()["detail"].lower()
        assert not local_path.exists()


@pytest.mark.asyncio
async def test_create_kb_from_file_tool_restores_durable_only_upload_from_minio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="kb-tool-minio-user")
    finally:
        db.close()

    local_path = tmp_path / "uploads" / f"user_{user.id}" / "kb-tool.txt"
    seeded = _seed_durable_uploaded_file(
        SessionLocal,
        minio_storage,
        user_id=user.id,
        filename="kb-tool.txt",
        content=b"kb tool durable content\n",
        local_path=local_path,
        file_id="33333333-3333-4333-8333-333333333333",
        mime_type="text/plain",
    )
    assert not seeded.path.exists()

    ingest_result = IngestionResult(
        status="success",
        doc_id="doc-1",
        parse_hash="parse-1",
        chunk_count=1,
        embedding_count=1,
        vector_count=1,
        completed_steps=[IngestionStepResult(name="register_document")],
        failed_step=None,
        message="ok",
        warnings=[],
        file_id=seeded.file_id,
    )
    observed_source_paths: list[str] = []

    def fake_run_document_ingestion(**kwargs: object) -> IngestionResult:
        source_path = str(kwargs["source_path"])
        observed_source_paths.append(source_path)
        assert Path(source_path).read_bytes() == b"kb tool durable content\n"
        return ingest_result

    service = AsyncMock()
    service.prepare_collection.return_value = "agent_file_kb"
    service.refresh_collection_metadata.return_value = None

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
        lambda user_id, is_admin: service,
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.pipelines.document_ingestion.run_document_ingestion",
        fake_run_document_ingestion,
    )

    tool = CreateKnowledgeBaseFromFileTool(user_id=user.id, is_admin=False)
    result = await tool.run_json_async(
        {"file_ids": [seeded.file_id], "collection_name": "agent_file_kb"}
    )

    assert result["success"] is True
    assert result["files_ingested"] == 1
    assert observed_source_paths == [str(seeded.path)]
    assert seeded.path.read_bytes() == b"kb tool durable content\n"


def test_download_serves_local_copy_when_minio_read_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    del minio_storage
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="file-local-read-outage-user")
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        uploaded = _upload_text_file(app.client, app.headers)
        file_id = uploaded["file_id"]
        record = _record(app.session_factory, file_id)
        local_path = Path(str(record.storage_path))
        assert local_path.exists()

        from xagent.core.file_storage.storage import FsspecFileStorage

        def fail_open_read(self: FsspecFileStorage, key: str) -> None:
            del key
            raise RuntimeError("simulated MinIO read outage")

        monkeypatch.setattr(FsspecFileStorage, "open_read", fail_open_read)

        download = app.client.get(
            f"/api/files/download/{file_id}",
            headers=app.headers,
        )

        assert download.status_code == 200
        assert download.content == b"source from minio\n"


def test_delete_removes_uploaded_file_from_db_local_disk_and_minio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="file-delete-user")
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        uploaded = _upload_text_file(app.client, app.headers)
        file_id = uploaded["file_id"]
        record = _record(app.session_factory, file_id)
        storage_key = str(record.storage_key)
        local_path = Path(str(record.storage_path))

        assert local_path.exists()
        assert minio_storage.exists(storage_key)

        delete = app.client.delete(f"/api/files/{file_id}", headers=app.headers)
        assert delete.status_code == 200
        assert not local_path.exists()
        assert not minio_storage.exists(storage_key)

        db = app.session_factory()
        try:
            assert (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                is None
            )
        finally:
            db.close()


def test_delete_keeps_db_row_when_durable_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="file-delete-failure-user")
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        uploaded = _upload_text_file(app.client, app.headers)
        file_id = uploaded["file_id"]
        record = _record(app.session_factory, file_id)
        storage_key = str(record.storage_key)
        local_path = Path(str(record.storage_path))

        from xagent.core.file_storage.storage import FsspecFileStorage

        real_delete = FsspecFileStorage.delete

        def fail_target_delete(self: FsspecFileStorage, key: str) -> None:
            if key == storage_key:
                raise RuntimeError("simulated durable delete failure")
            real_delete(self, key)

        monkeypatch.setattr(FsspecFileStorage, "delete", fail_target_delete)

        assert local_path.exists()
        assert minio_storage.exists(storage_key)

        delete = app.client.delete(f"/api/files/{file_id}", headers=app.headers)
        assert delete.status_code == 503

        assert local_path.exists()
        assert minio_storage.exists(storage_key)
        assert minio_storage.object_bytes(storage_key) == b"source from minio\n"

        db = app.session_factory()
        try:
            assert (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                is not None
            )
        finally:
            db.close()


def test_stale_kb_reconcile_retries_minio_object_cleanup_after_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="kb-reconcile-minio-user")
    finally:
        db.close()

    local_path = tmp_path / "uploads" / f"user_{user.id}" / "kb" / "stale-failed.md"
    seeded = _seed_durable_uploaded_file(
        SessionLocal,
        minio_storage,
        user_id=user.id,
        filename="stale-failed.md",
        content=b"stale failed kb file\n",
        local_path=local_path,
        file_id="22222222-2222-4222-8222-222222222222",
        mime_type="text/markdown",
    )
    db = SessionLocal()
    try:
        record = (
            db.query(UploadedFile).filter(UploadedFile.file_id == seeded.file_id).one()
        )
        record.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        db.commit()
    finally:
        db.close()

    from xagent.core.file_storage.storage import FsspecFileStorage

    real_delete = FsspecFileStorage.delete
    attempts = 0

    def fail_first_target_delete(self: FsspecFileStorage, key: str) -> None:
        nonlocal attempts
        if key == f"users/{user.id}/uploads/{seeded.file_id}/stale-failed.md":
            attempts += 1
            if attempts == 1:
                raise RuntimeError("simulated MinIO delete outage")
        real_delete(self, key)

    monkeypatch.setattr(FsspecFileStorage, "delete", fail_first_target_delete)

    db = SessionLocal()
    try:
        first = reconcile_uploaded_files(
            db,
            user_id=user.id,
            is_admin=False,
            stale_ttl_hours=1,
            delete_stale=True,
            deletable_statuses={"UNKNOWN"},
        )
        assert first["cleanup_errors"] == 1
        assert first["deleted"] == 0
        assert (
            db.query(UploadedFile)
            .filter(UploadedFile.file_id == seeded.file_id)
            .first()
            is not None
        )
        assert minio_storage.exists(
            f"users/{user.id}/uploads/{seeded.file_id}/stale-failed.md"
        )

        second = reconcile_uploaded_files(
            db,
            user_id=user.id,
            is_admin=False,
            stale_ttl_hours=1,
            delete_stale=True,
            deletable_statuses={"UNKNOWN"},
        )
        assert second["cleanup_errors"] == 0
        assert second["deleted"] == 1
        assert (
            db.query(UploadedFile)
            .filter(UploadedFile.file_id == seeded.file_id)
            .first()
            is None
        )
        assert not minio_storage.exists(
            f"users/{user.id}/uploads/{seeded.file_id}/stale-failed.md"
        )
        assert attempts == 2
    finally:
        db.close()


def test_file_routes_reject_cross_user_access_and_keep_minio_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    db = SessionLocal()
    try:
        owner = create_e2e_user(db, username="owner-user")
        other = create_e2e_user(db, username="other-user")
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=owner.username,
        user_id=owner.id,
    ) as app:
        uploaded = _upload_text_file(app.client, app.headers)
        file_id = uploaded["file_id"]
        record = _record(app.session_factory, file_id)
        storage_key = str(record.storage_key)
        other_headers = {
            "Authorization": (
                f"Bearer {build_access_token(username=other.username, user_id=other.id)}"
            )
        }

        for method, path in [
            ("GET", f"/api/files/download/{file_id}"),
            ("GET", f"/api/files/preview/{file_id}"),
            ("DELETE", f"/api/files/{file_id}"),
        ]:
            response = app.client.request(method, path, headers=other_headers)
            assert response.status_code == 403

        assert minio_storage.exists(storage_key)
        assert minio_storage.object_bytes(storage_key) == b"source from minio\n"
