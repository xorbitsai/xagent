from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.e2e.app_harness import (
    configure_e2e_app_environment,
    create_e2e_user,
    disable_external_app_services,
    init_e2e_db,
    run_e2e_app_client,
    seed_registered_local_file,
)
from tests.e2e.minio_harness import MinioStorage, run_minio_storage
from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.models.uploaded_file import UploadedFile

pytestmark = [pytest.mark.e2e, pytest.mark.docker]


@pytest.fixture
def minio_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[MinioStorage]:
    yield from run_minio_storage(monkeypatch)


def _record(session_factory: sessionmaker[Session], file_id: str) -> UploadedFile:
    db = session_factory()
    try:
        return db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
    finally:
        db.close()


def test_startup_sync_repairs_only_files_that_need_durable_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    uploads_dir = configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()

    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="startup-sync-user")
        user_id = user.id

        legacy_file_id = str(uuid4())
        existing_file_id = str(uuid4())
        missing_remote_file_id = str(uuid4())
        missing_local_file_id = str(uuid4())

        existing_key = f"users/{user_id}/uploads/{existing_file_id}/existing.txt"
        missing_remote_key = (
            f"users/{user_id}/uploads/{missing_remote_file_id}/missing-remote.txt"
        )
        missing_local_key = (
            f"users/{user_id}/uploads/{missing_local_file_id}/missing-local.txt"
        )
        existing_object = get_unscoped_file_storage().put_bytes(
            b"already durable\n",
            existing_key,
            content_type="text/plain",
        )

        legacy = seed_registered_local_file(
            db,
            uploads_dir=uploads_dir,
            user_id=user_id,
            filename="legacy.txt",
            content=b"legacy needs upload\n",
            file_id=legacy_file_id,
            mime_type="text/plain",
            storage_status="legacy",
        )
        seed_registered_local_file(
            db,
            uploads_dir=uploads_dir,
            user_id=user_id,
            filename="existing.txt",
            content=b"local should not overwrite remote\n",
            file_id=existing_file_id,
            mime_type="text/plain",
            storage_backend=existing_object.backend,
            storage_key=existing_object.key,
            storage_uri=existing_object.uri,
            checksum=existing_object.checksum,
            etag=existing_object.etag,
            storage_status="available",
        )
        seed_registered_local_file(
            db,
            uploads_dir=uploads_dir,
            user_id=user_id,
            filename="missing-remote.txt",
            content=b"remote should be repaired\n",
            file_id=missing_remote_file_id,
            mime_type="text/plain",
            storage_backend="s3",
            storage_key=missing_remote_key,
            storage_status="available",
        )
        missing_local = seed_registered_local_file(
            db,
            uploads_dir=uploads_dir,
            user_id=user_id,
            filename="missing-local.txt",
            content=b"this file disappears before startup\n",
            file_id=missing_local_file_id,
            mime_type="text/plain",
            storage_backend="s3",
            storage_key=missing_local_key,
            storage_status="available",
        )
        missing_local.path.unlink()
    finally:
        db.close()

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user_id,
    ) as app:
        setup_response = app.client.get("/api/auth/setup-status")
        assert setup_response.status_code == 200

        legacy_record = _record(app.session_factory, legacy_file_id)
        existing_record = _record(app.session_factory, existing_file_id)
        missing_remote_record = _record(app.session_factory, missing_remote_file_id)
        missing_local_record = _record(app.session_factory, missing_local_file_id)

        assert legacy_record.storage_backend == "s3"
        assert legacy_record.storage_status == "available"
        assert legacy_record.storage_key == (
            f"users/{user_id}/uploads/{legacy_file_id}/legacy.txt"
        )
        assert minio_storage.object_bytes(str(legacy_record.storage_key)) == (
            b"legacy needs upload\n"
        )
        assert legacy.path.exists()

        assert existing_record.storage_key == existing_key
        assert minio_storage.object_bytes(existing_key) == b"already durable\n"

        assert missing_remote_record.storage_key == missing_remote_key
        assert minio_storage.object_bytes(missing_remote_key) == (
            b"remote should be repaired\n"
        )

        assert missing_local_record.storage_key == missing_local_key
        assert not minio_storage.exists(missing_local_key)


def test_startup_file_storage_sync_runs_after_startup_but_gates_client_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    uploads_dir = configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()

    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="startup-gate-user")
        user_id = user.id
        legacy_file_id = str(uuid4())
        seed_registered_local_file(
            db,
            uploads_dir=uploads_dir,
            user_id=user_id,
            filename="startup-gate.txt",
            content=b"startup gate content\n",
            file_id=legacy_file_id,
            mime_type="text/plain",
            storage_status="legacy",
        )
    finally:
        db.close()

    sync_started = threading.Event()
    release_sync = threading.Event()
    close_app = threading.Event()
    client_request_started = threading.Event()
    client_request_completed = threading.Event()
    app_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    app_errors: queue.Queue[BaseException] = queue.Queue()
    request_errors: queue.Queue[BaseException] = queue.Queue()
    request_responses: queue.Queue[Any] = queue.Queue(maxsize=1)

    def _configure_app_module(app_module: Any) -> None:
        original_sync = app_module.run_startup_file_storage_sync

        def _delayed_startup_sync() -> None:
            sync_started.set()
            assert release_sync.wait(timeout=10)
            original_sync()

        app_module.run_startup_file_storage_sync = _delayed_startup_sync

    def _run_app() -> None:
        try:
            with run_e2e_app_client(
                monkeypatch,
                username=user.username,
                user_id=user_id,
                configure_app_module=_configure_app_module,
            ) as app:
                app_queue.put(app)
                assert close_app.wait(timeout=10)
        except BaseException as exc:
            app_errors.put(exc)

    app_thread = threading.Thread(target=_run_app, daemon=True)
    app_thread.start()

    try:
        app = app_queue.get(timeout=5)
        assert sync_started.wait(timeout=DB_PROGRESS_TIMEOUT)

        health_response = app.client.get("/health")
        assert health_response.status_code == 200

        ready_response = app.client.get("/ready")
        assert ready_response.status_code == 503
        assert ready_response.json()["status"] == "starting"

        def _make_client_request() -> None:
            try:
                client_request_started.set()
                response = app.client.get("/api/auth/setup-status")
                request_responses.put(response)
            except BaseException as exc:
                request_errors.put(exc)
            finally:
                client_request_completed.set()

        request_thread = threading.Thread(target=_make_client_request, daemon=True)
        request_thread.start()

        assert client_request_started.wait(timeout=5)
        assert not client_request_completed.wait(timeout=0.2)

        release_sync.set()
        request_thread.join(timeout=10)
        assert not request_thread.is_alive()
        assert request_errors.empty()

        client_response = request_responses.get_nowait()
        assert client_response.status_code == 200

        ready_response = app.client.get("/ready")
        assert ready_response.status_code == 200
        assert ready_response.json()["status"] == "ready"

        legacy_record = _record(app.session_factory, legacy_file_id)
        assert legacy_record.storage_backend == "s3"
        assert legacy_record.storage_status == "available"
        assert legacy_record.storage_key == (
            f"users/{user_id}/uploads/{legacy_file_id}/startup-gate.txt"
        )
        assert minio_storage.object_bytes(str(legacy_record.storage_key)) == (
            b"startup gate content\n"
        )
    finally:
        release_sync.set()
        close_app.set()
        app_thread.join(timeout=10)

    assert not app_thread.is_alive()
    assert app_errors.empty()


def test_startup_file_storage_sync_recovers_after_initial_storage_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    uploads_dir = configure_e2e_app_environment(monkeypatch, tmp_path=tmp_path)
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()

    db = SessionLocal()
    try:
        user = create_e2e_user(db, username="startup-retry-user")
        user_id = user.id
        legacy_file_id = str(uuid4())
        seed_registered_local_file(
            db,
            uploads_dir=uploads_dir,
            user_id=user_id,
            filename="startup-retry.txt",
            content=b"startup retry content\n",
            file_id=legacy_file_id,
            mime_type="text/plain",
            storage_status="legacy",
        )
    finally:
        db.close()

    first_attempt_failed = threading.Event()
    sync_completed = threading.Event()
    close_app = threading.Event()
    app_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    app_errors: queue.Queue[BaseException] = queue.Queue()

    def _configure_app_module(app_module: Any) -> None:
        original_sync = app_module.run_startup_file_storage_sync
        attempts = {"count": 0}

        def _fail_once_then_sync() -> None:
            attempts["count"] += 1
            if attempts["count"] == 1:
                first_attempt_failed.set()
                raise RuntimeError("s3 unavailable")
            original_sync()
            sync_completed.set()

        app_module.run_startup_file_storage_sync = _fail_once_then_sync
        app_module.FILE_STORAGE_STARTUP_SYNC_RETRY_INTERVAL_SECONDS = 0.01

    def _run_app() -> None:
        try:
            with run_e2e_app_client(
                monkeypatch,
                username=user.username,
                user_id=user_id,
                configure_app_module=_configure_app_module,
            ) as app:
                app_queue.put(app)
                assert close_app.wait(timeout=10)
        except BaseException as exc:
            app_errors.put(exc)

    app_thread = threading.Thread(target=_run_app, daemon=True)
    app_thread.start()

    try:
        app = app_queue.get(timeout=5)
        assert first_attempt_failed.wait(timeout=DB_PROGRESS_TIMEOUT)

        ready_response = app.client.get("/ready")
        assert ready_response.status_code == 503
        assert ready_response.json()["status"] == "error"

        failed_client_response = app.client.get("/api/auth/setup-status")
        assert failed_client_response.status_code == 503
        assert failed_client_response.json()["detail"] == (
            "Startup file storage sync failed"
        )

        assert sync_completed.wait(timeout=DB_PROGRESS_TIMEOUT)

        recovered_client_response = app.client.get("/api/auth/setup-status")
        assert recovered_client_response.status_code == 200

        ready_response = app.client.get("/ready")
        assert ready_response.status_code == 200
        assert ready_response.json()["status"] == "ready"

        legacy_record = _record(app.session_factory, legacy_file_id)
        assert legacy_record.storage_backend == "s3"
        assert legacy_record.storage_status == "available"
        assert legacy_record.storage_key == (
            f"users/{user_id}/uploads/{legacy_file_id}/startup-retry.txt"
        )
        assert minio_storage.object_bytes(str(legacy_record.storage_key)) == (
            b"startup retry content\n"
        )
    finally:
        close_app.set()
        app_thread.join(timeout=10)

    assert not app_thread.is_alive()
    assert app_errors.empty()
