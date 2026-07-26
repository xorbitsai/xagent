from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.e2e.minio_harness import (
    MinioStorage,
    PersistenceApp,
    run_file_persistence_app,
    run_minio_storage,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile

pytestmark = [pytest.mark.e2e, pytest.mark.docker]


LLM_RESPONSES_PATH = (
    Path(__file__).parent / "fixtures" / "file_persistence_minio_llm_responses.json"
)
INPUT_MATERIALIZATION_LLM_RESPONSES_PATH = (
    Path(__file__).parent / "fixtures" / "file_input_materialization_llm_responses.json"
)


@pytest.fixture
def minio_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[MinioStorage]:
    yield from run_minio_storage(monkeypatch)


@pytest.fixture
def persistence_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, minio_storage: MinioStorage
) -> Iterator[PersistenceApp]:
    del minio_storage
    yield from run_file_persistence_app(
        monkeypatch,
        tmp_path,
        llm_responses_path=LLM_RESPONSES_PATH,
    )


def _fetch_file_record(session_factory: sessionmaker[Session], file_id: str) -> Any:
    db = session_factory()
    try:
        return db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
    finally:
        db.close()


def _assert_task_output_generation_key(
    storage_key: str,
    *,
    user_id: int,
    task_id: int,
    file_id: str,
    relative_path: str,
) -> None:
    prefix = f"users/{user_id}/tasks/{task_id}/outputs/{file_id}/_versions/"
    assert storage_key.startswith(prefix)
    generation, actual_relative_path = storage_key.removeprefix(prefix).split("/", 1)
    assert UUID(generation)
    assert actual_relative_path == relative_path


def _wait_for_startup_file_storage_sync(client: TestClient) -> None:
    response = client.get("/api/auth/setup-status")
    assert response.status_code == 200


def _receive_until_task_completed(websocket: Any) -> dict[str, Any]:
    for _ in range(200):
        message = websocket.receive_json()
        if message.get("type") == "task_completed":
            return message
        if message.get("type") in {
            "task_error",
            "agent_error",
            "error",
        }:
            raise AssertionError(f"Task execution failed over websocket: {message}")
    raise AssertionError("Timed out waiting for task_completed websocket event")


def _receive_until_type(websocket: Any, event_type: str) -> dict[str, Any]:
    for _ in range(50):
        message = websocket.receive_json()
        if message.get("type") == event_type:
            return message
        if message.get("type") in {
            "task_error",
            "agent_error",
            "error",
        }:
            raise AssertionError(f"Unexpected websocket error: {message}")
    raise AssertionError(f"Timed out waiting for {event_type} websocket event")


def _receive_until_execution_error(websocket: Any) -> dict[str, Any]:
    for _ in range(200):
        message = websocket.receive_json()
        if message.get("type") in {"task_error", "agent_error", "error"}:
            return message
        if message.get("type") == "task_completed":
            raise AssertionError(f"Expected task_error, got completion: {message}")
    raise AssertionError("Timed out waiting for websocket execution error")


def test_task_uploads_agent_outputs_and_startup_sync_persist_to_minio(
    persistence_app: PersistenceApp,
    minio_storage: MinioStorage,
) -> None:
    client: TestClient = persistence_app.client
    headers = persistence_app.headers
    session_factory = persistence_app.session_factory
    user_id = persistence_app.user_id
    startup_repair_file_id = persistence_app.startup_repair_file_id
    token = persistence_app.token

    _wait_for_startup_file_storage_sync(client)
    startup_repair_record = _fetch_file_record(session_factory, startup_repair_file_id)
    assert startup_repair_record.storage_backend == "s3"
    assert startup_repair_record.storage_status == "available"
    assert startup_repair_record.storage_key == (
        f"users/{user_id}/uploads/{startup_repair_file_id}/startup-repair.txt"
    )
    assert minio_storage.object_bytes(startup_repair_record.storage_key) == (
        b"startup repair content\n"
    )

    create_response = client.post(
        "/api/chat/task/create",
        json={
            "title": "MinIO output task",
            "description": "Create static-output.txt",
            "execution_mode": "balanced",
        },
        headers=headers,
    )

    assert create_response.status_code == 200
    task_id = create_response.json()["task_id"]
    db = session_factory()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.PENDING
        assert task.execution_mode == "balanced"
    finally:
        db.close()

    response = client.post(
        "/api/files/upload",
        files=[
            ("files", ("alpha.txt", b"alpha upload\n", "text/plain")),
            ("files", ("beta.txt", b"beta upload\n", "text/plain")),
        ],
        data={"task_type": "general", "task_id": str(task_id)},
        headers=headers,
    )

    assert response.status_code == 200
    uploaded = response.json()["files"]
    assert len(uploaded) == 2
    for item, expected_content in zip(uploaded, [b"alpha upload\n", b"beta upload\n"]):
        record = _fetch_file_record(session_factory, item["file_id"])
        assert record.task_id == task_id
        assert record.storage_backend == "s3"
        assert record.storage_status == "available"
        assert record.storage_key
        assert record.file_size == len(expected_content)
        assert minio_storage.object_bytes(record.storage_key) == expected_content

    with client.websocket_connect(f"/ws/chat/{task_id}?token={token}") as websocket:
        _receive_until_type(websocket, "trace_event")
        websocket.send_json({"type": "execute_task"})
        execution_started = _receive_until_type(websocket, "execution_started")
        assert execution_started["type"] == "execution_started"
        completion_message = _receive_until_task_completed(websocket)

    assert completion_message["success"] is True
    db = session_factory()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.COMPLETED
        output_record = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.user_id == user_id,
                UploadedFile.task_id == task_id,
                UploadedFile.filename == "static-output.txt",
            )
            .one()
        )
        assert output_record.storage_backend == "s3"
        assert output_record.storage_status == "available"
        assert output_record.workspace_category == "output"
        assert output_record.workspace_relative_path == "output/static-output.txt"
        _assert_task_output_generation_key(
            output_record.storage_key,
            user_id=user_id,
            task_id=task_id,
            file_id=output_record.file_id,
            relative_path="output/static-output.txt",
        )
        assert minio_storage.object_bytes(output_record.storage_key) == (
            b"minio persistence e2e output\n"
        )
    finally:
        db.close()


def test_websocket_output_persistence_sends_error_and_rolls_back_when_minio_write_fails(
    persistence_app: PersistenceApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client: TestClient = persistence_app.client
    headers = persistence_app.headers
    session_factory = persistence_app.session_factory
    token = persistence_app.token

    create_response = client.post(
        "/api/chat/task/create",
        json={
            "title": "MinIO output outage task",
            "description": "Create static-output.txt while MinIO output write fails",
            "execution_mode": "balanced",
        },
        headers=headers,
    )

    assert create_response.status_code == 200
    task_id = create_response.json()["task_id"]

    from xagent.core.file_storage.storage import FsspecFileStorage

    real_put_file = FsspecFileStorage.put_file

    def fail_task_output_put_file(
        self: FsspecFileStorage,
        source: Path,
        key: str,
        content_type: str | None = None,
    ) -> Any:
        if f"/tasks/{task_id}/outputs/" in f"/{key}":
            raise RuntimeError("simulated MinIO output write outage")
        return real_put_file(self, source, key, content_type)

    monkeypatch.setattr(FsspecFileStorage, "put_file", fail_task_output_put_file)

    with client.websocket_connect(f"/ws/chat/{task_id}?token={token}") as websocket:
        _receive_until_type(websocket, "trace_event")
        websocket.send_json({"type": "execute_task"})
        execution_started = _receive_until_type(websocket, "execution_started")
        assert execution_started["type"] == "execution_started"
        error_message = _receive_until_execution_error(websocket)

    error_text = str(
        error_message.get("error") or error_message.get("message") or ""
    ).lower()
    assert "durable object" in error_text

    db = session_factory()
    try:
        assert (
            db.query(UploadedFile)
            .filter(
                UploadedFile.task_id == task_id,
                UploadedFile.filename == "static-output.txt",
            )
            .first()
            is None
        )
    finally:
        db.close()


def test_chat_task_materializes_missing_local_input_from_minio_before_agent_reads_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minio_storage: MinioStorage,
) -> None:
    app_generator = run_file_persistence_app(
        monkeypatch,
        tmp_path,
        llm_responses_path=INPUT_MATERIALIZATION_LLM_RESPONSES_PATH,
    )
    app = next(app_generator)
    try:
        client: TestClient = app.client
        headers = app.headers
        session_factory = app.session_factory
        user_id = app.user_id
        token = app.token

        create_response = client.post(
            "/api/chat/task/create",
            json={
                "title": "MinIO input task",
                "description": "Read source.txt and create a derived output",
                "execution_mode": "balanced",
            },
            headers=headers,
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        upload_response = client.post(
            "/api/files/upload",
            files={"file": ("source.txt", b"source from minio\n", "text/plain")},
            data={"task_type": "general", "task_id": str(task_id)},
            headers=headers,
        )
        assert upload_response.status_code == 200
        uploaded_file_id = upload_response.json()["file_id"]
        uploaded_record = _fetch_file_record(session_factory, uploaded_file_id)
        uploaded_local_path = Path(str(uploaded_record.storage_path))
        uploaded_storage_key = str(uploaded_record.storage_key)
        assert minio_storage.object_bytes(uploaded_storage_key) == (
            b"source from minio\n"
        )
        uploaded_local_path.unlink()

        with client.websocket_connect(f"/ws/chat/{task_id}?token={token}") as websocket:
            _receive_until_type(websocket, "trace_event")
            websocket.send_json(
                {
                    "type": "chat",
                    "message": "Read source.txt and create derived-from-input.txt",
                    "files": [
                        {
                            "file_id": uploaded_file_id,
                            "name": "source.txt",
                            "size": len(b"source from minio\n"),
                            "type": "text/plain",
                        }
                    ],
                }
            )
            completion_message = _receive_until_task_completed(websocket)

        assert completion_message["success"] is True
        assert uploaded_local_path.read_bytes() == b"source from minio\n"

        db = session_factory()
        try:
            task = db.query(Task).filter(Task.id == task_id).one()
            assert task.status == TaskStatus.COMPLETED
            output_record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.user_id == user_id,
                    UploadedFile.task_id == task_id,
                    UploadedFile.filename == "derived-from-input.txt",
                )
                .one()
            )
            assert output_record.storage_backend == "s3"
            assert output_record.storage_status == "available"
            assert minio_storage.object_bytes(str(output_record.storage_key)) == (
                b"derived from uploaded input: source from minio\n"
            )
        finally:
            db.close()
    finally:
        app_generator.close()
