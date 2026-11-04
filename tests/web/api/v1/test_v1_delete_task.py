"""Integration tests for ``DELETE /v1/chat/tasks/{task_id}``.

The endpoint deletes through the same shared orchestration as the Web UI
delete, so these tests pin the v1-specific parts (run-key ownership, the
running-task refusal, the 200/404 surface) and the shared cleanup outcome
(rows gone, workspace removed, pending obligations recorded).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import xagent.web.services.task_delete as task_delete_module
from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import set_execution_scope_snapshot_loader
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import (
    DAGExecution,
    Task,
    TaskConnectorRuntimeContext,
    TaskStatus,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_runtime_secret import TaskRuntimeSecret
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.agent_service_manager import get_agent_manager
from xagent.web.services.execution_scope_snapshot import (
    load_task_execution_scope_snapshot,
)
from xagent.web.services.task_cleanup_obligations import (
    CleanupObligationStatus,
    CleanupResourceKind,
    list_cleanup_obligations,
)
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_PENDING,
    COMMAND_PROCESSING,
)
from xagent.web.services.task_delete import (
    capture_workspace_cleanup_target_best_effort,
)
from xagent.web.services.task_runtime import (
    agent_config_with_task_extension_bindings,
    register_task_extension,
    unregister_task_extension,
)

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_start_task():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        yield


@pytest.fixture(autouse=True)
def _workspace_root(tmp_path, monkeypatch) -> Iterator[Path]:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")
    register_scope_resolver(None)
    set_execution_scope_snapshot_loader(load_task_execution_scope_snapshot)

    manager = get_agent_manager()
    manager._agents.clear()
    manager._agent_owner_ids.clear()
    manager._agent_run_ids.clear()
    manager._agent_run_generations.clear()
    manager._agent_cleanup_owner_ids.clear()

    yield uploads

    set_execution_scope_snapshot_loader(None)


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _create_agent_with_key(
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    headers = headers or _admin_headers()
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": f"v1 delete task agent {uuid.uuid4().hex[:8]}",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _create_task(full_key: str, agent_id: int) -> int:
    response = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": agent_id,
            "message": {"role": "user", "content": "delete me"},
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


def _force_status(task_id: int, status: TaskStatus) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = status
        task.control_state = status.value
        if status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            # A finished turn has released its lease and settled its start
            # command; creating the task claimed the first and may stage the other.
            db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.task_id == task_id
            ).update({"status": COMMAND_COMPLETED})
            task.lease_expires_at = None
        db.commit()
    finally:
        db.close()


def _seed_related_rows(
    *,
    task_id: int,
    owner_id: int,
    uploads: Path,
) -> None:
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"event-{task_id}",
                event_type="ai_message",
                timestamp=datetime.now(UTC),
                data={"content": "trace"},
            )
        )
        db.add(
            TraceMessageBlob(
                task_id=task_id,
                execution_id=f"exec-{task_id}",
                message_hash=f"hash-{task_id}",
                message_data={"content": "blob"},
                message_bytes=12,
            )
        )
        db.add(
            TraceCheckpointBlob(
                task_id=task_id,
                execution_id=f"exec-{task_id}",
                blob_kind="messages",
                blob_hash=f"blob-{task_id}",
                blob_data={"content": "checkpoint"},
                blob_bytes=16,
            )
        )
        db.add(
            TaskChatMessage(
                task_id=task_id,
                user_id=owner_id,
                role="user",
                content="delete me",
                message_type="user_message",
            )
        )
        db.add(DAGExecution(task_id=task_id))
        db.add(
            TaskConnectorRuntimeContext(
                task_id=task_id,
                connector_type="mcp",
                connector_id=1,
                context={"account_id": "acct"},
            )
        )
        db.add(
            TaskRuntimeSecret(
                task_id=task_id,
                turn_id=f"turn-{task_id}",
                run_id=f"run-{task_id}",
                owner_subject="owner",
                ciphertext="ciphertext",
            )
        )
        db.add(
            UploadedFile(
                user_id=owner_id,
                task_id=task_id,
                filename="attachment.txt",
                storage_path=str(uploads / f"attachment-{task_id}.txt"),
                file_size=4,
            )
        )
        db.commit()
    finally:
        db.close()


def _bound_file_count(task_id: int) -> int:
    db = _direct_db_session()
    try:
        return int(
            db.query(UploadedFile).filter(UploadedFile.task_id == task_id).count()
        )
    finally:
        db.close()


def _make_workspace(uploads: Path, task_id: int) -> Path:
    workspace = uploads / f"web_task_{task_id}"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("payload")
    return workspace


def test_owner_delete_removes_rows_workspace_and_readability(
    _workspace_root: Path,
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Task).filter(Task.id == task_id).one().user_id)
    finally:
        db.close()
    _seed_related_rows(task_id=task_id, owner_id=owner_id, uploads=_workspace_root)
    workspace = _make_workspace(_workspace_root, task_id)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 200, response.text
    assert response.json() == {
        "task_id": task_id,
        "workspace_cleanup_pending": False,
        "external_cleanup_pending": False,
    }
    assert not workspace.exists()

    for path in (
        f"/v1/chat/tasks/{task_id}",
        f"/v1/chat/tasks/{task_id}/steps",
        f"/v1/chat/tasks/{task_id}/events",
    ):
        read = client.get(path, headers=_bearer(full_key))
        assert read.status_code == 404, (path, read.text)
        assert read.json()["error"]["code"] == "task_not_found"

    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 0
        assert db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count() == 0
        assert (
            db.query(TraceMessageBlob)
            .filter(TraceMessageBlob.task_id == task_id)
            .count()
            == 0
        )
        assert (
            db.query(TraceCheckpointBlob)
            .filter(TraceCheckpointBlob.task_id == task_id)
            .count()
            == 0
        )
        assert (
            db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).count()
            == 0
        )
        assert (
            db.query(DAGExecution).filter(DAGExecution.task_id == task_id).count() == 0
        )
        assert (
            db.query(TaskConnectorRuntimeContext)
            .filter(TaskConnectorRuntimeContext.task_id == task_id)
            .count()
            == 0
        )
        assert (
            db.query(TaskRuntimeSecret)
            .filter(TaskRuntimeSecret.task_id == task_id)
            .count()
            == 0
        )
        assert _bound_file_count(task_id) == 0
        files = db.query(UploadedFile).filter(UploadedFile.filename == "attachment.txt")
        detached = [
            row for row in files if row.storage_path.endswith(f"-{task_id}.txt")
        ]
        assert len(detached) == 1
        assert detached[0].task_id is None
        assert detached[0].detached_reason == "task_deleted"
        assert list_cleanup_obligations(db) == []
    finally:
        db.close()


def test_delete_refuses_another_agents_task_without_touching_it(
    tmp_path: Path,
) -> None:
    agent_a, key_a = _create_agent_with_key()
    _ = agent_a
    agent_b, key_b = _create_agent_with_key()
    task_id = _create_task(key_b, agent_b)
    _force_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Task).filter(Task.id == task_id).one().user_id)
    finally:
        db.close()
    _seed_related_rows(task_id=task_id, owner_id=owner_id, uploads=tmp_path)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(key_a))

    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "task_not_found"
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count() == 1
    finally:
        db.close()


def test_delete_of_another_users_task_is_404() -> None:
    _admin_headers()
    other_headers = _register_second_user("delete-other", "delete-other-pass1")
    other_agent, other_key = _create_agent_with_key(other_headers)
    task_id = _create_task(other_key, other_agent)
    _force_status(task_id, TaskStatus.COMPLETED)
    _agent, own_key = _create_agent_with_key()

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(own_key))

    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "task_not_found"


def test_delete_missing_task_and_second_delete_are_404() -> None:
    agent_id, full_key = _create_agent_with_key()
    missing = client.delete("/v1/chat/tasks/987654321", headers=_bearer(full_key))
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "task_not_found"

    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    first = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert first.status_code == 200, first.text
    second = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert second.status_code == 404
    assert second.json()["error"]["code"] == "task_not_found"


@pytest.mark.parametrize("status", [TaskStatus.PENDING, TaskStatus.RUNNING])
def test_delete_refuses_an_active_task(status: TaskStatus, tmp_path: Path) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, status)
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Task).filter(Task.id == task_id).one().user_id)
    finally:
        db.close()
    _seed_related_rows(task_id=task_id, owner_id=owner_id, uploads=tmp_path)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count() == 1
    finally:
        db.close()


@pytest.mark.parametrize(
    "control_state",
    ["running", "pause_requested", "resume_requested"],
)
def test_delete_refuses_an_active_control_state(
    control_state: str, tmp_path: Path
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.control_state = control_state
        owner_id = int(task.user_id)
        db.commit()
    finally:
        db.close()
    _seed_related_rows(task_id=task_id, owner_id=owner_id, uploads=tmp_path)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count() == 1
    finally:
        db.close()


def _stage_command(task_id: int, status: str) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=int(task.user_id),
                command_id=f"start:{task_id}:{status}",
                kind="start",
                payload={},
                status=status,
            )
        )
        db.commit()
    finally:
        db.close()


@pytest.mark.parametrize("command_status", [COMMAND_PENDING, COMMAND_PROCESSING])
def test_delete_refuses_a_task_with_a_live_command(
    command_status: str, tmp_path: Path
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    _stage_command(task_id, command_status)
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Task).filter(Task.id == task_id).one().user_id)
    finally:
        db.close()
    _seed_related_rows(task_id=task_id, owner_id=owner_id, uploads=tmp_path)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .count()
            == 1
        )
        assert db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count() == 1
    finally:
        db.close()


def test_delete_ignores_a_finished_command(tmp_path: Path) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    _stage_command(task_id, COMMAND_COMPLETED)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 200, response.text


def test_delete_refuses_a_task_with_an_unexpired_lease() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)
        db.commit()
    finally:
        db.close()

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.lease_expires_at = datetime.now(UTC) - timedelta(minutes=5)
        db.commit()
    finally:
        db.close()
    expired = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert expired.status_code == 200, expired.text


def test_delete_rechecks_activity_inside_the_deletion_transaction(
    _workspace_root: Path,
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        owner_id = int(db.query(Task).filter(Task.id == task_id).one().user_id)
    finally:
        db.close()
    _seed_related_rows(task_id=task_id, owner_id=owner_id, uploads=_workspace_root)
    workspace = _make_workspace(_workspace_root, task_id)
    flipped = {"done": False}

    def _flip_status_then_capture(*args, **kwargs):
        if not flipped["done"]:
            flipped["done"] = True
            _force_status(task_id, TaskStatus.RUNNING)
        return capture_workspace_cleanup_target_best_effort(*args, **kwargs)

    with patch(
        "xagent.web.services.task_delete.capture_workspace_cleanup_target_best_effort",
        side_effect=_flip_status_then_capture,
    ):
        response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert flipped["done"] is True
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    assert workspace.exists()
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert db.query(TraceEvent).filter(TraceEvent.task_id == task_id).count() == 1
        assert (
            db.query(TaskChatMessage).filter(TaskChatMessage.task_id == task_id).count()
            >= 1
        )
    finally:
        db.close()


def test_delete_ignores_a_same_agent_task_not_created_by_sdk() -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.source = "web"
        db.commit()
    finally:
        db.close()

    read = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert read.status_code == 404, read.text
    assert read.json()["error"]["code"] == "task_not_found"
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "task_not_found"
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
    finally:
        db.close()


def test_failed_workspace_removal_leaves_a_pending_obligation(
    _workspace_root: Path, monkeypatch
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    _make_workspace(_workspace_root, task_id)

    def _explode(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(
        type(get_agent_manager()), "_cleanup_workspace_directory", _explode
    )

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 200, response.text
    assert response.json()["workspace_cleanup_pending"] is True
    assert response.json()["external_cleanup_pending"] is True
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 0
        [owed] = list_cleanup_obligations(db)
        assert owed.task_id == task_id
        assert owed.kind is CleanupResourceKind.WORKSPACE
        assert owed.status is CleanupObligationStatus.PENDING
        assert owed.attempts == 1
        assert "permission denied" in (owed.last_error or "")
    finally:
        db.close()


class _Provider:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.deleted: list[int] = []

    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context):
        return None

    async def public_metadata(self, context):
        return None

    async def on_task_deleted(self, context) -> None:
        if self.fail_delete:
            raise RuntimeError("provider is down")
        self.deleted.append(int(context.task_id))


@pytest.fixture
def provider() -> Iterator[tuple[str, _Provider]]:
    name = "delete_test_ext"
    instance = _Provider()
    register_task_extension(name, instance)
    try:
        yield name, instance
    finally:
        unregister_task_extension(name)


@pytest.fixture
def failing_provider() -> Iterator[str]:
    name = "delete_failing_ext"
    register_task_extension(name, _Provider(fail_delete=True))
    try:
        yield name
    finally:
        unregister_task_extension(name)


def _bind_extension(task_id: int, name: str) -> None:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.agent_config = agent_config_with_task_extension_bindings(
            task.agent_config, [name]
        )
        db.commit()
    finally:
        db.close()


def test_bound_extension_state_is_released(
    provider: tuple[str, _Provider],
) -> None:
    name, instance = provider
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    _bind_extension(task_id, name)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 200, response.text
    assert instance.deleted == [task_id]
    assert response.json()["external_cleanup_pending"] is False
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 0
        assert list_cleanup_obligations(db) == []
    finally:
        db.close()


def test_failing_bound_extension_preserves_the_task(failing_provider: str) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    _bind_extension(task_id, failing_provider)

    response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "internal_error"
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert list_cleanup_obligations(db) == []
    finally:
        db.close()


def test_active_task_is_refused_before_extension_state_is_released(
    provider: tuple[str, _Provider],
) -> None:
    name, instance = provider
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    _bind_extension(task_id, name)
    flipped = {"done": False}
    original = task_delete_module._refuse_if_task_active_sync

    def _flip_status_then_check(task_id_arg: int) -> None:
        if not flipped["done"]:
            flipped["done"] = True
            _force_status(task_id, TaskStatus.RUNNING)
        original(task_id_arg)

    with patch(
        "xagent.web.services.task_delete._refuse_if_task_active_sync",
        side_effect=_flip_status_then_check,
    ):
        response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert flipped["done"] is True
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    assert instance.deleted == []


def test_activity_after_the_unlocked_check_refuses_but_the_release_has_happened(
    provider: tuple[str, _Provider],
) -> None:
    """Pins the known window: the locked re-check still refuses the delete,
    but provider state was released before it ran."""
    name, instance = provider
    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    _bind_extension(task_id, name)
    flipped = {"done": False}

    def _flip_status_then_capture(*args, **kwargs):
        if not flipped["done"]:
            flipped["done"] = True
            _force_status(task_id, TaskStatus.RUNNING)
        return capture_workspace_cleanup_target_best_effort(*args, **kwargs)

    with patch(
        "xagent.web.services.task_delete.capture_workspace_cleanup_target_best_effort",
        side_effect=_flip_status_then_capture,
    ):
        response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert flipped["done"] is True
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "task_busy"
    assert instance.deleted == [task_id]
    db = _direct_db_session()
    try:
        assert db.query(Task).filter(Task.id == task_id).count() == 1
    finally:
        db.close()


def test_delete_detaches_and_closes_the_tasks_websocket_connections() -> None:
    from xagent.web.api.websocket import manager as websocket_manager

    agent_id, full_key = _create_agent_with_key()
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    connection = MagicMock()
    connection.close = AsyncMock()

    with patch.object(
        websocket_manager, "detach_task_connections", return_value=[connection]
    ) as detach:
        response = client.delete(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))

    assert response.status_code == 200, response.text
    detach.assert_called_once_with(task_id)
    connection.close.assert_awaited_once()
