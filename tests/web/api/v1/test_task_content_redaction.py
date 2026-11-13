"""Content redaction of a task whose row and usage must survive.

``redact_task_content`` is what a host calls when a user deletes their
conversation history while billing keeps reading the task's token counters.
These tests pin the split it promises -- every content row, file and object
gone; the task row, its metering columns and ``updated_at`` intact -- plus
idempotency, the busy refusal, and that the read surfaces still answer for
a redacted task rather than failing on missing content.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.shared.auth_database import auth_db_override
from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import set_execution_scope_snapshot_loader
from xagent.core.file_storage.factory import (
    get_unscoped_file_storage,
    get_user_file_storage,
)
from xagent.web.api.chat import chat_router
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import get_db
from xagent.web.models.task import (
    DAGExecution,
    Task,
    TaskConnectorRuntimeContext,
    TaskStatus,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.task_runtime_secret import TaskRuntimeSecret
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import task_content_redaction
from xagent.web.services.agent_service_manager import get_agent_manager
from xagent.web.services.execution_scope_snapshot import (
    load_task_execution_scope_snapshot,
)
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_PENDING,
)
from xagent.web.services.task_content_redaction import (
    TaskContentRedactionReport,
    redact_task_content,
)
from xagent.web.services.uploaded_file_store import UploadedFileStore

from ...services.task_interaction_schema_shared import make_row
from ..conftest import _admin_headers, _direct_db_session, _override_get_db, client
from .test_v1_delete_task import (
    _bearer,
    _create_agent_with_key,
    _create_task,
    _force_status,
)

pytestmark = pytest.mark.usefixtures("_test_db")

_chat_app = FastAPI()
_chat_app.include_router(chat_router)
_chat_app.dependency_overrides[get_db] = _override_get_db
_chat_app.dependency_overrides[get_auth_db] = auth_db_override(_override_get_db)
chat_client = TestClient(_chat_app)

#: The content columns the redaction blanks, with the value each is seeded at.
CONTENT_COLUMNS: dict[str, Any] = {
    "title": "secret title",
    "description": "secret description",
    "input": "secret input",
    "output": "secret output",
    "error_message": "secret error",
    "structured_output_record": {"output": "secret"},
    "process_description": "secret process",
    "examples": [{"input": "secret"}],
}

#: Columns billing and routing read, with the value each is seeded at.
KEPT_COLUMNS: dict[str, Any] = {
    "input_tokens": 120,
    "output_tokens": 45,
    "total_tokens": 165,
    "llm_calls": 3,
    "token_usage_details": {"model-a": {"input_tokens": 120}},
    "model_id": "model-a",
    "model_name": "provider/model-a",
    "completion_outcome": "completed",
}

UPDATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture(autouse=True)
def mock_start_task() -> Iterator[None]:
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ):
        yield


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point uploads and durable storage at ``tmp_path``; yields the uploads root."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    register_scope_resolver(None)
    set_execution_scope_snapshot_loader(load_task_execution_scope_snapshot)
    get_agent_manager()._agents.clear()

    yield uploads

    set_execution_scope_snapshot_loader(None)
    get_unscoped_file_storage.cache_clear()


def _object_exists(owner_id: int, storage_key: str) -> bool:
    return get_user_file_storage(owner_id).exists(storage_key)


def _settled_task(full_key: str, agent_id: int) -> int:
    """A COMPLETED SDK task carrying every content column and usage counter."""
    task_id = _create_task(full_key, agent_id)
    _force_status(task_id, TaskStatus.COMPLETED)
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(
            {**CONTENT_COLUMNS, **KEPT_COLUMNS, "updated_at": UPDATED_AT}
        )
        db.commit()
    finally:
        db.close()
    return task_id


def _seed_content(task_id: int, uploads: Path) -> tuple[str, Path, Path]:
    """One row of every content table, an uploaded file and a workspace.

    Returns the file's storage key, its local path and the workspace dir.
    """
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        owner_id = int(task.user_id)
        trace = TraceEvent(
            task_id=task_id,
            event_id=f"event-{task_id}",
            event_type="agent_execution_checkpoint",
            timestamp=datetime.now(UTC),
            data={"content": "secret trace"},
        )
        db.add(trace)
        db.flush()
        db.add_all(
            [
                TraceMessageBlob(
                    task_id=task_id,
                    execution_id="exec",
                    message_hash=f"m{task_id}",
                    message_data={"content": "secret"},
                    message_bytes=8,
                ),
                TraceCheckpointBlob(
                    task_id=task_id,
                    execution_id="exec",
                    blob_kind="messages",
                    blob_hash=f"c{task_id}",
                    blob_data={"content": "secret"},
                    blob_bytes=8,
                ),
                DAGExecution(task_id=task_id, current_plan={"goal": "secret"}),
                TaskChatMessage(
                    task_id=task_id,
                    user_id=owner_id,
                    role="user",
                    content="secret message",
                    message_type="user_message",
                ),
                TaskExecutionEvent(
                    event_id=f"exec-event-{task_id}",
                    task_id=task_id,
                    sequence=1,
                    scope_id="root",
                    idempotency_key="k1",
                    kind="assistant_message",
                    payload_version=1,
                    payload={"content": "secret"},
                    occurred_at=datetime.now(UTC),
                ),
                TaskConnectorRuntimeContext(
                    task_id=task_id,
                    connector_type="mcp",
                    connector_id=1,
                    context={"account": "secret"},
                ),
                TaskRuntimeSecret(
                    task_id=task_id,
                    turn_id="turn",
                    run_id="run",
                    owner_subject="owner",
                    ciphertext="secret",
                ),
                TaskAutoRecovery(
                    task_id=task_id,
                    run_id="run",
                    reason="worker_lost",
                    state="exhausted",
                    paused_state_version=1,
                    interrupted_at=datetime.now(UTC),
                    episode_started_at=datetime.now(UTC),
                ),
                TaskRecoveryEvent(task_id=task_id, run_id="run", event="interrupted"),
            ]
        )
        # An active interaction anchored on the trace row: the ordering
        # that keeps its CHECK satisfied is what this covers.
        db.add(
            TaskInteractionRequest(
                **make_row(task_id=task_id, resume_trace_event_id=int(trace.id), db=db)
            )
        )
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                command_id=f"turn-{task_id}",
                kind="message",
                payload={"content": "secret"},
                result={"output": "secret"},
                error="secret",
                status=COMMAND_COMPLETED,
            )
        )
        db.query(Task).filter(Task.id == task_id).update(
            {
                "last_checkpoint_event_id": trace.event_id,
                "last_checkpoint_trace_event_id": trace.id,
                "interaction_protocol_version": 1,
                "updated_at": UPDATED_AT,
            }
        )
        db.commit()

        local = uploads / f"user_{owner_id}" / "uploads" / f"attachment-{task_id}.txt"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("secret file")
        record = UploadedFileStore(db).create_from_local_path(
            local_path=local, user_id=owner_id, task_id=task_id
        )
        db.commit()
        storage_key = str(record.storage_key)
        assert _object_exists(owner_id, storage_key)
    finally:
        db.close()

    workspace = uploads / f"web_task_{task_id}"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("secret output file")
    return storage_key, local, workspace


def _content_counts(task_id: int) -> dict[str, int]:
    db = _direct_db_session()
    try:
        return {
            model.__tablename__: db.query(model)
            .filter(model.task_id == task_id)
            .count()
            for model in (
                TraceEvent,
                TraceMessageBlob,
                TraceCheckpointBlob,
                DAGExecution,
                TaskChatMessage,
                TaskExecutionEvent,
                TaskConnectorRuntimeContext,
                TaskRuntimeSecret,
                TaskAutoRecovery,
                TaskRecoveryEvent,
                TaskInteractionRequest,
                UploadedFile,
            )
        }
    finally:
        db.close()


def _redact(task_ids: list[int]) -> TaskContentRedactionReport:
    db = _direct_db_session()
    try:
        return redact_task_content(db, task_ids)
    finally:
        db.close()


def test_redaction_removes_content_and_keeps_the_task_and_its_usage(
    _roots: Path,
) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _settled_task(full_key, agent_id)
    bystander_id = _settled_task(full_key, agent_id)
    storage_key, local, workspace = _seed_content(task_id, _roots)
    _seed_content(bystander_id, _roots)

    report = _redact([task_id, task_id, 9_000_001])

    assert report == TaskContentRedactionReport(
        redacted=(task_id,), busy=(), missing=(9_000_001,)
    )
    assert set(_content_counts(task_id).values()) == {0}
    assert not local.exists()
    assert not workspace.exists()
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        owner_id = int(task.user_id)
        assert not _object_exists(owner_id, storage_key)
        assert {name: getattr(task, name) for name in CONTENT_COLUMNS} == {
            "title": "",
            **{name: None for name in CONTENT_COLUMNS if name != "title"},
        }
        assert {name: getattr(task, name) for name in KEPT_COLUMNS} == KEPT_COLUMNS
        assert task.status == TaskStatus.COMPLETED
        assert task.agent_id == agent_id
        assert task.updated_at.replace(tzinfo=UTC) == UPDATED_AT
        assert task.traces_expired_at is not None
        assert task.last_checkpoint_event_id is None
        assert task.last_checkpoint_trace_event_id is None
        assert task.interaction_protocol_version is None
        commands = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .all()
        )
        assert commands
        assert all(
            (command.payload, command.result, command.error) == ({}, None, None)
            for command in commands
        )
    finally:
        db.close()
    assert 0 not in _content_counts(bystander_id).values()


def test_redaction_is_idempotent(_roots: Path) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _settled_task(full_key, agent_id)
    _seed_content(task_id, _roots)

    first = _redact([task_id])
    second = _redact([task_id])

    assert first == second == TaskContentRedactionReport(redacted=(task_id,))
    assert set(_content_counts(task_id).values()) == {0}
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.input_tokens == KEPT_COLUMNS["input_tokens"]
        assert task.updated_at.replace(tzinfo=UTC) == UPDATED_AT
    finally:
        db.close()


def test_a_cleanup_failure_is_reported_and_the_batch_continues(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_id, full_key = _create_agent_with_key()
    failing_id = _settled_task(full_key, agent_id)
    next_id = _settled_task(full_key, agent_id)
    _seed_content(failing_id, _roots)
    _seed_content(next_id, _roots)
    manager = get_agent_manager()
    remove_agent = manager.remove_agent

    def flaky_remove_agent(task_id: int, *args: Any, **kwargs: Any) -> None:
        if task_id == failing_id:
            raise OSError("workspace is busy")
        remove_agent(task_id, *args, **kwargs)

    monkeypatch.setattr(manager, "remove_agent", flaky_remove_agent)

    report = _redact([failing_id, next_id])

    assert report == TaskContentRedactionReport(
        redacted=(next_id,), cleanup_failed=(failing_id,)
    )
    # The rows were committed before the cleanup ran.
    assert set(_content_counts(failing_id).values()) == {0}
    assert (_roots / f"web_task_{failing_id}").exists()
    assert not (_roots / f"web_task_{next_id}").exists()


@pytest.mark.parametrize("new_turn_after_commit", [False, True])
def test_cleanup_spares_a_runtime_a_later_turn_acquired(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, new_turn_after_commit: bool
) -> None:
    """The eviction is fenced on the runtime identity read under the lock.

    A turn that acquires the runtime between the commit and the cleanup owns
    the cached agent and its workspace; the cleanup must leave both alone.
    """
    agent_id, full_key = _create_agent_with_key()
    task_id = _settled_task(full_key, agent_id)
    _, _, workspace = _seed_content(task_id, _roots)
    manager = get_agent_manager()
    cached_agent = MagicMock()
    manager._agents[task_id] = cached_agent
    manager._agent_run_generations[task_id] = 1

    if new_turn_after_commit:

        def acquire_after_commit(_task_id: int) -> None:
            # Runs after the commit, before the eviction: a new acquisition.
            manager._agent_run_generations[task_id] = 2

        monkeypatch.setattr(
            task_content_redaction, "invalidate_task_cache", acquire_after_commit
        )

    report = _redact([task_id])

    assert report == TaskContentRedactionReport(redacted=(task_id,))
    if new_turn_after_commit:
        assert manager._agents[task_id] is cached_agent
        cached_agent.cleanup_workspace.assert_not_called()
        assert workspace.exists()
    else:
        assert task_id not in manager._agents
        cached_agent.cleanup_workspace.assert_called_once()
    manager._agents.pop(task_id, None)
    manager._agent_run_generations.pop(task_id, None)


@pytest.mark.parametrize("state", ["running", "pending_command"])
def test_redaction_leaves_a_busy_task_untouched(_roots: Path, state: str) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _settled_task(full_key, agent_id)
    _seed_content(task_id, _roots)
    db = _direct_db_session()
    try:
        if state == "running":
            db.query(Task).filter(Task.id == task_id).update(
                {"status": TaskStatus.RUNNING}
            )
        else:
            db.add(
                TaskExecutionCommand(
                    task_id=task_id,
                    command_id="queued-turn",
                    kind="message",
                    payload={"content": "queued"},
                    status=COMMAND_PENDING,
                )
            )
        db.commit()
    finally:
        db.close()
    before = _content_counts(task_id)

    report = _redact([task_id])

    assert report == TaskContentRedactionReport(busy=(task_id,))
    assert _content_counts(task_id) == before
    db = _direct_db_session()
    try:
        assert (
            db.query(Task.title).filter(Task.id == task_id).scalar()
            == (CONTENT_COLUMNS["title"])
        )
    finally:
        db.close()


def test_read_surfaces_answer_for_a_redacted_task(_roots: Path) -> None:
    agent_id, full_key = _create_agent_with_key()
    task_id = _settled_task(full_key, agent_id)
    _seed_content(task_id, _roots)
    headers = _admin_headers()
    # Warm the detail cache: the redaction pins ``updated_at``, the cache's
    # freshness token, so only its invalidation keeps this from being served.
    assert (
        chat_client.get(f"/api/chat/task/{task_id}", headers=headers).json()["title"]
        == (CONTENT_COLUMNS["title"])
    )

    _redact([task_id])

    detail = chat_client.get(f"/api/chat/task/{task_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["title"] == ""
    assert detail.json()["dag_data"] is None
    assert detail.json()["input_tokens"] == KEPT_COLUMNS["input_tokens"]
    assert (
        chat_client.get(f"/api/chat/task/{task_id}/status", headers=headers).status_code
        == 200
    )
    listed = chat_client.get("/api/chat/tasks", headers=headers)
    assert listed.status_code == 200, listed.text

    v1_task = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert v1_task.status_code == 200, v1_task.text
    assert v1_task.json()["output"] is None
    steps = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert steps.status_code == 200, steps.text
    assert steps.json()["steps"] == []
    assert steps.json()["steps_expired"] is True
    files = client.get(f"/v1/chat/tasks/{task_id}/files", headers=_bearer(full_key))
    assert files.status_code == 200, files.text
    assert files.json()["files"] == []

    log = client.get(f"/api/conversation-logs/{task_id}", headers=headers)
    assert log.status_code == 200, log.text
    assert log.json()["log"]["title"] == ""
