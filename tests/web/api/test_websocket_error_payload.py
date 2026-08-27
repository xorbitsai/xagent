import sys
import threading
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import _terminal_task_error_payload
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_lease_service import TaskLease, get_runner_id

from .conftest import _direct_db_session


def test_terminal_task_error_payload_marks_unowned_task_failed(_test_db):
    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Failing task",
            description="Failing task",
            status=TaskStatus.RUNNING,
            runner_id=None,
            lease_expires_at=None,
        )
        db.add(task)
        db.commit()
        task_id = task.id

        task_updates: list[str] = []

        def record_task_update(
            _connection,
            _cursor,
            statement,
            _parameters,
            _context,
            _executemany,
        ):
            if statement.lstrip().lower().startswith("update tasks"):
                task_updates.append(statement)

        engine = get_engine()
        event.listen(engine, "before_cursor_execute", record_task_update)
        try:
            payload = _terminal_task_error_payload(task_id, "Runtime error")
        finally:
            event.remove(engine, "before_cursor_execute", record_task_update)

        assert payload["type"] == "agent_error"
        assert payload["message"] == "Runtime error"
        assert payload["task"]["id"] == task_id
        assert payload["task"]["status"] == "failed"
        assert len(task_updates) == 1
        assert "tasks.runner_id IS NULL" in task_updates[0]

        db.expire_all()
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.FAILED
        assert persisted_task.runner_id is None
        assert persisted_task.lease_expires_at is None
        assert persisted_task.error_message == "Runtime error"
    finally:
        db.close()


def test_terminal_task_error_payload_persists_error_chat_message(_test_db):
    """Failures before agent execution (no trace events, e.g. sandbox
    capacity rejection) persist a client-safe assistant message while the
    task keeps the diagnostic detail for operators."""
    from xagent.web.models.chat_message import TaskChatMessage

    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Rejected task",
            description="Rejected task",
            status=TaskStatus.RUNNING,
            runner_id=None,
            lease_expires_at=None,
        )
        db.add(task)
        db.commit()
        task_id = task.id

        error_text = "Sandbox capacity limit reached (2 containers, cap 2)"
        _terminal_task_error_payload(task_id, error_text, event_type="task_error")

        db.expire_all()
        messages = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .all()
        )
        assert len(messages) == 1
        assert messages[0].message_type == "task_failure"
        assert messages[0].content == websocket_api.CLIENT_SAFE_TASK_FAILURE
        persisted_task = db.get(Task, task_id)
        assert persisted_task is not None
        assert persisted_task.error_message == error_text
    finally:
        db.close()


def test_terminal_task_error_payload_rejects_replacement_runner_same_run(_test_db):
    """A late legacy error cannot fail a run after another runner took it over.

    Lease recovery deliberately keeps ``run_id`` when a RUNNING task's expired
    lease is claimed by a replacement runner.  The runner predicate is therefore
    independently required; matching only ``run_id`` is not ownership proof.
    """
    db = _direct_db_session()
    try:
        user = User(username="replacement-owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Replacement runner",
            description="Replacement runner",
            status=TaskStatus.RUNNING,
            runner_id="replacement-runner",
            run_id="same-run",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            output="replacement output",
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
    finally:
        db.close()

    payload = _terminal_task_error_payload(
        task_id,
        "late old failure",
        expected_run_id="same-run",
    )

    assert payload is not None
    assert payload["task"]["status"] == TaskStatus.RUNNING.value
    db = _direct_db_session()
    try:
        persisted = db.query(Task).filter(Task.id == task_id).one()
        assert persisted.status == TaskStatus.RUNNING
        assert persisted.runner_id == "replacement-runner"
        assert persisted.run_id == "same-run"
        assert persisted.lease_expires_at is not None
        assert persisted.output == "replacement output"
        assert persisted.error_message is None
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "failure_type"),
    [
        ("chat", ValueError),
        ("chat", RuntimeError),
        ("execute", ValueError),
        ("execute", RuntimeError),
    ],
)
async def test_legacy_handler_does_not_steal_live_foreign_lease(
    _test_db,
    monkeypatch,
    handler_name,
    failure_type,
):
    """Legacy handlers must not mutate a task owned by another runner."""
    db = _direct_db_session()
    try:
        user = User(
            username=f"{handler_name}-{failure_type.__name__}", password_hash="hash"
        )
        db.add(user)
        db.commit()
        task = Task(
            user_id=user.id,
            title="Owned task",
            description="Owned task",
            status=TaskStatus.RUNNING,
            runner_id="replacement-runner",
            run_id="replacement-run",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            output="replacement output",
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        user_id = int(user.id)
    finally:
        db.close()

    class FailingAgentManager:
        async def get_agent_for_task(self, *args, **kwargs):
            raise failure_type("setup failed")

    fake_chat_module = ModuleType("xagent.web.api.chat")
    fake_chat_module.get_agent_manager = lambda: FailingAgentManager()  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "xagent.web.api.chat",
        fake_chat_module,
    )
    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        AsyncMock(),
    )
    broadcasts = AsyncMock()
    monkeypatch.setattr(websocket_api.manager, "broadcast_to_task", broadcasts)

    real_session_factory = websocket_api.get_session_local()
    main_thread_id = threading.get_ident()
    payload_session_threads: list[int] = []

    def tracked_get_session_local():
        payload_session_threads.append(threading.get_ident())
        return real_session_factory

    monkeypatch.setattr(
        websocket_api,
        "get_session_local",
        tracked_get_session_local,
    )

    message_data = {
        "message": "continue",
        "context": {},
        "user": SimpleNamespace(id=user_id, is_admin=False),
    }
    if handler_name == "chat":
        await websocket_api._handle_chat_message_unserialized(
            object(),
            task_id,
            message_data,
        )
    else:
        await websocket_api.handle_execute_task(object(), task_id, message_data)

    assert payload_session_threads
    assert all(thread_id != main_thread_id for thread_id in payload_session_threads)
    payload = broadcasts.await_args_list[-1].args[0]
    if handler_name == "execute":
        # The execute command now enters the shared lease-aware scheduler. A
        # live foreign lease stops it before agent setup, so task_info is the
        # last event rather than a synthetic setup error.
        assert payload["event_type"] == "task_info"
        assert payload["data"]["status"] == TaskStatus.RUNNING.value
    else:
        assert payload["task"]["status"] == TaskStatus.RUNNING.value

    db = _direct_db_session()
    try:
        persisted = db.query(Task).filter(Task.id == task_id).one()
        assert persisted.status == TaskStatus.RUNNING
        assert persisted.runner_id == "replacement-runner"
        assert persisted.run_id == "replacement-run"
        assert persisted.lease_expires_at is not None
        assert persisted.output == "replacement output"
        assert persisted.error_message is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_handle_chat_message_access_denied_does_not_fail_task(
    _test_db, monkeypatch
):
    db = _direct_db_session()
    try:
        owner = User(username="owner", password_hash="hash")
        intruder = User(username="intruder", password_hash="hash")
        db.add_all([owner, intruder])
        db.commit()

        task = Task(
            user_id=owner.id,
            title="Private task",
            description="Private task",
            status=TaskStatus.RUNNING,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = task.id
        intruder_id = intruder.id
    finally:
        db.close()

    sent_messages = []
    broadcasts = []

    async def fake_send_personal_message(message, websocket):
        sent_messages.append(message)

    async def fake_broadcast_to_task(message, broadcast_task_id):
        broadcasts.append((broadcast_task_id, message))

    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        fake_send_personal_message,
    )
    monkeypatch.setattr(
        websocket_api.manager,
        "broadcast_to_task",
        fake_broadcast_to_task,
    )

    await websocket_api.handle_chat_message(
        object(),
        task_id,
        {
            "message": "use this task",
            "user": SimpleNamespace(id=intruder_id, is_admin=False),
        },
    )

    assert broadcasts == []
    assert sent_messages
    assert sent_messages[0]["type"] == "error"
    assert "Access denied" in sent_messages[0]["message"]

    db = _direct_db_session()
    try:
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.RUNNING
        assert persisted_task.runner_id == get_runner_id()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_handle_execute_task_unauthenticated_does_not_fail_task(
    _test_db, monkeypatch
):
    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Private task",
            description="Private task",
            status=TaskStatus.RUNNING,
            runner_id=get_runner_id(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = task.id
    finally:
        db.close()

    sent_messages = []
    broadcasts = []

    async def fake_send_personal_message(message, websocket):
        sent_messages.append(message)

    async def fake_broadcast_to_task(message, broadcast_task_id):
        broadcasts.append((broadcast_task_id, message))

    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        fake_send_personal_message,
    )
    monkeypatch.setattr(
        websocket_api.manager,
        "broadcast_to_task",
        fake_broadcast_to_task,
    )

    await websocket_api.handle_execute_task(object(), task_id, {})

    assert broadcasts == []
    assert sent_messages
    assert sent_messages[0]["type"] == "error"
    assert "authentication required" in sent_messages[0]["message"].lower()

    db = _direct_db_session()
    try:
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.RUNNING
        assert persisted_task.runner_id == get_runner_id()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_execute_task_background_error_marks_task_failed(_test_db, monkeypatch):
    db = _direct_db_session()
    try:
        user = User(username="owner", password_hash="hash")
        db.add(user)
        db.commit()

        task = Task(
            user_id=user.id,
            title="Failing background task",
            description="Failing background task",
            status=TaskStatus.RUNNING,
            runner_id=None,
            lease_expires_at=None,
        )
        db.add(task)
        db.commit()
        task_id = task.id
        user_id = user.id
    finally:
        db.close()

    broadcasts = []

    async def fake_broadcast_to_task(message, broadcast_task_id):
        broadcasts.append((broadcast_task_id, message))

    class FailingAgentManager:
        async def get_agent_for_task(self, *args, **kwargs):
            raise RuntimeError("setup failed")

    monkeypatch.setattr(
        websocket_api.manager,
        "broadcast_to_task",
        fake_broadcast_to_task,
    )

    await websocket_api.execute_task_background(
        task_id=task_id,
        user_message="run",
        context={},
        agent_manager=FailingAgentManager(),
        task_owner_user_id=user_id,
    )

    assert broadcasts
    broadcast_task_id, payload = broadcasts[-1]
    assert broadcast_task_id == task_id
    assert payload["type"] == "task_error"
    assert payload["task"]["status"] == "failed"
    assert payload["error"] == "setup failed"

    db = _direct_db_session()
    try:
        persisted_task = db.query(Task).filter(Task.id == task_id).one()
        assert persisted_task.status == TaskStatus.FAILED
        assert persisted_task.runner_id is None
        assert persisted_task.lease_expires_at is None
        assert persisted_task.error_message == "setup failed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_orchestrated_background_error_defers_database_settlement(
    _test_db, monkeypatch
):
    """The lease owner settles before any terminal client notification.

    ``execute_task_background`` must neither open its own terminal Session nor
    announce FAILED while the durable row still carries the live run lease.
    The orchestrator emits the terminal event only after exact settlement.
    """
    db = _direct_db_session()
    try:
        user = User(username="orchestrated-owner", password_hash="hash")
        db.add(user)
        db.commit()
        task = Task(
            user_id=user.id,
            title="Orchestrated failure",
            description="Orchestrated failure",
            status=TaskStatus.RUNNING,
            runner_id="runner-a",
            run_id="run-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        user_id = int(user.id)
    finally:
        db.close()

    broadcasts: list[dict] = []

    async def fake_broadcast_to_task(message, _task_id):
        broadcasts.append(message)

    class FailingAgentManager:
        async def get_agent_for_task(self, *args, **kwargs):
            raise RuntimeError("setup failed before execution")

    monkeypatch.setattr(
        websocket_api.manager, "broadcast_to_task", fake_broadcast_to_task
    )
    terminal_writer = MagicMock()
    monkeypatch.setattr(websocket_api, "_terminal_task_error_payload", terminal_writer)

    with pytest.raises(RuntimeError, match="setup failed before execution"):
        await websocket_api.execute_task_background(
            task_id=task_id,
            user_message="run",
            context={},
            agent_manager=FailingAgentManager(),
            task_owner_user_id=user_id,
            expected_run_id="run-a",
            task_lease=TaskLease(task_id, "runner-a", "run-a"),
        )

    terminal_writer.assert_not_called()
    assert not any(event.get("type") == "task_error" for event in broadcasts)

    db = _direct_db_session()
    try:
        persisted = db.query(Task).filter(Task.id == task_id).one()
        assert persisted.status == TaskStatus.RUNNING
        assert persisted.runner_id == "runner-a"
        assert persisted.run_id == "run-a"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_orchestrated_background_pool_timeout_is_not_broadcast_as_failed(
    _test_db, monkeypatch
):
    """A fenced run kept for TTL recovery is not a FAILED task."""
    db = _direct_db_session()
    try:
        user = User(username="pool-timeout-owner", password_hash="hash")
        db.add(user)
        db.commit()
        task = Task(
            user_id=user.id,
            title="Pool timeout",
            description="Pool timeout",
            status=TaskStatus.RUNNING,
            runner_id="runner-a",
            run_id="run-a",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        user_id = int(user.id)
    finally:
        db.close()

    broadcasts: list[dict] = []

    async def fake_broadcast_to_task(message, _task_id):
        broadcasts.append(message)

    class PoolExhaustedAgentManager:
        async def get_agent_for_task(self, *args, **kwargs):
            raise SQLAlchemyTimeoutError("setup pool exhausted")

    monkeypatch.setattr(
        websocket_api.manager, "broadcast_to_task", fake_broadcast_to_task
    )
    terminal_writer = MagicMock()
    monkeypatch.setattr(websocket_api, "_terminal_task_error_payload", terminal_writer)

    with pytest.raises(SQLAlchemyTimeoutError, match="setup pool exhausted"):
        await websocket_api.execute_task_background(
            task_id=task_id,
            user_message="run",
            context={},
            agent_manager=PoolExhaustedAgentManager(),
            task_owner_user_id=user_id,
            expected_run_id="run-a",
            task_lease=TaskLease(task_id, "runner-a", "run-a"),
        )

    terminal_writer.assert_not_called()
    assert not any(event.get("type") == "task_error" for event in broadcasts)

    db = _direct_db_session()
    try:
        persisted = db.query(Task).filter(Task.id == task_id).one()
        assert persisted.status == TaskStatus.RUNNING
        assert persisted.runner_id == "runner-a"
        assert persisted.run_id == "run-a"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_background_failure_uses_worker_owned_setup_and_terminal_sessions(
    monkeypatch,
):
    """Setup and terminal persistence both stay off the event-loop thread."""
    main_thread_id = threading.get_ident()
    setup_threads: list[int] = []
    payload_calls: list[tuple[int, bool]] = []

    def load_snapshot(*_args, **_kwargs):
        setup_threads.append(threading.get_ident())
        return SimpleNamespace(
            task=SimpleNamespace(user_id=1, status=TaskStatus.RUNNING),
            runtime_user=SimpleNamespace(id=1, is_admin=False),
        )

    def terminal_payload(
        task_id,
        message,
        *,
        event_type="agent_error",
        expected_run_id=None,
        only_if_running=False,
    ):
        payload_calls.append((threading.get_ident(), only_if_running))
        return {"type": event_type, "message": message}

    class FailingAgentManager:
        async def get_agent_for_task(self, *args, **kwargs):
            raise RuntimeError("setup failed")

    monkeypatch.setattr(
        "xagent.web.services.task_setup_snapshot.load_task_setup_snapshot_sync",
        load_snapshot,
    )
    monkeypatch.setattr(
        websocket_api.background_task_manager,
        "wait_for_previous",
        AsyncMock(),
    )
    monkeypatch.setattr(
        websocket_api,
        "_terminal_task_error_payload",
        terminal_payload,
    )
    monkeypatch.setattr(
        websocket_api.manager,
        "broadcast_to_task",
        AsyncMock(),
    )

    await websocket_api.execute_task_background(
        task_id=42,
        user_message="run",
        context={},
        agent_manager=FailingAgentManager(),
        task_owner_user_id=1,
    )

    assert not hasattr(websocket_api, "_task_is_still_running")
    assert setup_threads and setup_threads[0] != main_thread_id
    assert len(payload_calls) == 1
    assert payload_calls[0][0] != main_thread_id
    assert payload_calls[0][1] is True
