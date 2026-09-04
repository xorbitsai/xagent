from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from anyio import BrokenResourceError, ClosedResourceError

from xagent.web.api import public_chat_access
from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    ConnectionManager,
    _with_current_task_control_state,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User


class _BlockingWebSocket:
    def __init__(self) -> None:
        self.receive_started = asyncio.Event()
        self.accepted = False
        self.closed: tuple[int, str] | None = None
        self.messages: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, *, code: int, reason: str) -> None:
        self.closed = (code, reason)

    async def send_text(self, message: str) -> None:
        self.messages.append(message)

    async def receive_text(self) -> str:
        self.receive_started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


class _ClosedWebSocket:
    def __init__(self, error_type: type[Exception]) -> None:
        self._error_type = error_type

    async def send_text(self, message: str) -> None:
        raise self._error_type


class _RejectIfReceivedWebSocket(_BlockingWebSocket):
    async def receive_text(self) -> str:
        self.receive_started.set()
        raise AssertionError("foreign task socket entered the receive loop")


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)


class _BlockingSendWebSocket(_RecordingWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def send_text(self, message: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        await super().send_text(message)


@pytest.fixture()
def current_task(tmp_path) -> Task:
    init_db(db_url=f"sqlite:///{tmp_path / 'task-state-events.db'}")
    db = next(get_db())
    try:
        user = User(username="event-user", password_hash="hash", is_admin=False)
        db.add(user)
        db.commit()
        task = Task(
            user_id=user.id,
            title="Event state",
            description="Event state",
            status=TaskStatus.RUNNING,
            execution_mode="auto",
            run_id="run-current",
            state_version=7,
            control_state="running",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        db.expunge(task)
        yield task
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.mark.asyncio
async def test_late_state_event_is_rewritten_to_current_snapshot(current_task) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "task_paused",
            "task_id": int(current_task.id),
            "status": "paused",
        }
    )

    assert event["type"] == "task_paused"
    assert event["run_id"] == "run-current"
    assert event["state_version"] == 7
    assert event["control_state"] == "running"
    assert event["status"] == "running"


@pytest.mark.asyncio
async def test_task_info_trace_gets_versioned_state_tuple(current_task) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "trace_event",
            "event_type": "task_info",
            "task_id": int(current_task.id),
            "data": {"id": int(current_task.id), "status": "paused"},
        }
    )

    assert event["state_version"] == 7
    assert event["data"] == {
        "id": int(current_task.id),
        "status": "running",
        "run_id": "run-current",
        "state_version": 7,
        "control_state": "running",
    }


@pytest.mark.asyncio
async def test_producer_snapshot_is_not_relabelled_as_a_newer_run(current_task) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "task_completed",
            "task_id": int(current_task.id),
            "run_id": "run-old",
            "state_version": 5,
            "control_state": "completed",
            "status": "completed",
            "result": "old result",
        }
    )

    assert event["run_id"] == "run-old"
    assert event["state_version"] == 5
    assert event["control_state"] == "completed"
    assert event["status"] == "completed"


@pytest.mark.asyncio
async def test_boolean_state_version_is_replaced_with_current_snapshot(
    current_task,
) -> None:
    event = await _with_current_task_control_state(
        {
            "type": "task_completed",
            "task_id": int(current_task.id),
            "run_id": "run-old",
            "state_version": True,
            "control_state": "completed",
            "status": "completed",
        }
    )

    assert event["run_id"] == "run-current"
    assert event["state_version"] == 7
    assert event["control_state"] == "running"
    assert event["status"] == "running"


@pytest.mark.asyncio
async def test_websocket_endpoint_disconnects_when_cancelled(
    current_task: Task,
    monkeypatch,
) -> None:
    db = next(get_db())
    task = db.query(Task).filter(Task.id == int(current_task.id)).one()
    task_id = int(current_task.id)
    owner_id = int(task.user_id)
    db.close()
    websocket = _BlockingWebSocket()
    connection_manager = ConnectionManager()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=owner_id, is_admin=False)),
    )
    monkeypatch.setattr(websocket_api, "handle_status_request", AsyncMock())

    endpoint = asyncio.create_task(
        websocket_api.websocket_chat_endpoint(websocket, task_id, "token")
    )
    await websocket.receive_started.wait()

    assert connection_manager.active_connections[task_id] == [websocket]

    endpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await endpoint

    assert task_id not in connection_manager.active_connections


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_kind", ["owner", "admin"])
async def test_private_websocket_registers_only_authorized_task_audiences(
    current_task: Task,
    monkeypatch,
    actor_kind: str,
) -> None:
    db = next(get_db())
    task = db.query(Task).filter(Task.id == int(current_task.id)).one()
    if actor_kind == "admin":
        actor = User(username="endpoint-admin", password_hash="hash", is_admin=True)
        db.add(actor)
        db.commit()
        actor_id = int(actor.id)
    else:
        actor_id = int(task.user_id)
    db.close()

    websocket = _BlockingWebSocket()
    connection_manager = ConnectionManager()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(
            return_value=SimpleNamespace(
                id=actor_id,
                is_admin=actor_kind == "admin",
            )
        ),
    )
    monkeypatch.setattr(websocket_api, "handle_status_request", AsyncMock())

    endpoint = asyncio.create_task(
        websocket_api.websocket_chat_endpoint(websocket, int(current_task.id), "token")
    )
    await websocket.receive_started.wait()

    assert websocket.accepted is True
    assert websocket.closed is None
    assert connection_manager.connections_for_task(int(current_task.id)) == [websocket]
    await connection_manager.broadcast_to_task(
        {"type": "authorized-private-event"},
        int(current_task.id),
    )
    assert json.loads(websocket.messages[-1])["type"] == "authorized-private-event"

    endpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await endpoint


@pytest.mark.asyncio
async def test_private_websocket_rejects_foreign_task_before_registration_or_receive(
    current_task: Task,
    monkeypatch,
) -> None:
    db = next(get_db())
    intruder = User(username="endpoint-intruder", password_hash="hash", is_admin=False)
    db.add(intruder)
    db.commit()
    intruder_id = int(intruder.id)
    db.close()

    websocket = _RejectIfReceivedWebSocket()
    connection_manager = ConnectionManager()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=intruder_id, is_admin=False)),
    )
    status = AsyncMock()
    monkeypatch.setattr(websocket_api, "handle_status_request", status)

    await websocket_api.websocket_chat_endpoint(
        websocket,
        int(current_task.id),
        "token",
    )

    assert websocket.accepted is True
    assert websocket.closed == (4003, "Task is no longer available.")
    assert websocket.receive_started.is_set() is False
    assert connection_manager.active_connections == {}
    status.assert_not_awaited()
    await connection_manager.broadcast_to_task(
        {"type": "must-not-reach-foreign"},
        int(current_task.id),
    )
    assert websocket.messages == []


@pytest.mark.asyncio
async def test_missing_private_task_stays_unregistered_until_recovery_moves_connection(
    current_task: Task,
    monkeypatch,
) -> None:
    missing_task_id = int(current_task.id) + 424242
    replacement_task_id = int(current_task.id)
    websocket = _BlockingWebSocket()
    connection_manager = ConnectionManager()

    async def recover_missing_task(_websocket, task_id, _message_data) -> None:
        assert task_id == missing_task_id
        assert connection_manager.active_connections == {}
        connection_manager.move_connection(websocket, replacement_task_id)
        await connection_manager.broadcast_to_task(
            {"type": "replacement-task-event"},
            replacement_task_id,
        )
        assert json.loads(websocket.messages[-1])["type"] == "replacement-task-event"

    websocket.receive_text = AsyncMock(
        side_effect=[json.dumps({"type": "chat"}), asyncio.CancelledError()]
    )
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(
            return_value=SimpleNamespace(id=int(current_task.user_id), is_admin=False)
        ),
    )
    monkeypatch.setattr(websocket_api, "handle_status_request", AsyncMock())
    monkeypatch.setattr(websocket_api, "handle_chat_message", recover_missing_task)

    with pytest.raises(asyncio.CancelledError):
        await websocket_api.websocket_chat_endpoint(websocket, missing_task_id, "token")

    assert websocket.accepted is True
    assert connection_manager.active_connections == {}


@pytest.mark.asyncio
async def test_websocket_endpoint_disconnects_moved_connection_when_cancelled(
    current_task: Task,
    monkeypatch,
) -> None:
    initial_task_id = int(current_task.id) + 424242
    moved_task_id = int(current_task.id)
    websocket = _BlockingWebSocket()
    connection_manager = ConnectionManager()

    async def move_connection_during_initial_status(*args) -> None:
        connection_manager.move_connection(websocket, moved_task_id)

    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(
            return_value=SimpleNamespace(
                id=int(current_task.user_id),
                is_admin=False,
            )
        ),
    )
    monkeypatch.setattr(
        websocket_api,
        "handle_status_request",
        AsyncMock(side_effect=move_connection_during_initial_status),
    )

    endpoint = asyncio.create_task(
        websocket_api.websocket_chat_endpoint(websocket, initial_task_id, "token")
    )
    await websocket.receive_started.wait()

    assert connection_manager.active_connections == {moved_task_id: [websocket]}

    endpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await endpoint

    assert connection_manager.active_connections == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_kind", ["public", "share"])
async def test_public_websocket_endpoint_disconnects_reassigned_connection_when_cancelled(
    monkeypatch,
    endpoint_kind: str,
) -> None:
    initial_task_id = 42
    moved_task_id = 99
    websocket = _BlockingWebSocket()
    connection_manager = ConnectionManager()
    principal = websocket_api.WebSocketPrincipal(id=7, is_admin=False)

    async def reassign_during_initial_status(*args) -> None:
        connection_manager.register_connection(websocket, moved_task_id)

    monkeypatch.setattr(public_chat_access, "manager", connection_manager)
    monkeypatch.setattr(
        public_chat_access,
        "_authorize_public_chat_websocket",
        AsyncMock(return_value=principal),
    )
    monkeypatch.setattr(
        public_chat_access,
        "_authorize_share_chat_websocket",
        AsyncMock(return_value=principal),
    )
    monkeypatch.setattr(
        public_chat_access,
        "handle_status_request",
        AsyncMock(side_effect=reassign_during_initial_status),
    )

    if endpoint_kind == "public":
        endpoint = asyncio.create_task(
            public_chat_access.public_chat_websocket_endpoint(
                websocket=websocket,
                task_id=initial_task_id,
                token="token",
                expected_auth_mode="widget",
            )
        )
    else:
        endpoint = asyncio.create_task(
            public_chat_access.share_chat_websocket_endpoint(
                websocket=websocket,
                task_id=initial_task_id,
                token="token",
            )
        )

    await websocket.receive_started.wait()

    assert connection_manager.active_connections == {moved_task_id: [websocket]}

    endpoint.cancel()
    with pytest.raises(asyncio.CancelledError):
        await endpoint

    assert connection_manager.active_connections == {}
    assert connection_manager._connection_task_ids == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [ClosedResourceError, BrokenResourceError],
)
async def test_broadcast_skips_closed_connection_and_reaches_live_connection(
    error_type: type[Exception],
) -> None:
    task_id = 42
    closed_websocket = _ClosedWebSocket(error_type)
    live_websocket = _RecordingWebSocket()
    connection_manager = ConnectionManager()
    connection_manager.register_connection(closed_websocket, task_id)
    connection_manager.register_connection(live_websocket, task_id)

    message = {"type": "diagnostic", "message": "still live"}
    await connection_manager.broadcast_to_task(message, task_id)

    assert live_websocket.messages == [json.dumps(message)]
    assert connection_manager.active_connections[task_id] == [live_websocket]


@pytest.mark.asyncio
async def test_broadcast_reraises_unexpected_connection_error() -> None:
    task_id = 42
    failed_websocket = _ClosedWebSocket(ValueError)
    connection_manager = ConnectionManager()
    connection_manager.register_connection(failed_websocket, task_id)

    with pytest.raises(ValueError):
        await connection_manager.broadcast_to_task({"type": "diagnostic"}, task_id)

    assert task_id not in connection_manager.active_connections


@pytest.mark.asyncio
async def test_broadcast_rechecks_membership_after_message_enrichment(
    monkeypatch,
) -> None:
    task_id = 42
    websocket = _RecordingWebSocket()
    connection_manager = ConnectionManager()
    connection_manager.register_connection(websocket, task_id)

    async def detach_during_enrichment(message, **kwargs):
        connection_manager.detach_task_connections(task_id)
        return message

    monkeypatch.setattr(
        websocket_api,
        "_with_current_task_control_state",
        detach_during_enrichment,
    )

    await connection_manager.broadcast_to_task({"type": "diagnostic"}, task_id)

    assert websocket.messages == []
    assert connection_manager.active_connections == {}


@pytest.mark.asyncio
async def test_broadcast_uses_membership_snapshot_without_cross_socket_blocking() -> (
    None
):
    task_id = 42
    moved_task_id = 99
    blocking_websocket = _BlockingSendWebSocket()
    moved_websocket = _RecordingWebSocket()
    connection_manager = ConnectionManager()
    connection_manager.register_connection(blocking_websocket, task_id)
    connection_manager.register_connection(moved_websocket, task_id)

    broadcast = asyncio.create_task(
        connection_manager.broadcast_to_task({"type": "diagnostic"}, task_id)
    )
    await blocking_websocket.send_started.wait()
    connection_manager.move_connection(moved_websocket, moved_task_id)
    blocking_websocket.release_send.set()
    await broadcast

    # The socket was authorized when fan-out began. Per-connection sends now
    # progress independently, so moving another socket cannot retroactively
    # cancel a frame that was already scheduled for this connection.
    assert moved_websocket.messages == [json.dumps({"type": "diagnostic"})]
    assert connection_manager.active_connections == {
        task_id: [blocking_websocket],
        moved_task_id: [moved_websocket],
    }


def test_detach_task_connections_removes_forward_and_reverse_membership() -> None:
    task_id = 42
    other_task_id = 99
    first_websocket = _RecordingWebSocket()
    second_websocket = _RecordingWebSocket()
    other_websocket = _RecordingWebSocket()
    connection_manager = ConnectionManager()
    connection_manager.register_connection(first_websocket, task_id)
    connection_manager.register_connection(second_websocket, task_id)
    connection_manager.register_connection(other_websocket, other_task_id)

    detached = connection_manager.detach_task_connections(task_id)

    assert detached == [first_websocket, second_websocket]
    assert connection_manager.active_connections == {other_task_id: [other_websocket]}
    assert connection_manager._connection_task_ids == {other_websocket: other_task_id}
