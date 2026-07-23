from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
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

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        self.receive_started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")


class _ClosedWebSocket:
    def __init__(self, error_type: type[Exception]) -> None:
        self._error_type = error_type

    async def send_text(self, message: str) -> None:
        raise self._error_type


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
async def test_websocket_endpoint_disconnects_when_cancelled(monkeypatch) -> None:
    task_id = 42
    websocket = _BlockingWebSocket()
    connection_manager = ConnectionManager()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=7)),
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
async def test_websocket_endpoint_disconnects_moved_connection_when_cancelled(
    monkeypatch,
) -> None:
    initial_task_id = 42
    moved_task_id = 99
    websocket = _BlockingWebSocket()
    connection_manager = ConnectionManager()

    async def move_connection_during_initial_status(*args) -> None:
        connection_manager.move_connection(websocket, moved_task_id)

    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(return_value=SimpleNamespace(id=7)),
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
    access_context = SimpleNamespace(user=SimpleNamespace(id=7))

    async def reassign_during_initial_status(*args) -> None:
        connection_manager.register_connection(websocket, moved_task_id)

    monkeypatch.setattr(public_chat_access, "manager", connection_manager)
    monkeypatch.setattr(
        public_chat_access,
        "db_session_context",
        lambda: nullcontext(object()),
    )
    monkeypatch.setattr(
        public_chat_access,
        "get_public_chat_user",
        lambda *args, **kwargs: access_context,
    )
    monkeypatch.setattr(
        public_chat_access,
        "get_share_chat_user",
        lambda *args, **kwargs: access_context,
    )
    monkeypatch.setattr(
        public_chat_access,
        "get_task_for_public_context",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        public_chat_access,
        "get_task_for_share_context",
        lambda *args, **kwargs: None,
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
async def test_broadcast_skips_connection_moved_during_fanout() -> None:
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

    assert moved_websocket.messages == []
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
