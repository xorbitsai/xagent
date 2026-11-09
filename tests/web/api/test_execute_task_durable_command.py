"""``execute_task`` is a durable START command, like ``chat``.

The WebSocket handler only accepts the command (acknowledgement, ``task_info``
and a staged ``START`` of kind ``existing``); the command dispatcher executes
it. Under a closed release hold the command therefore stays pending in the
database for whichever process claims it next.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from xagent import config as xagent_config
from xagent.web.api import websocket as websocket_api
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import (
    task_command_transport,
    task_event_bridge,
    task_execution,
    task_orchestrator,
)
from xagent.web.services.release_hold import ReleaseHold
from xagent.web.services.task_orchestrator import TaskTurnError

from .conftest import _direct_db_session


@pytest.fixture
def hold(monkeypatch: pytest.MonkeyPatch) -> ReleaseHold:
    fresh = ReleaseHold()
    # The dispatcher reads the hold here and, for the cap snapshot, in the
    # execution manager module.
    monkeypatch.setattr(task_command_transport, "release_hold", fresh)
    monkeypatch.setattr(task_execution, "release_hold", fresh)
    monkeypatch.setattr(websocket_api, "release_hold", fresh)
    return fresh


@pytest.fixture
def frames(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    return connection_manager


@pytest.fixture
def schedule(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """The scheduler the consumer hands the committed run to.

    Only the run itself is replaced: staging, the handoff's fence and lease
    acquisition and the command's completion are the real ones.
    """

    async def completed() -> None:
        return None

    mock = Mock(side_effect=lambda **_k: asyncio.create_task(completed()))
    monkeypatch.setattr(task_orchestrator, "_schedule_bg", mock)
    return mock


@pytest.fixture
def task_and_user(_test_db: None) -> tuple[int, Any]:
    db = _direct_db_session()
    try:
        user = User(username="execute-owner", password_hash="hash")
        db.add(user)
        db.commit()
        task = Task(
            user_id=int(user.id),
            title="Execute",
            description="Run the existing task",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        return int(task.id), SimpleNamespace(id=int(user.id), is_admin=False)
    finally:
        db.close()


def _commands(task_id: int) -> list[tuple[str, str, str]]:
    db = _direct_db_session()
    try:
        rows = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .order_by(TaskExecutionCommand.id)
            .all()
        )
        return [(str(r.kind), str(r.status), str(r.payload["kind"])) for r in rows]
    finally:
        db.close()


async def _dispatch() -> bool:
    return await task_command_transport.dispatch_one_task_command(
        command_execution_service.execute_durable_task_command
    )


async def _wait_for_schedule(task_id: int, schedule: Mock) -> None:
    # The START handoff commits the command as completed in a database thread
    # and only then, back on the event loop, hands the run to the scheduler.
    # When the handoff outlasts the handler's inline wait, the dispatch carries
    # on detached and the row can read completed before the scheduler is
    # called, so wait for the hand-off itself.
    for _ in range(100):
        if schedule.called:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(_commands(task_id))


@pytest.mark.asyncio
async def test_combined_execute_task_is_staged_and_run_by_the_dispatcher(
    task_and_user, hold, frames, schedule
) -> None:
    task_id, user = task_and_user
    websocket = MagicMock(name="origin")
    await websocket_api.handle_execute_task(websocket, task_id, {"user": user})
    await _wait_for_schedule(task_id, schedule)

    assert _commands(task_id) == [("start", "completed", "existing")]
    schedule.assert_called_once()
    assert schedule.call_args.kwargs["task_id"] == task_id
    assert schedule.call_args.kwargs["payload"].transcript_message == (
        "Run the existing task"
    )
    # The reply contract is unchanged and goes to the originating socket.
    frames.send_personal_message.assert_awaited_once()
    ack, target = frames.send_personal_message.await_args.args
    assert ack["type"] == "execution_started" and ack["task_id"] == task_id
    assert target is websocket
    first_broadcast = frames.broadcast_to_task.await_args_list[0].args[0]
    assert first_broadcast["event_type"] == "task_info"


@pytest.mark.asyncio
async def test_execute_task_stays_pending_under_a_closed_hold(
    task_and_user, hold, frames, schedule
) -> None:
    task_id, user = task_and_user
    hold.close("release r1")
    websocket = MagicMock(name="origin")
    await websocket_api.handle_execute_task(websocket, task_id, {"user": user})
    assert not await _dispatch()

    # Accepted and acknowledged, but nothing was started in this process.
    assert _commands(task_id) == [("start", "pending", "existing")]
    schedule.assert_not_called()
    assert frames.send_personal_message.await_args.args[0]["type"] == (
        "execution_started"
    )
    assert hold.snapshot()["refused"] == {}

    # The next process (or this one, once the hold lifts) claims and runs it.
    hold.reopen()
    assert await _dispatch()
    assert _commands(task_id) == [("start", "completed", "existing")]
    schedule.assert_called_once()


@pytest.mark.asyncio
async def test_duplicate_execute_task_frame_executes_once(
    task_and_user, hold, frames, schedule
) -> None:
    task_id, user = task_and_user
    hold.close("release r1")
    websocket = MagicMock(name="origin")
    await websocket_api.handle_execute_task(websocket, task_id, {"user": user})
    # The same frame again while the first command is still pending: nothing
    # new is staged and the socket is not torn down by the busy refusal.
    await websocket_api.handle_execute_task(websocket, task_id, {"user": user})
    assert _commands(task_id) == [("start", "pending", "existing")]

    hold.reopen()
    assert await _dispatch()
    assert not await _dispatch()
    schedule.assert_called_once()


@pytest.mark.asyncio
async def test_start_refused_at_consumption_is_reported_to_the_task(
    task_and_user, hold, frames, schedule
) -> None:
    task_id, user = task_and_user
    hold.close("release r1")
    await websocket_api.handle_execute_task(MagicMock(), task_id, {"user": user})
    db = _direct_db_session()
    try:
        db.query(Task).filter(Task.id == task_id).update(
            {Task.status: TaskStatus.RUNNING}
        )
        db.commit()
    finally:
        db.close()

    hold.reopen()
    await _dispatch()

    assert _commands(task_id) == [("start", "failed", "existing")]
    schedule.assert_not_called()
    errors = [
        call.args[0]
        for call in frames.broadcast_to_task.await_args_list
        if call.args[0].get("type") == "agent_error"
    ]
    assert errors and errors[0]["command_kind"] == "start"


@pytest.mark.asyncio
async def test_shared_mode_stages_the_same_start_pending_under_a_hold(
    task_and_user, hold, frames, schedule, monkeypatch
) -> None:
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv(xagent_config.TASK_EXECUTION_ROLE, "web")
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    task_id, user = task_and_user
    hold.close("release r1")
    websocket = MagicMock(name="origin")
    await websocket_api.handle_execute_task(websocket, task_id, {"user": user})

    assert _commands(task_id) == [("start", "pending", "existing")]
    schedule.assert_not_called()
    assert frames.send_personal_message.await_args.args[1] is websocket


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "error_code"),
    [("task_not_found", "task_unavailable"), ("other", "task_execution_failed")],
)
async def test_unstageable_execute_task_answers_the_client(
    task_and_user, hold, frames, monkeypatch, reason, error_code
) -> None:
    task_id, user = task_and_user
    monkeypatch.setattr(
        websocket_api.task_start_service,
        "execute_existing_task",
        AsyncMock(side_effect=TaskTurnError(reason)),
    )
    websocket = MagicMock(name="origin")

    await websocket_api.handle_execute_task(websocket, task_id, {"user": user})

    errors = [
        call.args[0]
        for call in frames.send_personal_message.await_args_list
        if call.args[0].get("type") == "error"
    ]
    assert [e["error_code"] for e in errors] == [error_code]
