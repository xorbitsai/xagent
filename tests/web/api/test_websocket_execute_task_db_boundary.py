"""Issue #935 regression tests for legacy WebSocket task execution."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import event

from xagent.web.api import chat as chat_api
from xagent.web.api import websocket as websocket_api
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_orchestrator as orchestrator_module
from xagent.web.services.task_orchestrator import (
    TaskTurnOrchestrator,
    TaskTurnPayload,
)

from .conftest import _direct_db_session


@pytest.mark.asyncio
async def test_legacy_execute_uses_worker_snapshot_and_primitive_scheduler_boundary(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No request Session or ORM row may survive into async execution."""
    db = _direct_db_session()
    try:
        user = User(username="legacy-execute-owner", password_hash="hash")
        db.add(user)
        db.commit()
        task = Task(
            user_id=int(user.id),
            title="Legacy execution",
            description="Run the existing task",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            process_description="Follow the saved process",
            examples=[{"input": "a", "output": "b"}],
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        user_id = int(user.id)
    finally:
        db.close()

    # The pre-fix handler directly calls these collaborators while retaining
    # its request Session. Keeping them harmless lets the assertion below
    # fail specifically because the primitive scheduler boundary is absent.
    agent_service = MagicMock()
    agent_service.set_outbound_message_handler = MagicMock()
    agent_service.set_execution_context_messages = MagicMock()
    agent_service.set_recovered_skill_context = MagicMock()
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    agent_manager.execute_task = AsyncMock(
        return_value={"success": True, "output": "done", "file_outputs": []}
    )
    monkeypatch.setattr(chat_api, "get_agent_manager", lambda: agent_manager)

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    async def completed_execution() -> None:
        return None

    main_thread_id = threading.get_ident()
    database_thread_ids: list[int] = []
    checked_out_connections = 0

    def record_database_thread(*_args: object) -> None:
        database_thread_ids.append(threading.get_ident())

    def record_checkout(*_args: object) -> None:
        nonlocal checked_out_connections
        checked_out_connections += 1

    def record_checkin(*_args: object) -> None:
        nonlocal checked_out_connections
        checked_out_connections -= 1

    scheduled_background = asyncio.create_task(completed_execution())

    async def schedule_at_closed_session_boundary(**_kwargs: object) -> asyncio.Task:
        assert checked_out_connections == 0
        return scheduled_background

    schedule = AsyncMock(side_effect=schedule_at_closed_session_boundary)
    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        schedule,
        raising=False,
    )

    engine = get_engine()
    event.listen(engine, "before_cursor_execute", record_database_thread)
    event.listen(engine, "checkout", record_checkout)
    event.listen(engine, "checkin", record_checkin)  # codespell:ignore checkin
    try:
        await websocket_api.handle_execute_task(
            MagicMock(),
            task_id,
            {"user": SimpleNamespace(id=user_id, is_admin=False)},
        )
    finally:
        event.remove(engine, "before_cursor_execute", record_database_thread)
        event.remove(engine, "checkout", record_checkout)
        event.remove(engine, "checkin", record_checkin)  # codespell:ignore checkin

    assert database_thread_ids
    assert all(thread_id != main_thread_id for thread_id in database_thread_ids)
    schedule.assert_awaited_once()
    schedule_call = schedule.await_args
    assert schedule_call.kwargs["task_id"] == task_id
    assert schedule_call.kwargs["task_owner_user_id"] == user_id
    assert schedule_call.kwargs["task_source"] == "internal"
    assert schedule_call.kwargs["actor_user_id"] == user_id
    assert schedule_call.kwargs["context"] == {
        "execution_mode": "balanced",
        "process_description": "Follow the saved process",
        "examples": [{"input": "a", "output": "b"}],
    }
    payload = schedule_call.kwargs["payload"]
    assert isinstance(payload, TaskTurnPayload)
    assert payload.transcript_message == "Run the existing task"
    assert payload.execution_message == "Run the existing task"

    # The legacy protocol still emits the immediate acknowledgement and
    # task-info event before waiting for the scheduled execution to finish.
    connection_manager.send_personal_message.assert_awaited_once()
    assert (
        connection_manager.send_personal_message.await_args.args[0]["type"]
        == "execution_started"
    )
    task_info = connection_manager.broadcast_to_task.await_args_list[0].args[0]
    assert task_info["event_type"] == "task_info"
    assert task_info["data"]["id"] == task_id
    assert task_info["data"]["status"] == TaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_existing_task_scheduler_reuses_shared_runtime_without_turn_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility entry schedules runtime work without a new message."""

    async def completed_execution() -> None:
        return None

    background_task = asyncio.create_task(completed_execution())
    schedule = MagicMock(return_value=background_task)
    monkeypatch.setattr(orchestrator_module, "_schedule_bg", schedule)

    payload = TaskTurnPayload(
        transcript_message="already stored description",
        execution_message="already stored description",
    )
    returned = await TaskTurnOrchestrator.schedule_existing_task_execution(
        task_id=23,
        task_owner_user_id=9,
        task_source="internal",
        payload=payload,
        context={"execution_mode": "balanced"},
        actor_user_id=11,
    )

    assert returned is background_task
    schedule.assert_called_once_with(
        task_id=23,
        task_owner_user_id=9,
        task_source="internal",
        payload=payload,
        force_fresh=False,
        context={"execution_mode": "balanced"},
    )
    await background_task
