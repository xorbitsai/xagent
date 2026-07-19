"""Owner/actor regression pins for the WebSocket control handlers.

An admin may operate on another user's task (admin bypass), but the agent
runtime must run as the task OWNER, not the admin. A non-admin who is not the
owner must be refused before any runtime is built. These pin the pause / resume
handlers directly (the focused unit tests cover get_agent_for_task in
isolation; here we exercise the handlers end to end).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.websocket import (
    _handle_pause_task_unserialized,
    _handle_resume_task_unserialized,
    background_task_manager,
    execute_resume_background,
    handle_chat_message,
    handle_pause_task,
    handle_resume_task,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_FAILED,
    DELIVERY_PENDING,
)
from xagent.web.services.task_execution_controller import StaleTaskRunError


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'owner_actor.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _user(db, username, *, is_admin=False) -> User:
    u = User(username=username, password_hash="x", is_admin=is_admin)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _task(db, owner_id: int, status: TaskStatus = TaskStatus.RUNNING) -> Task:
    t = Task(
        user_id=owner_id,
        title="t",
        description="d",
        status=status,
        execution_mode="balanced",
        source="sdk",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _register_current_resume(task_id: int) -> None:
    current = asyncio.current_task()
    assert current is not None
    background_task_manager.resume_tasks[task_id] = current


def _patched_manager_and_agent():
    """Return (patches contextmanagers, captured) wiring get_agent_manager +
    the module ``manager`` so the handler can run without real IO."""
    captured: dict = {}
    agent_service = MagicMock()
    agent_service.pause_execution = AsyncMock(return_value={"status": "paused"})
    agent_service.resume_execution = AsyncMock()
    agent_service.supports_live_control = MagicMock(return_value=False)

    async def _get_agent_for_task(task_id, db, *, user=None, task_owner_user_id=None):
        captured["task_owner_user_id"] = task_owner_user_id
        return agent_service

    mgr = MagicMock()
    mgr.get_agent_for_task = AsyncMock(side_effect=_get_agent_for_task)

    ws_manager = MagicMock()
    ws_manager.send_personal_message = AsyncMock()
    ws_manager.broadcast_to_task = AsyncMock()
    return captured, agent_service, mgr, ws_manager


@pytest.mark.asyncio
async def test_chat_admin_append_to_other_users_task_claims_as_owner(
    db_session,
) -> None:
    """The original #587 regression: an admin appending through
    ``handle_chat_message`` to a task owned by another user. The bug was the
    atomic claim using the actor id, so the owner's appendable task failed with
    ``TaskTurnNotFoundError``. Pin that ``begin_turn`` is invoked with
    ``task_owner_user_id == task.user_id`` (the owner), not the admin actor.
    """
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    # COMPLETED -> the WS path treats the follow-up as an APPEND turn.
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)

    ws_manager = MagicMock()
    ws_manager.broadcast_to_task = AsyncMock()
    ws_manager.send_personal_message = AsyncMock()
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "follow-up",
                "client_message_id": "client-turn-1",
                "user": admin,
                "files": [],
            },
        )
        for _ in range(100):
            if begin_turn.await_count:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable admin message was not dispatched in time")

    begin_turn.assert_awaited_once()
    assert begin_turn.await_args.kwargs["task_owner_user_id"] == int(owner.id)
    assert begin_turn.await_args.kwargs["payload"].turn_id == "client-turn-1"
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["client_message_id"] == "client-turn-1"
    assert accepted[0]["turn_id"] == "client-turn-1"


@pytest.mark.asyncio
async def test_chat_without_client_id_uses_durable_command_id_as_turn_id(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {"message": "server generated identity", "user": owner, "files": []},
        )
        for _ in range(100):
            if begin_turn.await_count:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable message was not dispatched in time")

    db_session.expire_all()
    command = (
        db_session.query(TaskExecutionCommand)
        .filter(TaskExecutionCommand.task_id == int(task.id))
        .one()
    )
    assert command.command_id.startswith("message:")
    assert command.payload["client_message_id"] == command.command_id
    assert begin_turn.await_args.kwargs["payload"].turn_id == command.command_id


@pytest.mark.asyncio
async def test_running_chat_message_is_persisted_before_resume(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=True)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    resume_bg = AsyncMock()
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Use the audio tool",
                "client_message_id": "live-turn-1",
                "user": owner,
                "files": [],
            },
        )
        for _ in range(100):
            if bg_mgr.register_reserved_resume.call_count:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable live message was not dispatched in time")

    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.role == "user",
        )
        .one()
    )
    assert stored.content == "Use the audio tool"
    assert stored.turn_id == "live-turn-1"
    assert agent.post_user_message.await_args.kwargs["turn_id"] == "live-turn-1"
    bg_mgr.register_reserved_resume.assert_called_once()
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["client_message_id"] == "live-turn-1"


@pytest.mark.asyncio
async def test_deferred_chat_message_is_acked_after_durable_command_commit(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=False)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    resume_bg = AsyncMock()
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True
    bg_mgr.running_tasks.get.return_value = None
    websocket = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_chat_message(
            websocket,
            int(task.id),
            {
                "message": "Wait for the checkpoint",
                "client_message_id": "deferred-turn-1",
                "user": owner,
                "files": [],
            },
        )
        for _ in range(100):
            db_session.expire_all()
            stored_command = (
                db_session.query(TaskExecutionCommand)
                .filter_by(task_id=int(task.id), command_id="deferred-turn-1")
                .one()
            )
            if stored_command.status == "pending":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("deferred command claim was not released in time")

    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["client_message_id"] == "deferred-turn-1"
    assert stored_command.status == "pending"
    assert not any(
        call.args[0].get("type") == "message_rejected"
        for call in ws_manager.send_personal_message.call_args_list
    )
    kwargs = resume_bg.call_args.kwargs
    assert kwargs["delivery_already_dispatched"] is False
    assert kwargs["delivery_websocket"] is None
    assert kwargs["delivery_client_message_id"] is None


@pytest.mark.asyncio
async def test_resume_registration_failure_preserves_dispatched_delivery(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=True)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True
    bg_mgr.running_tasks.get.return_value = None
    bg_mgr.register_reserved_resume.side_effect = RuntimeError("reservation lost")
    bg_handle = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", MagicMock()),
        patch(
            "xagent.web.api.websocket.asyncio.create_task",
            return_value=bg_handle,
        ),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Apply this safely",
                "client_message_id": "registration-failure-turn",
                "user": owner,
                "files": [],
            },
        )
        for _ in range(100):
            if bg_handle.cancel.called:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("resume registration failure was not handled in time")

    bg_handle.cancel.assert_called_once()
    db_session.expire_all()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "registration-failure-turn")
        .one()
    )
    assert stored.delivery_status == DELIVERY_DISPATCHED
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_retried_durable_message_is_accepted_without_reexecution(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Already delivered",
            message_type="user_message",
            turn_id="stable-turn-1",
        )
    )
    db_session.commit()
    agent_manager = MagicMock()
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Already delivered",
                "client_message_id": "stable-turn-1",
                "user": owner,
                "files": [],
            },
        )

    agent_manager.get_agent_for_task.assert_not_called()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.turn_id == "stable-turn-1",
        )
        .all()
    )
    assert len(stored) == 1
    accepted = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_accepted"
    ]
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_reusing_client_id_with_different_content_is_rejected(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.COMPLETED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Original content",
            message_type="user_message",
            turn_id="stable-turn-1",
        )
    )
    db_session.commit()
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Different content",
                "client_message_id": "stable-turn-1",
                "user": owner,
                "files": [],
            },
        )

    begin_turn.assert_not_awaited()
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["retry_with_new_id"] is True


@pytest.mark.asyncio
async def test_failed_durable_delivery_is_not_silently_accepted(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.FAILED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Retry after checkpoint failure",
            message_type="user_message",
            turn_id="failed-turn-1",
            delivery_status=DELIVERY_FAILED,
        )
    )
    db_session.commit()
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    begin_turn = AsyncMock()

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.services.task_orchestrator.TaskTurnOrchestrator.begin_turn",
            new=begin_turn,
        ),
    ):
        await handle_chat_message(
            MagicMock(),
            int(task.id),
            {
                "message": "Retry after checkpoint failure",
                "client_message_id": "failed-turn-1",
                "user": owner,
                "files": [],
            },
        )

    begin_turn.assert_not_awaited()
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["retry_with_new_id"] is True


@pytest.mark.asyncio
async def test_pause_admin_on_other_users_task_runs_as_owner(db_session) -> None:
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    task = _task(db_session, owner.id)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_pause_task(MagicMock(), int(task.id), {"user": admin})
        for _ in range(100):
            if "task_owner_user_id" in captured:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("durable pause command was not dispatched in time")

    # Built and paused as the OWNER, not the admin actor.
    assert captured["task_owner_user_id"] == int(owner.id)
    agent.pause_execution.assert_awaited_once()


@pytest.mark.asyncio
async def test_durable_pause_propagates_stale_run_error(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id)
    _captured, _agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket.apply_task_control_transition",
            side_effect=StaleTaskRunError("run rotated"),
        ),
        pytest.raises(StaleTaskRunError, match="run rotated"),
    ):
        await _handle_pause_task_unserialized(
            MagicMock(),
            int(task.id),
            {"user": owner, "_durable_ack_sent": True},
        )


@pytest.mark.asyncio
async def test_durable_resume_propagates_stale_run_error(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    _captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control.return_value = True
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.task_execution_controller.transition",
            new=AsyncMock(side_effect=StaleTaskRunError("run rotated")),
        ),
        pytest.raises(StaleTaskRunError, match="run rotated"),
    ):
        await _handle_resume_task_unserialized(
            MagicMock(),
            int(task.id),
            {"user": owner, "_durable_ack_sent": True},
        )
    bg_mgr.release_resume_reservation.assert_called_once_with(int(task.id))


@pytest.mark.asyncio
async def test_pause_non_owner_non_admin_is_refused(db_session) -> None:
    owner = _user(db_session, "owner")
    stranger = _user(db_session, "stranger")  # not admin, not owner
    task = _task(db_session, owner.id)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        # The handler authorizes the task away and handles the denial
        # internally; the point is that no owner runtime is built / paused.
        await handle_pause_task(MagicMock(), int(task.id), {"user": stranger})

    assert "task_owner_user_id" not in captured
    agent.pause_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_admin_on_other_users_task_runs_as_owner(db_session) -> None:
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    task = _task(db_session, owner.id)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": admin})

    assert captured["task_owner_user_id"] == int(owner.id)


@pytest.mark.asyncio
async def test_resume_rejection_embeds_known_control_state(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    task.run_id = "run-current"
    task.state_version = 7
    task.control_state = "running"
    db_session.commit()
    _captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control = MagicMock(return_value=True)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": owner})

        # The rejection is sent from a dispatched command that
        # ``dispatch_task_command_promptly`` may detach after its 50ms
        # deadline, so the "task" payload can land after this call returns.
        payload = None
        for _ in range(100):
            for call in ws_manager.send_personal_message.await_args_list:
                if call.args and "task" in call.args[0]:
                    payload = call.args[0]
                    break
            if payload is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("resume rejection payload did not arrive in time")

    assert payload["task"] == {
        "id": int(task.id),
        "run_id": "run-current",
        "state_version": 7,
        "control_state": "running",
        "status": "running",
    }


@pytest.mark.asyncio
async def test_resume_live_control_admin_runs_background_as_owner(db_session) -> None:
    """Live-control resume schedules ``execute_resume_background``; when an
    admin resumes another user's task it must run with the OWNER's
    UserContext, i.e. ``task_owner_user_id`` is the owner, not the admin."""
    owner = _user(db_session, "owner")
    admin = _user(db_session, "admin", is_admin=True)
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control = MagicMock(return_value=True)

    resume_bg = AsyncMock()
    transition = AsyncMock(
        return_value=SimpleNamespace(
            run_id="run-from-resume-transition",
            status=TaskStatus.PAUSED,
        )
    )
    bg_mgr = MagicMock()
    bg_mgr.running_tasks.get = MagicMock(return_value=None)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", resume_bg),
        patch(
            "xagent.web.api.websocket.task_execution_controller.transition",
            new=transition,
        ),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": admin})

    # Agent built as owner, and the background resume runs as owner.
    assert captured["task_owner_user_id"] == int(owner.id)
    resume_bg.assert_called_once()
    assert resume_bg.call_args.kwargs["task_owner_user_id"] == int(owner.id)
    assert resume_bg.call_args.kwargs["expected_run_id"] == "run-from-resume-transition"
    bg_mgr.reserve_resume.assert_called_once_with(int(task.id))
    bg_mgr.register_reserved_resume.assert_called_once()


@pytest.mark.asyncio
async def test_resume_registration_failure_cancels_coordinator(db_session) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()
    agent.supports_live_control = MagicMock(return_value=True)
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True
    bg_mgr.running_tasks.get.return_value = None
    bg_mgr.register_reserved_resume.side_effect = RuntimeError("reservation lost")
    bg_handle = MagicMock()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", MagicMock()),
        patch(
            "xagent.web.api.websocket.asyncio.create_task",
            return_value=bg_handle,
        ),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": owner})
        for _ in range(100):
            if bg_handle.cancel.called:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("resume command did not finish in time")

    bg_handle.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_execute_resume_background_rejects_owner_mismatch(db_session) -> None:
    """``execute_resume_background`` runs the resume under
    ``UserContext(task_owner_user_id)``. If a caller passes an owner id that
    disagrees with the task row, the symmetric guard (same as
    ``execute_task_background``) must fire before the agent resumes, so the
    runtime never executes as the wrong user."""
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id)

    agent = MagicMock()
    agent.resume_execution_by_id = AsyncMock()
    ws_manager = MagicMock()
    ws_manager.broadcast_to_task = AsyncMock()

    with (
        patch("xagent.web.api.websocket.acquire_task_lease", return_value=object()),
        patch("xagent.web.api.websocket.release_task_lease_with_workforce_sync"),
        patch("xagent.web.api.websocket.stop_task_lease_heartbeat", new=AsyncMock()),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id) + 999,  # != task owner
        )

    # Guard fired before the resume ran -- nothing executed as the wrong user.
    agent.resume_execution_by_id.assert_not_awaited()
    error_types = {
        msg.get("type")
        for (msg, _tid) in (
            call.args for call in ws_manager.broadcast_to_task.call_args_list
        )
        if isinstance(msg, dict)
    }
    assert "task_error" in error_types


@pytest.mark.asyncio
async def test_execute_resume_background_persists_assistant_for_live_turn(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    agent = MagicMock()
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="user",
                metadata={"turn_id": "live-turn-1"},
            )
        ]
    )
    agent.resume_execution_by_id = AsyncMock(
        return_value={
            "status": "completed",
            "success": True,
            "output": "Guidance applied",
            "agent_result": {"context": context},
        }
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
        )

    db_session.expire_all()
    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.role == "assistant",
        )
        .one()
    )
    assert stored.content == "Guidance applied"
    assert stored.turn_id == "live-turn-1"
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.output == "Guidance applied"


@pytest.mark.asyncio
async def test_execute_resume_background_persists_missing_checkpoint_failure(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    agent = MagicMock(
        resume_execution_by_id=AsyncMock(return_value=None),
    )
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-failure-turn",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with patch("xagent.web.api.websocket.manager", ws_manager):
        _register_current_resume(int(task.id))
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            delivery_turn_id="deferred-failure-turn",
        )

    db_session.expire_all()
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert "No resumable execution checkpoint" in str(task.error_message)
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "deferred-failure-turn")
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED
    failures = [
        call.args[0]
        for call in ws_manager.broadcast_to_task.call_args_list
        if call.args[0].get("type") == "task_error"
    ]
    assert len(failures) == 1
    assert failures[0]["task"]["status"] == TaskStatus.FAILED.value


@pytest.mark.asyncio
async def test_deferred_injection_failure_rejects_before_any_acceptance(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.PAUSED)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-injection-failure",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    agent = MagicMock(
        post_user_message=AsyncMock(return_value=False),
        resume_execution_by_id=AsyncMock(),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-injection-failure",
            },
            delivery_turn_id="deferred-injection-failure",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-injection-failure",
        )

    delivery_events = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") in {"message_accepted", "message_rejected"}
    ]
    assert [event["type"] for event in delivery_events] == ["message_rejected"]
    assert delivery_events[0]["retry_with_new_id"] is True
    db_session.expire_all()
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "deferred-injection-failure")
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED


@pytest.mark.asyncio
async def test_deferred_injection_rejects_before_post_when_lease_is_denied(
    db_session,
) -> None:
    owner = _user(db_session, "owner")
    task = _task(db_session, owner.id, status=TaskStatus.RUNNING)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(owner.id),
            role="user",
            content="Deferred guidance",
            message_type="user_message",
            turn_id="deferred-lease-denied",
            delivery_status=DELIVERY_PENDING,
        )
    )
    db_session.commit()
    agent = MagicMock(
        post_user_message=AsyncMock(return_value=True),
        resume_execution_by_id=AsyncMock(),
    )
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.websocket.acquire_task_lease", return_value=None),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
    ):
        await execute_resume_background(
            task_id=int(task.id),
            agent_service=agent,
            task_owner_user_id=int(owner.id),
            pending_user_message={
                "execution_message": "Deferred guidance",
                "display_message": "Deferred guidance",
                "files": [],
                "turn_id": "deferred-lease-denied",
            },
            delivery_turn_id="deferred-lease-denied",
            delivery_websocket=MagicMock(),
            delivery_client_message_id="deferred-lease-denied",
        )

    agent.post_user_message.assert_not_awaited()
    delivery_events = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") in {"message_accepted", "message_rejected"}
    ]
    assert [event["type"] for event in delivery_events] == ["message_rejected"]
    assert delivery_events[0]["retry_with_new_id"] is True
    db_session.expire_all()
    delivery = (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.turn_id == "deferred-lease-denied")
        .one()
    )
    assert delivery.delivery_status == DELIVERY_FAILED


@pytest.mark.asyncio
async def test_resume_non_owner_non_admin_is_refused(db_session) -> None:
    owner = _user(db_session, "owner")
    stranger = _user(db_session, "stranger")
    task = _task(db_session, owner.id)
    captured, agent, mgr, ws_manager = _patched_manager_and_agent()

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
    ):
        await handle_resume_task(MagicMock(), int(task.id), {"user": stranger})

    # Authorized away before any runtime is built; an error is sent back.
    assert "task_owner_user_id" not in captured
    ws_manager.send_personal_message.assert_awaited()
