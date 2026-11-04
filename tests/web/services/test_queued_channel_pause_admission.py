"""Queued channel START controls across the durable admission boundary."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F401, F811

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services import task_command_execution
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_orchestrator, task_start_consumer
from xagent.web.services.channel_runtime import SelectedChannelTask
from xagent.web.services.task_orchestrator import TaskTurnPayload

from .coordinator_command_shared import claim_task_command
from .test_task_execution_admission import (
    Execution,
    engine,
    enqueue,
    eventually,
    host,
)


@pytest.fixture(params=[None, "previous-run"])
async def queued_channel_start(host, monkeypatch, request):
    previous_run = request.param
    monkeypatch.setattr(shared, "get_session_local", lambda: host.sessions)
    monkeypatch.setattr(
        task_command_execution, "get_session_local", lambda: host.sessions
    )
    monkeypatch.setattr(task_start_consumer, "get_session_local", lambda: host.sessions)

    blocker = enqueue(host)
    active = Execution(host)
    assert await transport.dispatch_one_task_command(active)
    with host.sessions() as db:
        channel = UserChannel(
            user_id=host.user,
            channel_type="slack",
            channel_name="test",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.flush()
        task = Task(
            user_id=host.user,
            channel_id=channel.id,
            source="external",
            title="Queued channel turn",
            status=TaskStatus.COMPLETED if previous_run else TaskStatus.PENDING,
            control_state="completed" if previous_run else "pending",
            run_id=previous_run,
            output="Previous answer" if previous_run else None,
            state_version=0,
        )
        db.add(task)
        db.commit()
        selection = SelectedChannelTask(
            host.user,
            int(task.id),
            not previous_run,
            int(channel.id),
            "sender",
            previous_run,
            0,
        )
    turn = shared.SharedChannelTurn(selection, workspace=SimpleNamespace())
    start_id = shared._accept_channel_turn(turn, TaskTurnPayload("hello"), "ingress")
    turn.command_db_id = start_id
    turn.accepted = True
    turn.request_stop()
    assert turn.stop_task is not None
    await turn.stop_task
    with host.sessions() as db:
        pause = (
            db.query(TaskExecutionCommand)
            .filter_by(task_id=selection.task_id, kind="pause")
            .one()
        )
        pause_id = int(pause.id)
    try:
        yield SimpleNamespace(
            active=active,
            blocker=blocker,
            pause_id=pause_id,
            start_id=start_id,
            task_id=selection.task_id,
            previous_run=previous_run,
        )
    finally:
        active.finish.set()
        active.cleanup.set()


async def test_pause_settles_channel_start_stopped_before_admission(
    host, queued_channel_start, monkeypatch
):
    queued = queued_channel_start
    assert await transport.dispatch_one_task_command(
        task_command_execution.execute_durable_task_command,
        command_db_id=queued.pause_id,
    )
    with host.sessions() as db:
        start = db.get(TaskExecutionCommand, queued.start_id)
        pause = db.get(TaskExecutionCommand, queued.pause_id)
        events = {
            event.task_command_id: event.outcome
            for event in db.query(TaskCommandTerminalEvent)
            .filter(
                TaskCommandTerminalEvent.task_command_id.in_(
                    (queued.start_id, queued.pause_id)
                )
            )
            .all()
        }
        assert start.status == "failed"
        assert start.error == "Task stopped before execution started."
        assert start.result == {"rejection_reason": "cancelled_before_admission"}
        assert pause.status == "completed"
        if queued.previous_run:
            task = db.get(Task, queued.task_id)
            assert (task.run_id, task.output, task.status) == (
                queued.previous_run,
                "Previous answer",
                TaskStatus.COMPLETED,
            )
        else:
            task = db.get(Task, queued.task_id)
            assert (task.status, task.error_message) == (
                TaskStatus.FAILED,
                "Task stopped before execution started.",
            )
        assert events == {queued.start_id: "failed", queued.pause_id: "completed"}

    queued.active.finish.set()
    queued.active.cleanup.set()

    def blocker_released() -> bool:
        with host.sessions() as db:
            return db.get(TaskAdmissionTicket, queued.blocker.command_id) is None

    await eventually(blocker_released)
    scheduled = Mock()
    monkeypatch.setattr(task_orchestrator, "_schedule_bg", scheduled)
    assert not await transport.dispatch_one_task_command(
        task_command_execution.execute_durable_task_command,
        command_db_id=queued.start_id,
    )
    scheduled.assert_not_called()
    with host.sessions() as db:
        assert transport.retry_failed_task_command(db, queued.start_id)
        ticket = db.get(TaskAdmissionTicket, queued.start_id)
        assert (ticket.runner_id, ticket.owner_attempt_id) == (None, None)


@pytest.mark.parametrize("mutation", ["actor", "replacement", "stale_claim"])
async def test_pause_leaves_start_when_exact_fence_is_lost(
    host, queued_channel_start, mutation
):
    queued = queued_channel_start
    if mutation in {"actor", "replacement"}:
        with host.sessions() as db:
            if mutation == "actor":
                db.get(TaskExecutionCommand, queued.pause_id).actor_subject = "replaced"
            else:
                db.get(Task, queued.task_id).run_id = "replacement-run"
            db.commit()

    async def execute(command):
        if mutation == "stale_claim":
            with host.sessions() as db:
                db.get(TaskExecutionCommand, queued.pause_id).attempt_count += 1
                db.commit()
        return await task_command_execution.execute_durable_task_command(command)

    assert await transport.dispatch_one_task_command(
        execute, command_db_id=queued.pause_id
    )
    with host.sessions() as db:
        assert db.get(TaskExecutionCommand, queued.start_id).status == "pending"


async def test_pause_cannot_overtake_start_after_real_reservation(
    host, queued_channel_start
):
    queued = queued_channel_start
    queued.active.finish.set()
    queued.active.cleanup.set()

    def blocker_released() -> bool:
        with host.sessions() as db:
            return db.get(TaskAdmissionTicket, queued.blocker.command_id) is None

    await eventually(blocker_released)
    with host.sessions() as db:
        claimed = await claim_task_command(
            db, runner_id="worker-1", command_db_id=queued.start_id
        )
        assert claimed is not None
        task = db.get(Task, queued.task_id)
        ticket = db.get(TaskAdmissionTicket, queued.start_id)
        assert (ticket.runner_id, ticket.owner_attempt_id) == (
            task.runner_id,
            task.lease_attempt_id,
        )
    assert not await transport.dispatch_one_task_command(
        task_command_execution.execute_durable_task_command,
        command_db_id=queued.pause_id,
    )
    with host.sessions() as db:
        assert db.get(TaskExecutionCommand, queued.start_id).status == "processing"
        assert db.get(TaskExecutionCommand, queued.pause_id).status == "pending"
