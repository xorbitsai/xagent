"""Combined-mode consumption of ``execute_task``: the shared START handoff with a
local owner, and what the dispatcher tells the client when it fails."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.user import User
from xagent.web.services import task_command_execution as execution
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_orchestrator
from xagent.web.services import task_start_consumer as consumer
from xagent.web.services.task_command_transport import (
    ClaimedTaskCommand,
    SettledTaskCommand,
    TaskCommandDeferred,
    TaskCommandRejected,
)
from xagent.web.services.task_existing_command import stage_existing_execution
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
)

RUNNER = "local-1"


def _claim(command_db_id: int) -> ClaimedTaskCommand:
    with get_session_local()() as db:
        command = transport.claim_task_command(
            db, runner_id=RUNNER, command_db_id=command_db_id
        )
    assert command is not None
    return command


def _stage_and_claim(task_id: int, owner_id: int) -> ClaimedTaskCommand:
    return _claim(
        stage_existing_execution(
            task_id=task_id,
            task_owner_user_id=owner_id,
            task_description="saved task",
            context={},
            actor_user_id=owner_id,
        )
    )


@pytest.fixture
def combined(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    monkeypatch.setattr(consumer, "get_runner_id", lambda: RUNNER)
    init_db(db_url=f"sqlite:///{tmp_path / 'combined.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="Existing",
            source="internal",
            status=TaskStatus.COMPLETED,
            run_id="previous-run",
            state_version=3,
            control_state="completed",
        )
        db.add(task)
        db.commit()
        ids = (int(task.id), int(user.id))
    yield ids
    Base.metadata.drop_all(bind=get_engine())


def _task_and_command(task_id: int, command_db_id: int):
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        command = db.get(TaskExecutionCommand, command_db_id)
        return (
            (task.status, task.run_id, task.runner_id, task.state_version),
            (command.status, command.result, command.target_run_id),
        )


def test_handoff_opens_the_staged_run_and_completes_in_one_commit(combined):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)

    handoff = consumer._commit_handoff(command, None)

    (status, run_id, runner, _), (cmd_status, result, target) = _task_and_command(
        task_id, command.id
    )
    # The run is the one the START named, so staged facts and the runner agree.
    assert (status, run_id, runner) == (TaskStatus.RUNNING, target, RUNNER)
    assert cmd_status == "completed"
    assert result["run_id"] == target
    assert handoff.claimed.task_lease.run_id == target
    assert handoff.claimed.task_lease.runner_id == RUNNER


def test_redelivered_start_does_not_run_a_second_time(combined):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    consumer._commit_handoff(command, None)
    # The first run finished and the completion write was reclaimed: the row is
    # a processing START again while the task is no longer RUNNING.
    with get_session_local()() as db:
        db.get(TaskExecutionCommand, command.id).status = "processing"
        db.get(TaskExecutionCommand, command.id).claimed_by = RUNNER
        task = db.get(Task, task_id)
        task.status = TaskStatus.COMPLETED
        task.control_state = "completed"
        db.commit()

    settled = consumer._commit_handoff(command, None)

    assert isinstance(settled, SettledTaskCommand)
    assert settled.result == {"rejection_reason": "start_state_changed"}
    (status, run_id, runner, _), _ = _task_and_command(task_id, command.id)
    assert status == TaskStatus.COMPLETED and run_id == command.target_run_id


def test_start_is_rejected_when_the_task_moved_after_staging(combined):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    with get_session_local()() as db:
        db.get(Task, task_id).state_version = 9
        db.commit()

    settled = consumer._commit_handoff(command, None)

    assert settled.result == {"rejection_reason": "start_state_changed"}
    (status, run_id, runner, _), (cmd_status, _, _) = _task_and_command(
        task_id, command.id
    )
    assert (status, runner, cmd_status) == (TaskStatus.COMPLETED, None, "failed")
    assert run_id == "previous-run"


def test_actor_marked_task_is_rejected_with_its_own_reason(combined):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    with get_session_local()() as db:
        db.get(Task, task_id).agent_config = {
            MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True
        }
        db.commit()

    settled = consumer._commit_handoff(command, None)

    assert settled.result == {"rejection_reason": "actor_policy_required"}


def test_start_is_deferred_while_a_local_runner_is_registered(combined):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)

    with pytest.raises(TaskCommandDeferred) as deferred:
        consumer._commit_handoff(command, None, local_run_in_flight=True)

    # Nothing was applied, so the sender may safely retry.
    assert deferred.value.resend_safe is True
    (status, run_id, runner, _), (cmd_status, _, _) = _task_and_command(
        task_id, command.id
    )
    assert (status, run_id, runner) == (TaskStatus.COMPLETED, "previous-run", None)
    assert cmd_status == "processing"


@pytest.mark.asyncio
async def test_start_reads_runner_state_on_the_loop_and_defers(combined):
    from xagent.web.services.task_execution import background_task_manager

    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    running = asyncio.get_running_loop().create_future()
    background_task_manager.running_tasks[task_id] = running
    try:
        with pytest.raises(TaskCommandDeferred):
            await consumer.execute_task_start(command)
    finally:
        background_task_manager.running_tasks.pop(task_id, None)
        running.cancel()

    # Once the runner is gone the very same START goes through.
    settled = await _start_with_scheduling_stubbed(command)
    assert settled.result == {"run_id": command.target_run_id}


def test_non_existing_start_is_not_consumed_without_a_coordinator(combined):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    payload = {**command.payload, "kind": "create", "existing_context": None}
    with get_session_local()() as db:
        db.get(TaskExecutionCommand, command.id).payload = payload
        db.commit()
    command = ClaimedTaskCommand(**{**command.__dict__, "payload": payload})

    settled = consumer._commit_handoff(command, None)

    assert settled.result == {"rejection_reason": "unsupported_start"}


def test_v2_storage_start_and_run_share_one_run_id(combined):
    task_id, owner_id = combined
    with get_session_local()() as db:
        db.get(Task, task_id).conversation_storage_version = 2
        db.commit()
    command = _stage_and_claim(task_id, owner_id)

    consumer._commit_handoff(command, None)

    with get_session_local()() as db:
        fact = db.query(TaskExecutionEvent).filter_by(kind="input_accepted").one()
        assert fact.run_id == db.get(Task, task_id).run_id


@pytest.mark.asyncio
async def test_start_schedules_the_run_under_the_committed_lease(combined, monkeypatch):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    scheduled: dict = {}

    def fake_schedule(**kwargs):
        scheduled.update(kwargs)
        return asyncio.get_running_loop().create_future()

    monkeypatch.setattr(task_orchestrator, "_schedule_bg", fake_schedule)

    settled = await consumer.execute_task_start(command)

    assert settled.result == {"run_id": command.target_run_id}
    assert scheduled["run_id"] == command.target_run_id
    assert scheduled["task_lease"].run_id == command.target_run_id
    assert scheduled["task_lease"].runner_id == RUNNER


@pytest.mark.asyncio
async def test_scheduling_failure_settles_the_committed_lease(combined, monkeypatch):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    monkeypatch.setattr(
        task_orchestrator, "_schedule_bg", Mock(side_effect=RuntimeError("boom"))
    )
    settle = Mock()
    monkeypatch.setattr(task_orchestrator, "settle_task_lease_isolated", settle)
    monkeypatch.setattr(execution, "publish_task_event", AsyncMock())

    with pytest.raises(consumer.PostCommitStartFailure, match="boom"):
        await consumer.execute_task_start(command)

    settle.assert_called_once()
    assert settle.call_args.args[0].run_id == command.target_run_id


async def _start_with_scheduling_stubbed(command):
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        task_orchestrator,
        "_schedule_bg",
        lambda **_kw: asyncio.get_running_loop().create_future(),
    )
    try:
        return await consumer.execute_task_start(command)
    finally:
        patcher.undo()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_spent", [False, True])
async def test_post_commit_scheduling_failure_reaches_the_client_once(
    combined, monkeypatch, budget_spent
):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    if budget_spent:
        command = ClaimedTaskCommand(
            **{**command.__dict__, "failure_count": execution.MAX_COMMAND_FAILURES - 1}
        )
    monkeypatch.setattr(
        task_orchestrator, "_schedule_bg", Mock(side_effect=RuntimeError("boom"))
    )
    monkeypatch.setattr(task_orchestrator, "settle_task_lease_isolated", Mock())
    publish = AsyncMock()
    monkeypatch.setattr(execution, "publish_task_event", publish)

    with pytest.raises(consumer.PostCommitStartFailure) as raised:
        await execution._execute_and_report_local_start(
            command, consumer.execute_task_start
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    publish.assert_awaited_once()
    event = publish.await_args.args[0]
    assert event["type"] == "agent_error"
    assert event["message_code"] == "task_command_failed"


@pytest.mark.asyncio
async def test_post_commit_cancellation_is_not_reported_as_a_failure(
    combined, monkeypatch
):
    task_id, owner_id = combined
    command = _stage_and_claim(task_id, owner_id)
    monkeypatch.setattr(
        task_orchestrator, "_schedule_bg", Mock(side_effect=asyncio.CancelledError())
    )
    monkeypatch.setattr(task_orchestrator, "settle_task_lease_isolated", Mock())
    publish = AsyncMock()
    monkeypatch.setattr(execution, "publish_task_event", publish)

    with pytest.raises(asyncio.CancelledError):
        await consumer.execute_task_start(command)

    publish.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defer_count, reported",
    [(0, False), (transport.max_command_defers() - 1, True)],
)
async def test_local_start_deferral_is_reported_only_when_the_budget_is_spent(
    monkeypatch, defer_count, reported
):
    finished, broadcast = _reporting(monkeypatch)
    command = ClaimedTaskCommand(**{**_command(0).__dict__, "defer_count": defer_count})

    with pytest.raises(TaskCommandDeferred):
        await execution._execute_and_report_local_start(
            command, AsyncMock(side_effect=TaskCommandDeferred("busy"))
        )

    assert broadcast.await_count == int(reported)
    assert finished.call_count == int(reported)


@pytest.mark.asyncio
async def test_settled_refusal_broadcast_carries_the_staged_terminal_draft(
    monkeypatch,
):
    monkeypatch.setattr(execution, "finish_task_command_delivery", Mock())
    publish = AsyncMock()
    monkeypatch.setattr(execution, "publish_task_event", publish)
    settled = SettledTaskCommand({"rejection_reason": "start_state_changed"})

    result = await execution._execute_and_report_local_start(
        _command(0), AsyncMock(return_value=settled)
    )

    assert result is settled
    event = publish.await_args.args[0]
    assert event["message_code"] == "task_command_failed"
    assert event["resend_safe"] is False


def _reporting(monkeypatch):
    finished, broadcast = Mock(), AsyncMock()
    monkeypatch.setattr(execution, "finish_task_command_delivery", finished)
    monkeypatch.setattr(execution, "_broadcast_terminal_command_error", broadcast)
    monkeypatch.setattr(
        execution, "_terminal_command_event_draft", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(execution, "bind_terminal_event_draft", Mock())
    return finished, broadcast


def _command(failure_count: int) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=7,
        actor_user_id=1,
        command_id="turn-1",
        kind=transport.TaskCommandKind.START,
        payload={},
        target_run_id="run-1",
        attempt_count=1,
        failure_count=failure_count,
    )


@pytest.mark.asyncio
async def test_final_generic_failure_is_reported_once(monkeypatch):
    finished, broadcast = _reporting(monkeypatch)
    failing = AsyncMock(side_effect=ConnectionError("pool timeout"))

    with pytest.raises(ConnectionError):
        await execution._execute_and_report_local_start(
            _command(execution.MAX_COMMAND_FAILURES - 1), failing
        )

    finished.assert_called_once()
    broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_retryable_generic_failure_stays_silent(monkeypatch):
    finished, broadcast = _reporting(monkeypatch)

    with pytest.raises(ConnectionError):
        await execution._execute_and_report_local_start(
            _command(0), AsyncMock(side_effect=ConnectionError("pool timeout"))
        )

    finished.assert_not_called()
    broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_broadcast_does_not_replace_the_rejection(monkeypatch):
    _finished, broadcast = _reporting(monkeypatch)
    broadcast.side_effect = RuntimeError("socket gone")
    rejection = TaskCommandRejected("nope", reason="unsupported_start")

    with pytest.raises(TaskCommandRejected) as raised:
        await execution._execute_and_report_local_start(
            _command(0), AsyncMock(side_effect=rejection)
        )

    assert raised.value is rejection


@pytest.mark.asyncio
async def test_lost_claim_is_not_broadcast(monkeypatch):
    finished, broadcast = _reporting(monkeypatch)

    with pytest.raises(TaskCommandRejected):
        await execution._execute_and_report_local_start(
            _command(0),
            AsyncMock(side_effect=TaskCommandRejected("lost", reason="stale_claim")),
        )

    finished.assert_not_called()
    broadcast.assert_not_awaited()


def test_settling_cap_held_commands_fails_a_held_start(combined):
    from xagent.web.services.task_execution_admission import (
        settle_cap_held_commands_no_commit,
    )

    task_id, owner_id = combined
    stage_existing_execution(
        task_id=task_id,
        task_owner_user_id=owner_id,
        task_description="saved task",
        context={},
        actor_user_id=owner_id,
    )
    with get_session_local()() as db:
        control = TaskExecutionCommand(
            task_id=task_id,
            command_id="pause-1",
            kind="pause",
            payload={},
            actor_user_id=owner_id,
            target_state_version=0,
            status="completed",
        )
        db.add(control)
        db.commit()
        control_id = int(control.id)
    with get_session_local()() as db:
        settle_cap_held_commands_no_commit(db, control_id)
        db.commit()

    with get_session_local()() as db:
        start = db.query(TaskExecutionCommand).filter_by(kind="start").one()
        assert start.status == "failed"
        assert start.result == {"rejection_reason": "cancelled_before_admission"}
