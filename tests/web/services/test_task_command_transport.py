from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import inspect as sa_inspect

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    _load_command_message_delivery_status,
    execute_durable_task_command,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import (
    Base,
    get_db,
    get_engine,
    get_session_local,
    init_db,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    MAX_COMMAND_DEFERS,
    MAX_COMMAND_FAILURES,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    _claim_heartbeat,
    claim_task_command,
    defer_task_command,
    dispatch_one_task_command,
    dispatch_task_command_promptly,
    enqueue_task_command,
    fail_task_command,
    finish_task_command,
    load_task_command,
    notify_task_command_dispatcher,
    renew_task_command_claim,
    retry_failed_task_command,
    start_task_command_dispatcher,
    stop_task_command_dispatcher,
    task_has_live_foreign_runner,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'task-commands.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_running_task(db) -> tuple[User, Task]:
    user = User(username="command-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="Durable commands",
        description="Durable commands",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        run_id="run-1",
        runner_id="runner-a",
        lease_expires_at=datetime.utcnow() + timedelta(minutes=1),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return user, task


def test_enqueue_is_committed_and_idempotent(db_session) -> None:
    user, task = _create_running_task(db_session)
    payload = {"type": "pause_task"}

    first = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-1",
        kind=TaskCommandKind.PAUSE,
        payload=payload,
    )
    duplicate = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-1",
        kind=TaskCommandKind.PAUSE,
        payload=payload,
    )
    conflict = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-1",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )

    assert first.created is True
    assert duplicate.command_id == first.command_id
    assert duplicate.created is False
    assert duplicate.payload_matches is True
    assert conflict.payload_matches is False
    assert db_session.query(TaskExecutionCommand).count() == 1


def test_live_run_command_stays_with_owner_until_task_lease_expires(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="message-1",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat", "message": "new guidance"},
    )

    assert claim_task_command(db_session, runner_id="runner-b") is None
    owner_claim = claim_task_command(db_session, runner_id="runner-a")
    assert owner_claim is not None
    assert owner_claim.id == command.command_id

    # Simulate the owner crashing after claim. Once both the command claim and
    # task lease expire, another worker can replay the same durable command.
    row = db_session.query(TaskExecutionCommand).filter_by(id=command.command_id).one()
    row.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    recovered = claim_task_command(db_session, runner_id="runner-b")
    assert recovered is not None
    assert recovered.id == command.command_id
    assert recovered.attempt_count == 2


def test_reassigned_command_routes_only_to_the_current_live_owner(db_session) -> None:
    user, task = _create_running_task(db_session)
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="owner-takeover",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    task.runner_id = "runner-b"
    task.lease_expires_at = datetime.utcnow() + timedelta(minutes=1)
    db_session.commit()

    assert claim_task_command(db_session, runner_id="runner-c") is None
    current_owner_claim = claim_task_command(db_session, runner_id="runner-b")

    assert current_owner_claim is not None
    assert current_owner_claim.id == command.command_id


def test_same_command_row_has_a_single_concurrent_claim_winner(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="concurrent-claim",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    barrier = Barrier(2)

    def claim(runner_id: str) -> int | None:
        SessionLocal = get_session_local()
        with SessionLocal() as db:
            barrier.wait()
            claimed = claim_task_command(
                db,
                runner_id=runner_id,
                command_db_id=command.command_id,
            )
            return claimed.id if claimed is not None else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(claim, "runner-a")
        second = executor.submit(claim, "runner-b")
        winners = [value for value in (first.result(), second.result()) if value]

    assert winners == [command.command_id]


def test_stale_attempt_cannot_mutate_reclaimed_command(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="reclaimed-generation",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    first = claim_task_command(
        db_session,
        runner_id="runner-a",
        command_db_id=enqueued.command_id,
    )
    assert first is not None
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    row.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    second = claim_task_command(
        db_session,
        runner_id="runner-a",
        command_db_id=enqueued.command_id,
    )
    assert second is not None
    assert second.attempt_count == first.attempt_count + 1

    assert not renew_task_command_claim(
        enqueued.command_id,
        "runner-a",
        expected_attempt_count=first.attempt_count,
    )
    assert not fail_task_command(
        enqueued.command_id,
        "runner-a",
        "stale failure",
        expected_attempt_count=first.attempt_count,
    )
    assert not defer_task_command(
        enqueued.command_id,
        "runner-a",
        "stale deferral",
        expected_attempt_count=first.attempt_count,
    )
    assert not finish_task_command(
        enqueued.command_id,
        "runner-a",
        expected_attempt_count=first.attempt_count,
    )

    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    assert row.status == "processing"
    assert row.claimed_by == "runner-a"
    assert row.attempt_count == second.attempt_count


def test_live_foreign_runner_is_rechecked_before_cancel(db_session) -> None:
    _user, task = _create_running_task(db_session)

    assert task_has_live_foreign_runner(int(task.id), runner_id="runner-b") is True
    assert task_has_live_foreign_runner(int(task.id), runner_id="runner-a") is False

    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    assert task_has_live_foreign_runner(int(task.id), runner_id="runner-b") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({}, "Agent ID is missing or null in cancel command payload"),
        ({"agent_id": None}, "Agent ID is missing or null in cancel command payload"),
        ({"agent_id": "invalid"}, "Agent ID 'invalid' is invalid"),
    ],
)
async def test_cancel_command_rejects_invalid_agent_id_payload(
    db_session,
    payload,
    error: str,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="invalid-cancel",
        kind=TaskCommandKind.CANCEL,
        payload=payload,
        target_run_id=None,
        attempt_count=1,
    )

    with pytest.raises(ValueError, match=error):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_cancel_command_defers_on_a_live_foreign_owner(db_session) -> None:
    user, task = _create_running_task(db_session)
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="misrouted-cancel",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
        target_run_id=None,
        attempt_count=1,
    )

    with pytest.raises(TaskCommandDeferred, match="active task lease owner"):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [TaskCommandKind.PAUSE, TaskCommandKind.RESUME])
async def test_control_command_defers_on_a_live_foreign_owner(
    db_session,
    kind: TaskCommandKind,
) -> None:
    user, task = _create_running_task(db_session)
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id=f"misrouted-{kind.value}",
        kind=kind,
        payload={"type": f"{kind.value}_task"},
        target_run_id=None,
        attempt_count=1,
    )

    with pytest.raises(TaskCommandDeferred, match="active task lease owner"):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_cancel_command_does_not_require_persisted_actor(db_session) -> None:
    _user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="cancel-with-deleted-actor",
        kind=TaskCommandKind.CANCEL,
        payload={},
        target_run_id=None,
        attempt_count=1,
    )

    with (
        patch.object(websocket_api, "_load_command_actor") as load_actor,
        pytest.raises(ValueError, match="Agent ID is missing"),
    ):
        await execute_durable_task_command(command)
    load_actor.assert_not_called()


@pytest.mark.asyncio
async def test_only_terminal_command_failure_is_broadcast(db_session) -> None:
    _user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    transient = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="transient-cancel-failure",
        kind=TaskCommandKind.CANCEL,
        payload={},
        target_run_id=None,
        attempt_count=1,
        failure_count=0,
    )
    terminal = ClaimedTaskCommand(
        id=2,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="terminal-cancel-failure",
        kind=TaskCommandKind.CANCEL,
        payload={},
        target_run_id=None,
        attempt_count=1,
        failure_count=MAX_COMMAND_FAILURES - 1,
    )

    with patch.object(
        websocket_api.manager,
        "broadcast_to_task",
        new=AsyncMock(),
    ) as broadcast:
        with pytest.raises(ValueError, match="Agent ID is missing"):
            await execute_durable_task_command(transient)
        broadcast.assert_not_awaited()

        with pytest.raises(ValueError, match="Agent ID is missing"):
            await execute_durable_task_command(terminal)
        broadcast.assert_awaited_once()
        event, event_task_id = broadcast.await_args.args
        assert event_task_id == int(task.id)
        assert event["type"] == "agent_error"
        assert event["command_id"] == "terminal-cancel-failure"


@pytest.mark.asyncio
async def test_final_command_deferral_is_broadcast(db_session) -> None:
    _user, task = _create_running_task(db_session)
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=None,
        command_id="terminal-cancel-defer",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
        target_run_id=None,
        attempt_count=1,
        defer_count=MAX_COMMAND_DEFERS - 1,
    )

    with patch.object(
        websocket_api.manager,
        "broadcast_to_task",
        new=AsyncMock(),
    ) as broadcast:
        with pytest.raises(TaskCommandDeferred, match="active task lease owner"):
            await execute_durable_task_command(command)
        broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_command_is_rejected_after_run_rotation(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.run_id = "run-2"
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="stale-pause",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        target_run_id="run-1",
        attempt_count=1,
    )

    with pytest.raises(TaskCommandRejected, match="Task run changed"):
        await execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_stale_run_rejection_reason_is_persisted(db_session) -> None:
    user, task = _create_running_task(db_session)
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="stale-pause-result",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    task.run_id = "run-2"
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()

    assert await dispatch_one_task_command(
        execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_FAILED
    assert stored.result == {"rejection_reason": "stale_run"}


def test_later_command_cannot_overtake_unfinished_command(db_session) -> None:
    user, task = _create_running_task(db_session)
    first = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="pause-ordered",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    second = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="resume-ordered",
        kind=TaskCommandKind.RESUME,
        payload={"type": "resume_task"},
    )

    claimed_first = claim_task_command(db_session, runner_id="runner-a")
    assert claimed_first is not None
    assert claimed_first.id == first.command_id
    assert (
        claim_task_command(
            db_session,
            runner_id="runner-a",
            command_db_id=second.command_id,
        )
        is None
    )

    assert finish_task_command(first.command_id, "runner-a") is True
    claimed_second = claim_task_command(db_session, runner_id="runner-a")
    assert claimed_second is not None
    assert claimed_second.id == second.command_id


@pytest.mark.asyncio
async def test_dispatch_claims_and_completes_once(db_session) -> None:
    user, task = _create_running_task(db_session)
    # Untargeted commands may be consumed by the current test process.
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="cancel-once",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    seen: list[int] = []

    async def execute(command):
        seen.append(command.id)
        return {"ok": True}

    assert await dispatch_one_task_command(execute, command_db_id=enqueued.command_id)
    assert not await dispatch_one_task_command(
        execute, command_db_id=enqueued.command_id
    )
    db_session.expire_all()
    stored = (
        db_session.query(TaskExecutionCommand).filter_by(id=enqueued.command_id).one()
    )
    assert seen == [enqueued.command_id]
    assert stored.status == COMMAND_COMPLETED
    assert stored.result == {"ok": True}


@pytest.mark.asyncio
async def test_deferred_handoff_retries_without_consuming_failure_budget(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="deferred-message",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat", "message": "wait"},
    )

    async def defer(_command):
        raise TaskCommandDeferred("checkpoint is not ready")

    assert await dispatch_one_task_command(defer, command_db_id=enqueued.command_id)
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == "pending"
    assert stored.attempt_count == 1
    assert stored.failure_count == 0
    assert stored.defer_count == 1
    assert stored.claim_expires_at is not None

    stored.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    async def finish(_command):
        return {"applied": True}

    assert await dispatch_one_task_command(finish, command_db_id=enqueued.command_id)
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_COMPLETED
    assert stored.attempt_count == 2
    assert stored.failure_count == 0
    assert stored.defer_count == 1


@pytest.mark.asyncio
async def test_deferred_message_eventually_fails_and_unblocks_cancel(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    message = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="stuck-message",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat", "message": "wait forever"},
    )
    cancel = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="cancel-after-stuck-message",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    row = db_session.get(TaskExecutionCommand, message.command_id)
    assert row is not None
    row.defer_count = MAX_COMMAND_DEFERS - 1
    db_session.commit()

    async def defer(_command):
        raise TaskCommandDeferred("checkpoint never became ready")

    assert await dispatch_one_task_command(defer, command_db_id=message.command_id)
    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, message.command_id)
    assert row is not None
    assert row.status == COMMAND_FAILED
    assert row.failure_count == 0
    assert row.defer_count == MAX_COMMAND_DEFERS

    cancel_claim = claim_task_command(db_session)
    assert cancel_claim is not None
    assert cancel_claim.id == cancel.command_id


@pytest.mark.asyncio
async def test_real_failures_use_a_separate_bounded_budget(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    command = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="last-failure",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    claimed = claim_task_command(db_session, runner_id="runner-a")
    assert claimed is not None
    row = db_session.get(TaskExecutionCommand, command.command_id)
    assert row is not None
    row.failure_count = MAX_COMMAND_FAILURES - 1
    db_session.commit()

    assert fail_task_command(command.command_id, "runner-a", "still broken") is True
    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, command.command_id)
    assert row is not None
    assert row.status == COMMAND_FAILED
    assert row.failure_count == MAX_COMMAND_FAILURES


def test_failed_command_can_be_reset_for_explicit_retry(db_session) -> None:
    user, task = _create_running_task(db_session)
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="retry-terminal-command",
        kind=TaskCommandKind.CANCEL,
        payload={"agent_id": 1},
    )
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    row.status = COMMAND_FAILED
    row.failure_count = MAX_COMMAND_FAILURES
    row.defer_count = MAX_COMMAND_DEFERS
    row.error = "temporary cancellation failure"
    row.completed_at = datetime.utcnow()
    db_session.commit()

    assert retry_failed_task_command(
        db_session,
        enqueued.command_id,
        target_run_id="run-2",
        target_runner_id="runner-b",
    )
    db_session.expire_all()
    row = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert row is not None
    assert row.status == "pending"
    assert row.failure_count == 0
    assert row.defer_count == 0
    assert row.error is None
    assert row.completed_at is None
    assert row.target_run_id == "run-2"
    assert row.target_runner_id == "runner-b"


@pytest.mark.asyncio
async def test_recovery_dispatches_committed_message_across_run_rotation(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.input = "already committed"
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="committed-turn",
        kind=TaskCommandKind.MESSAGE,
        payload={
            "type": "chat",
            "message": "already committed",
            "client_message_id": "committed-turn",
            "files": [],
        },
    )
    first_claim = claim_task_command(db_session, runner_id="runner-a")
    assert first_claim is not None
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(user.id),
            role="user",
            content="already committed",
            message_type="user_message",
            turn_id="committed-turn",
            delivery_status="pending",
        )
    )
    row = db_session.query(TaskExecutionCommand).filter_by(id=enqueued.command_id).one()
    row.claim_expires_at = datetime.utcnow() - timedelta(seconds=1)
    # MESSAGE commands represent user intent for the task rather than a control
    # mutation on one run, so recovery deliberately applies them after rotation.
    task.run_id = "run-2"
    task.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
    db_session.commit()

    assert await dispatch_one_task_command(
        execute_durable_task_command,
        command_db_id=enqueued.command_id,
    )
    db_session.expire_all()
    messages = (
        db_session.query(TaskChatMessage)
        .filter_by(task_id=int(task.id), turn_id="committed-turn")
        .all()
    )
    assert len(messages) == 1
    assert messages[0].delivery_status == "dispatched"
    command = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert command is not None
    assert command.status == COMMAND_COMPLETED
    assert task.run_id == "run-2"


@pytest.mark.asyncio
async def test_dispatcher_recovers_command_that_predates_worker_start(
    db_session,
) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="startup-recovery",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    applied = asyncio.Event()

    async def execute(command):
        assert command.id == enqueued.command_id
        applied.set()
        return None

    start_task_command_dispatcher(execute)
    try:
        await asyncio.wait_for(applied.wait(), timeout=2)
        for _ in range(100):
            db_session.expire_all()
            stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
            if stored is not None and stored.status == COMMAND_COMPLETED:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("dispatcher did not complete recovered command")
    finally:
        await stop_task_command_dispatcher()
    db_session.expire_all()
    stored = db_session.get(TaskExecutionCommand, enqueued.command_id)
    assert stored is not None
    assert stored.status == COMMAND_COMPLETED


@pytest.mark.asyncio
async def test_dispatcher_recovers_unrelated_tasks_concurrently(db_session) -> None:
    user, first_task = _create_running_task(db_session)
    first_task.runner_id = None
    first_task.lease_expires_at = None
    second_task = Task(
        user_id=user.id,
        title="Second durable command",
        description="Second durable command",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        run_id="run-2",
        runner_id=None,
        lease_expires_at=None,
    )
    db_session.add(second_task)
    db_session.commit()
    first = enqueue_task_command(
        db_session,
        task_id=int(first_task.id),
        actor_user_id=int(user.id),
        command_id="slow-recovery",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    second = enqueue_task_command(
        db_session,
        task_id=int(second_task.id),
        actor_user_id=int(user.id),
        command_id="independent-recovery",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    async def execute(command):
        if command.id == first.command_id:
            await release_first.wait()
        elif command.id == second.command_id:
            second_started.set()
        return None

    start_task_command_dispatcher(execute)
    try:
        await asyncio.wait_for(second_started.wait(), timeout=1)
        release_first.set()
    finally:
        release_first.set()
        await stop_task_command_dispatcher()


def test_load_task_command_returns_an_explicit_detached_snapshot(db_session) -> None:
    user, task = _create_running_task(db_session)
    task.runner_id = None
    task.lease_expires_at = None
    db_session.commit()
    enqueued = enqueue_task_command(
        db_session,
        task_id=int(task.id),
        actor_user_id=int(user.id),
        command_id="detached-status",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
    )

    stored = load_task_command(enqueued.command_id)

    assert stored is not None
    assert sa_inspect(stored).detached
    assert stored.status == "pending"
    assert stored.error is None


def test_legacy_delivery_status_preserves_none(db_session) -> None:
    user, task = _create_running_task(db_session)
    db_session.add(
        TaskChatMessage(
            task_id=int(task.id),
            user_id=int(user.id),
            role="user",
            content="legacy",
            message_type="user_message",
            turn_id="legacy-delivery",
            delivery_status=None,
        )
    )
    db_session.commit()

    assert (
        _load_command_message_delivery_status(int(task.id), "legacy-delivery") is None
    )


@pytest.mark.asyncio
async def test_claim_heartbeat_survives_transient_database_error(
    monkeypatch,
) -> None:
    stop_event = asyncio.Event()
    attempts = 0

    def renew(
        _command_db_id: int,
        _runner_id: str,
        *,
        expected_attempt_count: int | None = None,
    ) -> bool:
        nonlocal attempts
        assert expected_attempt_count == 3
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary database outage")
        stop_event.set()
        return True

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.get_task_lease_heartbeat_seconds",
        lambda: 0.001,
    )
    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.renew_task_command_claim",
        renew,
    )

    await asyncio.wait_for(_claim_heartbeat(7, "runner-a", 3, stop_event), timeout=0.2)

    assert attempts == 2


@pytest.mark.asyncio
async def test_dispatcher_does_not_erase_wakeup_during_empty_claim(monkeypatch) -> None:
    second_claim = asyncio.Event()
    calls = 0

    async def fake_dispatch(_executor, *, command_db_id=None) -> bool:
        nonlocal calls
        del command_db_id
        calls += 1
        if calls == 1:
            notify_task_command_dispatcher()
            return False
        second_claim.set()
        return False

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.dispatch_one_task_command",
        fake_dispatch,
    )

    start_task_command_dispatcher(lambda _command: asyncio.sleep(0))
    try:
        await asyncio.wait_for(second_claim.wait(), timeout=0.25)
    finally:
        await stop_task_command_dispatcher()

    assert calls >= 2


@pytest.mark.asyncio
async def test_prompt_dispatch_observes_late_task_failure(monkeypatch, caplog) -> None:
    async def fail_after_handoff(_executor, *, command_db_id=None) -> bool:
        del command_db_id
        await asyncio.sleep(0.06)
        raise RuntimeError("late dispatch failure")

    async def execute(_command):
        return None

    monkeypatch.setattr(
        "xagent.web.services.task_command_transport.dispatch_one_task_command",
        fail_after_handoff,
    )
    caplog.set_level(logging.ERROR)

    await dispatch_task_command_promptly(execute, command_db_id=1)
    for _ in range(100):
        if "Detached prompt task command dispatch failed" in caplog.text:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("late prompt dispatch failure was not observed")

    assert "late dispatch failure" in caplog.text
