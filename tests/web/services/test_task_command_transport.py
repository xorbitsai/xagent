from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy import inspect as sa_inspect

from xagent.web.api.websocket import (
    _load_command_message_delivery_status,
    execute_durable_task_command,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    TaskCommandDeferred,
    TaskCommandKind,
    _claim_heartbeat,
    claim_task_command,
    dispatch_one_task_command,
    enqueue_task_command,
    finish_task_command,
    load_task_command,
    notify_task_command_dispatcher,
    start_task_command_dispatcher,
    stop_task_command_dispatcher,
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


@pytest.mark.asyncio
async def test_recovery_does_not_restart_a_committed_new_turn(db_session) -> None:
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

    def renew(_command_db_id: int, _runner_id: str) -> bool:
        nonlocal attempts
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

    await asyncio.wait_for(_claim_heartbeat(7, "runner-a", stop_event), timeout=0.2)

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
