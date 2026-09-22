"""Public dispatch keeps command admission and settlement under one task owner."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models import database
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as owners

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
async def host(engine, task_id, monkeypatch):
    factory = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database, "get_session_local", lambda: factory)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "worker")
    monkeypatch.setattr(runtime, "get_runner_id", lambda: "worker")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.02)
    registry = runtime.TaskCoordinatorRegistry(factory)
    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)
    monkeypatch.setattr(
        transport,
        "_claim_heartbeat",
        Mock(side_effect=AssertionError("independent heartbeat")),
    )
    try:
        yield factory, task_id, registry
    finally:
        await registry.close()


def enqueue(host, identity="message", **fields):
    factory, tid, _ = host
    with factory() as db:
        row = TaskExecutionCommand(
            task_id=tid, command_id=identity, kind="message", payload={}, **fields
        )
        db.add(row)
        db.commit()
        return row.id


async def test_duplicate_wakes_apply_once_and_hold_owner_through_receipt(
    host, monkeypatch
):
    factory, tid, registry = host
    cid = enqueue(host)
    applied = []
    entered, finish = asyncio.Event(), asyncio.Event()

    async def execute(command):
        coordinator = runtime.current_task_coordinator(tid)
        assert coordinator is not None
        with factory() as db:
            row = db.get(TaskExecutionCommand, cid)
            assert row.claimed_by is None and row.claim_expires_at is None
        applied.append(command.attempt_count)
        entered.set()
        await finish.wait()
        return {"applied": True}

    dispatches = [
        asyncio.create_task(transport.dispatch_one_task_command(execute))
        for _ in range(3)
    ]
    await asyncio.wait_for(entered.wait(), 3)
    finish.set()
    await asyncio.wait_for(asyncio.gather(*dispatches), 3)
    assert applied == [1]
    with factory() as db:
        row = db.get(TaskExecutionCommand, cid)
        assert row.status == "completed" and row.result == {"applied": True}


async def test_defer_uses_retry_time_without_a_command_lease(host):
    factory, _, _ = host
    cid = enqueue(host)

    async def defer(command):
        raise transport.TaskCommandDeferred("busy")

    assert await transport.dispatch_one_task_command(defer)
    assert not await transport.dispatch_one_task_command(defer)
    with factory() as db:
        row = db.get(TaskExecutionCommand, cid)
        assert row.status == "pending" and row.attempt_count == row.defer_count == 1
        assert row.retry_available_at is not None
        assert row.claim_expires_at is None and row.claimed_by is None
        row.retry_available_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    assert await transport.dispatch_one_task_command(defer)
    with factory() as db:
        assert db.get(TaskExecutionCommand, cid).attempt_count == 2


async def test_old_processing_claim_does_not_block_new_owner(host):
    factory, tid, _ = host
    cid = enqueue(
        host,
        status="processing",
        attempt_count=1,
        claimed_by="retired-worker",
        claim_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    async def reconcile(command):
        # Processing is replayed with its original identity and a new attempt;
        # the existing application handler must reconcile its persisted effects.
        assert command.command_id == "message" and command.attempt_count == 2
        assert runtime.current_task_coordinator(tid) is not None
        return {"reconciled": True}

    assert await transport.dispatch_one_task_command(reconcile)
    with factory() as db:
        assert db.get(TaskExecutionCommand, cid).result == {"reconciled": True}


async def test_replaced_owner_cannot_complete_command(host, monkeypatch):
    factory, tid, registry = host
    cid = enqueue(host)
    # Exercise the settlement fence before heartbeat-driven cancellation can
    # race it. Owner-loss cancellation is covered by the runtime tests.
    monkeypatch.setattr(registry, "_run_heartbeats", AsyncMock())

    async def execute(command):
        with factory() as db:
            task = db.get(Task, tid)
            task.lease_attempt_id = "replacement"
            db.commit()
        return {"stale": True}

    assert await transport.dispatch_one_task_command(execute)
    with factory() as db:
        row = db.get(TaskExecutionCommand, cid)
        assert row.status == "processing" and row.result is None
        assert db.get(Task, tid).lease_attempt_id == "replacement"


async def test_foreign_owner_wait_does_not_spend_command_budget(host):
    factory, tid, _ = host
    cid = enqueue(host)
    with factory() as db, db.begin():
        assert owners.acquire_task_lease_no_commit(db, tid, runner_id="other")
    assert not await transport.dispatch_one_task_command(Mock())
    with factory() as db:
        row = db.get(TaskExecutionCommand, cid)
        assert (row.attempt_count, row.failure_count, row.defer_count) == (0, 0, 0)


async def test_registry_renews_while_handler_runs_without_command_updates(host, engine):
    factory, tid, _ = host
    enqueue(host)
    statements = []

    def record(_conn, _cursor, statement, _params, _context, _many):
        if statement.lstrip().upper().startswith("UPDATE"):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:

        async def execute(command):
            statements.clear()
            await asyncio.sleep(0.09)
            assert any("last_heartbeat_at" in sql for sql in statements)
            assert not any("task_execution_commands" in sql for sql in statements)
            return {}

        assert await transport.dispatch_one_task_command(execute)
    finally:
        event.remove(engine, "before_cursor_execute", record)


async def test_busy_queue_head_does_not_starve_another_task(host):
    factory, tid, _ = host
    first_id = enqueue(host)
    with factory() as db:
        other = Task(user_id=db.get(Task, tid).user_id, title="other")
        db.add(other)
        db.flush()
        row = TaskExecutionCommand(
            task_id=other.id, command_id="other", kind="message", payload={}
        )
        db.add(row)
        db.commit()
        other_id = row.id
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def execute(command):
        seen.append(command.id)
        if command.id == first_id:
            entered.set()
            await release.wait()
        return {}

    first = asyncio.create_task(transport.dispatch_one_task_command(execute))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert await asyncio.wait_for(transport.dispatch_one_task_command(execute), 3)
        assert seen == [first_id, other_id]
    finally:
        release.set()
        await first


async def test_same_owner_rejects_older_command_attempt(host):
    factory, tid, registry = host
    cid = enqueue(host)

    async def execute(command):
        with factory() as db:
            db.get(TaskExecutionCommand, cid).attempt_count += 1
            db.commit()
        assert not transport.finish_task_command(
            cid,
            "worker",
            result={"stale": True},
            expected_attempt_count=command.attempt_count,
        )
        # The synthetic newer attempt has taken responsibility for the row.
        assert transport.finish_task_command(
            cid,
            "worker",
            result={"current": True},
            expected_attempt_count=command.attempt_count + 1,
        )
        return transport.SettledTaskCommand()

    assert await transport.dispatch_one_task_command(execute)
    with factory() as db:
        assert db.get(TaskExecutionCommand, cid).result == {"current": True}


async def test_shutdown_drains_command_while_registry_retains_lease(host):
    factory, tid, registry = host
    enqueue(host)
    entered, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(command):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            await release.wait()

    dispatch = asyncio.create_task(transport.dispatch_one_task_command(execute))
    await asyncio.wait_for(entered.wait(), 3)
    closing = asyncio.create_task(registry.close())
    await asyncio.wait_for(cleanup.wait(), 3)
    try:
        with factory() as db:
            before = db.get(Task, tid).last_heartbeat_at

        async def wait_for_renewal():
            while True:
                with factory() as db:
                    task = db.get(Task, tid)
                    assert task.runner_id == "worker"
                    if task.last_heartbeat_at > before:
                        return
                await asyncio.sleep(0.01)

        # Wait for a committed renewal, not a fixed database/scheduler latency.
        await asyncio.wait_for(wait_for_renewal(), 5)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 3)
        await asyncio.gather(dispatch, return_exceptions=True)


async def test_unrecoverable_running_owner_does_not_starve_unrelated_commands(
    host, monkeypatch
):
    factory, tid, registry = host
    first_id = enqueue(host)
    with factory() as db:
        blocked = db.get(Task, tid)
        blocked.status = TaskStatus.RUNNING
        blocked.runner_id = "retired"
        blocked.lease_attempt_id = "ambiguous-checkpoint-owner"
        blocked.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        other = Task(user_id=blocked.user_id, title="runnable")
        db.add(other)
        db.flush()
        row = TaskExecutionCommand(
            task_id=other.id, command_id="runnable", kind="message", payload={}
        )
        db.add(row)
        db.commit()
        other_id = row.id
    seen = []

    async def execute(command):
        seen.append(command.id)
        return {}

    ensure = AsyncMock(wraps=registry.ensure)
    monkeypatch.setattr(registry, "ensure", ensure)

    # Targeted dispatch must not consume some other command.
    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=first_id
    )
    assert await transport.dispatch_one_task_command(execute)
    assert seen == [other_id]
    assert not await transport.dispatch_one_task_command(execute)
    assert all(call.args[0] != tid for call in ensure.call_args_list)
    with factory() as db:
        row = db.get(TaskExecutionCommand, first_id)
        assert row.status == "pending"
        assert (row.attempt_count, row.failure_count, row.defer_count) == (0, 0, 0)
        assert db.get(Task, tid).lease_attempt_id == "ambiguous-checkpoint-owner"


@pytest.mark.parametrize(
    ("status", "runner", "expired"),
    [
        (TaskStatus.RUNNING, "worker", True),
        (TaskStatus.RUNNING, "retired", None),
        (TaskStatus.RUNNING, None, None),
        (TaskStatus.PAUSED, "retired", True),
        (TaskStatus.WAITING_FOR_USER, "retired", True),
    ],
)
async def test_candidate_filter_preserves_other_ownership_states(
    host, status, runner, expired
):
    factory, tid, _ = host
    cid = enqueue(host)
    with factory() as db, db.begin():
        task = db.get(Task, tid)
        task.status = status
        task.runner_id = runner
        task.lease_attempt_id = "existing" if runner else None
        task.lease_expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1) if expired else None
        )
    assert transport._find_command_candidate(cid)[0] == cid


async def test_recovered_running_task_reenters_dispatch_without_blacklist(host):
    from xagent.web.services.task_lease_recovery import (
        recover_task_lease_candidate_isolated,
    )
    from xagent.web.services.task_lease_service import get_expired_task_lease_candidates

    factory, tid, _ = host
    cid = enqueue(host)
    now = datetime.now(timezone.utc)
    with factory() as db, db.begin():
        task = db.get(Task, tid)
        task.status = TaskStatus.RUNNING
        task.runner_id = "retired"
        task.lease_attempt_id = "expired-owner"
        task.lease_expires_at = now - timedelta(seconds=1)
    execute = AsyncMock(return_value={"applied": True})
    assert not await transport.dispatch_one_task_command(execute)
    with factory() as db:
        candidates = get_expired_task_lease_candidates(db, cutoff=now, limit=1)
    assert len(candidates) == 1
    assert (
        recover_task_lease_candidate_isolated(candidates[0], recovered_at=now)
        == TaskStatus.FAILED
    )
    assert await transport.dispatch_one_task_command(execute, command_db_id=cid)
    execute.assert_awaited_once()


async def test_owner_acquisition_race_still_serves_next_candidate(host, monkeypatch):
    factory, tid, registry = host
    first_id = enqueue(host)
    with factory() as db, db.begin():
        other = Task(user_id=db.get(Task, tid).user_id, title="runnable")
        db.add(other)
        db.flush()
        row = TaskExecutionCommand(
            task_id=other.id, command_id="runnable", kind="message", payload={}
        )
        db.add(row)
        db.flush()
        other_id = row.id
    ensure = registry.ensure
    attempts = []

    async def race_acquisition(task_id):
        attempts.append(task_id)
        if task_id == tid:
            # The candidate was unowned when selected, but another process
            # wins before this dispatcher can acquire it.
            with factory() as db, db.begin():
                assert owners.acquire_task_lease_no_commit(
                    db, tid, runner_id="other-worker"
                )
        return await ensure(task_id)

    monkeypatch.setattr(registry, "ensure", race_acquisition)
    execute = AsyncMock(return_value={})
    assert await transport.dispatch_one_task_command(execute)
    assert attempts[0] == tid and len(attempts) == 2
    assert execute.await_args.args[0].id == other_id
    assert tid not in registry._coordinators
    with factory() as db:
        row = db.get(TaskExecutionCommand, first_id)
        assert (row.status, row.attempt_count, row.defer_count) == ("pending", 0, 0)
