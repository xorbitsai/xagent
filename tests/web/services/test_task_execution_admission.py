"""Execution admission through durable ingress, dispatch, and owner lifecycle."""

import asyncio
import multiprocessing
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_command_terminal_event import TaskCommandTerminalEvent
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_execution_controller import task_control_snapshot

engine = engine_fixture


@pytest.fixture
async def host(engine, monkeypatch):
    # Match production: concurrent readers must not block worker commits.
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(runtime, "get_task_lease_heartbeat_seconds", lambda: 0.05)
    monkeypatch.setattr(transport, "get_runner_id", lambda: "worker-1")
    registry = runtime.TaskCoordinatorRegistry(sessions)
    registry.runner_id = "worker-1"
    monkeypatch.setattr(runtime, "_registry", registry)
    from xagent.web.models import database

    monkeypatch.setattr(database, "get_session_local", lambda: sessions)
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = user.id
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("tenant:batch", 1, 20)
    )
    try:
        yield SimpleNamespace(
            sessions=sessions, user=user_id, registry=registry, engine=engine
        )
    finally:
        await registry.close()
        admission.set_task_admission_hook(None)


def enqueue(
    host, *, task_id=None, kind=transport.TaskCommandKind.START, command_id=None
):
    with host.sessions() as db:
        if task_id is None:
            task = Task(user_id=host.user, title="Work", status=TaskStatus.PENDING)
            db.add(task)
            db.flush()
            task_id = task.id
        return transport.enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=host.user,
            command_id=command_id or f"{kind.value}-{task_id}",
            kind=kind,
            payload={},
        )


async def eventually(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


async def dispatch_next(execute):
    async with asyncio.timeout(5):
        while not await transport.dispatch_one_task_command(execute):
            await asyncio.sleep(0.01)


async def test_capacity_wait_does_not_spend_command_attempts_and_drains(host):
    first, second = enqueue(host), enqueue(host)
    release = asyncio.Event()
    started = []

    async def execute(command):
        owner = runtime.current_task_coordinator(command.task_id)
        with host.sessions() as db, db.begin():
            execution = ownership.begin_task_execution_no_commit(
                db,
                owner.lease,
                expected=task_control_snapshot(db.get(Task, command.task_id)),
                new_run=True,
            )
        started.append(command.id)

        async def run():
            await release.wait()
            with host.sessions() as db, db.begin():
                task = db.get(Task, execution.lease.task_id)
                task.status = TaskStatus.COMPLETED
                task.control_state = "completed"

        owner.track_execution(asyncio.create_task(run()))
        return {}

    assert await transport.dispatch_one_task_command(execute)
    for _ in range(70):
        assert not await transport.dispatch_one_task_command(execute)
    waiting = transport.load_task_command(second.command_id)
    assert (waiting.status, waiting.attempt_count, waiting.defer_count) == (
        "pending",
        0,
        0,
    )
    assert started == [first.command_id]
    release.set()
    await dispatch_next(execute)
    assert started == [first.command_id, second.command_id]


async def test_cancel_overtakes_capacity_wait_and_prevents_later_execution(host):
    waiting = enqueue(host)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, waiting.command_id).task_id
    cancel = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.CANCEL)
    executed = []

    async def execute(command):
        executed.append(command.id)
        with host.sessions() as db, db.begin():
            task = db.get(Task, command.task_id)
            task.status = TaskStatus.FAILED
            task.control_state = "failed"
            task.state_version += 1
        return {}

    assert await transport.dispatch_one_task_command(
        execute, command_db_id=cancel.command_id
    )
    assert transport.load_task_command(waiting.command_id).status == "failed"
    assert not await transport.dispatch_one_task_command(execute)
    assert executed == [cancel.command_id]
    with host.sessions() as db:
        event = (
            db.query(TaskCommandTerminalEvent)
            .filter_by(task_command_id=waiting.command_id)
            .one()
        )
        assert event.outcome == "failed"
        assert event.outcome_version == 0


async def test_failed_command_retry_keeps_admission_and_pending_budget(host):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("retry", 1, 1)
    )
    first = enqueue(host)

    async def reject(command):
        raise transport.TaskCommandRejected("Refused", reason="test_refusal")

    assert await transport.dispatch_one_task_command(reject)
    active = enqueue(host)
    execute = Execution(host)
    await dispatch_next(execute)
    waiting = enqueue(host)
    with host.sessions() as db, pytest.raises(admission.AdmissionQueueFull):
        transport.retry_failed_task_command(db, first.command_id)
    assert transport.load_task_command(first.command_id).status == "failed"
    execute.finish.set()
    execute.cleanup.set()
    await dispatch_next(execute)
    await asyncio.sleep(0.1)
    with host.sessions() as db:
        assert transport.retry_failed_task_command(db, first.command_id)
    with host.sessions() as db:
        assert not transport.retry_failed_task_command(db, first.command_id)
    blocker = Execution(host)
    assert await transport.dispatch_one_task_command(
        blocker, command_db_id=first.command_id
    )
    other = enqueue(host)
    assert not await transport.dispatch_one_task_command(
        blocker, command_db_id=other.command_id
    )
    assert execute.started == [active.command_id, waiting.command_id]


async def test_prompt_dispatch_cannot_jump_older_waiting_work_in_its_bucket(host):
    first, second = enqueue(host), enqueue(host)
    executed = []

    async def execute(command):
        executed.append(command.id)
        return {}

    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=second.command_id
    )
    assert await transport.dispatch_one_task_command(
        execute, command_db_id=first.command_id
    )
    assert executed == [first.command_id]


class Execution:
    """Injected command executor with separately observable settlement and cleanup."""

    def __init__(self, host):
        self.host = host
        self.started = []
        self.finish = asyncio.Event()
        self.terminal = asyncio.Event()
        self.cleanup = asyncio.Event()

    async def __call__(self, command):
        owner = runtime.current_task_coordinator(command.task_id)
        with self.host.sessions() as db, db.begin():
            ownership.begin_task_execution_no_commit(
                db,
                owner.lease,
                expected=task_control_snapshot(db.get(Task, command.task_id)),
                new_run=True,
            )
        self.started.append(command.id)

        async def run():
            try:
                await self.finish.wait()
            finally:
                with self.host.sessions() as db, db.begin():
                    task = db.get(Task, command.task_id)
                    task.status = TaskStatus.COMPLETED
                    task.control_state = "completed"
                self.terminal.set()
            await self.cleanup.wait()

        owner.track_execution(asyncio.create_task(run()))
        return {}


async def test_terminal_status_does_not_release_capacity_before_cleanup(host):
    first, second = enqueue(host), enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    execute.finish.set()
    await asyncio.wait_for(execute.terminal.wait(), 5)
    assert not await transport.dispatch_one_task_command(execute)
    assert execute.started == [first.command_id]
    execute.cleanup.set()
    await dispatch_next(execute)
    assert execute.started == [first.command_id, second.command_id]


async def test_independent_bucket_runs_while_batch_waits_and_classification_is_durable(
    host,
):
    first, waiting = enqueue(host), enqueue(host)
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("tenant:interactive", 1, 20)
    )
    interactive = enqueue(host)
    # Workers use accepted classification, not an ambient hook on each dispatch.
    admission.set_task_admission_hook(None)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    assert await transport.dispatch_one_task_command(execute)
    assert not await transport.dispatch_one_task_command(execute)
    assert execute.started == [first.command_id, interactive.command_id]
    assert transport.load_task_command(waiting.command_id).status == "pending"


async def test_pending_budget_is_atomic_with_acceptance_and_duplicate_does_not_reclassify(
    host,
):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("limited", 1, 1)
    )
    first = enqueue(host)
    with pytest.raises(admission.AdmissionQueueFull):
        enqueue(host)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, first.command_id).task_id
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("other", 2, 2)
    )
    retry = enqueue(host, task_id=task_id)
    assert not retry.created
    assert retry.command_id == first.command_id
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    assert not await transport.dispatch_one_task_command(execute)
    assert execute.started == [first.command_id]


@pytest.mark.parametrize(
    "kind",
    [
        transport.TaskCommandKind.RESUME,
        transport.TaskCommandKind.RESUME_INPUT,
        transport.TaskCommandKind.MESSAGE,
    ],
)
async def test_continuations_use_the_same_capacity_gate(host, kind):
    enqueue(host)
    waiting = enqueue(host, kind=kind)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    assert not await transport.dispatch_one_task_command(execute)
    assert transport.load_task_command(waiting.command_id).attempt_count == 0


async def test_rejected_command_releases_its_admission(host):
    first, second = enqueue(host), enqueue(host)
    rejected = []

    async def execute(command):
        rejected.append(command.id)
        raise transport.TaskCommandRejected("Execution refused", reason="test_refusal")

    assert await transport.dispatch_one_task_command(execute)
    await dispatch_next(execute)
    assert rejected == [first.command_id, second.command_id]


async def test_live_guidance_joins_existing_execution_without_an_extra_slot(host):
    first = enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, first.command_id).task_id
    message = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)
    delivered = []

    async def inject(command):
        delivered.append(command.id)
        return {}

    assert await transport.dispatch_one_task_command(
        inject, command_db_id=message.command_id
    )
    assert delivered == [message.command_id]
    other = enqueue(host)
    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=other.command_id
    )
    execute.finish.set()
    execute.cleanup.set()
    await dispatch_next(execute)
    assert execute.started == [first.command_id, other.command_id]


async def test_policy_change_does_not_silently_change_existing_bucket(host):
    enqueue(host)
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("tenant:batch", 2, 20)
    )
    with pytest.raises(ValueError, match="policy"):
        enqueue(host)


async def test_next_turn_on_same_owner_does_not_retain_previous_turn_slot(host):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("turns", 2, 20)
    )
    first = enqueue(host)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, first.command_id).task_id
    second = enqueue(host, task_id=task_id, command_id="next-turn")
    third = enqueue(host)
    first_execution, next_execution = Execution(host), Execution(host)

    async def execute(command):
        return await (
            first_execution if command.id == first.command_id else next_execution
        )(command)

    assert await transport.dispatch_one_task_command(execute)
    next_dispatch = asyncio.create_task(
        transport.dispatch_one_task_command(execute, command_db_id=second.command_id)
    )
    # The coordinator must drain the first execution before applying the next START.
    await asyncio.sleep(0.1)
    assert not next_dispatch.done()
    first_execution.finish.set()
    first_execution.cleanup.set()
    assert await asyncio.wait_for(next_dispatch, 5)
    assert await transport.dispatch_one_task_command(
        execute, command_db_id=third.command_id
    )
    assert next_execution.started == [second.command_id, third.command_id]


async def test_crash_recovery_keeps_fifo_and_stale_release_cannot_free_new_owner(host):
    from tests.web.services.coordinator_command_shared import claim_for_owner

    first = enqueue(host)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, first.command_id).task_id
        old = ownership.acquire_task_lease_no_commit(
            db, task_id, runner_id="dead-worker"
        )
        db.commit()
        assert claim_for_owner(db, old, first.command_id) is not None
        # Crash after admission commit, before execution registration.
        db.get(Task, task_id).lease_expires_at = datetime.now(timezone.utc) - timedelta(
            seconds=1
        )
        db.commit()
    second = enqueue(host)
    execute = Execution(host)
    # Expiry alone does not free a slot; recovery must retire the acquisition.
    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=second.command_id
    )
    with host.sessions() as db, db.begin():
        assert ownership.recover_expired_idle_task_lease_no_commit(db, task_id)
    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=second.command_id
    )
    assert await transport.dispatch_one_task_command(
        execute, command_db_id=first.command_id
    )
    with host.sessions() as db, db.begin():
        admission.release_task_admissions(db, old)
    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=second.command_id
    )
    assert execute.started == [first.command_id]
    execute.finish.set()
    execute.cleanup.set()
    await dispatch_next(execute)
    assert execute.started == [first.command_id, second.command_id]


def _dispatch_process(url, barrier, release, results):
    """Independent process, connection pool, coordinator registry, and runner ID."""
    from xagent.web.models import database

    engine = create_engine(url)
    apply_sqlite_concurrency_pragmas(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    database.get_session_local = lambda: sessions
    worker = f"process-{os.getpid()}"
    transport.get_runner_id = lambda: worker

    async def main():
        registry = runtime.TaskCoordinatorRegistry(sessions)
        registry.runner_id = worker
        runtime._registry = registry
        host = SimpleNamespace(sessions=sessions)
        execute = Execution(host)

        async def recorded(command):
            outcome = await execute(command)
            results.put(("started", command.id))
            return outcome

        try:
            await asyncio.to_thread(barrier.wait, 40)
            for _ in range(8):
                await transport.dispatch_one_task_command(recorded)
            results.put(("scanned", worker))
            await asyncio.to_thread(release.wait, 40)
            execute.finish.set()
            execute.cleanup.set()
        finally:
            await registry.close()
            engine.dispose()

    try:
        asyncio.run(main())
    except BaseException as error:
        results.put(("error", repr(error)))
        raise


async def test_three_worker_processes_share_one_budget(host):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("shared", 2, 20)
    )
    accepted = [enqueue(host).command_id for _ in range(6)]
    if host.engine.dialect.name == "postgresql":
        with host.engine.connect() as connection:
            database = connection.scalar(text("SELECT current_database()"))
        url = (
            make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
            .set(database=database)
            .render_as_string(hide_password=False)
        )
    else:
        url = str(host.engine.url)
    context = multiprocessing.get_context("spawn")
    barrier, release, results = context.Barrier(3), context.Event(), context.Queue()
    workers = [
        context.Process(target=_dispatch_process, args=(url, barrier, release, results))
        for _ in range(3)
    ]
    started, scanned = [], []
    try:
        for worker in workers:
            worker.start()
        while len(scanned) < 3:
            kind, value = await asyncio.to_thread(results.get, True, 60)
            assert kind != "error", value
            (started if kind == "started" else scanned).append(value)
        assert sorted(started) == accepted[:2]
        assert all(
            transport.load_task_command(command).attempt_count == 0
            for command in accepted[2:]
        )
    finally:
        release.set()
        for worker in workers:
            await asyncio.to_thread(worker.join, 15)
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
        results.close()
    assert all(worker.exitcode == 0 for worker in workers)


async def test_deferred_joined_guidance_does_not_block_other_tasks_with_free_capacity(
    host,
):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("tenant:batch", 2, 2)
    )
    first = enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, first.command_id).task_id
    message = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)

    async def inject(command):
        # The joined guidance defers on runtime injection and re-waits as
        # pending without ever having held a slot of its own.
        raise transport.TaskCommandDeferred("waiting for runtime injection")

    await transport.dispatch_one_task_command(inject, command_db_id=message.command_id)
    deferred = transport.load_task_command(message.command_id)
    assert (deferred.status, deferred.attempt_count, deferred.defer_count) == (
        "pending",
        1,
        1,
    )

    other = enqueue(host)
    # Bucket capacity is 2 and only the first task holds a slot: the deferred
    # guidance is joinable work, not a capacity waiter, so it is not the head.
    assert await transport.dispatch_one_task_command(
        execute, command_db_id=other.command_id
    )
    assert execute.started == [first.command_id, other.command_id]
    # Both slots are held now; the deferred guidance still charges the pending
    # budget while it waits, so one more waiter fills the budget of two.
    third_task = enqueue(host)
    assert not await transport.dispatch_one_task_command(
        execute, command_db_id=third_task.command_id
    )
    with pytest.raises(admission.AdmissionQueueFull):
        enqueue(host)
    execute.finish.set()
    execute.cleanup.set()


async def test_interactive_head_blocked_by_its_own_batch_command_does_not_block_others(
    host,
):
    holder = enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    # Task T: batch START waits behind the saturated batch bucket, then its
    # interactive MESSAGE waits behind that START by per-task order.
    batch = enqueue(host)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, batch.command_id).task_id
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("tenant:interactive", 1, 20)
    )
    interactive = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)
    # Task J: interactive START from another task, accepted after T's MESSAGE.
    other = enqueue(host)
    assert await transport.dispatch_one_task_command(execute)
    assert execute.started == [holder.command_id, other.command_id]
    assert transport.load_task_command(batch.command_id).status == "pending"
    assert transport.load_task_command(interactive.command_id).status == "pending"
    # Same-task order and both capacity caps still hold: J now fills the
    # interactive bucket, so a newer interactive task waits on capacity.
    newer = enqueue(host)
    assert not await transport.dispatch_one_task_command(execute)
    for waiting in (batch, interactive, newer):
        assert transport.load_task_command(waiting.command_id).attempt_count == 0
    execute.finish.set()
    execute.cleanup.set()
