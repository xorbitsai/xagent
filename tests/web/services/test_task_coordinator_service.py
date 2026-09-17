"""Phase B ownership contracts against SQLite and PostgreSQL transactions."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from threading import Barrier, Event
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import psycopg2_kwargs
from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.services import task_coordinator_service as coordinator
from xagent.web.services.task_execution_controller import (
    control_state_for_status,
    task_control_snapshot,
)

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def database(engine, task_id):
    return sessionmaker(engine, expire_on_commit=False), task_id


def _business_state(task):
    return (
        task.status,
        task.control_state,
        task.state_version,
        task.run_id,
        task.input,
        task.output,
        task.error_message,
        task.last_checkpoint_event_id,
        task.last_checkpoint_trace_event_id,
        task.updated_at,
    )


@pytest.mark.parametrize("status", list(TaskStatus))
def test_ownership_does_not_change_execution_state(database, monkeypatch, status):
    factory, tid = database
    with factory() as db:
        task = db.get(Task, tid)
        task.status = status
        task.control_state = control_state_for_status(status).value
        task.state_version = 12
        task.run_id = None if status == TaskStatus.PENDING else "original-run"
        task.output = "previous output"
        db.commit()
        before = _business_state(task)
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        db.commit()
        db.refresh(task)
        assert _business_state(task) == before
        assert task.lease_attempt_id == lease.attempt_id
        assert task.runner_id == lease.runner_id

        # Renewal remains valid after TTL until a successor actually takes over.
        later = coordinator.utc_now() + timedelta(days=1)
        monkeypatch.setattr(coordinator, "utc_now", lambda: later)
        assert coordinator.renew_task_lease_no_commit(db, lease)
        db.commit()
        db.refresh(task)
        assert _business_state(task) == before
        heartbeat = task.last_heartbeat_at
        if heartbeat.tzinfo is None:  # SQLite returns a naive UTC datetime.
            heartbeat = heartbeat.replace(tzinfo=later.tzinfo)
        assert heartbeat == later
        assert coordinator.release_task_lease_no_commit(db, lease) is (
            status != TaskStatus.RUNNING
        )
        db.commit()
        db.refresh(task)
        assert _business_state(task) == before
        if status != TaskStatus.RUNNING:
            assert task.runner_id is None
            assert task.lease_attempt_id is None
            assert task.lease_expires_at is None


@pytest.mark.parametrize("runner", ["worker", "other-worker"])
@pytest.mark.parametrize("expired", [False, True])
def test_existing_lease_cannot_be_reacquired(database, runner, expired):
    factory, tid = database
    with factory() as db:
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        if expired:
            task = db.get(Task, tid)
            task.lease_expires_at = coordinator.utc_now() - timedelta(seconds=1)
        db.commit()
        assert (
            coordinator.acquire_task_lease_no_commit(db, tid, runner_id=runner) is None
        )
        db.commit()
        assert db.get(Task, tid).lease_attempt_id == lease.attempt_id


def test_acquisition_and_release_are_part_of_callers_transaction(database):
    factory, tid = database
    with factory() as db:
        rolled_back = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="a")
        assert rolled_back is not None
        db.rollback()
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="b")
        assert lease is not None
        db.commit()
        assert coordinator.release_task_lease_no_commit(db, lease)
        db.rollback()
        assert coordinator.acquire_task_lease_no_commit(db, tid, runner_id="a") is None
        db.rollback()
        assert coordinator.release_task_lease_no_commit(db, lease)
        db.commit()
        next_lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="b")
        assert next_lease is not None
        assert next_lease.attempt_id != lease.attempt_id
        db.commit()
        assert not coordinator.renew_task_lease_no_commit(db, lease)
        assert not coordinator.lock_task_lease_no_commit(db, lease)
        assert not coordinator.release_task_lease_no_commit(db, lease)
        db.commit()
        assert db.get(Task, tid).lease_attempt_id == next_lease.attempt_id


def test_missing_task_does_not_grant_ownership(database):
    factory, tid = database
    with factory() as db:
        db.delete(db.get(Task, tid))
        db.commit()
        assert coordinator.acquire_task_lease_no_commit(db, tid, runner_id="a") is None


@pytest.mark.parametrize("new_run", [False, True])
def test_start_preserves_lease_and_only_new_run_clears_checkpoint(database, new_run):
    factory, tid = database
    with factory() as db:
        task = db.get(Task, tid)
        task.status = TaskStatus.PAUSED
        task.control_state = "paused"
        task.run_id = "previous-run"
        task.state_version = 3
        task.output = "previous output"
        task.error_message = "previous diagnostic"
        checkpoint = TraceEvent(
            task_id=tid,
            event_id=str(uuid4()),
            event_type="agent_execution_checkpoint",
            timestamp=coordinator.utc_now(),
            data={},
        )
        db.add(checkpoint)
        db.flush()
        task.last_checkpoint_event_id = checkpoint.event_id
        task.last_checkpoint_trace_event_id = checkpoint.id
        db.commit()
        snapshot = task_control_snapshot(task)
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        db.commit()
        db.refresh(task)
        expiry = task.lease_expires_at
        execution = coordinator.begin_task_execution_no_commit(
            db, lease, expected=snapshot, new_run=new_run
        )
        assert execution is not None
        assert execution.lease is lease
        assert (execution.run_id != snapshot.run_id) is new_run
        db.commit()
        db.refresh(task)
        assert task.run_id == execution.run_id
        assert task.status == TaskStatus.RUNNING
        assert task.control_state == "running"
        assert task.state_version == 4
        assert task.lease_attempt_id == lease.attempt_id
        assert task.lease_expires_at == expiry
        assert task.last_checkpoint_event_id == (
            None if new_run else checkpoint.event_id
        )
        assert task.last_checkpoint_trace_event_id == (
            None if new_run else checkpoint.id
        )
        assert task.output == (None if new_run else "previous output")
        assert task.error_message == (None if new_run else "previous diagnostic")


@pytest.mark.parametrize("changed", ["run", "version", "control", "status", "task"])
def test_start_rejects_changed_admission_snapshot(database, changed):
    factory, tid = database
    with factory() as db:
        task = db.get(Task, tid)
        snapshot = task_control_snapshot(task)
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        if changed == "run":
            task.run_id = "changed"
        elif changed == "version":
            task.state_version += 1
        elif changed == "control":
            task.control_state = "pause_requested"
        elif changed == "status":
            task.status = TaskStatus.PAUSED
        else:
            snapshot = replace(snapshot, task_id=tid + 1)
        db.commit()
        before = _business_state(task)
        assert (
            coordinator.begin_task_execution_no_commit(
                db, lease, expected=snapshot, new_run=True
            )
            is None
        )
        db.commit()
        db.refresh(task)
        assert _business_state(task) == before


def test_run_transition_and_execution_writes_rollback_together(database):
    factory, tid = database
    with factory() as db:
        task = db.get(Task, tid)
        snapshot = task_control_snapshot(task)
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        db.commit()
        execution = coordinator.begin_task_execution_no_commit(
            db, lease, expected=snapshot, new_run=True
        )
        assert execution is not None
        assert coordinator.lock_task_execution_no_commit(db, execution)
        db.refresh(task)
        task.output = "uncommitted result"
        db.flush()
        db.rollback()
        assert _business_state(task)[:4] == (
            snapshot.status,
            snapshot.control_state.value,
            snapshot.state_version,
            snapshot.run_id,
        )
        assert task.output is None
        assert task.lease_attempt_id == lease.attempt_id
        assert not coordinator.lock_task_execution_no_commit(db, execution)


def test_old_run_context_does_not_follow_coordinators_next_run(database):
    factory, tid = database
    with factory() as db:
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        task = db.get(Task, tid)
        first = coordinator.begin_task_execution_no_commit(
            db, lease, expected=task_control_snapshot(task), new_run=True
        )
        assert first is not None
        db.commit()
        assert coordinator.lock_task_execution_no_commit(db, first)
        db.refresh(task)
        # Settlement updates business state, retaining ownership for receipts.
        task.output = "first result"
        task.status = TaskStatus.COMPLETED
        task.control_state = "completed"
        task.state_version += 1
        db.commit()
        assert coordinator.renew_task_lease_no_commit(db, lease)
        second = coordinator.begin_task_execution_no_commit(
            db, lease, expected=task_control_snapshot(task), new_run=True
        )
        assert second is not None
        db.commit()
        assert first.lease is second.lease
        assert first.run_id != second.run_id
        assert not coordinator.lock_task_execution_no_commit(db, first)
        assert coordinator.lock_task_execution_no_commit(db, second)
        assert coordinator.lock_task_lease_no_commit(db, lease)
        assert not coordinator.release_task_lease_no_commit(db, lease)


def test_acquisition_change_invalidates_execution_even_when_run_is_same(database):
    factory, tid = database
    with factory() as db:
        task = db.get(Task, tid)
        task.status = TaskStatus.PAUSED
        task.control_state = "paused"
        task.run_id = "retained-run"
        db.commit()
        first = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert first is not None
        old_context = coordinator.TaskExecutionContext(first, task.run_id)
        assert coordinator.release_task_lease_no_commit(db, first)
        second = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert second is not None
        db.commit()
        assert not coordinator.lock_task_execution_no_commit(db, old_context)
        assert coordinator.lock_task_execution_no_commit(
            db, coordinator.TaskExecutionContext(second, task.run_id)
        )
        assert (
            coordinator.begin_task_execution_no_commit(
                db, first, expected=task_control_snapshot(task), new_run=False
            )
            is None
        )


def test_running_execution_cannot_be_started_again(database):
    factory, tid = database
    with factory() as db:
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        task = db.get(Task, tid)
        first = coordinator.begin_task_execution_no_commit(
            db, lease, expected=task_control_snapshot(task), new_run=True
        )
        assert first is not None
        db.commit()
        db.refresh(task)
        snapshot = task_control_snapshot(task)
        for new_run in (False, True):
            assert (
                coordinator.begin_task_execution_no_commit(
                    db, lease, expected=snapshot, new_run=new_run
                )
                is None
            )


def test_resume_requires_a_real_run(database):
    factory, tid = database
    with factory() as db:
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        with pytest.raises(ValueError, match="existing run_id"):
            coordinator.begin_task_execution_no_commit(
                db,
                lease,
                expected=task_control_snapshot(db.get(Task, tid)),
                new_run=False,
            )
        assert db.get(Task, tid).run_id is None


def test_same_process_concurrent_acquisitions_have_one_winner(database):
    factory, tid = database
    barrier = Barrier(2)

    def acquire():
        with factory() as db:
            barrier.wait(timeout=5)
            lease = coordinator.acquire_task_lease_no_commit(
                db, tid, runner_id="same-worker"
            )
            db.commit()
            return lease

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(acquire) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert sum(lease is not None for lease in results) == 1


def test_execution_lock_precedes_concurrent_release(database):
    factory, tid = database
    with factory() as db:
        lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="worker")
        assert lease is not None
        task = db.get(Task, tid)
        task.run_id = "settling-run"
        task.status = TaskStatus.COMPLETED
        task.control_state = "completed"
        db.commit()
    execution = coordinator.TaskExecutionContext(lease, "settling-run")
    started = Event()
    finished = Event()

    def release():
        with factory() as db:
            started.set()
            result = coordinator.release_task_lease_no_commit(db, lease)
            db.commit()
            finished.set()
            return result

    with ThreadPoolExecutor(max_workers=1) as executor, factory() as db:
        assert coordinator.lock_task_execution_no_commit(db, execution)
        future = executor.submit(release)
        try:
            assert started.wait(timeout=5)
            assert not finished.wait(timeout=0.05)
            db.get(Task, tid).output = "settled before release"
            db.commit()
        finally:
            db.rollback()
        assert future.result(timeout=10)
    with factory() as db:
        assert db.get(Task, tid).output == "settled before release"
        assert not coordinator.lock_task_execution_no_commit(db, execution)


def _acquire_in_process(url, tid, runner, ready, start, results, hold=None):
    if url.startswith("postgresql"):
        import psycopg2

        engine = sa.create_engine(
            "postgresql://", creator=lambda: psycopg2.connect(**psycopg2_kwargs(url))
        )
    else:
        engine = sa.create_engine(url)
    try:
        ready.put(runner)
        assert start.wait(timeout=30)
        with sessionmaker(engine)() as db:
            lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id=runner)
            db.commit()
        results.put(lease)
        if hold is not None:
            assert hold.wait(timeout=30)
    finally:
        engine.dispose()


def _process_connection_url(engine):
    if engine.dialect.name == "postgresql":
        with engine.connect() as db:
            name = db.scalar(sa.text("SELECT current_database()"))
        url = sa.engine.make_url(os.environ["XAGENT_TEST_POSTGRES_URL"]).set(
            database=name
        )
        return url.render_as_string(hide_password=False)
    return str(engine.url)


def test_two_worker_processes_have_one_winner(engine, task_id):
    connection_url = _process_connection_url(engine)
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_acquire_in_process,
            args=(connection_url, task_id, f"worker-{index}", ready, start, results),
        )
        for index in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert {ready.get(timeout=30), ready.get(timeout=30)} == {
            "worker-0",
            "worker-1",
        }
        start.set()
        leases = [results.get(timeout=30), results.get(timeout=30)]
        assert sum(lease is not None for lease in leases) == 1
        winner = next(lease for lease in leases if lease is not None)
        with sessionmaker(engine)() as db:
            assert db.get(Task, task_id).lease_attempt_id == winner.attempt_id
            assert db.get(Task, task_id).runner_id == winner.runner_id
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        ready.close()
        results.close()


def test_killed_owner_is_reclaimed_only_after_expiry(engine, task_id):
    context = multiprocessing.get_context("spawn")
    ready, results = context.Queue(), context.Queue()
    start, hold = context.Event(), context.Event()
    process = context.Process(
        target=_acquire_in_process,
        args=(
            _process_connection_url(engine),
            task_id,
            "killed-worker",
            ready,
            start,
            results,
            hold,
        ),
    )
    try:
        process.start()
        assert ready.get(timeout=30) == "killed-worker"
        start.set()
        old = results.get(timeout=30)
        assert old is not None
        # Terminate after the acquisition commit, without a graceful release.
        process.terminate()
        process.join(timeout=10)
        assert process.exitcode is not None and process.exitcode != 0
        with sessionmaker(engine)() as db:
            assert not coordinator.recover_expired_idle_task_lease_no_commit(
                db, task_id
            )
            assert (
                coordinator.acquire_task_lease_no_commit(db, task_id, runner_id="new")
                is None
            )
            db.rollback()
            task = db.get(Task, task_id)
            task.lease_expires_at = coordinator.utc_now() - timedelta(seconds=1)
            db.commit()
            assert coordinator.recover_expired_idle_task_lease_no_commit(db, task_id)
            new = coordinator.acquire_task_lease_no_commit(db, task_id, runner_id="new")
            assert new is not None and new.attempt_id != old.attempt_id
            db.commit()
            assert not coordinator.renew_task_lease_no_commit(db, old)
            assert not coordinator.release_task_lease_no_commit(db, old)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        ready.close()
        results.close()


@pytest.mark.parametrize("status", list(TaskStatus))
def test_expired_idle_recovery_preserves_business_state(database, status):
    factory, tid = database
    with factory() as db:
        old = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="dead")
        task = db.get(Task, tid)
        task.status = status
        task.control_state = control_state_for_status(status).value
        task.run_id = "retained-run"
        task.output = "retained output"
        task.lease_expires_at = coordinator.utc_now() - timedelta(seconds=1)
        db.commit()
        before = _business_state(task)
        recovered = coordinator.recover_expired_idle_task_lease_no_commit(db, tid)
        assert recovered is (status != TaskStatus.RUNNING)
        db.rollback()
        assert db.get(Task, tid).lease_attempt_id == old.attempt_id
        assert (
            coordinator.recover_expired_idle_task_lease_no_commit(db, tid) is recovered
        )
        db.commit()
        db.refresh(task)
        assert _business_state(task) == before


def test_renewal_and_recovery_have_one_winner(database):
    factory, tid = database
    with factory() as db:
        old = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="old")
        db.get(Task, tid).lease_expires_at = coordinator.utc_now() - timedelta(
            seconds=1
        )
        db.commit()
    barrier = Barrier(2)

    def renew():
        with factory() as db:
            barrier.wait(timeout=5)
            result = coordinator.renew_task_lease_no_commit(db, old)
            db.commit()
            return result

    def recover():
        with factory() as db:
            barrier.wait(timeout=5)
            cleared = coordinator.recover_expired_idle_task_lease_no_commit(db, tid)
            lease = coordinator.acquire_task_lease_no_commit(db, tid, runner_id="new")
            db.commit()
            assert cleared is (lease is not None)
            return lease

    with ThreadPoolExecutor(max_workers=2) as executor:
        renewed, recovered = executor.submit(renew), executor.submit(recover)
        next_lease = recovered.result(timeout=10)
        assert renewed.result(timeout=10) is (next_lease is None)
    with factory() as db:
        winner = next_lease or old
        assert db.get(Task, tid).lease_attempt_id == winner.attempt_id
