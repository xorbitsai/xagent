"""Acquisition fencing against real SQLite and PostgreSQL task rows."""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.web.api.websocket import _task_lease_snapshot
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_lease_service as leases

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def lease_database(engine, task_id, monkeypatch):
    factory = sessionmaker(engine)
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: factory)
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_db", lambda: iter([factory()])
    )
    return factory, task_id


@pytest.mark.parametrize("change", ["none", "state_version", "expired", "waiting"])
def test_valid_renewal_and_settlement_survive_epoch_check(lease_database, change):
    factory, tid = lease_database
    with factory() as db:
        lease = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        row = db.get(Task, tid)
        if change == "state_version":
            row.state_version += 1
            row.control_state = "pause_requested"
        elif change == "expired":
            row.lease_expires_at = leases.utc_now() - timedelta(seconds=1)
        elif change == "waiting":
            row.status = TaskStatus.WAITING_FOR_USER
        db.commit()
        assert leases.refresh_task_lease(db, lease) == (
            leases.TaskLeaseRefreshState.SETTLEMENT_READY
            if change == "waiting"
            else leases.TaskLeaseRefreshState.REFRESHED
        )


def test_old_epoch_lost_but_new_epoch_renews(lease_database):
    factory, tid = lease_database
    with factory() as db:
        first = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        second = leases.acquire_task_lease(
            db, tid, runner_id="worker", expected_run_id=first.run_id
        )
        assert leases.refresh_task_lease(db, first) == leases.TaskLeaseRefreshState.LOST
        assert (
            leases.refresh_task_lease(db, second)
            == leases.TaskLeaseRefreshState.REFRESHED
        )
        db.get(Task, tid).status = TaskStatus.WAITING_FOR_USER
        db.commit()
        assert leases.refresh_task_lease(db, first) == leases.TaskLeaseRefreshState.LOST
        assert (
            leases.refresh_task_lease(db, second)
            == leases.TaskLeaseRefreshState.SETTLEMENT_READY
        )


@pytest.mark.asyncio
async def test_heartbeat_sharing_requires_epoch_key(lease_database, monkeypatch):
    factory, tid = lease_database
    with factory() as db:
        first = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        second = leases.acquire_task_lease(
            db, tid, runner_id="worker", expected_run_id=first.run_id
        )
    monkeypatch.setattr(leases, "get_task_lease_heartbeat_seconds", lambda: 0.005)
    manager = leases._TaskLeaseHeartbeatManager(asyncio.get_running_loop())
    old = manager.register(first)
    new = manager.register(second)
    try:
        await asyncio.wait_for(old.terminal_event.wait(), timeout=5)
        assert old._entry.outcome.lease_lost
        assert old._entry is not new._entry
        assert not new.terminal_event.is_set()
        assert not new._entry.outcome.lease_lost
    finally:
        await old.close()
        await new.close()
        await manager.wait_until_idle()


@pytest.mark.asyncio
async def test_routing_snapshot_requires_registered_holder(lease_database, monkeypatch):
    factory, tid = lease_database
    with factory() as db:
        lease = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        snapshot = _task_lease_snapshot(db.get(Task, tid))
    manager = leases._TaskLeaseHeartbeatManager(asyncio.get_running_loop())
    monkeypatch.setattr(leases, "_task_lease_heartbeat_manager", manager)
    assert leases.registered_task_lease(snapshot) is None
    registration = manager.register(lease)
    try:
        assert leases.registered_task_lease(snapshot) is lease
    finally:
        await registration.close()
        await manager.wait_until_idle()
    assert leases.registered_task_lease(snapshot) is None


@pytest.mark.asyncio
async def test_same_epoch_handoff_keeps_shared_heartbeat(lease_database, monkeypatch):
    factory, tid = lease_database
    with factory() as db:
        lease = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
    loop = asyncio.get_running_loop()
    refreshed = asyncio.Event()
    refresh_batch = leases.refresh_task_leases_isolated

    def observe_refresh(batch):
        result = refresh_batch(batch)
        loop.call_soon_threadsafe(refreshed.set)
        return result

    monkeypatch.setattr(leases, "refresh_task_leases_isolated", observe_refresh)
    monkeypatch.setattr(leases, "get_task_lease_heartbeat_seconds", lambda: 0.005)
    manager = leases._TaskLeaseHeartbeatManager(loop)
    first = manager.register(lease)
    successor = manager.register(lease)
    try:
        assert first._entry is successor._entry
        await asyncio.wait_for(refreshed.wait(), timeout=5)
        assert not (await first.close()).lease_lost
        refreshed.clear()
        await asyncio.wait_for(refreshed.wait(), timeout=5)
        assert not successor.terminal_event.is_set()
        assert not (await successor.close()).lease_lost
    finally:
        await first.close()
        await successor.close()
        await manager.wait_until_idle()


@pytest.mark.asyncio
async def test_epoch_key_preserves_soft_pool_timeout_and_recovery(monkeypatch):
    calls = 0

    def refresh(batch):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("simulated pool wait")
        return {
            leases._task_lease_key(item): leases.TaskLeaseRefreshState.REFRESHED
            for item in batch
        }

    monkeypatch.setattr(leases, "refresh_task_leases_isolated", refresh)
    monkeypatch.setattr(leases, "is_database_pool_timeout", lambda error: True)
    monkeypatch.setattr(leases, "get_task_lease_heartbeat_seconds", lambda: 0.005)
    manager = leases._TaskLeaseHeartbeatManager(asyncio.get_running_loop())
    outcomes = []
    recovered = asyncio.Event()
    settle = manager._settle_refresh_waiter

    def observe(entry, waiter, outcome, **kwargs):
        settle(entry, waiter, outcome, **kwargs)
        outcomes.append(outcome)
        if len(outcomes) >= 2:
            recovered.set()

    monkeypatch.setattr(manager, "_settle_refresh_waiter", observe)
    registration = manager.register(
        leases.TaskLease(
            task_id=1, runner_id="worker", run_id="run", attempt_id="epoch"
        )
    )
    try:
        await asyncio.wait_for(recovered.wait(), timeout=5)
        assert outcomes[0].pool_timeout is not None
        assert not outcomes[0].lease_lost
        assert outcomes[1].pool_timeout is None
        assert not outcomes[1].lease_lost
        assert not registration.terminal_event.is_set()
    finally:
        await registration.close()
        await manager.wait_until_idle()


@pytest.mark.parametrize("identity", ["superseded", "missing"])
@pytest.mark.parametrize(
    "writer",
    [
        "release",
        "fail",
        "managed",
        "finish",
        "result",
        "resume",
        "title",
        "a2a_input",
        "reply_input",
        "checkpoint",
        "outbound",
        "usage",
    ],
)
def test_old_acquisition_cannot_commit_any_execution_result(
    lease_database, monkeypatch, identity, writer
):
    from dataclasses import replace

    from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
    from xagent.core.agent.runner import UserMessageInjectionOutcome
    from xagent.core.agent.trace import TraceEvent as CoreTraceEvent
    from xagent.web.api import a2a, chat, websocket
    from xagent.web.api.trace_handlers import DatabaseTraceHandler
    from xagent.web.api.v1 import task_reply
    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.task import TraceEvent
    from xagent.web.models.task_execution_event import TaskExecutionEvent
    from xagent.web.services.managed_task_lease import (
        finalize_managed_task_lease_result,
    )
    from xagent.web.services.task_orchestrator import finish_turn

    factory, tid = lease_database
    monkeypatch.setattr(websocket, "get_db", lambda: iter([factory()]))
    monkeypatch.setattr(websocket, "get_session_local", lambda: factory)
    monkeypatch.setattr(a2a, "get_session_local", lambda: factory)
    monkeypatch.setattr(task_reply, "get_session_local", lambda: factory)
    with factory() as db:
        old = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        current = leases.acquire_task_lease(
            db, tid, runner_id="worker", expected_run_id=old.run_id
        )
        old = replace(current, attempt_id=None) if identity == "missing" else old
        task = db.get(Task, tid)
        user_id = task.user_id
        before = (task.title, task.input, task.output, task.error_message)

    empty_outputs = websocket._PreparedTaskFileOutputs((), (), ())
    with leases.bind_task_lease_context(old), factory() as db:
        if writer == "release":
            assert not leases.release_task_lease(db, old, status=TaskStatus.COMPLETED)
        elif writer == "fail":
            assert not leases.fail_and_release_task_lease_no_commit(
                db, old, error_message="stale"
            )
            db.commit()
        elif writer == "managed":
            assert not finalize_managed_task_lease_result(
                db, old, status=TaskStatus.COMPLETED, assistant_content="stale"
            )
        elif writer == "finish":
            assert not finish_turn(db, tid, task_lease=old)
        elif writer == "result":
            result = websocket._finalize_task_execution_result_isolated(
                task_id=tid,
                task_user_id=user_id,
                pre_run_status=TaskStatus.RUNNING,
                result={"status": "completed", "success": True, "output": "stale"},
                expected_run_id=old.run_id,
                task_lease=old,
                resolved_scope_segments=(),
                prepared_outputs=empty_outputs,
            )
            assert result.late_result
        elif writer == "resume":
            result = websocket._finalize_resumed_task(
                tid,
                status="completed",
                success=True,
                output="stale",
                task_owner_user_id=user_id,
                result={"output": "stale"},
                task_lease=old,
                prepared_outputs=empty_outputs,
            )
            assert result["late_result"]
        elif writer == "title":
            assert not chat._update_task_title_isolated(tid, "stale", task_lease=old)
        elif writer == "a2a_input":
            assert not a2a._update_a2a_resume_input_sync(
                old, "stale", None, UserMessageInjectionOutcome.NOT_POSTED
            )
        elif writer == "reply_input":
            assert not task_reply._update_reply_input_sync(old, "stale", None)
        elif writer == "usage":
            from xagent.web.tracking.task_tracker import (
                TokenUsage,
                _write_task_usage_sync,
            )

            assert not _write_task_usage_sync(
                tid,
                TokenUsage(input_tokens=99),
                old.run_id,
                old.runner_id,
                old.attempt_id,
            )
        elif writer == "checkpoint":
            handler = DatabaseTraceHandler(tid)
            event = CoreTraceEvent(
                CHECKPOINT_EVENT_TYPE,
                task_id=str(tid),
                data={
                    "checkpoint_type": CHECKPOINT_TYPE,
                    "execution_id": str(tid),
                    "snapshot": {"label": "stale"},
                },
            )
            with pytest.raises(RuntimeError, match="lease"):
                handler._save_trace_event(db, event)
        else:
            with pytest.raises(RuntimeError):
                websocket._persist_agent_outbound_event(
                    tid,
                    {
                        "type": "agent_message",
                        "event_id": "stale-message",
                        "data": {"message": "stale", "expect_response": True},
                    },
                )

    with factory() as db:
        task = db.get(Task, tid)
        assert task.status == TaskStatus.RUNNING
        assert task.lease_attempt_id == current.attempt_id
        assert (task.title, task.input, task.output, task.error_message) == before
        assert task.last_checkpoint_event_id is None
        assert task.last_checkpoint_trace_event_id is None
        assert db.query(TaskChatMessage).count() == 0
        assert db.query(TaskExecutionEvent).count() == 0
        assert db.query(TraceEvent).count() == 0


def test_expiry_snapshot_cannot_release_a_new_epoch_with_identical_timestamps(
    lease_database, monkeypatch
):
    factory, tid = lease_database
    now = leases.utc_now()
    monkeypatch.setattr(leases, "utc_now", lambda: now)
    with factory() as db:
        first = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        cutoff = now + timedelta(days=1)
        candidate = leases.get_expired_task_lease_candidates(
            db, cutoff=cutoff, limit=1
        )[0]
        second = leases.acquire_task_lease(
            db, tid, runner_id="worker", expected_run_id=first.run_id
        )
        assert second.attempt_id != candidate.attempt_id
        assert not leases.recover_expired_task_lease_no_commit(
            db,
            candidate,
            status=TaskStatus.FAILED,
            recovered_at=cutoff,
            error_message="expired",
        )
        db.commit()
        assert db.get(Task, tid).lease_attempt_id == second.attempt_id
        assert db.get(Task, tid).status == TaskStatus.RUNNING


def test_acquisition_and_old_renewal_race_preserves_the_winning_epoch(lease_database):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    factory, tid = lease_database
    with factory() as db:
        old = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
    barrier = Barrier(2)

    def acquire():
        with factory() as db:
            barrier.wait(timeout=5)
            return leases.acquire_task_lease(
                db, tid, runner_id="worker", expected_run_id=old.run_id
            )

    def renew():
        with factory() as db:
            barrier.wait(timeout=5)
            return leases.refresh_task_lease(db, old)

    with ThreadPoolExecutor(max_workers=2) as executor:
        acquisition = executor.submit(acquire)
        renewal = executor.submit(renew)
        current = acquisition.result(timeout=10)
        assert renewal.result(timeout=10) in {
            leases.TaskLeaseRefreshState.REFRESHED,
            leases.TaskLeaseRefreshState.LOST,
        }
    with factory() as db:
        assert db.get(Task, tid).lease_attempt_id == current.attempt_id
        assert leases.refresh_task_lease(db, old) == leases.TaskLeaseRefreshState.LOST
        assert (
            leases.refresh_task_lease(db, current)
            == leases.TaskLeaseRefreshState.REFRESHED
        )


@pytest.mark.asyncio
async def test_cancelled_acquisition_cleanup_cannot_release_its_successor(
    lease_database,
):
    import threading

    factory, tid = lease_database
    acquired = threading.Event()
    return_lease = threading.Event()
    holder = []
    cleanup_results = []

    def acquire():
        with factory() as db:
            lease = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        holder.append(lease)
        acquired.set()
        assert return_lease.wait(timeout=5)
        return lease

    def cleanup(lease):
        with factory() as db:
            cleanup_results.append(
                leases.release_task_lease(db, lease, status=TaskStatus.FAILED)
            )

    operation = asyncio.create_task(
        leases.acquire_task_lease_cancellation_safe(acquire, cleanup)
    )
    try:
        assert await asyncio.to_thread(acquired.wait, 5)
        operation.cancel()
        with factory() as db:
            current = leases.acquire_task_lease(
                db, tid, runner_id="worker", expected_run_id=holder[0].run_id
            )
        return_lease.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert cleanup_results == [False]
        with factory() as db:
            assert db.get(Task, tid).lease_attempt_id == current.attempt_id
            assert db.get(Task, tid).status == TaskStatus.RUNNING
    finally:
        return_lease.set()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_delayed_old_heartbeat_batch_cannot_renew_or_cancel_new_registration(
    lease_database, monkeypatch
):
    import threading

    factory, tid = lease_database
    with factory() as db:
        old_lease = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
    batch_ready = threading.Event()
    deliver_batch = threading.Event()
    original_refresh = leases.refresh_task_leases_isolated
    calls = 0

    def delayed_refresh(batch):
        nonlocal calls
        states = original_refresh(batch)
        calls += 1
        if calls == 1:
            batch_ready.set()
            assert deliver_batch.wait(timeout=5)
        return states

    monkeypatch.setattr(leases, "refresh_task_leases_isolated", delayed_refresh)
    monkeypatch.setattr(leases, "get_task_lease_heartbeat_seconds", lambda: 0.005)
    manager = leases._TaskLeaseHeartbeatManager(asyncio.get_running_loop())
    old = manager.register(old_lease)
    new = None
    try:
        assert await asyncio.to_thread(batch_ready.wait, 5)
        with factory() as db:
            current = leases.acquire_task_lease(
                db, tid, runner_id="worker", expected_run_id=old_lease.run_id
            )
        new = manager.register(current)
        deliver_batch.set()
        await asyncio.wait_for(old.terminal_event.wait(), timeout=5)
        assert not new.terminal_event.is_set()
        assert not new._entry.outcome.lease_lost
        with factory() as db:
            assert db.get(Task, tid).lease_attempt_id == current.attempt_id
    finally:
        deliver_batch.set()
        await old.close()
        if new is not None:
            await new.close()
        await manager.wait_until_idle()


@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.WAITING_FOR_USER])
def test_missing_epoch_cannot_renew_or_claim_settlement_ready(lease_database, status):
    from dataclasses import replace

    factory, tid = lease_database
    with factory() as db:
        lease = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
        db.get(Task, tid).status = status
        db.commit()
        assert (
            leases.refresh_task_lease(db, replace(lease, attempt_id=None))
            == leases.TaskLeaseRefreshState.LOST
        )


def test_settlement_lock_serializes_same_process_reacquisition(lease_database):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    factory, tid = lease_database
    with factory() as db:
        old = leases.acquire_task_lease(db, tid, runner_id="worker", new_run=True)
    started = Event()
    acquired = Event()

    def takeover():
        with factory() as db:
            started.set()
            result = leases.acquire_task_lease(
                db, tid, runner_id="worker", expected_run_id=old.run_id
            )
            acquired.set()
            return result

    with ThreadPoolExecutor(max_workers=1) as executor, factory() as db:
        assert leases.lock_task_lease_no_commit(db, old)
        future = executor.submit(takeover)
        try:
            assert started.wait(timeout=5)
            assert not acquired.wait(timeout=0.05)
            db.get(Task, tid).output = "committed by original owner"
            db.commit()
        finally:
            db.rollback()
        current = future.result(timeout=5)
    with factory() as db:
        assert db.get(Task, tid).output == "committed by original owner"
        assert db.get(Task, tid).lease_attempt_id == current.attempt_id
        assert not leases.lock_task_lease_no_commit(db, old)
