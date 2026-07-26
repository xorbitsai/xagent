from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.managed_task_lease import (
    ManagedTaskLease,
    claim_managed_task_lease,
    claim_managed_task_lease_isolated,
    finalize_managed_task_lease_result,
    finalize_managed_task_lease_result_isolated,
    start_managed_task_lease,
)
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    acquire_task_lease,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'managed-lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_task(db) -> Task:
    user = User(username="managed-lease-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    task = Task(
        user_id=user.id,
        title="Managed lease",
        description="Managed lease",
        status=TaskStatus.PENDING,
        execution_mode="auto",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_finalize_managed_result_commits_exact_status_and_transcript(
    db_session,
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None

    assert (
        finalize_managed_task_lease_result(
            db_session,
            lease,
            status=TaskStatus.COMPLETED,
            assistant_content="completed inline",
            interactions=[{"type": "question", "content": "continue?"}],
            message_type="assistant_message",
        )
        is True
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.runner_id is None
    assert task.lease_expires_at is None
    stored = (
        db_session.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == int(task.id),
            TaskChatMessage.role == "assistant",
        )
        .one()
    )
    assert "completed inline" in stored.content


def test_finalize_managed_result_rejects_replacement_owner(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    task.runner_id = "replacement-runner"
    task.run_id = "replacement-run"
    db_session.commit()

    assert (
        finalize_managed_task_lease_result(
            db_session,
            lease,
            status=TaskStatus.COMPLETED,
            assistant_content="stale result",
        )
        is False
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "replacement-runner"
    assert (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == int(task.id))
        .count()
        == 0
    )


@pytest.mark.asyncio
async def test_isolated_finalize_rejects_replacement_runner_with_same_run_id(
    db_session,
) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    assert lease.run_id is not None

    task.runner_id = "replacement-runner"
    task.run_id = lease.run_id
    db_session.commit()

    assert (
        await finalize_managed_task_lease_result_isolated(
            lease,
            status=TaskStatus.FAILED,
        )
        is False
    )

    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "replacement-runner"
    assert task.run_id == lease.run_id
    assert task.lease_expires_at is not None


@pytest.mark.asyncio
async def test_isolated_finalize_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()

    def blocking_finalize(*_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return True

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease."
        "_finalize_managed_task_lease_result_sync",
        blocking_finalize,
    )

    settlement = asyncio.create_task(
        finalize_managed_task_lease_result_isolated(
            TaskLease(task_id=7, runner_id="runner-a", run_id="run-a"),
            status=TaskStatus.FAILED,
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    await asyncio.sleep(0)

    assert settlement.done() is False
    allow_worker.set()
    assert await settlement is True


@pytest.mark.asyncio
async def test_managed_finalize_stops_heartbeat_before_exact_settlement() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=17, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    events: list[str] = []

    async def stop_heartbeat(*_args, **_kwargs) -> TaskLeaseHeartbeatOutcome:
        events.append("stop-heartbeat")
        return TaskLeaseHeartbeatOutcome()

    async def finalize(*_args, **_kwargs) -> bool:
        events.append("finalize")
        return True

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            side_effect=stop_heartbeat,
        ),
        patch(
            "xagent.web.services.managed_task_lease."
            "finalize_managed_task_lease_result_isolated",
            side_effect=finalize,
        ),
    ):
        assert (
            await managed.finalize_result(
                status=TaskStatus.COMPLETED,
                assistant_content="done",
            )
            is True
        )
        assert await managed.close() is False

    assert events == ["stop-heartbeat", "finalize"]


@pytest.mark.asyncio
async def test_claim_projects_workforce_run_to_running_in_claim_transaction(
    db_session,
) -> None:
    user = User(username="workforce-claim-user", password_hash="hash", is_admin=False)
    manager = Agent(user=user, name="Managed claim manager")
    db_session.add_all([user, manager])
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Managed claim workforce",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = Task(
        user_id=user.id,
        title="Managed workforce claim",
        description="Managed workforce claim",
        status=TaskStatus.PENDING,
        execution_mode="auto",
        agent_id=manager.id,
        agent_config={},
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="pending",
        snapshot={},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": int(run.id)}
    db_session.commit()

    managed = claim_managed_task_lease(db_session, int(task.id))

    assert managed is not None
    db_session.refresh(run)
    assert run.status == "running"
    assert await managed.close() is True


@pytest.mark.asyncio
async def test_claim_new_run_clears_stale_terminal_snapshot(db_session) -> None:
    task = _create_task(db_session)
    task.status = TaskStatus.COMPLETED
    task.output = "previous answer"
    task.error_message = "previous error"
    db_session.commit()

    managed = claim_managed_task_lease(db_session, int(task.id))

    assert managed is not None
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.output is None
    assert task.error_message is None
    assert await managed.close() is True


@pytest.mark.asyncio
async def test_isolated_claim_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    lease = TaskLease(task_id=23, runner_id="runner-a", run_id="run-a")
    managed = object()

    def blocking_claim(_task_id: int) -> TaskLease:
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return lease

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease._claim_managed_task_lease_sync",
        blocking_claim,
    )
    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease.start_managed_task_lease",
        lambda claimed: managed if claimed is lease else None,
    )

    claim = asyncio.create_task(claim_managed_task_lease_isolated(23))
    assert await asyncio.to_thread(worker_started.wait, 1)
    await asyncio.sleep(0)
    assert claim.done() is False

    allow_worker.set()
    assert await claim is managed


@pytest.mark.asyncio
async def test_isolated_claim_cancellation_settles_late_committed_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    cleanup_finished = threading.Event()
    lease = TaskLease(task_id=24, runner_id="runner-a", run_id="run-a")

    def blocking_claim(_task_id: int) -> TaskLease:
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return lease

    def cleanup(claimed: TaskLease) -> bool:
        assert claimed is lease
        cleanup_finished.set()
        return True

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease._claim_managed_task_lease_sync",
        blocking_claim,
    )
    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
        cleanup,
    )

    claim = asyncio.create_task(claim_managed_task_lease_isolated(24))
    assert await asyncio.to_thread(worker_started.wait, 1)
    claim.cancel()
    await asyncio.sleep(0)
    assert claim.done() is False

    allow_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await claim
    assert cleanup_finished.is_set()


def test_claim_rolls_back_lease_when_running_projection_fails(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _create_task(db_session)

    def fail_projection(*_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
        raise RuntimeError("workforce projection failed")

    monkeypatch.setattr(
        "xagent.web.services.managed_task_lease.sync_workforce_run_status",
        fail_projection,
    )

    with pytest.raises(RuntimeError, match="workforce projection failed"):
        claim_managed_task_lease(db_session, int(task.id))

    db_session.refresh(task)
    assert task.status == TaskStatus.PENDING
    assert task.runner_id is None
    assert task.run_id is None


@pytest.mark.asyncio
async def test_managed_lease_releases_terminal_task(db_session) -> None:
    task = _create_task(db_session)
    managed = claim_managed_task_lease(db_session, int(task.id))
    assert managed is not None
    assert claim_managed_task_lease(db_session, int(task.id)) is None
    task.status = TaskStatus.COMPLETED
    task.control_state = "completed"
    db_session.commit()

    assert await managed.close() is True
    assert await managed.close() is False
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.runner_id is None
    assert task.lease_expires_at is None


@pytest.mark.asyncio
async def test_managed_lease_fails_an_unfinished_task(db_session) -> None:
    task = _create_task(db_session)
    lease = acquire_task_lease(db_session, int(task.id), new_run=True)
    assert lease is not None
    managed = start_managed_task_lease(lease)

    assert await managed.close() is True
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.control_state == "failed"
    assert task.runner_id is None


@pytest.mark.asyncio
async def test_managed_lease_heartbeat_timeout_skips_release_checkout() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=7, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    timeout = SQLAlchemyTimeoutError("heartbeat pool timeout")

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome(pool_timeout=timeout)),
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync"
        ) as release,
    ):
        assert await managed.close() is False

    release.assert_not_called()


@pytest.mark.asyncio
async def test_managed_lease_close_drains_release_before_cancellation() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=8, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    release_started = threading.Event()
    allow_release = threading.Event()

    def blocking_release(_lease: TaskLease) -> bool:
        release_started.set()
        assert allow_release.wait(timeout=2)
        return True

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            new=AsyncMock(return_value=TaskLeaseHeartbeatOutcome()),
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
            side_effect=blocking_release,
        ),
    ):
        closing = asyncio.create_task(managed.close())
        assert await asyncio.to_thread(release_started.wait, 1)
        closing.cancel()
        await asyncio.sleep(0.02)
        assert not closing.done()
        allow_release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing


@pytest.mark.asyncio
async def test_managed_lease_close_drains_heartbeat_before_cancellation() -> None:
    heartbeat_task = asyncio.create_task(asyncio.sleep(0))
    managed = ManagedTaskLease(
        lease=TaskLease(task_id=9, runner_id="runner-a", run_id="run-a"),
        stop_event=asyncio.Event(),
        heartbeat_task=heartbeat_task,  # type: ignore[arg-type]
    )
    heartbeat_stop_started = asyncio.Event()
    allow_heartbeat_stop = asyncio.Event()

    async def blocking_heartbeat_stop(*_args) -> TaskLeaseHeartbeatOutcome:
        heartbeat_stop_started.set()
        await allow_heartbeat_stop.wait()
        return TaskLeaseHeartbeatOutcome()

    with (
        patch(
            "xagent.web.services.managed_task_lease.stop_task_lease_heartbeat",
            side_effect=blocking_heartbeat_stop,
        ),
        patch(
            "xagent.web.services.managed_task_lease._release_managed_task_lease_sync",
            return_value=True,
        ) as release,
    ):
        closing = asyncio.create_task(managed.close())
        await heartbeat_stop_started.wait()
        closing.cancel()
        await asyncio.sleep(0.02)
        assert not closing.done()
        allow_heartbeat_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await closing

    release.assert_called_once_with(managed.lease)
