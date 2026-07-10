"""Tests for task execution leases."""

from datetime import timedelta

import pytest

from xagent.core.agent.checkpoint import CHECKPOINT_TYPE, LEGACY_CHECKPOINT_TYPES
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import ExecutionMode, Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.task_lease_service import (
    acquire_task_lease,
    mark_task_paused_if_stale,
    refresh_task_lease,
    release_task_lease,
    utc_now,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_task(db, *, status=TaskStatus.PENDING) -> Task:
    user = User(username="lease-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    task = Task(
        user_id=user.id,
        title="Lease test",
        description="Lease test",
        status=status,
        execution_mode="auto",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_task_model_default_execution_mode_is_auto(db_session) -> None:
    user = User(username="default-mode-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=user.id,
        title="Default mode",
        description="Default mode",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    assert task.execution_mode == "auto"
    assert task.execution_mode_enum == ExecutionMode.AUTO


def test_task_lease_acquire_refresh_and_release(db_session) -> None:
    task = _create_task(db_session)

    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")

    assert lease is not None
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "runner-a"
    assert task.lease_expires_at is not None
    assert task.run_id == lease.run_id
    assert task.state_version == 1
    assert task.control_state == "running"

    assert acquire_task_lease(db_session, int(task.id), runner_id="runner-b") is None
    assert refresh_task_lease(db_session, lease) is True
    assert release_task_lease(db_session, lease, status=TaskStatus.COMPLETED) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.state_version == 2
    assert task.control_state == "completed"
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_lease_acquire_rejects_a_superseded_run(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.run_id = "current-run"
    task.control_state = "running"
    task.state_version = 3
    db_session.commit()

    assert (
        acquire_task_lease(
            db_session,
            int(task.id),
            runner_id="runner-a",
            expected_run_id="old-run",
        )
        is None
    )
    db_session.refresh(task)
    assert task.run_id == "current-run"
    assert task.state_version == 3


def test_old_lease_cannot_refresh_or_release_a_new_run(db_session) -> None:
    task = _create_task(db_session)
    old_lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")
    assert old_lease is not None

    task.run_id = "new-run"
    task.status = TaskStatus.RUNNING
    task.control_state = "running"
    task.runner_id = "runner-a"
    db_session.commit()

    assert refresh_task_lease(db_session, old_lease) is False
    assert release_task_lease(db_session, old_lease, status=TaskStatus.FAILED) is False
    db_session.refresh(task)
    assert task.run_id == "new-run"
    assert task.status == TaskStatus.RUNNING


def test_stale_running_task_with_checkpoint_becomes_paused(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.add(
        TraceEvent(
            task_id=task.id,
            event_id="checkpoint-1",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
            },
        )
    )
    db_session.commit()

    assert mark_task_paused_if_stale(db_session, task) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_stale_running_task_ignores_child_agent_checkpoint(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.add(
        TraceEvent(
            task_id=task.id,
            build_id="agent_123_child",
            event_id="child-checkpoint-1",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": CHECKPOINT_TYPE,
                "snapshot": {"type": "checkpoint"},
            },
        )
    )
    db_session.commit()

    assert mark_task_paused_if_stale(db_session, task) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED
    assert task.runner_id is None
    assert task.lease_expires_at is None


def test_stale_running_task_with_legacy_checkpoint_becomes_paused(db_session) -> None:
    task = _create_task(db_session, status=TaskStatus.RUNNING)
    task.runner_id = "dead-runner"
    task.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.add(
        TraceEvent(
            task_id=task.id,
            event_id="legacy-checkpoint-1",
            event_type="system_update_general",
            timestamp=utc_now(),
            step_id=None,
            parent_event_id=None,
            data={
                "checkpoint_type": next(iter(LEGACY_CHECKPOINT_TYPES)),
                "snapshot": {"type": "checkpoint"},
            },
        )
    )
    db_session.commit()

    assert mark_task_paused_if_stale(db_session, task) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.PAUSED
    assert task.runner_id is None
    assert task.lease_expires_at is None
