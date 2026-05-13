"""Tests for task execution leases."""

from datetime import timedelta

import pytest

from xagent.core.agent_v2.checkpoint import CHECKPOINT_TYPE
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus, TraceEvent
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


def test_task_lease_acquire_refresh_and_release(db_session) -> None:
    task = _create_task(db_session)

    lease = acquire_task_lease(db_session, int(task.id), runner_id="runner-a")

    assert lease is not None
    db_session.refresh(task)
    assert task.status == TaskStatus.RUNNING
    assert task.runner_id == "runner-a"
    assert task.lease_expires_at is not None

    assert acquire_task_lease(db_session, int(task.id), runner_id="runner-b") is None
    assert refresh_task_lease(db_session, lease) is True
    assert release_task_lease(db_session, lease, status=TaskStatus.COMPLETED) is True
    db_session.refresh(task)
    assert task.status == TaskStatus.COMPLETED
    assert task.runner_id is None
    assert task.lease_expires_at is None


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
