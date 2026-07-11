from __future__ import annotations

import pytest

from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.managed_task_lease import (
    claim_managed_task_lease,
    start_managed_task_lease,
)
from xagent.web.services.task_lease_service import acquire_task_lease


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
