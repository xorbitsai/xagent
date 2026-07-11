"""Test the thread-isolated wrapper around ``acquire_task_lease``.

Background:
    Profiling on 2026-05-20 showed the bare ``acquire_task_lease(bg_db,
    task_id)`` call inside ``_schedule_bg._runner`` accounted for 3.75s
    of synchronous DB write on the main event loop (issue #427). The
    write itself is a normal conditional UPDATE; the cost is just the
    DB round-trip on a busy worker. Wrapping it in
    ``acquire_task_lease_isolated`` lets ``_runner`` call it via
    ``asyncio.to_thread`` so the loop is released.

What this test pins:

    * The wrapper opens its own ``SessionLocal``, commits, and closes
      that session whether the inner ``acquire_task_lease`` succeeds
      or fails. Session leak here would eventually exhaust the
      connection pool under load.
    * Functional equivalence with the original helper: same TaskLease
      shape on success, ``None`` on contention.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_lease_service import (
    TaskLease,
    acquire_task_lease_isolated,
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


def _create_user(db) -> User:
    user = User(username="lease-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_task(db, user_id: int, status: TaskStatus = TaskStatus.PENDING) -> Task:
    task = Task(
        user_id=user_id,
        title="lease test",
        description="lease",
        status=status,
        execution_mode="flash",
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_isolated_acquire_returns_lease_on_clean_task(db_session) -> None:
    """Happy path: a PENDING task with no live lease returns a
    ``TaskLease`` whose ``task_id`` matches and whose ``runner_id`` is
    this process's id."""
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id))

    lease = acquire_task_lease_isolated(int(task.id))

    assert lease is not None
    assert isinstance(lease, TaskLease)
    assert lease.task_id == int(task.id)
    # ``runner_id`` defaults to this process's identity; we don't pin
    # the exact value (varies per PID) but it must be a non-empty
    # string -- the lease is useless without one.
    assert isinstance(lease.runner_id, str)
    assert lease.runner_id


def test_isolated_acquire_returns_none_when_lease_taken(db_session) -> None:
    """A task already RUNNING with a live lease owned by some other
    runner must yield ``None`` (the running-elsewhere short-circuit
    that ``_runner`` relies on to skip ``finish_turn``)."""
    from datetime import datetime, timedelta, timezone

    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id), status=TaskStatus.RUNNING)

    # Simulate another worker holding a live lease.
    task.runner_id = "other-host:9999:dead"
    task.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=120)
    db_session.commit()

    lease = acquire_task_lease_isolated(int(task.id))
    assert lease is None


def test_new_run_claim_is_atomic_across_sessions(db_session) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, int(user.id))
    task_id = int(task.id)
    barrier = Barrier(2)

    def claim(runner_id: str) -> TaskLease | None:
        barrier.wait()
        return acquire_task_lease_isolated(
            task_id,
            runner_id=runner_id,
            new_run=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, ["runner-a", "runner-b"]))

    winners = [lease for lease in results if lease is not None]
    assert len(winners) == 1
    db_session.expire_all()
    stored = db_session.query(Task).filter(Task.id == task_id).one()
    assert stored.run_id == winners[0].run_id
    assert stored.runner_id == winners[0].runner_id


def test_isolated_acquire_closes_session_on_success() -> None:
    """The wrapper must close its session even on the success path.
    Otherwise a long-running worker bleeds a connection-pool slot per
    bg task, which negates the point of moving acquire off-loop.
    """
    fake_session = MagicMock(name="fake_session")
    fake_factory = MagicMock(return_value=fake_session)

    fake_lease = TaskLease(task_id=42, runner_id="runner-x")

    with (
        patch(
            "xagent.web.models.database.get_session_local",
            return_value=fake_factory,
        ),
        patch(
            "xagent.web.services.task_lease_service.acquire_task_lease",
            return_value=fake_lease,
        ),
    ):
        result = acquire_task_lease_isolated(42)

    assert result is fake_lease
    fake_session.close.assert_called_once()


def test_isolated_acquire_closes_session_on_inner_exception() -> None:
    """Same as above for the failure path: ``acquire_task_lease`` may
    raise on connectivity issues; the wrapper's ``finally`` clause
    must still close the session.
    """
    fake_session = MagicMock(name="fake_session")
    fake_factory = MagicMock(return_value=fake_session)

    boom = RuntimeError("simulated DB connectivity failure")

    with (
        patch(
            "xagent.web.models.database.get_session_local",
            return_value=fake_factory,
        ),
        patch(
            "xagent.web.services.task_lease_service.acquire_task_lease",
            side_effect=boom,
        ),
    ):
        with pytest.raises(RuntimeError, match="simulated DB connectivity failure"):
            acquire_task_lease_isolated(42)

    fake_session.close.assert_called_once()
