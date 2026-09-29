"""The live-control lease fences on real PostgreSQL row locking.

The SQLite suite (test_pause_resume_lease_fence.py) pins the decisions. This
pins what only an MVCC database shows: a takeover that commits while the
fenced write waits on the row lock is seen by the write's re-evaluated WHERE
clause, so the write refuses instead of landing on the successor's run.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

import pytest
from sqlalchemy import text, update
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models import database
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import task_command_execution as commands
from xagent.web.services.task_command_execution import (
    ClientVisibleTaskCommandDeferred,
)
from xagent.web.services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    transition_task_control_state_sync,
)
from xagent.web.services.task_lease_service import TaskLease

pytestmark = pytest.mark.postgresql

# Bounds a regression that blocks on a row lock; a passing run never waits.
_BLOCKED_SECONDS = 30
OWNER = TaskLease(task_id=0, runner_id="owner", run_id="live-run", attempt_id="a")


@pytest.fixture()
def pg_sessions(monkeypatch) -> Iterator[sessionmaker[Session]]:
    with disposable_database_factory("pause_lease_fence") as make_database:
        engine = make_database("t")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Production code resolves its Session factory and engine globally.
        monkeypatch.setattr(database, "_engine", engine)
        monkeypatch.setattr(database, "_SessionLocal", sessions)
        yield sessions


def _leased_task(pg_sessions: sessionmaker[Session]) -> TaskLease:
    with pg_sessions() as db:
        user = User(username="pause-fence-owner", password_hash="x")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="t",
            description="d",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            run_id=OWNER.run_id,
            runner_id=OWNER.runner_id,
            lease_attempt_id=OWNER.attempt_id,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            control_state=TaskControlState.RUNNING.value,
            state_version=3,
        )
        db.add(task)
        db.commit()
        return TaskLease(
            task_id=int(task.id),
            runner_id=OWNER.runner_id,
            run_id=OWNER.run_id,
            attempt_id=OWNER.attempt_id,
        )


def _wait_until_blocked(pg_sessions: sessionmaker[Session], pending: Future) -> None:
    deadline = time.monotonic() + _BLOCKED_SECONDS
    with pg_sessions() as observer:
        while time.monotonic() < deadline:
            assert not pending.done(), pending.exception()
            waiting = observer.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock'"
                )
            ).scalar_one()
            if waiting:
                return
            observer.rollback()
            time.sleep(0.05)
    raise AssertionError("fenced write never waited on the takeover's row lock")


def _pause(lease: TaskLease) -> Callable[[], Any]:
    return lambda: commands._apply_pause_requested_isolated(
        lease.task_id, expected_run_id=lease.run_id, owner_leases=(lease,)
    )


def _handoff(lease: TaskLease) -> Callable[[], Any]:
    return lambda: transition_task_control_state_sync(
        lease.task_id,
        TaskControlState.RESUME_REQUESTED,
        expected_run_id=lease.run_id,
        fence_live_owner=True,
        owner_lease=lease,
    )


@pytest.mark.parametrize(
    ("write", "refusal", "applied_state"),
    [
        (_pause, ClientVisibleTaskCommandDeferred, "pause_requested"),
        (_handoff, StaleTaskRunError, "resume_requested"),
    ],
    ids=["pause", "resume-handoff"],
)
@pytest.mark.parametrize("takeover_commits", [True, False], ids=["commit", "abort"])
def test_fenced_write_rechecks_a_concurrent_takeover(
    pg_sessions, write, refusal, applied_state, takeover_commits: bool
) -> None:
    lease = _leased_task(pg_sessions)
    takeover = pg_sessions()
    try:
        # An expired-lease takeover keeps the run and mints a new attempt.
        takeover.execute(
            update(Task)
            .where(Task.id == lease.task_id)
            .values(runner_id="successor", lease_attempt_id="b")
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(write(lease))
            _wait_until_blocked(pg_sessions, pending)
            if takeover_commits:
                takeover.commit()
                with pytest.raises(refusal):
                    pending.result(timeout=_BLOCKED_SECONDS)
            else:
                takeover.rollback()
                pending.result(timeout=_BLOCKED_SECONDS)
    finally:
        takeover.rollback()
        takeover.close()

    with pg_sessions() as db:
        stored = db.get(Task, lease.task_id)
        assert stored is not None
        if takeover_commits:
            assert stored.control_state == TaskControlState.RUNNING.value
            assert stored.state_version == 3
            assert stored.lease_attempt_id == "b"
        else:
            assert stored.control_state == applied_state
            assert stored.state_version == 4
            assert stored.lease_attempt_id == lease.attempt_id
