"""Tests for the stale-preview-run reaper (M2: orphaned preview workforce runs).

Preview runs (workforce builder "test before save", ``workforce_id`` NULL)
are invalidated client-side only -- a closed tab, crashed browser, or network
drop leaves one running server-side forever. ``reap_stale_preview_workforce_runs``
is the scheduled backstop; these tests cover its selection and cancellation
logic in isolation from the Celery task that calls it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import get_db, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import WorkforceRun
from xagent.web.services.workforce_runtime import reap_stale_preview_workforce_runs


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'preview_reap.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()


def _make_user(db) -> int:
    user = User(
        username="preview-owner",
        email="preview-owner@example.com",
        password_hash="x",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return int(user.id)


def _make_agent(db, user_id: int) -> int:
    agent = Agent(
        user_id=user_id,
        name="Preview Manager",
        description="d",
        instructions="i",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return int(agent.id)


def _make_preview_run(
    db,
    *,
    user_id: int,
    agent_id: int,
    task_status: TaskStatus,
    run_status: str,
    created_at: datetime,
) -> tuple[int, int]:
    task = Task(
        user_id=user_id,
        title="preview",
        description="hi",
        status=task_status,
        agent_id=agent_id,
        source="internal",
    )
    db.add(task)
    db.flush()
    run = WorkforceRun(
        workforce_id=None,
        task_id=int(task.id),
        user_id=user_id,
        status=run_status,
        is_preview=True,
        snapshot={},
    )
    db.add(run)
    db.flush()
    run.created_at = created_at
    db.commit()
    return int(task.id), int(run.id)


def test_reaps_stale_active_preview_run_and_collects_pause_target(db_session) -> None:
    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    stale_created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    running_task_id, running_run_id = _make_preview_run(
        db_session,
        user_id=user_id,
        agent_id=agent_id,
        task_status=TaskStatus.RUNNING,
        run_status="running",
        created_at=stale_created_at,
    )

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert len(pause_targets) == 1
    assert pause_targets[0].run_id == running_run_id
    assert pause_targets[0].task_id == running_task_id

    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == running_run_id).one()
    assert run.status == "cancelled"
    assert run.completed_at is not None


def test_leaves_fresh_preview_run_untouched(db_session) -> None:
    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    fresh_created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    _task_id, run_id = _make_preview_run(
        db_session,
        user_id=user_id,
        agent_id=agent_id,
        task_status=TaskStatus.PENDING,
        run_status="pending",
        created_at=fresh_created_at,
    )

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert pause_targets == []
    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
    assert run.status == "pending"
    assert run.completed_at is None


def test_leaves_terminal_preview_run_untouched(db_session) -> None:
    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    stale_created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    _task_id, run_id = _make_preview_run(
        db_session,
        user_id=user_id,
        agent_id=agent_id,
        task_status=TaskStatus.COMPLETED,
        run_status="completed",
        created_at=stale_created_at,
    )

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert pause_targets == []
    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
    assert run.status == "completed"


def test_does_not_reap_active_runs_belonging_to_a_saved_workforce(db_session) -> None:
    """Guards the ``workforce_id IS NULL`` filter -- must never touch real runs."""
    from xagent.web.models.workforce import Workforce

    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    workforce = Workforce(
        owner_user_id=user_id,
        scope_type="user",
        scope_id=str(user_id),
        name="Real Workforce",
        description="d",
        manager_agent_id=agent_id,
        status="active",
    )
    db_session.add(workforce)
    db_session.commit()
    db_session.refresh(workforce)

    stale_created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    task = Task(
        user_id=user_id,
        title="real run",
        description="hi",
        status=TaskStatus.RUNNING,
        agent_id=agent_id,
        source="internal",
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=int(workforce.id),
        task_id=int(task.id),
        user_id=user_id,
        status="running",
        is_preview=False,
        snapshot={},
    )
    db_session.add(run)
    db_session.flush()
    run.created_at = stale_created_at
    db_session.commit()
    run_id = int(run.id)

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert pause_targets == []
    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
    assert run.status == "running"
