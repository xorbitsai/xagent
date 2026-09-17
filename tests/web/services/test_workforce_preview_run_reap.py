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
    last_activity_at: datetime | None = None,
    task_heartbeat_at: datetime | None = None,
) -> tuple[int, int]:
    task = Task(
        user_id=user_id,
        title="preview",
        description="hi",
        status=task_status,
        agent_id=agent_id,
        source="internal",
    )
    if task_heartbeat_at is not None:
        task.last_heartbeat_at = task_heartbeat_at
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
    # Mirrors reality: a freshly created row's last_activity_at starts equal
    # to created_at (both stamped "now" at row-creation time) and only
    # diverges once a real turn bumps it via sync_workforce_run_status.
    # Overridable so tests can simulate a conversation with real activity
    # after an old created_at (PR review round 8, F-NEW-1).
    run.last_activity_at = (
        last_activity_at if last_activity_at is not None else created_at
    )
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


def test_leaves_an_old_but_actively_multi_turn_preview_run_untouched(
    db_session,
) -> None:
    """PR #1060 review round 8, F-NEW-1: created_at alone can't distinguish a
    genuinely-abandoned preview run from one that's mid-conversation but has
    simply been open a long time. sync_workforce_run_status resets
    status/completed_at every turn but never touched created_at (fixed by
    also bumping last_activity_at, which the reaper now keys off instead).
    This run's created_at is old enough to be stale on its own, but its
    last_activity_at reflects a turn that just happened -- the reaper must
    not reap it."""
    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    stale_created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    recent_activity = datetime.now(timezone.utc) - timedelta(minutes=1)
    _task_id, run_id = _make_preview_run(
        db_session,
        user_id=user_id,
        agent_id=agent_id,
        task_status=TaskStatus.RUNNING,
        run_status="running",
        created_at=stale_created_at,
        last_activity_at=recent_activity,
    )

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert pause_targets == []
    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
    assert run.status == "running"
    assert run.completed_at is None


def test_leaves_a_run_mid_single_long_turn_untouched(db_session) -> None:
    """Self-review finding after PR #1060 round 8: sync_workforce_run_status
    only bumps last_activity_at at turn boundaries (start/end), so a single
    turn that runs longer than the stale window on its own -- one long,
    tool-heavy execution with no status transition in between -- would leave
    last_activity_at stale mid-execution even though the run is genuinely,
    currently active. The reaper now also considers the task's own
    last_heartbeat_at (refreshed every ~20s for the whole duration of an
    active execution), which is the more direct "is this actually still
    running right now" signal for exactly this case."""
    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    stale_turn_start = datetime.now(timezone.utc) - timedelta(hours=3)
    recent_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=15)
    _task_id, run_id = _make_preview_run(
        db_session,
        user_id=user_id,
        agent_id=agent_id,
        task_status=TaskStatus.RUNNING,
        run_status="running",
        created_at=stale_turn_start,
        last_activity_at=stale_turn_start,
        task_heartbeat_at=recent_heartbeat,
    )

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert pause_targets == []
    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
    assert run.status == "running"
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
    """Guards the ``is_preview IS TRUE`` filter -- must never touch a real
    (non-preview) run, even one belonging to a saved workforce."""
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


def test_reaps_stale_edit_mode_preview_run_of_a_saved_workforce(db_session) -> None:
    """PR #1060 review, F5: filtering on ``workforce_id IS NULL`` alone
    missed edit-mode preview runs -- a test message sent against an
    already-saved workforce has a real, non-null workforce_id with
    is_preview True, and had no cleanup path at all before this fix
    (cancel_active_workforce_runs is only reachable from the archive
    endpoint). Arguably the more common leak, since editing an existing
    workforce is the primary post-launch workflow."""
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
        title="edit-mode preview",
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
        is_preview=True,
        snapshot={},
    )
    db_session.add(run)
    db_session.flush()
    run.created_at = stale_created_at
    run.last_activity_at = stale_created_at
    db_session.commit()
    run_id = int(run.id)
    task_id = int(task.id)

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert len(pause_targets) == 1
    assert pause_targets[0].run_id == run_id
    assert pause_targets[0].task_id == task_id
    # PR #1060 review, F1: actor_user_id is carried per-target from the
    # run's own owner (WorkforceRun.user_id is nullable=False).
    assert pause_targets[0].actor_user_id == user_id

    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
    assert run.status == "cancelled"


def test_reaps_stale_run_whose_task_already_finished_without_a_pause_target(
    db_session,
) -> None:
    """PR #1060 review (test quality): the negative half of the
    pause-target gate. A stale run stuck in an ACTIVE status (e.g.
    "pending") whose Task already reached a terminal state must still be
    cancelled, but with no PAUSE dispatch -- only a still-RUNNING task
    needs one."""
    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    stale_created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    _task_id, run_id = _make_preview_run(
        db_session,
        user_id=user_id,
        agent_id=agent_id,
        task_status=TaskStatus.COMPLETED,
        run_status="pending",
        created_at=stale_created_at,
    )

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200
    )

    assert pause_targets == []
    run = db_session.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
    assert run.status == "cancelled"
    assert run.completed_at is not None


def test_limit_caps_the_number_of_runs_reaped_per_call(db_session) -> None:
    """PR #1060 review (test quality): ``limit`` was never exercised."""
    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    stale_created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    run_ids = []
    for _ in range(3):
        _task_id, run_id = _make_preview_run(
            db_session,
            user_id=user_id,
            agent_id=agent_id,
            task_status=TaskStatus.RUNNING,
            run_status="running",
            created_at=stale_created_at,
        )
        run_ids.append(run_id)

    pause_targets = reap_stale_preview_workforce_runs(
        db_session, stale_after_seconds=7200, limit=2
    )

    assert len(pause_targets) == 2
    reaped_ids = {target.run_id for target in pause_targets}
    remaining = (
        db_session.query(WorkforceRun).filter(WorkforceRun.status == "running").all()
    )
    assert len(remaining) == 1
    assert {run.id for run in remaining} | reaped_ids == set(run_ids)


def test_reap_commits_so_a_fresh_session_sees_the_cancellation(
    db_session, tmp_path
) -> None:
    """PR #1060 review (test quality): every other test here asserts through
    the same session that made the change -- a missing internal
    ``db.commit()`` in ``reap_stale_preview_workforce_runs`` would still
    pass those. This function is called from a Celery task on a session
    that gets closed afterward, so the commit is load-bearing in
    production; verify durability via an independent session against the
    same on-disk database instead of the session under test."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    user_id = _make_user(db_session)
    agent_id = _make_agent(db_session, user_id)
    stale_created_at = datetime.now(timezone.utc) - timedelta(hours=3)
    _task_id, run_id = _make_preview_run(
        db_session,
        user_id=user_id,
        agent_id=agent_id,
        task_status=TaskStatus.RUNNING,
        run_status="running",
        created_at=stale_created_at,
    )

    reap_stale_preview_workforce_runs(db_session, stale_after_seconds=7200)

    fresh_engine = create_engine(f"sqlite:///{tmp_path / 'preview_reap.db'}")
    FreshSessionLocal = sessionmaker(bind=fresh_engine)
    fresh_db = FreshSessionLocal()
    try:
        run = fresh_db.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
        assert run.status == "cancelled"
    finally:
        fresh_db.close()
        fresh_engine.dispose()
