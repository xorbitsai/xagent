"""What the retention purge leaves behind for readers (#2565).

Runs on SQLite and PostgreSQL through the shared ``engine`` fixture, like the
other purge tests: the ``ON DELETE SET NULL`` the tombstone must read ahead of,
the account-deletion cascade and the pinned ``onupdate`` columns are all
behaviours worth seeing on both.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.expired_task import ExpiredTaskTombstone
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerType
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.expired_tasks import find_expired_task
from xagent.web.services.task_deletion import purge_task_rows
from xagent.web.services.task_retention_purge import (
    RetentionPurgeAction,
    purge_task,
)
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
)

engine = engine_fixture

NOW = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
CONVERSATION_DAYS = 365
TRACE_DAYS = 90
#: A fixed clock for the rows' own ``onupdate`` columns, far from :data:`NOW`,
#: so a write that let them fire is visible as a changed value.
EARLIER = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_inherited_retention_env(monkeypatch):
    from xagent import config as _config

    for name in _config.RETENTION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sessions(engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sa.orm.sessionmaker(bind=engine, autoflush=False)


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite drops the offset on the way back; PostgreSQL keeps it."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _make_user(db: Session, username: str = "owner") -> int:
    user = User(username=username, password_hash="unused")
    db.add(user)
    db.flush()
    return int(user.id)


def _make_agent(db: Session, user_id: int) -> int:
    agent = Agent(user_id=user_id, name=f"agent-{user_id}")
    db.add(agent)
    db.flush()
    return int(agent.id)


def _make_task(
    db: Session,
    *,
    user_id: int,
    days_old: int,
    agent_id: int | None = None,
    source: str = "sdk",
    is_visible: bool = True,
    agent_config: dict | None = None,
) -> int:
    task = Task(
        user_id=user_id,
        title="expired-task fixture",
        status=TaskStatus.COMPLETED,
        last_activity_at=NOW - timedelta(days=days_old),
        created_at=NOW - timedelta(days=days_old + 1),
        agent_id=agent_id,
        source=source,
        is_visible=is_visible,
        agent_config=agent_config,
    )
    db.add(task)
    db.flush()
    return int(task.id)


def _make_trace(db: Session, task_id: int, suffix: str = "1") -> None:
    db.add(
        TraceEvent(
            task_id=task_id,
            event_id=f"evt-{task_id}-{suffix}",
            event_type="task_start",
            timestamp=NOW,
            data={},
        )
    )
    db.flush()


def _make_workforce_run(db: Session, *, user_id: int, task_id: int) -> tuple[int, int]:
    manager_id = _make_agent(db, user_id)
    workforce = Workforce(
        owner_user_id=user_id,
        scope_type="user",
        scope_id=str(user_id),
        name="expiry workforce",
        manager_agent_id=manager_id,
        status="active",
    )
    db.add(workforce)
    db.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task_id,
        user_id=user_id,
        status="completed",
        snapshot={"version": 1},
        last_activity_at=EARLIER,
    )
    db.add(run)
    db.flush()
    return int(workforce.id), int(run.id)


def _make_trigger_run(
    db: Session, *, user_id: int, agent_id: int, task_id: int, key: str = "run-1"
) -> int:
    trigger = AgentTrigger(
        user_id=user_id,
        agent_id=agent_id,
        type=TriggerType.SCHEDULED.value,
        name="expiry trigger",
        config={"interval_seconds": 60},
    )
    db.add(trigger)
    db.flush()
    run = TriggerRun(
        trigger_id=trigger.id,
        task_id=task_id,
        status="completed",
        idempotency_key=key,
        updated_at=EARLIER,
    )
    db.add(run)
    db.flush()
    return int(run.id)


def _purge(db: Session, task_id: int, *, dry_run: bool = False) -> RetentionPurgeAction:
    return purge_task(
        db,
        task_id,
        now=NOW,
        conversation_days=CONVERSATION_DAYS,
        trace_days=TRACE_DAYS,
        dry_run=dry_run,
    )


# --- conversation expiry -------------------------------------------------------


def test_conversation_expiry_leaves_a_content_free_tombstone(sessions) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        agent_id = _make_agent(db, user_id)
        task_id = _make_task(
            db, user_id=user_id, days_old=400, agent_id=agent_id, is_visible=False
        )
        created_at = db.get(Task, task_id).created_at
        db.commit()

        assert _purge(db, task_id) is RetentionPurgeAction.PURGED_CONVERSATION

    with sessions() as db:
        assert db.get(Task, task_id) is None
        tombstone = db.get(ExpiredTaskTombstone, task_id)
        assert tombstone is not None
        assert tombstone.user_id == user_id
        assert tombstone.agent_id == agent_id
        assert tombstone.workforce_id is None
        assert tombstone.source == "sdk"
        assert tombstone.is_visible is False
        assert tombstone.is_channel_plumbing is False
        assert _as_utc(tombstone.task_created_at) == _as_utc(created_at)
        assert _as_utc(tombstone.expired_at) == NOW
        # Nothing but predicate inputs and timestamps: no title, description,
        # input or config can be carried across.
        assert set(ExpiredTaskTombstone.__table__.columns.keys()) == {
            "task_id",
            "user_id",
            "agent_id",
            "workforce_id",
            "source",
            "is_visible",
            "is_channel_plumbing",
            "task_created_at",
            "expired_at",
        }


@pytest.mark.parametrize(
    ("marker", "expected"),
    [(True, True), (False, False), ("true", False), (1, False), (None, False)],
)
def test_only_a_literal_true_marker_is_channel_plumbing(
    sessions, marker, expected
) -> None:
    config = (
        {}
        if marker is None
        else {MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: marker}
    )
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=400, agent_config=config)
        db.commit()
        _purge(db, task_id)

    with sessions() as db:
        assert db.get(ExpiredTaskTombstone, task_id).is_channel_plumbing is expected


def test_the_workforce_is_read_before_the_delete_clears_the_run_pointer(
    sessions,
) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=400)
        workforce_id, run_id = _make_workforce_run(db, user_id=user_id, task_id=task_id)
        db.commit()

        _purge(db, task_id)

    with sessions() as db:
        assert db.get(ExpiredTaskTombstone, task_id).workforce_id == workforce_id
        run = db.get(WorkforceRun, run_id)
        assert run.task_id is None
        assert _as_utc(run.task_expired_at) == NOW
        # The outcome of the run is not the expiry's to change.
        assert run.status == "completed"
        # The preview-run reaper reads this; an expiry must not freshen it.
        assert _as_utc(run.last_activity_at) == EARLIER


def test_trigger_runs_are_marked_without_touching_their_status_or_clock(
    sessions,
) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        agent_id = _make_agent(db, user_id)
        task_id = _make_task(db, user_id=user_id, days_old=400, agent_id=agent_id)
        run_id = _make_trigger_run(
            db, user_id=user_id, agent_id=agent_id, task_id=task_id
        )
        db.commit()

        _purge(db, task_id)

    with sessions() as db:
        run = db.get(TriggerRun, run_id)
        assert run.task_id is None
        assert _as_utc(run.task_expired_at) == NOW
        assert run.status == "completed"
        assert _as_utc(run.updated_at) == EARLIER


def test_runs_of_other_tasks_are_not_marked(sessions) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        agent_id = _make_agent(db, user_id)
        expired = _make_task(db, user_id=user_id, days_old=400, agent_id=agent_id)
        kept = _make_task(db, user_id=user_id, days_old=10, agent_id=agent_id)
        kept_run = _make_trigger_run(
            db, user_id=user_id, agent_id=agent_id, task_id=kept, key="kept"
        )
        db.commit()

        _purge(db, expired)

    with sessions() as db:
        run = db.get(TriggerRun, kept_run)
        assert run.task_id == kept
        assert run.task_expired_at is None


def test_a_dry_run_records_nothing(sessions) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        agent_id = _make_agent(db, user_id)
        task_id = _make_task(db, user_id=user_id, days_old=400, agent_id=agent_id)
        run_id = _make_trigger_run(
            db, user_id=user_id, agent_id=agent_id, task_id=task_id
        )
        db.commit()

        assert (
            _purge(db, task_id, dry_run=True)
            is RetentionPurgeAction.PURGED_CONVERSATION
        )

    with sessions() as db:
        assert db.get(Task, task_id) is not None
        assert db.get(ExpiredTaskTombstone, task_id) is None
        assert db.get(TriggerRun, run_id).task_expired_at is None


def test_a_refused_task_records_nothing(sessions) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=400)
        db.get(Task, task_id).status = TaskStatus.RUNNING
        db.commit()

        assert _purge(db, task_id) is RetentionPurgeAction.SKIPPED_BUSY

    with sessions() as db:
        assert db.get(ExpiredTaskTombstone, task_id) is None


def test_a_failed_purge_rolls_the_records_back_with_the_delete(
    sessions, monkeypatch
) -> None:
    """Records and delete commit together, or a reader is told a lie."""
    import xagent.web.services.task_retention_purge as purge_module

    def fail(*_args, **_kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(purge_module, "purge_task_rows", fail)
    with sessions() as db:
        user_id = _make_user(db)
        agent_id = _make_agent(db, user_id)
        task_id = _make_task(db, user_id=user_id, days_old=400, agent_id=agent_id)
        run_id = _make_trigger_run(
            db, user_id=user_id, agent_id=agent_id, task_id=task_id
        )
        db.commit()

        with pytest.raises(RuntimeError, match="delete failed"):
            _purge(db, task_id)

    with sessions() as db:
        assert db.get(Task, task_id) is not None
        assert db.get(ExpiredTaskTombstone, task_id) is None
        assert db.get(TriggerRun, run_id).task_expired_at is None


def test_user_initiated_deletion_records_nothing(sessions) -> None:
    """An owner's delete keeps answering not-found: no tombstone, no marker."""
    with sessions() as db:
        user_id = _make_user(db)
        agent_id = _make_agent(db, user_id)
        task_id = _make_task(db, user_id=user_id, days_old=400, agent_id=agent_id)
        run_id = _make_trigger_run(
            db, user_id=user_id, agent_id=agent_id, task_id=task_id
        )
        db.commit()

        purge_task_rows(db, task_id=task_id)
        db.commit()

    with sessions() as db:
        assert db.get(ExpiredTaskTombstone, task_id) is None
        run = db.get(TriggerRun, run_id)
        assert run.task_id is None
        assert run.task_expired_at is None


# --- tombstone lifecycle -------------------------------------------------------


def _expire_one(sessions) -> tuple[int, int, int]:
    with sessions() as db:
        user_id = _make_user(db)
        agent_id = _make_agent(db, user_id)
        task_id = _make_task(db, user_id=user_id, days_old=400, agent_id=agent_id)
        db.commit()
        _purge(db, task_id)
    return user_id, agent_id, task_id


def test_account_deletion_removes_the_tombstone(sessions) -> None:
    user_id, agent_id, task_id = _expire_one(sessions)
    with sessions() as db:
        db.execute(sa.delete(Agent).where(Agent.id == agent_id))
        db.execute(sa.delete(User).where(User.id == user_id))
        db.commit()

    with sessions() as db:
        assert db.get(ExpiredTaskTombstone, task_id) is None


def test_the_account_deletion_path_removes_the_tombstone(
    sessions, monkeypatch, tmp_path
) -> None:
    """Through ``admin_users``, which deletes the user via the ORM.

    The tombstone has no ORM relationship to ``User``, so it goes only by the
    database cascade; this pins that the real deletion path reaches it.
    """
    from xagent.web.api import admin_users

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(admin_users, "get_session_local", lambda: sessions)
    # No agent: this helper deletes the user without deleting its agents,
    # which is outside what is being pinned here.
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=400)
        db.commit()
        _purge(db, task_id)
    assert find_expired_task_exists(sessions, task_id)

    assert admin_users._delete_user_rows_sync(user_id=user_id) is not None

    with sessions() as db:
        assert db.get(User, user_id) is None
        assert db.get(ExpiredTaskTombstone, task_id) is None


def find_expired_task_exists(sessions, task_id: int) -> bool:
    with sessions() as db:
        return find_expired_task(db, task_id) is not None


def test_workforce_deletion_clears_the_tombstone_workforce(sessions) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=400)
        workforce_id, _run_id = _make_workforce_run(
            db, user_id=user_id, task_id=task_id
        )
        db.commit()
        _purge(db, task_id)

    with sessions() as db:
        db.execute(sa.delete(Workforce).where(Workforce.id == workforce_id))
        db.commit()

    with sessions() as db:
        tombstone = db.get(ExpiredTaskTombstone, task_id)
        assert tombstone is not None
        assert tombstone.workforce_id is None


def test_agent_deletion_clears_the_tombstone_agent(sessions) -> None:
    _user_id, agent_id, task_id = _expire_one(sessions)
    with sessions() as db:
        db.execute(sa.delete(Agent).where(Agent.id == agent_id))
        db.commit()

    with sessions() as db:
        tombstone = db.get(ExpiredTaskTombstone, task_id)
        assert tombstone is not None
        assert tombstone.agent_id is None


def test_find_expired_task_returns_the_tombstone(sessions) -> None:
    _user_id, _agent_id, task_id = _expire_one(sessions)
    with sessions() as db:
        tombstone = find_expired_task(db, task_id)
        assert tombstone is not None
        assert tombstone.task_id == task_id


def test_find_expired_task_never_shadows_a_live_task(sessions) -> None:
    """SQLite reuses the highest deleted id; the live task must win."""
    user_id, _agent_id, task_id = _expire_one(sessions)
    with sessions() as db:
        db.add(Task(id=task_id, user_id=user_id, title="reused id"))
        db.commit()

    with sessions() as db:
        assert find_expired_task(db, task_id) is None


def test_find_expired_task_is_none_for_an_unknown_id(sessions) -> None:
    with sessions() as db:
        assert find_expired_task(db, 424242) is None


# --- trace expiry --------------------------------------------------------------


def test_trace_expiry_stamps_traces_expired_at(sessions) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=100)
        _make_trace(db, task_id)
        db.commit()

        assert _purge(db, task_id) is RetentionPurgeAction.PURGED_TRACES

    with sessions() as db:
        task = db.get(Task, task_id)
        assert _as_utc(task.traces_expired_at) == NOW
        # Trace expiry keeps the task, so it leaves no tombstone.
        assert db.get(ExpiredTaskTombstone, task_id) is None


def test_a_later_trace_expiry_restamps(sessions) -> None:
    """New turns write new trace rows, which can expire in their turn."""
    later = NOW + timedelta(days=1)
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=100)
        _make_trace(db, task_id, "1")
        db.commit()
        _purge(db, task_id)

        _make_trace(db, task_id, "2")
        db.commit()
        assert (
            purge_task(
                db,
                task_id,
                now=later,
                conversation_days=CONVERSATION_DAYS,
                trace_days=TRACE_DAYS,
            )
            is RetentionPurgeAction.PURGED_TRACES
        )

    with sessions() as db:
        assert _as_utc(db.get(Task, task_id).traces_expired_at) == later


def test_nothing_to_purge_does_not_write_the_task_row(sessions) -> None:
    """No trace and no pointer: no stamp, and no dead tuple per sweep."""
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=100)
        db.commit()

        assert _purge(db, task_id) is RetentionPurgeAction.NOTHING_TO_PURGE

    with sessions() as db:
        assert db.get(Task, task_id).traces_expired_at is None


def test_trace_dry_run_does_not_stamp(sessions) -> None:
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, days_old=100)
        _make_trace(db, task_id)
        db.commit()

        _purge(db, task_id, dry_run=True)

    with sessions() as db:
        assert db.get(Task, task_id).traces_expired_at is None
