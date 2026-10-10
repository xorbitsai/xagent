"""The auto-resume sweeper: dispatch, give-ups, housekeeping, drain, loop.

Nothing records ``scheduled`` rows yet, so every case seeds them directly.
Runs on SQLite (scan + compare-and-swap) and, when
``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL (``FOR UPDATE OF tasks
SKIP LOCKED``) through the shared ``engine`` fixture; the cases that need two
real transactions are PostgreSQL only.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery, TaskRecoveryEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import ops_signals, task_auto_resume
from xagent.web.services.task_auto_recovery import (
    TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
)
from xagent.web.services.task_auto_resume import (
    DISPATCH_CHECK_GRACE_SECONDS,
    TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR,
    TASK_AUTO_RESUME_DISPATCH_FAILED_TRIGGER_ERROR,
    TASK_AUTO_RESUME_EXHAUSTED_TRIGGER_ERROR,
    TASK_AUTO_RESUME_EXPIRED_TRIGGER_ERROR,
    TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR,
    auto_resume_command_id,
    public_auto_resume_view,
    run_auto_resume_tick,
)
from xagent.web.services.task_command_transport import COMMAND_ID_PATTERN
from xagent.web.services.task_execution_admission import AdmissionQueueFull
from xagent.web.services.task_lease_service import utc_now

engine = engine_fixture

RUN = "run-auto-resume"
PV = 5

_ENV = (
    "XAGENT_TASK_AUTO_RESUME_ENABLED",
    "XAGENT_TASK_AUTO_RESUME_MAX_PER_TICK",
    "XAGENT_TASK_AUTO_RESUME_MAX_INFLIGHT",
    "XAGENT_TASK_AUTO_RESUME_WINDOW_SECONDS",
    "XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN",
    "XAGENT_TASK_AUTO_RESUME_SHORT_MAX_NO_PROGRESS",
    "XAGENT_TASK_AUTO_RESUME_LLM_MAX_ELAPSED_SECONDS",
    "XAGENT_SHARED_TASK_EXECUTION_ENABLED",
)


@pytest.fixture(autouse=True)
def _clean_signals():
    ops_signals.clear_degradation(ops_signals.TASK_AUTO_RESUME_UNAVAILABLE)
    yield
    ops_signals.clear_degradation(ops_signals.TASK_AUTO_RESUME_UNAVAILABLE)


@pytest.fixture
def factory(engine, monkeypatch) -> sessionmaker:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine, autoflush=False)
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: result)
    # Selects the PostgreSQL row-lock path or the SQLite CAS path.
    monkeypatch.setattr("xagent.web.models.database.get_engine", lambda: engine)
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return result


@pytest.fixture
def notified(monkeypatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: calls.append(1)
    )
    return calls


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _user(db: Session) -> User:
    user = User(username=f"auto-resume-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    return user


def _seed(
    db: Session,
    now: datetime,
    *,
    user: User | None = None,
    state: str = "scheduled",
    reason: str = "lease_expired",
    due_in: float | None = -1,
    interrupted_ago: float = 60,
    episode_ago: float | None = None,
    no_progress: int = 0,
    total: int = 0,
    last_command_id: str | None = None,
    run_id: str = RUN,
    paused_state_version: int = PV,
    **task_fields: Any,
) -> int:
    """A PAUSED task at its fence and its recovery row."""
    fields: dict[str, Any] = {
        "status": TaskStatus.PAUSED,
        "control_state": "paused",
        "run_id": run_id,
        "state_version": paused_state_version,
    }
    fields.update(task_fields)
    task = Task(
        user_id=(user or _user(db)).id,
        title="Auto resume",
        description="auto resume test",
        execution_mode="balanced",
        **fields,
    )
    db.add(task)
    db.flush()
    db.add(
        TaskAutoRecovery(
            task_id=task.id,
            run_id=run_id,
            reason=reason,
            state=state,
            paused_state_version=paused_state_version,
            interrupted_at=now - timedelta(seconds=interrupted_ago),
            episode_started_at=now
            - timedelta(
                seconds=episode_ago if episode_ago is not None else interrupted_ago
            ),
            no_progress_resumes=no_progress,
            total_resumes=total,
            next_attempt_at=(
                now + timedelta(seconds=due_in) if due_in is not None else None
            ),
            last_command_id=last_command_id,
        )
    )
    db.flush()
    return int(task.id)


def _trigger(
    db: Session, user: User, task_id: int, *, later_run: bool = False
) -> tuple[int, int | None]:
    agent = Agent(user_id=user.id, name="auto resume agent")
    db.add(agent)
    db.flush()
    trigger = AgentTrigger(
        user_id=user.id,
        agent_id=agent.id,
        type=TriggerType.SCHEDULED.value,
        name="tick",
        config={},
    )
    db.add(trigger)
    db.flush()
    run = TriggerRun(
        trigger_id=trigger.id,
        task_id=task_id,
        status=TriggerRunStatus.RUNNING.value,
        idempotency_key=f"tick-{task_id}-1",
        started_at=utc_now(),
    )
    db.add(run)
    db.flush()
    later = None
    if later_run:
        next_run = TriggerRun(
            trigger_id=trigger.id,
            status=TriggerRunStatus.RUNNING.value,
            idempotency_key=f"tick-{task_id}-2",
            started_at=utc_now(),
        )
        db.add(next_run)
        db.flush()
        later = int(next_run.id)
    return int(run.id), later


def _trigger_task(
    db: Session, now: datetime, *, later_run: bool = False, **kwargs: Any
) -> tuple[int, int]:
    user = _user(db)
    task_id = _seed(
        db,
        now,
        user=user,
        source="trigger",
        agent_config={"trigger_type": "scheduled"},
        **kwargs,
    )
    run_id, _later = _trigger(db, user, task_id, later_run=later_run)
    return task_id, run_id


def _row(factory: sessionmaker, task_id: int) -> TaskAutoRecovery:
    with factory() as db:
        row = db.get(TaskAutoRecovery, task_id)
        db.expunge_all()
        return row


def _events(factory: sessionmaker, task_id: int) -> list[TaskRecoveryEvent]:
    with factory() as db:
        events = list(
            db.scalars(
                sa.select(TaskRecoveryEvent)
                .where(TaskRecoveryEvent.task_id == task_id)
                .order_by(TaskRecoveryEvent.id)
            )
        )
        db.expunge_all()
        return events


def _commands(factory: sessionmaker, task_id: int) -> list[TaskExecutionCommand]:
    with factory() as db:
        commands = list(
            db.scalars(
                sa.select(TaskExecutionCommand)
                .where(TaskExecutionCommand.task_id == task_id)
                .order_by(TaskExecutionCommand.id)
            )
        )
        db.expunge_all()
        return commands


def _trigger_run(factory: sessionmaker, run_id: int) -> TriggerRun:
    with factory() as db:
        run = db.get(TriggerRun, run_id)
        db.expunge_all()
        return run


def _sqlite_only(engine) -> None:
    if engine.dialect.name != "sqlite":
        pytest.skip("covered on SQLite; the PostgreSQL path is pinned separately")


# --------------------------------------------------------------------------
# Command id and public view
# --------------------------------------------------------------------------


def test_command_id_is_unique_per_fence_and_attempt_and_fits_the_column():
    assert auto_resume_command_id(5, 1, "run-1") == "auto-resume:5:1:run-1"
    long_run = "r" * 64
    command_id = auto_resume_command_id(123456, 17, long_run)
    assert len(command_id) == 64
    assert command_id.startswith("auto-resume:123456:17:")
    assert COMMAND_ID_PATTERN.fullmatch(command_id)
    assert COMMAND_ID_PATTERN.fullmatch(
        auto_resume_command_id(1, 1, "123e4567-e89b-12d3-a456-426614174000")
    )
    # A counter restart at a new fence never reuses an id.
    assert auto_resume_command_id(6, 1, long_run) != auto_resume_command_id(
        5, 1, long_run
    )


@pytest.mark.parametrize(
    ("state", "detail", "expected"),
    [
        ("scheduled", None, ("scheduled", None)),
        ("dispatched", None, ("resuming", None)),
        ("exhausted", None, ("stopped", "limit_reached")),
        ("expired", None, ("stopped", "expired")),
        ("superseded", None, ("stopped", "superseded")),
        ("dispatch_failed", None, ("stopped", "failed")),
        ("ineligible", "ineligible:preview", ("stopped", "ineligible")),
        ("manual", "auto_resume_disabled", ("stopped", "disabled")),
        ("manual", "channel_hold_pending", None),
        ("manual", None, None),
        ("stale", "dispatch:run_id", None),
        ("disabled", None, None),
    ],
)
def test_public_view_exposes_only_the_client_contract(state, detail, expected):
    at = datetime(2026, 10, 10, 8, 0, 12, tzinfo=timezone.utc)
    view = public_auto_resume_view(
        state,
        state_detail=detail,
        reason="lease_expired",
        attempt=2,
        next_attempt_at=at,
    )
    if expected is None:
        assert view is None
        return
    assert (view["status"], view["stop_reason"]) == expected
    assert set(view) == {
        "status",
        "stop_reason",
        "reason",
        "attempt",
        "next_attempt_at",
    }
    assert view["next_attempt_at"] == (
        "2026-10-10T08:00:12+00:00" if state == "scheduled" else None
    )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def test_due_row_is_dispatched_as_one_fenced_resume_command(factory, notified):
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now, no_progress=1, total=2)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert (report.dispatched, report.staged_commands) == (1, 1)
    row = _row(factory, task_id)
    expected_id = f"auto-resume:{PV}:3:{RUN}"
    assert row.state == "dispatched"
    assert (row.no_progress_resumes, row.total_resumes) == (2, 3)
    assert row.last_command_id == expected_id
    assert _aware(row.next_attempt_at) == now + timedelta(
        seconds=DISPATCH_CHECK_GRACE_SECONDS
    )
    (command,) = _commands(factory, task_id)
    assert command.command_id == expected_id
    assert command.kind == "resume"
    assert command.status == "pending"
    assert command.target_run_id == RUN
    assert command.target_state_version == PV
    assert command.payload == {
        "type": "resume_task",
        "auto_resume": {
            "expected_run_id": RUN,
            "expected_state_version": PV,
            "reason": "lease_expired",
            "attempt": 3,
        },
    }
    (event,) = _events(factory, task_id)
    assert (event.event, event.reason, event.attempt, event.run_id) == (
        "auto_resumed",
        "lease_expired",
        3,
        RUN,
    )
    assert event.detail == {"kind": "normal"}
    assert notified == [1]
    (notice,) = report.notices
    assert (notice.task_id, notice.run_id, notice.state_version) == (
        task_id,
        RUN,
        PV,
    )
    assert notice.auto_resume["status"] == "resuming"
    assert notice.auto_resume["attempt"] == 3
    message = notice.message()
    assert message["type"] == "task_recovery_update"
    assert "total_resumes" not in str(message)


def test_dispatch_does_not_require_an_empty_runner(factory, notified):
    # A PAUSED task may hold an idle shared coordinator's owner lease.
    now = utc_now()
    with factory() as db:
        task_id = _seed(
            db,
            now,
            runner_id="idle-coordinator",
            lease_expires_at=now + timedelta(minutes=5),
        )
        db.commit()

    assert run_auto_resume_tick(now=now).dispatched == 1
    assert _row(factory, task_id).state == "dispatched"


def test_row_not_yet_due_is_untouched(factory, notified):
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now, due_in=5)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.dispatched == 0
    assert _row(factory, task_id).state == "scheduled"
    assert _commands(factory, task_id) == []
    assert _events(factory, task_id) == []
    assert notified == []


@pytest.mark.parametrize(
    ("field", "task_fields"),
    [
        ("status", {"status": TaskStatus.WAITING_FOR_USER}),
        ("control_state", {"control_state": "resume_requested"}),
        ("run_id", {"run_id": "run-other"}),
        ("state_version", {"state_version": PV + 1}),
    ],
)
def test_each_fence_column_makes_the_row_stale(factory, notified, field, task_fields):
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        task = db.get(Task, task_id)
        for key, value in task_fields.items():
            setattr(task, key, value)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert (report.dispatched, report.stale) == (0, 1)
    row = _row(factory, task_id)
    assert row.state == "stale"
    assert row.state_detail == f"dispatch:{field}"
    assert row.next_attempt_at is None
    assert _commands(factory, task_id) == []
    (event,) = _events(factory, task_id)
    assert event.event == "stale"
    assert event.detail["mismatch"] == field
    assert event.detail["at"] == "dispatch"
    # A person's own action produced it; nothing is announced.
    assert report.notices == []
    assert notified == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TaskStatus.RUNNING, TriggerRunStatus.RUNNING.value),
        (TaskStatus.WAITING_FOR_USER, TriggerRunStatus.RUNNING.value),
        (TaskStatus.COMPLETED, TriggerRunStatus.COMPLETED.value),
        (TaskStatus.FAILED, TriggerRunStatus.FAILED.value),
    ],
)
def test_stale_never_fails_a_trigger_run_it_only_mirrors_an_ended_task(
    factory, notified, status, expected
):
    now = utc_now()
    with factory() as db:
        task_id, trigger_run_id = _trigger_task(db, now)
        db.get(Task, task_id).status = status
        db.commit()

    assert run_auto_resume_tick(now=now).stale == 1

    run = _trigger_run(factory, trigger_run_id)
    assert run.status == expected
    if status is TaskStatus.COMPLETED:
        assert run.error_message is None


def _workforce_missing(db: Session, now: datetime) -> int:
    return _seed(db, now, agent_config={"workforce_run_id": 999_999})


def _channel_task(db: Session, now: datetime) -> int:
    user = _user(db)
    channel = UserChannel(
        user_id=user.id, channel_type="slack", channel_name="ops", config={}
    )
    db.add(channel)
    db.flush()
    return _seed(db, now, user=user, channel_id=channel.id)


@pytest.mark.parametrize(
    ("case", "state", "detail", "event", "stop_reason"),
    [
        (
            "workforce_missing",
            "ineligible",
            "ineligible:workforce_run_missing",
            "ineligible",
            "ineligible",
        ),
        ("channel", "manual", "channel_hold_pending", "ineligible", None),
        ("expired", "expired", None, "expired", "expired"),
        ("no_progress", "exhausted", None, "exhausted", "limit_reached"),
        ("total", "exhausted", None, "exhausted", "limit_reached"),
        ("elapsed", "exhausted", None, "exhausted", "limit_reached"),
        ("no_policy", "manual", None, "ineligible", None),
    ],
)
def test_dispatch_time_limits_stop_without_a_command(
    factory, notified, monkeypatch, case, state, detail, event, stop_reason
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    now = utc_now()
    with factory() as db:
        if case == "workforce_missing":
            task_id = _workforce_missing(db, now)
        elif case == "channel":
            task_id = _channel_task(db, now)
        elif case == "expired":
            task_id = _seed(db, now, interrupted_ago=86_400 + 60)
        elif case == "no_progress":
            task_id = _seed(db, now, no_progress=3, total=3)
        elif case == "total":
            task_id = _seed(db, now, total=20)
        elif case == "elapsed":
            task_id = _seed(
                db, now, reason="llm_unavailable", episode_ago=7_200, total=5
            )
        else:
            task_id = _seed(db, now, reason="user_pause")
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.dispatched == 0
    row = _row(factory, task_id)
    assert (row.state, row.state_detail) == (state, detail)
    assert row.next_attempt_at is None
    assert _commands(factory, task_id) == []
    (recorded,) = _events(factory, task_id)
    assert recorded.event == event
    assert recorded.detail["state"] == state
    assert recorded.detail["at"] == "dispatch"
    assert [n.auto_resume["stop_reason"] for n in report.notices] == (
        [stop_reason] if stop_reason else []
    )
    assert notified == []


@pytest.mark.parametrize(
    ("case", "state", "message"),
    [
        ("superseded", "superseded", TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR),
        ("expired", "expired", TASK_AUTO_RESUME_EXPIRED_TRIGGER_ERROR),
        ("exhausted", "exhausted", TASK_AUTO_RESUME_EXHAUSTED_TRIGGER_ERROR),
        ("no_policy", "manual", TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR),
    ],
)
def test_dispatch_give_up_fails_the_running_trigger_run(
    factory, notified, case, state, message
):
    now = utc_now()
    kwargs: dict[str, Any] = {
        "superseded": {"later_run": True},
        "expired": {"interrupted_ago": 86_400 + 60},
        "exhausted": {"total": 20},
        "no_policy": {"reason": "user_pause"},
    }[case]
    with factory() as db:
        task_id, trigger_run_id = _trigger_task(db, now, **kwargs)
        db.commit()

    run_auto_resume_tick(now=now)

    assert _row(factory, task_id).state == state
    run = _trigger_run(factory, trigger_run_id)
    assert run.status == TriggerRunStatus.FAILED.value
    assert run.error_message == message
    assert _commands(factory, task_id) == []


def test_unsuperseded_trigger_task_is_dispatched_and_its_run_stays_running(
    factory, notified
):
    now = utc_now()
    with factory() as db:
        task_id, trigger_run_id = _trigger_task(db, now)
        db.commit()

    assert run_auto_resume_tick(now=now).dispatched == 1
    assert _trigger_run(factory, trigger_run_id).status == "running"
    assert _events(factory, task_id)[0].detail == {"kind": "trigger_scheduled"}


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_max_per_tick_bounds_dispatches(factory, notified, monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_PER_TICK", "2")
    now = utc_now()
    with factory() as db:
        ids = [_seed(db, now, due_in=-10 + index) for index in range(3)]
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert (report.dispatched, report.budget) == (2, 2)
    assert [_row(factory, task_id).state for task_id in ids] == [
        "dispatched",
        "dispatched",
        "scheduled",
    ]
    assert notified == [1]


def _inflight_running(db: Session, now: datetime) -> None:
    """A: an auto-resumed run still RUNNING (row confirmed, check cleared)."""
    _seed(
        db,
        now,
        state="dispatched",
        due_in=None,
        status=TaskStatus.RUNNING,
        control_state="running",
        run_id="run-a",
        state_version=PV + 2,
        paused_state_version=PV,
    )


def _inflight_queued(db: Session, now: datetime) -> None:
    """B: a dispatched command whose fence has not moved, checked later."""
    _seed(db, now, state="dispatched", due_in=30, run_id="run-b")


@pytest.mark.parametrize(
    ("inflight", "expected"),
    [((), 0), (("running",), 1), (("queued",), 1), (("running", "queued"), 2)],
)
def test_max_inflight_counts_running_and_queued_resumes(
    factory, notified, monkeypatch, inflight, expected
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_INFLIGHT", "3")
    now = utc_now()
    with factory() as db:
        for kind in inflight:
            (_inflight_running if kind == "running" else _inflight_queued)(db, now)
        # A finished resumed run and a paused row past its fence are not in
        # flight.
        _seed(
            db,
            now,
            state="dispatched",
            due_in=None,
            status=TaskStatus.COMPLETED,
            run_id="run-done",
        )
        ids = [_seed(db, now, due_in=-10 + index) for index in range(4)]
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.inflight == expected
    assert report.budget == 3 - expected
    assert report.dispatched == 3 - expected
    assert sum(_row(factory, i).state == "dispatched" for i in ids) == 3 - expected


def test_give_ups_still_advance_when_no_dispatch_budget_is_left(
    factory, notified, monkeypatch
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_INFLIGHT", "1")
    now = utc_now()
    with factory() as db:
        _inflight_queued(db, now)
        expired = _seed(db, now, due_in=-20, interrupted_ago=86_400 + 60)
        stale = _seed(db, now, due_in=-15, state_version=PV + 1)
        waiting = _seed(db, now, due_in=-10)
        after = _seed(db, now, due_in=-5, interrupted_ago=86_400 + 60)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert (report.budget, report.dispatched) == (0, 0)
    assert _row(factory, expired).state == "expired"
    assert _row(factory, stale).state == "stale"
    # A row that needs a dispatch is left unwritten and the scan goes on, so
    # it cannot hold back the give-ups behind it.
    assert _row(factory, waiting).state == "scheduled"
    assert _row(factory, after).state == "expired"
    assert _commands(factory, waiting) == []
    assert notified == []


# --------------------------------------------------------------------------
# Races and admission
# --------------------------------------------------------------------------


def test_a_lost_compare_and_swap_writes_nothing(factory, notified, engine, monkeypatch):
    _sqlite_only(engine)
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()
    # A second sweeper read the row before the first one dispatched it.
    with factory() as seen:
        stale_task = seen.get(Task, task_id)
        stale_row = seen.get(TaskAutoRecovery, task_id)
        seen.expunge_all()

    assert run_auto_resume_tick(now=now).dispatched == 1

    monkeypatch.setattr(
        task_auto_resume,
        "_load_locked_no_commit",
        lambda _db, _id: (stale_task, stale_row),
    )
    with factory() as db:
        result = task_auto_resume._process_due_candidate_no_commit(
            db, task_id, now=now, can_dispatch=True
        )
        db.rollback()
    assert result.outcome is task_auto_resume._Outcome.RACED
    assert len(_commands(factory, task_id)) == 1
    assert [e.event for e in _events(factory, task_id)] == ["auto_resumed"]
    assert _row(factory, task_id).total_resumes == 1


def _process_with_stale_view(
    factory: sessionmaker, monkeypatch, task_id: int, now: datetime, change
) -> Any:
    """Process ``task_id`` from a view read before ``change`` committed."""
    with factory() as seen:
        stale_task = seen.get(Task, task_id)
        stale_row = seen.get(TaskAutoRecovery, task_id)
        seen.expunge_all()
    with factory() as db:
        change(db)
        db.commit()
    monkeypatch.setattr(
        task_auto_resume,
        "_load_locked_no_commit",
        lambda _db, _id: (stale_task, stale_row),
    )
    with factory() as db:
        result = task_auto_resume._process_due_candidate_no_commit(
            db, task_id, now=now, can_dispatch=True
        )
        if result.writes:
            db.commit()
        else:
            db.rollback()
    return result


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("status", TaskStatus.WAITING_FOR_USER),
        ("control_state", "resume_requested"),
        ("run_id", "run-other"),
        ("state_version", PV + 1),
    ],
)
def test_dispatch_cas_rechecks_each_fence_column_in_the_database(
    factory, notified, monkeypatch, column, value
):
    # SQLite has no row lock: the task moved after the sweeper read it, and
    # only the CAS's EXISTS over ``tasks`` can notice.
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()

    def move(db: Session) -> None:
        setattr(db.get(Task, task_id), column, value)

    result = _process_with_stale_view(factory, monkeypatch, task_id, now, move)

    assert result.outcome is task_auto_resume._Outcome.RACED
    assert _row(factory, task_id).state == "scheduled"
    assert _commands(factory, task_id) == []
    assert _events(factory, task_id) == []


@pytest.mark.parametrize("column", ["paused_state_version", "run_id"])
def test_stale_cas_does_not_touch_a_rerecorded_row(
    factory, notified, monkeypatch, column
):
    # The sweeper saw the task moved off the row's fence; meanwhile a new
    # interruption re-recorded the row at a new fence. Closing it as stale
    # would drop a live schedule.
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now, state_version=PV + 1)
        db.commit()

    def rerecord(db: Session) -> None:
        row = db.get(TaskAutoRecovery, task_id)
        if column == "paused_state_version":
            row.paused_state_version = PV + 1
        else:
            row.run_id = "run-next"
            db.get(Task, task_id).run_id = "run-next"

    result = _process_with_stale_view(factory, monkeypatch, task_id, now, rerecord)

    assert result.outcome is task_auto_resume._Outcome.RACED
    row = _row(factory, task_id)
    assert row.state == "scheduled"
    assert _events(factory, task_id) == []


def test_give_up_cas_rechecks_the_fence(factory, notified, monkeypatch):
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now, interrupted_ago=86_400 + 60)
        db.commit()

    def resume_by_hand(db: Session) -> None:
        db.get(Task, task_id).state_version = PV + 1

    result = _process_with_stale_view(
        factory, monkeypatch, task_id, now, resume_by_hand
    )

    assert result.outcome is task_auto_resume._Outcome.RACED
    assert _row(factory, task_id).state == "scheduled"


@pytest.mark.parametrize("existing", ["pending", "completed", "foreign"])
def test_an_existing_command_with_the_same_id_commits_only_if_it_will_run(
    factory, notified, caplog, existing
):
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        task = db.get(Task, task_id)
        if existing == "foreign":
            db.add(
                TaskExecutionCommand(
                    task_id=task_id,
                    actor_user_id=task.user_id,
                    command_id=auto_resume_command_id(PV, 1, RUN),
                    kind="resume",
                    payload={"type": "resume_task"},
                    status="pending",
                )
            )
        db.commit()
    if existing != "foreign":
        # An earlier dispatch of this exact (fence, attempt) whose CAS a
        # second sweeper then saw undone: the row reads scheduled again.
        assert run_auto_resume_tick(now=now).dispatched == 1
        with factory() as db:
            row = db.get(TaskAutoRecovery, task_id)
            row.state, row.total_resumes, row.no_progress_resumes = "scheduled", 0, 0
            row.next_attempt_at = now - timedelta(seconds=1)
            command = db.scalars(
                sa.select(TaskExecutionCommand).where(
                    TaskExecutionCommand.task_id == task_id
                )
            ).one()
            command.status = existing
            db.commit()
        notified.clear()
    events_before = len(_events(factory, task_id))

    with caplog.at_level(logging.WARNING, logger=task_auto_resume.__name__):
        report = run_auto_resume_tick(now=now)

    assert "already existed" in caplog.text
    assert len(_commands(factory, task_id)) == 1
    assert report.staged_commands == 0
    assert notified == []
    row = _row(factory, task_id)
    if existing == "pending":
        # That command is this dispatch and will still run.
        assert report.dispatched == 1
        assert row.state == "dispatched"
        assert len(_events(factory, task_id)) == events_before + 1
    else:
        # A finished command or someone else's payload: nothing commits.
        assert (report.dispatched, report.raced) == (0, 1)
        assert (row.state, row.total_resumes) == ("scheduled", 0)
        assert len(_events(factory, task_id)) == events_before


def test_full_admission_queue_postpones_and_stops_the_tick(
    factory, notified, monkeypatch
):
    def full(_db, _command):
        raise AdmissionQueueFull("Execution queue is full")

    monkeypatch.setattr(
        "xagent.web.services.task_execution_admission.stage_task_admission", full
    )
    now = utc_now()
    with factory() as db:
        first = _seed(db, now, due_in=-10)
        second = _seed(db, now, due_in=-5)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.admission_full
    assert report.dispatched == 0
    row = _row(factory, first)
    assert row.state == "scheduled"
    assert (row.no_progress_resumes, row.total_resumes) == (0, 0)
    assert row.last_command_id is None
    assert _aware(row.next_attempt_at) == now + timedelta(seconds=30)
    assert _commands(factory, first) == []
    assert _events(factory, first) == []
    # The bucket is shared: the next candidate is not even tried.
    assert _aware(_row(factory, second).next_attempt_at) == now - timedelta(seconds=5)
    assert notified == []


def test_candidate_failure_is_isolated(factory, notified, monkeypatch):
    now = utc_now()
    with factory() as db:
        broken = _seed(db, now, due_in=-10)
        healthy = _seed(db, now, due_in=-5)
        db.commit()
    original = task_auto_resume.auto_recovery_eligibility

    def eligibility(db, task):
        if task.id == broken:
            raise RuntimeError("boom")
        return original(db, task)

    monkeypatch.setattr(task_auto_resume, "auto_recovery_eligibility", eligibility)

    report = run_auto_resume_tick(now=now)

    assert (report.failed, report.dispatched) == (1, 1)
    assert _row(factory, broken).state == "scheduled"
    assert _row(factory, healthy).state == "dispatched"


def test_postgresql_candidate_query_locks_tasks_with_skip_locked():
    captured: list[Any] = []

    class _Db:
        def execute(self, statement):
            captured.append(statement)

            class _Result:
                def first(self):
                    return None

            return _Result()

    task_auto_resume.select_next_candidate_for_update(
        _Db(),
        task_auto_resume.TaskAutoRecoveryState.SCHEDULED,
        now=utc_now(),
        after=(utc_now(), 3),
    )
    sql = " ".join(str(captured[0].compile(dialect=postgresql.dialect())).split())
    assert "FOR UPDATE OF tasks SKIP LOCKED" in sql
    assert "task_auto_recovery.next_attempt_at <=" in sql
    assert (
        "ORDER BY task_auto_recovery.next_attempt_at, task_auto_recovery.task_id" in sql
    )
    assert "LIMIT" in sql


# --------------------------------------------------------------------------
# PostgreSQL: partitioning and lock order
# --------------------------------------------------------------------------


def _postgresql_only(engine) -> None:
    if engine.dialect.name != "postgresql":
        pytest.skip("needs two real PostgreSQL transactions")


def test_postgresql_workers_partition_candidates(factory, engine):
    _postgresql_only(engine)
    now = utc_now()
    with factory() as db:
        first = _seed(db, now, due_in=-10)
        second = _seed(db, now, due_in=-5)
        db.commit()
    state = task_auto_resume.TaskAutoRecoveryState.SCHEDULED
    with factory() as one, factory() as two:
        picked_one = task_auto_resume.select_next_candidate_for_update(
            one, state, now=now
        )
        # A worker that waited on the first one's lock would fail here, not hang.
        two.execute(sa.text("SET LOCAL lock_timeout = '1s'"))
        picked_two = task_auto_resume.select_next_candidate_for_update(
            two, state, now=now
        )
        assert picked_one is not None and picked_two is not None
        assert (picked_one[1], picked_two[1]) == (first, second)
        one.rollback()
        two.rollback()


def test_postgresql_sweeper_skips_a_task_a_settlement_holds_without_deadlock(
    factory, engine, notified
):
    """A settlement holds ``tasks`` and then writes the recovery row.

    Locking the recovery row first (``FOR UPDATE OF task_auto_recovery``)
    would let the sweeper take it, then block on the key-share lock its
    command and event inserts need on ``tasks`` while the settlement blocks
    on the recovery row: a deadlock. Locking ``tasks`` with SKIP LOCKED
    leaves the row to a later tick. Every sweeper statement runs under a
    short ``lock_timeout`` so a regression fails here instead of hanging.
    """

    _postgresql_only(engine)

    @sa.event.listens_for(engine, "connect")
    def short_lock_timeout(connection, _record):
        # Outside a transaction, so the pool's reset-on-return keeps it.
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout = '1500ms'")
        connection.autocommit = False

    try:
        _sweeper_skips_a_held_task(factory, engine)
    finally:
        sa.event.remove(engine, "connect", short_lock_timeout)


def _sweeper_skips_a_held_task(factory: sessionmaker, engine) -> None:
    engine.dispose()
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()

    holder = factory()
    try:
        # The settlement: lock the task row first, as every settling path does.
        holder.execute(
            sa.select(Task.id).where(Task.id == task_id).with_for_update()
        ).one()
        outcome: dict[str, Any] = {}

        def sweep() -> None:
            try:
                outcome["report"] = run_auto_resume_tick(now=now)
            except BaseException as exc:  # pragma: no cover - reported below
                outcome["error"] = exc

        sweeper = threading.Thread(target=sweep)
        sweeper.start()
        sweeper.join(timeout=10)
        assert not sweeper.is_alive()
        assert "error" not in outcome, outcome.get("error")
        report = outcome["report"]
        assert (report.dispatched, report.failed) == (0, 0)

        # The settlement now writes the recovery row and commits unhindered.
        holder.execute(sa.text("SET LOCAL lock_timeout = '1500ms'"))
        holder.execute(
            sa.update(TaskAutoRecovery)
            .where(TaskAutoRecovery.task_id == task_id)
            .values(last_error="settled")
        )
        holder.commit()
    finally:
        holder.close()

    assert _row(factory, task_id).state == "scheduled"
    # Released, the next tick dispatches it.
    assert run_auto_resume_tick(now=now).dispatched == 1
    assert _row(factory, task_id).state == "dispatched"


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------


def _dispatched(
    db: Session, now: datetime, *, command_status: str | None, **kwargs: Any
) -> int:
    command_id = auto_resume_command_id(PV, 1, RUN)
    task_id = _seed(
        db,
        now,
        state="dispatched",
        last_command_id=command_id,
        no_progress=kwargs.pop("no_progress", 1),
        total=kwargs.pop("total", 1),
        **kwargs,
    )
    if command_status is not None:
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=db.get(Task, task_id).user_id,
                command_id=command_id,
                kind="resume",
                payload={"type": "resume_task"},
                status=command_status,
                result=(
                    {"rejection_reason": "stale_run"}
                    if command_status == "failed"
                    else {"resume_outcome": "already_in_progress"}
                    if command_status == "completed"
                    else None
                ),
            )
        )
        db.flush()
    return task_id


@pytest.mark.parametrize("command_status", ["failed", None])
def test_housekeeping_marks_a_failed_or_lost_dispatch(
    factory, notified, command_status
):
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status=command_status)
        user = db.get(User, db.get(Task, task_id).user_id)
        trigger_run_id, _later = _trigger(db, user, task_id)
        db.commit()

    report = run_auto_resume_tick(now=now)

    row = _row(factory, task_id)
    assert row.state == "dispatch_failed"
    assert row.next_attempt_at is None
    (event,) = _events(factory, task_id)
    assert event.event == "dispatch_failed"
    assert event.detail["at"] == "housekeeping"
    assert event.detail["rejection_reason"] == (
        "stale_run" if command_status == "failed" else None
    )
    run = _trigger_run(factory, trigger_run_id)
    assert run.status == TriggerRunStatus.FAILED.value
    assert run.error_message == TASK_AUTO_RESUME_DISPATCH_FAILED_TRIGGER_ERROR
    assert [n.auto_resume["stop_reason"] for n in report.notices] == ["failed"]


def test_housekeeping_reschedules_a_completed_resume_that_did_not_happen(
    factory, notified, monkeypatch
):
    monkeypatch.setattr(task_auto_resume, "_AUTO_RESUME_RNG", random.Random(7))
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status="completed")
        db.commit()

    report = run_auto_resume_tick(now=now)

    row = _row(factory, task_id)
    assert row.state == "scheduled"
    # The dispatch happened; its counters stand.
    assert (row.no_progress_resumes, row.total_resumes) == (1, 1)
    # Second short backoff (20s) plus at most 10s of jitter.
    assert now + timedelta(seconds=20) <= _aware(row.next_attempt_at)
    assert _aware(row.next_attempt_at) <= now + timedelta(seconds=30)
    (event,) = _events(factory, task_id)
    assert (event.event, event.attempt) == ("scheduled", 2)
    assert event.detail == {"at": "housekeeping", "after": "already_in_progress"}
    (notice,) = report.notices
    assert notice.auto_resume["status"] == "scheduled"
    assert notice.auto_resume["next_attempt_at"] is not None


def test_housekeeping_stops_a_completed_resume_at_its_limit(factory, notified):
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(
            db, now, command_status="completed", no_progress=3, total=3
        )
        db.commit()

    run_auto_resume_tick(now=now)

    assert _row(factory, task_id).state == "exhausted"
    (event,) = _events(factory, task_id)
    assert event.event == "exhausted"
    assert event.detail["after"] == "already_in_progress"


def test_housekeeping_cas_misses_a_row_redispatched_since_it_was_read(
    factory, notified, monkeypatch
):
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status="failed")
        db.commit()
    with factory() as seen:
        stale_task = seen.get(Task, task_id)
        stale_row = seen.get(TaskAutoRecovery, task_id)
        seen.expunge_all()
    redispatched = auto_resume_command_id(PV, 2, RUN)
    with factory() as db:
        db.get(TaskAutoRecovery, task_id).last_command_id = redispatched
        db.commit()
    monkeypatch.setattr(
        task_auto_resume,
        "_load_locked_no_commit",
        lambda _db, _id: (stale_task, stale_row),
    )

    with factory() as db:
        result = task_auto_resume._housekeep_candidate_no_commit(db, task_id, now=now)
        db.rollback()

    assert result.outcome is task_auto_resume._Outcome.RACED
    row = _row(factory, task_id)
    assert (row.state, row.last_command_id) == ("dispatched", redispatched)


@pytest.mark.parametrize("command_status", ["completed", "failed", "pending", None])
def test_housekeeping_confirms_once_the_fence_moved(factory, notified, command_status):
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status=command_status)
        task = db.get(Task, task_id)
        task.status = TaskStatus.RUNNING
        task.control_state = "running"
        task.state_version = PV + 2
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.confirmed == 1
    row = _row(factory, task_id)
    assert (row.state, row.next_attempt_at) == ("dispatched", None)
    assert _events(factory, task_id) == []
    assert report.notices == []


@pytest.mark.parametrize("command_status", ["pending", "processing"])
def test_housekeeping_rechecks_a_queued_command_later(
    factory, notified, command_status
):
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status=command_status)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.rearmed == 1
    row = _row(factory, task_id)
    assert row.state == "dispatched"
    assert _aware(row.next_attempt_at) == now + timedelta(
        seconds=DISPATCH_CHECK_GRACE_SECONDS
    )
    assert _events(factory, task_id) == []


def test_housekeeping_ignores_dispatches_not_yet_due(factory, notified):
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status=None, due_in=10)
        db.commit()

    run_auto_resume_tick(now=now)

    assert _row(factory, task_id).state == "dispatched"


# --------------------------------------------------------------------------
# Drain with the switch off
# --------------------------------------------------------------------------


def test_switch_off_drains_scheduled_rows_to_manual(factory, notified, monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    monkeypatch.setattr(task_auto_resume, "_last_inflight", 7)
    now = utc_now()
    with factory() as db:
        task_id, trigger_run_id = _trigger_task(db, now, due_in=3_600)
        moved = _seed(db, now, state_version=PV + 1)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.inflight is None
    assert task_auto_resume._last_inflight == 0
    row = _row(factory, task_id)
    assert (row.state, row.state_detail) == ("manual", "auto_resume_disabled")
    assert row.next_attempt_at is None
    (event,) = _events(factory, task_id)
    assert event.event == "skipped_disabled"
    assert event.detail["at"] == "drain"
    run = _trigger_run(factory, trigger_run_id)
    assert run.status == TriggerRunStatus.FAILED.value
    assert run.error_message == TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR
    assert _row(factory, moved).state == "stale"
    assert _commands(factory, task_id) == []
    assert [n.auto_resume["stop_reason"] for n in report.notices] == ["disabled"]
    assert notified == []


@pytest.mark.parametrize("command_status", ["completed", "failed", None])
def test_switch_off_closes_a_dispatch_whose_command_ended(
    factory, notified, monkeypatch, command_status
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status=command_status)
        user = db.get(User, db.get(Task, task_id).user_id)
        trigger_run_id, _later = _trigger(db, user, task_id)
        db.commit()

    report = run_auto_resume_tick(now=now)

    row = _row(factory, task_id)
    assert (row.state, row.state_detail) == ("manual", "auto_resume_disabled")
    assert [e.event for e in _events(factory, task_id)] == ["skipped_disabled"]
    run = _trigger_run(factory, trigger_run_id)
    assert (run.status, run.error_message) == (
        TriggerRunStatus.FAILED.value,
        TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR,
    )
    assert [n.auto_resume["stop_reason"] for n in report.notices] == ["disabled"]


def test_switch_off_leaves_a_queued_dispatch_to_the_claim_guard(
    factory, notified, monkeypatch
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    now = utc_now()
    with factory() as db:
        queued = _dispatched(db, now, command_status="pending")
        moved = _dispatched(db, now, command_status="completed", state_version=PV + 2)
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert (report.rearmed, report.confirmed) == (1, 1)
    row = _row(factory, queued)
    assert row.state == "dispatched"
    assert _aware(row.next_attempt_at) == now + timedelta(
        seconds=DISPATCH_CHECK_GRACE_SECONDS
    )
    assert (_row(factory, moved).state, _row(factory, moved).next_attempt_at) == (
        "dispatched",
        None,
    )
    assert _events(factory, queued) == []


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------


class _Stop(Exception):
    pass


async def _run_loop(monkeypatch, outcomes: list[Any], *, clock: list[float]):
    """Drive the loop through ``outcomes`` (exceptions or reports)."""
    sleeps: list[float] = []
    states: list[bool] = []

    async def run_inline(operation):
        return operation()

    def tick(*, now):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def sleep(delay):
        sleeps.append(delay)
        states.append(
            ops_signals.TASK_AUTO_RESUME_UNAVAILABLE
            in ops_signals.active_degradations()
        )
        clock[0] += delay
        if not outcomes:
            raise asyncio.CancelledError

    monkeypatch.setattr(task_auto_resume, "run_db_io_cancellation_safe", run_inline)
    monkeypatch.setattr(task_auto_resume, "run_auto_resume_tick", tick)
    monkeypatch.setattr(task_auto_resume.asyncio, "sleep", sleep)
    monkeypatch.setattr(task_auto_resume, "_clock", lambda: clock[0])
    with pytest.raises(asyncio.CancelledError):
        await task_auto_resume.run_task_auto_resume_loop(poll_interval_seconds=5)
    return sleeps, states


@pytest.mark.asyncio
async def test_loop_backs_off_raises_the_signal_and_recovers(monkeypatch, caplog):
    ok = task_auto_resume.AutoResumeTickReport()
    outcomes: list[Any] = [RuntimeError("db down")] * 5 + [ok, ok]
    with caplog.at_level(logging.WARNING, logger=task_auto_resume.__name__):
        sleeps, signal = await _run_loop(monkeypatch, outcomes, clock=[0.0])

    assert sleeps == [10, 20, 40, 60, 60, 5, 5]
    assert signal == [False, False, True, True, True, False, False]
    # Logged on the first failure, then at most once a minute.
    failures = [r for r in caplog.records if "tick failed" in r.getMessage()]
    assert [r.getMessage() for r in failures] == [
        "Auto-resume tick failed (failures=1)",
        "Auto-resume tick failed (failures=4)",
        "Auto-resume tick failed (failures=5)",
    ]


@pytest.mark.asyncio
async def test_loop_publishes_each_ticks_notices(monkeypatch):
    published: list[Any] = []

    async def publish(notices):
        published.extend(notices)

    monkeypatch.setattr(task_auto_resume, "publish_recovery_notices", publish)
    notice = task_auto_resume.RecoveryNotice(
        task_id=1, run_id="r", state_version=2, auto_resume={"status": "resuming"}
    )
    report = task_auto_resume.AutoResumeTickReport(notices=(notice,))
    await _run_loop(monkeypatch, [report], clock=[0.0])
    assert published == [notice]


@pytest.mark.asyncio
async def test_loop_lets_cancellation_through(monkeypatch):
    async def run_inline(operation):
        raise asyncio.CancelledError

    monkeypatch.setattr(task_auto_resume, "run_db_io_cancellation_safe", run_inline)
    with pytest.raises(asyncio.CancelledError):
        await task_auto_resume.run_task_auto_resume_loop(poll_interval_seconds=5)


@pytest.mark.asyncio
async def test_publish_is_best_effort(monkeypatch):
    sent: list[tuple[dict[str, Any], int]] = []

    async def sink(message, task_id):
        sent.append((message, task_id))
        if len(sent) == 1:
            raise RuntimeError("socket gone")

    monkeypatch.setattr("xagent.web.services.task_events._task_event_sink", sink)
    notices = [
        task_auto_resume.RecoveryNotice(
            task_id=task_id, run_id="r", state_version=2, auto_resume={}
        )
        for task_id in (1, 2)
    ]
    await task_auto_resume.publish_recovery_notices(notices)
    assert [task_id for _message, task_id in sent] == [1, 2]
    assert sent[0][0]["type"] == "task_recovery_update"


# --------------------------------------------------------------------------
# Command order
# --------------------------------------------------------------------------


def test_user_commands_queue_behind_a_pending_auto_resume(factory, notified):
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        claim_task_command,
        finish_task_command,
        stage_task_command,
    )

    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()
    assert run_auto_resume_tick(now=now).dispatched == 1
    with factory() as db:
        user_id = int(db.get(Task, task_id).user_id)
        for command_id, kind in (
            ("message-1", TaskCommandKind.MESSAGE),
            ("pause-1", TaskCommandKind.PAUSE),
        ):
            stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=user_id,
                command_id=command_id,
                kind=kind,
                payload={"type": kind.value},
            )
        db.commit()

    claimed: list[str] = []
    with factory() as db:
        first = claim_task_command(db, runner_id="worker")
        assert first is not None
        claimed.append(first.command_id)
        # Per-task order: nothing else is claimable while it is unfinished.
        assert claim_task_command(db, runner_id="worker") is None
    assert finish_task_command(first.id, "worker", result={})
    with factory() as db:
        second = claim_task_command(db, runner_id="worker")
        assert second is not None
        claimed.append(second.command_id)

    assert claimed == [auto_resume_command_id(PV, 1, RUN), "message-1"]


def test_transport_refuses_the_reserved_prefix_to_every_caller_but_the_sweeper(
    factory,
):
    from xagent.web.services.task_command_transport import (
        TaskCommandKind,
        enqueue_task_command,
        stage_task_command,
    )

    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        user_id = int(db.get(Task, task_id).user_id)
        db.commit()
    for kind in (TaskCommandKind.RESUME, TaskCommandKind.MESSAGE):
        with factory() as db:
            with pytest.raises(ValueError, match="reserved"):
                stage_task_command(
                    db,
                    task_id=task_id,
                    actor_user_id=user_id,
                    command_id=" auto-resume:5:1:run",
                    kind=kind,
                    payload={},
                )
            with pytest.raises(ValueError, match="reserved"):
                enqueue_task_command(
                    db,
                    task_id=task_id,
                    actor_user_id=user_id,
                    command_id="auto-resume:5:1:run",
                    kind=kind,
                    payload={},
                )
    assert _commands(factory, task_id) == []
    with factory() as db:
        staged = stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=user_id,
            command_id="auto-resume:5:1:run",
            kind=TaskCommandKind.RESUME,
            payload={},
            reserved=True,
        )
        db.commit()
    assert staged.created


# --------------------------------------------------------------------------
# Tick robustness
# --------------------------------------------------------------------------


def test_nothing_due_costs_one_exists_and_no_counts(factory, notified, monkeypatch):
    now = utc_now()
    with factory() as db:
        _seed(db, now, due_in=60)
        _seed(db, now, state="dispatched", due_in=60, run_id="run-b")
        _seed(db, now, state="manual", due_in=None, run_id="run-c")
        db.commit()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("the inert tick must stop at the EXISTS")

    monkeypatch.setattr(task_auto_resume, "count_inflight", unexpected)
    monkeypatch.setattr(task_auto_resume, "_for_each_candidate", unexpected)

    report = run_auto_resume_tick(now=now)

    assert (report.dispatched, report.inflight) == (0, None)


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_due_dispatched_rows_alone_make_the_tick_run(
    factory, notified, monkeypatch, enabled
):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", enabled)
    now = utc_now()
    with factory() as db:
        task_id = _dispatched(db, now, command_status="pending")
        db.commit()

    assert run_auto_resume_tick(now=now).rearmed == 1
    assert _aware(_row(factory, task_id).next_attempt_at) > now


def test_dispatch_cas_requires_the_row_to_still_be_scheduled(
    factory, notified, monkeypatch
):
    # Another sweeper closed the row (here: drained it with the switch off)
    # without touching the task or the counters.
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()

    def drained(db: Session) -> None:
        db.get(TaskAutoRecovery, task_id).state = "manual"

    result = _process_with_stale_view(factory, monkeypatch, task_id, now, drained)

    assert result.outcome is task_auto_resume._Outcome.RACED
    assert _commands(factory, task_id) == []
    assert _row(factory, task_id).state == "manual"


def test_dispatch_cas_pins_the_attempt_it_read(factory, notified, monkeypatch):
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()

    def dispatched_meanwhile(db: Session) -> None:
        db.get(TaskAutoRecovery, task_id).total_resumes = 1

    result = _process_with_stale_view(
        factory, monkeypatch, task_id, now, dispatched_meanwhile
    )

    assert result.outcome is task_auto_resume._Outcome.RACED
    assert _commands(factory, task_id) == []
    assert _row(factory, task_id).total_resumes == 1


def test_naive_now_is_read_as_utc(factory, notified):
    now = utc_now()
    with factory() as db:
        due = _seed(db, now, due_in=-1)
        later = _seed(db, now, due_in=30, run_id="run-later")
        db.commit()

    report = run_auto_resume_tick(now=now.replace(tzinfo=None))

    assert report.dispatched == 1
    assert _row(factory, due).state == "dispatched"
    assert _row(factory, later).state == "scheduled"
    assert _aware(_row(factory, due).next_attempt_at) == now + timedelta(
        seconds=DISPATCH_CHECK_GRACE_SECONDS
    )


def test_pool_timeout_aborts_the_tick_and_is_not_a_candidate_failure(
    factory, notified, monkeypatch
):
    from sqlalchemy.exc import TimeoutError as PoolTimeout

    now = utc_now()
    with factory() as db:
        first = _seed(db, now, due_in=-10)
        second = _seed(db, now, due_in=-5)
        db.commit()
    failures: list[str] = []
    real_counter = task_auto_resume.increment_counter
    monkeypatch.setattr(
        task_auto_resume,
        "increment_counter",
        lambda name, **kw: (failures.append(name), real_counter(name, **kw)),
    )

    def exhausted(db, task):
        raise PoolTimeout("QueuePool limit reached")

    monkeypatch.setattr(task_auto_resume, "auto_recovery_eligibility", exhausted)

    with pytest.raises(PoolTimeout):
        run_auto_resume_tick(now=now)

    assert "xagent.task.auto_resume.candidate_failed" not in failures
    assert _row(factory, first).state == "scheduled"
    assert _row(factory, second).state == "scheduled"


def test_integrity_error_while_staging_is_a_race_and_writes_nothing(
    factory, notified, monkeypatch
):
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()

    def conflict(*_args, **_kwargs):
        raise sa.exc.IntegrityError("INSERT", {}, Exception("duplicate"))

    monkeypatch.setattr(task_auto_resume, "stage_task_command", conflict)

    report = run_auto_resume_tick(now=now)

    assert (report.dispatched, report.raced, report.failed) == (0, 1, 0)
    row = _row(factory, task_id)
    assert (row.state, row.total_resumes) == ("scheduled", 0)
    assert _events(factory, task_id) == []
    assert notified == []


def test_notify_fires_even_when_the_tick_fails_after_a_dispatch(
    factory, notified, monkeypatch
):
    from sqlalchemy.exc import TimeoutError as PoolTimeout

    now = utc_now()
    with factory() as db:
        first = _seed(db, now, due_in=-10)
        _seed(db, now, due_in=-5, run_id="run-second")
        db.commit()
    real = task_auto_resume._process_due_candidate_no_commit
    calls = 0

    def second_times_out(db, task_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PoolTimeout("QueuePool limit reached")
        return real(db, task_id, **kwargs)

    monkeypatch.setattr(
        task_auto_resume, "_process_due_candidate_no_commit", second_times_out
    )

    with pytest.raises(PoolTimeout):
        run_auto_resume_tick(now=now)

    assert _row(factory, first).state == "dispatched"
    assert notified == [1]


def test_scan_and_housekeeping_limits_bound_each_tick(factory, notified, monkeypatch):
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_PER_TICK", "1")
    now = utc_now()
    with factory() as db:
        # Housekeeping takes at most MAX_PER_TICK rows.
        queued = [
            _dispatched(db, now, command_status="pending", due_in=-10 + i)
            for i in range(2)
        ]
        # The scan takes at most 4 x MAX_PER_TICK rows, give-ups included.
        expired = [
            _seed(
                db,
                now,
                due_in=-20 + i,
                interrupted_ago=86_400 + 60,
                run_id=f"run-{i}",
            )
            for i in range(5)
        ]
        db.commit()

    report = run_auto_resume_tick(now=now)

    assert report.rearmed == 1
    assert [_aware(_row(factory, t).next_attempt_at) > now for t in queued] == [
        True,
        False,
    ]
    assert report.gave_up == 4
    assert [_row(factory, t).state for t in expired] == ["expired"] * 4 + ["scheduled"]


def test_failure_while_postponing_after_a_full_queue_fails_the_tick(
    factory, notified, monkeypatch
):
    def full(_db, _command):
        raise AdmissionQueueFull("Execution queue is full")

    def broken(*_args, **_kwargs):
        raise RuntimeError("postpone failed")

    monkeypatch.setattr(
        "xagent.web.services.task_execution_admission.stage_task_admission", full
    )
    monkeypatch.setattr(task_auto_resume, "_postpone_after_admission_full", broken)
    now = utc_now()
    with factory() as db:
        task_id = _seed(db, now)
        db.commit()

    with pytest.raises(RuntimeError, match="postpone failed"):
        run_auto_resume_tick(now=now)

    # Nothing committed: the row is still due and the next tick retries it.
    row = _row(factory, task_id)
    assert (row.state, row.total_resumes) == ("scheduled", 0)
    assert _aware(row.next_attempt_at) == now - timedelta(seconds=1)
    assert notified == []


def test_dispatch_delay_is_observed(factory, notified, monkeypatch):
    observed: list[tuple[str, float, str]] = []
    monkeypatch.setattr(
        task_auto_resume,
        "observe_value",
        lambda name, value, *, unit, attributes=None: observed.append(
            (name, value, unit)
        ),
    )
    now = utc_now()
    with factory() as db:
        _seed(db, now, due_in=-12)
        db.commit()

    run_auto_resume_tick(now=now)

    assert observed == [("xagent.task.auto_resume.dispatch_delay_seconds", 12.0, "s")]
