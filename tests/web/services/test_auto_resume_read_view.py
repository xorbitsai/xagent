"""``auto_resume`` in task reads: the notice for the recovery row as it stands.

``current_auto_recovery_view`` reports, beside the interruption, the fence
(``run_id``, ``state_version``) and the ``auto_resume`` payload the live
``task_recovery_update`` notice carries for the row's current state. These
tests pin the mapping, the fence, that a read after each committed
transition agrees with the notice that transition published, that the
within-fence order clients derive from ``(status, attempt)`` strictly
increases, and the known cases where a read is ``None`` but a notice was
published.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models import database
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import TaskAutoRecovery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_auto_resume
from xagent.web.services.task_auto_recovery import (
    current_auto_recovery_view,
    record_interruption_no_commit,
)
from xagent.web.services.task_auto_resume import (
    RecoveryNotice,
    check_auto_resume_claim_sync,
    current_auto_resume_view,
    public_auto_resume_view,
    recorded_notice,
    run_auto_resume_tick,
)
from xagent.web.services.task_lease_service import utc_now
from xagent.web.services.task_settlement_report import RecordedInterruption

engine = engine_fixture

RUN = "run-read-view"
PV = 9
DUE = datetime(2026, 10, 10, 8, 0, 12, tzinfo=timezone.utc)

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


@pytest.fixture
def factory(engine, monkeypatch) -> Iterator[sessionmaker]:
    Base.metadata.create_all(engine)
    result = sessionmaker(engine, autoflush=False)
    # The globals behind the accessors, as test_task_auto_resume does.
    monkeypatch.setattr(database, "_SessionLocal", result)
    monkeypatch.setattr(database, "_engine", engine)
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        task_auto_resume, "notify_task_command_dispatcher", lambda: None
    )
    yield result


def _task(db: Session, **fields: Any) -> int:
    user = User(username=f"read-view-{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    db.flush()
    values: dict[str, Any] = {
        "status": TaskStatus.PAUSED,
        "control_state": "paused",
        "run_id": RUN,
        "state_version": PV,
    }
    values.update(fields)
    task = Task(
        user_id=user.id,
        title="Read view",
        description="read view test",
        execution_mode="balanced",
        **values,
    )
    db.add(task)
    db.flush()
    return int(task.id)


def _seed(db: Session, row: dict[str, Any], **task_fields: Any) -> int:
    task_id = _task(db, **task_fields)
    at = datetime(2026, 10, 10, 8, 0, 0, tzinfo=timezone.utc)
    db.add(
        TaskAutoRecovery(
            task_id=task_id,
            **{
                "run_id": RUN,
                "reason": "lease_expired",
                "state": "scheduled",
                "paused_state_version": PV,
                "interrupted_at": at,
                "episode_started_at": at,
                **row,
            },
        )
    )
    db.flush()
    return task_id


def _read(factory: sessionmaker, task_id: int) -> dict[str, Any] | None:
    """A request's read: a fresh session, as the routes use."""
    with factory() as db:
        return current_auto_recovery_view(db, db.get(Task, task_id))


def _k(auto_resume: dict[str, Any]) -> tuple[int, int]:
    """The within-fence order ``current_auto_resume_view`` documents."""
    attempt = int(auto_resume["attempt"])
    return {
        "resuming": (attempt, 0),
        "scheduled": (attempt - 1, 1),
        "stopped": (attempt, 2),
    }[auto_resume["status"]]


# --------------------------------------------------------------------------
# S1: row state -> read
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {"state": "scheduled", "total_resumes": 2, "next_attempt_at": DUE},
            ("scheduled", None, 3, "2026-10-10T08:00:12+00:00"),
        ),
        # The row's time is when housekeeping looks again; not for clients.
        (
            {"state": "dispatched", "total_resumes": 3, "next_attempt_at": DUE},
            ("resuming", None, 3, None),
        ),
        (
            {"state": "exhausted", "total_resumes": 4},
            ("stopped", "limit_reached", 4, None),
        ),
        ({"state": "expired", "total_resumes": 1}, ("stopped", "expired", 1, None)),
        ({"state": "superseded"}, ("stopped", "superseded", 0, None)),
        (
            {"state": "dispatch_failed", "total_resumes": 2},
            ("stopped", "failed", 2, None),
        ),
        # Decision U1: a recorded switched-off ``manual`` was never announced,
        # and the drain's identical row cannot be told from it.
        ({"state": "manual", "state_detail": "auto_resume_disabled"}, None),
        ({"state": "manual"}, None),
        ({"state": "manual", "state_detail": "supersede_check_failed"}, None),
        ({"state": "manual", "state_detail": "channel_hold_pending"}, None),
        ({"state": "manual", "state_detail": "interaction_pending"}, None),
        # Only a person's cancel writes it, announced at this very fence.
        (
            {"state": "manual", "state_detail": "user_cancelled", "total_resumes": 2},
            ("stopped", "cancelled", 2, None),
        ),
        ({"state": "ineligible", "state_detail": "ineligible:source_a2a"}, None),
        ({"state": "ineligible", "state_detail": "ineligible:preview"}, None),
        ({"state": "stale", "state_detail": "dispatch:run_id"}, None),
        ({"state": "disabled"}, None),
    ],
)
def test_row_state_maps_to_the_notice_contract(factory, row, expected):
    with factory() as db:
        task_id = _seed(db, row)
        db.commit()

    view = _read(factory, task_id)

    assert view is not None
    assert (view["run_id"], view["state_version"], view["reason"]) == (
        RUN,
        PV,
        "lease_expired",
    )
    assert view["interrupted_at"] == "2026-10-10T08:00:00+00:00"
    auto_resume = view["auto_resume"]
    if expected is None:
        assert auto_resume is None
        return
    assert (
        auto_resume["status"],
        auto_resume["stop_reason"],
        auto_resume["attempt"],
        auto_resume["next_attempt_at"],
    ) == expected
    assert auto_resume["reason"] == "lease_expired"
    # Exactly the notice's keys: no counters, details or command ids.
    assert set(auto_resume) == set(
        public_auto_resume_view(
            "scheduled",
            state_detail=None,
            reason="x",
            attempt=1,
            next_attempt_at=None,
        )
    )
    assert set(view) == {
        "reason",
        "interrupted_at",
        "run_id",
        "state_version",
        "auto_resume",
    }


# --------------------------------------------------------------------------
# S2: the fence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task_fields", "whole_view_hidden"),
    [
        ({"run_id": "run-later"}, True),
        # A resume handing over (``resume_requested``) moved the version.
        ({"state_version": PV + 1, "control_state": "resume_requested"}, True),
        # A rolled-back automatic resume: paused again, two versions on.
        ({"state_version": PV + 2}, True),
        ({"control_state": "resume_requested"}, False),
        ({"status": TaskStatus.RUNNING, "control_state": "running"}, True),
        (
            {
                "status": TaskStatus.WAITING_FOR_USER,
                "control_state": "waiting_for_user",
            },
            True,
        ),
        ({"status": TaskStatus.FAILED, "control_state": "failed"}, True),
    ],
    ids=[
        "run_id",
        "resume_requested",
        "rolled_back",
        "control_state",
        "running",
        "waiting",
        "failed",
    ],
)
def test_auto_resume_is_shown_only_at_the_sweepers_fence(
    factory, task_fields, whole_view_hidden
):
    with factory() as db:
        task_id = _seed(
            db,
            {"state": "dispatched", "total_resumes": 1, "next_attempt_at": DUE},
            **task_fields,
        )
        db.commit()

    view = _read(factory, task_id)

    if whole_view_hidden:
        assert view is None
    else:
        assert view is not None and view["auto_resume"] is None


def test_the_row_alone_decides_without_a_query(factory):
    """Pure: the same answer from detached objects, no session involved."""
    task = Task(
        status=TaskStatus.PAUSED, control_state="paused", run_id=RUN, state_version=PV
    )
    row = TaskAutoRecovery(
        task_id=1,
        run_id=RUN,
        reason="llm_unavailable",
        state="scheduled",
        paused_state_version=PV,
        total_resumes=0,
        next_attempt_at=DUE,
    )
    assert current_auto_resume_view(task, row) == {
        "status": "scheduled",
        "stop_reason": None,
        "reason": "llm_unavailable",
        "attempt": 1,
        "next_attempt_at": "2026-10-10T08:00:12+00:00",
    }
    task.state_version = PV + 1
    assert current_auto_resume_view(task, row) is None


# --------------------------------------------------------------------------
# S3: every committed transition reads as the notice it published
# --------------------------------------------------------------------------


def _record(
    factory: sessionmaker, *, interrupted_ago: timedelta = timedelta(seconds=30)
) -> tuple[int, RecoveryNotice | None, datetime]:
    """Record an interruption of a PAUSED task as settlement does."""
    now = utc_now()
    with factory() as db:
        task_id = _task(db)
        task = db.get(Task, task_id)
        row = record_interruption_no_commit(
            db,
            task=task,
            reason=InterruptionReason.LEASE_EXPIRED,
            task_status=TaskStatus.PAUSED,
            interrupted_at=now - interrupted_ago,
            progress_marker=None,
            gated_by_settlement_switches=False,
            now=now,
        )
        assert row is not None
        recorded = RecordedInterruption.from_row(row)
        db.commit()
    return task_id, recorded_notice(recorded), now


def _due(factory: sessionmaker, task_id: int) -> datetime:
    with factory() as db:
        value = db.get(TaskAutoRecovery, task_id).next_attempt_at
    assert value is not None
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)) + timedelta(
        seconds=1
    )


def _end_command(factory: sessionmaker, task_id: int, status: str) -> None:
    with factory() as db:
        command = db.scalars(
            sa.select(TaskExecutionCommand)
            .where(TaskExecutionCommand.task_id == task_id)
            .order_by(TaskExecutionCommand.id.desc())
        ).first()
        command.status = status
        command.result = (
            {"resume_outcome": "already_in_progress"}
            if status == "completed"
            else {"rejection_reason": "stale_run"}
        )
        db.commit()


def _tick(factory: sessionmaker, task_id: int, now: datetime) -> RecoveryNotice:
    report = run_auto_resume_tick(now=now)
    (notice,) = [n for n in report.notices if n.task_id == task_id]
    return notice


def _assert_reads_as(
    factory: sessionmaker, task_id: int, notice: RecoveryNotice
) -> dict[str, Any]:
    view = _read(factory, task_id)
    assert view is not None
    assert (view["run_id"], view["state_version"]) == (
        notice.run_id,
        notice.state_version,
    )
    assert view["auto_resume"] == notice.auto_resume
    return view["auto_resume"]


def test_each_transition_reads_as_its_notice_in_increasing_order(factory):
    task_id, recorded, now = _record(factory)
    assert recorded is not None
    seen = [_assert_reads_as(factory, task_id, recorded)]

    # Dispatch, the resume does not happen, housekeeping reschedules, the
    # second dispatch's command fails.
    at = _due(factory, task_id)
    seen.append(_assert_reads_as(factory, task_id, _tick(factory, task_id, at)))
    _end_command(factory, task_id, "completed")
    at = _due(factory, task_id)
    seen.append(_assert_reads_as(factory, task_id, _tick(factory, task_id, at)))
    at = _due(factory, task_id)
    seen.append(_assert_reads_as(factory, task_id, _tick(factory, task_id, at)))
    _end_command(factory, task_id, "failed")
    at = _due(factory, task_id)
    seen.append(_assert_reads_as(factory, task_id, _tick(factory, task_id, at)))

    assert [(s["status"], s["attempt"]) for s in seen] == [
        ("scheduled", 1),
        ("resuming", 1),
        ("scheduled", 2),
        ("resuming", 2),
        ("stopped", 2),
    ]
    assert seen[-1]["stop_reason"] == "failed"
    keys = [_k(s) for s in seen]
    assert keys == sorted(set(keys)), keys


def test_a_rolled_back_resume_reads_as_its_claims_notice_at_the_new_fence(
    factory,
):
    """``resume_task`` wrote RESUME_REQUESTED, failed to start and wrote
    PAUSED back: two versions on. Until the command's retry claims it the
    row keeps its old fence and the whole view is hidden; the claim moves
    the row to the new fence and schedules it again, and the read follows."""
    task_id, _recorded, _now = _record(factory)
    resuming = _tick(factory, task_id, _due(factory, task_id))
    _assert_reads_as(factory, task_id, resuming)
    with factory() as db:
        db.get(Task, task_id).state_version = PV + 2
        command = db.scalars(
            sa.select(TaskExecutionCommand).where(
                TaskExecutionCommand.task_id == task_id
            )
        ).one()
        command_id, payload = command.command_id, command.payload["auto_resume"]
        db.commit()
    assert _read(factory, task_id) is None

    decision = check_auto_resume_claim_sync(
        task_id=task_id,
        command_id=command_id,
        auto_resume=payload,
        status=TaskStatus.PAUSED,
        control_state="paused",
        run_id=RUN,
        state_version=PV + 2,
        attempt_count=2,
    )

    assert decision.why == "rolled_back"
    (notice,) = decision.notices
    assert (notice.state_version, notice.auto_resume["status"]) == (
        PV + 2,
        "scheduled",
    )
    _assert_reads_as(factory, task_id, notice)


def test_a_stop_straight_from_scheduled_orders_after_it(factory):
    """A give-up from ``scheduled`` reports the resumes made, not the next."""
    task_id, recorded, now = _record(factory)
    assert recorded is not None
    # Expired by the time it falls due.
    stopped = _tick(factory, task_id, now + timedelta(days=30))

    assert _assert_reads_as(factory, task_id, stopped)["status"] == "stopped"
    assert _k(recorded.auto_resume) < _k(stopped.auto_resume)


def test_a_stop_recorded_at_once_reads_as_its_notice(factory):
    task_id, recorded, _now = _record(factory, interrupted_ago=timedelta(days=30))

    assert recorded is not None
    assert (recorded.auto_resume["status"], recorded.auto_resume["stop_reason"]) == (
        "stopped",
        "expired",
    )
    _assert_reads_as(factory, task_id, recorded)


# --------------------------------------------------------------------------
# S4: known differences (design section 2.3 / 3.3): a read is None where a
# notice was published. Each is a stop the fence never moves past, so a
# client treats the None as terminal, as the docstring says.
# --------------------------------------------------------------------------


def test_switched_off_manual_reads_none_whether_recorded_or_drained(
    factory, monkeypatch
):
    """Decision U1. Recorded with the switch off, nothing is announced; the
    switch-off drain announces ``stopped/disabled``. The two rows are
    identical, so both read as ``None``."""
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    recorded_id, recorded, _now = _record(factory)
    assert recorded is None

    monkeypatch.delenv("XAGENT_TASK_AUTO_RESUME_ENABLED")
    drained_id, scheduled, now = _record(factory)
    assert scheduled is not None
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_ENABLED", "false")
    drained = _tick(factory, drained_id, now)
    assert (drained.auto_resume["status"], drained.auto_resume["stop_reason"]) == (
        "stopped",
        "disabled",
    )

    columns = ("state", "state_detail", "next_attempt_at", "total_resumes")
    with factory() as db:
        rows = [db.get(TaskAutoRecovery, i) for i in (recorded_id, drained_id)]
        assert [tuple(getattr(r, c) for c in columns) for r in rows] == [
            ("manual", "auto_resume_disabled", None, 0)
        ] * 2
    for task_id in (recorded_id, drained_id):
        view = _read(factory, task_id)
        assert view is not None and view["auto_resume"] is None


def test_dispatch_time_ineligible_reads_none(factory):
    task_id, _recorded, now = _record(factory)
    with factory() as db:
        db.get(Task, task_id).source = "a2a"
        db.commit()

    stopped = _tick(factory, task_id, _due(factory, task_id))

    assert (stopped.auto_resume["status"], stopped.auto_resume["stop_reason"]) == (
        "stopped",
        "ineligible",
    )
    view = _read(factory, task_id)
    assert view is not None and view["auto_resume"] is None


def test_failed_resume_past_a_rollback_reads_none(factory):
    """The notice names the rolled-back pause's version; the row keeps its
    old fence, so the whole view is hidden."""
    task_id, _recorded, now = _record(factory)
    _tick(factory, task_id, _due(factory, task_id))
    _end_command(factory, task_id, "failed")
    with factory() as db:
        db.get(Task, task_id).state_version = PV + 2
        db.commit()

    stopped = _tick(factory, task_id, _due(factory, task_id))

    assert (stopped.state_version, stopped.auto_resume["stop_reason"]) == (
        PV + 2,
        "failed",
    )
    assert _read(factory, task_id) is None


# --------------------------------------------------------------------------
# P1: the read takes no row lock
# --------------------------------------------------------------------------


def test_postgresql_read_does_not_wait_for_a_sweeper_holding_the_rows(factory, engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("row locks are PostgreSQL's")
    with factory() as db:
        task_id = _seed(db, {"state": "scheduled", "next_attempt_at": DUE})
        db.commit()

    holder = factory()
    try:
        holder.execute(
            sa.select(Task.id).where(Task.id == task_id).with_for_update()
        ).all()
        holder.execute(
            sa.update(TaskAutoRecovery)
            .where(TaskAutoRecovery.task_id == task_id)
            .values(state="dispatched", total_resumes=1)
        )
        result: list[Any] = []

        def read() -> None:
            with factory() as db:
                db.execute(sa.text("SET LOCAL lock_timeout = '2s'"))
                result.append(current_auto_recovery_view(db, db.get(Task, task_id)))

        reader = threading.Thread(target=read)
        reader.start()
        reader.join(timeout=10)
        assert not reader.is_alive()
    finally:
        holder.rollback()
        holder.close()

    (view,) = result
    assert view is not None, "read waited on a row lock"
    # The committed state, not the holder's uncommitted one.
    assert view["auto_resume"]["status"] == "scheduled"
