"""Retention anchor and eligibility predicate (#2562).

Runs against both SQLite and a disposable PostgreSQL, via the shared ``engine``
fixture, because three of the properties under test are dialect-sensitive:
``DateTime(timezone=True)`` round-trips tz-naive on SQLite and aware on
PostgreSQL, ``SELECT ... FOR UPDATE`` is a no-op on one and a real row lock on
the other, and the boolean leg expressions in the select list compile
differently on each.

How the acceptance criterion "the anchor is not moved by token updates,
heartbeats, or checkpoint-pointer maintenance" is covered here: the token
tracker and heartbeat renewal are driven behaviourally through their real
writers, and
``test_last_activity_at_is_written_only_by_the_retention_module`` covers
checkpoint-pointer maintenance -- and every other current and future writer of
the ``tasks`` row -- by asserting no other module writes the column at all.
That static leg is the load-bearing one; the two behavioural tests exist
because the token tracker is the writer that fires every 15 seconds, so a
regression there is the one that would actually reach production.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.core.model.chat.token_context import TokenUsage
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import (
    persist_assistant_message,
    persist_user_message,
    persist_user_message_no_commit,
)
from xagent.web.services.task_coordinator_service import (
    acquire_task_lease_no_commit,
    renew_task_lease_no_commit,
)
from xagent.web.services.task_retention import (
    RETENTION_LIVE_COMMAND_STATUSES,
    RETENTION_TERMINAL_STATUSES,
    RetentionDisposition,
    assess_task_retention,
    count_quiescent_tasks,
    count_retention_candidates,
    is_retention_eligible,
    retention_candidate_condition,
    retention_cutoff,
    retention_lease_clear_condition,
    retention_quiescent_condition,
    touch_task_last_activity,
)
from xagent.web.tracking.task_tracker import _commit_task_usage_if_owned

engine = engine_fixture

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


@pytest.fixture
def sessions(engine):
    """Sessions shaped like production's.

    ``autoflush=False`` matches models/database.py, which is what the
    application actually hands these functions. The sessionmaker default is
    the opposite, and testing under it would exercise a configuration no
    deployment runs.
    """
    Base.metadata.create_all(engine)
    return sa.orm.sessionmaker(bind=engine, autoflush=False)


def _make_user(db: Session, username: str = "owner") -> int:
    user = User(username=username, password_hash="unused")
    db.add(user)
    db.flush()
    return int(user.id)


def _make_task(
    db: Session,
    *,
    user_id: int,
    status: TaskStatus = TaskStatus.COMPLETED,
    anchor: datetime | None = None,
    created_at: datetime | None = None,
    lease_expires_at: datetime | None = None,
) -> int:
    task = Task(
        user_id=user_id,
        title="task",
        status=status,
        last_activity_at=anchor,
        lease_expires_at=lease_expires_at,
    )
    if created_at is not None:
        task.created_at = created_at
    db.add(task)
    db.flush()
    return int(task.id)


def _assess(db: Session, task_id: int, **kwargs):
    params = {"now": NOW, "conversation_days": 365, "trace_days": 90}
    params.update(kwargs)
    return assess_task_retention(db, task_id, **params)


# --------------------------------------------------------------------------
# The anchor
# --------------------------------------------------------------------------


def test_anchor_advances_and_never_regresses(sessions):
    """A later message moves the anchor; an out-of-order earlier one does not.

    The regression direction is the dangerous one: it shortens retention, so a
    late-arriving write for an earlier message would delete data early.
    """
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id)
        db.commit()

        touch_task_last_activity(db, task_id, when=NOW - timedelta(days=10))
        db.commit()
        first = _as_utc(db.get(Task, task_id).last_activity_at)
        assert first == NOW - timedelta(days=10)

        touch_task_last_activity(db, task_id, when=NOW - timedelta(days=1))
        db.commit()
        db.expire_all()
        assert _as_utc(db.get(Task, task_id).last_activity_at) == NOW - timedelta(
            days=1
        )

        touch_task_last_activity(db, task_id, when=NOW - timedelta(days=30))
        db.commit()
        db.expire_all()
        assert _as_utc(db.get(Task, task_id).last_activity_at) == NOW - timedelta(
            days=1
        )


def test_anchor_write_does_not_disturb_updated_at(sessions):
    """The anchor write pins ``updated_at`` instead of letting onupdate fire.

    An anchor write is conversational bookkeeping. Anything reading
    ``updated_at`` as "last execution activity" must not see it move.
    """
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id)
        db.commit()
        before = _as_utc(db.get(Task, task_id).updated_at)

        touch_task_last_activity(db, task_id, when=NOW)
        db.commit()
        db.expire_all()
        task = db.get(Task, task_id)
        assert _as_utc(task.updated_at) == before
        assert _as_utc(task.last_activity_at) == NOW


def test_token_tracker_write_moves_updated_at_but_not_the_anchor(sessions):
    """The writer that fires every 15 seconds must not postpone expiry.

    This is the mechanism that makes ``updated_at`` unusable as an anchor, and
    the one regression here that would reach production quickly.
    """
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=400))
        db.commit()
        anchor_before = _as_utc(db.get(Task, task_id).last_activity_at)
        updated_before = _as_utc(db.get(Task, task_id).updated_at)

        assert _commit_task_usage_if_owned(
            db, task_id, TokenUsage(input_tokens=5, output_tokens=7, llm_calls=1)
        )
        db.expire_all()
        task = db.get(Task, task_id)
        assert task.input_tokens == 5
        assert _as_utc(task.last_activity_at) == anchor_before
        assert _as_utc(task.updated_at) >= updated_before


def test_lease_heartbeat_renewal_does_not_move_the_anchor(sessions):
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(
            db,
            user_id=user_id,
            status=TaskStatus.RUNNING,
            anchor=NOW - timedelta(days=5),
        )
        db.commit()
        lease = acquire_task_lease_no_commit(db, task_id, runner_id="worker")
        db.commit()
        assert lease is not None
        anchor_before = _as_utc(db.get(Task, task_id).last_activity_at)

        assert renew_task_lease_no_commit(db, lease)
        db.commit()
        db.expire_all()
        assert _as_utc(db.get(Task, task_id).last_activity_at) == anchor_before


@pytest.mark.parametrize(
    "persist",
    [
        pytest.param("persist_user_message", id="user-committing"),
        pytest.param("persist_assistant_message", id="assistant-committing"),
        pytest.param("persist_user_message_no_commit", id="user-staging"),
    ],
)
def test_message_persist_paths_anchor_the_task(sessions, persist):
    """Every transcript write is a conversational activity and must anchor.

    The staging variant is checked in the same transaction the caller would
    commit, which is where the hook has to land: an anchor written outside the
    caller's transaction would survive a rolled-back message.
    """
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id)
        db.commit()
        assert db.get(Task, task_id).last_activity_at is None

        if persist == "persist_user_message":
            persist_user_message(db, task_id, user_id, "hello")
        elif persist == "persist_assistant_message":
            persist_assistant_message(db, task_id, user_id, "hi there")
        else:
            assert (
                persist_user_message_no_commit(db, task_id, user_id, "hello")
                is not None
            )
            db.commit()

        db.expire_all()
        assert db.get(Task, task_id).last_activity_at is not None
        assert db.query(TaskChatMessage).filter_by(task_id=task_id).count() == 1


def test_staged_message_and_anchor_roll_back_together(sessions):
    """The hook rides the caller's transaction, so a rollback undoes both."""
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id)
        db.commit()

        persist_user_message_no_commit(db, task_id, user_id, "hello")
        db.rollback()
        db.expire_all()
        assert db.get(Task, task_id).last_activity_at is None
        assert db.query(TaskChatMessage).filter_by(task_id=task_id).count() == 0


#: Ways a module can name ``tasks.last_activity_at`` in an expression or in
#: raw SQL. Deliberately qualified: ``workforce_runs`` has carried a column of
#: the same name since revision 20260802, so an unqualified scan would match
#: that unrelated table and its reaper.
TASK_ANCHOR_SPELLINGS = ("Task.last_activity_at", "tasks SET last_activity_at")


def test_last_activity_at_is_written_only_by_the_retention_module():
    """No module outside the retention module may name the task anchor.

    This is what keeps the column meaning "last conversational activity"
    rather than "last write of any kind", and it covers checkpoint-pointer
    maintenance along with every other ``tasks`` writer -- present and
    future -- without needing one test per writer.

    Limits, stated so this is not mistaken for a boundary: it catches the
    SQLAlchemy-expression and raw-SQL spellings, which is how every current
    writer of this row names a column. A dynamic write
    (``setattr(task, "last_activity_at", ...)``) would evade it. Adding a
    legitimate writer means editing ``allowed`` on purpose, which is the
    point.
    """
    source_root = pathlib.Path(__file__).resolve().parents[3] / "src" / "xagent"
    allowed = {
        # Declares the column.
        source_root / "web" / "models" / "task.py",
        # Owns the only write.
        source_root / "web" / "services" / "task_retention.py",
        # Creates and backfills it.
        source_root / "migrations" / "versions" / "20260922_task_last_activity_at.py",
    }
    found = {
        path
        for path in source_root.rglob("*.py")
        if any(
            spelling in path.read_text(encoding="utf-8")
            for spelling in TASK_ANCHOR_SPELLINGS
        )
    }
    unexpected = sorted(str(p.relative_to(source_root)) for p in found - allowed)
    assert not unexpected, f"unexpected writers of tasks.last_activity_at: {unexpected}"


def test_null_anchor_falls_back_to_created_at(sessions):
    """A task that never carried a message still ages.

    Comparing a NULL anchor against a cutoff yields NULL, which every WHERE
    clause reads as "not eligible" -- so without the coalesce these rows would
    be silently immortal.
    """
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(
            db, user_id=user_id, anchor=None, created_at=NOW - timedelta(days=400)
        )
        db.commit()
        assessment = _assess(db, task_id)
        assert assessment.disposition is RetentionDisposition.CONVERSATION_EXPIRED
        assert assessment.anchor == NOW - timedelta(days=400)


def _insert_unanchored_message(
    db: Session, *, task_id: int, user_id: int, created_at: datetime
) -> None:
    """Persist a transcript row the way a pre-#2571 binary does.

    Deliberately not through ``chat_history_service``: those paths call
    ``touch_task_last_activity``, and the writer this models is one that does
    not know the column exists (#2580).
    """
    db.add(
        TaskChatMessage(
            task_id=task_id,
            user_id=user_id,
            role="user",
            content="written by a binary that never touches the anchor",
            message_type="text",
            created_at=created_at,
        )
    )
    db.flush()


@pytest.mark.parametrize(
    ("stored_days", "message_days", "expected_days", "expected"),
    [
        pytest.param(
            400, 10, 10, RetentionDisposition.NOT_ELIGIBLE, id="stale-anchor-retained"
        ),
        pytest.param(
            400, 200, 200, RetentionDisposition.TRACE_EXPIRED, id="stale-downgraded"
        ),
        pytest.param(
            None, 10, 10, RetentionDisposition.NOT_ELIGIBLE, id="null-anchor-retained"
        ),
        pytest.param(
            10, 400, 10, RetentionDisposition.NOT_ELIGIBLE, id="older-message-ignored"
        ),
    ],
)
def test_assessment_anchor_follows_a_message_the_stored_anchor_missed(
    sessions, stored_days, message_days, expected_days, expected
):
    """The locked assessment measures from the newest message, not the column.

    A writer that never touches ``last_activity_at`` -- an old binary during a
    rolling deploy, or after a rollback past #2571 -- leaves the stored anchor
    behind the transcript. Reading only the column would make that
    conversation look older than it is and expire it early (#2580). The
    ``older-message-ignored`` case pins the other direction: a message older
    than the stored anchor must not pull it back.
    """
    with sessions() as db:
        user_id = _make_user(db)
        stored = None if stored_days is None else NOW - timedelta(days=stored_days)
        task_id = _make_task(
            db,
            user_id=user_id,
            anchor=stored,
            created_at=NOW - timedelta(days=500),
        )
        _insert_unanchored_message(
            db,
            task_id=task_id,
            user_id=user_id,
            created_at=NOW - timedelta(days=message_days),
        )
        db.commit()

        assessment = _assess(db, task_id)
        assert assessment.disposition is expected
        assert assessment.anchor == NOW - timedelta(days=expected_days)
        db.expire_all()
        # The fix reads the transcript; it does not repair the column.
        assert _as_utc(db.get(Task, task_id).last_activity_at) == stored


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------


def test_terminal_statuses_are_exactly_completed_and_failed():
    """PENDING is not terminal: it holds user input that has never run."""
    assert set(RETENTION_TERMINAL_STATUSES) == {TaskStatus.COMPLETED, TaskStatus.FAILED}
    assert TaskStatus.PENDING not in RETENTION_TERMINAL_STATUSES


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
    ],
)
def test_non_terminal_statuses_are_refused(sessions, status):
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(
            db, user_id=user_id, status=status, anchor=NOW - timedelta(days=4000)
        )
        db.commit()
        assessment = _assess(db, task_id)
        assert assessment.disposition is RetentionDisposition.NOT_ELIGIBLE
        assert assessment.quiescent is False
        assert "status" in assessment.blockers
        assert count_retention_candidates(db, now=NOW, days=1) == 0


def test_live_lease_is_refused_and_expired_lease_is_not(sessions):
    """Expiry means no worker is executing; it is not permission to restart."""
    with sessions() as db:
        user_id = _make_user(db)
        live = _make_task(
            db,
            user_id=user_id,
            anchor=NOW - timedelta(days=400),
            lease_expires_at=NOW + timedelta(minutes=5),
        )
        stale = _make_task(
            db,
            user_id=user_id,
            anchor=NOW - timedelta(days=400),
            lease_expires_at=NOW - timedelta(minutes=5),
        )
        db.commit()

        live_assessment = _assess(db, live)
        assert live_assessment.disposition is RetentionDisposition.NOT_ELIGIBLE
        assert live_assessment.blockers == ("lease",)
        assert (
            _assess(db, stale).disposition is RetentionDisposition.CONVERSATION_EXPIRED
        )


@pytest.mark.parametrize("command_status", list(RETENTION_LIVE_COMMAND_STATUSES))
def test_pending_or_processing_command_is_refused(sessions, command_status):
    """A command insert leaves status and ``updated_at`` untouched.

    So neither column can stand in for this leg -- an accepted-but-unexecuted
    turn is invisible to both.
    """
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=400))
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                command_id=f"cmd-{command_status}",
                kind="start",
                payload={},
                status=command_status,
            )
        )
        db.commit()
        assessment = _assess(db, task_id)
        assert assessment.disposition is RetentionDisposition.NOT_ELIGIBLE
        assert assessment.blockers == ("commands",)
        assert count_retention_candidates(db, now=NOW, days=1) == 0
        assert count_quiescent_tasks(db, now=NOW) == 0


@pytest.mark.parametrize("command_status", ["completed", "failed"])
def test_settled_command_does_not_block(sessions, command_status):
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=400))
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                command_id=f"cmd-{command_status}",
                kind="start",
                payload={},
                status=command_status,
            )
        )
        db.commit()
        assert _assess(db, task_id).disposition is (
            RetentionDisposition.CONVERSATION_EXPIRED
        )


def test_three_dispositions_by_anchor_age(sessions):
    with sessions() as db:
        user_id = _make_user(db)
        fresh = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=10))
        trace_only = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=100))
        whole = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=400))
        db.commit()
        assert _assess(db, fresh).disposition is RetentionDisposition.NOT_ELIGIBLE
        assert _assess(db, trace_only).disposition is RetentionDisposition.TRACE_EXPIRED
        assert _assess(db, whole).disposition is (
            RetentionDisposition.CONVERSATION_EXPIRED
        )


def test_conversation_expiry_subsumes_traces_when_periods_are_inverted(sessions):
    """A longer trace period than conversation period still deletes the task.

    The purge path selected by CONVERSATION_EXPIRED removes traces with the
    task, so reporting it is accurate rather than a period being ignored.
    """
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=200))
        db.commit()
        assessment = _assess(db, task_id, conversation_days=90, trace_days=365)
        assert assessment.disposition is RetentionDisposition.CONVERSATION_EXPIRED


def test_unlimited_retention_expires_nothing(sessions):
    """``None`` days is the "unlimited" policy option, not a cutoff long past."""
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=40000))
        db.commit()
        assessment = _assess(db, task_id, conversation_days=None, trace_days=None)
        assert assessment.disposition is RetentionDisposition.NOT_ELIGIBLE
        assert assessment.quiescent is True
        assert retention_cutoff(now=NOW, days=None) is None
        assert count_retention_candidates(db, now=NOW, days=None) == 0


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: retention_cutoff(now=datetime(2026, 9, 22, 12), days=90),
            id="cutoff",
        ),
        pytest.param(
            lambda: retention_lease_clear_condition(now=datetime(2026, 9, 22, 12)),
            id="lease-condition",
        ),
        pytest.param(
            lambda: retention_quiescent_condition(now=datetime(2026, 9, 22, 12)),
            id="quiescent-condition",
        ),
        pytest.param(
            lambda: retention_candidate_condition(
                now=datetime(2026, 9, 22, 12), days=90
            ),
            id="candidate-condition",
        ),
    ],
)
def test_naive_now_is_refused_at_the_entry(call):
    """A naive ``now`` is differently wrong on each path, so it is refused.

    PostgreSQL would resolve it against the session time zone and answer
    silently, while the Python comparison would raise. A retention cutoff
    silently shifted by the server's UTC offset deletes the wrong rows.
    """
    with pytest.raises(ValueError, match="timezone-aware"):
        call()


def test_naive_now_is_refused_by_the_assessment(sessions):
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id)
        db.commit()
        with pytest.raises(ValueError, match="timezone-aware"):
            assess_task_retention(
                db,
                task_id,
                now=datetime(2026, 9, 22, 12),
                conversation_days=90,
                trace_days=90,
            )


@pytest.mark.parametrize("offset_hours", [0, 8, -5])
def test_paths_agree_across_time_zones_for_one_instant(sessions, offset_hours):
    """The same instant must decide the same way however the caller spells it.

    Rejecting naive values was not enough: an aware ``+08:00`` reached the SQL
    leg with its offset dropped by SQLite's DATETIME bind, so the scanning
    path expired a task the locked path did not. #2563 picking a batch with
    ``datetime.now().astimezone()`` would have deleted up to one UTC offset
    short of the retention period.
    """
    instant = datetime(2026, 9, 22, 4, 0, 0, tzinfo=timezone.utc)
    spelled = instant.astimezone(timezone(timedelta(hours=offset_hours)))
    assert spelled == instant
    # Four hours newer than the 90-day cutoff: inside the window, and inside
    # it by less than any of the offsets under test.
    anchor = instant - timedelta(days=90) + timedelta(hours=4)
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, anchor=anchor)
        db.commit()

        per_task = is_retention_eligible(
            db, task_id, now=spelled, conversation_days=90, trace_days=90
        )
        scanned = (
            db.execute(
                sa.select(Task.id).where(
                    retention_candidate_condition(now=spelled, days=90)
                )
            )
            .scalars()
            .all()
        )
        assert per_task is False
        assert list(scanned) == []
        assert per_task is (task_id in scanned)


def test_anchor_write_normalizes_a_non_utc_when(sessions):
    """The write side shares the contract, since it feeds what the reads use."""
    instant = datetime(2026, 9, 22, 4, 0, 0, tzinfo=timezone.utc)
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id)
        db.commit()
        touch_task_last_activity(
            db, task_id, when=instant.astimezone(timezone(timedelta(hours=8)))
        )
        db.commit()
        db.expire_all()
        assert _as_utc(db.get(Task, task_id).last_activity_at) == instant


def test_naive_when_is_refused_by_the_anchor_write(sessions):
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id)
        db.commit()
        with pytest.raises(ValueError, match="timezone-aware"):
            touch_task_last_activity(db, task_id, when=datetime(2026, 9, 22, 12))


def test_missing_task_is_not_eligible(sessions):
    with sessions() as db:
        assert _assess(db, 999_999).blockers == ("missing",)
        assert (
            is_retention_eligible(
                db, 999_999, now=NOW, conversation_days=1, trace_days=1
            )
            is False
        )


@pytest.mark.parametrize("offset_seconds", [-1, 0, 1])
def test_per_task_and_set_paths_agree_at_the_expiry_boundary(sessions, offset_seconds):
    """The boundary is the only place the locked and scanning paths could drift.

    One evaluates the cutoff comparison in Python against an anchor read back
    from the database, the other evaluates it in SQL. They share
    ``retention_cutoff`` but not the comparison itself, so the inclusive edge
    is pinned here rather than left to review.
    """
    cutoff = retention_cutoff(now=NOW, days=90)
    assert cutoff is not None
    anchor = cutoff + timedelta(seconds=offset_seconds)
    with sessions() as db:
        user_id = _make_user(db)
        task_id = _make_task(db, user_id=user_id, anchor=anchor)
        db.commit()

        per_task = is_retention_eligible(
            db, task_id, now=NOW, conversation_days=90, trace_days=90
        )
        scanned = (
            db.execute(
                sa.select(Task.id).where(
                    retention_candidate_condition(now=NOW, days=90)
                )
            )
            .scalars()
            .all()
        )
        assert per_task is (task_id in scanned)
        assert per_task is (offset_seconds <= 0)


def test_candidate_condition_composes_with_a_callers_own_query(sessions):
    """The condition is the unit #2563 inherits, so it must compose cleanly.

    A batching caller adds its own ordering and limit around it rather than
    receiving a prebuilt statement; this pins that it can.
    """
    with sessions() as db:
        user_id = _make_user(db)
        ids = [
            _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=400))
            for _ in range(5)
        ]
        _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=1))
        db.commit()

        batched = (
            db.execute(
                sa.select(Task.id)
                .where(retention_candidate_condition(now=NOW, days=365))
                .order_by(Task.id)
                .limit(2)
            )
            .scalars()
            .all()
        )
        assert list(batched) == sorted(ids)[:2]
        every = (
            db.execute(
                sa.select(Task.id)
                .where(retention_candidate_condition(now=NOW, days=365))
                .order_by(Task.id)
            )
            .scalars()
            .all()
        )
        assert list(every) == sorted(ids)


def test_counts_exclude_live_tasks(sessions):
    with sessions() as db:
        user_id = _make_user(db)
        _make_task(db, user_id=user_id, anchor=NOW - timedelta(days=400))
        _make_task(
            db,
            user_id=user_id,
            status=TaskStatus.RUNNING,
            anchor=NOW - timedelta(days=400),
        )
        db.commit()
        assert count_quiescent_tasks(db, now=NOW) == 1
        assert count_retention_candidates(db, now=NOW, days=365) == 1
