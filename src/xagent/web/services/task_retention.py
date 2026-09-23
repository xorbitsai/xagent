"""Retention anchor and the DB-authoritative eligibility predicate (#2562).

Nothing here deletes anything. This module answers one question -- *may this
task's conversation or traces be expired right now?* -- and writes the one
column that question is anchored on. The purge that acts on the answer is
#2563; the retention periods it is asked about are a policy decision still
open in #2567.

Two consumers must never disagree about eligibility: the purge, which
evaluates one task under a row lock before deleting it, and the read-only
preview, which counts eligible tasks across the table without locking
anything. They agree here by construction rather than by review: both are
built from the same three leaf conditions (:func:`retention_terminal_condition`,
:func:`retention_lease_clear_condition`,
:func:`retention_commands_clear_condition`), the same cutoff arithmetic
(:func:`retention_cutoff`), and one normalization of the caller's clock
(:func:`_utc`). That last one is not housekeeping: an aware non-UTC ``now``
used to make the two paths answer differently on SQLite, because its
``DATETIME`` bind keeps the wall clock and drops the offset.
``test_task_retention.py`` pins the agreement at the expiry boundary and
across time zones, which are the two places the paths could plausibly
drift.

What the row lock does and does not fence
-----------------------------------------
:func:`assess_task_retention` takes ``SELECT ... FOR UPDATE`` on the task
row before evaluating. On PostgreSQL that makes the *status* and *lease* legs
authoritative: every writer of those columns goes through a conditional
``UPDATE tasks``, which cannot commit while the lock is held.

**On SQLite it fences nothing.** The dialect has no row locks and SQLAlchemy
compiles the clause away entirely, so the guarantee below is PostgreSQL's
alone. SQLite is this repo's default store, so #2563 cannot treat a
successful assessment as a deletion licence there without supplying its own
serialization.

It also fences command insertion for the producers that take the task row
first, which is why the "no live command" leg is meaningful rather than
advisory. Verified producers that do: ``reserve_task_start_no_commit``
(task_orchestrator.py) and ``task_resume_command.py``, whose
``updated_at = updated_at`` self-writes exist precisely to lock the row and
decide admission by rowcount -- note that they pin the timestamp rather than
advance it; ``stage_task_command`` via task_start_protocol.py and
shared_channel_execution.py, which use ``with_for_update`` explicitly; and
the A2A cancel path, which conditionally updates the task row.

Three producers were *not* verified to lock the task row:
``api/websocket.py``, ``services/task_interaction_service.py``, and
``services/workforce_runtime.py``. A command inserted by one of those
between this assessment and a later delete would not be fenced by this
lock. #2563 must confirm or fence them before it deletes on this predicate's
word; this module deliberately does not claim it already holds.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import cast

from sqlalchemy import and_, exists, false, func, or_, select, update
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ..models.task import Task, TaskStatus, TraceEvent, task_status_predicate
from ..models.task_command import TaskExecutionCommand
from .task_command_transport import COMMAND_PENDING, COMMAND_PROCESSING

#: Statuses from which no further execution can begin without a new,
#: separately-fenced admission. Written as a positive membership test rather
#: than "not one of the live statuses": a status added to ``TaskStatus``
#: later must be classified deliberately, and until it is, it is retained.
#: ``PENDING`` is *not* terminal -- an accepted-but-unstarted task holds user
#: input that has never run.
RETENTION_TERMINAL_STATUSES: tuple[TaskStatus, ...] = (
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
)

#: Command statuses that mean accepted work still owes execution. Imported
#: from the transport rather than restated, so the predicate cannot end up
#: admitting a status the command executor still considers live.
RETENTION_LIVE_COMMAND_STATUSES: tuple[str, ...] = (
    COMMAND_PENDING,
    COMMAND_PROCESSING,
)


class RetentionDisposition(enum.Enum):
    """What, if anything, may be expired for a task.

    ``CONVERSATION_EXPIRED`` subsumes traces: the purge path it selects
    removes the whole task, traces included. It is therefore reported even
    when the trace period alone would not have elapsed, which is the case
    whenever ``trace_days`` is configured longer than ``conversation_days``.
    """

    NOT_ELIGIBLE = "not_eligible"
    TRACE_EXPIRED = "trace_expired"
    CONVERSATION_EXPIRED = "conversation_expired"


@dataclass(frozen=True)
class RetentionAssessment:
    """The predicate's answer, with enough detail to log or explain it."""

    disposition: RetentionDisposition
    #: The anchor actually used, normalized to UTC. ``None`` when the task row
    #: does not exist.
    anchor: datetime | None
    #: ``False`` when the task is live in some way; the ``blockers`` tuple
    #: then says which legs failed.
    quiescent: bool
    #: Leg names that refused this task: any of ``"missing"`` (no such row),
    #: ``"status"``, ``"lease"``, ``"commands"``, or ``"anchor"`` -- the last
    #: meaning the row is quiescent but carries neither ``last_activity_at``
    #: nor ``created_at``, so there is nothing to measure a period against.
    #: Empty when the task is quiescent and has an anchor.
    blockers: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return self.disposition is not RetentionDisposition.NOT_ELIGIBLE


def retention_anchor() -> ColumnElement[datetime]:
    """The SQL expression for "last conversational activity".

    Coalesced to ``created_at`` so a task that never carried a message, or
    whose backfill has not reached it, still ages: comparing a NULL anchor
    against a cutoff yields NULL, which every WHERE clause reads as "not
    eligible" -- silently immortal rows, forever.
    """
    return func.coalesce(Task.last_activity_at, Task.created_at)


def _utc(now: datetime) -> datetime:
    """Refuse a naive ``now`` and convert an aware one to UTC.

    Both halves are load-bearing, and for the same reason: everything
    downstream must see one instant expressed one way.

    *Refusing naive* -- a naive value is differently wrong on each side. The
    SQL path binds it into a ``timestamptz`` comparison that PostgreSQL
    resolves against the session time zone and answers silently; the Python
    path compares it against a UTC-normalized anchor and raises.

    *Converting aware* -- accepting ``+08:00`` unconverted is worse than
    refusing it, because it fails only on one dialect. SQLAlchemy's SQLite
    ``DATETIME`` binds the wall-clock components and drops the offset, so the
    SQL leg shifts the cutoff by the offset while the Python leg, comparing
    against an anchor normalized by :func:`_as_utc`, does not. The same
    instant then yields "expired" from the scanning path and "not expired"
    from the locked path, and #2563 deletes tasks up to one UTC offset short
    of the retention period.

    Every entry point that accepts a datetime routes through here:
    :func:`retention_cutoff`, :func:`retention_lease_clear_condition`,
    :func:`assess_task_retention`, and :func:`touch_task_last_activity`.
    """
    if now.tzinfo is None:
        raise ValueError(
            "retention predicates require a timezone-aware `now`; "
            "pass datetime.now(timezone.utc)"
        )
    return now.astimezone(timezone.utc)


def retention_cutoff(*, now: datetime, days: int | None) -> datetime | None:
    """The instant an anchor must not be newer than to have expired.

    ``None`` days means unlimited retention, one of the options open in
    #2567, and returns ``None`` -- not a cutoff far in the past.
    """
    now = _utc(now)
    if days is None:
        return None
    return now - timedelta(days=days)


def retention_terminal_condition() -> ColumnElement[bool]:
    """Leg 1: the task has reached a status from which nothing runs.

    Goes through ``task_status_predicate`` rather than ``Task.status.in_()``
    because that is this repo's single typed entry point for status SQL -- it
    is what keeps every predicate writing the enum *names* the column actually
    stores. Its return is untyped, hence the cast.
    """
    return cast(
        "ColumnElement[bool]", task_status_predicate.in_(RETENTION_TERMINAL_STATUSES)
    )


def retention_lease_clear_condition(*, now: datetime) -> ColumnElement[bool]:
    """Leg 2: no execution lease is live.

    A NULL lease is unowned; an elapsed one is abandoned. Note that expiry
    alone is not permission to *restart* the task (lease recovery classifies
    it first), but it is enough to establish that no worker is executing.
    """
    return or_(Task.lease_expires_at.is_(None), Task.lease_expires_at <= _utc(now))


def retention_commands_clear_condition() -> ColumnElement[bool]:
    """Leg 3: no accepted command is still owed execution.

    ``enqueue_task_command`` is a plain insert -- a task can hold a pending
    command with its status and ``updated_at`` untouched, so neither of those
    columns can stand in for this check.
    """
    return ~exists(
        select(1).where(
            TaskExecutionCommand.task_id == Task.id,
            TaskExecutionCommand.status.in_(RETENTION_LIVE_COMMAND_STATUSES),
        )
    )


def retention_quiescent_condition(*, now: datetime) -> ColumnElement[bool]:
    """All three legs: the task is not live in any sense the purge must respect."""
    return and_(
        retention_terminal_condition(),
        retention_lease_clear_condition(now=now),
        retention_commands_clear_condition(),
    )


def retention_expiry_condition(
    *, now: datetime, days: int | None
) -> ColumnElement[bool]:
    """Whether the anchor has aged past ``days``."""
    cutoff = retention_cutoff(now=now, days=days)
    if cutoff is None:
        return false()
    return retention_anchor() <= cutoff


def retention_candidate_condition(
    *, now: datetime, days: int | None
) -> ColumnElement[bool]:
    """Quiescent *and* expired: the filter a purge or a count scans on.

    Composable on purpose -- callers add their own predicates (a team, an id
    range, a batch limit) around it instead of restating eligibility.
    """
    return and_(
        retention_quiescent_condition(now=now),
        retention_expiry_condition(now=now, days=days),
    )


def count_retention_candidates(db: Session, *, now: datetime, days: int | None) -> int:
    """How many tasks would be eligible at ``days``. Takes no locks."""
    return int(
        db.execute(
            select(func.count())
            .select_from(Task)
            .where(retention_candidate_condition(now=now, days=days))
        ).scalar_one()
    )


def count_retention_candidate_trace_events(
    db: Session, *, now: datetime, days: int | None
) -> int:
    """How many ``trace_events`` rows hang off the tasks eligible at ``days``."""
    return int(
        db.execute(
            select(func.count())
            .select_from(TraceEvent)
            .where(
                TraceEvent.task_id.in_(
                    select(Task.id).where(
                        retention_candidate_condition(now=now, days=days)
                    )
                )
            )
        ).scalar_one()
    )


def count_quiescent_tasks(db: Session, *, now: datetime) -> int:
    """Tasks that are purgeable in principle, before any period is applied.

    Reported by the preview so the per-period numbers are auditable: the
    difference against the table total is the live-task population, not a
    retention effect.
    """
    return int(
        db.execute(
            select(func.count())
            .select_from(Task)
            .where(retention_quiescent_condition(now=now))
        ).scalar_one()
    )


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize a timestamp read back from the database to aware UTC.

    ``DateTime(timezone=True)`` columns come back tz-naive from SQLite, which
    stores the naked timestamp; PostgreSQL returns them aware. Same
    normalization the RUNNING-lease guard in task_orchestrator.py applies, for
    the same reason -- so a comparison written once is correct on both.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def assess_task_retention(
    db: Session,
    task_id: int,
    *,
    now: datetime,
    conversation_days: int | None,
    trace_days: int | None,
) -> RetentionAssessment:
    """Decide what may be expired for one task.

    The task row is locked with ``SELECT ... FOR UPDATE`` before the legs are
    evaluated, and the lock is held until the caller's transaction ends. Read
    the module docstring before treating that as a deletion licence: it is
    PostgreSQL-only, it covers the status and lease legs, and it does not
    cover the command leg -- ``task_execution_commands`` is a separate table
    that this lock never touches.

    There is no unlocked variant. Callers that must scan without locking --
    the preview, and whatever #2563 uses to choose a batch -- compose
    :func:`retention_candidate_condition` into their own query instead.

    The legs are evaluated in SQL, not re-implemented in Python, so this shares
    one definition with the set-scanning path. Only the expiry comparison
    happens here, against :func:`retention_cutoff` -- the same function the SQL
    path uses.
    """
    now = _utc(now)
    locked = db.execute(
        select(Task.id).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if locked is None:
        return RetentionAssessment(
            disposition=RetentionDisposition.NOT_ELIGIBLE,
            anchor=None,
            quiescent=False,
            blockers=("missing",),
        )

    row = db.execute(
        select(
            retention_anchor().label("anchor"),
            retention_terminal_condition().label("terminal"),
            retention_lease_clear_condition(now=now).label("lease_clear"),
            retention_commands_clear_condition().label("commands_clear"),
        ).where(Task.id == task_id)
    ).one_or_none()
    if row is None:
        return RetentionAssessment(
            disposition=RetentionDisposition.NOT_ELIGIBLE,
            anchor=None,
            quiescent=False,
            blockers=("missing",),
        )

    anchor = _as_utc(row.anchor)
    blockers = tuple(
        name
        for name, passed in (
            ("status", bool(row.terminal)),
            ("lease", bool(row.lease_clear)),
            ("commands", bool(row.commands_clear)),
        )
        if not passed
    )
    quiescent = not blockers
    if not quiescent or anchor is None:
        return RetentionAssessment(
            disposition=RetentionDisposition.NOT_ELIGIBLE,
            anchor=anchor,
            quiescent=quiescent,
            blockers=blockers if blockers else ("anchor",),
        )

    conversation_cutoff = retention_cutoff(now=now, days=conversation_days)
    trace_cutoff = retention_cutoff(now=now, days=trace_days)
    if conversation_cutoff is not None and anchor <= conversation_cutoff:
        disposition = RetentionDisposition.CONVERSATION_EXPIRED
    elif trace_cutoff is not None and anchor <= trace_cutoff:
        disposition = RetentionDisposition.TRACE_EXPIRED
    else:
        disposition = RetentionDisposition.NOT_ELIGIBLE
    return RetentionAssessment(disposition=disposition, anchor=anchor, quiescent=True)


def is_retention_eligible(
    db: Session,
    task_id: int,
    *,
    now: datetime,
    conversation_days: int | None,
    trace_days: int | None,
) -> bool:
    """Boolean form of :func:`assess_task_retention` for call sites that only branch.

    Prefer the assessment itself where the *reason* matters: a purge that logs
    "skipped, pending command" is debuggable, one that logs "skipped" is not.
    """
    return assess_task_retention(
        db,
        task_id,
        now=now,
        conversation_days=conversation_days,
        trace_days=trace_days,
    ).eligible


def touch_task_last_activity(
    db: Session, task_id: int, *, when: datetime | None = None
) -> None:
    """Advance the retention anchor because a conversational message was persisted.

    Called from the ``task_chat_messages`` insert paths and nowhere else --
    ``test_task_retention.py`` asserts that no other module writes this column,
    which is what keeps the anchor meaning "conversation", not "activity".

    Three properties this statement is shaped to hold:

    * **It never moves the anchor backwards.** A late write for an earlier
      message would otherwise shorten retention -- the dangerous direction,
      since it deletes data early. The guard also makes the call idempotent
      and order-independent.

      The guard compares two clocks, which is worth stating plainly: this
      stamp is the *application* clock, while ``task_chat_messages.created_at``
      and therefore the migration's backfill use the *database* clock
      (``server_default=func.now()``). If the database clock leads, the first
      touches after a backfill compare against a slightly-future anchor and
      are dropped, leaving the anchor stale by at most the skew until the
      application clock passes it. That bound -- seconds, against retention
      periods measured in months -- is why this is documented rather than
      fixed. The obvious fix is not free: ``func.now()`` renders as
      ``CURRENT_TIMESTAMP`` on SQLite, whose second-precision value sorts
      below a stored microsecond-precision one within the same second, which
      would drop touches on a far more routine path than clock skew.
    * **It does not disturb ``updated_at``.** Naming the column in the SET
      clause suppresses its ``onupdate``, the same idiom the lease writers
      use. An anchor write is not execution activity and must not be read as
      such by anything watching ``updated_at``.
    * **It does not flush the caller's pending work.** The staging
      (``_no_commit``) insert paths hand their transaction back to a caller
      that expects to commit it themselves; an autoflush here would surface
      their INSERT's errors inside this call instead.
    """
    moment = _utc(when) if when is not None else datetime.now(timezone.utc)
    with db.no_autoflush:
        db.execute(
            update(Task)
            .where(
                Task.id == task_id,
                or_(
                    Task.last_activity_at.is_(None),
                    Task.last_activity_at < moment,
                ),
            )
            .values(last_activity_at=moment, updated_at=Task.updated_at)
            .execution_options(synchronize_session=False)
        )
