"""Dispatch automatic resumes of interrupted tasks, and re-check them at claim.

The sweeper turns due ``scheduled`` rows of ``task_auto_recovery`` into
durable RESUME commands; ``resume_task`` re-checks each such command when it
is claimed (:func:`check_auto_resume_claim_sync`), because the task may have
moved on while the command waited. Rows become ``scheduled`` when an
interruption is recorded (``task_auto_recovery.record_interruption_no_commit``);
with nothing due a tick is a single ``EXISTS`` over the due index.

Every process that runs lease recovery runs a sweeper; correctness rests on
the compare-and-swaps below, not on a singleton.

Lock order, shared with every other writer of these rows: ``tasks`` ->
``task_auto_recovery`` -> ``trigger_runs``. Settlement and lease recovery
write the ``tasks`` row before they upsert the recovery row, so on
PostgreSQL the sweeper locks ``tasks`` first (``FOR UPDATE OF tasks SKIP
LOCKED``). Locking the recovery row first would deadlock with such a
settlement: the sweeper's command and event inserts take a key-share lock on
``tasks`` through their foreign keys. SQLite ignores row locks; there every
transition is a compare-and-swap that re-checks the task fence, the way
lease recovery's SQLite path does.

The fence is ``tasks.status = PAUSED``, ``control_state = 'paused'`` and the
row's ``run_id`` and ``paused_state_version``. It deliberately ignores
``runner_id``: in shared mode a PAUSED task may hold an idle coordinator's
owner lease, whose acquisition and release do not move ``state_version``;
``resume_task`` defers while another process holds a live lease.

A dispatch is one transaction: the CAS to ``dispatched``, its event, and the
command (staged last, as ``stage_task_command`` requires). The dispatcher is
notified after the commit. ``dispatched`` rows reuse ``next_attempt_at`` as
the time housekeeping next checks the dispatch, and clear it once the fence
has moved (the resume took effect or a person acted).

With ``XAGENT_TASK_AUTO_RESUME_ENABLED`` off nothing is dispatched and
nothing waits for the switch to come back: the tick drains ``scheduled``
rows to ``manual(auto_resume_disabled)`` and closes ``dispatched`` rows whose
command already ended the same way, failing their TriggerRun; a command
still queued is dropped by the claim-time guard.

A person can stop a pending resume
(:func:`cancel_pending_auto_resume_no_commit`), which moves the version like
a superseding turn.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterable

from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from ...config import (
    get_task_auto_resume_enabled,
    get_task_auto_resume_max_inflight,
    get_task_auto_resume_max_per_tick,
)
from ...core.agent.interruption import InterruptionReason, is_database_unavailable
from ...core.runtime_performance import (
    increment_counter,
    observe_value,
    register_observable_gauge,
)
from ..models.task import Task, TaskStatus, task_status_predicate
from ..models.task_auto_recovery import (
    TaskAutoRecovery,
    TaskAutoRecoveryState,
    TaskRecoveryEvent,
    TaskRecoveryEventType,
)
from ..models.task_command import TaskExecutionCommand
from ..utils.db_timezone import format_datetime_for_api
from .db_runtime import is_database_pool_timeout, run_db_io_cancellation_safe
from .ops_signals import (
    TASK_AUTO_RESUME_UNAVAILABLE,
    clear_degradation,
    register_degradation,
)
from .task_auto_recovery import (
    AUTO_RESUME_DISABLED_DETAIL,
    AUTO_RESUME_USER_CANCELLED_DETAIL,
    PLANNED_RECOVERY_EVENTS,
    TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    auto_recovery_eligibility,
    is_auto_resume_command_id,
    is_status_only_source,
)
from .task_auto_recovery_policy import (
    auto_resume_limit,
    auto_resume_recovery_state,
    next_auto_resume_at,
    scheduled_trigger_superseded,
    staleness_window_seconds,
)
from .task_command_transport import (
    AUTO_RESUME_COMMAND_PREFIX,
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PENDING,
    COMMAND_PROCESSING,
    COMMAND_TERMINAL,
    TaskCommandKind,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_dispatch_cap import EXECUTION_OPENING_KINDS
from .task_execution_admission import AdmissionQueueFull
from .task_execution_controller import TaskControlState
from .task_lease_service import utc_now
from .task_settlement_report import RecordedInterruption

logger = logging.getLogger(__name__)

# A dispatched command is checked again this long after dispatch (and after
# every check that finds it still queued).
DISPATCH_CHECK_GRACE_SECONDS = 60
# A full admission queue pushes the candidate this far out.
ADMISSION_FULL_DELAY_SECONDS = 30
# So does a refused staging that is neither a full queue nor a race nor a
# database outage: a configuration or programming fault repeats on every
# attempt, so retrying it on every tick only adds load. Only ``ValueError``
# and ``RuntimeError`` (``_EXPECTED_STAGING_REFUSALS``) count as refused; any
# other class still postpones the row but counts as a failed candidate.
STAGING_REFUSED_DELAY_SECONDS = 60
# The classes of ``stage_task_command`` failure that are a refusal, not a bug:
# a rejected command, an owner or task that moved, an admission wait.
_EXPECTED_STAGING_REFUSALS = (ValueError, RuntimeError)
_COMMAND_ID_MAX_CHARS = 64
# Consecutive failed ticks before TASK_AUTO_RESUME_UNAVAILABLE is raised.
_UNAVAILABLE_AFTER_FAILURES = 3
_FAILURE_LOG_INTERVAL_SECONDS = 60.0
_MAX_FAILURE_BACKOFF_SECONDS = 60
_MAX_FAILURE_BACKOFF_EXPONENT = 6

# TriggerRun errors for a scheduled resume that automatic recovery stopped.
TASK_AUTO_RESUME_EXHAUSTED_TRIGGER_ERROR = (
    "Task paused after a system interruption; automatic resume stopped after "
    "repeated attempts. Manual resume is required."
)
TASK_AUTO_RESUME_EXPIRED_TRIGGER_ERROR = (
    "Task paused after a system interruption; it was not resumed within the "
    "automatic-resume window. Manual resume is required."
)
TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR = (
    "Task paused after a system interruption and was superseded by a later "
    "scheduled run."
)
TASK_AUTO_RESUME_DISPATCH_FAILED_TRIGGER_ERROR = (
    "Task paused after a system interruption; automatic resume could not be "
    "started. Manual resume is required."
)
TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR = (
    "Task paused after a system interruption; automatic resume is disabled. "
    "Manual resume is required."
)

# Module-level so tests can seed it; drawn only by housekeeping reschedules.
_AUTO_RESUME_RNG = random.Random()

# Cached by each tick that measures it, for the in-flight gauge.
_last_inflight = 0

_SCHEDULED = TaskAutoRecoveryState.SCHEDULED
_DISPATCHED = TaskAutoRecoveryState.DISPATCHED


def auto_resume_command_id(paused_state_version: int, attempt: int, run_id: str) -> str:
    """The id of the RESUME that dispatches ``attempt`` at one fence.

    ``(task_id, command_id)`` is unique. The fence version grows with every
    interruption and ``attempt`` with every dispatch at one fence, so two
    dispatches never share an id even after the counters restart. The unique
    part leads; truncation to the column only shortens the run id.
    """

    return f"{AUTO_RESUME_COMMAND_PREFIX}{int(paused_state_version)}:{int(attempt)}:{run_id}"[
        :_COMMAND_ID_MAX_CHARS
    ]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Public notices (``task_recovery_update``)
# --------------------------------------------------------------------------

_STOP_REASONS = {
    TaskAutoRecoveryState.EXHAUSTED.value: "limit_reached",
    TaskAutoRecoveryState.EXPIRED.value: "expired",
    TaskAutoRecoveryState.SUPERSEDED.value: "superseded",
    TaskAutoRecoveryState.DISPATCH_FAILED.value: "failed",
    TaskAutoRecoveryState.INELIGIBLE.value: "ineligible",
}


def public_auto_resume_view(
    state: str,
    *,
    state_detail: str | None,
    reason: str,
    attempt: int | None,
    next_attempt_at: datetime | None,
) -> dict[str, Any] | None:
    """The client contract for one automatic-resume transition, or ``None``.

    Counters, ``state_detail``, ``last_error`` and command ids stay
    server-side. ``stale`` is not announced: a person's own action produced
    it, and that action announced itself. ``manual`` reached from an
    announced state (a scheduled or dispatched row) is ``stopped``:
    ``disabled`` when the sweeper or the guard closed it for a switched-off
    executor, ``cancelled`` when a person cancelled the pending resume, else
    ``ineligible`` (a reason with no policy: automatic resume will not act on
    the task). Callers decide whether a transition is announced at all; a
    recorded ``manual`` is not.
    """

    if state == _SCHEDULED.value:
        status, stop_reason = "scheduled", None
    elif state == _DISPATCHED.value:
        status, stop_reason = "resuming", None
    elif state in _STOP_REASONS:
        status, stop_reason = "stopped", _STOP_REASONS[state]
    elif state == TaskAutoRecoveryState.MANUAL.value:
        status = "stopped"
        stop_reason = {
            AUTO_RESUME_DISABLED_DETAIL: "disabled",
            AUTO_RESUME_USER_CANCELLED_DETAIL: "cancelled",
        }.get(state_detail or "", "ineligible")
    else:
        return None
    return {
        "status": status,
        "stop_reason": stop_reason,
        "reason": reason,
        "attempt": attempt,
        "next_attempt_at": (
            format_datetime_for_api(next_attempt_at) if status == "scheduled" else None
        ),
    }


def current_auto_resume_view(
    task: Task, row: TaskAutoRecovery
) -> dict[str, Any] | None:
    """The ``auto_resume`` a task read reports for ``row``, or ``None``.

    The ``task_recovery_update`` notice for the row's current state, computed
    from the row alone (no query), and only at the sweeper's fence
    (:func:`_fence_mismatch`). ``attempt`` is derived as at record time
    (:meth:`RecordedInterruption.from_row`); ``stale`` and ``disabled`` are
    never announced (:func:`public_auto_resume_view`).

    Every ``manual`` and ``ineligible`` reads as ``None``, except ``manual``
    (``user_cancelled``). Recording one announces nothing, and the same state
    reached later -- ``manual`` (``auto_resume_disabled``) by the switch-off
    drain or the claim guard, ``ineligible`` at dispatch -- was announced
    ``stopped``, but the row cannot tell the two apart. Nor is a give-up past
    a rolled-back resume shown: its notice names a later version than the
    row's fence. ``manual`` (``user_cancelled``) is the one ``manual`` that
    can be told apart: only a person's cancel
    (:func:`cancel_pending_auto_resume_no_commit`) writes it, never a
    recording, and in the same transaction it moves the version and the row
    to that new fence, so the row reads ``stopped`` / ``cancelled`` exactly
    as the cancel announced it.

    Invariant: a row that reads as ``None`` here (``manual``, ``ineligible``)
    never returns to ``scheduled`` at the same fence; scheduling it again
    must advance ``state_version``. Any later writer (the interaction gate)
    must keep this, or a client would discard a live ``scheduled`` notice as
    older than the ``None`` it read. ``manual`` (``user_cancelled``) keeps it
    too, and reads ``stopped``, the last word of the fence the cancel
    created: only three writers ever write ``scheduled`` -- recording an
    interruption (always at the larger version its settlement wrote),
    housekeeping's reschedule and the rolled-back claim's re-fence (both
    only out of ``dispatched``) -- so nothing schedules that row again at
    its fence.

    Ordering, for clients merging reads with live notices: a later
    ``(run_id, state_version)`` wins, as between live notices. Within one
    fence ``state_version`` does not move, so order by ``k(status,
    attempt)``: ``(attempt, 0)`` for ``resuming``, ``(attempt - 1, 1)`` for
    ``scheduled`` and ``(attempt, 2)`` for ``stopped``. It strictly
    increases along a fence's transitions (``scheduled`` -> ``resuming`` ->
    ``scheduled`` -> ... -> ``stopped``); equal keys (a postponed
    ``scheduled``) keep the later arrival, and ``stopped`` is the fence's
    last word.

    Terminal ``None``: only ``auto_recovery`` present with ``auto_resume:
    None`` at the fence a client holds ends that fence's notices (a state
    never announced, or one of the stops above), so the client drops the
    notice it holds. A whole ``auto_recovery: None`` (a failed read, or a
    rolled-back resume before its claim moves the row to the new fence)
    says nothing about automatic resume: the client keeps its notice until
    the fence changes.

    Never cached on its own: the sweeper's transitions do not write
    ``tasks``, so every read computes it afresh or keys its cache on it.
    """

    if _fence_mismatch(task, row) is not None:
        return None
    state = str(row.state)
    # Also recorded, unannounced; indistinguishable from the later ones. A
    # cancel is never recorded and always announced at the row's fence.
    if state == TaskAutoRecoveryState.INELIGIBLE.value or (
        state == TaskAutoRecoveryState.MANUAL.value
        and row.state_detail != AUTO_RESUME_USER_CANCELLED_DETAIL
    ):
        return None
    return public_auto_resume_view(
        state,
        state_detail=row.state_detail,
        reason=str(row.reason),
        attempt=RecordedInterruption.from_row(row).attempt,
        next_attempt_at=row.next_attempt_at,
    )


@dataclass(frozen=True)
class RecoveryNotice:
    """A committed transition to announce as ``task_recovery_update``."""

    task_id: int
    run_id: str
    # The row's fence; a client already past it ignores the notice.
    state_version: int
    auto_resume: dict[str, Any]

    def message(self) -> dict[str, Any]:
        return {
            "type": "task_recovery_update",
            "task_id": self.task_id,
            "run_id": self.run_id,
            "state_version": self.state_version,
            "auto_resume": self.auto_resume,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }


def _notice(
    row: TaskAutoRecovery,
    state: TaskAutoRecoveryState,
    *,
    state_detail: str | None,
    attempt: int | None,
    next_attempt_at: datetime | None = None,
    state_version: int | None = None,
) -> RecoveryNotice | None:
    """``state_version`` names the pause the client shows when it is not the
    row's fence (a give-up past a rolled-back resume); the row's otherwise."""
    view = public_auto_resume_view(
        state.value,
        state_detail=state_detail,
        reason=str(row.reason),
        attempt=attempt,
        next_attempt_at=next_attempt_at,
    )
    if view is None:
        return None
    return RecoveryNotice(
        task_id=int(row.task_id),
        run_id=str(row.run_id),
        state_version=(
            int(row.paused_state_version) if state_version is None else state_version
        ),
        auto_resume=view,
    )


# Recorded states announced at record time. A recorded ``manual``,
# ``ineligible`` or ``disabled`` changes nothing for the client: the pause it
# already shows is all there is.
_RECORDED_ANNOUNCED = frozenset(state.value for state in PLANNED_RECOVERY_EVENTS)


def recorded_notice(recorded: RecordedInterruption | None) -> RecoveryNotice | None:
    """The notice for what recording an interruption committed, if any."""

    if recorded is None or recorded.state not in _RECORDED_ANNOUNCED:
        return None
    view = public_auto_resume_view(
        recorded.state,
        state_detail=recorded.state_detail,
        reason=recorded.reason,
        attempt=recorded.attempt,
        next_attempt_at=recorded.next_attempt_at,
    )
    if view is None:
        return None
    return RecoveryNotice(
        task_id=recorded.task_id,
        run_id=recorded.run_id,
        state_version=recorded.paused_state_version,
        auto_resume=view,
    )


def trigger_error_for_recorded(
    recorded: RecordedInterruption | None, *, default: str
) -> str | None:
    """The TriggerRun error for a recorded PAUSED interruption, or ``None``.

    ``None`` keeps the run RUNNING: a scheduled resume settles it later. A
    stop names why automatic resume will not run; anything else (``manual``,
    ``ineligible``, or nothing recorded) is the caller's ``default``.
    """

    if recorded is None:
        return default
    if recorded.state == _SCHEDULED.value:
        return None
    stop = _STOPS.get(TaskAutoRecoveryState(recorded.state))
    if recorded.state in _RECORDED_ANNOUNCED and stop is not None:
        return stop[1]
    return default


async def publish_recovery_notices(
    notices: Iterable[RecoveryNotice],
) -> tuple[dict[str, Any], ...]:
    """Broadcast committed notices; best effort, a failure is only logged.

    Returns each notice's message as built for the broadcast (timestamp
    included), whether or not that broadcast went through.
    """

    from .task_events import publish_task_event

    messages: list[dict[str, Any]] = []
    for notice in notices:
        message = notice.message()
        messages.append(message)
        try:
            await publish_task_event(message, notice.task_id)
        except Exception:
            logger.warning(
                "task_recovery_update committed but broadcast failed for task %s",
                notice.task_id,
                exc_info=True,
            )
    return tuple(messages)


# --------------------------------------------------------------------------
# Fence and shared transition helpers
# --------------------------------------------------------------------------


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def _fence_mismatch(task: Task, row: TaskAutoRecovery) -> str | None:
    """The first fence column that no longer matches ``row``, or ``None``."""

    if task.status != TaskStatus.PAUSED:
        return "status"
    if task.control_state != TaskControlState.PAUSED.value:
        return "control_state"
    if task.run_id != row.run_id:
        return "run_id"
    if int(task.state_version or 0) != int(row.paused_state_version):
        return "state_version"
    return None


def _task_at_fence(task_id: int, run_id: str, paused_state_version: int) -> Any:
    """``EXISTS`` over ``tasks`` for the fence, for compare-and-swaps."""

    return (
        select(Task.id)
        .where(
            Task.id == task_id,
            task_status_predicate.eq(TaskStatus.PAUSED),
            Task.control_state == TaskControlState.PAUSED.value,
            Task.run_id == run_id,
            Task.state_version == paused_state_version,
        )
        .exists()
    )


def _row_cas(
    row: TaskAutoRecovery,
    from_state: TaskAutoRecoveryState,
    *,
    fenced: bool,
) -> list[Any]:
    """The compare-and-swap predicate for a transition out of ``from_state``."""

    predicates: list[Any] = [
        TaskAutoRecovery.task_id == int(row.task_id),
        TaskAutoRecovery.state == from_state.value,
        TaskAutoRecovery.run_id == row.run_id,
        TaskAutoRecovery.paused_state_version == int(row.paused_state_version),
    ]
    if from_state is _DISPATCHED:
        predicates.append(TaskAutoRecovery.last_command_id == row.last_command_id)
    if fenced:
        predicates.append(
            _task_at_fence(
                int(row.task_id), str(row.run_id), int(row.paused_state_version)
            )
        )
    return predicates


# Each terminal state's event and TriggerRun error. A dispatch-time
# ``manual`` (a reason with no policy) is the task being out of automatic
# recovery's reach, recorded as an ``ineligible`` event whose detail names
# the state; a disabled switch passes its own event and error.
# ``stale`` never fails a TriggerRun.
_STOPS: dict[TaskAutoRecoveryState, tuple[TaskRecoveryEventType, str | None]] = {
    TaskAutoRecoveryState.EXHAUSTED: (
        TaskRecoveryEventType.EXHAUSTED,
        TASK_AUTO_RESUME_EXHAUSTED_TRIGGER_ERROR,
    ),
    TaskAutoRecoveryState.EXPIRED: (
        TaskRecoveryEventType.EXPIRED,
        TASK_AUTO_RESUME_EXPIRED_TRIGGER_ERROR,
    ),
    TaskAutoRecoveryState.SUPERSEDED: (
        TaskRecoveryEventType.SUPERSEDED,
        TASK_AUTO_RESUME_SUPERSEDED_TRIGGER_ERROR,
    ),
    TaskAutoRecoveryState.DISPATCH_FAILED: (
        TaskRecoveryEventType.DISPATCH_FAILED,
        TASK_AUTO_RESUME_DISPATCH_FAILED_TRIGGER_ERROR,
    ),
    TaskAutoRecoveryState.INELIGIBLE: (
        TaskRecoveryEventType.INELIGIBLE,
        TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    ),
    TaskAutoRecoveryState.MANUAL: (
        TaskRecoveryEventType.INELIGIBLE,
        TASK_INTERRUPTION_PAUSED_TRIGGER_ERROR,
    ),
    TaskAutoRecoveryState.STALE: (TaskRecoveryEventType.STALE, None),
}
_DISABLED_STOP = (
    TaskRecoveryEventType.SKIPPED_DISABLED,
    TASK_AUTO_RESUME_DISABLED_TRIGGER_ERROR,
)
# No TriggerRun error: a person's cancel leaves the run RUNNING, as a
# person's pause does (#2177); the resumed run's settlement ends it.
_CANCELLED_STOP: tuple[TaskRecoveryEventType, str | None] = (
    TaskRecoveryEventType.SKIPPED_USER_INTENT,
    None,
)


class _Outcome(str, Enum):
    DISPATCHED = "dispatched"
    GAVE_UP = "gave_up"
    STALE = "stale"
    RESCHEDULED = "rescheduled"
    CONFIRMED = "confirmed"
    # Housekeeping found the command still queued; checked again later.
    REARMED = "rearmed"
    # Nothing written: the row moved, or a CAS lost.
    RACED = "raced"
    # Nothing written: a dispatch was due but the budget is spent.
    CAPACITY = "capacity"


@dataclass(frozen=True)
class _CandidateResult:
    outcome: _Outcome
    notice: RecoveryNotice | None = None
    staged: bool = False
    dispatch_delay_seconds: float | None = None

    @property
    def writes(self) -> bool:
        return self.outcome not in (_Outcome.RACED, _Outcome.CAPACITY)


_RACED = _CandidateResult(_Outcome.RACED)


def _give_up_no_commit(
    db: Session,
    task: Task,
    row: TaskAutoRecovery,
    new_state: TaskAutoRecoveryState,
    *,
    from_state: TaskAutoRecoveryState,
    at: str,
    state_detail: str | None = None,
    event_detail: dict[str, Any] | None = None,
    stop: tuple[TaskRecoveryEventType, str | None] | None = None,
    fenced: bool | None = None,
    log_level: int = logging.WARNING,
    counter: str = "xagent.task.auto_resume.gave_up",
) -> _CandidateResult:
    """Stop automatic recovery of ``row`` in ``new_state``; no commit.

    The CAS re-checks the fence except for ``stale``, which exists because
    the fence no longer holds; ``fenced`` overrides that for a caller that
    has already established why the fence moved. TriggerRun: a give-up with
    an error fails the task's PENDING/RUNNING runs with it (a ``stop``
    without one, a person's cancel, leaves them); ``stale`` only mirrors a
    task that already ended, since otherwise a person resumed or changed it
    and that run's own settlement reports the real outcome. That includes a
    task left paused on the same run, as after a user's resume that failed
    to start: like any user pause, its run stays RUNNING until a later run
    settles it (#2177), and deleting the task fails it. ``stop`` overrides
    the state's ``(event, TriggerRun error)``; ``log_level`` that of the
    give-up's log line, and ``counter`` the metric it is counted on (a
    person's cancel is not a system give-up).
    """

    from .task_orchestrator import sync_trigger_run_status

    event, trigger_error = stop or _STOPS[new_state]
    stale = new_state is TaskAutoRecoveryState.STALE
    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(
                *_row_cas(
                    row, from_state, fenced=not stale if fenced is None else fenced
                )
            )
            .values(
                state=new_state.value, state_detail=state_detail, next_attempt_at=None
            )
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        increment_counter(
            "xagent.task.auto_resume.skipped", attributes={"outcome": "raced"}
        )
        return _RACED
    detail: dict[str, Any] = {"at": at, "state": new_state.value}
    if state_detail is not None:
        detail["state_detail"] = state_detail
    if event_detail:
        detail.update(event_detail)
    attempts = int(row.total_resumes or 0)
    db.add(
        TaskRecoveryEvent(
            task_id=int(row.task_id),
            run_id=row.run_id,
            event=event.value,
            reason=row.reason,
            attempt=attempts,
            detail=detail,
        )
    )
    if trigger_error is not None:
        sync_trigger_run_status(
            db, task, TaskStatus.PAUSED, error_message=trigger_error
        )
    elif task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        sync_trigger_run_status(db, task, TaskStatus(task.status))
    db.flush()
    if stale:
        increment_counter("xagent.task.auto_resume.stale", attributes={"operation": at})
        logger.info(
            "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
            "attempt=%s next_attempt_at=None fence_mismatch=%s",
            row.task_id,
            row.run_id,
            row.reason,
            new_state.value,
            attempts,
            (event_detail or {}).get("mismatch"),
        )
        return _CandidateResult(_Outcome.STALE)
    increment_counter(
        counter,
        attributes={"outcome": new_state.value, "operation": at},
    )
    logger.log(
        log_level,
        "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
        "attempt=%s next_attempt_at=None state_detail=%s",
        row.task_id,
        row.run_id,
        row.reason,
        new_state.value,
        attempts,
        state_detail,
    )
    return _CandidateResult(
        _Outcome.GAVE_UP,
        _notice(
            row,
            new_state,
            state_detail=state_detail,
            attempt=attempts,
            # Past a rolled-back resume the task rests paused on the same
            # run at a later version; the client shows that pause, so name
            # it, or a viewer holding that version drops the notice as old.
            state_version=None if fenced is not False else int(task.state_version or 0),
        ),
    )


def _load_locked_no_commit(
    db: Session, task_id: int
) -> tuple[Task | None, TaskAutoRecovery | None]:
    """Read the task, then the recovery row, in lock order.

    On PostgreSQL the caller already holds the ``tasks`` row lock; locking
    the recovery row then may wait, but only on a writer that also holds
    ``tasks`` first, so it cannot close a cycle.
    """

    task = db.get(Task, task_id, populate_existing=True)
    row = db.get(
        TaskAutoRecovery, task_id, with_for_update=True, populate_existing=True
    )
    return task, row


def _limit_state(
    row: TaskAutoRecovery, kind: str | None, now: datetime
) -> TaskAutoRecoveryState | None:
    limit = auto_resume_limit(
        reason=row.reason,
        kind=kind,
        interrupted_at=row.interrupted_at,
        episode_started_at=row.episode_started_at,
        no_progress_resumes=int(row.no_progress_resumes or 0),
        total_resumes=int(row.total_resumes or 0),
        now=now,
    )
    return None if limit is None else auto_resume_recovery_state(limit)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


class _DispatchRefused(Exception):
    """A dispatch whose staging was refused, carrying the fence the candidate
    was seen at. The scan postpones that one row and goes on."""

    delay_seconds: int

    def __init__(self, task_id: int, paused_state_version: int) -> None:
        super().__init__(task_id, paused_state_version)
        self.task_id = task_id
        self.paused_state_version = paused_state_version


class _AdmissionFull(_DispatchRefused):
    """``AdmissionQueueFull``: the candidate's admission bucket is full."""

    delay_seconds = ADMISSION_FULL_DELAY_SECONDS


class _StagingRefused(_DispatchRefused):
    """``stage_task_command`` raised something other than a full queue, a race
    or a database outage (chained as ``__cause__``). The row is postponed
    either way; ``scan_due`` counts it as refused or as failed by the class."""

    delay_seconds = STAGING_REFUSED_DELAY_SECONDS


def _process_due_candidate_no_commit(
    db: Session, task_id: int, *, now: datetime, can_dispatch: bool
) -> _CandidateResult:
    """Dispatch one due ``scheduled`` row, or stop its automatic recovery.

    Stopping never consumes the dispatch budget, so rows that only need to
    be closed keep moving while in-flight resumes are at the cap. Writes,
    in order: the CAS, its event, then the command, which must be the last
    write before the caller commits.
    """

    task, row = _load_locked_no_commit(db, task_id)
    if (
        task is None
        or row is None
        or row.state != _SCHEDULED.value
        or row.next_attempt_at is None
        or _utc(row.next_attempt_at) > now
    ):
        return _RACED
    mismatch = _fence_mismatch(task, row)
    if mismatch is not None:
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.STALE,
            from_state=_SCHEDULED,
            at="dispatch",
            state_detail=f"dispatch:{mismatch}",
            event_detail={"mismatch": mismatch},
        )
    # Recomputed: the task's kind decides its window and may have changed.
    eligibility = auto_recovery_eligibility(db, task)
    if not eligibility.eligible:
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.INELIGIBLE,
            from_state=_SCHEDULED,
            at="dispatch",
            state_detail=eligibility.detail,
        )
    if scheduled_trigger_superseded(db, task):
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.SUPERSEDED,
            from_state=_SCHEDULED,
            at="dispatch",
        )
    limit_state = _limit_state(row, eligibility.kind, now)
    if limit_state is not None:
        return _give_up_no_commit(
            db, task, row, limit_state, from_state=_SCHEDULED, at="dispatch"
        )
    if not can_dispatch:
        return _CandidateResult(_Outcome.CAPACITY)

    run_id = str(row.run_id)
    paused_state_version = int(row.paused_state_version)
    attempt = int(row.total_resumes or 0) + 1
    command_id = auto_resume_command_id(paused_state_version, attempt, run_id)
    next_check_at = now + timedelta(seconds=DISPATCH_CHECK_GRACE_SECONDS)
    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(
                *_row_cas(row, _SCHEDULED, fenced=True),
                # The attempt (and so the command id) derives from it: a view
                # read before another dispatch must not reuse that attempt.
                TaskAutoRecovery.total_resumes == int(row.total_resumes or 0),
            )
            .values(
                state=_DISPATCHED.value,
                state_detail=None,
                no_progress_resumes=TaskAutoRecovery.no_progress_resumes + 1,
                total_resumes=TaskAutoRecovery.total_resumes + 1,
                last_command_id=command_id,
                next_attempt_at=next_check_at,
            )
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        increment_counter(
            "xagent.task.auto_resume.skipped", attributes={"outcome": "raced"}
        )
        return _RACED
    db.add(
        TaskRecoveryEvent(
            task_id=task_id,
            run_id=run_id,
            event=TaskRecoveryEventType.AUTO_RESUMED.value,
            reason=row.reason,
            attempt=attempt,
            detail={"kind": eligibility.kind},
        )
    )
    db.flush()
    try:
        staged = stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=int(task.user_id),
            command_id=command_id,
            kind=TaskCommandKind.RESUME,
            payload={
                "type": "resume_task",
                "auto_resume": {
                    "expected_run_id": run_id,
                    "expected_state_version": paused_state_version,
                    "reason": row.reason,
                    "attempt": attempt,
                },
            },
            target_run_id=run_id,
            reserved=True,
        )
    except AdmissionQueueFull as exc:
        raise _AdmissionFull(task_id, paused_state_version) from exc
    except IntegrityError:
        # A race on the command row: ``_settle_candidate`` counts it.
        raise
    except Exception as exc:
        if is_database_unavailable(exc) or is_database_pool_timeout(exc):
            raise
        raise _StagingRefused(task_id, paused_state_version) from exc
    if not staged.created:
        # Only a race that dispatched this exact (fence, attempt) gets here.
        # Commit ``dispatched`` only if that command is this dispatch and is
        # still going to run; otherwise roll the CAS back.
        live = staged.payload_matches and staged.status not in COMMAND_TERMINAL
        increment_counter(
            "xagent.task.auto_resume.skipped",
            attributes={"outcome": "command_exists" if live else "raced"},
        )
        logger.warning(
            "component=auto-resume task_id=%s run_id=%s command %s already "
            "existed when dispatching attempt %s (status=%s, matches=%s)",
            task_id,
            run_id,
            command_id,
            attempt,
            staged.status,
            staged.payload_matches,
        )
        if not live:
            return _RACED
    increment_counter(
        "xagent.task.auto_resume.dispatched",
        attributes={
            "operation": str(row.reason),
            "outcome": str(attempt) if attempt < 4 else "4+",
        },
    )
    logger.info(
        "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
        "attempt=%s next_attempt_at=%s",
        task_id,
        run_id,
        row.reason,
        _DISPATCHED.value,
        attempt,
        next_check_at,
    )
    return _CandidateResult(
        _Outcome.DISPATCHED,
        _notice(row, _DISPATCHED, state_detail=None, attempt=attempt),
        staged=staged.created,
        dispatch_delay_seconds=(now - _utc(row.next_attempt_at)).total_seconds(),
    )


def _postpone_candidate(
    task_id: int,
    *,
    paused_state_version: int,
    now: datetime,
    delay_seconds: int,
) -> None:
    """Push a candidate its dispatch was refused for; a single-row CAS.

    Fresh session: the candidate's own transaction was rolled back. Matches
    only a row still ``scheduled`` at ``paused_state_version``.
    """

    from ..models.database import get_session_local

    with get_session_local()() as db:
        try:
            db.execute(
                update(TaskAutoRecovery)
                .where(
                    TaskAutoRecovery.task_id == task_id,
                    TaskAutoRecovery.state == _SCHEDULED.value,
                    TaskAutoRecovery.paused_state_version == paused_state_version,
                )
                .values(next_attempt_at=now + timedelta(seconds=delay_seconds))
                .execution_options(synchronize_session=False)
            )
            db.commit()
        except Exception:
            db.rollback()
            raise


# --------------------------------------------------------------------------
# Housekeeping of dispatched rows
# --------------------------------------------------------------------------


def _housekeep_candidate_no_commit(
    db: Session, task_id: int, *, now: datetime, enabled: bool = True
) -> _CandidateResult:
    """Check one ``dispatched`` row whose check time has come.

    While the task still rests ``paused`` on the same run, nothing has
    resumed it, whatever the version: a rolled-back resume moves the version
    by two. So at that state a queued or failed command is never taken for
    a confirmation:

    - command queued (including a retry waiting out its backoff after a
      rollback, which the claim guard normally re-arms): check again after
      the grace period (with the switch off, the claim-time guard drops it);
    - command failed: the resume itself failed. A person acting first makes
      the claim guard complete the command as ``auto_skipped``, so a failed
      one is the resume failing -- rare after a rollback, since the claim
      guard re-arms the first retry, so only when every attempt failed;
    - fence otherwise moved: the resume took effect or a person acted;
      confirm by clearing ``next_attempt_at`` (no event);
    - switch off, command ended (or missing), the resume not having taken
      effect: the row rests in ``manual(auto_resume_disabled)``, as a
      drained row does;
    - command failed or missing otherwise: ``dispatch_failed``;
    - command completed at an unchanged fence (e.g. ``already_in_progress``):
      the resume did not happen; schedule it again unless a limit stops it.
    """

    task, row = _load_locked_no_commit(db, task_id)
    if (
        task is None
        or row is None
        or row.state != _DISPATCHED.value
        or row.next_attempt_at is None
        or _utc(row.next_attempt_at) > now
    ):
        return _RACED
    command = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.command_id == row.last_command_id,
        )
        .one_or_none()
        if row.last_command_id is not None
        else None
    )
    at_fence = _fence_mismatch(task, row) is None
    fence_moved = not at_fence
    paused_on_run = (
        task.status == TaskStatus.PAUSED
        and task.control_state == TaskControlState.PAUSED.value
        and task.run_id == row.run_id
    )
    queued = command is not None and command.status not in COMMAND_TERMINAL
    failed = command is not None and command.status == COMMAND_FAILED
    if fence_moved and paused_on_run and (queued or failed):
        fence_moved = False
    if fence_moved or queued:
        # Confirmed, or still queued: only the next check time changes.
        updated = _rowcount(
            db.execute(
                update(TaskAutoRecovery)
                .where(*_row_cas(row, _DISPATCHED, fenced=at_fence))
                .values(
                    next_attempt_at=(
                        None
                        if fence_moved
                        else now + timedelta(seconds=DISPATCH_CHECK_GRACE_SECONDS)
                    )
                )
                .execution_options(synchronize_session=False)
            )
        )
        if updated != 1:
            return _RACED
        if fence_moved:
            logger.info(
                "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
                "attempt=%s next_attempt_at=None confirmed",
                row.task_id,
                row.run_id,
                row.reason,
                _DISPATCHED.value,
                row.total_resumes,
            )
            increment_counter("xagent.task.auto_resume.confirmed")
            return _CandidateResult(_Outcome.CONFIRMED)
        # A growing count means dispatched commands sit in the queue.
        increment_counter("xagent.task.auto_resume.rearmed")
        return _CandidateResult(_Outcome.REARMED)
    if not enabled:
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.MANUAL,
            from_state=_DISPATCHED,
            at="drain",
            state_detail=AUTO_RESUME_DISABLED_DETAIL,
            stop=_DISABLED_STOP,
            fenced=at_fence,
        )
    command_result: dict[str, Any] = (
        command.result
        if command is not None and isinstance(command.result, dict)
        else {}
    )
    if command is None or command.status == COMMAND_FAILED:
        rejection = command_result.get("rejection_reason")
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.DISPATCH_FAILED,
            from_state=_DISPATCHED,
            at="housekeeping",
            event_detail={
                "command": "missing" if command is None else "failed",
                "rejection_reason": (rejection if isinstance(rejection, str) else None),
            },
            # A failed resume that rolled back moved the version; the row's
            # own CAS still pins this dispatch.
            fenced=at_fence,
        )

    assert command.status == COMMAND_COMPLETED
    resume_outcome = command_result.get("resume_outcome")
    after = resume_outcome if isinstance(resume_outcome, str) else "unknown"
    return _reschedule_dispatched_no_commit(
        db, task, row, now=now, at="housekeeping", after=after
    )


def _reschedule_dispatched_no_commit(
    db: Session,
    task: Task,
    row: TaskAutoRecovery,
    *,
    now: datetime,
    at: str,
    after: str,
) -> _CandidateResult:
    """A dispatch that did not resume the task: schedule it again or stop.

    The counters stand: the dispatch happened, and the limits are what keep
    this from looping.
    """

    limit_state = _limit_state(row, auto_recovery_eligibility(db, task).kind, now)
    if limit_state is not None:
        return _give_up_no_commit(
            db,
            task,
            row,
            limit_state,
            from_state=_DISPATCHED,
            at=at,
            event_detail={"after": after},
        )
    next_attempt_at = next_auto_resume_at(
        row.reason, int(row.no_progress_resumes or 0), now, _AUTO_RESUME_RNG
    )
    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, _DISPATCHED, fenced=True))
            .values(
                state=_SCHEDULED.value,
                state_detail=None,
                next_attempt_at=next_attempt_at,
            )
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        return _RACED
    attempt = int(row.total_resumes or 0) + 1
    db.add(
        TaskRecoveryEvent(
            task_id=int(row.task_id),
            run_id=row.run_id,
            event=TaskRecoveryEventType.SCHEDULED.value,
            reason=row.reason,
            attempt=attempt,
            next_attempt_at=next_attempt_at,
            detail={"at": at, "after": after},
        )
    )
    db.flush()
    increment_counter(
        "xagent.task.auto_resume.rescheduled", attributes={"outcome": after}
    )
    logger.info(
        "component=auto-resume task_id=%s run_id=%s reason=%s state=%s "
        "attempt=%s next_attempt_at=%s after=%s",
        row.task_id,
        row.run_id,
        row.reason,
        _SCHEDULED.value,
        attempt,
        next_attempt_at,
        after,
    )
    return _CandidateResult(
        _Outcome.RESCHEDULED,
        _notice(
            row,
            _SCHEDULED,
            state_detail=None,
            attempt=attempt,
            next_attempt_at=next_attempt_at,
        ),
    )


def _drain_candidate_no_commit(db: Session, task_id: int) -> _CandidateResult:
    """Stop one ``scheduled`` row because automatic resume is switched off.

    Not retroactive: a row stays stopped when the switch comes back. A row
    whose fence already moved is ``stale`` like at dispatch.
    """

    task, row = _load_locked_no_commit(db, task_id)
    if task is None or row is None or row.state != _SCHEDULED.value:
        return _RACED
    mismatch = _fence_mismatch(task, row)
    if mismatch is not None:
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.STALE,
            from_state=_SCHEDULED,
            at="drain",
            state_detail=f"drain:{mismatch}",
            event_detail={"mismatch": mismatch},
        )
    return _give_up_no_commit(
        db,
        task,
        row,
        TaskAutoRecoveryState.MANUAL,
        from_state=_SCHEDULED,
        at="drain",
        state_detail=AUTO_RESUME_DISABLED_DETAIL,
        stop=_DISABLED_STOP,
    )


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------

_Cursor = tuple[datetime, int]


def _use_postgresql_partitioning() -> bool:
    """Return whether candidate row locking can partition sweeper workers."""

    from ..models.database import get_engine

    return get_engine().dialect.name == "postgresql"


def _candidates_statement(
    state: TaskAutoRecoveryState,
    *,
    now: datetime | None,
    after: _Cursor | None,
) -> Any:
    """Rows in ``state``, due by ``now`` when given, in ``(time, task)`` order.

    Joined to ``tasks`` so the PostgreSQL path can lock the task row (and
    skip a row whose task another transaction holds) before the recovery
    row.
    """

    statement = (
        select(TaskAutoRecovery.task_id, TaskAutoRecovery.next_attempt_at)
        .join(Task, Task.id == TaskAutoRecovery.task_id)
        .where(
            TaskAutoRecovery.state == state.value,
            TaskAutoRecovery.next_attempt_at.is_not(None),
        )
    )
    if now is not None:
        statement = statement.where(TaskAutoRecovery.next_attempt_at <= now)
    if after is not None:
        after_at, after_task_id = after
        statement = statement.where(
            or_(
                TaskAutoRecovery.next_attempt_at > after_at,
                and_(
                    TaskAutoRecovery.next_attempt_at == after_at,
                    TaskAutoRecovery.task_id > after_task_id,
                ),
            )
        )
    return statement.order_by(
        TaskAutoRecovery.next_attempt_at, TaskAutoRecovery.task_id
    )


def select_next_candidate_for_update(
    db: Session,
    state: TaskAutoRecoveryState,
    *,
    now: datetime | None,
    after: _Cursor | None = None,
) -> _Cursor | None:
    """Lock and return one candidate's ``tasks`` row, skipping held ones.

    PostgreSQL only: renders ``FOR UPDATE OF tasks SKIP LOCKED``, so peer
    sweepers partition the candidates and a row whose task a settlement
    holds is left for a later tick. The caller processes the candidate in
    this same transaction.
    """

    picked = db.execute(
        _candidates_statement(state, now=now, after=after)
        .with_for_update(of=Task, skip_locked=True)
        .limit(1)
    ).first()
    if picked is None:
        return None
    return picked.next_attempt_at, int(picked.task_id)


def _candidates(
    state: TaskAutoRecoveryState, *, now: datetime | None, limit: int
) -> Iterator[tuple[Session, int]]:
    """Yield up to ``limit`` candidates, each in its own open transaction.

    The consumer commits or rolls back. PostgreSQL locks each candidate's
    task row in that transaction; SQLite scans one page up front and relies
    on the compare-and-swaps.
    """

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    if _use_postgresql_partitioning():
        cursor: _Cursor | None = None
        for _ in range(limit):
            with SessionLocal() as db:
                cursor = select_next_candidate_for_update(
                    db, state, now=now, after=cursor
                )
                if cursor is None:
                    return
                yield db, cursor[1]
        return

    with SessionLocal() as scan_db:
        task_ids = [
            int(task_id)
            for task_id, _at in scan_db.execute(
                _candidates_statement(state, now=now, after=None).limit(limit)
            ).all()
        ]
    for task_id in task_ids:
        with SessionLocal() as db:
            yield db, task_id


# --------------------------------------------------------------------------
# Tick
# --------------------------------------------------------------------------


@dataclass
class AutoResumeTickReport:
    """What one sweeper tick committed."""

    dispatched: int = 0
    gave_up: int = 0
    stale: int = 0
    rescheduled: int = 0
    confirmed: int = 0
    rearmed: int = 0
    raced: int = 0
    failed: int = 0
    staged_commands: int = 0
    # Measured only when a due dispatch could follow.
    inflight: int | None = None
    budget: int | None = None
    admission_full: bool = False
    notices: list[RecoveryNotice] = field(default_factory=list)

    def record(self, result: _CandidateResult) -> None:
        field_name = {
            _Outcome.DISPATCHED: "dispatched",
            _Outcome.GAVE_UP: "gave_up",
            _Outcome.STALE: "stale",
            _Outcome.RESCHEDULED: "rescheduled",
            _Outcome.CONFIRMED: "confirmed",
            _Outcome.REARMED: "rearmed",
            _Outcome.RACED: "raced",
        }.get(result.outcome)
        if field_name is not None:
            setattr(self, field_name, getattr(self, field_name) + 1)
        if result.staged:
            self.staged_commands += 1
        if result.notice is not None:
            self.notices.append(result.notice)


def _settle_candidate(
    db: Session,
    task_id: int,
    report: AutoResumeTickReport,
    work: Callable[[], _CandidateResult],
) -> _CandidateResult | None:
    """Run one candidate's work and end its transaction.

    A failure is logged and skipped so one bad row cannot stall the rest,
    except a pool timeout, which ends the tick as lease recovery does.
    ``_DispatchRefused`` propagates for the dispatch path to handle.
    """

    try:
        result = work()
        if result.writes:
            db.commit()
        else:
            db.rollback()
    except _DispatchRefused:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        increment_counter(
            "xagent.task.auto_resume.skipped", attributes={"outcome": "raced"}
        )
        result = _RACED
    except Exception as exc:
        db.rollback()
        if is_database_pool_timeout(exc):
            raise
        report.failed += 1
        increment_counter("xagent.task.auto_resume.candidate_failed")
        logger.exception(
            "component=auto-resume task_id=%s candidate failed; retrying next tick",
            task_id,
        )
        return None
    report.record(result)
    return result


def housekeep_dispatched(
    now: datetime, report: AutoResumeTickReport, *, limit: int, enabled: bool = True
) -> None:
    """Check up to ``limit`` dispatched rows whose check time has come."""

    for db, task_id in _candidates(_DISPATCHED, now=now, limit=limit):
        _settle_candidate(
            db,
            task_id,
            report,
            lambda: _housekeep_candidate_no_commit(
                db, task_id, now=now, enabled=enabled
            ),
        )


def drain_disabled_rows(report: AutoResumeTickReport, *, limit: int) -> None:
    """Stop up to ``limit`` scheduled rows, due or not, with the switch off."""

    for db, task_id in _candidates(_SCHEDULED, now=None, limit=limit):
        _settle_candidate(
            db, task_id, report, lambda: _drain_candidate_no_commit(db, task_id)
        )


def has_due_work(db: Session, now: datetime, *, enabled: bool = True) -> bool:
    """One ``EXISTS`` over the due index: is there anything for a tick to do?

    Due ``dispatched`` rows need housekeeping either way; ``scheduled`` rows
    need dispatching once due, or draining (due or not) with the switch off.
    """

    scheduled = TaskAutoRecovery.state == _SCHEDULED.value
    if enabled:
        scheduled = and_(scheduled, TaskAutoRecovery.next_attempt_at <= now)
    return bool(
        db.execute(
            select(
                select(TaskAutoRecovery.task_id)
                .where(
                    or_(
                        scheduled,
                        and_(
                            TaskAutoRecovery.state == _DISPATCHED.value,
                            TaskAutoRecovery.next_attempt_at <= now,
                        ),
                    )
                )
                .exists()
            )
        ).scalar()
    )


def count_inflight(db: Session) -> int:
    """Automatic resumes in flight across every process.

    A: auto-resumed runs still RUNNING (driven from the task status index);
    bounded by the running tasks, and a row leaves it when its run stops. A
    confirmed row stays ``dispatched``, so a run some person paused and
    resumed by hand since -- a command other than the sweeper's after the
    dispatch -- is not counted: it is not an automatic resume any more.
    B: dispatches whose command is still queued (pending claim, waiting for
    admission or deferred) at an unchanged fence. A row whose command has
    ended stops counting even before housekeeping closes it, so B cannot
    grow with unconfirmed rows. The two never overlap: a running resumed run
    has moved past the fence version.
    """

    dispatch = aliased(TaskExecutionCommand)
    later = aliased(TaskExecutionCommand)
    person_acted = exists().where(
        later.task_id == Task.id,
        later.id
        > select(dispatch.id)
        .where(
            dispatch.task_id == Task.id,
            dispatch.command_id == TaskAutoRecovery.last_command_id,
        )
        # Nested in ``exists()``, so it does not correlate on its own; without
        # this it reads every task's dispatch and PostgreSQL rejects the rows.
        .correlate(Task, TaskAutoRecovery)
        .scalar_subquery(),
        # Exact, case-sensitive prefix, as the transport reserves it.
        ~and_(
            later.kind == "resume",
            func.substr(later.command_id, 1, len(AUTO_RESUME_COMMAND_PREFIX))
            == AUTO_RESUME_COMMAND_PREFIX,
        ),
    )
    running = db.execute(
        select(func.count())
        .select_from(Task)
        .join(
            TaskAutoRecovery,
            and_(
                TaskAutoRecovery.task_id == Task.id,
                TaskAutoRecovery.run_id == Task.run_id,
            ),
        )
        .where(
            task_status_predicate.eq(TaskStatus.RUNNING),
            TaskAutoRecovery.state == _DISPATCHED.value,
            ~person_acted,
        )
    ).scalar_one()
    queued = db.execute(
        select(func.count())
        .select_from(TaskAutoRecovery)
        .join(Task, Task.id == TaskAutoRecovery.task_id)
        .join(
            TaskExecutionCommand,
            and_(
                TaskExecutionCommand.task_id == TaskAutoRecovery.task_id,
                TaskExecutionCommand.command_id == TaskAutoRecovery.last_command_id,
            ),
        )
        .where(
            TaskAutoRecovery.state == _DISPATCHED.value,
            Task.run_id == TaskAutoRecovery.run_id,
            Task.state_version == TaskAutoRecovery.paused_state_version,
            TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
        )
    ).scalar_one()
    return int(running) + int(queued)


def scan_due(
    now: datetime,
    report: AutoResumeTickReport,
    *,
    dispatch_budget: int,
    scan_limit: int,
) -> None:
    """Dispatch or close up to ``scan_limit`` due scheduled rows.

    Once the budget is spent, a row that needs a dispatch is left unwritten
    and the scan goes on, so rows behind it can still expire, go stale or be
    superseded. A row whose staging is refused is postponed on its own
    (``_DispatchRefused``) and the scan goes on too: a host may bucket
    admission per team, so one full bucket says nothing about the next row.
    A team with more due rows than ``scan_limit`` ahead of everyone else still
    uses up the scan, so another team's row waits at most
    ``ceil(rows / scan_limit)`` ticks plus the one that dispatches it; each
    further due row of a full bucket costs one refused staging. A refusal
    that is not ``ValueError`` or ``RuntimeError`` counts as a failed
    candidate (ERROR, ``candidate_failed``) while the row is still postponed.
    A postpone that fails on a database outage or pool timeout ends the tick;
    any other postpone failure is logged and the scan goes on.
    """

    for db, task_id in _candidates(_SCHEDULED, now=now, limit=scan_limit):
        can_dispatch = report.dispatched < dispatch_budget
        try:
            result = _settle_candidate(
                db,
                task_id,
                report,
                lambda: _process_due_candidate_no_commit(
                    db, task_id, now=now, can_dispatch=can_dispatch
                ),
            )
        except _DispatchRefused as refused:
            if isinstance(refused, _AdmissionFull):
                report.admission_full = True
                increment_counter(
                    "xagent.task.auto_resume.skipped",
                    attributes={"outcome": "admission_full"},
                )
                logger.info(
                    "component=auto-resume task_id=%s admission queue is full; "
                    "postponing %ss",
                    refused.task_id,
                    refused.delay_seconds,
                )
            elif isinstance(refused.__cause__, _EXPECTED_STAGING_REFUSALS):
                increment_counter(
                    "xagent.task.auto_resume.skipped",
                    attributes={"outcome": "staging_refused"},
                )
                logger.warning(
                    "component=auto-resume task_id=%s staging the resume was "
                    "refused; postponing %ss",
                    refused.task_id,
                    refused.delay_seconds,
                    exc_info=refused.__cause__,
                )
            else:
                report.failed += 1
                increment_counter("xagent.task.auto_resume.candidate_failed")
                logger.error(
                    "component=auto-resume task_id=%s candidate failed; postponing %ss",
                    refused.task_id,
                    refused.delay_seconds,
                    exc_info=refused.__cause__,
                )
            try:
                _postpone_candidate(
                    refused.task_id,
                    paused_state_version=refused.paused_state_version,
                    now=now,
                    delay_seconds=refused.delay_seconds,
                )
            except Exception as exc:
                if is_database_pool_timeout(exc) or is_database_unavailable(exc):
                    raise
                logger.warning(
                    "component=auto-resume task_id=%s postponing the refused "
                    "candidate failed; it is retried next tick",
                    refused.task_id,
                    exc_info=True,
                )
            continue
        if result is not None and result.dispatch_delay_seconds is not None:
            observe_value(
                "xagent.task.auto_resume.dispatch_delay_seconds",
                max(0.0, result.dispatch_delay_seconds),
                unit="s",
            )


def run_auto_resume_tick(*, now: datetime | None = None) -> AutoResumeTickReport:
    """One sweeper pass; synchronous, for a worker thread.

    One ``EXISTS`` first: with nothing due (always, until something records
    ``scheduled`` rows) the tick ends there. With the switch off it drains
    ``scheduled`` rows and closes ended ``dispatched`` ones (see the module
    docstring). Otherwise housekeeping runs first (it frees in-flight
    slots), then due rows are dispatched within
    ``min(MAX_PER_TICK, MAX_INFLIGHT - inflight)``. The dispatcher is
    notified once, after the dispatches have committed, even if the tick
    then fails.
    """

    global _last_inflight
    from ..models.database import get_session_local

    report = AutoResumeTickReport()
    now = _utc(now or utc_now())
    max_per_tick = get_task_auto_resume_max_per_tick()
    enabled = get_task_auto_resume_enabled()
    try:
        if not enabled:
            # Nothing is dispatched while off; queued commands meet the guard.
            _last_inflight = 0
        with get_session_local()() as db:
            if not has_due_work(db, now, enabled=enabled):
                return report
        if not enabled:
            drain_disabled_rows(report, limit=max_per_tick)
            housekeep_dispatched(now, report, limit=max_per_tick, enabled=False)
            return report
        housekeep_dispatched(now, report, limit=max_per_tick)
        with get_session_local()() as db:
            inflight = count_inflight(db)
        _last_inflight = inflight
        budget = max(
            0, min(max_per_tick, get_task_auto_resume_max_inflight() - inflight)
        )
        report.inflight, report.budget = inflight, budget
        if budget == 0:
            increment_counter(
                "xagent.task.auto_resume.skipped",
                attributes={"outcome": "inflight_cap"},
            )
        scan_due(now, report, dispatch_budget=budget, scan_limit=4 * max_per_tick)
        return report
    except Exception as exc:
        # Candidates before the failure committed; the loop still announces
        # them (see ``committed_tick_report``).
        exc.__dict__[_PARTIAL_REPORT_ATTR] = report
        raise
    finally:
        if report.staged_commands:
            notify_task_command_dispatcher()


_PARTIAL_REPORT_ATTR = "auto_resume_partial_report"


def committed_tick_report(exc: BaseException) -> AutoResumeTickReport | None:
    """What a tick that raised ``exc`` had committed before it failed."""

    report = getattr(exc, _PARTIAL_REPORT_ATTR, None)
    return report if isinstance(report, AutoResumeTickReport) else None


def _clock() -> float:
    return time.monotonic()


async def run_task_auto_resume_loop(*, poll_interval_seconds: int) -> None:
    """Run sweeper ticks until cancelled.

    A failed tick backs off exponentially (capped at 60s), is logged on the
    first failure and then at most once a minute, and after three in a row
    raises TASK_AUTO_RESUME_UNAVAILABLE until a tick succeeds.
    """

    register_observable_gauge(
        "xagent.task.auto_resume.inflight",
        lambda: float(_last_inflight),
        unit="{task}",
        description="Automatic resumes in flight, as of this process's last count",
    )
    failures = 0
    last_failure_log = float("-inf")
    while True:
        try:
            report = await run_db_io_cancellation_safe(
                lambda: run_auto_resume_tick(now=utc_now())
            )
            await publish_recovery_notices(report.notices)
            if failures:
                logger.info(
                    "Auto-resume sweeper recovered after %s failed tick(s)", failures
                )
            failures = 0
            clear_degradation(TASK_AUTO_RESUME_UNAVAILABLE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            partial = committed_tick_report(exc)
            if partial is not None:
                await publish_recovery_notices(partial.notices)
            failures += 1
            increment_counter("xagent.task.auto_resume.tick_failed")
            now_monotonic = _clock()
            if (
                failures == 1
                or now_monotonic - last_failure_log >= _FAILURE_LOG_INTERVAL_SECONDS
            ):
                last_failure_log = now_monotonic
                if is_database_pool_timeout(exc):
                    logger.warning(
                        "Auto-resume tick skipped after database pool timeout "
                        "(failures=%s)",
                        failures,
                    )
                else:
                    logger.warning(
                        "Auto-resume tick failed (failures=%s)",
                        failures,
                        exc_info=True,
                    )
            if failures >= _UNAVAILABLE_AFTER_FAILURES:
                register_degradation(
                    TASK_AUTO_RESUME_UNAVAILABLE,
                    f"{failures} consecutive failed auto-resume ticks",
                )
        delay = (
            poll_interval_seconds
            if failures == 0
            else min(
                poll_interval_seconds
                * int(2 ** min(failures, _MAX_FAILURE_BACKOFF_EXPONENT)),
                _MAX_FAILURE_BACKOFF_SECONDS,
            )
        )
        await asyncio.sleep(delay)


# --------------------------------------------------------------------------
# Claim-time guard
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoResumeClaimDecision:
    """Whether ``resume_task`` may go on with an automatic RESUME."""

    proceed: bool
    why: str | None = None
    notices: tuple[RecoveryNotice, ...] = ()


_PROCEED = AutoResumeClaimDecision(proceed=True)


def _valid_auto_resume_payload(auto_resume: Any) -> bool:
    if not isinstance(auto_resume, dict):
        return False
    run_id = auto_resume.get("expected_run_id")
    version = auto_resume.get("expected_state_version")
    return (
        isinstance(run_id, str)
        and bool(run_id)
        and isinstance(version, int)
        and not isinstance(version, bool)
    )


def check_auto_resume_claim_sync(
    *,
    task_id: int,
    command_id: Any,
    auto_resume: Any,
    status: Any,
    control_state: Any,
    run_id: str | None,
    state_version: int | None,
    attempt_count: Any,
) -> AutoResumeClaimDecision:
    """Decide, at claim, whether an automatic RESUME still applies.

    ``status`` .. ``state_version`` are the snapshot ``resume_task`` admits
    from; its RESUME_REQUESTED transition is fenced on the same version, so
    a row that moves after this check makes the transition defer and the
    retry lands here again.

    - Not the sweeper's command (no reserved id) or a malformed payload:
      ``ValueError``, which ``resume_task`` rejects as an invalid payload.
    - The recovery row no longer rests in ``dispatched`` naming this command:
      skip (``not_current``) and leave the row to whatever owns it now.
    - This command's own earlier attempt already wrote RESUME_REQUESTED and
      died: proceed, or the task would stay ``resume_requested``. Commands
      on a task are serialized, so only that attempt can have written it
      (exactly one version past the fence), whatever the switch says now.
    - At the fence with the switch on: proceed.
    - Otherwise skip: ``auto_resume_disabled`` at the fence (the row rests
      in ``manual`` and a RUNNING TriggerRun fails), else ``stale`` (a
      person moved the task on; WAITING_FOR_USER and PAUSE_REQUESTED land
      here) -- except a retry after this command's own resume was rolled
      back, which is scheduled again (``rolled_back``).
    """

    if not is_auto_resume_command_id(command_id) or not _valid_auto_resume_payload(
        auto_resume
    ):
        raise ValueError("invalid auto_resume payload")
    expected_run_id = str(auto_resume["expected_run_id"])
    expected_version = int(auto_resume["expected_state_version"])
    task_status = TaskStatus(status)
    version = int(state_version or 0)
    retry = int(attempt_count or 0) > 1
    own_retry = (
        retry
        and task_status is TaskStatus.PAUSED
        and control_state == TaskControlState.RESUME_REQUESTED.value
        and run_id == expected_run_id
        and version == expected_version + 1
    )
    at_fence = (
        task_status is TaskStatus.PAUSED
        and control_state == TaskControlState.PAUSED.value
        and run_id == expected_run_id
        and version == expected_version
    )
    proceed = own_retry or (at_fence and get_task_auto_resume_enabled())
    why = None if proceed else AUTO_RESUME_DISABLED_DETAIL if at_fence else "stale"
    decision = _settle_claim(
        task_id=task_id, command_id=command_id, why=why, retry=retry
    )
    if not decision.proceed:
        logger.info(
            "auto resume skipped task_id=%s run_id=%s why=%s component=auto-resume",
            task_id,
            expected_run_id,
            decision.why,
        )
    return decision


def _settle_claim(
    *, task_id: int, command_id: str, why: str | None, retry: bool
) -> AutoResumeClaimDecision:
    """Check the recovery row behind a claim; close it when the claim skips.

    Locks in the shared order (tasks, then the recovery row) only on the
    paths that write.
    """

    from ..models.database import get_session_local

    with get_session_local()() as db:
        try:
            if why is not None and _use_postgresql_partitioning():
                db.execute(select(Task.id).where(Task.id == task_id).with_for_update())
            task, row = _load_locked_no_commit(db, task_id)
            if (
                task is None
                or row is None
                or row.state != _DISPATCHED.value
                or row.last_command_id != command_id
            ):
                db.rollback()
                return AutoResumeClaimDecision(proceed=False, why=why or "not_current")
            if why is None:
                db.rollback()
                return _PROCEED
            if why == AUTO_RESUME_DISABLED_DETAIL:
                result = _give_up_no_commit(
                    db,
                    task,
                    row,
                    TaskAutoRecoveryState.MANUAL,
                    from_state=_DISPATCHED,
                    at="claim",
                    state_detail=AUTO_RESUME_DISABLED_DETAIL,
                    stop=_DISABLED_STOP,
                )
            elif retry and _rolled_back_by_this_claim(db, task, row, command_id):
                why = "rolled_back"
                result = _refence_rolled_back_claim_no_commit(db, task, row)
            else:
                mismatch = _fence_mismatch(task, row)
                result = _give_up_no_commit(
                    db,
                    task,
                    row,
                    TaskAutoRecoveryState.STALE,
                    from_state=_DISPATCHED,
                    at="claim",
                    state_detail=f"claim:{mismatch}" if mismatch else "claim",
                    event_detail={"mismatch": mismatch},
                )
            if result.writes:
                db.commit()
            else:
                db.rollback()
        except Exception:
            db.rollback()
            raise
    return AutoResumeClaimDecision(
        proceed=False,
        why=why,
        notices=(result.notice,) if result.notice is not None else (),
    )


def _rolled_back_by_this_claim(
    db: Session, task: Task, row: TaskAutoRecovery, command_id: str
) -> bool:
    """Whether the task is back at PAUSED only because this command undid it.

    ``resume_task`` writes RESUME_REQUESTED (+1) and, if starting the resumed
    run then fails, writes PAUSED back (+1). Nothing else ran in between:
    commands are serialized and no user command reached the task since the
    dispatch.
    """

    from .task_auto_recovery import _user_command_since_dispatch

    return bool(
        task.status == TaskStatus.PAUSED
        and task.control_state == TaskControlState.PAUSED.value
        and task.run_id == row.run_id
        and int(task.state_version or 0) == int(row.paused_state_version) + 2
        and not _user_command_since_dispatch(db, int(task.id), command_id)
    )


def _refence_rolled_back_claim_no_commit(
    db: Session, task: Task, row: TaskAutoRecovery
) -> _CandidateResult:
    """Move the row to the task's current fence, then schedule it again.

    With the switch off it stops instead, as a drained row does, so turning
    the switch back on never resumes it retroactively.
    """

    updated = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, _DISPATCHED, fenced=False))
            .values(paused_state_version=int(task.state_version or 0))
            .execution_options(synchronize_session=False)
        )
    )
    if updated != 1:
        return _RACED
    db.refresh(row)
    if not get_task_auto_resume_enabled():
        return _give_up_no_commit(
            db,
            task,
            row,
            TaskAutoRecoveryState.MANUAL,
            from_state=_DISPATCHED,
            at="claim",
            state_detail=AUTO_RESUME_DISABLED_DETAIL,
            stop=_DISABLED_STOP,
        )
    return _reschedule_dispatched_no_commit(
        db, task, row, now=utc_now(), at="claim", after="resume_rolled_back"
    )


# --------------------------------------------------------------------------
# Channel reply hold
# --------------------------------------------------------------------------

# Read side only: absorbs clock skew between the host that dispatches and the
# host that reads a channel reply. Dispatch compares the same
# ``interrupted_at`` with the channel window, so a ``scheduled`` row past
# ``interrupted_at + window`` is never dispatched again.
CHANNEL_HOLD_CLOCK_GRACE_SECONDS = 60


def _hold_snapshot(db: Session, task_id: int) -> Any:
    """The task, its recovery row and the row's dispatched command, read once,
    with whether any RESUME of the task (automatic or not) is still queued or
    running.

    One statement on purpose: under PostgreSQL's READ COMMITTED two reads
    could straddle a resume that was interrupted and recorded again, pairing
    one interruption's ``state_version`` with another's
    ``paused_state_version`` and releasing a reply that should wait.
    """

    command_status = (
        select(TaskExecutionCommand.status)
        .where(
            TaskExecutionCommand.task_id == TaskAutoRecovery.task_id,
            TaskExecutionCommand.command_id == TaskAutoRecovery.last_command_id,
        )
        .correlate(TaskAutoRecovery)
        .scalar_subquery()
    )
    resume_in_flight = (
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == Task.id,
            TaskExecutionCommand.kind == TaskCommandKind.RESUME.value,
            TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
        )
        .correlate(Task)
        .exists()
    )
    return db.execute(
        select(
            Task.status,
            Task.control_state,
            Task.run_id,
            Task.state_version,
            TaskAutoRecovery.run_id.label("row_run_id"),
            TaskAutoRecovery.state.label("row_state"),
            TaskAutoRecovery.paused_state_version,
            TaskAutoRecovery.interrupted_at,
            command_status.label("dispatch_status"),
            resume_in_flight.label("resume_in_flight"),
        )
        .select_from(Task)
        .outerjoin(TaskAutoRecovery, TaskAutoRecovery.task_id == Task.id)
        .where(Task.id == task_id)
    ).one_or_none()


def _auto_resume_holds(snap: Any, *, run_id: str, now: datetime) -> bool:
    """Snapshot of a task PAUSED on ``run_id``: is an automatic resume still
    due or starting?

    A dispatched command still queued or running holds whatever the version
    (a retry after a rolled-back resume moves it by two). No deadline bounds
    that branch: the command's own liveness does (an expired claim is
    reclaimed and run; a command that fails leaves this branch, and the row
    then holds only until housekeeping closes it as ``dispatch_failed`` or
    the window ends); housekeeping only re-arms a command that is still
    queued. Every other hold ends at the channel window from the latest
    interruption, plus a clock grace: past it the sweeper can no longer
    dispatch the row.
    """

    if snap.row_run_id != run_id:
        return False
    if snap.row_state == _DISPATCHED.value and snap.dispatch_status in (
        COMMAND_PENDING,
        COMMAND_PROCESSING,
    ):
        return True
    if snap.row_state not in (
        _SCHEDULED.value,
        _DISPATCHED.value,
        TaskAutoRecoveryState.STALE.value,
    ):
        return False
    deadline = _utc(snap.interrupted_at) + timedelta(
        seconds=staleness_window_seconds("channel") + CHANNEL_HOLD_CLOCK_GRACE_SECONDS
    )
    if now > deadline:
        return False
    # A resume (automatic, or a person's from the web, which the sweeper may
    # meanwhile have closed as ``stale``) is handing off: wait for its run.
    if snap.control_state == TaskControlState.RESUME_REQUESTED.value:
        return True
    return (
        snap.row_state in (_SCHEDULED.value, _DISPATCHED.value)
        and snap.control_state == TaskControlState.PAUSED.value
        and int(snap.state_version or 0) == int(snap.paused_state_version)
    )


def channel_reply_waits_for_auto_resume(
    db: Session, *, task_id: int, run_id: str, now: datetime
) -> bool:
    """Whether a channel reply for ``run_id`` waits for an automatic resume.

    The caller (``_read_channel_result``) replies "interrupted" on False and
    reads again later on True. False when one consistent read shows the task
    PAUSED on ``run_id`` with no automatic resume pending or starting, and
    when the task is gone (deleted since the caller read it), so that reply
    is "interrupted" too. A task that moved otherwise since the caller read
    it is True: the next read's other branches decide.
    """

    snap = _hold_snapshot(db, task_id)
    if snap is None:
        return False
    if snap.status != TaskStatus.PAUSED or snap.run_id != run_id:
        return True
    return _auto_resume_holds(snap, run_id=run_id, now=now)


def resume_hand_off_can_complete(db: Session, task: Task, *, now: datetime) -> bool:
    """Whether the resume ``task`` is handing off (PAUSED, ``resume_requested``)
    can still start its run, so a new channel turn or a status-only append
    should wait (busy).

    True while a RESUME command of the task (automatic or a person's) is still
    queued or running, or while an automatic resume of its run still holds
    the channel reply (``_auto_resume_holds``: within the channel window and
    its grace). Otherwise the hand-off is stranded -- its command ended but
    the resumed run never took the lease -- and refusing every turn would
    strand the channel or the status-only caller with it. One read, no locks; the caller holds the task
    row lock.
    """

    snap = _hold_snapshot(db, int(task.id))
    if snap is None:
        return False
    if snap.resume_in_flight:
        return True
    return snap.run_id is not None and _auto_resume_holds(
        snap, run_id=str(snap.run_id), now=now
    )


def supersede_pending_auto_resume_no_commit(
    db: Session, task: Task, *, now: datetime
) -> bool:
    """A new turn accepted on ``task`` replaces the automatic resume its run
    waits for; no commit.

    Caller: channel acceptance, holding the task row lock (FOR UPDATE) after
    reserving the START in this transaction, before reading the version the
    START is staged at. Lock order is the module's: ``tasks``, then the
    recovery row. Acts only while the resume has not begun its hand-off
    (``control_state='paused'``). It moves ``state_version`` so that the
    RESUME_REQUESTED transition of an already-claimed automatic RESUME, fenced
    on the version it admitted from, can no longer succeed; it defers, and
    the claim guard skips the retry (``stale``: its snapshot is past the
    fence, and the row no longer names the command). Marking the row ``stale`` alone
    would let that transition resume the old run and reject the new START.
    """

    if (
        task.status != TaskStatus.PAUSED
        or task.control_state != TaskControlState.PAUSED.value
        or task.run_id is None
    ):
        return False
    snap = _hold_snapshot(db, int(task.id))
    if snap is None or not _auto_resume_holds(snap, run_id=str(task.run_id), now=now):
        return False
    row = db.get(
        TaskAutoRecovery, int(task.id), with_for_update=True, populate_existing=True
    )
    if row is None or row.state not in (_SCHEDULED.value, _DISPATCHED.value):
        return False
    bumped = _rowcount(
        db.execute(
            update(Task)
            .where(
                Task.id == task.id,
                Task.run_id == task.run_id,
                Task.state_version == task.state_version,
                task_status_predicate.eq(TaskStatus.PAUSED),
                Task.control_state == TaskControlState.PAUSED.value,
            )
            .values(state_version=func.coalesce(Task.state_version, 0) + 1)
            .execution_options(synchronize_session=False)
        )
    )
    if bumped != 1:
        # Not under the caller's lock; leave the resume alone.
        return False
    _give_up_no_commit(
        db,
        task,
        row,
        TaskAutoRecoveryState.STALE,
        from_state=TaskAutoRecoveryState(row.state),
        at="new_turn",
        state_detail="new_turn",
        event_detail={"mismatch": "new_turn"},
        fenced=False,
    )
    increment_counter("xagent.task.auto_resume.superseded_by_turn")
    return True


# --------------------------------------------------------------------------
# Cancel by a person
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoResumeCancel:
    """What :func:`cancel_pending_auto_resume_no_commit` did."""

    cancelled: bool
    # Lost a compare-and-swap; nothing was written, so the caller may read
    # again and retry.
    raced: bool = False
    run_id: str | None = None
    # The task's version after the cancel; also the row's new fence.
    state_version: int | None = None
    notice: RecoveryNotice | None = None


_NOTHING_TO_CANCEL = AutoResumeCancel(cancelled=False)
_CANCEL_RACED = AutoResumeCancel(cancelled=False, raced=True)


def _auto_resume_pending(db: Session, task: Task, row: TaskAutoRecovery | None) -> bool:
    """Whether an automatic resume of ``task``'s paused run may still start.

    At the fence a ``scheduled`` row will be dispatched and a ``dispatched``
    one is queued, running, or about to be rescheduled by housekeeping
    (whatever its command's status). Off the fence only one dispatch can
    still resume the run: a retry after the resume this command rolled back
    (:func:`_rolled_back_by_this_claim`: two versions on, no other command
    since), while the command is queued or running. A missing or ended
    command there means nothing is pending: housekeeping or the guard closes
    the row without resuming. A hand-off in progress (``resume_requested``)
    is past cancelling.
    """

    if (
        row is None
        or task.status != TaskStatus.PAUSED
        or task.control_state != TaskControlState.PAUSED.value
        or task.run_id is None
        or row.run_id != task.run_id
        or row.state not in (_SCHEDULED.value, _DISPATCHED.value)
    ):
        return False
    if int(task.state_version or 0) == int(row.paused_state_version):
        return True
    if row.state != _DISPATCHED.value or row.last_command_id is None:
        return False
    status = db.execute(
        select(TaskExecutionCommand.status).where(
            TaskExecutionCommand.task_id == task.id,
            TaskExecutionCommand.command_id == row.last_command_id,
        )
    ).scalar_one_or_none()
    return status in (COMMAND_PENDING, COMMAND_PROCESSING) and (
        _rolled_back_by_this_claim(db, task, row, str(row.last_command_id))
    )


def supersede_pending_auto_resume_for_append_no_commit(db: Session, task: Task) -> bool:
    """A status-only caller's append, accepted on ``task``, replaces the
    automatic resume its paused run waits for; no commit.

    Caller: ``_accept_turn_no_commit`` for a status-only source in shared
    execution, holding the ``tasks`` row lock, with ``task`` re-read after it
    and the START already reserved in this transaction, before reading the
    version the START is staged at. Lock order is the module's: ``tasks``,
    then the recovery row.

    Acts only while :func:`_auto_resume_pending` says an automatic resume of
    the run may still start; otherwise nothing is written and ``False`` is
    returned. It moves ``state_version`` and the row's fence to the new version
    in one step, as :func:`cancel_pending_auto_resume_no_commit` does, so an
    automatic RESUME already queued or claimed (fenced on the version it
    admitted from) defers and its claim guard skips it. The row rests in
    ``manual(new_turn)`` at the new fence: unlike a person's cancel this is no
    user intent, and nothing schedules a ``manual`` row again at its fence.

    A compare-and-swap that misses means the task or the row moved since it was
    read (the caller's lock prevents that on PostgreSQL; SQLite ignores row
    locks); it raises ``TaskTurnError("busy")`` so the whole acceptance rolls
    back and nothing is accepted.
    """

    from .task_orchestrator import TaskTurnError

    row = db.get(
        TaskAutoRecovery, int(task.id), with_for_update=True, populate_existing=True
    )
    if not _auto_resume_pending(db, task, row):
        return False
    assert row is not None
    from_state = TaskAutoRecoveryState(row.state)
    task_version = int(task.state_version or 0)
    row_as_read = select(TaskAutoRecovery.task_id).where(
        *_row_cas(row, from_state, fenced=False)
    )
    bumped = _rowcount(
        db.execute(
            update(Task)
            .where(
                Task.id == task.id,
                Task.run_id == task.run_id,
                Task.state_version == task.state_version,
                task_status_predicate.eq(TaskStatus.PAUSED),
                Task.control_state == TaskControlState.PAUSED.value,
                row_as_read.exists(),
            )
            .values(state_version=func.coalesce(Task.state_version, 0) + 1)
            .execution_options(synchronize_session=False)
        )
    )
    if bumped != 1:
        raise TaskTurnError("busy")
    moved = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, from_state, fenced=False))
            .values(paused_state_version=task_version + 1)
            .execution_options(synchronize_session=False)
        )
    )
    if moved != 1:
        raise TaskTurnError("busy")
    db.refresh(task)
    db.refresh(row)
    result = _give_up_no_commit(
        db,
        task,
        row,
        TaskAutoRecoveryState.MANUAL,
        from_state=from_state,
        at="sdk_append",
        state_detail="new_turn",
        # The event type is ``stale``, like the channel supersede's, so one
        # timeline query (``stale`` with ``mismatch = new_turn``) covers both;
        # the row's own state says why it rests in ``manual``. The event's
        # detail already carries the row's state and state_detail.
        event_detail={"mismatch": "new_turn"},
        stop=(TaskRecoveryEventType.STALE, None),
        fenced=True,
        log_level=logging.INFO,
        counter="xagent.task.auto_resume.superseded_by_turn",
    )
    if result.outcome is not _Outcome.GAVE_UP:
        raise TaskTurnError("busy")
    return True


def cancel_pending_auto_resume_no_commit(
    db: Session, task: Task, *, at: str, actor_user_id: int | None = None
) -> AutoResumeCancel:
    """A person stops the automatic resume ``task``'s paused run waits for;
    no commit.

    Caller: holds the ``tasks`` row lock, and ``task`` reflects that locked
    read (selected ``FOR UPDATE``, or refreshed after it); lock order is the
    module's, ``tasks`` then the recovery row, then reads of commands.
    ``at`` names the caller (e.g. ``channel_stop``) in the event, the log
    and the metrics; ``actor_user_id``, when given, the person who cancelled
    (an admin may cancel another user's task), in the event.

    ``raced``: the first write's compare-and-swap, which also re-checks the
    recovery row as read, lost (SQLite ignores the row locks), and nothing
    was written; the caller may read again and retry.
    Once it lands nothing else can move either row -- the locks on
    PostgreSQL, the write lock it took on SQLite -- so a later write that
    misses raises instead. On any exception the caller rolls back.

    Moves ``state_version`` so that an automatic RESUME already claimed --
    even one past the claim guard -- cannot make its RESUME_REQUESTED
    transition, fenced on the version it admitted from: it defers, and the
    guard skips the retry (the row no longer names it), completing it as
    ``auto_skipped``. The row moves to the new version (its fence) and rests
    in ``manual(user_cancelled)``, announced and read as ``stopped`` /
    ``cancelled``; nothing schedules a ``manual`` row again at that fence.
    A TriggerRun stays RUNNING, as after a person's pause (#2177): the run
    that is later resumed by hand settles it. Queued admissions are left
    alone: a RESUME waiting in one completes as ``auto_skipped`` once
    admitted.
    """

    row = db.get(
        TaskAutoRecovery, int(task.id), with_for_update=True, populate_existing=True
    )
    if not _auto_resume_pending(db, task, row):
        return _NOTHING_TO_CANCEL
    assert row is not None
    from_state = TaskAutoRecoveryState(row.state)
    task_version = int(task.state_version or 0)
    paused_state_version = int(row.paused_state_version)
    row_as_read = select(TaskAutoRecovery.task_id).where(
        *_row_cas(row, from_state, fenced=False)
    )
    bumped = _rowcount(
        db.execute(
            update(Task)
            .where(
                Task.id == task.id,
                Task.run_id == task.run_id,
                Task.state_version == task.state_version,
                task_status_predicate.eq(TaskStatus.PAUSED),
                Task.control_state == TaskControlState.PAUSED.value,
                row_as_read.exists(),
            )
            .values(state_version=func.coalesce(Task.state_version, 0) + 1)
            .execution_options(synchronize_session=False)
        )
    )
    if bumped != 1:
        return _CANCEL_RACED
    moved = _rowcount(
        db.execute(
            update(TaskAutoRecovery)
            .where(*_row_cas(row, from_state, fenced=False))
            .values(paused_state_version=task_version + 1)
            .execution_options(synchronize_session=False)
        )
    )
    if moved != 1:
        raise RuntimeError(f"auto-resume row of task {task.id} moved under a cancel")
    db.refresh(task)
    db.refresh(row)
    result = _give_up_no_commit(
        db,
        task,
        row,
        TaskAutoRecoveryState.MANUAL,
        from_state=from_state,
        at=at,
        state_detail=AUTO_RESUME_USER_CANCELLED_DETAIL,
        event_detail={
            "from": from_state.value,
            "task_version": task_version,
            "paused_state_version": paused_state_version,
            **({} if actor_user_id is None else {"actor_user_id": actor_user_id}),
        },
        stop=_CANCELLED_STOP,
        fenced=True,
        log_level=logging.INFO,
        counter="xagent.task.auto_resume.cancelled",
    )
    if result.outcome is not _Outcome.GAVE_UP:
        raise RuntimeError(f"auto-resume row of task {task.id} moved under a cancel")
    return AutoResumeCancel(
        cancelled=True,
        run_id=str(row.run_id),
        state_version=task_version + 1,
        notice=result.notice,
    )


# --------------------------------------------------------------------------
# Terminal view for status-only callers
# --------------------------------------------------------------------------

# Client-visible: what a caller that only reads the task's status is told when
# automatic resume has stopped and nothing is queued that could still move the
# task. Keyed by the recovery row's state; :func:`_with_reason` completes each
# text with the row's interruption reason.
_TERMINAL_VIEW_ERRORS = {
    TaskAutoRecoveryState.EXHAUSTED.value: (
        "Task failed after a system interruption; "
        "automatic resume stopped after repeated attempts"
    ),
    TaskAutoRecoveryState.EXPIRED.value: (
        "Task failed after a system interruption; "
        "it was not resumed within the automatic-resume window"
    ),
    TaskAutoRecoveryState.DISPATCH_FAILED.value: (
        "Task failed after a system interruption; automatic resume could not be started"
    ),
    TaskAutoRecoveryState.INELIGIBLE.value: (
        "Task failed after a system interruption; "
        "it can no longer be resumed automatically"
    ),
    TaskAutoRecoveryState.SUPERSEDED.value: (
        "Task failed after a system interruption; "
        "it can no longer be resumed automatically"
    ),
}
_TERMINAL_VIEW_DISABLED_ERROR = (
    "Task failed after a system interruption; automatic resume is disabled"
)
_TERMINAL_VIEW_NEW_TURN_ERROR = "Task failed: the new message could not be started."
_TERMINAL_VIEW_MANUAL_ERROR = (
    "Task failed after a system interruption; automatic resume will not continue it"
)
# ``state_detail`` that
# :func:`supersede_pending_auto_resume_for_append_no_commit` writes on a
# ``manual`` row when a caller's new message replaces a pending resume.
_NEW_TURN_DETAIL = "new_turn"
# Row states that end automatic resume for good, whatever their detail.
_TERMINAL_VIEW_STOP_STATES = frozenset(_TERMINAL_VIEW_ERRORS)


def _with_reason(text: str, reason: Any) -> str:
    """A stop text completed with the row's interruption ``reason``."""

    return f"{text} (reason: {reason})."


# Commands of a person that can still change the paused task (a START or
# MESSAGE of the caller is not one: a queued one is an opening command, and
# one that ran moved the run).
_PERSON_CONTROL_KINDS = (
    TaskCommandKind.RESUME.value,
    TaskCommandKind.RESUME_INPUT.value,
    TaskCommandKind.PAUSE.value,
    TaskCommandKind.CANCEL.value,
)


@dataclass(frozen=True)
class TerminalView:
    """What a status-only caller is told about a task automatic resume left.

    ``state`` is the recovery row's state, ``error`` the client-visible text,
    ``stopped_at`` when the row last changed (the stop) and ``reason`` the
    row's interruption reason.
    """

    state: str
    error: str
    stopped_at: datetime
    reason: str


def _terminal_view_snapshot(db: Session, task_id: int) -> Any:
    """The task, its recovery row and the command facts, in one statement."""

    last_dispatch = (
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == TaskAutoRecovery.task_id,
            TaskExecutionCommand.command_id == TaskAutoRecovery.last_command_id,
        )
        .correlate(TaskAutoRecovery)
        .scalar_subquery()
    )
    person_control_since = (
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == Task.id,
            TaskExecutionCommand.id > last_dispatch,
            TaskExecutionCommand.kind.in_(_PERSON_CONTROL_KINDS),
            # Exact, case-sensitive prefix, as ``_user_command_since_dispatch``
            # tests it: ``startswith`` compiles to LIKE, which ignores ASCII
            # case on SQLite, so a person's ``AUTO-RESUME:...`` command would
            # pass for the sweeper's.
            func.substr(
                TaskExecutionCommand.command_id, 1, len(AUTO_RESUME_COMMAND_PREFIX)
            )
            != AUTO_RESUME_COMMAND_PREFIX,
        )
        .correlate(Task, TaskAutoRecovery)
        .exists()
    )
    opening_queued = (
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == Task.id,
            TaskExecutionCommand.kind.in_(EXECUTION_OPENING_KINDS),
            TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
        )
        .correlate(Task)
        .exists()
    )
    start_rejected = (
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == Task.id,
            TaskExecutionCommand.kind == TaskCommandKind.START.value,
            TaskExecutionCommand.status == COMMAND_FAILED,
            TaskExecutionCommand.target_state_version == Task.state_version,
        )
        .correlate(Task)
        .exists()
    )
    return db.execute(
        select(
            Task.status,
            Task.control_state,
            Task.run_id,
            Task.state_version,
            Task.source,
            TaskAutoRecovery.run_id.label("row_run_id"),
            TaskAutoRecovery.state.label("row_state"),
            TaskAutoRecovery.state_detail,
            TaskAutoRecovery.reason,
            TaskAutoRecovery.paused_state_version,
            TaskAutoRecovery.updated_at,
            last_dispatch.is_not(None).label("last_dispatch_exists"),
            start_rejected.label("start_rejected"),
            opening_queued.label("opening_queued"),
            person_control_since.label("person_control_since"),
        )
        .select_from(Task)
        .outerjoin(TaskAutoRecovery, TaskAutoRecovery.task_id == Task.id)
        .where(Task.id == task_id)
    ).one_or_none()


def status_only_terminal_view(db: Session, task_id: int) -> TerminalView | None:
    """Whether a status-only caller should read ``task_id`` as failed, and why.

    A read model for the public ``/v1`` task reads only: its one caller is
    ``_load_task_info_snapshot``, which calls it only for a task it loaded
    PAUSED with control state ``paused``; it is never written anywhere, and
    the task row stays PAUSED. A caller
    that sees nothing but the status cannot tell a task waiting for its next
    automatic resume from one that stopped for good, so it would wait on the
    second forever. The answer is a :class:`TerminalView` when all hold:

    1. the task's source is a status-only one;
    2. the task is PAUSED with control state ``paused``;
    3. the recovery row describes the task's current run;
    4. the row's reason is not a person's pause, and the row either ended
       automatic resume (``exhausted``, ``expired``, ``dispatch_failed``,
       ``ineligible``, ``superseded``) or rests in ``manual`` for a reason
       other than a person's own cancel (which is the person's intent);
    5. the task is at the row's fence, or two versions on (a resume that
       rolled back) when the row is ``dispatch_failed`` or ``manual``, its
       last dispatch command still exists (without it nothing can tell
       whether a person acted since), and no person has resumed, paused or
       cancelled the task since (the sweeper's own commands do not count);
    6. no command that could open an execution is queued or running (an
       append's START, a person's resume).

    Anything else, including a task without a row, is ``None``: the caller
    keeps waiting and the next read decides again.

    The text names the row's reason. When a START staged at the task's
    current ``state_version`` was rejected, the text says the new message
    could not be started, whatever the row's state; this changes the text
    only, never whether the view is terminal.

    One statement on purpose, like the channel reply hold's snapshot: under
    PostgreSQL's READ COMMITTED two reads could straddle a resume that was
    interrupted and recorded again, pairing one interruption's task version
    with another's fence. It runs in a SAVEPOINT; a failed read is logged and
    reads as ``None`` rather than as terminal, because a false ``failed``
    would end the caller's wait on a task that may still resume, while a
    missed one is corrected by the next read.
    """

    try:
        with db.begin_nested():
            snap = _terminal_view_snapshot(db, task_id)
    except Exception:
        logger.warning(
            "Reading the terminal view of task %s failed; reporting none",
            task_id,
            exc_info=True,
        )
        return None
    if (
        snap is None
        or not is_status_only_source(snap.source)
        or snap.status != TaskStatus.PAUSED
        or snap.control_state != TaskControlState.PAUSED.value
        or snap.row_run_id is None
        or snap.row_run_id != snap.run_id
        or snap.reason == InterruptionReason.USER_PAUSE.value
    ):
        return None
    state = str(snap.row_state)
    detail = snap.state_detail
    new_turn = False
    if state == TaskAutoRecoveryState.MANUAL.value:
        if detail == AUTO_RESUME_USER_CANCELLED_DETAIL:
            return None
        if detail == AUTO_RESUME_DISABLED_DETAIL:
            text = _TERMINAL_VIEW_DISABLED_ERROR
        elif detail == _NEW_TURN_DETAIL:
            text, new_turn = _TERMINAL_VIEW_MANUAL_ERROR, True
        else:
            text = _TERMINAL_VIEW_MANUAL_ERROR
    elif state in _TERMINAL_VIEW_STOP_STATES:
        text = _TERMINAL_VIEW_ERRORS[state]
    else:
        return None
    gap = int(snap.state_version or 0) - int(snap.paused_state_version)
    if gap != 0 and not (
        gap == 2
        and state
        in (
            TaskAutoRecoveryState.DISPATCH_FAILED.value,
            TaskAutoRecoveryState.MANUAL.value,
        )
        and snap.last_dispatch_exists
        and not snap.person_control_since
    ):
        return None
    if snap.opening_queued:
        return None
    error = (
        _TERMINAL_VIEW_NEW_TURN_ERROR
        if new_turn or snap.start_rejected
        else _with_reason(text, snap.reason)
    )
    return TerminalView(
        state=state,
        error=error,
        stopped_at=snap.updated_at,
        reason=str(snap.reason),
    )
