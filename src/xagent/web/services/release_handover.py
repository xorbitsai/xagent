"""Hand unfinished runs of a draining process to its successor.

A host that replaces this process on purpose (the SaaS thin ingress Gate)
drains up to a deadline. Runs that are still executing at the deadline are
not re-run from a checkpoint: they stop through the run's own pause, at the
step boundary it defines and in command order, and the successor resumes
them. That pause interrupts the work in flight (see below), so only a run
with nothing it cannot safely repeat is paused: a run with a tool call in
flight or unconfirmed tool effects is not handed over and keeps draining.

This module is the engine half and adds no new execution path:

- :func:`begin_handover_sync` stages one ordinary durable PAUSE command for
  each RUNNING run owned by the given runners that
  :func:`~.task_auto_recovery.shutdown_handoff_eligible` accepts (the one
  predicate for it: status-only sdk tasks are not, nor are channel tasks
  unless their host marked the reply for delivery, and they only drain) and whose checkpoint is ``RECOVERABLE`` with no tool call that may
  have started (:func:`_checkpoint_allows_handover`, the read lease recovery
  uses). A run with a tool call in flight or an unconfirmed tool effect, with
  no resumable checkpoint, or whose checkpoint cannot be read now is counted
  ``kept`` and drains, like a channel run: pausing it would cancel the tool
  and leave a ``manual`` row nobody resumes. A run in the middle of an LLM
  call with no tool started is handed over. The lease owner applies the PAUSE
  exactly as it applies a user's pause: the command completes when the pause
  is *requested*, and the run rests ``PAUSED`` at its next step boundary
  with its checkpoint intact. The pause is the run's ordinary one:
  ``request_interrupt`` cancels the LLM call (and any tool call) in flight.
  One race remains: a tool call that starts after the PAUSE was staged is
  interrupted too, and its checkpoint then rests ``PAUSED`` with a ``manual``
  row (``unknown_tool_effect``, or ``tool_in_flight`` for a legacy
  checkpoint) for its user, never resumed automatically
  (:func:`record_handover_pause_no_commit`). Closing it would need the PAUSE
  to wait for the step to finish, which is the engine's pause semantics and
  not changed here.
- The completed PAUSE command, in the reserved ``release-handover:``
  namespace, is the durable marker that the pause was the platform's
  (:func:`~.task_auto_recovery.handover_pause_owns_run`: the run has not
  left the stop the hand-over produced, judged by ``state_version``, and no
  person paused or cancelled it). Whoever then sees the run
  PAUSED records it as a ``shutdown`` interruption: this module's
  :func:`settle_handover_sync` as an accelerator, and without it lease
  recovery (a run that died while PAUSE_REQUESTED) and the auto-resume
  sweeper (:func:`~.task_auto_resume.promote_handover_pauses`). One routine,
  :func:`record_handover_pause_no_commit`, does the recording for the settle
  step and the sweeper. So an aborted release or a controller that died
  after the pause applied does not strand the run as a user pause: the
  process that is still serving it resumes it locally, and a successor
  resumes it after a real handover.
- The resume is the existing automatic resume: its sweeper dispatches a
  RESUME command, and ``resume_task`` re-checks it when it is claimed, so a
  user pause, cancel or message that moved ``state_version`` keeps the run
  paused. A host whose hold is closed (``release_hold``) leaves the resume
  command pending, where the successor claims it after it activates.

Runs whose user already asked for a pause or cancel (queued before or after
the hand-over, in command order or not) are the user's and are left alone.
Nothing here times out or queues work; the only work it cancels is what
``request_interrupt`` cancels in a run it pauses (above): an LLM call, and a
tool call only in the race above.

Host contract, enforced where it can be:

- ``begin`` after the hold closed and before it is sealed (a sealed host
  refuses the PAUSE attempt, which would then wait forever): otherwise
  :class:`HandoverOrderError`.
- ``settle`` only while the hold is closed: a draining host whose hold is
  still open would let its own sweeper resume the run locally. The hold is
  per process: in a worker pool the host must order ``begin`` and ``settle``
  against every member's hold, as these checks see only the caller's.
- Both are idempotent per release id, and the host calls them repeatedly
  until ``settle`` reports ``unsettled == 0``, then publishes
  ``HandoverReport.notices`` (``publish_recovery_notices``). Release ids must
  be unique: a command id is release id + task + run, so reusing one for a
  later release would find its old command.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import get_task_auto_resume_enabled
from ...core.agent.interruption import InterruptionReason
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_auto_recovery import TaskAutoRecovery, TaskAutoRecoveryState
from ..models.task_command import TaskExecutionCommand
from .release_hold import release_hold
from .task_auto_recovery import (
    _project_interruption_pause_no_commit,
    _settlement_candidate,
    handover_pause_owner,
    person_stop_command_clause,
    record_interruption_no_commit,
    resolution_progress_marker,
    resolution_tool_call_in_flight,
    shutdown_handoff_eligible,
    verdict_interruption_reason,
)
from .task_auto_resume import RecoveryNotice, recorded_notice
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_TERMINAL,
    HANDOVER_COMMAND_PREFIX,
    TaskCommandKind,
    command_id_has_prefix,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_execution_controller import TaskControlState
from .task_lease_service import (
    CheckpointRecoveryVerdict,
    resolve_checkpoint_recovery_with_data,
)
from .task_settlement_report import RecordedInterruption

logger = logging.getLogger(__name__)


class HandoverOrderError(RuntimeError):
    """The host called a hand-over step outside its place in a release."""


@dataclass(frozen=True)
class HandoverReport:
    """What one handover call saw; counts of runs, never of work cancelled.

    ``unsettled`` is the runs whose handover is still in flight: the command
    is not terminal, the run has not reached its step boundary, or the
    recording is not yet possible. The host keeps waiting while it is
    positive. ``kept`` are runs left to drain (not eligible, or paused by
    their user); ``failed`` are runs whose handover command failed, which
    stay RUNNING and drain to the host's own budget. ``notices`` are the
    ``task_recovery_update`` notices of what ``settle`` committed, for the
    host to publish.
    """

    staged: int = 0
    kept: int = 0
    recorded: int = 0
    unsettled: int = 0
    failed: int = 0
    notices: tuple[RecoveryNotice, ...] = ()


class RecordOutcome(Enum):
    RECORDED = "recorded"
    # The checkpoint cannot be read this round (or recording failed): retry.
    PENDING = "pending"
    # No resumable checkpoint (or an unknown tool effect): the run stays paused
    # for its user, with a ``manual`` row saying why. Committed like RECORDED.
    NOT_RECOVERABLE = "not_recoverable"
    # Recorded as ``shutdown``, but automatic resume will not take the run
    # (its window or attempts are spent): it stays paused for its user.
    STOPPED = "stopped"
    # Not (or no longer) a handed-over pause, or already recorded.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RecordResult:
    outcome: RecordOutcome
    notice: RecoveryNotice | None = None


_SKIPPED = RecordResult(RecordOutcome.SKIPPED)
# Rows that say automatic resume will not run the paused run.
_AUTO_RESUME_SPENT_STATES = (
    TaskAutoRecoveryState.EXHAUSTED.value,
    TaskAutoRecoveryState.EXPIRED.value,
)


def _release_prefix(release_id: str) -> str:
    digest = hashlib.sha256(release_id.encode()).hexdigest()[:16]
    return f"{HANDOVER_COMMAND_PREFIX}{digest}:"


def handover_command_id(release_id: str, task_id: int, run_id: str) -> str:
    """The idempotent PAUSE command id for one run of ``task_id`` in a release."""
    run = hashlib.sha256(run_id.encode()).hexdigest()[:8]
    return f"{_release_prefix(release_id)}{task_id}:{run}"


def _release_commands(db: Session, release_id: str) -> list[TaskExecutionCommand]:
    prefix = _release_prefix(release_id)
    return list(
        db.execute(
            select(TaskExecutionCommand)
            .where(
                command_id_has_prefix(TaskExecutionCommand.command_id, prefix),
                TaskExecutionCommand.kind == TaskCommandKind.PAUSE.value,
            )
            .order_by(TaskExecutionCommand.id)
        )
        .scalars()
        .all()
    )


def _person_stop_queued(db: Session, task_id: int, run_id: str) -> bool:
    """A person's PAUSE or CANCEL of the run is not yet terminal."""
    return (
        db.execute(
            select(TaskExecutionCommand.id)
            .where(
                TaskExecutionCommand.task_id == task_id,
                TaskExecutionCommand.status.not_in(COMMAND_TERMINAL),
                person_stop_command_clause(TaskExecutionCommand, run_id),
            )
            .limit(1)
        ).first()
        is not None
    )


def _checkpoint_allows_handover(db: Session, task_id: int) -> bool:
    """Whether the run's checkpoint shows it can be handed over right now.

    The read lease recovery and the settle step use
    (``resolve_checkpoint_recovery_with_data``), asked before the PAUSE is
    staged. Only a ``RECOVERABLE`` checkpoint with no tool call that may have
    started qualifies: a run in the middle of an LLM call, with no tool
    started, is recoverable and is handed over. Anything else stays with the
    drain, because pausing it would only cancel the work in flight and leave
    a ``manual`` row nobody resumes: a checkpoint to resume from does not
    exist (``NOT_RECOVERABLE``), a tool call is in flight or its effect is
    unconfirmed (``UNKNOWN_TOOL_EFFECT``, or a legacy checkpoint
    :func:`~.task_auto_recovery.resolution_tool_call_in_flight` flags), or
    the checkpoint cannot be read now (``INDETERMINATE``). A read that
    raises propagates: the caller's SAVEPOINT rolls back and the run is kept.
    Kept apart from ``shutdown_handoff_eligible``, which the orchestrator's
    shutdown path shares and which judges the task, not its checkpoint.
    """

    task = db.get(Task, task_id)
    if task is None or task.status != TaskStatus.RUNNING or task.run_id is None:
        return False
    resolution = resolve_checkpoint_recovery_with_data(db, _settlement_candidate(task))
    return (
        resolution.verdict is CheckpointRecoveryVerdict.RECOVERABLE
        and not resolution_tool_call_in_flight(resolution)
    )


def begin_handover_sync(
    release_id: str, runner_ids: frozenset[str] | set[str] | list[str]
) -> HandoverReport:
    """Stage a handover PAUSE for every eligible run of ``runner_ids``.

    Safe to repeat (the host calls it while it waits): a run that already has
    this release's command is not staged twice. A run whose staging fails
    stays on the drain and the others are still staged. A run whose user has
    a pause or cancel queued is left to that user, and so is a run whose
    checkpoint shows a tool call in flight or cannot be read (see
    :func:`_checkpoint_allows_handover`).

    The PAUSE carries the task owner as its actor, like the sweeper's RESUME:
    executing a command needs a real actor. Staging snapshots the run's
    ``state_version`` (the command's ``target_state_version``, the run being
    RUNNING under the lock taken here): the anchor the hand-over's ownership
    of the stop it produces is judged by
    (:func:`~.task_auto_recovery.handover_pause_owner`).
    """

    if not release_hold.closed or release_hold.sealed:
        raise HandoverOrderError(
            "begin_handover needs the release hold closed and not yet sealed"
        )
    runners = set(runner_ids)
    if not runners or not get_task_auto_resume_enabled():
        # Nothing could resume a paused run: leave every run to the drain.
        return HandoverReport()
    staged = 0
    kept = 0
    with get_session_local()() as db:
        # Plain values: a rolled-back SAVEPOINT expires the session's objects,
        # and a later task's row must not be reloaded outside its own try.
        rows = [
            (
                int(task.id),
                str(task.run_id) if task.run_id is not None else None,
                int(task.user_id),
                str(task.control_state),
            )
            for task in db.execute(
                select(Task)
                .where(Task.status == TaskStatus.RUNNING, Task.runner_id.in_(runners))
                .order_by(Task.id)
                .with_for_update()
            )
            .scalars()
            .all()
        ]
        for task_id, run_id, user_id, control_state in rows:
            if run_id is None:
                kept += 1
                continue
            command_id = handover_command_id(release_id, task_id, run_id)
            created = 0
            left_to_drain = False
            try:
                # Every read and the staging in one SAVEPOINT: on PostgreSQL a
                # failed statement aborts the transaction, and the others'
                # commands (and the final commit) must survive it.
                with db.begin_nested():
                    already = db.execute(
                        select(TaskExecutionCommand.id).where(
                            TaskExecutionCommand.task_id == task_id,
                            TaskExecutionCommand.command_id == command_id,
                        )
                    ).first()
                    if already is not None:
                        # This release's pause is in flight or failed; settle says
                        # which.
                        continue
                    if (
                        control_state != TaskControlState.RUNNING.value
                        or _person_stop_queued(db, task_id, run_id)
                        or not shutdown_handoff_eligible(db, task_id)
                        or not _checkpoint_allows_handover(db, task_id)
                    ):
                        # A pause (the user's) already moves it, it is not a
                        # run the successor would resume, or it has a tool
                        # call in flight (or no readable checkpoint): it
                        # keeps draining.
                        left_to_drain = True
                    else:
                        created = int(
                            stage_task_command(
                                db,
                                task_id=task_id,
                                actor_user_id=user_id,
                                command_id=command_id,
                                kind=TaskCommandKind.PAUSE,
                                payload={},
                                target_run_id=run_id,
                                handover=True,
                            ).created
                        )
            except Exception:
                logger.exception(
                    "Staging the handover pause of task %s failed", task_id
                )
                kept += 1
                continue
            staged += created
            kept += int(left_to_drain)
        if staged:
            db.commit()
        else:
            db.rollback()
    if staged:
        notify_task_command_dispatcher()
    return HandoverReport(staged=staged, kept=kept)


def record_handover_pause_no_commit(
    db: Session, task: Task, now: datetime
) -> RecordResult:
    """Record ``task``'s handed-over pause as ``shutdown``; the caller commits.

    ``task`` is the locked row. Shared by :func:`settle_handover_sync` and the
    auto-resume sweeper. Only a task resting PAUSED whose pause
    :func:`~.task_auto_recovery.handover_pause_owner` attributes to a
    hand-over, with no recovery row at its current fence yet, is recorded. A
    run with no resumable checkpoint is left paused (``NOT_RECOVERABLE``), one
    whose legacy checkpoint shows a started tool call rests as
    ``manual(tool_in_flight)``, as after a crash, one whose automatic resume
    window or attempts are spent is ``STOPPED``, and one whose checkpoint
    cannot be read now is ``PENDING``. A trigger task's TriggerRun follows the
    recording as in every settling path. ``interrupted_at`` is when the pause
    completed. Idempotent per ``(run, state_version)``; a run handed over
    again in a later release is recorded again.
    """

    if (
        task.status != TaskStatus.PAUSED
        or task.control_state != TaskControlState.PAUSED.value
        or task.run_id is None
    ):
        return _SKIPPED
    row = db.get(TaskAutoRecovery, task.id)
    if (
        row is not None
        and row.run_id == task.run_id
        and int(row.paused_state_version) == int(task.state_version or 0)
    ):
        return _SKIPPED
    owner = handover_pause_owner(db, int(task.id), str(task.run_id))
    if owner is None:
        return _SKIPPED
    savepoint = db.begin_nested()
    try:
        resolution = resolve_checkpoint_recovery_with_data(
            db, _settlement_candidate(task)
        )
        # A recoverable run is recorded for the successor; without a checkpoint
        # to resume from (or with a tool whose effect is unknown) it stays
        # paused for its user and the row says so, as after a crash. Without a
        # row the sweeper would look at it again on every tick.
        reason = verdict_interruption_reason(
            resolution.verdict,
            control_state=cast(str | None, task.control_state),
            recoverable_reason=InterruptionReason.SHUTDOWN,
            handover_pause=True,
        )
        if reason is None:
            # The checkpoint cannot be read this round: retry.
            savepoint.rollback()
            return RecordResult(RecordOutcome.PENDING)
        recorded_row = record_interruption_no_commit(
            db,
            task=task,
            reason=reason,
            task_status=TaskStatus.PAUSED,
            # The run stopped when its pause completed, not when this noticed:
            # a sweeper that finds it late must not stretch the staleness
            # window measured from it.
            interrupted_at=min(owner.completed_at or now, now),
            now=now,
            progress_marker=resolution_progress_marker(db, int(task.id), resolution),
            gated_by_settlement_switches=False,
            tool_call_in_flight=resolution_tool_call_in_flight(resolution),
        )
        if recorded_row is None:
            savepoint.rollback()
            return _SKIPPED
        recorded = RecordedInterruption.from_row(recorded_row)
        # As every settling path does: a trigger task's TriggerRun follows the
        # recording (it stays RUNNING while a resume is scheduled, else ends).
        _project_interruption_pause_no_commit(db, task, recorded)
    except Exception:
        savepoint.rollback()
        logger.exception("Recording the handover pause of task %s failed", task.id)
        return RecordResult(RecordOutcome.PENDING)
    savepoint.commit()
    if reason is not InterruptionReason.SHUTDOWN:
        outcome = RecordOutcome.NOT_RECOVERABLE
    elif recorded.state in _AUTO_RESUME_SPENT_STATES:
        outcome = RecordOutcome.STOPPED
    else:
        outcome = RecordOutcome.RECORDED
    return RecordResult(outcome, recorded_notice(recorded))


def settle_handover_sync(release_id: str) -> HandoverReport:
    """Record each run that reached its safe point so the successor resumes it.

    Returns how many pauses were recorded now and how many are still pending
    (see :func:`record_handover_pause_no_commit` for what is recorded). A run
    that moved on since (its run changed, it ended, or its user resumed,
    paused or cancelled it) is not recorded: it is no longer the handover's.
    The host publishes ``notices`` after this returns. An accelerator: the
    auto-resume sweeper records any handed-over pause this call never saw.
    """

    if not release_hold.closed:
        raise HandoverOrderError("settle_handover needs the release hold closed")
    recorded = 0
    unsettled = 0
    failed = 0
    notices: list[RecoveryNotice] = []
    wrote = False
    with get_session_local()() as db:
        for command in _release_commands(db, release_id):
            if command.status not in COMMAND_TERMINAL:
                unsettled += 1
                continue
            if command.status != COMMAND_COMPLETED:
                failed += 1
                continue
            task = db.execute(
                select(Task).where(Task.id == command.task_id).with_for_update()
            ).scalar_one_or_none()
            if task is None or task.run_id != command.target_run_id:
                continue
            if task.status == TaskStatus.RUNNING:
                if task.control_state == TaskControlState.PAUSE_REQUESTED.value:
                    # The command completes when the pause is requested; the run
                    # rests PAUSED at its next step boundary. Tearing down now
                    # would leave it to lease recovery.
                    unsettled += 1
                continue
            result = record_handover_pause_no_commit(
                db, task, datetime.now(timezone.utc)
            )
            if result.outcome is RecordOutcome.PENDING:
                unsettled += 1
            elif result.outcome is RecordOutcome.RECORDED:
                recorded += 1
                wrote = True
                if result.notice is not None:
                    notices.append(result.notice)
            elif result.outcome in (
                RecordOutcome.NOT_RECOVERABLE,
                RecordOutcome.STOPPED,
            ):
                # A committed outcome publishes its notice too (an expired or
                # exhausted row is stopped now), without counting as recorded.
                wrote = True
                if result.notice is not None:
                    notices.append(result.notice)
        if wrote:
            db.commit()
        else:
            db.rollback()
    return HandoverReport(
        recorded=recorded,
        unsettled=unsettled,
        failed=failed,
        notices=tuple(notices),
    )
