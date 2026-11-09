"""Hand unfinished runs of a draining process to its successor.

A host that replaces this process on purpose (the SaaS thin ingress Gate)
drains up to a deadline. Runs that are still executing at the deadline must
neither be cancelled nor re-run from a checkpoint: they stop at the step
boundary the run's own pause already defines, in command order, and the
successor resumes them.

This module is the engine half and adds no new execution path:

- :func:`begin_handover_sync` stages one ordinary durable PAUSE command for
  each eligible RUNNING run owned by the given runners. The lease owner
  applies it exactly as it applies a user's pause: the command completes when
  the pause is *requested*, and the run rests ``PAUSED`` at its next step
  boundary with its checkpoint intact. ``request_interrupt`` cancels a tool
  call that is in flight, so a legacy-checkpoint run is resumed only when its
  checkpoint shows no started tool call (see :func:`settle_handover_sync`).
- :func:`settle_handover_sync` then records that pause as a ``shutdown``
  interruption (``task_auto_recovery``). The resume is the existing automatic
  resume: its sweeper dispatches a RESUME command, and ``resume_task``
  re-checks it when it is claimed, so a user pause, cancel or message that
  moved ``state_version`` keeps the run paused. A host whose hold is closed
  (``release_hold``) leaves the resume command pending, where the successor
  claims it after it activates.

Only runs automatic recovery would resume are touched, and channel tasks stay
on the host's drain: a channel reader that finds the run PAUSED before its
recovery row exists would answer "interrupted" although the successor will
resume it. Runs whose user already asked for a pause, or whose handover
pause a user pause or cancel followed, are the user's and are left alone.
Nothing here cancels, times out or queues work.

Host contract, enforced where it can be:

- ``begin`` after the hold closed and before it is sealed (a sealed host
  refuses the PAUSE attempt, which would then wait forever): otherwise
  :class:`HandoverOrderError`.
- ``settle`` only while the hold is closed: a draining host whose hold is
  still open would let its own sweeper resume the run locally.
- Both are idempotent per release id, and the host calls them repeatedly
  until ``settle`` reports ``unsettled == 0``. Release ids must be unique: a
  command id is release id + task + run, so reusing one for a later release
  would find its old command.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...config import get_task_auto_resume_enabled
from ...core.agent.interruption import InterruptionReason
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_auto_recovery import TaskAutoRecovery
from ..models.task_command import TaskExecutionCommand
from .release_hold import release_hold
from .task_auto_recovery import (
    auto_recovery_eligibility,
    record_interruption_no_commit,
    resolution_progress_marker,
    resolution_tool_call_in_flight,
)
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_TERMINAL,
    HANDOVER_COMMAND_PREFIX,
    TaskCommandKind,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_execution_controller import TaskControlState
from .task_lease_service import (
    CheckpointRecoveryVerdict,
    TaskLeaseRecoveryCandidate,
    resolve_checkpoint_recovery_with_data,
)

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
    stay RUNNING and drain to the host's own budget.
    """

    staged: int = 0
    kept: int = 0
    recorded: int = 0
    unsettled: int = 0
    failed: int = 0


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
                # Exact and case-sensitive (``LIKE`` is neither on SQLite).
                func.substr(TaskExecutionCommand.command_id, 1, len(prefix)) == prefix,
                TaskExecutionCommand.kind == TaskCommandKind.PAUSE.value,
            )
            .order_by(TaskExecutionCommand.id)
        ).scalars()
    )


def begin_handover_sync(
    release_id: str, runner_ids: frozenset[str] | set[str] | list[str]
) -> HandoverReport:
    """Stage a handover PAUSE for every eligible run of ``runner_ids``.

    Safe to repeat (the host calls it while it waits): a run that already has
    this release's command is not staged twice. A run whose staging fails
    stays on the drain and the others are still staged.
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
        tasks = list(
            db.execute(
                select(Task)
                .where(Task.status == TaskStatus.RUNNING, Task.runner_id.in_(runners))
                .order_by(Task.id)
                .with_for_update()
            ).scalars()
        )
        for task in tasks:
            if task.run_id is None:
                kept += 1
                continue
            command_id = handover_command_id(release_id, int(task.id), str(task.run_id))
            savepoint = db.begin_nested()
            try:
                already = db.execute(
                    select(TaskExecutionCommand.id).where(
                        TaskExecutionCommand.task_id == task.id,
                        TaskExecutionCommand.command_id == command_id,
                    )
                ).first()
                if already is not None:
                    # This release's pause is in flight or failed; settle says which.
                    savepoint.commit()
                    continue
                if (
                    task.control_state != TaskControlState.RUNNING.value
                    or task.channel_id is not None
                    or not auto_recovery_eligibility(db, task).eligible
                ):
                    # A pause (the user's) already moves it, or it is not a run
                    # the successor would resume: it keeps draining.
                    savepoint.commit()
                    kept += 1
                    continue
                result = stage_task_command(
                    db,
                    task_id=int(task.id),
                    actor_user_id=cast(int | None, task.user_id),
                    command_id=command_id,
                    kind=TaskCommandKind.PAUSE,
                    payload={},
                    target_run_id=str(task.run_id),
                    handover=True,
                )
                savepoint.commit()
                staged += int(result.created)
            except Exception:
                savepoint.rollback()
                logger.exception(
                    "Staging the handover pause of task %s failed", task.id
                )
                kept += 1
        if staged:
            db.commit()
        else:
            db.rollback()
    if staged:
        notify_task_command_dispatcher()
    return HandoverReport(staged=staged, kept=kept)


def _user_pause_or_cancel_since(db: Session, command: TaskExecutionCommand) -> bool:
    """A user's PAUSE or CANCEL reached the task after the handover pause.

    Another release's handover pause cannot follow this one here: ``begin``
    stages only for a run still RUNNING, which this one is not until resumed.
    """
    return (
        db.execute(
            select(TaskExecutionCommand.id)
            .where(
                TaskExecutionCommand.task_id == command.task_id,
                TaskExecutionCommand.id > command.id,
                TaskExecutionCommand.kind.in_(
                    (TaskCommandKind.PAUSE.value, TaskCommandKind.CANCEL.value)
                ),
            )
            .limit(1)
        ).first()
        is not None
    )


def _candidate(task: Task, now: datetime) -> TaskLeaseRecoveryCandidate:
    return TaskLeaseRecoveryCandidate(
        task_id=int(task.id),
        runner_id=str(task.runner_id) if task.runner_id is not None else None,
        run_id=str(task.run_id),
        lease_expires_at=now,
        attempt_id=getattr(task, "lease_attempt_id", None),
        state_version=int(task.state_version or 0),
        last_checkpoint_event_id=(
            str(task.last_checkpoint_event_id)
            if task.last_checkpoint_event_id is not None
            else None
        ),
        last_checkpoint_trace_event_id=(
            int(task.last_checkpoint_trace_event_id)
            if task.last_checkpoint_trace_event_id is not None
            else None
        ),
        control_state=cast(str | None, task.control_state),
    )


def settle_handover_sync(release_id: str) -> HandoverReport:
    """Record each run that reached its safe point so the successor resumes it.

    Returns how many pauses were recorded now and how many are still pending.
    A run that moved on since (its run changed, or its user resumed, paused or
    cancelled it) is not recorded: it is no longer the handover's. A run whose
    checkpoint cannot be read this round is pending; one with no resumable
    checkpoint is left paused, and one whose legacy checkpoint shows a started
    tool call rests as ``manual(tool_in_flight)``, as after a crash.
    Idempotent: a pause already recorded at the task's ``state_version`` is
    skipped, while a run handed over again in a later release is recorded
    again.
    """

    if not release_hold.closed:
        raise HandoverOrderError("settle_handover needs the release hold closed")
    recorded = 0
    unsettled = 0
    failed = 0
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
                    # would leave it to lease recovery, which records a user pause.
                    unsettled += 1
                continue
            if (
                task.status != TaskStatus.PAUSED
                or task.control_state != TaskControlState.PAUSED.value
                or _user_pause_or_cancel_since(db, command)
            ):
                continue
            row = db.get(TaskAutoRecovery, task.id)
            if (
                row is not None
                and row.run_id == task.run_id
                and row.reason == InterruptionReason.SHUTDOWN.value
                and int(row.paused_state_version) == int(task.state_version or 0)
            ):
                continue
            now = datetime.now(timezone.utc)
            savepoint = db.begin_nested()
            try:
                resolution = resolve_checkpoint_recovery_with_data(
                    db, _candidate(task, now)
                )
                if resolution.verdict is not CheckpointRecoveryVerdict.RECOVERABLE:
                    # No checkpoint to resume from this round (or ever): leave it
                    # paused for its user rather than schedule a resume that fails.
                    savepoint.rollback()
                    if resolution.verdict is CheckpointRecoveryVerdict.INDETERMINATE:
                        unsettled += 1
                    else:
                        logger.warning(
                            "Handover pause of task %s has no resumable checkpoint (%s)",
                            task.id,
                            resolution.verdict.value,
                        )
                    continue
                record_interruption_no_commit(
                    db,
                    task=task,
                    reason=InterruptionReason.SHUTDOWN,
                    task_status=TaskStatus.PAUSED,
                    interrupted_at=now,
                    now=now,
                    progress_marker=resolution_progress_marker(
                        db, int(task.id), resolution
                    ),
                    gated_by_settlement_switches=False,
                    tool_call_in_flight=resolution_tool_call_in_flight(resolution),
                )
            except Exception:
                savepoint.rollback()
                logger.exception(
                    "Recording the handover pause of task %s failed", task.id
                )
                unsettled += 1
                continue
            savepoint.commit()
            recorded += 1
        if recorded:
            db.commit()
        else:
            db.rollback()
    return HandoverReport(recorded=recorded, unsettled=unsettled, failed=failed)
