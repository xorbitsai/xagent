"""Hand unfinished runs of a draining process to its successor.

A host that replaces this process on purpose (the SaaS thin-Gate release)
drains up to a deadline. Runs that are still executing at the deadline must
neither be cancelled nor re-run from a checkpoint: they stop at the safe
point the run's own pause already defines (between steps, in command order)
and the successor resumes them.

This module is the engine half and adds no new execution path:

- :func:`begin_handover_sync` stages one ordinary durable PAUSE command for
  each eligible RUNNING run owned by the given runners. The lease owner
  applies it exactly as it applies a user's pause, so the run rests PAUSED at
  a step boundary with its checkpoint intact.
- :func:`settle_handover_sync` then records that pause as a ``shutdown``
  interruption (``task_auto_recovery``). The resume is the existing automatic
  resume: its sweeper dispatches a RESUME command, and ``resume_task``
  re-checks it when it is claimed, so a user pause, cancel or message that
  arrived meanwhile (it moves ``state_version``) keeps the run paused. A host
  that is closed (``release_hold``) leaves the resume command pending, where
  the successor claims it after it activates.

Only runs automatic recovery would resume (``auto_recovery_eligibility``) are
touched; the rest keep draining and the host's own budget decides. A run whose
user already asked for a pause, or whose pause is already queued, is the
user's and is left alone. Nothing here cancels, times out or queues work, and
every call is idempotent for the same release id.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import get_task_auto_resume_enabled
from ...core.agent.interruption import InterruptionReason
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_auto_recovery import TaskAutoRecovery
from ..models.task_command import TaskExecutionCommand
from .task_auto_recovery import (
    auto_recovery_eligibility,
    record_interruption_no_commit,
)
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_TERMINAL,
    TaskCommandKind,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_execution_controller import TaskControlState

logger = logging.getLogger(__name__)

HANDOVER_COMMAND_PREFIX = "release-handover:"


@dataclass(frozen=True)
class HandoverReport:
    """What one handover call saw; counts of runs, never of work cancelled.

    ``unsettled`` is the runs whose handover pause is still in flight (their
    command is not terminal, or its pause has not yet been recorded): the host
    keeps waiting while it is positive. ``kept`` are runs this call left to
    drain: ineligible for automatic recovery, or paused by their user.
    """

    staged: int = 0
    kept: int = 0
    recorded: int = 0
    unsettled: int = 0


def _release_key(release_id: str) -> str:
    return hashlib.sha256(release_id.encode()).hexdigest()[:20]


def handover_command_id(release_id: str, task_id: int) -> str:
    """The idempotent PAUSE command id for ``task_id`` in this release."""
    return f"{HANDOVER_COMMAND_PREFIX}{_release_key(release_id)}:{task_id}"


def _release_commands(db: Session, release_id: str) -> list[TaskExecutionCommand]:
    prefix = f"{HANDOVER_COMMAND_PREFIX}{_release_key(release_id)}:"
    return list(
        db.execute(
            select(TaskExecutionCommand).where(
                TaskExecutionCommand.command_id.like(f"{prefix}%"),
                TaskExecutionCommand.kind == TaskCommandKind.PAUSE.value,
            )
        ).scalars()
    )


def begin_handover_sync(
    release_id: str, runner_ids: frozenset[str] | set[str] | list[str]
) -> HandoverReport:
    """Stage a handover PAUSE for every eligible run of ``runner_ids``.

    Safe to call again (the host repeats it while it waits): a run that
    already has this release's command is not staged twice.
    """

    runners = set(runner_ids)
    if not runners or not get_task_auto_resume_enabled():
        # Nothing could resume a paused run: leave every run to the drain.
        return HandoverReport(kept=0)
    staged = 0
    kept = 0
    with get_session_local()() as db:
        tasks = list(
            db.execute(
                select(Task)
                .where(Task.status == TaskStatus.RUNNING, Task.runner_id.in_(runners))
                .with_for_update()
            ).scalars()
        )
        for task in tasks:
            if task.run_id is None or task.control_state != (
                TaskControlState.RUNNING.value
            ):
                # A pause (the user's) or another control already moves it.
                kept += 1
                continue
            if not auto_recovery_eligibility(db, task).eligible:
                kept += 1
                continue
            command_id = handover_command_id(release_id, int(task.id))
            existing = db.execute(
                select(TaskExecutionCommand.id).where(
                    TaskExecutionCommand.task_id == task.id,
                    TaskExecutionCommand.command_id == command_id,
                )
            ).first()
            if existing is not None:
                continue
            stage_task_command(
                db,
                task_id=int(task.id),
                actor_user_id=cast(int | None, task.user_id),
                command_id=command_id,
                kind=TaskCommandKind.PAUSE,
                payload={},
                target_run_id=str(task.run_id),
            )
            staged += 1
        if staged:
            db.commit()
        else:
            db.rollback()
    if staged:
        notify_task_command_dispatcher()
    return HandoverReport(staged=staged, kept=kept)


def settle_handover_sync(release_id: str) -> HandoverReport:
    """Record each completed handover pause so the successor resumes it.

    Returns how many pauses were recorded now and how many are still pending.
    A run that moved on since (its user resumed, cancelled or wrote to it, so
    its run or ``state_version`` changed) is not recorded: it is no longer the
    handover's. Idempotent: a run already recorded for its run id is skipped.
    """

    recorded = 0
    unsettled = 0
    with get_session_local()() as db:
        commands = _release_commands(db, release_id)
        for command in commands:
            if command.status not in COMMAND_TERMINAL:
                unsettled += 1
                continue
            if command.status != COMMAND_COMPLETED:
                continue
            task = db.execute(
                select(Task).where(Task.id == command.task_id).with_for_update()
            ).scalar_one_or_none()
            if (
                task is None
                or task.status != TaskStatus.PAUSED
                or task.control_state != TaskControlState.PAUSED.value
                or task.run_id != command.target_run_id
            ):
                continue
            row = db.get(TaskAutoRecovery, task.id)
            if (
                row is not None
                and row.run_id == task.run_id
                and row.reason == InterruptionReason.SHUTDOWN.value
            ):
                continue
            now = datetime.now(timezone.utc)
            savepoint = db.begin_nested()
            try:
                record_interruption_no_commit(
                    db,
                    task=task,
                    reason=InterruptionReason.SHUTDOWN,
                    task_status=TaskStatus.PAUSED,
                    interrupted_at=now,
                    now=now,
                    progress_marker=None,
                    gated_by_settlement_switches=False,
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
    return HandoverReport(recorded=recorded, unsettled=unsettled)
