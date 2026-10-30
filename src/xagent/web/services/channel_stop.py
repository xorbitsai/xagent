"""A channel /stop applied to the sender's selected task in the database.

The ingress's in-memory stop reaches only a turn it still observes, and its
PAUSE does nothing to a run that already rests PAUSED while an automatic
resume of it is pending. This service runs first: it cancels that resume
(``cancel_pending_auto_resume_no_commit``), tells the ingress when nothing
of the task runs, and, when the ingress no longer tracks the conversation,
pauses the run whose reply the chat still waits for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ...core.runtime_performance import increment_counter
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_channel_delivery import TaskChannelDelivery
from ..models.task_command import TaskExecutionCommand
from .channel_runtime import _load_channel_owner_sync
from .db_runtime import run_db_io_cancellation_safe
from .execution_result_projection import interrupted_channel_result
from .shared_channel_execution import stamp_channel_result_no_commit
from .task_auto_resume import (
    RecoveryNotice,
    cancel_pending_auto_resume_no_commit,
    publish_recovery_notices,
    resume_hand_off_can_complete,
)
from .task_command_transport import (
    AUTO_RESUME_COMMAND_PREFIX,
    COMMAND_TERMINAL,
    TaskCommandKind,
    notify_task_command_dispatcher,
    stage_task_command,
)
from .task_execution_controller import TaskControlState
from .task_orchestrator import invalidate_task_cache_best_effort

logger = logging.getLogger(__name__)

_ATTEMPTS = 3
# Commands that start or continue a run of the task.
_RUN_COMMAND_KINDS = (
    TaskCommandKind.START.value,
    TaskCommandKind.RESUME.value,
    TaskCommandKind.RESUME_INPUT.value,
    TaskCommandKind.MESSAGE.value,
)


class ChannelStopOutcome(str, Enum):
    CANCELLED = "cancelled"  # a pending automatic resume was cancelled
    PAUSE_QUEUED = "pause_queued"  # a PAUSE is queued for the run this chat awaits
    IDLE = "idle"  # nothing of the task runs or will run
    ACTIVE = "active"  # leave it to the ingress's in-memory stop
    UNAVAILABLE = "unavailable"  # no selected task of this channel and owner


@dataclass(frozen=True)
class ChannelStopResult:
    outcome: ChannelStopOutcome
    # The START whose held reply the cancel released: deliver it now.
    held_command_db_id: int | None = None
    notices: tuple[RecoveryNotice, ...] = ()


def _release_held_delivery_no_commit(
    db: Session, *, task_id: int, run_id: str, channel_id: int
) -> int | None:
    """Make the pending reply of ``run_id``'s START claimable now; returns
    the START's id when that reply is still pending. A reply another sender
    holds keeps its lease."""
    starts = select(TaskExecutionCommand.id).where(
        TaskExecutionCommand.task_id == task_id,
        TaskExecutionCommand.kind == TaskCommandKind.START.value,
        TaskExecutionCommand.target_run_id == run_id,
    )
    pending = (
        TaskChannelDelivery.command_id.in_(starts),
        TaskChannelDelivery.channel_id == channel_id,
        TaskChannelDelivery.status == "pending",
    )
    held = db.execute(
        select(TaskChannelDelivery.command_id)
        .where(*pending)
        .order_by(TaskChannelDelivery.command_id)
        .limit(1)
    ).scalar_one_or_none()
    if held is None:
        return None
    db.execute(
        update(TaskChannelDelivery)
        .where(
            TaskChannelDelivery.command_id == held,
            TaskChannelDelivery.claim_token.is_(None),
            *pending,
        )
        .values(available_at=datetime.now(timezone.utc))
        .execution_options(synchronize_session=False)
    )
    return int(held)


def settle_channel_reply_after_cancel_no_commit(
    db: Session, *, task_id: int, run_id: str, channel_id: int
) -> int | None:
    """The channel reply of ``run_id`` once a person cancelled its automatic
    resume: "interrupted" (first writer wins, whatever a later resume by
    hand answers), and its pending reply claimable now. Returns the START
    whose held reply was released. Caller holds the ``tasks`` row lock; lock
    order: tasks, the recovery row, commands, deliveries. Without a stored
    START or a delivery row (no shared execution) both steps do nothing."""

    stamp_channel_result_no_commit(
        db,
        task_id=task_id,
        run_id=run_id,
        channel_result=interrupted_channel_result(),
    )
    return _release_held_delivery_no_commit(
        db, task_id=task_id, run_id=run_id, channel_id=channel_id
    )


def _idle(db: Session, task: Task) -> bool:
    """Nothing of ``task`` runs or will run: it rests (paused by a person or
    ended) and no command that starts a run is unfinished. A queued
    automatic RESUME does not count; with no resume pending its guard skips
    it."""

    rests = task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) or (
        task.status == TaskStatus.PAUSED
        and task.control_state == TaskControlState.PAUSED.value
    )
    if not rests:
        return False
    unfinished = db.execute(
        select(TaskExecutionCommand.id)
        .where(
            TaskExecutionCommand.task_id == task.id,
            TaskExecutionCommand.status.not_in(COMMAND_TERMINAL),
            TaskExecutionCommand.kind.in_(_RUN_COMMAND_KINDS),
            func.substr(
                TaskExecutionCommand.command_id, 1, len(AUTO_RESUME_COMMAND_PREFIX)
            )
            != AUTO_RESUME_COMMAND_PREFIX,
        )
        .limit(1)
    ).first()
    return unfinished is None


def _awaits_reply(db: Session, task: Task, channel_id: int) -> bool:
    """This channel still waits for the reply of ``task``'s current run."""

    if task.run_id is None:
        return False
    return (
        db.execute(
            select(TaskChannelDelivery.command_id)
            .join(
                TaskExecutionCommand,
                TaskExecutionCommand.id == TaskChannelDelivery.command_id,
            )
            .where(
                TaskChannelDelivery.channel_id == channel_id,
                TaskChannelDelivery.status == "pending",
                TaskExecutionCommand.task_id == task.id,
                TaskExecutionCommand.kind == TaskCommandKind.START.value,
                TaskExecutionCommand.target_run_id == task.run_id,
            )
            .limit(1)
        ).first()
        is not None
    )


def _pause_queued(db: Session, task: Task) -> bool:
    """A pause of ``task``'s current run is already requested or queued."""

    if task.control_state == TaskControlState.PAUSE_REQUESTED.value:
        return True
    return (
        db.execute(
            select(TaskExecutionCommand.id)
            .where(
                TaskExecutionCommand.task_id == task.id,
                TaskExecutionCommand.kind == TaskCommandKind.PAUSE.value,
                TaskExecutionCommand.target_run_id == task.run_id,
                TaskExecutionCommand.status.not_in(COMMAND_TERMINAL),
            )
            .limit(1)
        ).first()
        is not None
    )


def _stop_once_no_commit(
    db: Session,
    *,
    channel_id: int,
    external_user_id: str,
    task_id: int,
    pause_awaited_run: bool,
) -> tuple[ChannelStopResult, bool] | None:
    """One attempt; ``None`` when the cancel lost a race. The flag says
    whether a PAUSE was staged."""

    owner = _load_channel_owner_sync(
        db, channel_id=channel_id, external_user_id=external_user_id
    )
    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None or task.channel_id != channel_id or task.user_id != owner.user_id:
        return ChannelStopResult(ChannelStopOutcome.UNAVAILABLE), False

    cancel = cancel_pending_auto_resume_no_commit(db, task, at="channel_stop")
    if cancel.raced:
        return None
    if cancel.cancelled:
        held = settle_channel_reply_after_cancel_no_commit(
            db, task_id=task_id, run_id=str(cancel.run_id), channel_id=channel_id
        )
        notices = (cancel.notice,) if cancel.notice is not None else ()
        return (
            ChannelStopResult(ChannelStopOutcome.CANCELLED, held, notices),
            False,
        )

    if _idle(db, task):
        return ChannelStopResult(ChannelStopOutcome.IDLE), False

    if pause_awaited_run and _awaits_reply(db, task, channel_id):
        if _pause_queued(db, task):
            return ChannelStopResult(ChannelStopOutcome.PAUSE_QUEUED), False
        if task.status == TaskStatus.RUNNING or (
            task.status == TaskStatus.PAUSED
            and task.control_state == TaskControlState.RESUME_REQUESTED.value
            and resume_hand_off_can_complete(db, task, now=datetime.now(timezone.utc))
        ):
            stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=owner.user_id,
                command_id=f"stop:{uuid4().hex}",
                kind=TaskCommandKind.PAUSE,
                payload={},
                target_run_id=str(task.run_id),
            )
            return ChannelStopResult(ChannelStopOutcome.PAUSE_QUEUED), True

    return ChannelStopResult(ChannelStopOutcome.ACTIVE), False


def stop_channel_task_sync(
    *, channel_id: int, external_user_id: str, task_id: int, pause_awaited_run: bool
) -> ChannelStopResult:
    """Apply a channel /stop to the sender's selected task; one transaction.

    Raises ChannelAuthorizationError / ChannelConfigurationError like
    ``_load_channel_owner_sync``. A lost compare-and-swap (SQLite) is read
    again, at most three times; after that the ingress falls back to its
    in-memory stop (ACTIVE). Only a cancel or a staged PAUSE commits.

    ``pause_awaited_run`` (the ingress tracks no turn of the sender) pauses
    a RUNNING run, or a resume still handing off, whose START reply this
    channel awaits. A PENDING task -- its START queued, not yet admitted --
    is not paused: the PAUSE would target a run that has not started. That
    is ACTIVE, and the ingress's in-memory stop handles it.
    """

    result = ChannelStopResult(ChannelStopOutcome.ACTIVE)
    staged = False
    for _ in range(_ATTEMPTS):
        with get_session_local()() as db:
            attempt = _stop_once_no_commit(
                db,
                channel_id=channel_id,
                external_user_id=external_user_id,
                task_id=task_id,
                pause_awaited_run=pause_awaited_run,
            )
            if attempt is None:
                db.rollback()
                continue
            result, staged = attempt
            if result.outcome is ChannelStopOutcome.CANCELLED or staged:
                db.commit()
            else:
                db.rollback()
        break
    else:
        logger.warning("Channel stop of task %s kept losing a race", task_id)
    if result.outcome is ChannelStopOutcome.CANCELLED:
        # The cancel moved ``state_version``; a cached detail or status read
        # keyed on ``updated_at`` (one-second resolution on SQLite) would
        # otherwise keep serving the old fence.
        invalidate_task_cache_best_effort(task_id)
    if staged:
        notify_task_command_dispatcher()
    increment_counter(
        "xagent.channel.stop", attributes={"outcome": result.outcome.value}
    )
    return result


async def stop_channel_task(
    *, channel_id: int, external_user_id: str, task_id: int, pause_awaited_run: bool
) -> ChannelStopResult:
    """``stop_channel_task_sync`` off the loop, then its notices published."""

    result = await run_db_io_cancellation_safe(
        lambda: stop_channel_task_sync(
            channel_id=channel_id,
            external_user_id=external_user_id,
            task_id=task_id,
            pause_awaited_run=pause_awaited_run,
        )
    )
    await publish_recovery_notices(result.notices)
    return result
