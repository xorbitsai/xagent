"""A person cancels, from the web UI, the automatic resume a paused run waits for.

One transaction per attempt, shaped like the channel /stop
(``channel_stop.stop_channel_task_sync``): lock the task row, apply
``cancel_pending_auto_resume_no_commit``, read again after a lost
compare-and-swap (SQLite; at most three attempts), and commit only a cancel.
After the commit the task's cached responses are dropped -- a write within
the second of the cached ``updated_at`` would otherwise be served the old
``state_version`` -- and the cancel's ``task_recovery_update`` is published.
Lock order is the module's: tasks, the recovery row, commands, deliveries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.runtime_performance import increment_counter
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from .channel_stop import settle_channel_reply_after_cancel_no_commit
from .db_runtime import run_db_io_cancellation_safe
from .task_auto_recovery import current_auto_recovery_view, is_status_only_source
from .task_auto_resume import (
    RecoveryNotice,
    cancel_pending_auto_resume_no_commit,
    publish_recovery_notices,
)
from .task_execution_controller import TaskControlState
from .task_orchestrator import invalidate_task_cache_best_effort

logger = logging.getLogger(__name__)

_ATTEMPTS = 3
_AT = "web_cancel"


class WebAutoResumeCancelOutcome(str, Enum):
    CANCELLED = "cancelled"  # a pending automatic resume was cancelled
    ALREADY_CANCELLED = "already_cancelled"  # a person cancelled it before
    RESUME_IN_PROGRESS = "resume_in_progress"  # a resume is handing off
    NOT_PENDING = "not_pending"  # no automatic resume of this run to cancel
    # Not answered with 200; the route maps them to errors.
    MISSING = "missing"  # no such task the caller may change
    NOT_CANCELLABLE = "not_cancellable"  # a status-only (SDK) task
    RACED = "raced"  # every attempt lost a compare-and-swap


@dataclass(frozen=True)
class WebAutoResumeCancelResult:
    outcome: WebAutoResumeCancelOutcome
    # The task the outcome was decided on, read under its lock (a cancel's
    # own writes included): status, control_state, run_id, state_version and
    # auto_recovery, as GET reads them. ``None`` exactly when no task was
    # answered for: MISSING (no such task the caller may change),
    # NOT_CANCELLABLE and RACED.
    task: dict[str, Any] | None = None
    notices: tuple[RecoveryNotice, ...] = ()
    # The ``task_recovery_update`` messages published for ``notices``, the
    # very dicts broadcast (set by :func:`cancel_auto_resume_for_web`).
    messages: tuple[dict[str, Any], ...] = ()


def _result(outcome: WebAutoResumeCancelOutcome) -> WebAutoResumeCancelResult:
    return WebAutoResumeCancelResult(outcome)


def _cancel_once_no_commit(
    db: Session,
    *,
    task_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
    expected_run_id: str | None,
) -> WebAutoResumeCancelResult | None:
    """One attempt; ``None`` when the cancel lost a race."""

    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None or not (actor_is_admin or int(task.user_id) == actor_user_id):
        return _result(WebAutoResumeCancelOutcome.MISSING)
    if is_status_only_source(task.source):
        return _result(WebAutoResumeCancelOutcome.NOT_CANCELLABLE)
    if expected_run_id is not None and task.run_id != expected_run_id:
        return WebAutoResumeCancelResult(
            WebAutoResumeCancelOutcome.NOT_PENDING, task=_task_snapshot(db, task)
        )

    cancel = cancel_pending_auto_resume_no_commit(
        db, task, at=_AT, actor_user_id=actor_user_id
    )
    if cancel.raced:
        return None
    if cancel.cancelled:
        if task.channel_id is not None:
            settle_channel_reply_after_cancel_no_commit(
                db,
                task_id=task_id,
                run_id=str(cancel.run_id),
                channel_id=int(task.channel_id),
            )
        notices = (cancel.notice,) if cancel.notice is not None else ()
        return WebAutoResumeCancelResult(
            WebAutoResumeCancelOutcome.CANCELLED,
            task=_task_snapshot(db, task),
            notices=notices,
        )

    snapshot = _task_snapshot(db, task)
    if (
        task.status == TaskStatus.PAUSED
        and task.control_state == TaskControlState.RESUME_REQUESTED.value
    ):
        outcome = WebAutoResumeCancelOutcome.RESUME_IN_PROGRESS
    elif _stopped_by_a_cancel(snapshot["auto_recovery"]):
        outcome = WebAutoResumeCancelOutcome.ALREADY_CANCELLED
    else:
        outcome = WebAutoResumeCancelOutcome.NOT_PENDING
    return WebAutoResumeCancelResult(outcome, task=snapshot)


def _stopped_by_a_cancel(view: dict[str, Any] | None) -> bool:
    auto_resume = (view or {}).get("auto_resume")
    return (
        auto_resume is not None
        and auto_resume.get("status") == "stopped"
        and auto_resume.get("stop_reason") == "cancelled"
    )


def _task_snapshot(db: Session, task: Task) -> dict[str, Any]:
    """``task`` as the next GET reads it, plain data. Taken under the lock,
    before the commit or rollback, so it is what the outcome was decided on
    (a cancel's own writes included) and nothing fallible runs between a
    committed cancel and its broadcast."""

    status = task.status
    return {
        "status": status.value if hasattr(status, "value") else str(status),
        "control_state": str(task.control_state or "idle"),
        "run_id": task.run_id,
        "state_version": int(task.state_version or 0),
        "auto_recovery": current_auto_recovery_view(db, task),
    }


def cancel_auto_resume_for_web_sync(
    *,
    task_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
    expected_run_id: str | None = None,
) -> WebAutoResumeCancelResult:
    """Cancel ``task_id``'s pending automatic resume for a person on the web.

    The caller is the task's owner or an admin; anyone else reads MISSING.
    ``expected_run_id`` (the run whose notice the person saw): a task now on
    another run is NOT_PENDING, and nothing is written. Only a cancel
    commits; after it the task's cached responses are dropped, here rather
    than in the async wrapper so that a disconnected client cannot skip it.
    """

    result = _result(WebAutoResumeCancelOutcome.RACED)
    for _ in range(_ATTEMPTS):
        with get_session_local()() as db:
            attempt = _cancel_once_no_commit(
                db,
                task_id=task_id,
                actor_user_id=actor_user_id,
                actor_is_admin=actor_is_admin,
                expected_run_id=expected_run_id,
            )
            if attempt is None:
                db.rollback()
                continue
            if attempt.outcome is WebAutoResumeCancelOutcome.CANCELLED:
                db.commit()
                invalidate_task_cache_best_effort(task_id)
            else:
                db.rollback()
            result = attempt
        break
    else:
        logger.warning(
            "Web cancel of automatic resume of task %s kept losing a race", task_id
        )
    increment_counter(
        "xagent.web.auto_resume_cancel", attributes={"outcome": result.outcome.value}
    )
    return result


async def cancel_auto_resume_for_web(
    *,
    task_id: int,
    actor_user_id: int,
    actor_is_admin: bool,
    expected_run_id: str | None = None,
) -> WebAutoResumeCancelResult:
    """:func:`cancel_auto_resume_for_web_sync` off the loop, then its notices
    published; ``messages`` are the very dicts broadcast, timestamp included."""

    result = await run_db_io_cancellation_safe(
        lambda: cancel_auto_resume_for_web_sync(
            task_id=task_id,
            actor_user_id=actor_user_id,
            actor_is_admin=actor_is_admin,
            expected_run_id=expected_run_id,
        )
    )
    if not result.notices:
        return result
    messages = await publish_recovery_notices(result.notices)
    return replace(result, messages=messages)
