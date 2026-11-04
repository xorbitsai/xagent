"""Abandon a waiting task's pending question without deleting its history.

The endpoint behind this service closes the paused run and leaves the task
``completed``, so the caller can append a fresh turn (with a fresh connector
runtime context) on the same task. The pending interaction row is terminated,
not deleted; legacy question rows are relabelled exactly as
``supersede_legacy_question_rows`` already does for a structured publication.

Terminal reason: the interaction table's ``terminal_reason`` CHECK has a
closed vocabulary, so this reuses ``run_superseded`` -- the existing "the run
that asked no longer owns this slot" outcome -- instead of adding a
``cancelled`` member that would need a migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...config import get_shared_task_execution_enabled
from ..models.chat_message import TaskChatMessage
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus, TraceEvent
from ..models.task_command import TaskExecutionCommand
from .chat_history_service import (
    QUESTION_MESSAGE_TYPE,
    SUPERSEDED_MESSAGE_TYPE,
    supersede_legacy_question_rows,
)
from .db_runtime import run_db_io_cancellation_safe
from .task_command_transport import (
    COMMAND_PENDING,
    COMMAND_PROCESSING,
    TaskCommandKind,
)
from .task_execution_controller import TaskControlState, task_control_snapshot
from .task_interaction_close import terminate_active_interaction_for_question_cancel
from .task_interaction_read import get_pending_interaction_question
from .task_lease_service import task_settlement_ownership_values
from .task_orchestrator import (
    TaskTurnOrchestrator,
    invalidate_task_cache_best_effort,
    sync_trigger_run_status,
)
from .task_resume import _RESUMABLE_CONTROL_STATES
from .task_runtime import (
    QUESTION_CANCELLED_AGENT_CONFIG_KEY,
    question_cancel_marker,
)
from .task_start import SdkTaskScope, resolve_sdk_task
from .trace_event_staging import stage_trace_event_row
from .workforce_runtime import sync_workforce_run_status


class TaskQuestionCancelBusy(Exception):
    """A reply claimed the waiting task first; retrying can succeed."""


class TaskQuestionCancelNotWaiting(Exception):
    """The task is not waiting and carries no cancelled-question marker."""


@dataclass(frozen=True)
class TaskQuestionCancelResult:
    task_id: int
    status: str
    run_id: str | None
    state_version: int


def _task_run_id(task: Task) -> str | None:
    return str(task.run_id) if task.run_id is not None else None


def _pending_resume_command_exists(task_id: int) -> Any:
    """A queued/in-flight reply owns the answer slot; cancel must not steal it."""
    return sa.exists(
        select(1).where(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.kind == TaskCommandKind.RESUME_INPUT.value,
            TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
        )
    )


def _pending_resume_command_in_flight(db: Session, task_id: int) -> bool:
    return bool(db.scalar(select(_pending_resume_command_exists(task_id))))


def _latest_question_row(db: Session, task_id: int) -> TaskChatMessage | None:
    return (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
            TaskChatMessage.message_type.in_(
                (QUESTION_MESSAGE_TYPE, SUPERSEDED_MESSAGE_TYPE)
            ),
        )
        .order_by(TaskChatMessage.id.desc())
        .first()
    )


def _trace_text(row: TraceEvent) -> str:
    data: Any = row.data if isinstance(row.data, dict) else {}
    value = data.get("message")
    if value is None:
        value = data.get("content")
    return value.strip() if isinstance(value, str) else ""


def _question_trace_exists(
    db: Session,
    *,
    task_id: int,
    question: str,
    question_row: TaskChatMessage | None,
) -> bool:
    """Whether ``/steps`` already has an assistant trace for this question.

    Rows written by the live runtime carry ``source_event_id``; older or
    test-seeded transcript rows do not, so fall back to content matching
    scoped after the task's latest ``user_message`` -- a repeated question in
    a later turn must still be materialized.
    """

    if question_row is not None:
        source_event_id = getattr(question_row, "source_event_id", None)
        if source_event_id:
            exists = (
                db.query(TraceEvent.id)
                .filter(
                    TraceEvent.task_id == task_id,
                    TraceEvent.event_id == str(source_event_id),
                )
                .first()
            )
            if exists is not None:
                return True
    wanted = question.strip()
    latest_user_trace_id = (
        db.query(func.max(TraceEvent.id))
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.event_type == "user_message",
        )
        .scalar()
        or 0
    )
    rows = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.event_type == "agent_message",
            TraceEvent.id > int(latest_user_trace_id),
        )
        .all()
    )
    return any(_trace_text(row) == wanted for row in rows)


def _materialize_question_step(
    db: Session,
    *,
    task: Task,
    question: str | None,
    interactions: list[dict[str, Any]] | None,
    question_row: TaskChatMessage | None,
) -> None:
    """Give the cancelled question a trace row so ``/steps`` can project it.

    Live questions already have an ``agent_message`` trace; a transcript-only
    row (older data, or a task created without the outbound writer) has none.
    """

    if (
        task.conversation_storage_version != 1
        or task.traces_expired_at is not None
        or not question
        or not question.strip()
    ):
        return
    task_id = int(task.id)
    if _question_trace_exists(
        db, task_id=task_id, question=question, question_row=question_row
    ):
        return
    timestamp = getattr(question_row, "created_at", None) or datetime.now(timezone.utc)
    row_id = getattr(question_row, "id", None)
    identity = row_id if row_id is not None else (task.run_id or "unknown")
    stage_trace_event_row(
        db,
        task_id=task_id,
        build_id=None,
        event_id=f"v1:cancel-question:{task_id}:{identity}",
        event_type="agent_message",
        timestamp=timestamp,
        step_id=None,
        parent_event_id=None,
        data={
            "message": question,
            "message_type": "question",
            "expect_response": True,
            "visible": True,
            "metadata": {"interactions": interactions or []},
        },
        checkpoint_lease=None,
    )


def _result(task: Task) -> TaskQuestionCancelResult:
    return TaskQuestionCancelResult(
        task_id=int(task.id),
        status=task.status.value,
        run_id=_task_run_id(task),
        state_version=int(task.state_version or 0),
    )


def _result_or_raise(task: Task) -> TaskQuestionCancelResult:
    """Return the idempotent replay result, or the right 409 for this state."""

    if task.status == TaskStatus.RUNNING:
        raise TaskQuestionCancelBusy
    if task.status == TaskStatus.COMPLETED:
        marker = question_cancel_marker(task.agent_config)
        if marker is not None and marker.get("run_id") == _task_run_id(task):
            return _result(task)
    raise TaskQuestionCancelNotWaiting


def _cancel_waiting_task(
    db: Session, task: Task, scope: SdkTaskScope
) -> TaskQuestionCancelResult:
    task_id = int(task.id)
    run_id = _task_run_id(task)
    observed_state_version = int(task.state_version or 0)
    now = datetime.now(timezone.utc)
    agent_config: dict[str, Any] = (
        dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
    )
    agent_config[QUESTION_CANCELLED_AGENT_CONFIG_KEY] = {
        "run_id": run_id,
        "cancelled_at": now.isoformat(),
    }
    # This request does not run inside the owning coordinator. Under shared
    # execution that coordinator may still be draining its handle, so keep
    # runner/attempt/lease as they are and let it release ownership after the
    # drain; clearing them here would let another worker acquire the task and
    # start an append before the drain barrier, and free the admission slot.
    ownership: dict[str, Any] = (
        {}
        if get_shared_task_execution_enabled()
        else task_settlement_ownership_values(task_id)
    )
    values: dict[Any, Any] = {
        Task.status: TaskStatus.COMPLETED,
        Task.control_state: TaskControlState.COMPLETED.value,
        Task.state_version: func.coalesce(Task.state_version, 0) + 1,
        Task.completion_outcome: None,
        Task.output: None,
        Task.error_message: None,
        Task.agent_config: agent_config,
        **ownership,
    }
    # Same acceptance set as the reply prelease, plus the exact run/version
    # read above: a reply that finishes one round and asks again must not be
    # cancelled by this stale request.
    run_fence = Task.run_id.is_(None) if run_id is None else Task.run_id == run_id
    claimed = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.source == "sdk",
            Task.status == TaskStatus.WAITING_FOR_USER,
            Task.control_state.in_(tuple(_RESUMABLE_CONTROL_STATES)),
            run_fence,
            func.coalesce(Task.state_version, 0) == observed_state_version,
            ~_pending_resume_command_exists(task_id),
        )
        .update(values, synchronize_session=False)
    )
    if claimed != 1:
        db.rollback()
        fresh = resolve_sdk_task(task_id, scope, db)
        if fresh.status == TaskStatus.WAITING_FOR_USER and (
            fresh.control_state == TaskControlState.RESUME_REQUESTED.value
            or _pending_resume_command_in_flight(db, task_id)
        ):
            raise TaskQuestionCancelBusy
        return _result_or_raise(fresh)

    completed = db.query(Task).filter(Task.id == task_id).populate_existing().one()
    # Read the question before terminating/relabelling the rows that carry it.
    # A native row this reader cannot read (T3) means "no question to show";
    # never fall back to a previous turn's already-superseded transcript row.
    question, interactions = get_pending_interaction_question(db, completed)
    question_row = _latest_question_row(db, task_id)

    terminate_active_interaction_for_question_cancel(db, task_id=task_id, run_id=run_id)
    supersede_legacy_question_rows(db, task_id=task_id)
    _materialize_question_step(
        db,
        task=completed,
        question=question,
        interactions=interactions,
        question_row=question_row,
    )
    sync_workforce_run_status(db, completed, TaskStatus.COMPLETED)
    sync_trigger_run_status(db, completed, TaskStatus.COMPLETED)
    if completed.conversation_storage_version == 2:
        from .task_execution_event_writer import (
            append_fact_no_commit,
            stage_result_fact_no_commit,
        )

        append_fact_no_commit(
            db,
            task_id=task_id,
            kind="control_state_changed",
            key=f"control:{completed.state_version}",
            run_id=_task_run_id(completed),
            payload=task_control_snapshot(completed).as_dict(),
        )
        # ``cancelled`` would inject "Previous execution was cancelled." into
        # the next turn; abandoning a question is not a cancelled execution.
        stage_result_fact_no_commit(db, completed, {"status": "completed"})

    result = _result(completed)
    db.commit()
    invalidate_task_cache_best_effort(task_id)
    return result


def _cancel_sync(*, task_id: int, scope: SdkTaskScope) -> TaskQuestionCancelResult:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = resolve_sdk_task(task_id, scope, db)
        if task.status != TaskStatus.WAITING_FOR_USER:
            return _result_or_raise(task)
        if not get_shared_task_execution_enabled():
            # Refuse a runner still finishing after WAITING_FOR_USER committed,
            # exactly as append does; ownership is checked first so 404 is opaque.
            TaskTurnOrchestrator.ensure_no_background_turn(task_id)
        return _cancel_waiting_task(db, task, scope)


async def cancel_task_question(
    *, task_id: int, scope: SdkTaskScope
) -> TaskQuestionCancelResult:
    """Authorize and cancel one waiting task's pending question."""

    return await run_db_io_cancellation_safe(
        lambda: _cancel_sync(task_id=task_id, scope=scope)
    )
