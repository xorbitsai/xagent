"""Cancel one SDK-source task through the durable command channel.

``POST /v1/chat/tasks/{task_id}/cancel`` is split in two halves that live
here. The request half authorizes the caller, refuses a target that cannot
be cancelled, persists a CANCEL command and waits a bounded time for it.
The execution half runs wherever that command is applied -- the process
that owns the task -- and is the SDK counterpart of the A2A and external
cores, which load their targets by ``Task.source`` and so cannot reach an
SDK task.

One judgement decides whether a task was cancelled:
``is_sdk_cancelled`` -- FAILED carrying a sentence only a cancellation
writes. The cancelled run's own settlement writes that sentence when the
run was stopped by an explicit ``cancel_task`` call
(``sdk_cancel_settlement_text``); this core writes the same row when no
settlement did. Whichever writer lands first owns the terminal row, and
the other one changes nothing.

Only a RUNNING task is cancelled here. A task with a turn still queued to
start, one waiting for the caller's answer and a paused one are refused as
busy, without a write.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic
from typing import Any

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from ..models.chat_message import TaskChatMessage
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from .db_runtime import run_db_io_cancellation_safe
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    TaskCommandKind,
    TaskCommandRejected,
    TaskCommandTaskMissing,
    dispatch_one_task_command,
    enqueue_task_command,
    load_task_command,
    retry_failed_task_command,
)
from .task_execution_controller import StaleTaskRunError, TaskControlState

logger = logging.getLogger(__name__)

SDK_TASK_SOURCE = "sdk"

# Command payload scope that routes a CANCEL command to this core. Only a
# server-side producer writes it; client frames naming a scope are refused.
SDK_COMMAND_SCOPE = "sdk"

# The durable text of a cancelled SDK task, read back by the caller as the
# task's ``error``. It is also the cancellation judgement, so rewording it
# means adding the new sentence to ``_SDK_CANCEL_MESSAGES`` and keeping
# this one there: rows and processes carrying the old text must still be
# recognised.
SDK_CANCEL_ERROR_MESSAGE = "Task was cancelled by the caller."

_SDK_CANCEL_MESSAGES = frozenset({SDK_CANCEL_ERROR_MESSAGE})

# Machine-readable companion of the text on the public read surface.
SDK_CANCEL_ERROR_CODE = "task_cancelled"

# Passed to ``BackgroundTaskManager.cancel_task``. Nominal, not a bound:
# that call returns once the cancelled coroutine has unwound, and its
# cleanup drains database work that cannot be interrupted.
SDK_CANCEL_WAIT_SECONDS = 5.0

# How long the request waits for its command before answering "accepted".
SDK_CANCEL_REQUEST_WAIT_SECONDS = 10.0

_COMMAND_POLL_SECONDS = 0.05

_TERMINAL_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED})


def is_sdk_cancelled(status: object, error_message: object) -> bool:
    """Whether a task row is the outcome of an SDK cancellation.

    The row's ``error_message`` always describes its current run: a new run
    clears it when it starts and every other terminal write replaces it, so
    the sentence cannot outlive the run it was written for.
    """

    return status == TaskStatus.FAILED and error_message in _SDK_CANCEL_MESSAGES


def sdk_cancel_error_code(status: object, error_message: object) -> str | None:
    """The public ``error_code`` of a task row: set only for a cancellation."""

    return SDK_CANCEL_ERROR_CODE if is_sdk_cancelled(status, error_message) else None


def sdk_cancel_settlement_text(
    task_source: str | None, *, explicit_cancel: bool
) -> str | None:
    """Settlement text for an SDK run stopped by an explicit ``cancel_task``.

    ``None`` for every other source and for a cancellation nobody asked for
    (a host shutting down), which keep the wording they have always had.
    """

    if task_source == SDK_TASK_SOURCE and explicit_cancel:
        return SDK_CANCEL_ERROR_MESSAGE
    return None


def sdk_cancel_result_fact(error_message: str | None) -> dict[str, Any]:
    """Result fact for a failed run, naming a cancellation as one.

    The next turn's context reports ``cancelled`` as such instead of as a
    failure.
    """

    if error_message in _SDK_CANCEL_MESSAGES:
        return {"status": "cancelled", "error": error_message}
    return {"error": error_message}


class SdkTaskCancelBusy(Exception):
    """Nothing was cancelled, and retrying can succeed."""


class SdkTaskNotCancellable(Exception):
    """The task already ended on its own; no retry can cancel it."""


class SdkTaskCancelFailed(Exception):
    """The command could not be applied; the task state is unchanged by it."""


@dataclass(frozen=True)
class SdkTaskCancelResult:
    task_id: int
    status: str
    run_id: str | None
    state_version: int
    cancelled_run_id: str | None
    error_code: str | None
    # False while the command is still being applied.
    cancelled: bool


@dataclass(frozen=True)
class _CancelCommand:
    command_db_id: int


def _task_run_id(task: Task) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _result(task: Task) -> SdkTaskCancelResult:
    cancelled = is_sdk_cancelled(task.status, task.error_message)
    run_id = _task_run_id(task)
    return SdkTaskCancelResult(
        task_id=int(task.id),
        status=task.status.value,
        run_id=run_id,
        state_version=int(task.state_version or 0),
        cancelled_run_id=run_id if cancelled else None,
        error_code=sdk_cancel_error_code(task.status, task.error_message),
        cancelled=cancelled,
    )


def _prepare_cancel_sync(
    *, task_id: int, scope: Any
) -> SdkTaskCancelResult | _CancelCommand:
    """Authorize, classify the target and persist its command, in one read."""

    from .task_auto_resume import status_only_terminal_view
    from .task_start import TaskStartRejected, resolve_sdk_task
    from .task_start_protocol import has_queued_start

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # Read in the order the public task read uses: until a queued START
        # is handed off, the row still shows the previous turn, so judging
        # the row alone would answer for a turn the caller is not asking
        # about.
        queued = has_queued_start(db, task_id)
        task = resolve_sdk_task(task_id, scope, db)
        if queued:
            raise SdkTaskCancelBusy(
                "A turn of this task is queued to start and cannot be "
                "cancelled yet; retry once it is running."
            )
        if is_sdk_cancelled(task.status, task.error_message):
            return _result(task)
        if task.status in _TERMINAL_STATUSES:
            raise SdkTaskNotCancellable
        if (
            task.status == TaskStatus.PAUSED
            and task.control_state == TaskControlState.PAUSED.value
            and status_only_terminal_view(db, int(task.id)) is not None
        ):
            # Reported as failed on the read surface; nothing is left to stop.
            raise SdkTaskNotCancellable
        if task.status != TaskStatus.RUNNING:
            raise SdkTaskCancelBusy(
                f"A task that is {task.status.value} cannot be cancelled yet."
            )
        target_state_version = int(task.state_version or 0)
        try:
            enqueued = _enqueue_cancel(db, task, target_state_version)
        except TaskCommandTaskMissing as exc:
            # Deleted between the read above and the insert.
            raise TaskStartRejected("task_not_found") from exc
        if not enqueued.payload_matches:
            raise SdkTaskCancelBusy(
                "Another command holds this cancel identity; retry."
            )
        if enqueued.status == COMMAND_FAILED:
            retry_failed_task_command(db, enqueued.command_id)
        return _CancelCommand(command_db_id=enqueued.command_id)


def _enqueue_cancel(db: Session, task: Task, target_state_version: int) -> Any:
    task_id = int(task.id)
    return enqueue_task_command(
        db,
        task_id=task_id,
        actor_user_id=int(task.user_id) if task.user_id is not None else None,
        # Keyed on the state read here: a repeat against the same state
        # is the same command, and a repeat after the state moved is a
        # new one aimed at the new state.
        command_id=f"cancel:{task_id}:{target_state_version}",
        kind=TaskCommandKind.CANCEL,
        payload={
            "agent_id": int(task.agent_id),
            "target_state_version": target_state_version,
            "scope": SDK_COMMAND_SCOPE,
        },
    )


def _load_result_sync(*, task_id: int, scope: Any) -> SdkTaskCancelResult:
    """Read the task as it stands once the command is over or overdue.

    Same read order as ``_prepare_cancel_sync``: a turn queued since then is
    not described by the row, so the row's outcome is not reported for it.
    """

    from .task_start import resolve_sdk_task
    from .task_start_protocol import has_queued_start

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        queued = has_queued_start(db, task_id)
        result = _result(resolve_sdk_task(task_id, scope, db))
        if queued and result.cancelled:
            raise SdkTaskCancelBusy(
                "A new turn of this task was queued while the cancel was "
                "applied; retry."
            )
        return result


async def request_sdk_task_cancel(*, task_id: int, scope: Any) -> SdkTaskCancelResult:
    """Authorize and cancel one SDK task, waiting a bounded time for it.

    ``scope`` is the calling key's ``SdkTaskScope``. A task outside it, a
    missing one and a non-SDK one raise the same ``TaskStartRejected`` the
    other task endpoints answer 404 with, before anything is read about the
    task's state.

    Returns ``cancelled=True`` once the task is the outcome of a
    cancellation, including when it already was. Returns ``cancelled=False``
    when the command was persisted but not applied within
    ``SDK_CANCEL_REQUEST_WAIT_SECONDS``; calling again reaches the same
    command.
    """

    prepared = await run_db_io_cancellation_safe(
        lambda: _prepare_cancel_sync(task_id=task_id, scope=scope)
    )
    if isinstance(prepared, SdkTaskCancelResult):
        return prepared

    from .task_command_execution import execute_durable_task_command

    # Applied here when this process owns the run; otherwise the owner's
    # dispatcher picks the durable row up and the poll below observes it.
    await dispatch_one_task_command(
        execute_durable_task_command,
        command_db_id=prepared.command_db_id,
    )
    deadline = monotonic() + SDK_CANCEL_REQUEST_WAIT_SECONDS
    while True:
        stored = await run_db_io_cancellation_safe(
            lambda: load_task_command(prepared.command_db_id)
        )
        if stored is not None and stored.status in {COMMAND_COMPLETED, COMMAND_FAILED}:
            result = await run_db_io_cancellation_safe(
                lambda: _load_result_sync(task_id=task_id, scope=scope)
            )
            # The row decides, not the command: a command that lost to
            # another cancel of the same run still finds the task cancelled.
            if result.cancelled:
                return result
            if (
                stored.status == COMMAND_FAILED
                and stored.rejection_reason == "stale_run"
            ) or stored.status == COMMAND_COMPLETED:
                raise SdkTaskCancelBusy(
                    "The task changed state while the cancel was applied; retry."
                )
            raise SdkTaskCancelFailed(stored.error or "cancel command failed")
        if monotonic() >= deadline:
            return await run_db_io_cancellation_safe(
                lambda: _load_result_sync(task_id=task_id, scope=scope)
            )
        await asyncio.sleep(_COMMAND_POLL_SECONDS)


def _load_sdk_task(db: Session, *, task_id: int, agent_id: int) -> Task:
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.agent_id == agent_id,
            Task.source == SDK_TASK_SOURCE,
        )
        .populate_existing()
        .first()
    )
    if task is None:
        raise TaskCommandRejected(
            f"task {task_id} is not an SDK task of agent {agent_id}",
            reason="task_not_found",
        )
    return task


def _is_settled_sdk_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Whether this exact target already reached its cancel outcome.

    True for the command's own finalize replayed, and for the settlement of
    the run this command cancelled: both leave the target run FAILED at the
    command's state version or one past it, carrying the cancellation
    sentence. A run that failed on its own carries another text and is not
    this command's to report.
    """

    return (
        is_sdk_cancelled(task.status, task.error_message)
        and _task_run_id(task) == expected_run_id
        and int(task.state_version or 0)
        in {expected_state_version, expected_state_version + 1}
    )


def _stale(task: Task, expected_run_id: str | None, expected_state_version: int) -> str:
    return (
        f"task {task.id} changed from run/version "
        f"{expected_run_id}/{expected_state_version} to "
        f"{_task_run_id(task)}/{int(task.state_version or 0)}/{task.status.value}"
    )


def _load_cancelable_sdk_task_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Return whether the exact cancel target is already settled."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _load_sdk_task(db, task_id=task_id, agent_id=agent_id)
        if _is_settled_sdk_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return True
        if (
            _task_run_id(task) != expected_run_id
            or int(task.state_version or 0) != expected_state_version
            or task.status != TaskStatus.RUNNING
        ):
            raise TaskCommandRejected(
                _stale(task, expected_run_id, expected_state_version),
                reason="stale_run",
            )
        return False


def _close_cancelled_turn_no_commit(db: Session, task: Task) -> str | None:
    """Stage what a failed run's settlement leaves beside the terminal row.

    The assistant line mirrors the settlement's own -- the generic failure
    text under the task-failure provenance -- so a reader of the transcript
    cannot tell which writer ended the run. The delivery row of the turn is
    closed as ``dispatched``: its message may have been consumed, so it must
    not be offered for a resend.

    The running turn is the newest pending user row, which is sound only
    because a RUNNING task accepts no new turn. Returns that turn's id.
    """

    from .assistant_history_safety import TASK_FAILURE_MESSAGE_TYPE
    from .chat_history_service import (
        DELIVERY_DISPATCHED,
        DELIVERY_PENDING,
        mark_user_message_delivery,
        persist_assistant_message_no_commit,
    )
    from .client_error_messages import CLIENT_SAFE_TASK_FAILURE

    task_id = int(task.id)
    row = (
        db.query(TaskChatMessage.turn_id)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "user",
            TaskChatMessage.delivery_status == DELIVERY_PENDING,
        )
        .order_by(TaskChatMessage.id.desc())
        .first()
    )
    turn_id = str(row[0]) if row is not None and row[0] is not None else None
    if task.user_id is not None:
        persist_assistant_message_no_commit(
            db,
            task_id=task_id,
            user_id=int(task.user_id),
            content=CLIENT_SAFE_TASK_FAILURE,
            message_type=TASK_FAILURE_MESSAGE_TYPE,
            turn_id=turn_id,
        )
    if turn_id is not None:
        mark_user_message_delivery(
            db, task_id=task_id, turn_id=turn_id, status=DELIVERY_DISPATCHED
        )
    return turn_id


def _finalize_sdk_cancel_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Persist cancellation for one exact target; return whether this wrote it.

    Raises ``StaleTaskRunError`` before any write, or after a rollback, when
    the target is neither still running as read nor already settled.
    """

    from .task_coordinator_runtime import current_task_coordinator
    from .task_coordinator_service import (
        lock_task_lease_no_commit,
        task_lease_predicate,
    )
    from .task_execution_event_writer import stage_result_fact_no_commit
    from .task_lease_service import task_settlement_ownership_values
    from .task_orchestrator import sync_trigger_run_status
    from .workforce_runtime import sync_workforce_run_status

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        coordinator = current_task_coordinator(task_id)
        if coordinator is not None:
            assert coordinator.lease is not None
            if not lock_task_lease_no_commit(db, coordinator.lease):
                raise StaleTaskRunError(f"task {task_id} cancellation owner changed")
        task = _load_sdk_task(db, task_id=task_id, agent_id=agent_id)
        if _is_settled_sdk_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            db.rollback()
            return False
        if (
            _task_run_id(task) != expected_run_id
            or int(task.state_version or 0) != expected_state_version
            or task.status != TaskStatus.RUNNING
        ):
            db.rollback()
            raise StaleTaskRunError(
                _stale(task, expected_run_id, expected_state_version)
            )

        statement = (
            update(Task)
            .where(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == SDK_TASK_SOURCE,
                func.coalesce(Task.state_version, 0) == expected_state_version,
                Task.status == TaskStatus.RUNNING,
            )
            .values(
                status=TaskStatus.FAILED,
                control_state=TaskControlState.FAILED.value,
                state_version=expected_state_version + 1,
                **task_settlement_ownership_values(task_id),
                output=None,
                error_message=SDK_CANCEL_ERROR_MESSAGE,
            )
        )
        if expected_run_id is None:
            statement = statement.where(Task.run_id.is_(None))
        else:
            statement = statement.where(Task.run_id == expected_run_id)
        if coordinator is not None:
            assert coordinator.lease is not None
            statement = statement.where(task_lease_predicate(coordinator.lease))

        # ``task`` is already in the identity map, so RETURNING hands back
        # that same object; populate_existing refreshes its attributes.
        updated = db.execute(
            statement.returning(Task).execution_options(
                synchronize_session=False,
                populate_existing=True,
            )
        ).scalar_one_or_none()
        if updated is None:
            db.rollback()
            raise StaleTaskRunError(
                f"task {task_id} changed while finalizing cancel target "
                f"{expected_run_id}/{expected_state_version}"
            )
        sync_workforce_run_status(db, updated, TaskStatus.FAILED)
        sync_trigger_run_status(db, updated, TaskStatus.FAILED)
        turn_id = _close_cancelled_turn_no_commit(db, updated)
        stage_result_fact_no_commit(
            db,
            updated,
            sdk_cancel_result_fact(SDK_CANCEL_ERROR_MESSAGE),
            turn_id=turn_id,
        )
        db.commit()
        return True


async def _broadcast_sdk_cancel_terminal_event(task_id: int) -> None:
    """Wake the task's event readers once its cancellation is durable.

    The cancelled run's settlement does not announce a cancellation, and a
    reader of the task's event stream would otherwise learn of it only from
    its own periodic row read.
    """

    from .task_event_display import publish_task_result
    from .task_execution import create_terminal_task_error_event

    try:
        await publish_task_result(
            create_terminal_task_error_event(task_id, SDK_CANCEL_ERROR_MESSAGE),
            task_id,
        )
    except Exception:
        # The terminal row is already committed; a failed notification must
        # not turn that durable success into a retried command.
        logger.warning(
            "task %s cancellation was committed but its terminal broadcast failed",
            task_id,
            exc_info=True,
        )


async def cancel_sdk_task_unserialized(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> None:
    """Cancel one exact durable-command target while the caller owns its gate.

    Every expected failure leaves as ``TaskCommandRejected`` or
    ``StaleTaskRunError`` (which the dispatcher rejects as ``stale_run``),
    so the command ends at once with its reason instead of being retried.
    """

    already_settled = await run_db_io_cancellation_safe(
        lambda: _load_cancelable_sdk_task_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
    )
    if not already_settled:
        from .task_execution import background_task_manager
        from .task_orchestrator import invalidate_task_cache_best_effort

        # Returns once the run coroutine has unwound, which normally means
        # its own settlement already wrote the cancelled row.
        await background_task_manager.cancel_task(
            task_id, timeout_seconds=SDK_CANCEL_WAIT_SECONDS
        )

        def finalize() -> bool:
            return _finalize_sdk_cancel_sync(
                task_id=task_id,
                agent_id=agent_id,
                expected_run_id=expected_run_id,
                expected_state_version=expected_state_version,
            )

        try:
            wrote = await run_db_io_cancellation_safe(finalize)
        except StaleTaskRunError:
            # A settlement can commit between this command's read and its
            # conditional UPDATE. Every stale result is raised before any
            # write or after a rollback, so one more pass applies the same
            # exact checks: a settled cancel now matches, and any other
            # change to the target is still rejected.
            wrote = await run_db_io_cancellation_safe(finalize)
        if wrote:
            invalidate_task_cache_best_effort(task_id)
            # No settlement wrote the row, so no coroutine had unwound here.
            # One that registered since the call above is stopped now; its
            # settlement finds the row terminal and writes nothing.
            await background_task_manager.cancel_task(
                task_id, timeout_seconds=SDK_CANCEL_WAIT_SECONDS
            )
    await _broadcast_sdk_cancel_terminal_event(task_id)
