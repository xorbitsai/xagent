"""Cancel one SDK-source task through the durable command channel.

``POST /v1/chat/tasks/{task_id}/cancel`` is split in two halves that live
here. The request half authorizes the caller, refuses a target that cannot
be cancelled, persists a CANCEL command and polls a bounded time for it.
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

A turn still queued to start is a target of its own: until its START is
handed off the task row shows the turn before it, so the request names the
queued run and the core judges that run, not the row. Cancelling it before
it starts fails its START and, for an appended turn, leaves the row -- the
previous turn's result -- untouched. A queued START is a target only while
it can still hand off (``start_can_hand_off``): one the START consumer would
reject because the row moved on is treated as absent, and the row is judged
as it stands. A repeat of a cancel that withdrew the newest turn rejoins
that cancel's command and is answered for that turn
(``_cancel_of_newest_turn``), since the row no longer shows it.

A task waiting for the caller's answer and a paused one have no run
coroutine to stop: the core writes their terminal row itself, after closing
the pending question or stopping the automatic resume the paused run was
waiting for.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from ..models.chat_message import TaskChatMessage
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_command import TaskExecutionCommand
from .db_runtime import run_db_io_cancellation_safe
from .task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PENDING,
    COMMAND_PROCESSING,
    EnqueuedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
    TaskCommandTaskMissing,
    dispatch_one_task_command,
    enqueue_task_command,
    load_task_command,
    retry_failed_task_command,
)
from .task_execution_admission import (
    STOPPED_BEFORE_START,
    STOPPED_BEFORE_START_REASON,
    waiting_admission,
)
from .task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    task_control_snapshot,
)

if TYPE_CHECKING:
    # ``task_start`` imports the orchestrator, which imports this module.
    from .task_start import SdkTaskScope

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

# Read by a caller whose cancel was aimed at a run that is no longer the
# task's current one.
_AIMED_RUN_ENDED_MESSAGE = "The run this cancel was aimed at has already ended."

_TERMINAL_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED})

# The statuses of a run a cancel command can end.
_CANCELLABLE_STATUSES = frozenset(
    {TaskStatus.RUNNING, TaskStatus.WAITING_FOR_USER, TaskStatus.PAUSED}
)


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
    """The run already ended on its own; no retry can cancel it.

    The message, when set, is what the caller reads.
    """


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
    # The run the command was persisted against: the one this request
    # answers for, whatever run the row shows by the time it is read back.
    # For a turn queued to start it is that turn's run, which the row does
    # not show until the START is handed off.
    aimed_run_id: str | None
    aimed_at_queued_turn: bool


def _task_run_id(task: Task) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _paused_reads_failed(db: Session, task: Task) -> bool:
    """Whether the task read reports this PAUSED row as ``failed``.

    The row stays PAUSED; automatic resume gave up, or the turn appended to
    it could not be started, and ``status_only_terminal_view`` is what the
    read surface derives ``failed`` from.
    """

    from .task_auto_resume import status_only_terminal_view

    if (
        task.status != TaskStatus.PAUSED
        or task.control_state != TaskControlState.PAUSED.value
    ):
        return False
    return status_only_terminal_view(db, int(task.id)) is not None


def _result(
    db: Session,
    task: Task,
    *,
    cancelled: bool,
    cancelled_run_id: str | None = None,
    queued: bool = False,
) -> SdkTaskCancelResult:
    """The task row as the cancel response reports it.

    ``cancelled`` is the caller's verdict on the run it was asked about,
    which is not always the run the row holds. The status is the one the task
    read reports: a completed or failed row reads as ``pending`` with no code
    while a START is waiting (``queued``; ``QUEUED_START_STATUSES``), and a
    paused row reads as ``failed`` when ``_paused_reads_failed``.
    """

    from .task_start_protocol import QUEUED_START_STATUSES

    shown_pending = queued and task.status in QUEUED_START_STATUSES
    if shown_pending:
        status = TaskStatus.PENDING.value
    elif _paused_reads_failed(db, task):
        status = TaskStatus.FAILED.value
    else:
        status = task.status.value
    return SdkTaskCancelResult(
        task_id=int(task.id),
        status=status,
        run_id=_task_run_id(task),
        state_version=int(task.state_version or 0),
        cancelled_run_id=cancelled_run_id if cancelled else None,
        error_code=(
            None
            if shown_pending
            else sdk_cancel_error_code(task.status, task.error_message)
        ),
        cancelled=cancelled,
    )


def _row_result(db: Session, task: Task) -> SdkTaskCancelResult:
    """The result for a command aimed at the run the row holds."""

    return _result(
        db,
        task,
        cancelled=is_sdk_cancelled(task.status, task.error_message),
        cancelled_run_id=_task_run_id(task),
    )


def _queued_start(db: Session, task_id: int) -> TaskExecutionCommand | None:
    """The START of ``task_id`` not handed off yet, as ``has_queued_start``
    counts it (at most one exists)."""

    from .task_start_protocol import queued_start_conditions

    return (
        db.query(TaskExecutionCommand)
        .filter(*queued_start_conditions(task_id))
        .order_by(TaskExecutionCommand.id.desc())
        .first()
    )


def _cancel_of_newest_turn(db: Session, task: Task) -> TaskExecutionCommand | None:
    """This endpoint's cancel command aimed at the task's newest START, if any.

    A cancel of a turn queued to start is persisted under an identity made of
    that turn's run and the version the START was accepted against
    (``_cancel_command_id``), so the newest START names the one command a
    cancel of that turn can be. Once the command has withdrawn the turn, the
    row still shows the turn before it (or, for a new task, holds no run) and
    no START is queued, so a repeat finds the turn only through this
    identity -- whatever state the command is in, including while it is
    still being applied after the withdrawal was committed. A START stopped
    some other way has no such command and is not found.

    The command answers only while the row is still as that cancel found or
    left it: withdrawing an appended turn does not move the row, so the
    command's version is still the row's. A row that moved since -- for
    example the turn before was resumed and runs again -- is answered from
    the row, as any other call would be. A new task is the exception: its
    cancel wrote the row itself, which then holds no run and reads as
    cancelled. A task paused on request whose appended turn was withdrawn
    therefore keeps answering that withdrawal, and is not itself cancelled,
    until the row moves.
    """

    newest = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == int(task.id),
            TaskExecutionCommand.kind == TaskCommandKind.START.value,
        )
        .order_by(TaskExecutionCommand.id.desc())
        .populate_existing()
        .first()
    )
    if (
        newest is None
        or newest.target_run_id is None
        or newest.target_state_version is None
    ):
        return None
    run_id = str(newest.target_run_id)
    if run_id == _task_run_id(task):
        return None
    accepted_version = int(newest.target_state_version)
    cancelled_new_task = _task_run_id(task) is None and is_sdk_cancelled(
        task.status, task.error_message
    )
    if not cancelled_new_task and accepted_version != int(task.state_version or 0):
        return None
    cancel = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == int(task.id),
            TaskExecutionCommand.command_id
            == _cancel_command_id(int(task.id), accepted_version, run_id),
            TaskExecutionCommand.kind == TaskCommandKind.CANCEL.value,
        )
        .populate_existing()
        .one_or_none()
    )
    if cancel is None:
        return None
    payload: dict[str, Any] = cancel.payload if isinstance(cancel.payload, dict) else {}
    if (
        payload.get("scope") != SDK_COMMAND_SCOPE
        or payload.get("queued_run_id") != run_id
    ):
        return None
    return cancel


def _unapplied_cancel_of_row(db: Session, task: Task) -> TaskExecutionCommand | None:
    """This endpoint's cancel aimed at the row as it stands, while unapplied.

    Found by the identity ``_enqueue_cancel`` gives a cancel of the run the
    row holds at its current version (``_cancel_command_id`` with no queued
    run), and only while it is pending or processing. A repeat that finds
    it rejoins it instead of judging the row: the row may already read as
    ended -- a paused run whose automatic resume gave up -- while the
    command will still cancel it.
    """

    cancel = (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == int(task.id),
            TaskExecutionCommand.command_id
            == _cancel_command_id(int(task.id), int(task.state_version or 0), None),
            TaskExecutionCommand.kind == TaskCommandKind.CANCEL.value,
            TaskExecutionCommand.status.in_((COMMAND_PENDING, COMMAND_PROCESSING)),
        )
        .populate_existing()
        .one_or_none()
    )
    if cancel is None:
        return None
    payload: dict[str, Any] = cancel.payload if isinstance(cancel.payload, dict) else {}
    if payload.get("scope") != SDK_COMMAND_SCOPE:
        return None
    return cancel


def _start_of_run(
    db: Session, task_id: int, run_id: str
) -> TaskExecutionCommand | None:
    """The START that opens exactly ``run_id``, whatever its state.

    Matched on the run so that a later turn's START is never taken for the
    one a cancel was aimed at.
    """

    return (
        db.query(TaskExecutionCommand)
        .filter(
            TaskExecutionCommand.task_id == task_id,
            TaskExecutionCommand.kind == TaskCommandKind.START.value,
            TaskExecutionCommand.target_run_id == run_id,
        )
        .order_by(TaskExecutionCommand.id.desc())
        .populate_existing()
        .first()
    )


def _start_was_stopped(start: TaskExecutionCommand | None) -> bool:
    if start is None or start.status != COMMAND_FAILED:
        return False
    result: dict[str, Any] = start.result if isinstance(start.result, dict) else {}
    return bool(result.get("rejection_reason") == STOPPED_BEFORE_START_REASON)


def _ended_without_cancel(db: Session, task: Task) -> bool:
    """Whether the task's run ended and was not an SDK cancellation.

    Covers a COMPLETED or FAILED row and a paused row the read surface
    reports as failed (``_paused_reads_failed``).
    """

    if is_sdk_cancelled(task.status, task.error_message):
        return False
    return task.status in _TERMINAL_STATUSES or _paused_reads_failed(db, task)


def _prepare_cancel_sync(
    *, task_id: int, scope: SdkTaskScope
) -> SdkTaskCancelResult | _CancelCommand:
    """Authorize, classify the target and persist its command, in one read.

    The target is the queued START that can still hand off, else the row. A
    repeat of a cancel that already withdrew the newest turn finds no such
    START and rejoins that cancel's command (``_cancel_of_newest_turn``):
    the same command to poll while it is still being applied, the same
    answer once it completed. A repeat of a cancel of the row that is still
    unapplied at the row's version rejoins it too
    (``_unapplied_cancel_of_row``), before the row is judged as ended.
    """

    from .task_start import resolve_sdk_task
    from .task_start_protocol import start_can_hand_off

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # Read in the order the public task read uses: until a queued START
        # is handed off, the row still shows the previous turn, so judging
        # the row alone would answer for a turn the caller is not asking
        # about.
        queued = _queued_start(db, task_id)
        task = resolve_sdk_task(task_id, scope, db)
        if queued is not None and not start_can_hand_off(queued, task):
            # Never starts: the row moved on after the START was accepted,
            # so the turn it would open is not what a cancel can reach.
            queued = None
        if queued is not None:
            if queued.target_run_id is None:
                raise SdkTaskCancelBusy(
                    "A turn of this task is queued to start and cannot be "
                    "cancelled yet; retry once it is running."
                )
            return _stage_cancel(db, task, queued_run_id=str(queued.target_run_id))
        # A repeat of the cancel that withdrew the newest turn rejoins that
        # command and is answered for that turn, not for whatever the row
        # shows of the turn before it: polled like the first call while the
        # command is still being applied, answered as cancelled once it
        # completed. A failed command is not a cancellation, so the row
        # decides.
        cancel = _cancel_of_newest_turn(db, task)
        if cancel is not None:
            withdrawn_run_id = str(cancel.payload["queued_run_id"])
            if cancel.status == COMMAND_COMPLETED:
                return _result(
                    db, task, cancelled=True, cancelled_run_id=withdrawn_run_id
                )
            if cancel.status in {COMMAND_PENDING, COMMAND_PROCESSING}:
                return _CancelCommand(
                    command_db_id=int(cancel.id),
                    aimed_run_id=withdrawn_run_id,
                    aimed_at_queued_turn=True,
                )
        if is_sdk_cancelled(task.status, task.error_message):
            return _row_result(db, task)
        # A repeat of a cancel still unapplied against the row as it stands
        # rejoins it, even when the row already reads as ended: that command
        # decides, as it did for the first call.
        unapplied = _unapplied_cancel_of_row(db, task)
        if unapplied is not None:
            return _CancelCommand(
                command_db_id=int(unapplied.id),
                aimed_run_id=_task_run_id(task),
                aimed_at_queued_turn=False,
            )
        if _ended_without_cancel(db, task):
            raise SdkTaskNotCancellable
        if task.status not in _CANCELLABLE_STATUSES:
            raise SdkTaskCancelBusy(
                f"A task that is {task.status.value} cannot be cancelled yet."
            )
        if task.status == TaskStatus.WAITING_FOR_USER and _answer_in_flight(db, task):
            # The answer owns the waiting run until it resumes it; the
            # resumed run is then cancellable.
            raise SdkTaskCancelBusy(
                "An answer is being applied to this task; retry once it runs."
            )
        return _stage_cancel(db, task, queued_run_id=None)


def _answer_in_flight(db: Session, task: Task) -> bool:
    """Whether a reply already claimed the waiting run or is queued to, by
    the reply path's own acceptance rules."""

    from .task_question_cancel import _pending_resume_command_exists
    from .task_resume import _RESUMABLE_CONTROL_STATES

    if task.control_state not in _RESUMABLE_CONTROL_STATES:
        return True
    return bool(db.scalar(select(_pending_resume_command_exists(int(task.id)))))


def _stage_cancel(
    db: Session, task: Task, *, queued_run_id: str | None
) -> _CancelCommand:
    """Persist the cancel command for the state just read.

    ``queued_run_id`` aims it at the turn queued to start as that run;
    otherwise it is aimed at the run the row holds.
    """

    from .task_start import TaskStartRejected

    target_state_version = int(task.state_version or 0)
    # Read before the insert commits and expires ``task``: this is the run
    # the command targets.
    aimed_run_id = queued_run_id if queued_run_id is not None else _task_run_id(task)
    try:
        enqueued = _enqueue_cancel(
            db, task, target_state_version, queued_run_id=queued_run_id
        )
    except TaskCommandTaskMissing as exc:
        # Deleted between the caller's read and the insert.
        raise TaskStartRejected("task_not_found") from exc
    if not enqueued.payload_matches:
        raise SdkTaskCancelBusy("Another command holds this cancel identity; retry.")
    if enqueued.status == COMMAND_FAILED:
        retry_failed_task_command(db, enqueued.command_id)
    return _CancelCommand(
        command_db_id=enqueued.command_id,
        aimed_run_id=aimed_run_id,
        aimed_at_queued_turn=queued_run_id is not None,
    )


def _cancel_command_id(
    task_id: int, target_state_version: int, queued_run_id: str | None
) -> str:
    """The identity of the cancel command for the state it is aimed at.

    Keyed on the state the request read: a repeat against the same state is
    the same command, and a repeat after the state moved is a new one aimed
    at the new state. An appended turn does not move the row while it is
    queued, so its run is part of the key; once such a turn is withdrawn,
    ``_cancel_of_newest_turn`` finds the command again by this key.
    """

    command_id = f"cancel:{task_id}:{target_state_version}"
    if queued_run_id is not None:
        command_id = f"{command_id}:{queued_run_id}"
    return command_id


def _enqueue_cancel(
    db: Session,
    task: Task,
    target_state_version: int,
    *,
    queued_run_id: str | None,
) -> EnqueuedTaskCommand:
    task_id = int(task.id)
    command_id = _cancel_command_id(task_id, target_state_version, queued_run_id)
    payload: dict[str, Any] = {
        "agent_id": int(task.agent_id),
        "target_state_version": target_state_version,
        "scope": SDK_COMMAND_SCOPE,
    }
    if queued_run_id is not None:
        payload["queued_run_id"] = queued_run_id
    return enqueue_task_command(
        db,
        task_id=task_id,
        actor_user_id=int(task.user_id) if task.user_id is not None else None,
        command_id=command_id,
        kind=TaskCommandKind.CANCEL,
        payload=payload,
        # A queued turn's cancel targets that turn's run, not the run the row
        # still shows. A process that does not read ``queued_run_id`` then
        # finds the run changed and rejects the command without a write,
        # instead of taking an earlier cancelled run for this one.
        target_run_id=queued_run_id,
    )


def _load_result_sync(
    *,
    task_id: int,
    scope: SdkTaskScope,
    aimed_run_id: str | None,
    aimed_at_queued_turn: bool,
    command_completed: bool,
    command_db_id: int | None = None,
) -> SdkTaskCancelResult:
    """Read the outcome once the command is over or overdue.

    ``aimed_run_id`` is the run the command was persisted against. A
    completed command cancelled that run, so the answer names it as
    ``cancelled_run_id`` while every other field describes the task now,
    which may already show a later run. Otherwise the answer is about the
    aimed run only:

    - a turn queued to start is not on the row until its START is handed
      off. While its START waits the command is still pending; a START
      stopped before it began is that turn cancelled; any other START
      outcome means the turn is over. A START the row moved away from is
      not stopped by the command (``stale_run``): while it is still pending
      the call answers busy, or 202 at the deadline, and once its consumer
      rejects it the turn reads as over.
    - a row that shows another run means the aimed one is over, and
      retrying would aim at the new one, so it raises
      ``SdkTaskNotCancellable`` with ``_AIMED_RUN_ENDED_MESSAGE``.
    - an aimed run that ended on its own raises ``SdkTaskNotCancellable``
      exactly as the request would have on arrival, whether or not a later
      turn is queued behind it. The exception is a paused run that reads
      ``failed`` while the command is still pending or processing (the
      deadline's read, which passes ``command_db_id``) and the row is still
      at the version the command is aimed at: its automatic resume
      gave up during the wait, which does not move that version or stop the
      command from cancelling it, so the call answers that the cancel is
      still being applied.

    A START waiting for a later turn reads as ``pending`` with no code, and
    a paused row that has stopped for good reads as ``failed``, as on the
    task read.
    """

    from .task_start import resolve_sdk_task

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # Same read order as ``_prepare_cancel_sync``.
        queued = _queued_start(db, task_id) is not None
        task = resolve_sdk_task(task_id, scope, db)
        if command_completed:
            return _result(
                db, task, cancelled=True, cancelled_run_id=aimed_run_id, queued=queued
            )
        if _task_run_id(task) != aimed_run_id:
            if aimed_at_queued_turn and aimed_run_id is not None:
                start = _start_of_run(db, task_id, aimed_run_id)
                if _start_was_stopped(start):
                    return _result(
                        db,
                        task,
                        cancelled=True,
                        cancelled_run_id=aimed_run_id,
                        queued=queued,
                    )
                if start is not None and start.status in {
                    COMMAND_PENDING,
                    COMMAND_PROCESSING,
                }:
                    return _result(db, task, cancelled=False, queued=queued)
            raise SdkTaskNotCancellable(_AIMED_RUN_ENDED_MESSAGE)
        if _ended_without_cancel(db, task) and not _still_cancels_paused_row(
            db, task, command_db_id
        ):
            raise SdkTaskNotCancellable
        return _result(
            db,
            task,
            cancelled=is_sdk_cancelled(task.status, task.error_message),
            cancelled_run_id=aimed_run_id,
            queued=queued,
        )


def _still_cancels_paused_row(
    db: Session, task: Task, command_db_id: int | None
) -> bool:
    """Whether the unapplied command ``command_db_id`` still cancels this
    paused row: it is the pending or processing cancel aimed at the version
    the row is at now, the same command a repeat call rejoins.

    The command is matched by its identity, which carries the version its
    executor fences on. The command row's own version column is read when
    the command is staged and can already be past the version the request
    aimed at, so it does not say what the command will cancel."""

    if command_db_id is None or task.status != TaskStatus.PAUSED:
        return False
    cancel = _unapplied_cancel_of_row(db, task)
    return cancel is not None and int(cancel.id) == command_db_id


async def request_sdk_task_cancel(
    *, task_id: int, scope: SdkTaskScope
) -> SdkTaskCancelResult:
    """Authorize and cancel one SDK task, polling a bounded time for it.

    The poll is bounded; a cancel applied in this process first waits for
    the cancelled run to stop, which has no upper bound.

    ``scope`` is the calling key's ``SdkTaskScope``. A task outside it, a
    missing one and a non-SDK one raise the same ``TaskStartRejected`` the
    other task endpoints answer 404 with, before anything is read about the
    task's state.

    Returns ``cancelled=True`` once the task is the outcome of a
    cancellation, including when it already was, once the command that
    cancelled the aimed run completed (``cancelled_run_id``), and when an
    earlier call's command that withdrew the newest turn completed, which
    names that turn. Returns ``cancelled=False`` when the command was
    persisted but not applied within ``SDK_CANCEL_REQUEST_WAIT_SECONDS``;
    calling again reaches the same command, including after it withdrew a
    queued turn and before it is marked completed. Raises
    ``SdkTaskNotCancellable`` when the aimed run ended without being
    cancelled, whichever way the wait ends, except a paused run whose
    automatic resume gave up while the command is still unapplied at the
    deadline and the row is still at the version the command is aimed at: the command
    still cancels it, so that answer is ``cancelled=False``, and calling
    again rejoins the same command.
    """

    prepared = await run_db_io_cancellation_safe(
        lambda: _prepare_cancel_sync(task_id=task_id, scope=scope)
    )
    if isinstance(prepared, SdkTaskCancelResult):
        return prepared

    from .task_command_execution import execute_durable_task_command

    aimed_run_id = prepared.aimed_run_id
    deadline = monotonic() + SDK_CANCEL_REQUEST_WAIT_SECONDS
    # Applied here when this process owns the run; otherwise the owner's
    # dispatcher picks the durable row up and the poll below observes it.
    await dispatch_one_task_command(
        execute_durable_task_command,
        command_db_id=prepared.command_db_id,
    )
    while True:
        stored = await run_db_io_cancellation_safe(
            lambda: load_task_command(prepared.command_db_id)
        )
        if stored is not None and stored.status in {COMMAND_COMPLETED, COMMAND_FAILED}:
            command_completed = stored.status == COMMAND_COMPLETED
            result = await run_db_io_cancellation_safe(
                lambda: _load_result_sync(
                    task_id=task_id,
                    scope=scope,
                    aimed_run_id=aimed_run_id,
                    aimed_at_queued_turn=prepared.aimed_at_queued_turn,
                    command_completed=command_completed,
                )
            )
            if command_completed:
                return result
            # The row decides, not the command: a command that lost to
            # another cancel of the same run still finds the task cancelled.
            if result.cancelled:
                return result
            if stored.rejection_reason == "stale_run":
                raise SdkTaskCancelBusy(
                    "The task changed state while the cancel was applied; retry."
                )
            raise SdkTaskCancelFailed(stored.error or "cancel command failed")
        if monotonic() >= deadline:
            return await run_db_io_cancellation_safe(
                lambda: _load_result_sync(
                    task_id=task_id,
                    scope=scope,
                    aimed_run_id=aimed_run_id,
                    aimed_at_queued_turn=prepared.aimed_at_queued_turn,
                    command_completed=False,
                    command_db_id=prepared.command_db_id,
                )
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
    the run this command cancelled: both leave the target run FAILED at or
    past the command's state version (a paused run's stopped automatic
    resume moves it once more), carrying the cancellation sentence. A run
    that failed on its own carries another text and is not this command's
    to report.
    """

    return (
        is_sdk_cancelled(task.status, task.error_message)
        and _task_run_id(task) == expected_run_id
        and int(task.state_version or 0) >= expected_state_version
    )


def _is_exact_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
    expected_status: TaskStatus | None,
) -> bool:
    """Whether the row is still the target this command was aimed at.

    ``expected_status`` ``None`` accepts any status a cancel can end: the
    command was persisted against the run and version alone.
    """

    if expected_status is None:
        status_matches = task.status in _CANCELLABLE_STATUSES
    else:
        status_matches = task.status == expected_status
    return (
        _task_run_id(task) == expected_run_id
        and int(task.state_version or 0) == expected_state_version
        and status_matches
    )


def _stale(task: Task, expected_run_id: str | None, expected_state_version: int) -> str:
    return (
        f"task {task.id} changed from run/version "
        f"{expected_run_id}/{expected_state_version} to "
        f"{_task_run_id(task)}/{int(task.state_version or 0)}/{task.status.value}"
    )


def _carries_display_events(task: Task, cancelled_from: TaskStatus | None) -> bool:
    """Whether the terminal broadcast should carry the turn's display events.

    Only a version-2 task has them, and only a run that ran has the result
    fact they are read from. A running, waiting or paused run gets one from
    whichever writer ended it (``None`` is one of those, read before its
    status was known); a new task cancelled before it started has none, and
    publishing them without it fails.
    """

    if cancelled_from == TaskStatus.PENDING:
        return False
    return bool(task.conversation_storage_version == 2)


def _load_cancelable_sdk_task_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
    expected_status: TaskStatus | None,
    terminal_event_state: dict[str, Any],
) -> TaskStatus | None:
    """The status to cancel the exact target from, or ``None`` when it is
    already settled.

    ``expected_status`` is given for a queued turn, whose target was just
    resolved; otherwise any status a cancel can end is accepted as read. A
    settled target fills ``terminal_event_state`` with the control snapshot
    its terminal broadcast carries.
    """

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _load_sdk_task(db, task_id=task_id, agent_id=agent_id)
        if _is_settled_sdk_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            if _carries_display_events(task, expected_status):
                terminal_event_state.update(task_control_snapshot(task).as_dict())
            return None
        if not _is_exact_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
            expected_status=expected_status,
        ):
            raise TaskCommandRejected(
                _stale(task, expected_run_id, expected_state_version),
                reason="stale_run",
            )
        return TaskStatus(task.status)


def _close_cancelled_turn_no_commit(
    db: Session, task: Task, *, was_running: bool
) -> str | None:
    """Stage what a failed run's settlement leaves beside the terminal row.

    The assistant line mirrors the settlement's own -- the generic failure
    text under the task-failure provenance -- so a reader of the transcript
    cannot tell which writer ended the run. The delivery row of the turn is
    closed as ``dispatched``: its message may have been consumed, so it must
    not be offered for a resend.

    The running turn is the newest pending user row, which is sound only
    because a RUNNING task accepts no new turn. A waiting or paused run has
    no such guarantee -- a paused task accepts an append -- and its turn's
    delivery was settled when the run stopped, so only the line is written.
    Returns the closed turn's id.
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
    turn_id: str | None = None
    if was_running:
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


def _stop_auto_resume_no_commit(
    db: Session,
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> int:
    """Stop the automatic resume a paused run waits for; return the version
    the task holds afterwards.

    Without this the sweeper could still dispatch the resume it had
    scheduled. A run with nothing pending -- none was scheduled, or it gave
    up -- is left as it is. Raises ``StaleTaskRunError`` when the row read
    under the lock is no longer the run and version the cancel was aimed
    at, or when the stop lost a race, so the caller reads again.
    """

    from .task_auto_resume import cancel_pending_auto_resume_no_commit

    # The stop requires the task row locked and read under that lock.
    locked = (
        db.query(Task)
        .filter(Task.id == task.id)
        .with_for_update()
        .populate_existing()
        .one()
    )
    if (
        _task_run_id(locked) != expected_run_id
        or int(locked.state_version or 0) != expected_state_version
    ):
        db.rollback()
        raise StaleTaskRunError(_stale(locked, expected_run_id, expected_state_version))
    stopped = cancel_pending_auto_resume_no_commit(db, locked, at="sdk_cancel")
    if stopped.raced:
        db.rollback()
        raise StaleTaskRunError(
            f"task {task.id} automatic resume moved while it was being stopped"
        )
    return int(locked.state_version or 0)


def _finalize_sdk_cancel_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
    expected_status: TaskStatus,
    terminal_event_state: dict[str, Any],
) -> bool:
    """Persist cancellation for one exact target; return whether this wrote it.

    ``expected_status`` is the status the target was read in. PENDING is a
    new task whose START is still queued: it never ran, so it gets the
    terminal row alone and its START's own settlement closes the turn. A
    PAUSED run first has its pending automatic resume stopped, which moves
    the version this write is then fenced on. A WAITING_FOR_USER run must
    still be unanswered, and has its question closed with the row.

    Raises ``StaleTaskRunError`` before any write, or after a rollback, when
    the target is neither still as read nor already settled. When the
    target is settled or written here, it fills ``terminal_event_state``
    with the control snapshot the terminal broadcast carries, once the row
    is durable. The task's cached read is invalidated after this call's own
    commit.
    """

    from .chat_history_service import supersede_legacy_question_rows
    from .task_coordinator_runtime import current_task_coordinator
    from .task_coordinator_service import (
        lock_task_lease_no_commit,
        task_lease_predicate,
    )
    from .task_execution_event_writer import stage_result_fact_no_commit
    from .task_interaction_close import (
        terminate_active_interaction_for_question_cancel,
    )
    from .task_interaction_read import get_pending_interaction_question
    from .task_lease_service import task_settlement_ownership_values
    from .task_orchestrator import (
        invalidate_task_cache_best_effort,
        sync_trigger_run_status,
    )
    from .task_question_cancel import (
        _latest_question_row,
        _materialize_question_step,
        _pending_resume_command_exists,
    )
    from .task_resume import _RESUMABLE_CONTROL_STATES
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
            if _carries_display_events(task, expected_status):
                settled_state = task_control_snapshot(task).as_dict()
            else:
                settled_state = {}
            db.rollback()
            terminal_event_state.update(settled_state)
            return False
        if not _is_exact_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
            expected_status=expected_status,
        ):
            db.rollback()
            raise StaleTaskRunError(
                _stale(task, expected_run_id, expected_state_version)
            )

        fence_version = expected_state_version
        if expected_status == TaskStatus.PAUSED:
            fence_version = _stop_auto_resume_no_commit(
                db,
                task,
                expected_run_id=expected_run_id,
                expected_state_version=expected_state_version,
            )
        statement = (
            update(Task)
            .where(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == SDK_TASK_SOURCE,
                func.coalesce(Task.state_version, 0) == fence_version,
                Task.status == expected_status,
            )
            .values(
                status=TaskStatus.FAILED,
                control_state=TaskControlState.FAILED.value,
                state_version=fence_version + 1,
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
        if expected_status == TaskStatus.WAITING_FOR_USER:
            statement = statement.where(
                Task.control_state.in_(tuple(_RESUMABLE_CONTROL_STATES)),
                ~_pending_resume_command_exists(task_id),
            )

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
        if expected_status == TaskStatus.WAITING_FOR_USER:
            # After the task row's write, which is the lock order the
            # interaction writers require, and in the order abandoning the
            # question uses: the question is read before the rows that carry
            # it are closed, and a question with no trace row is given one
            # so the task's steps still show it.
            question, interactions = get_pending_interaction_question(db, updated)
            question_row = _latest_question_row(db, task_id)
            terminate_active_interaction_for_question_cancel(
                db, task_id=task_id, run_id=expected_run_id
            )
            supersede_legacy_question_rows(db, task_id=task_id)
            _materialize_question_step(
                db,
                task=updated,
                question=question,
                interactions=interactions,
                question_row=question_row,
            )
        sync_workforce_run_status(db, updated, TaskStatus.FAILED)
        sync_trigger_run_status(db, updated, TaskStatus.FAILED)
        if expected_status != TaskStatus.PENDING:
            turn_id = _close_cancelled_turn_no_commit(
                db, updated, was_running=expected_status == TaskStatus.RUNNING
            )
            stage_result_fact_no_commit(
                db,
                updated,
                sdk_cancel_result_fact(SDK_CANCEL_ERROR_MESSAGE),
                turn_id=turn_id,
            )
        event_state = (
            task_control_snapshot(updated).as_dict()
            if _carries_display_events(updated, expected_status)
            else {}
        )
        db.commit()
        terminal_event_state.update(event_state)
        invalidate_task_cache_best_effort(task_id)
        return True


async def _broadcast_sdk_cancel_terminal_event(
    task_id: int, terminal_event_state: dict[str, Any]
) -> None:
    """Wake the task's event readers once its cancellation is durable.

    The cancelled run's settlement does not announce a cancellation, and a
    reader of the task's event stream would otherwise learn of it only from
    its own periodic row read. ``terminal_event_state`` is the control
    snapshot of a version-2 task; its presence is what makes the broadcast
    carry the turn's display events.
    """

    from .task_event_display import publish_task_result
    from .task_execution import create_terminal_task_error_event

    try:
        await publish_task_result(
            {
                **create_terminal_task_error_event(task_id, SDK_CANCEL_ERROR_MESSAGE),
                **terminal_event_state,
            },
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


@dataclass(frozen=True)
class _CancelTarget:
    run_id: str | None
    state_version: int
    status: TaskStatus


def _stop_queued_start_no_commit(
    db: Session, task_id: int, start: TaskExecutionCommand
) -> bool:
    """Fail a START that no executor has claimed, leaving the task row alone.

    Stopped while it is pending and either never attempted or waiting for an
    admission ticket, the same START a cancel is dispatched past
    (``waiting_admission``): one sent back to the admission queue after an
    attempt is still stopped, one an executor claimed and may have acted on
    is not. An append's failure belongs to its command: the row keeps the
    previous turn's result. The turn's delivery row and runtime values are
    released by the START's own settlement. Returns False when the START
    moved first.
    """

    from .task_command_terminal_events import (
        TerminalTaskEventDraft,
        stage_terminal_event,
    )
    from .task_coordinator_runtime import current_task_coordinator
    from .task_coordinator_service import lock_task_lease_no_commit
    from .task_start_consumer import settle_failed_start_no_commit

    coordinator = current_task_coordinator(task_id)
    if coordinator is not None:
        assert coordinator.lease is not None
        if not lock_task_lease_no_commit(db, coordinator.lease):
            raise StaleTaskRunError(f"task {task_id} cancellation owner changed")
    stopped = db.execute(
        update(TaskExecutionCommand)
        .where(
            TaskExecutionCommand.id == start.id,
            TaskExecutionCommand.status == COMMAND_PENDING,
            or_(
                TaskExecutionCommand.attempt_count == 0,
                waiting_admission(TaskExecutionCommand.id),
            ),
        )
        .values(
            status=COMMAND_FAILED,
            error=STOPPED_BEFORE_START,
            completed_at=func.now(),
            result={"rejection_reason": STOPPED_BEFORE_START_REASON},
        )
        .returning(TaskExecutionCommand.id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    if stopped is None:
        return False
    db.refresh(start)
    settle_failed_start_no_commit(db, start)
    stage_terminal_event(
        db,
        command_db_id=int(start.id),
        draft=TerminalTaskEventDraft(message_code=None, resend_safe=False),
    )
    return True


def _resolve_queued_cancel_target_sync(
    *, task_id: int, agent_id: int, queued_run_id: str
) -> _CancelTarget | None:
    """Turn "the turn queued as ``queued_run_id``" into what to cancel now.

    Returns the task-row target to cancel, or ``None`` when nothing is left
    to do: the turn was already cancelled, or its START was stopped before
    it began (by this call, for an appended turn). Raises
    ``TaskCommandRejected`` when the turn ended on its own or never started,
    including when its START can no longer hand off because the row moved
    after it was accepted (checked before anything is written), and
    ``TaskCommandDeferred`` while its START is being handed off.
    """

    from .task_start_protocol import start_can_hand_off

    stale = TaskCommandRejected(
        f"queued run {queued_run_id} of task {task_id} is no longer cancellable",
        reason="stale_run",
    )
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # One more pass covers a START that is handed off, or stopped by
        # another control, between this read and the write below.
        for _attempt in range(2):
            task = _load_sdk_task(db, task_id=task_id, agent_id=agent_id)
            if _task_run_id(task) == queued_run_id:
                # The queued turn started. Fenced on the version read now:
                # the handoff and the run itself moved it since the request.
                # It is cancelled the way its current state is, which may
                # already be waiting for an answer or paused.
                if is_sdk_cancelled(task.status, task.error_message):
                    return None
                if task.status not in _CANCELLABLE_STATUSES:
                    raise stale
                return _CancelTarget(
                    queued_run_id,
                    int(task.state_version or 0),
                    TaskStatus(task.status),
                )
            start = _start_of_run(db, task_id, queued_run_id)
            if _start_was_stopped(start):
                return None
            if start is None or start.status == COMMAND_FAILED:
                raise stale
            if start.status == COMMAND_PROCESSING:
                raise TaskCommandDeferred(
                    f"START of run {queued_run_id} is being handed off"
                )
            if start.status == COMMAND_PENDING:
                if _new_task_already_cancelled(task, start):
                    # A new task's cancel wrote the row and the START is
                    # settled only once that command completes. Applied
                    # again before then -- its completion never committed --
                    # the cancel is already done; the row has moved past the
                    # START, which must not read as the turn having ended.
                    return None
                if not start_can_hand_off(start, task):
                    # The row moved on after the START was accepted, so the
                    # START will never open this run and there is nothing
                    # of the turn to cancel. Nothing is written.
                    raise stale
                if task.status == TaskStatus.PENDING:
                    # A new task that never ran: the row itself is cancelled
                    # and the completed command settles the START.
                    return _CancelTarget(
                        _task_run_id(task),
                        int(task.state_version or 0),
                        TaskStatus.PENDING,
                    )
                if _stop_queued_start_no_commit(db, task_id, start):
                    db.commit()
                    return None
            db.rollback()
        raise stale


def _new_task_already_cancelled(task: Task, start: TaskExecutionCommand) -> bool:
    """Whether ``start`` opens a new task whose row a cancel already wrote."""

    from .task_start_protocol import TaskStartPayload

    if _task_run_id(task) is not None or not is_sdk_cancelled(
        task.status, task.error_message
    ):
        return False
    try:
        payload = TaskStartPayload.model_validate(start.payload)
    except ValueError:
        return False
    return bool(payload.kind == "create")


async def cancel_sdk_task_unserialized(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
    queued_run_id: str | None = None,
) -> None:
    """Cancel one exact durable-command target while the caller owns its gate.

    ``queued_run_id`` aims the command at a turn that was queued when it
    was requested; the run and version to cancel are then read here.

    Every expected failure leaves as ``TaskCommandRejected`` or
    ``StaleTaskRunError`` (which the dispatcher rejects as ``stale_run``),
    so the command ends at once with its reason instead of being retried.
    """

    expected_status: TaskStatus | None = None
    if queued_run_id is not None:
        target = await run_db_io_cancellation_safe(
            lambda: _resolve_queued_cancel_target_sync(
                task_id=task_id, agent_id=agent_id, queued_run_id=queued_run_id
            )
        )
        if target is None:
            return
        expected_run_id = target.run_id
        expected_state_version = target.state_version
        expected_status = target.status

    terminal_event_state: dict[str, Any] = {}
    cancel_from = await run_db_io_cancellation_safe(
        lambda: _load_cancelable_sdk_task_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
            expected_status=expected_status,
            terminal_event_state=terminal_event_state,
        )
    )
    if cancel_from is not None:
        from .task_execution import background_task_manager

        if cancel_from == TaskStatus.RUNNING:
            # Returns once the run coroutine has unwound, which normally
            # means its own settlement already wrote the cancelled row. A
            # waiting or paused run has no coroutine; this core writes its
            # row.
            await background_task_manager.cancel_task(
                task_id, timeout_seconds=SDK_CANCEL_WAIT_SECONDS
            )

        def finalize() -> bool:
            return _finalize_sdk_cancel_sync(
                task_id=task_id,
                agent_id=agent_id,
                expected_run_id=expected_run_id,
                expected_state_version=expected_state_version,
                expected_status=cancel_from,
                terminal_event_state=terminal_event_state,
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
            # No settlement wrote the row, so no coroutine had unwound here.
            # One that registered since the target was read is stopped now;
            # its settlement finds the row terminal and writes nothing.
            await background_task_manager.cancel_task(
                task_id, timeout_seconds=SDK_CANCEL_WAIT_SECONDS
            )
    await _broadcast_sdk_cancel_terminal_event(task_id, terminal_event_state)
