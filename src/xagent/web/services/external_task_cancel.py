"""Cancel execution of one external-source task turn.

The durable command channel already carries A2A cancels; its execution core
(``a2a.py``) loads its target with ``Task.source == "a2a"``, so an external
task cannot travel through it. This module is that core's external-scope
counterpart: the same load / hard-cancel / fenced-finalize shape against
``Task.source == "external"`` rows, with the differences the external
surface needs.

  - The visitor has no other channel to learn the turn ended, so the core
    broadcasts the terminal event itself once its finalize commits. The
    settlement path in ``task_orchestrator`` broadcasts only for setup/run
    errors, never for a cancellation.
  - The wait for the running coroutine is longer than the A2A one, because
    the external turn's finalize races the settlement that the cancelled
    coroutine is about to perform. See ``EXTERNAL_CANCEL_WAIT_SECONDS``.
  - Every expected failure leaves as ``TaskCommandRejected``. Whether the
    visitor hears about it is the dispatcher's per-reason policy
    (``EXTERNAL_CANCEL_BROADCAST_REJECTION_REASONS``): a stale target
    broadcasts, because the stop press it answers gets no other signal,
    while a stop that no longer has a target at all stays terminal without
    a client-visible error frame.
  - The finalize commits more than the terminal row: the interruption
    transcript line the visitor reads and the stopped turn's delivery row
    are staged in the same transaction, and the shared task cache is
    invalidated once that commit lands.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from ..models.chat_message import TaskChatMessage
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from .assistant_history_safety import CLIENT_SAFE_FAILURE_MESSAGE_TYPE
from .chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
    mark_user_message_delivery,
    persist_assistant_message_no_commit,
)
from .db_runtime import run_db_io_cancellation_safe
from .task_command_transport import TaskCommandRejected
from .task_execution_controller import TaskControlState

logger = logging.getLogger(__name__)

EXTERNAL_TASK_SOURCE = "external"

# Command payload scope that routes a CANCEL command to this core. A command
# without it keeps the A2A execution path it has always had.
EXTERNAL_COMMAND_SCOPE = "external"

_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}

# Cancellation waits for the run coroutine to unwind, once per handle - the
# running handle and the resume coordinator - so the nominal cost of one
# external cancel is twice this value. Nominal, not bounded:
# ``BackgroundTaskManager.cancel_task`` waits with ``asyncio.wait_for``,
# which on timeout cancels the coroutine and then waits for that
# cancellation to finish, and the coroutine's cleanup drains uninterruptible
# database work in a worker thread before it lets the cancellation through
# (``run_db_io_cancellation_safe``). A slow database therefore extends the
# hold past this value with no hard upper bound.
#
# The value is deliberately far above the A2A path's 0.5s: whichever writer
# lands first owns the terminal row, and letting the cancelled coroutine
# settle first keeps its lease and delivery reconciliation in the hands of
# the coroutine that ran the turn. Raising it raises the typical queueing
# delay for every other command on this process; lowering it makes the
# overlap window more likely.
EXTERNAL_CANCEL_WAIT_SECONDS = 5.0

# A cancelled turn and a worker shutting down both cut the response short,
# so the text every consumer of the interruption sees states the outcome
# without claiming a cause.
EXTERNAL_TURN_INTERRUPTED_MESSAGE = "This response was interrupted."

# Written to the durable row only when a cancel command actually applied,
# which is what keeps the two causes apart for whoever reads the row later.
EXTERNAL_CANCEL_ERROR_MESSAGE = "Stopped by the visitor."

# Broadcast, never stored. An exhausted cancel command whose target is
# still alive has to say so, and the sentence has to be actionable: the
# visitor can press stop again. Writing it to ``task.error_message`` would
# put a text the replay judgement does not recognise on a live row, so this
# constant has exactly one consumer - the terminal command broadcast.
EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE = (
    "Stopping this response didn't go through — please try again."
)

# ``TaskCommandRejected`` reasons the durable dispatcher answers with a live
# ``agent_error`` broadcast when an external-scope CANCEL lands on them
# (#2009). Membership is decided per reason, never as a blanket:
#
# - ``stale_run`` broadcasts. It is the one rejection the real producer's
#   stop press can reach — the target's run/version moved between the
#   producer's read and the dispatch — and nothing else answers that press.
#   Its producers are this module's own checks above plus three sites in
#   ``websocket.py``'s dispatcher: the common pre-dispatch target-run-id
#   comparison (ahead of either execution core), the CANCEL payload's
#   state-version validation, and the ``StaleTaskRunError`` wrap around the
#   cancel cores. Every one of them rides this same gate — it matches on
#   the reason, not the raise site. The broadcast wording is read
#   from the task's current status (``external_cancel_exhausted_message``),
#   so it stays true for a live successor ("didn't go through — please try
#   again") and for a target that settled on its own ("This response was
#   interrupted."). A target that is ``COMPLETED`` is the one exception: the
#   broadcast site suppresses the live frame there, because the task's own
#   completion frame already answered the visitor and "interrupted" would
#   be false for a run that finished.
# - ``task_not_found`` stays silent. The status read has no row to consult,
#   and the fallback wording would invite retrying a stop against a task
#   that no longer exists.
# - ``unsupported_scope`` is structurally absent: it is raised only for a
#   payload whose scope is not ``external``, and the dispatcher's gate
#   classifies external cancels by that same scope, so no membership here
#   could ever make it broadcast.
EXTERNAL_CANCEL_BROADCAST_REJECTION_REASONS = frozenset({"stale_run"})


def external_cancel_exhausted_message(task_status: TaskStatus | None) -> str:
    """Wording for a cancel command that ran out of budget, by what happened.

    ``EXTERNAL_TURN_INTERRUPTED_MESSAGE`` asserts the response is over, so it
    is only true once the task reached ``COMPLETED`` or ``FAILED``. ``PAUSED``
    and ``WAITING_FOR_USER`` look stopped to the frontend - its
    ``isStoppedTaskStatus`` counts them, because that function answers
    "should the spinner stop" - but the run behind them can still produce
    more output, so on those the stop did not take.

    ``None`` means the status could not be read: the row is gone, or the
    database did not answer. The sentence that asserts nothing about the
    turn is the safe one there too.
    """

    if task_status in _TERMINAL_STATUSES:
        return EXTERNAL_TURN_INTERRUPTED_MESSAGE
    return EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE


# The two sentences a cancelled external turn can be settled with: this
# module's own finalize writes the first, the cancelled run's settlement
# writes the second (``task_orchestrator`` derives it from the task
# source). A row carrying neither was failed by something other than a
# cancel, and this command has no claim on it.
_SETTLED_BY_CANCEL_MESSAGES = frozenset(
    {EXTERNAL_CANCEL_ERROR_MESSAGE, EXTERNAL_TURN_INTERRUPTED_MESSAGE}
)


def _task_run_id(task: Task) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _load_external_task(db: Session, *, task_id: int, agent_id: int) -> Task:
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.agent_id == agent_id,
            Task.source == EXTERNAL_TASK_SOURCE,
        )
        .first()
    )
    if task is None:
        raise TaskCommandRejected(
            f"task {task_id} is not an external task of agent {agent_id}",
            reason="task_not_found",
        )
    return task


def _is_settled_external_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Report whether this exact target already reached its cancel outcome.

    The judgement is the durable state tuple plus the settling sentence, not
    a marker column: the same command replayed after its own finalize, and
    the settlement of the run this command cancelled, both leave the exact
    target run FAILED at the command's own state version or one past it,
    carrying one of the two texts only a cancellation writes.

    The sentence is load bearing, not decoration. Without it the same tuple
    also describes a run that failed on its own - a provider timeout, a
    setup error - and this command would report that failure as a
    cancellation it had performed, and broadcast a terminal event claiming
    the visitor's stop had landed. A tuple that matches with any other text
    falls through to ``_assert_external_cancel_target``, which rejects it as
    ``stale_run`` (the version has moved, or the task is already terminal),
    writing nothing itself; that run's own settlement owns its transcript
    and its event. The dispatcher still broadcasts that rejection for the
    reasons in ``EXTERNAL_CANCEL_BROADCAST_REJECTION_REASONS``, except when
    the target completed - there the live frame is suppressed because the
    completion frame already answered the visitor.
    """

    return (
        task.status == TaskStatus.FAILED
        and _task_run_id(task) == expected_run_id
        and int(task.state_version or 0)
        in {expected_state_version, expected_state_version + 1}
        and task.error_message in _SETTLED_BY_CANCEL_MESSAGES
    )


def _assert_external_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> None:
    """Reject a cancel command whose immutable task-state target is stale."""

    current_run_id = _task_run_id(task)
    current_state_version = int(task.state_version or 0)
    if (
        current_run_id != expected_run_id
        or current_state_version != expected_state_version
        or task.status in _TERMINAL_STATUSES
    ):
        raise TaskCommandRejected(
            f"task {task.id} changed from run/version "
            f"{expected_run_id}/{expected_state_version} to "
            f"{current_run_id}/{current_state_version}/{task.status.value}",
            reason="stale_run",
        )


def _load_cancelable_external_task_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Return whether the exact cancel target is already settled."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _load_external_task(db, task_id=task_id, agent_id=agent_id)
        if _is_settled_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return True
        _assert_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
        return False


def _mark_cancelled_turn_delivery_dispatched(
    db: Session, task_id: int, *, turn_id: str | None
) -> None:
    """Close the cancelled turn's delivery row without committing ``db``.

    A delivery row is closed by whichever coroutine settles the turn it
    belongs to. When this finalize wins that race the settlement is fenced
    out, so nothing else ever moves the row off ``pending`` and every later
    resend of the same client message id is refused forever.
    ``dispatched`` rather than ``failed`` is the honest target: the run had
    already started, so the message may have been consumed and a retry
    invitation could double-execute it.

    ``turn_id`` names the exact row when the command carries one, which is
    the only way to be certain the row closed belongs to the turn the stop
    was aimed at. It is a producer-reported value, not an authorization
    fact: even a value forged for another task falls through the
    ``task_id`` filter in ``mark_user_message_delivery`` and lands nothing,
    the same way ``end_user_id`` is audit data only.

    Without one the target is the newest pending user row on the task. That
    fallback is only sound because a task cannot take a new turn while it is
    running: ``_claim_turn_no_commit`` in ``task_orchestrator`` filters an
    APPEND on ``_APPENDABLE_STATUSES`` - COMPLETED, FAILED, PAUSED - which
    excludes RUNNING, so a cancel target that is still running has at most
    one pending user row and "newest" and "the running turn's" name the same
    row. If that filter ever admits RUNNING, this fallback starts closing
    another turn's row and every caller must pass ``turn_id``.
    """

    if turn_id is None:
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
        if row is None or row[0] is None:
            return
        turn_id = str(row[0])
    mark_user_message_delivery(
        db,
        task_id=task_id,
        turn_id=turn_id,
        status=DELIVERY_DISPATCHED,
    )


def _persist_interruption_transcript_no_commit(db: Session, task: Task) -> None:
    """Stage the one assistant line the stopped turn owes its reader.

    Exactly one writer produces it. When the cancelled run's settlement
    wins the race it writes the line itself (``settle_task_lease_isolated``
    in ``task_orchestrator``), and this finalize never runs its fenced
    UPDATE, so it writes nothing. When this finalize wins, the settlement's
    UPDATE requires ``status == RUNNING`` and matches no row, so it skips
    its own write and this line is the only one. The replay short circuit
    above returns before reaching here, so a redelivered command adds none.

    The row mirrors the settlement's exactly - same content, same explicit
    client-safe failure provenance, no turn id - so the reader cannot tell
    which writer won. ``content_is_reconciled`` is set because the content is
    a server-owned constant with no file references to resolve.

    A task row without an owner gets no line from either writer: the
    settlement guards on ``task.user_id is not None`` and so does this.
    External rows are created with the agent owner's id and never hit that
    branch; it is a guard against a NULL column, not a normal path.
    """

    if task.user_id is None:
        return
    persist_assistant_message_no_commit(
        db,
        task_id=int(task.id),
        user_id=int(task.user_id),
        content=EXTERNAL_TURN_INTERRUPTED_MESSAGE,
        message_type=CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
        content_is_reconciled=True,
    )


def _invalidate_task_cache_after_commit(task_id: int) -> None:
    """Drop the task's cached projections once its terminal row is committed.

    The import is deferred because ``task_orchestrator`` imports this module
    at module level for the settlement text; a module-level import back
    would close the cycle. The orchestrator's own wrapper is the one called
    rather than ``hot_path_cache.invalidate_task_cache`` directly, so a cache
    outage stays non-fatal here exactly as it is on the settlement path.
    """

    from .task_orchestrator import invalidate_task_cache_best_effort

    invalidate_task_cache_best_effort(task_id)


def _finalize_external_cancel_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
    turn_id: str | None,
) -> None:
    """Atomically persist cancellation for one exact task-state target."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _load_external_task(db, task_id=task_id, agent_id=agent_id)
        if _is_settled_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return
        _assert_external_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )

        statement = (
            update(Task)
            .where(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == EXTERNAL_TASK_SOURCE,
                func.coalesce(Task.state_version, 0) == expected_state_version,
                Task.status.notin_(_TERMINAL_STATUSES),
            )
            .values(
                status=TaskStatus.FAILED,
                control_state=TaskControlState.FAILED.value,
                state_version=expected_state_version + 1,
                runner_id=None,
                lease_attempt_id=None,
                lease_expires_at=None,
                last_heartbeat_at=None,
                error_message=EXTERNAL_CANCEL_ERROR_MESSAGE,
            )
        )
        if expected_run_id is None:
            statement = statement.where(Task.run_id.is_(None))
        else:
            statement = statement.where(Task.run_id == expected_run_id)

        updated = db.execute(
            statement.returning(Task).execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if updated is None:
            db.rollback()
            raise TaskCommandRejected(
                f"task {task_id} changed while finalizing cancel target "
                f"{expected_run_id}/{expected_state_version}",
                reason="stale_run",
            )
        _persist_interruption_transcript_no_commit(db, updated)
        _mark_cancelled_turn_delivery_dispatched(db, task_id, turn_id=turn_id)
        if updated.conversation_storage_version == 2:
            from .task_execution_event_writer import stage_result_fact_no_commit

            db.refresh(updated)
            stage_result_fact_no_commit(
                db,
                updated,
                {"status": "cancelled", "error": EXTERNAL_CANCEL_ERROR_MESSAGE},
            )
        db.commit()
        _invalidate_task_cache_after_commit(task_id)


async def _broadcast_external_cancel_terminal_event(task_id: int) -> None:
    from ..api.websocket import create_terminal_task_error_event
    from ..api.websocket import manager as websocket_manager

    try:
        await websocket_manager.broadcast_to_task(
            create_terminal_task_error_event(
                task_id,
                EXTERNAL_TURN_INTERRUPTED_MESSAGE,
            ),
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


async def cancel_external_task_unserialized(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
    turn_id: str | None = None,
) -> None:
    """Cancel one exact durable-command target while the caller owns its gate."""

    already_settled = await run_db_io_cancellation_safe(
        lambda: _load_cancelable_external_task_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
    )
    if not already_settled:
        from ..api.websocket import background_task_manager

        await background_task_manager.cancel_task(
            task_id,
            timeout_seconds=EXTERNAL_CANCEL_WAIT_SECONDS,
        )
        await run_db_io_cancellation_safe(
            lambda: _finalize_external_cancel_sync(
                task_id=task_id,
                agent_id=agent_id,
                expected_run_id=expected_run_id,
                expected_state_version=expected_state_version,
                turn_id=turn_id,
            )
        )
    await _broadcast_external_cancel_terminal_event(task_id)
