"""``POST /v1/chat/tasks/{task_id}/reply``: answer a waiting task's question.

A task in ``waiting_for_user`` has a paused execution, not an idle one --
its agent asked a question via ``ask_user_question`` / a ``send_message``
call with ``expect_response=True`` and is suspended waiting for the
answer. Answering it means resuming that exact paused run, not starting
a new turn the way ``POST /v1/chat/tasks/{task_id}/messages`` does. This
module implements that resume, mirroring the shape of the A2A
``message:send`` input-required resume in ``web/api/a2a.py``
(``_resume_input_required_a2a_task``) with four differences:

  - Scoped to ``Task.source == "sdk"`` plus the v1 SDK ownership rules
    (agent-bound or workforce-bound key) instead of A2A's
    ``Task.source == "a2a"`` plus a bare ``agent_id`` match.
  - Only ``WAITING_FOR_USER`` is accepted; ``PAUSED`` is left to the
    existing append endpoint, which already handles it and already has
    its own failure policy for that status.
  - Errors are translated to the stable ``V1ApiError`` envelope instead
    of A2A's own error shape.
  - The synthetic ``turn_id`` is derived from a fresh UUID
    (``v1:reply:{task_id}:{uuid4()}``) rather than an A2A message id,
    because the v1 request body carries no client-supplied message
    identity to derive one from. This endpoint intentionally does not
    implement idempotency: a client-side retry after a timeout can post
    the answer twice. Callers that want to avoid that should ``GET`` the
    task and confirm it has left ``waiting_for_user`` before retrying.

The rest of the resume mechanics -- the two-phase prelease (claim, then
either commit past it or restore it exactly), the heartbeat guarding the
injection, and the fail-closed handling of every checkpoint outcome --
are intentionally duplicated from ``a2a.py`` rather than factored into a
shared helper. The two call sites answer to different protocols with
independently evolving error contracts; a shared helper would couple
them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func

from ....core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointReadError,
)
from ...models.database import get_session_local
from ...models.task import Task, TaskStatus
from ...schemas.v1 import ReplyRequest, ReplyResponse
from ...services.db_runtime import (
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ...services.task_execution_controller import TaskControlState
from ...services.task_interaction_close import (
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction,
)
from ...services.task_interaction_schema import interaction_requests_table_exists
from ...services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
    TaskLeaseLostError,
    acquire_task_lease_cancellation_safe,
    acquire_task_lease_no_commit,
    bind_task_lease_context,
    release_task_lease_no_commit,
    run_task_lease_heartbeat,
    run_while_task_lease_owned,
    stop_task_lease_heartbeat,
)
from .deps import ApiKeyPrincipal, record_key_usage
from .errors import V1ApiError, V1ErrorCode
from .tasks import _resolve_task_or_404, _validate_owner_scope

logger = logging.getLogger(__name__)

# Control states a waiting task may be claimed from. Mirrors the a2a
# resume prelease's acceptance set for WAITING_FOR_USER: a task parked
# there sits at either the plain WAITING_FOR_USER control state, or, on
# a database left over from an interrupted prior resume attempt, IDLE.
_RESUMABLE_CONTROL_STATES = frozenset(
    {
        TaskControlState.IDLE.value,
        TaskControlState.WAITING_FOR_USER.value,
    }
)


@dataclass(frozen=True)
class _ReplyContext:
    """Detached, already-authorized inputs the resume phase needs."""

    task_id: int
    agent_id: int
    task_owner_user_id: int
    run_id: str | None
    text: str


def _status_rejection(status: TaskStatus) -> V1ApiError:
    """Map a non-waiting task status to its reply-rejection error.

    RUNNING is the one truly transient case (a real turn is in flight;
    retrying later can succeed), so it keeps the generic retryable
    ``task_busy``. Every other non-waiting status means "there is
    nothing to reply to right now" -- ``no_pending_interaction``, not
    retryable as such. The caller's remedy differs by status: for
    paused / completed / failed, ``append`` starts the next turn; for
    pending, the task hasn't started executing yet -- it is not in
    ``_APPENDABLE_STATUSES`` either, so there is no turn to append to
    yet. The only remedy is to wait for it to leave pending.
    """
    if status == TaskStatus.RUNNING:
        return V1ApiError(V1ErrorCode.TASK_BUSY, 409)
    return V1ApiError(V1ErrorCode.NO_PENDING_INTERACTION, 409)


def _prepare_reply_context_sync(
    *,
    task_id: int,
    principal: ApiKeyPrincipal,
    request: ReplyRequest,
) -> _ReplyContext:
    """Authorize the call and validate the body against the task's status.

    A pure read -- no mutation happens here. The actual state transition
    is the prelease claim in :func:`_acquire_reply_prelease_sync`, run
    afterwards under ``acquire_task_lease_cancellation_safe``.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        _validate_owner_scope(
            principal,
            request_agent_id=request.agent_id,
            request_workforce_id=request.workforce_id,
        )
        if request.message.files:
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                422,
                message=(
                    "files are not accepted on reply; attach files via a "
                    "follow-up call to the task append endpoint "
                    "(POST .../messages) instead"
                ),
            )
        if task.status != TaskStatus.WAITING_FOR_USER:
            raise _status_rejection(task.status)
        return _ReplyContext(
            task_id=int(task.id),
            agent_id=int(task.agent_id),
            task_owner_user_id=int(task.user_id),
            run_id=str(task.run_id) if task.run_id is not None else None,
            text=request.message.content,
        )


# Four different mechanisms keep this claim from racing every other writer
# that can touch a WAITING_FOR_USER task, none of which overlap by accident:
#
# 1. Append (task_orchestrator._claim_turn_no_commit): its claim filters
#    Task.status.in_(_APPENDABLE_STATUSES), and this PR removed
#    WAITING_FOR_USER from that set. A row this claim can match (status ==
#    WAITING_FOR_USER) is therefore structurally outside append's status
#    filter -- the two claims can never both succeed on the same row, by
#    construction of the two status sets being disjoint.
# 2. A second reply on the same task: the claim below is a single
#    conditional UPDATE (status == WAITING_FOR_USER AND control_state in
#    _RESUMABLE_CONTROL_STATES); whichever request's UPDATE commits first
#    flips control_state away from those values, so a concurrent request's
#    UPDATE matches zero rows and returns None. Covered by
#    test_concurrent_reply_race_exactly_one_winner.
# 3. A2A's own resume (a2a._acquire_a2a_resume_prelease_sync): its claim
#    filters Task.source == "a2a"; this claim filters Task.source == "sdk".
#    A task row has exactly one source value, so the two filters can never
#    both match the same row.
# 4. The WebSocket live-control resume path: unlike the three cases above,
#    it does NOT filter by Task.source at all, so it is not excluded from a
#    "sdk" row by a source predicate. Its exclusivity instead comes from
#    two other mechanisms: the in-process reservation in
#    background_task_manager (reserve_resume / resume_tasks) that allows
#    only one live resume coroutine per task_id, and the exact-run-fenced
#    lease acquired below, which a second claimant with a stale or missing
#    run_id cannot pass.
def _acquire_reply_prelease_sync(
    *,
    task_id: int,
    agent_id: int,
    previous_run_id: str | None,
    result_holder: dict[str, Any],
) -> TaskLease | None:
    """Claim and commit one exact reply-resume lease in a worker transaction.

    ``agent_id`` re-checks the task row against the value
    ``_resolve_task_or_404`` already authorized for this call -- ownership
    itself (agent-bound vs workforce-bound key) was established there;
    this claim only needs to confirm the row's identity hasn't moved
    since, which a scalar ``agent_id`` match does without a join.

    ``result_holder`` is filled in with the post-claim ``state_version``
    / ``control_state`` before commit, so the caller can report the
    response's real values without a second query racing the
    background resume this claim is about to hand off to. It stays
    empty when the claim fails.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        claimed = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "sdk",
                Task.status == TaskStatus.WAITING_FOR_USER,
                Task.control_state.in_(_RESUMABLE_CONTROL_STATES),
            )
            .update(
                {
                    Task.control_state: TaskControlState.RESUME_REQUESTED.value,
                    Task.state_version: func.coalesce(Task.state_version, 0) + 1,
                },
                synchronize_session=False,
            )
        )
        if claimed != 1:
            db.rollback()
            return None
        # A legacy row with run_id already NULL passes previous_run_id=None
        # here. acquire_task_lease_no_commit mints a fresh UUID for that
        # case rather than reusing anything, and (since the row isn't a
        # live RUNNING row with a run id) also clears both checkpoint
        # pointer columns (last_checkpoint_event_id /
        # last_checkpoint_trace_event_id) as part of the same UPDATE. If
        # this reply later fails closed, releasing the lease back to
        # waiting_for_user does not restore those two columns -- they stay
        # cleared. This is not a behavior change worth guarding against:
        # a row with no run_id never had a run-fenced checkpoint to
        # recover in the first place, so nothing resumable is lost.
        task_lease = acquire_task_lease_no_commit(
            db,
            task_id,
            expected_run_id=previous_run_id,
        )
        if task_lease is None or task_lease.run_id is None:
            db.rollback()
            return None
        refreshed = (
            db.query(Task.state_version, Task.control_state)
            .filter(Task.id == task_id)
            .one()
        )
        result_holder["state_version"] = int(refreshed.state_version or 0)
        result_holder["control_state"] = str(refreshed.control_state or "idle")
        db.commit()
        return task_lease


def _restore_reply_prelease_sync(task_lease: TaskLease) -> bool:
    """Release one exact reply prelease in a worker-owned short Session.

    A successful restore retains the run id and its tagged checkpoint, so
    a client retry after a lost response is idempotent at the runner
    boundary. If ownership changed, no row is mutated and the current
    owner remains the sole lifecycle authority.

    Mirrors the A2A prelease restore (``a2a._restore_a2a_resume_prelease_sync``):
    a successful restore is an abandoned resume, not a completed one, so it
    also reconciles the interaction marker via
    ``clear_interaction_marker_if_unpaired`` -- see that function's docstring
    for the NOT EXISTS semantics. No lock read precedes this call:
    ``release_task_lease_no_commit``'s own tasks UPDATE writes only
    non-key columns and is already the first statement this transaction
    directs at tasks or task_interaction_requests.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        restored = release_task_lease_no_commit(
            db,
            task_lease,
            status=TaskStatus.WAITING_FOR_USER,
        )
        if restored:
            assert task_lease.run_id is not None
            clear_interaction_marker_if_unpaired(
                db, task_id=task_lease.task_id, run_id=task_lease.run_id
            )
            db.commit()
        else:
            db.rollback()
        return restored


async def _restore_reply_prelease_isolated(task_lease: TaskLease) -> bool:
    return await run_db_io_cancellation_safe(
        lambda: _restore_reply_prelease_sync(task_lease)
    )


# The exact non-key column set the resume-input fence UPDATE below writes.
# Shared with test_interaction_close_lock_ordering.py's static guard, which
# asserts the UPDATE's values keys equal this set exactly -- see
# a2a.RESUME_INPUT_FENCE_UPDATE_COLUMNS for the fence's no-lock-read
# argument this set backs.
RESUME_INPUT_FENCE_UPDATE_COLUMNS = frozenset({"input", "output", "error_message"})


def _update_reply_input_sync(task_lease: TaskLease, text: str) -> bool:
    """Persist the reply's input only while the exact prelease remains current."""
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        updated = (
            db.query(Task)
            .filter(
                Task.id == task_lease.task_id,
                Task.status == TaskStatus.RUNNING,
                Task.runner_id == task_lease.runner_id,
                Task.run_id == task_lease.run_id,
            )
            .update(
                {
                    Task.input: text,
                    Task.output: None,
                    Task.error_message: None,
                },
                synchronize_session=False,
            )
        )
        if updated != 1:
            db.rollback()
            return False
        # Mirrors a2a._update_a2a_resume_input_sync's fence-ordering argument:
        # this update() call is the fence, not a new transaction, so a
        # rollback here would undo it together with the input write -- the
        # point is that ownership and the interaction close are one atomic
        # fact. No lock read either -- the fence UPDATE above writes only
        # non-key columns (input, output, error_message) and is already the
        # first statement this transaction directs at tasks or
        # task_interaction_requests, so it satisfies the same ordering and
        # strength obligation a dedicated lock read would. The table-presence
        # gate sits here, immediately before the close call and after the
        # fence UPDATE, for the same reason as the A2A site: the gate only
        # inspects the catalog and takes no row lock, so it does not count
        # as preceding the fence UPDATE in the sense the ordering obligation
        # means.
        assert task_lease.run_id is not None
        if interaction_requests_table_exists(db):
            close_legacy_resume_interaction(
                db, task_id=task_lease.task_id, run_id=task_lease.run_id
            )
        db.commit()
        return True


async def _schedule_waiting_reply_resume(
    *,
    task_id: int,
    agent_service: Any,
    task_owner_user_id: int,
    task_lease: TaskLease,
    heartbeat_stop: asyncio.Event,
    heartbeat_task: "asyncio.Task[TaskLeaseHeartbeatOutcome]",
) -> None:
    from .. import websocket

    if task_lease.task_id != task_id or task_lease.run_id is None:
        raise ValueError("Reply resume scheduling requires an exact task lease")
    if not websocket.background_task_manager.reserve_resume(task_id):
        # A same-process resume is already registered for this task despite
        # the exclusive DB claim above -- report it the same way a caller
        # would read a lost race on the claim itself, rather than a bare
        # 500: retrying (after the other resume settles) can succeed.
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409)
    previous_task = websocket.background_task_manager.running_tasks.get(task_id)
    bg_task: "asyncio.Task[None] | None" = None
    try:
        bg_task = asyncio.create_task(
            websocket.execute_resume_background(
                task_id=task_id,
                agent_service=agent_service,
                task_owner_user_id=task_owner_user_id,
                expected_run_id=task_lease.run_id,
                previous_task=previous_task,
                preacquired_lease=task_lease,
                preacquired_heartbeat_stop=heartbeat_stop,
                preacquired_heartbeat_task=heartbeat_task,
                # The prelease claimed this task out of waiting_for_user;
                # hand that over so a checkpoint the resume cannot read
                # restores it instead of failing it terminally.
                preacquired_prior_status=TaskStatus.WAITING_FOR_USER,
            )
        )
        websocket.background_task_manager.register_reserved_resume(task_id, bg_task)
    except BaseException:
        if bg_task is not None:
            await cancel_and_drain_async_task(bg_task)
        websocket.background_task_manager.release_resume_reservation(task_id)
        raise


async def reply_to_task(
    *,
    task_id: int,
    request: ReplyRequest,
    principal: ApiKeyPrincipal,
) -> ReplyResponse:
    """Resume a ``waiting_for_user`` task with the user's answer.

    Args:
        task_id: Path parameter; the target task's primary key.
        request: Validated :class:`ReplyRequest`.
        principal: Key-bound owner from the auth dependency.

    Returns:
        :class:`ReplyResponse` with ``status='running'`` and the same
        ``run_id`` the task was waiting on.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task not found, not owned by the key, or
            body.agent_id / body.workforce_id doesn't match the bound
            owner.
        V1ApiError 422: body.message.files is non-empty, or
            body.agent_id is missing for an agent-bound key.
        V1ApiError 409: ``task_busy`` (task is RUNNING, or the resume
            lease was lost to a concurrent reply -- retryable);
            ``no_pending_interaction`` (task is not currently waiting
            on a question); ``interaction_not_resumable`` (the task's
            saved progress cannot be resumed -- NOT retryable, the task
            stays in waiting_for_user and the caller must start a new
            task).
        V1ApiError 503: ``temporarily_unavailable`` -- the saved
            progress could not be read due to a transient failure;
            retryable.
        500: any other unexpected error (V1 envelope via the global
            handler), including a lease lost mid-resume.
    """
    ctx = await run_db_io_cancellation_safe(
        lambda: _prepare_reply_context_sync(
            task_id=task_id,
            principal=principal,
            request=request,
        )
    )

    prelease_info: dict[str, Any] = {}
    task_lease = await acquire_task_lease_cancellation_safe(
        lambda: _acquire_reply_prelease_sync(
            task_id=ctx.task_id,
            agent_id=ctx.agent_id,
            previous_run_id=ctx.run_id,
            result_holder=prelease_info,
        ),
        lambda acquired: _restore_reply_prelease_sync(acquired),
    )
    if task_lease is None or task_lease.run_id is None:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409)

    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(task_lease, heartbeat_stop)
    )
    ownership_transferred = False
    prelease_cleanup_done = False

    async def stop_and_restore_prelease() -> bool:
        nonlocal prelease_cleanup_done
        if prelease_cleanup_done:
            return False
        prelease_cleanup_done = True
        try:
            outcome = await stop_task_lease_heartbeat(heartbeat_task, heartbeat_stop)
        except BaseException:
            logger.error(
                "Reply prelease heartbeat failed before task %s could be "
                "restored; retaining run %s for TTL recovery",
                task_id,
                task_lease.run_id,
                exc_info=True,
            )
            return False
        if outcome.requires_ttl_recovery:
            logger.error(
                "Reply prelease for task %s became unhealthy; retaining "
                "run %s for TTL recovery (lost=%s, pool_timeout=%s)",
                task_id,
                task_lease.run_id,
                outcome.lease_lost,
                outcome.pool_timeout is not None,
            )
            return False
        return await _restore_reply_prelease_isolated(task_lease)

    try:

        async def inject_user_message() -> tuple[Any, bool]:
            from .. import chat

            agent_service = await chat.get_agent_manager().get_agent_for_task(
                task_id,
                None,
                task_owner_user_id=ctx.task_owner_user_id,
            )
            posted = await agent_service.post_user_message(
                str(task_id),
                execution_message=ctx.text,
                display_message=ctx.text,
                turn_id=f"v1:reply:{task_id}:{uuid4()}",
                request_interrupt=False,
                reason="V1 interaction response",
            )
            return agent_service, bool(posted)

        with bind_task_lease_context(task_lease):
            agent_service, posted = await run_while_task_lease_owned(
                inject_user_message(),
                heartbeat_task,
            )

        if not posted:
            # No run-fenced checkpoint is available to resume onto. Never
            # fabricate a run or fall back to a transcript replay: release
            # the exact prelease back to waiting_for_user and fail closed
            # so the caller must start a new task explicitly.
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            if not await drain_async_task_cancellation_safe(cleanup_task):
                raise TaskLeaseLostError(
                    f"Task {task_id} lease changed before reply fallback"
                )
            raise V1ApiError(V1ErrorCode.INTERACTION_NOT_RESUMABLE, 409)

        updated = await run_db_io_cancellation_safe(
            lambda: _update_reply_input_sync(task_lease, ctx.text)
        )
        if not updated:
            raise TaskLeaseLostError(
                f"Task {task_id} lease changed before reply resume scheduling"
            )

        await _schedule_waiting_reply_resume(
            task_id=task_id,
            agent_service=agent_service,
            task_owner_user_id=ctx.task_owner_user_id,
            task_lease=task_lease,
            heartbeat_stop=heartbeat_stop,
            heartbeat_task=heartbeat_task,
        )
        ownership_transferred = True
    except CheckpointReadError as exc:
        if not ownership_transferred and not prelease_cleanup_done:
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            if not await drain_async_task_cancellation_safe(cleanup_task):
                raise TaskLeaseLostError(
                    f"Task {task_id} lease changed before reply "
                    "checkpoint-failure fallback"
                ) from exc
        if isinstance(exc, CheckpointCorruptError):
            raise V1ApiError(V1ErrorCode.INTERACTION_NOT_RESUMABLE, 409) from exc
        if isinstance(exc, CheckpointAccessRefusedError):
            if exc.reason == "superseded_legacy":
                raise V1ApiError(V1ErrorCode.INTERACTION_NOT_RESUMABLE, 409) from exc
            raise V1ApiError(V1ErrorCode.TASK_BUSY, 409) from exc
        # CheckpointUnavailableError, or any future CheckpointReadError
        # subclass this dispatch does not yet know about: treat it
        # conservatively as retryable rather than assuming a terminal
        # failure, so an unrecognized failure mode never silently
        # collapses into a data-losing branch.
        raise V1ApiError(V1ErrorCode.TEMPORARILY_UNAVAILABLE, 503) from exc
    except BaseException:
        if not ownership_transferred and not prelease_cleanup_done:
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            await drain_async_task_cancellation_safe(cleanup_task)
        raise
    finally:
        if not ownership_transferred and not prelease_cleanup_done:
            await stop_task_lease_heartbeat(heartbeat_task, heartbeat_stop)

    await record_key_usage(str(principal.key.key_prefix))

    return ReplyResponse(
        task_id=ctx.task_id,
        agent_id=ctx.agent_id,
        workforce_id=(
            int(principal.workforce.id) if principal.workforce is not None else None
        ),
        status=TaskStatus.RUNNING.value,
        accepted_at=datetime.now(timezone.utc),
        run_id=task_lease.run_id,
        state_version=prelease_info["state_version"],
        control_state=prelease_info["control_state"],
    )
