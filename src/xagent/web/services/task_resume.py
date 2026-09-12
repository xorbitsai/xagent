"""Runner-owned A2A and SDK reply recovery.

The two recovery flows retain their existing claim, injection and cleanup
semantics. Callers provide detached inputs and translate failures to their
protocol; leases, agents and background tasks stay inside this module.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, assert_never
from uuid import uuid4

from sqlalchemy import func

from ...core.agent.checkpoint import CheckpointReadError, CheckpointUnavailableError
from ...core.agent.runner import UserMessageInjectionOutcome
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from . import agent_service_manager as agent_runtime_service
from . import task_execution as task_execution_service
from .db_runtime import (
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from .task_execution_controller import TaskControlState
from .task_interaction_close import (
    ActiveInteractionAbsent,
    ActiveInteractionFound,
    ActiveInteractionUnavailable,
    active_interaction_id_sync,
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction,
)
from .task_interaction_schema import interaction_requests_table_exists
from .task_lease_service import (
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
    task_lease_attempt_predicate,
)

logger = logging.getLogger(__name__)


class TaskResumeBusyError(Exception):
    """Another execution owns the task or has already claimed its resume."""


class TaskResumeNotWaitingError(Exception):
    """The SDK task has no pending reply to resume."""


class TaskResumeNotResumableError(Exception):
    """No run-fenced checkpoint accepted the reply."""


class TaskResumeRetryableError(CheckpointUnavailableError):
    """A temporary read failed before injection and its lease was restored."""


@dataclass(frozen=True)
class TaskReplyInput:
    """Authorized task snapshot and reply, detached from the request session."""

    task_id: int
    agent_id: int
    task_owner_user_id: int
    run_id: str | None
    status: TaskStatus
    text: str


@dataclass(frozen=True)
class TaskReplyResumeResult:
    """Claim state captured before the registered resume can advance it."""

    run_id: str
    state_version: int
    control_state: str


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


async def _schedule_waiting_a2a_resume(
    *,
    task_id: int,
    agent_service: Any,
    task_owner_user_id: int,
    task_lease: TaskLease,
    heartbeat_stop: asyncio.Event,
    heartbeat_task: asyncio.Task[TaskLeaseHeartbeatOutcome],
    resumable_status: TaskStatus,
) -> None:
    from .task_execution import (
        background_task_manager,
        execute_resume_background,
    )

    if task_lease.task_id != task_id or task_lease.run_id is None:
        raise ValueError("A2A resume scheduling requires an exact task lease")
    if not background_task_manager.reserve_resume(task_id):
        raise RuntimeError(f"Task {task_id} already has a resume in progress")
    previous_task = background_task_manager.running_tasks.get(task_id)
    bg_task: asyncio.Task[None] | None = None
    try:
        bg_task = asyncio.create_task(
            execute_resume_background(
                task_id=task_id,
                agent_service=agent_service,
                task_owner_user_id=task_owner_user_id,
                expected_run_id=task_lease.run_id,
                previous_task=previous_task,
                preacquired_lease=task_lease,
                preacquired_heartbeat_stop=heartbeat_stop,
                preacquired_heartbeat_task=heartbeat_task,
                # The prelease claimed this task out of an input-required
                # status; hand that over so a checkpoint the resume cannot
                # read restores it instead of failing it terminally.
                preacquired_prior_status=resumable_status,
            )
        )
        background_task_manager.register_reserved_resume(
            task_id,
            bg_task,
            run_id=task_lease.run_id,
        )
    except BaseException:
        if bg_task is not None:
            await cancel_and_drain_async_task(bg_task)
        background_task_manager.release_resume_reservation(task_id)
        raise


def _acquire_a2a_resume_prelease_sync(
    *,
    task_id: int,
    agent_id: int,
    resumable_status: TaskStatus,
    previous_run_id: str | None,
) -> TaskLease | None:
    """Claim and commit one exact A2A resume lease in a worker transaction."""

    resumable_control_states = {
        TaskControlState.IDLE.value,
        (
            TaskControlState.PAUSED.value
            if resumable_status == TaskStatus.PAUSED
            else TaskControlState.WAITING_FOR_USER.value
        ),
    }
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        claimed = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
                Task.status == resumable_status,
                Task.control_state.in_(resumable_control_states),
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
        task_lease = acquire_task_lease_no_commit(
            db,
            task_id,
            expected_run_id=previous_run_id,
        )
        if task_lease is None or task_lease.run_id is None:
            db.rollback()
            return None
        db.commit()
        return task_lease


def _restore_a2a_resume_prelease_sync(
    task_lease: TaskLease,
    *,
    status: TaskStatus = TaskStatus.WAITING_FOR_USER,
) -> bool:
    """Release one exact A2A prelease in a worker-owned short Session.

    A successful restore retains the run id and its tagged checkpoint, so a
    retry with the same A2A message id is idempotent at the runner boundary.
    If ownership changed, no row is mutated and the current owner remains the
    sole lifecycle authority.

    Two callers reach this today: the isolated wrapper right below, used by
    the no-checkpoint and checkpoint-read-error fallbacks; and the cancel
    settlement callback ``acquire_task_lease_cancellation_safe`` passes to
    ``resume_a2a_task``'s lease acquisition, which fires on
    a cancellation with no ``posted`` outcome to consult at all -- the
    marker clear below has to hold on that path too, not only on the two
    ``posted``-gated fallbacks.
    """

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        restored = release_task_lease_no_commit(
            db,
            task_lease,
            status=status,
        )
        if restored:
            # Mirror of the WebSocket lease restore: this is an abandoned
            # resume, not a completed one, so only the marker may need
            # reconciling -- see clear_interaction_marker_if_unpaired's
            # docstring for the NOT EXISTS semantics. No lock read precedes
            # this statement; release_task_lease_no_commit's own tasks
            # UPDATE writes only non-key columns and is already the first
            # statement this transaction directs at tasks or
            # task_interaction_requests.
            assert task_lease.run_id is not None
            clear_interaction_marker_if_unpaired(
                db, task_id=task_lease.task_id, run_id=task_lease.run_id
            )
            db.commit()
        else:
            db.rollback()
        return restored


async def _restore_a2a_resume_prelease_isolated(
    task_lease: TaskLease,
    *,
    status: TaskStatus,
) -> bool:
    return await run_db_io_cancellation_safe(
        lambda: _restore_a2a_resume_prelease_sync(
            task_lease,
            status=status,
        )
    )


# The exact non-key column set the resume-input fence UPDATE below writes.
# Shared with test_interaction_close_lock_ordering.py's static guard, which
# asserts the UPDATE's values keys equal this set exactly: the fence's
# no-lock-read argument (see the inline comment inside
# _update_a2a_resume_input_sync) depends on every one of these columns being
# a non-key column, so a future change widening the UPDATE's values must
# widen this constant too, deliberately, not just add a key to a dict
# literal the guard never looks at again.
RESUME_INPUT_FENCE_UPDATE_COLUMNS = frozenset({"input", "output", "error_message"})


def _update_a2a_resume_input_sync(
    task_lease: TaskLease,
    text: str,
    interaction_id: int | None,
    injection_outcome: UserMessageInjectionOutcome,
) -> bool:
    """Persist A2A input only while the exact prelease remains current.

    ``interaction_id`` is the active interaction row the caller observed
    before injecting the message, passed in rather than read here: this
    function's session does not open until after the injection has already
    committed. See ``task_interaction_close``'s module docstring for why
    the read has to precede the injection.

    ``injection_outcome`` is the caller's own ``post_user_message`` report,
    not re-derived here: a replayed turn id must not retire the close, and
    only the call that produced the short circuit can say whether this was
    one.
    """

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        updated = (
            db.query(Task)
            .filter(
                Task.id == task_lease.task_id,
                Task.status == TaskStatus.RUNNING,
                Task.runner_id == task_lease.runner_id,
                task_lease_attempt_predicate(task_lease),
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
        # This update() call is the fence above, not a new transaction: a
        # rollback here would undo it together with the input write, which
        # is the point -- ownership and the interaction close are one
        # atomic fact. No run_db_io_cancellation_safe wrap: the caller
        # already wraps this whole function in one. No lock read either --
        # the fence UPDATE above writes only non-key columns (input,
        # output, error_message) and is already the first statement this
        # transaction directs at tasks or task_interaction_requests, so it
        # satisfies the same ordering and strength obligation a dedicated
        # lock read would. If a future change adds a key column (or any
        # column covered by a unique index) to that UPDATE's values, this
        # judgment call must be redone -- the lock strength that UPDATE
        # takes would change.
        #
        # The table-presence gate sits here, immediately before the close
        # call and after the fence UPDATE, not at the top of the function:
        # the gate only inspects the catalog and takes no row lock, so it
        # does not count as preceding the fence UPDATE in the sense the
        # ordering obligation above means.
        assert task_lease.run_id is not None
        if interaction_requests_table_exists(db):
            # A retried A2A message replays this site's deterministic turn
            # id (f"a2a:{task_id}:{message_id}"), and inject_user_message
            # short-circuits a repeated turn id without persisting
            # anything. interaction_id above was read fresh by this
            # attempt, so on such a replay it names whatever the resumed
            # agent has staged since rather than the question this call
            # answered, and closing on it would retire a live question.
            # See task_interaction_close's module docstring for the rule,
            # the other sites, and why the v1 reply resume-input path
            # needs no guard at all.
            if injection_outcome is UserMessageInjectionOutcome.POSTED_FRESH:
                close_legacy_resume_interaction(
                    db,
                    task_id=task_lease.task_id,
                    run_id=task_lease.run_id,
                    interaction_id=interaction_id,
                )
        db.commit()
        return True


async def resume_a2a_task(
    *,
    agent_id: int,
    task_owner_user_id: int,
    task_id: int,
    previous_run_id: str | None,
    resumable_status: TaskStatus,
    text: str,
    message_id: str,
    preacquired_lease: TaskLease | None = None,
) -> bool:
    if resumable_status not in {
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
    }:
        return False
    from .task_execution_host import enqueues_task_turns

    if enqueues_task_turns() and preacquired_lease is None:
        from uuid import NAMESPACE_URL, uuid5

        from .task_resume_command import enqueue_resume_input

        await enqueue_resume_input(
            TaskReplyInput(
                task_id=task_id,
                agent_id=agent_id,
                task_owner_user_id=task_owner_user_id,
                run_id=previous_run_id,
                status=resumable_status,
                text=text,
            ),
            source="a2a",
            message_id=message_id,
            command_id=uuid5(NAMESPACE_URL, f"a2a-reply:{task_id}:{message_id}").hex,
        )
        return True
    task_lease = preacquired_lease or await acquire_task_lease_cancellation_safe(
        lambda: _acquire_a2a_resume_prelease_sync(
            task_id=task_id,
            agent_id=agent_id,
            resumable_status=resumable_status,
            previous_run_id=previous_run_id,
        ),
        lambda acquired: _restore_a2a_resume_prelease_sync(
            acquired,
            status=resumable_status,
        ),
    )
    if task_lease is None or task_lease.run_id is None:
        raise TaskResumeBusyError

    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        run_task_lease_heartbeat(task_lease, heartbeat_stop)
    )
    ownership_transferred = False
    message_posted = False
    prelease_cleanup_done = False

    async def stop_and_restore_prelease() -> bool:
        nonlocal prelease_cleanup_done
        if prelease_cleanup_done:
            return False
        prelease_cleanup_done = True
        try:
            outcome = await stop_task_lease_heartbeat(
                heartbeat_task,
                heartbeat_stop,
            )
        except BaseException:
            logger.error(
                "A2A prelease heartbeat failed before task %s could be restored; "
                "retaining run %s for TTL recovery",
                task_id,
                task_lease.run_id,
                exc_info=True,
            )
            return False
        if outcome.requires_ttl_recovery:
            logger.error(
                "A2A prelease for task %s became unhealthy; retaining run %s "
                "for TTL recovery (lost=%s, pool_timeout=%s)",
                task_id,
                task_lease.run_id,
                outcome.lease_lost,
                outcome.pool_timeout is not None,
            )
            return False
        return await _restore_a2a_resume_prelease_isolated(
            task_lease,
            status=resumable_status,
        )

    try:
        # Read before the injection below, not inside
        # _update_a2a_resume_input_sync, whose session opens only afterwards.
        # See task_interaction_close's module docstring for why.
        #
        # This site's turn id is deterministic (f"a2a:{task_id}:{message_id}"
        # below), so a retried A2A message replays the same turn.
        # AgentRunner.inject_user_message reports that replay explicitly
        # (see UserMessageInjectionOutcome) instead of folding it into the
        # same truthy value a fresh write produces, and
        # _update_a2a_resume_input_sync reads that report to decide whether
        # to skip its close call -- see the guard inside that function for
        # why a replay must skip it.
        active_interaction_read = await run_db_io_cancellation_safe(
            lambda: active_interaction_id_sync(task_id)
        )
        # Translate the three-state read into the `int | None` this site's
        # close call takes. Absent and Unavailable both become `None` here
        # -- but that is not folding Unavailable into Absent, it is this
        # call's own contract: `None` means "bind the close to no primary
        # key", so it matches zero rows and retires no question either
        # way, which is the safe outcome for a read that could not be
        # made, not a claim that nothing was ever active. What then
        # happens to the task's marker differs between the two, and this
        # value is not what decides it -- the clear beside the close runs
        # its own check (see active_interaction_id_sync's docstring).
        # Written as three branches, not
        # `interaction_id if isinstance(..., ActiveInteractionFound) else
        # None`, so a reader (and mypy) sees Unavailable handled on its own
        # line rather than merged into Absent's.
        if isinstance(active_interaction_read, ActiveInteractionFound):
            active_interaction_id = active_interaction_read.interaction_id
        elif isinstance(active_interaction_read, ActiveInteractionAbsent):
            active_interaction_id = None
        elif isinstance(active_interaction_read, ActiveInteractionUnavailable):
            active_interaction_id = None
            logger.info(
                "active interaction read unavailable (reason=%s) for "
                "task_id=%s; the legacy resume close will match no row",
                active_interaction_read.reason,
                task_id,
            )
        else:
            assert_never(active_interaction_read)

        async def inject_user_message() -> tuple[Any, UserMessageInjectionOutcome]:
            from .agent_service_manager import get_agent_manager

            agent_service = await get_agent_manager().get_agent_for_task(
                task_id,
                None,
                task_owner_user_id=task_owner_user_id,
            )
            posted = await agent_service.post_user_message(
                str(task_id),
                execution_message=text,
                display_message=text,
                turn_id=f"a2a:{task_id}:{message_id}",
                request_interrupt=False,
                reason="A2A input-required response",
            )
            return agent_service, posted

        with bind_task_lease_context(task_lease):
            agent_service, posted = await run_while_task_lease_owned(
                inject_user_message(),
                heartbeat_task,
            )

        message_posted = bool(posted)
        if not posted:
            # Untagged or otherwise unreadable legacy checkpoints are never
            # resumed under a fabricated run or transcript fallback. Release
            # the exact prelease back to the prior input-required state and
            # fail closed so a caller must start a new task explicitly.
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            if not await drain_async_task_cancellation_safe(cleanup_task):
                raise TaskLeaseLostError(
                    f"Task {task_id} lease changed before A2A fallback"
                )
            raise TaskResumeNotResumableError

        updated = await run_db_io_cancellation_safe(
            lambda: _update_a2a_resume_input_sync(
                task_lease, text, active_interaction_id, posted
            )
        )
        if not updated:
            raise TaskLeaseLostError(
                f"Task {task_id} lease changed before A2A resume scheduling"
            )

        await _schedule_waiting_a2a_resume(
            task_id=task_id,
            agent_service=agent_service,
            task_owner_user_id=task_owner_user_id,
            task_lease=task_lease,
            heartbeat_stop=heartbeat_stop,
            heartbeat_task=heartbeat_task,
            resumable_status=resumable_status,
        )
        ownership_transferred = True
    except CheckpointReadError as exc:
        # Ownership was never transferred, so restore the exact prelease
        # to the prior input-required status exactly like the absent-
        # checkpoint fallback above, then let the API translate the failure.
        # Keep the same ownership_transferred/prelease_cleanup_done guard as
        # the sibling except BaseException handler below.
        if not ownership_transferred and not prelease_cleanup_done:
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            if not await drain_async_task_cancellation_safe(cleanup_task):
                raise TaskLeaseLostError(
                    f"Task {task_id} lease changed before A2A checkpoint-failure fallback"
                ) from exc
            if not message_posted and isinstance(exc, CheckpointUnavailableError):
                # AgentRunner reads the checkpoint baseline before mutating
                # the context. A read failure here did not inject this reply.
                raise TaskResumeRetryableError(str(exc)) from exc
        raise
    except BaseException:
        if not ownership_transferred and not prelease_cleanup_done:
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            await drain_async_task_cancellation_safe(cleanup_task)
        raise
    finally:
        if not ownership_transferred and not prelease_cleanup_done:
            await stop_task_lease_heartbeat(heartbeat_task, heartbeat_stop)
    return True


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
# 3. A2A's own resume (_acquire_a2a_resume_prelease_sync): its claim
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
        # cleared. The premise that used to justify this as costless -- "a
        # row with no run_id never had a run-fenced checkpoint to recover in
        # the first place" -- no longer holds: the root-checkpoint read path
        # can resolve an untagged partition and read a checkpoint row
        # written before the run-partition field existed, so such a row can
        # have something resumable to lose. What survives of the original
        # reasoning is narrower and is about where the resume looks rather
        # than whether anything is there: with both pointer columns
        # cleared, that read path's by-primary-key anchor has nothing to
        # anchor on and falls back to its own legacy scan, which is keyed
        # on task/checkpoint-type/execution-id and partition rather than on
        # either cleared column. That scan reaches the row the pointer used
        # to name only when the row carries an execution identity of its
        # own: the scan filters on one, while the pointer path names a row
        # unconditionally and is deliberately the more permissive of the
        # two (see ``_load_pk_anchored_checkpoint``'s own docstring). For a
        # row predating that field, the cleared pointer was the only way to
        # reach it. Whether to keep clearing unconditionally has not been
        # re-decided under the corrected premise (see #2024); this comment
        # records the premise correctly rather than standing on the retired
        # one.
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

    Mirrors the A2A prelease restore (``_restore_a2a_resume_prelease_sync``):
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


def _update_reply_input_sync(
    task_lease: TaskLease, text: str, interaction_id: int | None
) -> bool:
    """Persist the reply's input only while the exact prelease remains current.

    ``interaction_id`` is the active interaction row the caller observed
    before injecting the message, passed in rather than read here: this
    function's session does not open until after the injection has already
    committed. See ``task_interaction_close``'s module docstring for why
    the read has to precede the injection.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        updated = (
            db.query(Task)
            .filter(
                Task.id == task_lease.task_id,
                Task.status == TaskStatus.RUNNING,
                Task.runner_id == task_lease.runner_id,
                task_lease_attempt_predicate(task_lease),
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
        # Mirrors _update_a2a_resume_input_sync's fence-ordering argument:
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
                db,
                task_id=task_lease.task_id,
                run_id=task_lease.run_id,
                interaction_id=interaction_id,
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
    if task_lease.task_id != task_id or task_lease.run_id is None:
        raise ValueError("Reply resume scheduling requires an exact task lease")
    if not task_execution_service.background_task_manager.reserve_resume(task_id):
        # A same-process resume is already registered for this task despite
        # the exclusive DB claim above -- report it the same way a caller
        # would read a lost race on the claim itself, rather than a bare
        # 500: retrying (after the other resume settles) can succeed.
        raise TaskResumeBusyError
    previous_task = task_execution_service.background_task_manager.running_tasks.get(
        task_id
    )
    bg_task: "asyncio.Task[None] | None" = None
    try:
        bg_task = asyncio.create_task(
            task_execution_service.execute_resume_background(
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
        task_execution_service.background_task_manager.register_reserved_resume(
            task_id,
            bg_task,
            run_id=task_lease.run_id,
        )
    except BaseException:
        if bg_task is not None:
            await cancel_and_drain_async_task(bg_task)
        task_execution_service.background_task_manager.release_resume_reservation(
            task_id
        )
        raise


async def resume_task_reply(
    ctx: TaskReplyInput,
    *,
    preacquired_lease: TaskLease | None = None,
    preacquired_state: dict[str, Any] | None = None,
    turn_id: str | None = None,
) -> TaskReplyResumeResult:
    """Resume an authorized SDK reply and return after background registration.

    Each SDK request gets a fresh turn ID, preserving its existing lack of
    client-message deduplication. The observed status preserves the initial
    rejection semantics; the subsequent conditional claim checks current state.
    """
    task_id = ctx.task_id
    # RUNNING is transient: a turn is in flight, so retrying later can
    # succeed (task_busy). Other non-waiting states have no interaction to
    # answer (no_pending_interaction). For PAUSED / COMPLETED / FAILED,
    # callers should append a new turn. PENDING cannot accept an append
    # yet; callers must wait for the task to leave that state.
    if ctx.status != TaskStatus.WAITING_FOR_USER:
        if ctx.status == TaskStatus.RUNNING:
            raise TaskResumeBusyError
        raise TaskResumeNotWaitingError

    from .task_execution_host import enqueues_task_turns

    if enqueues_task_turns() and preacquired_lease is None:
        from .task_resume_command import enqueue_resume_input

        return await enqueue_resume_input(ctx, source="sdk", message_id="")
    prelease_info: dict[str, Any] = preacquired_state or {}
    task_lease = preacquired_lease or await acquire_task_lease_cancellation_safe(
        lambda: _acquire_reply_prelease_sync(
            task_id=ctx.task_id,
            agent_id=ctx.agent_id,
            previous_run_id=ctx.run_id,
            result_holder=prelease_info,
        ),
        lambda acquired: _restore_reply_prelease_sync(acquired),
    )
    if task_lease is None or task_lease.run_id is None:
        raise TaskResumeBusyError

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
        # Read before the injection below, not inside
        # _update_reply_input_sync, whose session opens only afterwards.
        # See task_interaction_close's module docstring for why.
        active_interaction_read = await run_db_io_cancellation_safe(
            lambda: active_interaction_id_sync(task_id)
        )
        # Translate the three-state read into the `int | None` this site's
        # close call takes. Absent and Unavailable both become `None` here
        # -- but that is not folding Unavailable into Absent, it is this
        # call's own contract: `None` means "bind the close to no primary
        # key", so it matches zero rows and retires no question either
        # way, which is the safe outcome for a read that could not be
        # made, not a claim that nothing was ever active. What then
        # happens to the task's marker differs between the two, and this
        # value is not what decides it -- the clear beside the close runs
        # its own check (see active_interaction_id_sync's docstring).
        # Written as three branches, not
        # `interaction_id if isinstance(..., ActiveInteractionFound) else
        # None`, so a reader (and mypy) sees Unavailable handled on its own
        # line rather than merged into Absent's.
        if isinstance(active_interaction_read, ActiveInteractionFound):
            active_interaction_id = active_interaction_read.interaction_id
        elif isinstance(active_interaction_read, ActiveInteractionAbsent):
            active_interaction_id = None
        elif isinstance(active_interaction_read, ActiveInteractionUnavailable):
            active_interaction_id = None
            logger.info(
                "active interaction read unavailable (reason=%s) for "
                "task_id=%s; the legacy resume close will match no row",
                active_interaction_read.reason,
                task_id,
            )
        else:
            assert_never(active_interaction_read)

        async def inject_user_message() -> tuple[Any, bool]:
            agent_service = (
                await agent_runtime_service.get_agent_manager().get_agent_for_task(
                    task_id,
                    None,
                    task_owner_user_id=ctx.task_owner_user_id,
                )
            )
            posted = await agent_service.post_user_message(
                str(task_id),
                execution_message=ctx.text,
                display_message=ctx.text,
                turn_id=turn_id or f"v1:reply:{task_id}:{uuid4()}",
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
            raise TaskResumeNotResumableError

        updated = await run_db_io_cancellation_safe(
            lambda: _update_reply_input_sync(
                task_lease, ctx.text, active_interaction_id
            )
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
        raise
    except BaseException:
        if not ownership_transferred and not prelease_cleanup_done:
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            await drain_async_task_cancellation_safe(cleanup_task)
        raise
    finally:
        if not ownership_transferred and not prelease_cleanup_done:
            await stop_task_lease_heartbeat(heartbeat_task, heartbeat_stop)

    return TaskReplyResumeResult(
        run_id=task_lease.run_id,
        state_version=prelease_info["state_version"],
        control_state=prelease_info["control_state"],
    )
