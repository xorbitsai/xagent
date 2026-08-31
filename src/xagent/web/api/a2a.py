from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any, Mapping

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import String, and_, cast, func, or_, update

from ...core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointReadError,
)
from ..models.agent import Agent
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..services.a2a_protocol import (
    A2A_VERSION,
    ALL_TASK_STATES,
    A2AAgentCardSnapshot,
    A2ATaskPageSnapshot,
    A2ATaskSnapshot,
    a2a_error,
    a2a_json_response,
    a2a_task_state_filter,
    build_agent_card,
    extract_message_text,
    is_published_agent,
    message_context_id,
    message_task_id,
    new_context_id,
    sse_task_artifacts,
    sse_task_snapshot,
    sse_task_update,
    task_context_id,
    task_state,
    task_to_a2a,
)
from ..services.db_runtime import (
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ..services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    TaskCommandKind,
    dispatch_one_task_command,
    enqueue_task_command,
    load_task_command,
    retry_failed_task_command,
)
from ..services.task_execution_controller import (
    StaleTaskRunError,
    TaskControlState,
    task_execution_controller,
)
from ..services.task_interaction_close import (
    active_interaction_id_sync,
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction,
)
from ..services.task_interaction_schema import interaction_requests_table_exists
from ..services.task_lease_service import (
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
from ..services.task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
    _ClaimedTurn,
)
from .v1.deps import (
    AgentPrincipalSnapshot,
    RuntimeApiKeySnapshot,
    get_agent_from_api_key,
    record_key_usage,
)
from .v1.errors import V1ApiError

router = APIRouter(prefix="/api/a2a", tags=["a2a"])
logger = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)
_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}
_STREAM_END_STATES = {
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_AUTH_REQUIRED",
}
A2A_BLOCKING_WAIT_TIMEOUT_SECONDS = 60.0
A2A_STREAM_MAX_DURATION_SECONDS = 60.0 * 60.0


async def _get_a2a_agent_from_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot]:
    _validate_a2a_version(request)
    try:
        return await get_agent_from_api_key(credentials)
    except V1ApiError as exc:
        raise a2a_error(
            "invalid_api_key",
            exc.message,
            status_code=exc.http_status,
        ) from exc


def _load_published_agent_card_sync(agent_id: int) -> A2AAgentCardSnapshot:
    """Load public Agent Card fields in one worker-owned Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if agent is None or not is_published_agent(agent):
            raise a2a_error("agent_not_found", "Agent not found.", status_code=404)
        return A2AAgentCardSnapshot.from_agent(agent)


async def _load_published_agent_card_isolated(
    agent_id: int,
) -> A2AAgentCardSnapshot:
    return await run_db_io_cancellation_safe(
        lambda: _load_published_agent_card_sync(agent_id)
    )


def _task_run_id(task: Task) -> str | None:
    run_id = getattr(task, "run_id", None)
    return str(run_id) if run_id is not None else None


def _require_bound_agent(path_agent_id: int, agent: AgentPrincipalSnapshot) -> None:
    if int(agent.id) != int(path_agent_id) or not is_published_agent(agent):
        raise a2a_error("agent_not_found", "Agent not found.", status_code=404)


async def _schedule_waiting_a2a_resume(
    *,
    task_id: int,
    agent_service: Any,
    task_owner_user_id: int,
    task_lease: TaskLease,
    heartbeat_stop: asyncio.Event,
    heartbeat_task: asyncio.Task[TaskLeaseHeartbeatOutcome],
    resumable_status: TaskStatus,
    connector_runtime_turn_id: str,
) -> None:
    from .chat import get_agent_manager
    from .websocket import background_task_manager, execute_resume_background

    if task_lease.task_id != task_id or task_lease.run_id is None:
        raise ValueError("A2A resume scheduling requires an exact task lease")
    if not background_task_manager.reserve_resume(task_id):
        raise RuntimeError(f"Task {task_id} already has a resume in progress")
    previous_task = background_task_manager.running_tasks.get(task_id)
    bg_task: asyncio.Task[None] | None = None
    try:
        # This request is now the sole admitted owner of the cached agent's
        # tool_config for this task (see websocket.py's message-triggered
        # and explicit-resume handlers for the same reasoning) - only now
        # is it safe to sync the connector runtime turn binding to the turn
        # whose message was just injected, so a reconnected app's tools
        # rebuild against it instead of leaving a losing resume's turn/
        # cache context on the shared agent this one is about to execute
        # under. Inside the try so a raise here still releases the
        # reservation below, instead of leaking it.
        get_agent_manager().sync_connector_runtime_turn(
            task_id, connector_runtime_turn_id
        )
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
    ``_resume_input_required_a2a_task``'s lease acquisition, which fires on
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
) -> bool:
    """Persist A2A input only while the exact prelease remains current.

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
            close_legacy_resume_interaction(
                db,
                task_id=task_lease.task_id,
                run_id=task_lease.run_id,
                interaction_id=interaction_id,
            )
        db.commit()
        return True


async def _resume_input_required_a2a_task(
    *,
    agent_id: int,
    task_owner_user_id: int,
    task: A2ATaskSnapshot,
    text: str,
    message_id: str,
) -> bool:
    task_id = int(task.id)
    resumable_status = task.status
    if resumable_status not in {
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
    }:
        return False
    task_lease = await acquire_task_lease_cancellation_safe(
        lambda: _acquire_a2a_resume_prelease_sync(
            task_id=task_id,
            agent_id=agent_id,
            resumable_status=resumable_status,
            previous_run_id=task.run_id,
        ),
        lambda acquired: _restore_a2a_resume_prelease_sync(
            acquired,
            status=resumable_status,
        ),
    )
    if task_lease is None or task_lease.run_id is None:
        raise a2a_error(
            "unsupported_operation",
            "Task is currently running and cannot accept a new message.",
            status_code=400,
            details={"taskId": task_id},
        )

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
        # below), so a retried A2A message replays the same turn:
        # AgentRunner.inject_user_message short-circuits a repeated turn id
        # by returning the existing context without persisting anything, and
        # reports the same truthy `posted` a first attempt does. On such a
        # replay the id read here is not the question the replayed message
        # answered but whatever the resumed agent has asked since, and the
        # close would retire a live question. Nothing can be retired today:
        # the only INSERT into task_interaction_requests is
        # stage_interaction_request (task_interaction_staging.py), which has
        # no caller in src/ and is held at none by
        # tests/web/services/test_interaction_staging_production_gate.py, so
        # this read returns None on every call. Closing the window is a
        # precondition on the change that wires the first production writer,
        # stated once at the online WebSocket injection site (websocket.py)
        # and binding here identically.
        active_interaction_id = await run_db_io_cancellation_safe(
            lambda: active_interaction_id_sync(task_id)
        )

        a2a_turn_id = f"a2a:{task_id}:{message_id}"

        async def inject_user_message() -> tuple[Any, bool]:
            from .chat import get_agent_manager

            agent_service = await get_agent_manager().get_agent_for_task(
                task_id,
                None,
                task_owner_user_id=task_owner_user_id,
            )
            posted = await agent_service.post_user_message(
                str(task_id),
                execution_message=text,
                display_message=text,
                turn_id=a2a_turn_id,
                request_interrupt=False,
                reason="A2A input-required response",
            )
            return agent_service, bool(posted)

        with bind_task_lease_context(task_lease):
            agent_service, posted = await run_while_task_lease_owned(
                inject_user_message(),
                heartbeat_task,
            )

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
            raise a2a_error(
                "unsupported_operation",
                "No run-fenced checkpoint is available for this task.",
                status_code=400,
                details={"taskId": task_id},
            )

        updated = await run_db_io_cancellation_safe(
            lambda: _update_a2a_resume_input_sync(
                task_lease, text, active_interaction_id
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
            connector_runtime_turn_id=a2a_turn_id,
        )
        ownership_transferred = True
    except CheckpointReadError as exc:
        # Ownership was never transferred, so restore the exact prelease
        # to the prior input-required status exactly like the absent-
        # checkpoint fallback above, then translate the failure instead of
        # letting it escape as a raw exception. Same ownership_transferred/
        # prelease_cleanup_done guard as the sibling except BaseException
        # handler below, for symmetry.
        if not ownership_transferred and not prelease_cleanup_done:
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            if not await drain_async_task_cancellation_safe(cleanup_task):
                raise TaskLeaseLostError(
                    f"Task {task_id} lease changed before A2A checkpoint-failure fallback"
                ) from exc
        if isinstance(exc, CheckpointCorruptError):
            raise a2a_error(
                "unsupported_operation",
                "The task's saved progress is unreadable.",
                status_code=400,
                details={"taskId": task_id},
            ) from exc
        if isinstance(exc, CheckpointAccessRefusedError):
            message = {
                "lease_mismatch": (
                    "This task is currently owned by a different execution "
                    "and cannot accept a new message."
                ),
                "superseded_legacy": (
                    "This task's checkpoint history has been superseded by "
                    "a newer run and cannot accept a new message."
                ),
            }.get(
                exc.reason,
                "Task is currently running and cannot accept a new message.",
            )
            raise a2a_error(
                "unsupported_operation",
                message,
                status_code=400,
                details={"taskId": task_id},
            ) from exc
        # CheckpointUnavailableError, or any future CheckpointReadError
        # subclass this dispatch does not yet know about: treat it
        # conservatively as retryable rather than assuming a terminal or
        # policy failure, so an unrecognized failure mode never silently
        # collapses into a data-losing branch.
        raise a2a_error(
            "temporarily_unavailable",
            "The task's saved progress could not be read. Please retry.",
            status_code=503,
            details={"taskId": task_id},
        ) from exc
    except BaseException:
        if not ownership_transferred and not prelease_cleanup_done:
            cleanup_task = asyncio.create_task(stop_and_restore_prelease())
            await drain_async_task_cancellation_safe(cleanup_task)
        raise
    finally:
        if not ownership_transferred and not prelease_cleanup_done:
            await stop_task_lease_heartbeat(heartbeat_task, heartbeat_stop)
    return True


def _validate_a2a_version(request: Request) -> None:
    requested = request.headers.get("A2A-Version")
    if requested is None:
        requested = request.query_params.get("A2A-Version")
    if requested is None or not requested.strip():
        raise a2a_error(
            "version_not_supported",
            "A2A-Version header or query parameter is required.",
            status_code=400,
            details={"supportedVersions": A2A_VERSION},
        )
    requested = requested.strip()
    version_parts = requested.split(".")
    compatible = (
        len(version_parts) in {2, 3}
        and all(part.isdecimal() for part in version_parts)
        and version_parts[0] == A2A_VERSION.split(".", maxsplit=1)[0]
    )
    if not compatible:
        raise a2a_error(
            "version_not_supported",
            f"A2A protocol version {requested!r} is not supported.",
            status_code=400,
            details={"supportedVersions": A2A_VERSION},
        )


def _validate_send_configuration(body: Mapping[str, Any]) -> bool:
    configuration = body.get("configuration")
    if configuration is None:
        return False
    if not isinstance(configuration, Mapping):
        raise a2a_error(
            "invalid_argument",
            "configuration must be a JSON object.",
            status_code=400,
            details={"field": "configuration"},
        )
    if configuration.get("taskPushNotificationConfig") is not None:
        raise a2a_error(
            "push_notification_not_supported",
            "This agent does not support A2A push notifications.",
            status_code=400,
        )
    accepted_modes = configuration.get("acceptedOutputModes")
    if accepted_modes is not None:
        if not isinstance(accepted_modes, list) or not all(
            isinstance(mode, str) for mode in accepted_modes
        ):
            raise a2a_error(
                "invalid_argument",
                "acceptedOutputModes must be an array of media types.",
                status_code=400,
                details={"field": "configuration.acceptedOutputModes"},
            )
        if accepted_modes and "text/plain" not in accepted_modes:
            raise a2a_error(
                "content_type_not_supported",
                "This agent currently returns text/plain output only.",
                status_code=400,
                details={"supportedMediaType": "text/plain"},
            )
    return_immediately = configuration.get("returnImmediately", False)
    if not isinstance(return_immediately, bool):
        raise a2a_error(
            "invalid_argument",
            "returnImmediately must be a boolean.",
            status_code=400,
            details={"field": "configuration.returnImmediately"},
        )
    return return_immediately


@dataclass(frozen=True)
class _A2ATurnPreparation:
    """Detached result of the worker-owned A2A preparation transaction."""

    task: A2ATaskSnapshot
    created_task: bool
    kind: TurnKind
    payload: TaskTurnPayload
    claimed_turn: _ClaimedTurn | None


def _prepare_a2a_turn_sync(
    *,
    agent_id: int,
    task_owner_user_id: int,
    agent_execution_mode: str,
    text: str,
    context_id: str | None,
    task_id: int | None,
) -> _A2ATurnPreparation:
    """Create/claim or validate an A2A turn in one worker-owned transaction."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        payload = TaskTurnPayload(transcript_message=text)
        created_task = task_id is None
        if task_id is None:
            context_id = context_id or new_context_id()
            task = Task(
                user_id=task_owner_user_id,
                title=(text[:50] or "A2A task"),
                description=text,
                status=TaskStatus.PENDING,
                agent_id=agent_id,
                input=text,
                source="a2a",
                is_visible=False,
                execution_mode=agent_execution_mode,
                agent_config={"a2a_context_id": context_id},
            )
            db.add(task)
            db.flush()
            claimed_turn = TaskTurnOrchestrator.claim_created_turn_no_commit(
                db,
                task_id=int(task.id),
                task_owner_user_id=task_owner_user_id,
                payload=payload,
            )
            db.flush()
            db.refresh(task)
            task_snapshot = A2ATaskSnapshot.from_task(task)
            db.commit()
            kind = TurnKind.CREATE
        else:
            existing_task = (
                db.query(Task)
                .filter(
                    Task.id == task_id,
                    Task.agent_id == agent_id,
                    Task.user_id == task_owner_user_id,
                    Task.source == "a2a",
                )
                .first()
            )
            if existing_task is None:
                raise a2a_error(
                    "task_not_found",
                    "Task not found.",
                    status_code=404,
                )
            task = existing_task
            if task.status in _TERMINAL_STATUSES:
                raise a2a_error(
                    "unsupported_operation",
                    "Messages cannot be appended to a terminal A2A task.",
                    status_code=400,
                    details={"taskId": task.id},
                )
            stored_context_id = task_context_id(task)
            if context_id is not None and context_id != stored_context_id:
                raise a2a_error(
                    "invalid_argument",
                    "The supplied contextId does not match the referenced task.",
                    status_code=400,
                    details={"taskId": task.id, "contextId": context_id},
                )
            agent_config: dict[str, Any] = (
                dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
            )
            if not agent_config.get("a2a_context_id"):
                agent_config["a2a_context_id"] = stored_context_id
                setattr(task, "agent_config", agent_config)
                db.commit()
                db.refresh(task)
            task_snapshot = A2ATaskSnapshot.from_task(task)
            claimed_turn = None
            kind = TurnKind.APPEND
        return _A2ATurnPreparation(
            task=task_snapshot,
            created_task=created_task,
            kind=kind,
            payload=payload,
            claimed_turn=claimed_turn,
        )


async def _start_a2a_turn(
    *,
    agent_id: int,
    task_owner_user_id: int,
    agent_execution_mode: str,
    text: str,
    message_id: str,
    context_id: str | None,
    task_id: int | None,
) -> A2ATaskSnapshot:
    async def start_unserialized() -> A2ATaskSnapshot:
        preparation = await run_db_io_cancellation_safe(
            lambda: _prepare_a2a_turn_sync(
                agent_id=agent_id,
                task_owner_user_id=task_owner_user_id,
                agent_execution_mode=agent_execution_mode,
                text=text,
                context_id=context_id,
                task_id=task_id,
            )
        )

        prepared_task = preparation.task
        if prepared_task.status in {
            TaskStatus.PAUSED,
            TaskStatus.WAITING_FOR_USER,
        }:
            await _resume_input_required_a2a_task(
                agent_id=agent_id,
                task_owner_user_id=task_owner_user_id,
                task=prepared_task,
                text=text,
                message_id=message_id,
            )
            fresh = await _fetch_fresh_a2a_task_isolated(
                agent_id,
                prepared_task.id,
            )
            if fresh is None:
                raise a2a_error(
                    "task_not_found",
                    "Task not found.",
                    status_code=404,
                )
            return fresh

        try:
            if preparation.created_task:
                if preparation.claimed_turn is None:
                    raise RuntimeError(
                        "created A2A task did not stage its initial turn"
                    )
                await TaskTurnOrchestrator.schedule_claimed_create_turn(
                    task_id=prepared_task.id,
                    task_owner_user_id=task_owner_user_id,
                    actor_user_id=task_owner_user_id,
                    payload=preparation.payload,
                    claimed=preparation.claimed_turn,
                )
            else:
                await TaskTurnOrchestrator.begin_turn(
                    task_id=prepared_task.id,
                    task_owner_user_id=task_owner_user_id,
                    actor_user_id=task_owner_user_id,
                    payload=preparation.payload,
                    kind=preparation.kind,
                    force_fresh=False,
                )
        except TaskTurnNotFoundError as exc:
            raise a2a_error(
                "task_not_found",
                "Task not found.",
                status_code=404,
            ) from exc
        except TaskTurnError as exc:
            raise a2a_error(
                "unsupported_operation",
                "Task is currently running and cannot accept a new message.",
                status_code=400,
                details={"taskId": prepared_task.id},
            ) from exc

        fresh = await _fetch_fresh_a2a_task_isolated(
            agent_id,
            prepared_task.id,
        )
        if fresh is None:
            raise a2a_error(
                "task_not_found",
                "Task not found.",
                status_code=404,
            )
        return fresh

    if task_id is not None:
        async with task_execution_controller.command(task_id):
            return await start_unserialized()
    start_task = asyncio.create_task(start_unserialized())
    return await drain_async_task_cancellation_safe(start_task)


async def _json_body(request: Request) -> Mapping[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise a2a_error(
            "invalid_request",
            "Request body must be valid JSON.",
            status_code=400,
        ) from exc
    if not isinstance(body, Mapping):
        raise a2a_error(
            "invalid_argument", "Request body must be a JSON object.", status_code=400
        )
    return body


def _message_payload(body: Mapping[str, Any]) -> Mapping[str, Any]:
    message = body.get("message")
    if not isinstance(message, Mapping):
        raise a2a_error(
            "invalid_argument",
            "Request body must include a message object.",
            status_code=400,
        )
    return message


def _message_id(message: Mapping[str, Any]) -> str:
    value = message.get("messageId")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise a2a_error(
        "invalid_argument",
        "message.messageId must be a non-empty string.",
        status_code=400,
        details={"field": "message.messageId"},
    )


def _fetch_fresh_a2a_task(
    agent_id: int,
    task_id: int,
) -> A2ATaskSnapshot | None:
    """Load one detached A2A task snapshot in a worker-owned Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        fresh = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        return A2ATaskSnapshot.from_task(fresh) if fresh is not None else None


async def _fetch_fresh_a2a_task_isolated(
    agent_id: int,
    task_id: int,
) -> A2ATaskSnapshot | None:
    return await run_db_io_cancellation_safe(
        lambda: _fetch_fresh_a2a_task(agent_id, task_id)
    )


async def _load_a2a_task_or_error(
    agent_id: int,
    task_id: int,
) -> A2ATaskSnapshot:
    task = await _fetch_fresh_a2a_task_isolated(agent_id, task_id)
    if task is None:
        raise a2a_error("task_not_found", "Task not found.", status_code=404)
    return task


def _task_stream_ended(task: Task | A2ATaskSnapshot) -> bool:
    return task_state(task) in _STREAM_END_STATES


def _task_stream_response(
    agent_id: int,
    task: A2ATaskSnapshot,
) -> StreamingResponse:
    started_task_id = int(task.id)

    async def _events() -> Any:
        deadline = monotonic() + A2A_STREAM_MAX_DURATION_SECONDS
        yield sse_task_snapshot(task)
        if _task_stream_ended(task):
            return
        previous_state = task_state(task)
        previous_output = str(task.output or "")
        previous_error = str(task.error_message or "")
        artifact_finalized = bool(previous_output) and _task_stream_ended(task)
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.5, remaining))
            fresh = await _fetch_fresh_a2a_task_isolated(
                agent_id,
                started_task_id,
            )
            if fresh is None:
                return
            fresh_output = str(fresh.output or "")
            fresh_state = task_state(fresh)
            fresh_error = str(fresh.error_message or "")
            stream_ended = _task_stream_ended(fresh)
            if fresh_output and fresh_output != previous_output:
                append = bool(previous_output) and fresh_output.startswith(
                    previous_output
                )
                chunk = fresh_output[len(previous_output) :] if append else fresh_output
                artifacts = sse_task_artifacts(
                    fresh,
                    text=chunk,
                    append=append,
                    last_chunk=stream_ended,
                )
                if artifacts:
                    yield artifacts
                artifact_finalized = stream_ended
            elif stream_ended and fresh_output and not artifact_finalized:
                artifacts = sse_task_artifacts(
                    fresh,
                    text=fresh_output,
                    append=False,
                    last_chunk=True,
                )
                if artifacts:
                    yield artifacts
                artifact_finalized = True
            if fresh_state != previous_state or fresh_error != previous_error:
                yield sse_task_update(fresh)
            previous_state = fresh_state
            previous_output = fresh_output
            previous_error = fresh_error
            if stream_ended:
                return

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"A2A-Version": A2A_VERSION},
    )


async def _wait_for_task(
    agent_id: int,
    task: A2ATaskSnapshot,
) -> A2ATaskSnapshot:
    if _task_stream_ended(task):
        return task
    task_id = int(task.id)
    deadline = monotonic() + A2A_BLOCKING_WAIT_TIMEOUT_SECONDS
    fresh = task
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return fresh
        await asyncio.sleep(min(0.25, remaining))
        fetched = await _fetch_fresh_a2a_task_isolated(agent_id, task_id)
        if fetched is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        fresh = fetched
        if _task_stream_ended(fresh):
            return fresh


def _page_offset(page_token: str | None) -> int:
    if page_token is None or page_token == "":
        return 0
    if page_token.isdecimal():
        return int(page_token)
    raise a2a_error(
        "invalid_argument",
        "pageToken is invalid.",
        status_code=400,
        details={"field": "pageToken"},
    )


def _load_a2a_task_page_sync(
    *,
    agent_id: int,
    context_id: str | None,
    status: str | None,
    status_timestamp_after: datetime | None,
    offset: int,
    page_size: int,
) -> A2ATaskPageSnapshot:
    """Load one stable A2A task page in a worker-owned Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        query = db.query(Task).filter(
            Task.agent_id == agent_id,
            Task.source == "a2a",
        )
        if context_id is not None:
            stored_context_id = Task.agent_config["a2a_context_id"].as_string()
            query = query.filter(
                or_(
                    stored_context_id == context_id,
                    and_(
                        stored_context_id.is_(None),
                        cast(Task.id, String) == context_id,
                    ),
                )
            )
        if status is not None:
            query = query.filter(a2a_task_state_filter(status))
        if status_timestamp_after is not None:
            query = query.filter(Task.updated_at > status_timestamp_after)

        total_size = query.count()
        rows = query.order_by(Task.id.desc()).offset(offset).limit(page_size).all()
        tasks = tuple(A2ATaskSnapshot.from_task(task) for task in rows)
        next_offset = offset + len(tasks)
        return A2ATaskPageSnapshot(
            tasks=tasks,
            next_page_token=(str(next_offset) if next_offset < total_size else ""),
            page_size=page_size,
            total_size=total_size,
        )


async def _load_a2a_task_page_isolated(
    *,
    agent_id: int,
    context_id: str | None,
    status: str | None,
    status_timestamp_after: datetime | None,
    offset: int,
    page_size: int,
) -> A2ATaskPageSnapshot:
    return await run_db_io_cancellation_safe(
        lambda: _load_a2a_task_page_sync(
            agent_id=agent_id,
            context_id=context_id,
            status=status,
            status_timestamp_after=status_timestamp_after,
            offset=offset,
            page_size=page_size,
        )
    )


@router.get("/agents/{agent_id}/.well-known/agent-card.json")
async def get_agent_card_well_known(
    agent_id: int,
    request: Request,
) -> Any:
    agent = await _load_published_agent_card_isolated(agent_id)
    return a2a_json_response(build_agent_card(agent, request))


@router.post("/agents/{agent_id}/message:send")
async def send_message(
    agent_id: int,
    request: Request,
    authed: tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot] = Depends(
        _get_a2a_agent_from_api_key
    ),
) -> Any:
    agent, key = authed
    _require_bound_agent(agent_id, agent)
    bound_agent_id = int(agent.id)
    task_owner_user_id = int(agent.user_id)
    agent_execution_mode = str(agent.execution_mode)
    key_prefix = str(key.key_prefix)
    body = await _json_body(request)
    return_immediately = _validate_send_configuration(body)
    message = _message_payload(body)
    text = extract_message_text(message)
    message_id = _message_id(message)
    context_id = message_context_id(message, body)
    task_id = message_task_id(message, body)
    task = await _start_a2a_turn(
        agent_id=bound_agent_id,
        task_owner_user_id=task_owner_user_id,
        agent_execution_mode=agent_execution_mode,
        text=text,
        message_id=message_id,
        context_id=context_id,
        task_id=task_id,
    )
    await record_key_usage(key_prefix)
    if not return_immediately:
        task = await _wait_for_task(bound_agent_id, task)
    return a2a_json_response({"task": task_to_a2a(task)})


@router.post("/agents/{agent_id}/message:stream")
async def stream_message(
    agent_id: int,
    request: Request,
    authed: tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot] = Depends(
        _get_a2a_agent_from_api_key
    ),
) -> StreamingResponse:
    agent, key = authed
    _require_bound_agent(agent_id, agent)
    bound_agent_id = int(agent.id)
    task_owner_user_id = int(agent.user_id)
    agent_execution_mode = str(agent.execution_mode)
    key_prefix = str(key.key_prefix)
    body = await _json_body(request)
    _validate_send_configuration(body)
    message = _message_payload(body)
    text = extract_message_text(message)
    message_id = _message_id(message)
    context_id = message_context_id(message, body)
    task_id = message_task_id(message, body)
    task = await _start_a2a_turn(
        agent_id=bound_agent_id,
        task_owner_user_id=task_owner_user_id,
        agent_execution_mode=agent_execution_mode,
        text=text,
        message_id=message_id,
        context_id=context_id,
        task_id=task_id,
    )
    await record_key_usage(key_prefix)
    return _task_stream_response(bound_agent_id, task)


@router.get("/agents/{agent_id}/tasks/{task_id}")
async def get_task(
    agent_id: int,
    task_id: int,
    authed: tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot] = Depends(
        _get_a2a_agent_from_api_key
    ),
) -> Any:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    task = await _load_a2a_task_or_error(int(agent.id), task_id)
    return a2a_json_response(task_to_a2a(task))


@router.get("/agents/{agent_id}/tasks")
async def list_tasks(
    agent_id: int,
    context_id: str | None = Query(default=None, alias="contextId"),
    status: str | None = Query(default=None),
    page_size: int = Query(default=50, ge=1, le=100, alias="pageSize"),
    page_token: str | None = Query(default=None, alias="pageToken"),
    include_artifacts: bool = Query(default=False, alias="includeArtifacts"),
    status_timestamp_after: datetime | None = Query(
        default=None,
        alias="statusTimestampAfter",
        description=(
            "Filter by the timestamp exposed in each A2A task status; "
            "this is backed by Task.updated_at."
        ),
    ),
    authed: tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot] = Depends(
        _get_a2a_agent_from_api_key
    ),
) -> Any:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    if status is not None:
        if status not in ALL_TASK_STATES:
            raise a2a_error(
                "invalid_argument",
                f"Unknown A2A task status: {status}",
                status_code=400,
                details={"field": "status"},
            )

    offset = _page_offset(page_token)
    page = await _load_a2a_task_page_isolated(
        agent_id=int(agent.id),
        context_id=context_id,
        status=status,
        status_timestamp_after=status_timestamp_after,
        offset=offset,
        page_size=page_size,
    )
    return a2a_json_response(
        {
            "tasks": [
                task_to_a2a(task, include_artifacts=include_artifacts)
                for task in page.tasks
            ],
            "nextPageToken": page.next_page_token,
            "pageSize": page.page_size,
            "totalSize": page.total_size,
        }
    )


@router.api_route(
    "/agents/{agent_id}/tasks/{task_id}:subscribe", methods=["GET", "POST"]
)
async def subscribe_task(
    agent_id: int,
    task_id: int,
    authed: tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot] = Depends(
        _get_a2a_agent_from_api_key
    ),
) -> StreamingResponse:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    bound_agent_id = int(agent.id)
    task_snapshot = await _load_a2a_task_or_error(bound_agent_id, task_id)
    if task_snapshot.status in _TERMINAL_STATUSES:
        raise a2a_error(
            "unsupported_operation",
            "A terminal task cannot be subscribed to.",
            status_code=400,
            details={"taskId": task_snapshot.id},
        )
    return _task_stream_response(bound_agent_id, task_snapshot)


@router.post("/agents/{agent_id}/tasks/{task_id}:cancel")
async def cancel_task(
    agent_id: int,
    task_id: int,
    authed: tuple[AgentPrincipalSnapshot, RuntimeApiKeySnapshot] = Depends(
        _get_a2a_agent_from_api_key
    ),
) -> Any:
    agent, _key = authed
    _require_bound_agent(agent_id, agent)
    bound_agent_id = int(agent.id)
    task_owner_user_id = int(agent.user_id)
    command = await run_db_io_cancellation_safe(
        lambda: _prepare_a2a_cancel_command_sync(
            task_id=task_id,
            agent_id=bound_agent_id,
            task_owner_user_id=task_owner_user_id,
        )
    )

    from .websocket import execute_durable_task_command

    # Apply immediately when this process owns the target run. If another
    # worker owns it, that worker's dispatcher observes the durable row and
    # completes it; polling here preserves the synchronous A2A cancel contract.
    await dispatch_one_task_command(
        execute_durable_task_command,
        command_db_id=command.command_db_id,
    )
    deadline = monotonic() + 10.0
    while True:
        stored = await run_db_io_cancellation_safe(
            lambda: load_task_command(command.command_db_id)
        )
        if stored is not None and stored.status == COMMAND_COMPLETED:
            fresh = await _fetch_fresh_a2a_task_isolated(bound_agent_id, task_id)
            if fresh is None:
                raise a2a_error(
                    "task_not_found",
                    "Task not found.",
                    status_code=404,
                )
            return a2a_json_response(task_to_a2a(fresh))
        if stored is not None and stored.status == COMMAND_FAILED:
            rejection_reason = (
                stored.result.get("rejection_reason")
                if isinstance(stored.result, dict)
                else None
            )
            if rejection_reason == "stale_run":
                raise a2a_error(
                    "invalid_request",
                    "Task run changed before cancellation was applied; retry the request.",
                    status_code=409,
                    details={
                        "taskId": task_id,
                        "commandId": command.command_identity,
                    },
                )
            raise a2a_error(
                "internal_error",
                str(stored.error or "Task cancellation failed."),
                status_code=500,
            )
        if monotonic() >= deadline:
            raise a2a_error(
                "temporarily_unavailable",
                "Task cancellation was accepted but is still being applied.",
                status_code=503,
                details={"taskId": task_id, "commandId": command.command_identity},
            )
        await asyncio.sleep(0.05)


@dataclass(frozen=True)
class _A2ACancelCommand:
    command_db_id: int
    command_identity: str


def _prepare_a2a_cancel_command_sync(
    *,
    task_id: int,
    agent_id: int,
    task_owner_user_id: int,
) -> _A2ACancelCommand:
    """Authorize and persist one A2A cancel command in a short transaction."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.user_id == task_owner_user_id,
                Task.source == "a2a",
            )
            .first()
        )
        if task is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        target_state_version = int(task.state_version or 0)
        command_identity = f"cancel:{task_id}:{target_state_version}"
        enqueued = enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=task_owner_user_id,
            command_id=command_identity,
            kind=TaskCommandKind.CANCEL,
            payload={
                "agent_id": agent_id,
                "target_state_version": target_state_version,
            },
        )
        if not enqueued.payload_matches:
            raise a2a_error(
                "invalid_request",
                "Cancel command identity conflicts with a different request.",
                status_code=409,
            )
        if enqueued.status == COMMAND_FAILED:
            retry_failed_task_command(db, enqueued.command_id)
        return _A2ACancelCommand(
            command_db_id=enqueued.command_id,
            command_identity=command_identity,
        )


def _load_cancelable_a2a_task_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> A2ATaskSnapshot:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        if task is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        if _is_completed_a2a_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return A2ATaskSnapshot.from_task(task)
        _assert_a2a_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
        return A2ATaskSnapshot.from_task(task)


def _assert_a2a_cancel_target(
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
    ):
        raise StaleTaskRunError(
            f"task {task.id} changed from run/version "
            f"{expected_run_id}/{expected_state_version} to "
            f"{current_run_id}/{current_state_version}"
        )


def _is_completed_a2a_cancel_target(
    task: Task,
    *,
    expected_run_id: str | None,
    expected_state_version: int,
) -> bool:
    """Validate an idempotent cancel replay against its immutable target."""

    agent_config: dict[str, Any] = (
        task.agent_config if isinstance(task.agent_config, dict) else {}
    )
    if agent_config.get("a2a_state") != "TASK_STATE_CANCELED":
        return False

    current_run_id = _task_run_id(task)
    current_state_version = int(task.state_version or 0)
    is_exact_completion = (
        current_run_id == expected_run_id
        and current_state_version
        in {expected_state_version, expected_state_version + 1}
        and task.status == TaskStatus.FAILED
        and task.control_state == TaskControlState.FAILED.value
    )
    if not is_exact_completion:
        raise StaleTaskRunError(
            f"task {task.id} has a canceled marker outside command target "
            f"{expected_run_id}/{expected_state_version}; current state is "
            f"{current_run_id}/{current_state_version}/"
            f"{task.status.value}/{task.control_state}"
        )
    return True


def _finalize_a2a_cancel_sync(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
    local_cancel_requested: bool,
) -> A2ATaskSnapshot:
    """Atomically persist cancellation for one exact task-state target."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
            )
            .first()
        )
        if task is None:
            raise a2a_error("task_not_found", "Task not found.", status_code=404)
        agent_config: dict[str, Any] = (
            dict(task.agent_config) if isinstance(task.agent_config, dict) else {}
        )
        if _is_completed_a2a_cancel_target(
            task,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        ):
            return A2ATaskSnapshot.from_task(task)

        current_run_id = _task_run_id(task)
        current_state_version = int(task.state_version or 0)
        same_run = current_run_id == expected_run_id
        settled_local_cancel = (
            local_cancel_requested
            and same_run
            and current_state_version == expected_state_version + 1
            and task.status == TaskStatus.FAILED
            and task.control_state == TaskControlState.FAILED.value
            and task.runner_id is None
            and task.lease_expires_at is None
        )
        direct_cancel = (
            same_run
            and current_state_version == expected_state_version
            and task.status not in _TERMINAL_STATUSES
        )
        if not settled_local_cancel and not direct_cancel:
            raise StaleTaskRunError(
                f"task {task.id} cannot finalize cancel target "
                f"{expected_run_id}/{expected_state_version}; current state is "
                f"{current_run_id}/{current_state_version}/"
                f"{task.status.value}/{task.control_state}"
            )

        agent_config["a2a_state"] = "TASK_STATE_CANCELED"
        target_state_version = current_state_version
        final_state_version = (
            current_state_version
            if settled_local_cancel
            else expected_state_version + 1
        )
        statement = (
            update(Task)
            .where(
                Task.id == task_id,
                Task.agent_id == agent_id,
                Task.source == "a2a",
                func.coalesce(Task.state_version, 0) == target_state_version,
            )
            .values(
                agent_config=agent_config,
                status=TaskStatus.FAILED,
                control_state=TaskControlState.FAILED.value,
                state_version=final_state_version,
                runner_id=None,
                lease_attempt_id=None,
                lease_expires_at=None,
                last_heartbeat_at=None,
                output=None,
                error_message="Task canceled by A2A client.",
            )
        )
        if expected_run_id is None:
            statement = statement.where(Task.run_id.is_(None))
        else:
            statement = statement.where(Task.run_id == expected_run_id)
        if settled_local_cancel:
            statement = statement.where(
                Task.status == TaskStatus.FAILED,
                Task.control_state == TaskControlState.FAILED.value,
                Task.runner_id.is_(None),
                Task.lease_expires_at.is_(None),
            )
        else:
            statement = statement.where(Task.status.notin_(_TERMINAL_STATUSES))

        updated = db.execute(
            statement.returning(Task).execution_options(synchronize_session=False)
        ).scalar_one_or_none()
        if updated is None:
            db.rollback()
            raise StaleTaskRunError(
                f"task {task_id} changed while finalizing cancel target "
                f"{expected_run_id}/{expected_state_version}"
            )
        snapshot = A2ATaskSnapshot.from_task(updated)
        db.commit()
        return snapshot


async def _cancel_task_unserialized(
    *,
    task_id: int,
    agent_id: int,
    expected_run_id: str | None,
    expected_state_version: int,
) -> Any:
    """Cancel one exact durable-command target while the caller owns its gate."""

    task = await run_db_io_cancellation_safe(
        lambda: _load_cancelable_a2a_task_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
        )
    )
    if task.agent_config.get("a2a_state") == "TASK_STATE_CANCELED":
        return a2a_json_response(task_to_a2a(task))
    if task.status in _TERMINAL_STATUSES:
        raise a2a_error(
            "task_not_cancelable",
            "Task is not in a cancelable state.",
            status_code=400,
            details={"taskId": task.id},
        )

    from .websocket import background_task_manager

    cancel_outcome = await background_task_manager.cancel_task(task.id)
    finalized = await run_db_io_cancellation_safe(
        lambda: _finalize_a2a_cancel_sync(
            task_id=task_id,
            agent_id=agent_id,
            expected_run_id=expected_run_id,
            expected_state_version=expected_state_version,
            local_cancel_requested=cancel_outcome.requested,
        )
    )
    return a2a_json_response(task_to_a2a(finalized))
