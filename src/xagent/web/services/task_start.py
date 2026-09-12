"""Runner-owned acceptance and scheduling for HTTP and legacy task starts.

Each entry preserves its transport's existing transaction and cancellation
semantics. Protocol adapters pass detached identities and inputs; claims,
connector runtime values and background task handles stay inside the runner.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from ...config import get_shared_task_execution_enabled
from ..models.agent import Agent
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.workforce import WorkforceRun
from .a2a_protocol import A2ATaskSnapshot, new_context_id, task_context_id
from .a2a_task_read import load_a2a_task_snapshot
from .connector_runtime import (
    bind_create_connector_runtime_plan,
    persist_create_connector_runtime_context,
    pop_ephemeral_runtime_values,
    prepare_append_connector_runtime,
    prepare_create_connector_runtime,
    store_ephemeral_runtime_values,
)
from .db_runtime import drain_async_task_cancellation_safe, run_db_io_cancellation_safe
from .file_turn import (
    append_uploaded_files_context,
    build_uploaded_files_context,
    normalize_attachments_for_persistence,
    resolve_turn_file_infos,
)
from .managed_file_ref import (
    DurableObjectIntegrityError,
    DurableStorageOperationError,
    log_durable_storage_fault,
)
from .task_execution_controller import task_execution_controller
from .task_orchestrator import (
    TaskTurnError,
    TaskTurnFileBindingError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
    TurnStarted,
    _PreparedTurn,
    _retire_turn_session_best_effort,
    commit_claimed_turn_or_reconcile,
    timezone_schedule_context,
)
from .task_resume import resume_a2a_task

logger = logging.getLogger(__name__)
_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}


class SdkCreateTurnRejected(TaskTurnError):
    """A CREATE scheduling rejection, distinct from preparation failures."""


class TaskStartRejected(Exception):
    """A domain rejection for the API to translate to its protocol."""

    def __init__(
        self, reason: str, *, task_id: int | None = None, context_id: str | None = None
    ):
        super().__init__(reason)
        self.reason = reason
        self.task_id = task_id
        self.context_id = context_id


@dataclass(frozen=True)
class SdkTaskScope:
    """Authenticated SDK owner; exactly one owner ID is set."""

    agent_id: int | None
    workforce_id: int | None


@dataclass(frozen=True)
class TaskStartResult:
    """Public start state without the orchestration task handle.

    accepted_at is the creation timestamp for CREATE, and the committed turn
    update timestamp for APPEND, matching the existing protocol responses.
    """

    task_id: int
    agent_id: int
    status: TaskStatus
    accepted_at: datetime | None
    run_id: str
    state_version: int
    control_state: str


def _resolve_turn_files(
    *,
    file_ids: list[str],
    owner_user_id: int,
    db: Session,
    task_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Resolve every ``file_id`` up front (all-or-nothing) WITHOUT binding.

    Called before the task is committed (create) or the turn is claimed
    (append), so a bad/unowned/already-bound id fails with 400 before any
    task row is created or mutated -- no orphan task, no binding stuck to a
    turn that later 409s. Actual binding happens via :func:`bind_turn_files`
    only after the turn is committed to running.
    """
    if not file_ids:
        return []
    try:
        file_infos, missing = resolve_turn_file_infos(
            file_ids=file_ids,
            owner_user_id=owner_user_id,
            db=db,
            task_id=task_id,
        )
    except DurableObjectIntegrityError:
        # Precedes the parent arm: permanent corruption, already logged at
        # ERROR with both checksums where it is raised. The envelope is left
        # as-is deliberately -- reclassifying it for the SDK is a compatibility
        # change, not a logging one -- but it must not also emit a
        # transient-outage warning.
        raise
    except DurableStorageOperationError as exc:
        # Transient storage fault, not a client error -- 503 so SDK can retry.
        # The SDK adapter returns a 503 envelope without a traceback, so
        # this log line is the only record of the provider fault (#1467).
        # ``task_id`` is None on the create path -- one of this function's two
        # callers, not an edge case -- so the owner and the requested ids carry
        # the identification there.
        log_durable_storage_fault(
            logger,
            "turn attachment resolution",
            exc,
            task_id=task_id,
            owner_user_id=owner_user_id,
            file_ids=",".join(file_ids),
        )
        raise
    if missing:
        raise TaskTurnFileBindingError(missing)
    return file_infos


def _turn_payload(content: str, file_infos: list[dict[str, Any]]) -> TaskTurnPayload:
    """Build a :class:`TaskTurnPayload`, file-enriching the execution channel.

    Consolidates the payload construction shared by create and append so the
    transcript-vs-execution split can't drift between the two entry points.
    """
    if not file_infos:
        return TaskTurnPayload(transcript_message=content)
    context = build_uploaded_files_context(file_infos)
    return TaskTurnPayload(
        transcript_message=content,
        execution_message=append_uploaded_files_context(content, context),
        attachments=normalize_attachments_for_persistence(file_infos) or None,
        file_ids=tuple(str(info["file_id"]) for info in file_infos),
    )


def _store_connector_runtime_values_or_fail(
    *,
    task_id: int,
    turn_id: str,
    values_by_ref: dict,
    db: Session,
) -> None:
    try:
        if get_shared_task_execution_enabled():
            from .task_runtime_secrets import stage_runtime_values

            stage_runtime_values(
                db, task_id=task_id, turn_id=turn_id, values_by_ref=values_by_ref
            )
        else:
            store_ephemeral_runtime_values(turn_id, values_by_ref)
    except Exception as exc:
        pop_ephemeral_runtime_values(turn_id)
        logger.warning(
            "Connector runtime setup failed for task %s turn %s",
            task_id,
            turn_id,
        )
        raise TaskStartRejected("connector_runtime_setup_failed") from exc


@dataclass(frozen=True)
class _PreparedCreateTaskStart:
    """Detached result of the atomic create-and-claim transaction."""

    task_id: int
    agent_id: int
    task_owner_user_id: int
    created_at: datetime
    payload: TaskTurnPayload
    claimed_turn: _PreparedTurn


@dataclass(frozen=True)
class _PreparedAppendTurn:
    """Detached result of the atomic append claim transaction."""

    task_id: int
    agent_id: int
    task_owner_user_id: int
    payload: TaskTurnPayload
    claimed_turn: _PreparedTurn


def _prepare_created_task_isolated(
    *,
    agent_id: int,
    task_owner_user_id: int,
    message: str,
    file_ids: tuple[str, ...],
    connector_runtime_context: tuple[dict[str, Any], ...],
    timezone: str | None = None,
) -> _PreparedCreateTaskStart:
    """Create, claim, and commit the first turn in one DB transaction."""

    SessionLocal = get_session_local()
    db = SessionLocal()
    payload: TaskTurnPayload | None = None
    owned_task_id = 0
    try:
        agent = db.get(Agent, agent_id)
        if agent is None:
            raise TaskStartRejected("agent_not_found")

        file_infos = _resolve_turn_files(
            file_ids=list(file_ids),
            owner_user_id=task_owner_user_id,
            db=db,
            task_id=None,
        )
        task = Task(
            user_id=task_owner_user_id,
            title=message[:50] or "SDK task",
            description=message,
            status=TaskStatus.PENDING,
            agent_id=agent_id,
            input=message,
            source="sdk",
            is_visible=False,
        )
        runtime_plan = prepare_create_connector_runtime(
            db=db,
            agent=agent,
            task_source="sdk",
            connector_user_id=task_owner_user_id,
            payload_items=connector_runtime_context,
        )
        bind_create_connector_runtime_plan(task=task, plan=runtime_plan)

        db.add(task)
        db.flush()
        task_id = int(task.id)
        owned_task_id = task_id
        persist_create_connector_runtime_context(
            db=db,
            task_id=task_id,
            plan=runtime_plan,
        )
        payload = _turn_payload(message, file_infos)
        if get_shared_task_execution_enabled():
            _store_connector_runtime_values_or_fail(
                task_id=task_id,
                turn_id=payload.turn_id,
                db=db,
                values_by_ref=runtime_plan.ephemeral_by_ref,
            )

        claimed_turn = TaskTurnOrchestrator.claim_created_turn_no_commit(
            db,
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            context=timezone_schedule_context(timezone),
        )
        if not get_shared_task_execution_enabled():
            _store_connector_runtime_values_or_fail(
                task_id=task_id,
                turn_id=payload.turn_id,
                db=db,
                values_by_ref=runtime_plan.ephemeral_by_ref,
            )

        created_at = db.query(Task.created_at).filter(Task.id == task_id).scalar()
        if created_at is None:
            raise RuntimeError("created task has no creation timestamp")
        commit_claimed_turn_or_reconcile(
            db,
            task_id=task_id,
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            claimed=claimed_turn,
        )
        return _PreparedCreateTaskStart(
            task_id=task_id,
            agent_id=agent_id,
            task_owner_user_id=task_owner_user_id,
            created_at=created_at,
            payload=payload,
            claimed_turn=claimed_turn,
        )
    except Exception:
        db.rollback()
        if payload is not None and not get_shared_task_execution_enabled():
            pop_ephemeral_runtime_values(payload.turn_id)
        raise
    finally:
        _retire_turn_session_best_effort(db, task_id=owned_task_id)


def validate_sdk_owner_scope(
    scope: SdkTaskScope,
    *,
    request_agent_id: int | None,
    request_workforce_id: int | None,
) -> None:
    """Validate a request body's owner-scoping fields against the key.

    Shared by every ``/v1/chat/tasks/{task_id}/*`` write that reuses the
    ``AppendMessageRequest``-shaped owner fields (append, reply): an
    agent-bound key must pass a matching ``agent_id`` and no
    ``workforce_id``; a workforce-bound key must not pass ``agent_id``
    and, if it passes ``workforce_id``, it must match the bound
    workforce. Task ownership itself must already be resolved (e.g. via
    :func:`resolve_sdk_task`) before calling this, so an unrelated
    task stays an opaque ``task_not_found`` even when the body also
    names the wrong owner.
    """
    if scope.agent_id is not None:
        if request_agent_id is None:
            raise TaskStartRejected("agent_id_required")
        if request_agent_id != scope.agent_id:
            raise TaskStartRejected("agent_not_found")
        if request_workforce_id is not None:
            raise TaskStartRejected("workforce_not_found")
    else:
        assert scope.workforce_id is not None
        if request_agent_id is not None:
            raise TaskStartRejected("agent_not_found")
        if (
            request_workforce_id is not None
            and request_workforce_id != scope.workforce_id
        ):
            raise TaskStartRejected("workforce_not_found")


def _prepare_append_turn_isolated(
    *,
    task_id: int,
    scope: SdkTaskScope,
    request_agent_id: int | None,
    request_workforce_id: int | None,
    message: str,
    file_ids: tuple[str, ...],
    connector_runtime_context: tuple[dict[str, Any], ...],
) -> _PreparedAppendTurn:
    """Resolve append authorization and runtime inputs into detached values."""

    SessionLocal = get_session_local()
    db = SessionLocal()
    payload: TaskTurnPayload | None = None
    try:
        task = resolve_sdk_task(task_id, scope, db)
        task_owner_user_id = int(task.user_id)

        # Task ownership is resolved first so an unrelated task remains an
        # opaque task_not_found even when the body also names the wrong owner.
        validate_sdk_owner_scope(
            scope,
            request_agent_id=request_agent_id,
            request_workforce_id=request_workforce_id,
        )

        runtime_agent = db.get(Agent, int(task.agent_id))
        if runtime_agent is None:
            raise TaskStartRejected("runtime_agent_missing")
        runtime_plan = prepare_append_connector_runtime(
            db=db,
            agent=runtime_agent,
            task=task,
            connector_user_id=task_owner_user_id,
            payload_items=connector_runtime_context,
        )

        file_infos = _resolve_turn_files(
            file_ids=list(file_ids),
            owner_user_id=task_owner_user_id,
            db=db,
            task_id=int(task.id),
        )
        payload = _turn_payload(message, file_infos)
        if get_shared_task_execution_enabled():
            _store_connector_runtime_values_or_fail(
                task_id=int(task.id),
                turn_id=payload.turn_id,
                db=db,
                values_by_ref=runtime_plan.ephemeral_by_ref,
            )

        claimed_turn = TaskTurnOrchestrator.claim_append_turn_no_commit(
            db,
            task_id=int(task.id),
            task_owner_user_id=task_owner_user_id,
            payload=payload,
        )
        if not get_shared_task_execution_enabled():
            _store_connector_runtime_values_or_fail(
                task_id=int(task.id),
                turn_id=payload.turn_id,
                db=db,
                values_by_ref=runtime_plan.ephemeral_by_ref,
            )

        prepared = _PreparedAppendTurn(
            task_id=int(task.id),
            agent_id=int(task.agent_id),
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            claimed_turn=claimed_turn,
        )
        commit_claimed_turn_or_reconcile(
            db,
            task_id=int(task.id),
            task_owner_user_id=task_owner_user_id,
            payload=payload,
            claimed=claimed_turn,
        )
        return prepared
    except Exception:
        db.rollback()
        if payload is not None and not get_shared_task_execution_enabled():
            pop_ephemeral_runtime_values(payload.turn_id)
        raise
    finally:
        _retire_turn_session_best_effort(db, task_id=task_id)


def resolve_sdk_task(task_id: int, scope: SdkTaskScope, db: Session) -> Task:
    """Resolve a task_id against the calling key's ownership AND
    SDK-source scope.

    Returns the :class:`Task` row when the task:

      1. Exists.
      2. Belongs to the key's owner -- directly (``Task.agent_id`` for
         an agent-bound key) or through a workforce run
         (``WorkforceRun.task_id`` for a workforce-bound key; the
         binding is 1:1 unique).
      3. Was created by the SDK (``source == "sdk"``).

    Any other case — missing row, row belongs to a different owner,
    or row was created by the Web UI / internal paths — raises
    :class:`TaskStartRejected` with ``task_not_found``. Missing and
    inaccessible tasks share one rejection so adapters can preserve
    non-disclosing error responses.

    The ``source == "sdk"`` filter exists because an SDK API key
    binds to an owner, not to a particular product surface. Without
    it, an SDK client could read or append to any task the Web UI
    created under the same owner (the user's own historical Web UI
    chats or workforce runs, for example). Whether that's intentional
    is a product decision, but the safe default for a public SDK is to
    scope lookups to tasks the SDK itself created — ``POST
    /v1/chat/tasks`` and ``POST /v1/workforces/{id}/runs`` both write
    ``source="sdk"`` so this is well-defined.

    Args:
        task_id: Path parameter from the route.
        scope: Detached owner IDs resolved by API-key authentication.
        db: SQLAlchemy session.

    Raises:
        TaskStartRejected: task missing, not owned by
            the calling key, or not created by the SDK.
    """
    query = db.query(Task).filter(
        Task.id == task_id,
        Task.source == "sdk",
    )
    if scope.agent_id is not None:
        query = query.filter(Task.agent_id == scope.agent_id)
    else:
        assert scope.workforce_id is not None
        query = query.join(WorkforceRun, WorkforceRun.task_id == Task.id).filter(
            WorkforceRun.workforce_id == scope.workforce_id
        )
    task = query.first()
    if task is None:
        raise TaskStartRejected("task_not_found")
    return task


async def create_sdk_task(
    *,
    agent_id: int,
    task_owner_user_id: int,
    actor_user_id: int,
    message: str,
    file_ids: tuple[str, ...],
    connector_runtime_context: tuple[dict[str, Any], ...],
    timezone: str | None,
) -> TaskStartResult:
    async def _prepare_and_schedule() -> tuple[_PreparedCreateTaskStart, TurnStarted]:
        prepared = await run_db_io_cancellation_safe(
            lambda: _prepare_created_task_isolated(
                agent_id=agent_id,
                task_owner_user_id=task_owner_user_id,
                message=message,
                file_ids=file_ids,
                connector_runtime_context=connector_runtime_context,
                timezone=timezone,
            )
        )
        try:
            started = await TaskTurnOrchestrator.schedule_claimed_create_turn(
                task_id=prepared.task_id,
                task_owner_user_id=prepared.task_owner_user_id,
                actor_user_id=actor_user_id,
                payload=prepared.payload,
                claimed=prepared.claimed_turn,
                context=timezone_schedule_context(timezone),
            )
        except TaskTurnError as exc:
            pop_ephemeral_runtime_values(prepared.payload.turn_id)
            raise SdkCreateTurnRejected(exc.reason) from exc
        except BaseException:
            pop_ephemeral_runtime_values(prepared.payload.turn_id)
            raise
        return prepared, started

    # The owned child starts before the first await. If the caller is cancelled
    # after RUNNING commits, scheduling (or its exact-lease compensation) still
    # settles before cancellation propagates.
    start_task = asyncio.create_task(_prepare_and_schedule())
    prepared, started = await drain_async_task_cancellation_safe(start_task)

    return TaskStartResult(
        task_id=prepared.task_id,
        agent_id=prepared.agent_id,
        status=started.status,
        accepted_at=prepared.created_at,
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
    )


async def append_sdk_turn(
    *,
    task_id: int,
    scope: SdkTaskScope,
    actor_user_id: int,
    request_agent_id: int | None,
    request_workforce_id: int | None,
    message: str,
    file_ids: tuple[str, ...],
    connector_runtime_context: tuple[dict[str, Any], ...],
) -> TaskStartResult:
    async def _prepare_and_begin() -> tuple[_PreparedAppendTurn, TurnStarted]:
        # A runner may still be finishing local cleanup after its terminal
        # status is visible in the database. Reject that tail window before
        # the domain-owned transaction stages any append mutation. The atomic
        # status predicate inside the transaction remains authoritative across
        # workers and concurrent requests.
        TaskTurnOrchestrator.ensure_no_background_turn(task_id)
        prepared = await run_db_io_cancellation_safe(
            lambda: _prepare_append_turn_isolated(
                task_id=task_id,
                scope=scope,
                request_agent_id=request_agent_id,
                request_workforce_id=request_workforce_id,
                message=message,
                file_ids=file_ids,
                connector_runtime_context=connector_runtime_context,
            )
        )
        try:
            started = await TaskTurnOrchestrator.schedule_claimed_turn(
                task_id=prepared.task_id,
                task_owner_user_id=prepared.task_owner_user_id,
                actor_user_id=actor_user_id,
                payload=prepared.payload,
                claimed=prepared.claimed_turn,
                kind=TurnKind.APPEND,
            )
        except BaseException:
            pop_ephemeral_runtime_values(prepared.payload.turn_id)
            raise
        return prepared, started

    start_task = asyncio.create_task(_prepare_and_begin())
    prepared, started = await drain_async_task_cancellation_safe(start_task)
    return TaskStartResult(
        task_id=prepared.task_id,
        agent_id=prepared.agent_id,
        status=started.status,
        accepted_at=started.updated_at,
        run_id=started.run_id,
        state_version=started.state_version,
        control_state=started.control_state,
    )


@dataclass(frozen=True)
class _A2ATurnPreparation:
    """Detached result of the worker-owned A2A preparation transaction."""

    task: A2ATaskSnapshot
    created_task: bool
    kind: TurnKind
    payload: TaskTurnPayload
    claimed_turn: _PreparedTurn | None


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
                raise TaskTurnNotFoundError(task_id)
            task = existing_task
            if task.status in _TERMINAL_STATUSES:
                raise TaskStartRejected("a2a_terminal", task_id=int(task.id))
            stored_context_id = task_context_id(task)
            if context_id is not None and context_id != stored_context_id:
                raise TaskStartRejected(
                    "a2a_context_mismatch", task_id=int(task.id), context_id=context_id
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


async def start_a2a_turn(
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
            await resume_a2a_task(
                agent_id=agent_id,
                task_owner_user_id=task_owner_user_id,
                task_id=prepared_task.id,
                previous_run_id=prepared_task.run_id,
                resumable_status=prepared_task.status,
                text=text,
                message_id=message_id,
            )
            fresh = await load_a2a_task_snapshot(
                agent_id,
                prepared_task.id,
            )
            if fresh is None:
                raise TaskTurnNotFoundError(prepared_task.id)
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
            raise TaskTurnNotFoundError(prepared_task.id) from exc
        except TaskTurnError as exc:
            raise TaskStartRejected("a2a_busy", task_id=prepared_task.id) from exc

        fresh = await load_a2a_task_snapshot(
            agent_id,
            prepared_task.id,
        )
        if fresh is None:
            raise TaskTurnNotFoundError(prepared_task.id)
        return fresh

    if task_id is not None:
        async with task_execution_controller.command(task_id):
            return await start_unserialized()
    start_task = asyncio.create_task(start_unserialized())
    return await drain_async_task_cancellation_safe(start_task)


async def execute_existing_task(
    *,
    task_id: int,
    task_owner_user_id: int,
    task_source: str | None,
    task_description: str,
    context: dict[str, Any],
    actor_user_id: int,
) -> None:
    """Run the legacy command to completion without returning a task handle."""
    from .task_execution_host import enqueues_task_turns

    if enqueues_task_turns():
        from .task_completion import wait_for_task_run
        from .task_existing_command import enqueue_existing_execution

        run_id = await run_db_io_cancellation_safe(
            lambda: enqueue_existing_execution(
                task_id=task_id,
                task_owner_user_id=task_owner_user_id,
                task_description=task_description,
                context=context,
                actor_user_id=actor_user_id,
            )
        )
        await wait_for_task_run(task_id, run_id)
        return
    background_task = await TaskTurnOrchestrator.schedule_existing_task_execution(
        task_id=task_id,
        task_owner_user_id=task_owner_user_id,
        task_source=task_source,
        payload=TaskTurnPayload(
            transcript_message=task_description,
            execution_message=task_description,
        ),
        context=context,
        actor_user_id=actor_user_id,
    )
    # Legacy WebSocket execute_task must not return until execution finishes.
    # Keep that ordering even though scheduling owns the background handle.
    await background_task
