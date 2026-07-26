from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.web.models.database import (
    get_session_local,
    release_db_connection_if_clean,
)
from xagent.web.models.task import ExecutionMode, Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun

from .connector_runtime import (
    bind_connector_runtime_selection_snapshot,
    prepare_connector_runtime_selection_snapshot,
)
from .db_runtime import (
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from .task_orchestrator import (
    TaskTurnOrchestrator,
    TaskTurnPayload,
    _ClaimedTurn,
)
from .workforce_access import ensure_workforce_access, get_workforce_policy
from .workforce_errors import WorkforceRunError, WorkforceRunErrorCode
from .workforce_lifecycle import acquire_workforce_lifecycle_fence
from .workforce_runtime import sync_workforce_run_status
from .workforce_snapshot import (
    build_workforce_snapshot,
    build_workforce_task_config,
    normalize_text,
)


@dataclass(frozen=True)
class WorkforceRunRecordResult:
    workforce_run: WorkforceRun
    task: Task
    created: bool = True


@dataclass(frozen=True)
class WorkforceTaskStartSnapshot:
    """Detached task fields consumed after the worker transaction closes."""

    id: int
    agent_id: int
    title: str
    status: TaskStatus
    created_at: datetime | None
    run_id: str | None
    state_version: int
    control_state: str
    channel_id: int | None
    channel_name: str | None


@dataclass(frozen=True)
class WorkforceRunStartSnapshot:
    """Detached run fields consumed after the worker transaction closes."""

    id: int
    status: str


@dataclass(frozen=True)
class WorkforceRunStartResult:
    workforce_run: WorkforceRunStartSnapshot
    task: WorkforceTaskStartSnapshot
    background_task: asyncio.Task[None] | None
    created: bool = True


@dataclass(frozen=True)
class _PreparedWorkforceRunStart:
    workforce_run: WorkforceRunStartSnapshot
    task: WorkforceTaskStartSnapshot
    payload: TaskTurnPayload | None
    claimed_turn: _ClaimedTurn | None
    created: bool


@dataclass(frozen=True)
class _NormalizedWorkforceRunRequest:
    message: str
    selected_file_ids: tuple[str, ...]
    execution_mode: str | None
    is_preview: bool
    is_visible: bool
    source: str
    idempotency_key: str | None
    extra_agent_config: dict[str, Any] | None


def normalize_execution_mode(value: str | None) -> str:
    normalized = (value or ExecutionMode.BALANCED.value).strip().lower()
    allowed = {mode.value for mode in ExecutionMode}
    if normalized not in allowed:
        raise WorkforceRunError(
            status_code=400,
            detail="Invalid execution mode",
            code=WorkforceRunErrorCode.INVALID_EXECUTION_MODE,
        )
    return normalized


def _normalize_selected_file_ids(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        file_id = value.strip()
        if not file_id or file_id in seen:
            continue
        normalized.append(file_id)
        seen.add(file_id)
    return normalized


def _build_task_title(workforce: Workforce, message: str) -> str:
    title = f"{workforce.name}: {message}"
    return title[:50] + "..." if len(title) > 50 else title


def _merge_agent_config(
    task_config: dict[str, Any], extra_agent_config: dict[str, Any] | None
) -> dict[str, Any]:
    """Overlay caller-supplied config keys (e.g. share-channel markers) onto
    the snapshot-built task config. The built config wins on key collisions so
    callers can never clobber runtime-critical keys like ``workforce_run_id``.
    """
    if not extra_agent_config:
        return task_config
    return {**extra_agent_config, **task_config}


def _normalize_run_source(value: str | None) -> str:
    normalized = (value or "internal").strip().lower()
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid run source")
    return normalized


def _normalize_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > 128:
        raise WorkforceRunError(
            status_code=400,
            detail="Invalid idempotency key",
            code=WorkforceRunErrorCode.INVALID_IDEMPOTENCY_KEY,
        )
    return normalized


def _normalize_workforce_run_request(
    *,
    message: str,
    selected_file_ids: list[str] | None,
    execution_mode: str | None,
    is_preview: bool,
    is_visible: bool,
    source: str | None,
    idempotency_key: str | None,
    extra_agent_config: dict[str, Any] | None,
) -> _NormalizedWorkforceRunRequest:
    """Normalize caller input once before either transaction owner runs."""

    return _NormalizedWorkforceRunRequest(
        message=normalize_text(message, "message", required=True),
        selected_file_ids=tuple(_normalize_selected_file_ids(selected_file_ids)),
        execution_mode=execution_mode,
        is_preview=is_preview,
        is_visible=is_visible,
        source=_normalize_run_source(source),
        idempotency_key=_normalize_idempotency_key(idempotency_key),
        extra_agent_config=(
            dict(extra_agent_config) if extra_agent_config is not None else None
        ),
    )


def _replay_existing_run_by_idempotency_key(
    db: Session, workforce_id: int, idempotency_key: str
) -> WorkforceRunRecordResult | None:
    """Resolve an idempotency-key replay to the original run, if any.

    Raises 409 when the key was already used but its task is gone
    (``task_id`` is ``SET NULL`` on task deletion): the original result can
    no longer be replayed, and inserting a fresh run under the same key
    would only trip the unique index.
    """
    existing = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.workforce_id == workforce_id,
            WorkforceRun.idempotency_key == idempotency_key,
        )
        .first()
    )
    if existing is None:
        return None
    if existing.task is None:
        raise WorkforceRunError(
            status_code=409,
            detail="Idempotency key was already used by a run whose task no longer exists",
            code=WorkforceRunErrorCode.IDEMPOTENCY_CONFLICT,
        )
    return WorkforceRunRecordResult(
        workforce_run=existing,
        task=cast(Task, existing.task),
        created=False,
    )


def _bind_selected_files_to_task(
    db: Session,
    user: User,
    task: Task,
    selected_file_ids: list[str],
) -> None:
    if not selected_file_ids:
        return

    uploaded_files = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.file_id.in_(selected_file_ids),
            UploadedFile.user_id == int(user.id),
            UploadedFile.storage_status != "compensating",
            or_(UploadedFile.task_id.is_(None), UploadedFile.task_id == int(task.id)),
        )
        .all()
    )
    found_file_ids = {str(uploaded_file.file_id) for uploaded_file in uploaded_files}
    missing_file_ids = [
        file_id for file_id in selected_file_ids if file_id not in found_file_ids
    ]
    if missing_file_ids:
        raise WorkforceRunError(
            status_code=404,
            detail="Selected file not found",
            code=WorkforceRunErrorCode.FILE_NOT_FOUND,
        )

    for uploaded_file in uploaded_files:
        if uploaded_file.task_id is None:
            uploaded_file.task_id = int(task.id)


def _create_workforce_run_record_no_commit(
    db: Session,
    user: User,
    workforce: Workforce | None,
    *,
    request: _NormalizedWorkforceRunRequest,
) -> WorkforceRunRecordResult:
    """Stage a PENDING WorkforceRun and Task without ending the transaction."""

    workforce = ensure_workforce_access(db, user, workforce, action="run")
    workforce_id = int(workforce.id)
    if request.idempotency_key is not None:
        replayed = _replay_existing_run_by_idempotency_key(
            db,
            workforce_id,
            request.idempotency_key,
        )
        if replayed is not None:
            return replayed

    selected_files = list(request.selected_file_ids)
    # Revalidate archived/active under the lifecycle fence so archive and run
    # creation cannot pass each other between validation and insert.
    workforce = ensure_workforce_access(
        db,
        user,
        acquire_workforce_lifecycle_fence(db, workforce_id),
        action="run",
    )
    snapshot = build_workforce_snapshot(
        db,
        user,
        workforce,
        is_preview=request.is_preview,
    )
    policy = get_workforce_policy()
    policy.before_workforce_run(db, user, workforce)
    manager_execution_mode = normalize_execution_mode(
        request.execution_mode or cast(Any, workforce.manager_agent).execution_mode
    )

    task = Task(
        user_id=int(user.id),
        title=_build_task_title(workforce, request.message),
        description=request.message,
        status=TaskStatus.PENDING,
        agent_id=int(workforce.manager_agent_id),
        agent_config=_merge_agent_config(
            build_workforce_task_config(
                snapshot,
                selected_file_ids=selected_files,
            ),
            request.extra_agent_config,
        ),
        execution_mode=manager_execution_mode,
        source=request.source,
        is_visible=request.is_visible,
    )
    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db,
        agent=cast(Any, workforce.manager_agent),
        connector_user_id=int(user.id),
    )
    bind_connector_runtime_selection_snapshot(task=task, selected_refs=selected_refs)
    db.add(task)
    db.flush()

    _bind_selected_files_to_task(db, user, task, selected_files)

    workforce_run = WorkforceRun(
        workforce_id=int(workforce.id),
        task_id=int(task.id),
        user_id=int(user.id),
        status="pending",
        is_preview=request.is_preview,
        idempotency_key=request.idempotency_key,
        snapshot=snapshot,
    )
    db.add(workforce_run)
    db.flush()

    setattr(
        task,
        "agent_config",
        _merge_agent_config(
            build_workforce_task_config(
                snapshot,
                selected_file_ids=selected_files,
                workforce_run_id=int(workforce_run.id),
            ),
            request.extra_agent_config,
        ),
    )
    policy.after_workforce_run_created(db, user, workforce, workforce_run, task)
    return WorkforceRunRecordResult(workforce_run=workforce_run, task=task)


def create_workforce_run_record(
    db: Session,
    user: User,
    workforce: Workforce | None,
    *,
    message: str,
    selected_file_ids: list[str] | None = None,
    execution_mode: str | None = None,
    is_preview: bool = False,
    is_visible: bool = True,
    source: str | None = None,
    idempotency_key: str | None = None,
    extra_agent_config: dict[str, Any] | None = None,
) -> WorkforceRunRecordResult:
    """Create a committed PENDING run for a caller-owned dispatch phase."""

    request = _normalize_workforce_run_request(
        message=message,
        selected_file_ids=selected_file_ids,
        execution_mode=execution_mode,
        is_preview=is_preview,
        is_visible=is_visible,
        source=source,
        idempotency_key=idempotency_key,
        extra_agent_config=extra_agent_config,
    )
    workforce_id = int(workforce.id) if workforce is not None else None
    try:
        record = _create_workforce_run_record_no_commit(
            db,
            user,
            workforce,
            request=request,
        )
        if not record.created:
            return record
        db.commit()
    except IntegrityError:
        db.rollback()
        if workforce_id is not None and request.idempotency_key is not None:
            replayed = _replay_existing_run_by_idempotency_key(
                db,
                workforce_id,
                request.idempotency_key,
            )
            if replayed is not None:
                return replayed
        raise
    except Exception:
        db.rollback()
        raise

    db.refresh(record.task)
    db.refresh(record.workforce_run)
    return record


def _build_start_snapshots(
    db: Session,
    *,
    task_id: int,
    workforce_run_id: int,
) -> tuple[WorkforceTaskStartSnapshot, WorkforceRunStartSnapshot]:
    """Project the explicit result graph before the worker Session closes."""

    task_row = (
        db.query(
            Task.id,
            Task.agent_id,
            Task.title,
            Task.status,
            Task.created_at,
            Task.run_id,
            Task.state_version,
            Task.control_state,
            Task.channel_id,
            Task.channel_name,
        )
        .filter(Task.id == task_id)
        .one()
    )
    run_row = (
        db.query(
            WorkforceRun.id,
            WorkforceRun.status,
        )
        .filter(WorkforceRun.id == workforce_run_id)
        .one()
    )

    return (
        WorkforceTaskStartSnapshot(
            id=int(task_row.id),
            agent_id=int(task_row.agent_id),
            title=str(task_row.title),
            status=cast(TaskStatus, task_row.status),
            created_at=cast(datetime | None, task_row.created_at),
            run_id=(str(task_row.run_id) if task_row.run_id is not None else None),
            state_version=int(task_row.state_version or 0),
            control_state=str(task_row.control_state or "idle"),
            channel_id=(
                int(task_row.channel_id) if task_row.channel_id is not None else None
            ),
            channel_name=(
                str(task_row.channel_name)
                if task_row.channel_name is not None
                else None
            ),
        ),
        WorkforceRunStartSnapshot(
            id=int(run_row.id),
            status=str(run_row.status),
        ),
    )


def _create_claimed_workforce_run_isolated(
    *,
    user_id: int,
    workforce_id: int,
    request: _NormalizedWorkforceRunRequest,
) -> _PreparedWorkforceRunStart:
    """Create and claim an interactive run in one worker-owned transaction."""

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        workforce = db.get(Workforce, workforce_id)
        record = _create_workforce_run_record_no_commit(
            db,
            user,
            workforce,
            request=request,
        )
        if not record.created:
            task_snapshot, run_snapshot = _build_start_snapshots(
                db,
                task_id=int(record.task.id),
                workforce_run_id=int(record.workforce_run.id),
            )
            return _PreparedWorkforceRunStart(
                workforce_run=run_snapshot,
                task=task_snapshot,
                payload=None,
                claimed_turn=None,
                created=False,
            )

        payload = TaskTurnPayload(transcript_message=request.message)
        claimed_turn = TaskTurnOrchestrator.claim_created_turn_no_commit(
            db,
            task_id=int(record.task.id),
            task_owner_user_id=user_id,
            payload=payload,
        )
        sync_workforce_run_status(db, record.task, TaskStatus.RUNNING)
        db.flush()
        task_snapshot, run_snapshot = _build_start_snapshots(
            db,
            task_id=int(record.task.id),
            workforce_run_id=int(record.workforce_run.id),
        )
        db.commit()
    except IntegrityError:
        db.rollback()
        if request.idempotency_key is not None:
            replayed = _replay_existing_run_by_idempotency_key(
                db,
                workforce_id,
                request.idempotency_key,
            )
            if replayed is not None:
                task_snapshot, run_snapshot = _build_start_snapshots(
                    db,
                    task_id=int(replayed.task.id),
                    workforce_run_id=int(replayed.workforce_run.id),
                )
                return _PreparedWorkforceRunStart(
                    workforce_run=run_snapshot,
                    task=task_snapshot,
                    payload=None,
                    claimed_turn=None,
                    created=False,
                )
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return _PreparedWorkforceRunStart(
        workforce_run=run_snapshot,
        task=task_snapshot,
        payload=payload,
        claimed_turn=claimed_turn,
        created=True,
    )


async def _start_normalized_workforce_run(
    *,
    user_id: int,
    workforce_id: int,
    request: _NormalizedWorkforceRunRequest,
) -> WorkforceRunStartResult:
    """Create and schedule a normalized run without a caller-owned Session."""

    async def _create_and_schedule() -> WorkforceRunStartResult:
        prepared = await run_db_io_cancellation_safe(
            lambda: _create_claimed_workforce_run_isolated(
                user_id=user_id,
                workforce_id=workforce_id,
                request=request,
            )
        )
        if not prepared.created:
            return WorkforceRunStartResult(
                workforce_run=prepared.workforce_run,
                task=prepared.task,
                background_task=None,
                created=False,
            )
        if prepared.payload is None or prepared.claimed_turn is None:
            raise RuntimeError("created workforce run did not stage its initial turn")

        started = await TaskTurnOrchestrator.schedule_claimed_create_turn(
            task_id=prepared.task.id,
            task_owner_user_id=user_id,
            actor_user_id=user_id,
            payload=prepared.payload,
            claimed=prepared.claimed_turn,
        )
        return WorkforceRunStartResult(
            workforce_run=prepared.workforce_run,
            task=prepared.task,
            background_task=started.background_task,
        )

    # Create the owner task before the first await. Once RUNNING is committed,
    # scheduling (or its compensated failure) must settle before cancellation
    # reaches the request.
    start_task = asyncio.create_task(_create_and_schedule())
    return await drain_async_task_cancellation_safe(start_task)


async def create_workforce_run_by_id(
    *,
    user_id: int,
    workforce_id: int,
    message: str,
    selected_file_ids: list[str] | None = None,
    execution_mode: str | None = None,
    is_preview: bool = False,
    is_visible: bool = True,
    source: str | None = None,
    idempotency_key: str | None = None,
    extra_agent_config: dict[str, Any] | None = None,
) -> WorkforceRunStartResult:
    """Create a run from detached identities.

    API adapters that already authenticated an owner should use this entry
    point so no request Session or attached ORM row survives into the async
    turn-start boundary.
    """

    request = _normalize_workforce_run_request(
        message=message,
        selected_file_ids=selected_file_ids,
        execution_mode=execution_mode,
        is_preview=is_preview,
        is_visible=is_visible,
        source=source,
        idempotency_key=idempotency_key,
        extra_agent_config=extra_agent_config,
    )
    return await _start_normalized_workforce_run(
        user_id=int(user_id),
        workforce_id=int(workforce_id),
        request=request,
    )


async def create_workforce_run(
    db: Session,
    user: User,
    workforce: Workforce | None,
    *,
    message: str,
    selected_file_ids: list[str] | None = None,
    execution_mode: str | None = None,
    is_preview: bool = False,
    is_visible: bool = True,
    source: str | None = None,
    idempotency_key: str | None = None,
    extra_agent_config: dict[str, Any] | None = None,
) -> WorkforceRunStartResult:
    """Compatibility entry point for callers that still own ORM identities."""

    if workforce is None:
        raise HTTPException(status_code=404, detail="Workforce not found")
    user_id = int(user.id)
    workforce_id = int(workforce.id)
    request = _normalize_workforce_run_request(
        message=message,
        selected_file_ids=selected_file_ids,
        execution_mode=execution_mode,
        is_preview=is_preview,
        is_visible=is_visible,
        source=source,
        idempotency_key=idempotency_key,
        extra_agent_config=extra_agent_config,
    )
    if not release_db_connection_if_clean(db):
        raise RuntimeError("request DB transaction is not clean at turn boundary")
    return await _start_normalized_workforce_run(
        user_id=user_id,
        workforce_id=workforce_id,
        request=request,
    )
