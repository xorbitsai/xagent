from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.web.models.task import ExecutionMode, Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun

from .connector_runtime import (
    bind_connector_runtime_selection_snapshot,
    prepare_connector_runtime_selection_snapshot,
)
from .task_orchestrator import TaskTurnOrchestrator, TaskTurnPayload, TurnKind
from .workforce_access import ensure_workforce_access, get_workforce_policy
from .workforce_errors import WorkforceRunError, WorkforceRunErrorCode
from .workforce_lifecycle import acquire_workforce_lifecycle_fence
from .workforce_runtime import mark_workforce_task_status, sync_workforce_run_status
from .workforce_snapshot import (
    build_workforce_snapshot,
    build_workforce_task_config,
    normalize_text,
)


@dataclass(frozen=True)
class WorkforceRunStartResult:
    workforce_run: WorkforceRun
    task: Task
    # None when an idempotency_key matched an existing run and no new turn
    # was started (created is False in that case).
    background_task: asyncio.Task[None] | None
    created: bool = True


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


def _replay_existing_run_by_idempotency_key(
    db: Session, workforce: Workforce, idempotency_key: str
) -> WorkforceRunStartResult | None:
    """Resolve an idempotency-key replay to the original run, if any.

    Raises 409 when the key was already used but its task is gone
    (``task_id`` is ``SET NULL`` on task deletion): the original result can
    no longer be replayed, and inserting a fresh run under the same key
    would only trip the unique index.
    """
    existing = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.workforce_id == int(workforce.id),
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
    return WorkforceRunStartResult(
        workforce_run=existing,
        task=cast(Task, existing.task),
        background_task=None,
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
) -> WorkforceRunStartResult:
    """Create the WorkforceRun + pending Task without starting the turn.

    Synchronous creation half of ``create_workforce_run``; callers with their
    own dispatch phase (the trigger pipeline's prepare step) use this directly
    and start the turn later through the orchestrator. ``background_task`` is
    always None in the returned result.
    """
    workforce = ensure_workforce_access(db, user, workforce, action="run")
    workforce_id = int(workforce.id)
    normalized_message = normalize_text(message, "message", required=True)
    normalized_source = _normalize_run_source(source)
    normalized_idempotency_key = _normalize_idempotency_key(idempotency_key)

    if normalized_idempotency_key is not None:
        replayed = _replay_existing_run_by_idempotency_key(
            db, workforce, normalized_idempotency_key
        )
        if replayed is not None:
            return replayed

    selected_files = _normalize_selected_file_ids(selected_file_ids)

    try:
        # The lifecycle fence takes a real row lock (a no-op UPDATE, so it
        # also locks on SQLite where SELECT FOR UPDATE is ignored) and
        # re-reads the current row; building the snapshot below re-validates
        # archived/active UNDER that lock, closing the TOCTOU window against
        # a concurrent archive whose cancellation sweep could not see this
        # run's uncommitted insert.
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
            is_preview=is_preview,
        )
        policy = get_workforce_policy()
        policy.before_workforce_run(db, user, workforce)
        manager_execution_mode = normalize_execution_mode(
            execution_mode or cast(Any, workforce.manager_agent).execution_mode
        )

        task = Task(
            user_id=int(user.id),
            title=_build_task_title(workforce, normalized_message),
            description=normalized_message,
            status=TaskStatus.PENDING,
            agent_id=int(workforce.manager_agent_id),
            agent_config=_merge_agent_config(
                build_workforce_task_config(
                    snapshot,
                    selected_file_ids=selected_files,
                ),
                extra_agent_config,
            ),
            execution_mode=manager_execution_mode,
            source=normalized_source,
            is_visible=is_visible,
        )
        selected_refs = prepare_connector_runtime_selection_snapshot(
            db=db,
            agent=cast(Any, workforce.manager_agent),
            connector_user_id=int(user.id),
        )
        bind_connector_runtime_selection_snapshot(
            task=task, selected_refs=selected_refs
        )
        db.add(task)
        db.flush()

        _bind_selected_files_to_task(db, user, task, selected_files)

        workforce_run = WorkforceRun(
            workforce_id=int(workforce.id),
            task_id=int(task.id),
            user_id=int(user.id),
            status="pending",
            is_preview=is_preview,
            idempotency_key=normalized_idempotency_key,
            snapshot=snapshot,
        )
        db.add(workforce_run)
        db.flush()

        workforce_run_id = int(workforce_run.id)
        setattr(
            task,
            "agent_config",
            _merge_agent_config(
                build_workforce_task_config(
                    snapshot,
                    selected_file_ids=selected_files,
                    workforce_run_id=workforce_run_id,
                ),
                extra_agent_config,
            ),
        )
        policy.after_workforce_run_created(db, user, workforce, workforce_run, task)
        db.commit()
    except IntegrityError:
        # Two concurrent calls with the same idempotency_key both passed the
        # pre-insert lookup; the unique index let exactly one win. Return the
        # winner's run instead of surfacing the constraint violation.
        db.rollback()
        if normalized_idempotency_key is not None:
            replayed = _replay_existing_run_by_idempotency_key(
                db, workforce, normalized_idempotency_key
            )
            if replayed is not None:
                return replayed
        raise
    except Exception:
        db.rollback()
        raise

    db.refresh(task)
    db.refresh(workforce_run)

    return WorkforceRunStartResult(
        workforce_run=workforce_run,
        task=task,
        background_task=None,
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
    record = create_workforce_run_record(
        db,
        user,
        workforce,
        message=message,
        selected_file_ids=selected_file_ids,
        execution_mode=execution_mode,
        is_preview=is_preview,
        is_visible=is_visible,
        source=source,
        idempotency_key=idempotency_key,
        extra_agent_config=extra_agent_config,
    )
    if not record.created:
        return record

    task = record.task
    workforce_run = record.workforce_run
    task_id = int(task.id)

    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=task_id,
            task_owner_user_id=int(user.id),
            # Workforce runs as the requesting user; actor == owner here.
            actor_user_id=int(user.id),
            # Intentionally equal to the normalized message: create_workforce_run_record
            # sets task.description = normalize_text(message). Read it back from the
            # record so this wrapper stays a thin dispatch layer; keep them in sync if
            # description derivation ever changes.
            payload=TaskTurnPayload(transcript_message=str(task.description or "")),
            kind=TurnKind.CREATE,
            force_fresh=False,
        )
        background_task = started.background_task
    except Exception:
        db.rollback()
        fresh_task = db.get(Task, task_id)
        if fresh_task is not None:
            mark_workforce_task_status(
                db,
                fresh_task,
                TaskStatus.FAILED,
                error_message="Workforce run failed to start",
                clear_output=True,
            )
            db.commit()
        raise

    db.refresh(task)
    if sync_workforce_run_status(db, task, task.status):
        db.commit()
        db.refresh(workforce_run)

    return WorkforceRunStartResult(
        workforce_run=workforce_run,
        task=task,
        background_task=background_task,
    )
