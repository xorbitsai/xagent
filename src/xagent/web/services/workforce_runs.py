from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy import or_
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
    background_task: asyncio.Task[None]


def normalize_execution_mode(value: str | None) -> str:
    normalized = (value or ExecutionMode.BALANCED.value).strip().lower()
    allowed = {mode.value for mode in ExecutionMode}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail="Invalid execution mode")
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
        raise HTTPException(status_code=404, detail="Selected file not found")

    for uploaded_file in uploaded_files:
        if uploaded_file.task_id is None:
            uploaded_file.task_id = int(task.id)


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
) -> WorkforceRunStartResult:
    workforce = ensure_workforce_access(db, user, workforce, action="run")
    normalized_message = normalize_text(message, "message", required=True)

    selected_files = _normalize_selected_file_ids(selected_file_ids)
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

    try:
        task = Task(
            user_id=int(user.id),
            title=_build_task_title(workforce, normalized_message),
            description=normalized_message,
            status=TaskStatus.PENDING,
            agent_id=int(workforce.manager_agent_id),
            agent_config=build_workforce_task_config(
                snapshot,
                selected_file_ids=selected_files,
            ),
            execution_mode=manager_execution_mode,
            source="internal",
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
            snapshot=snapshot,
        )
        db.add(workforce_run)
        db.flush()

        workforce_run_id = int(workforce_run.id)
        setattr(
            task,
            "agent_config",
            build_workforce_task_config(
                snapshot,
                selected_file_ids=selected_files,
                workforce_run_id=workforce_run_id,
            ),
        )
        policy.after_workforce_run_created(db, user, workforce, workforce_run, task)
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(task)
    db.refresh(workforce_run)
    task_id = int(task.id)

    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=task_id,
            task_owner_user_id=int(user.id),
            # Workforce runs as the requesting user; actor == owner here.
            actor_user_id=int(user.id),
            payload=TaskTurnPayload(transcript_message=normalized_message),
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
