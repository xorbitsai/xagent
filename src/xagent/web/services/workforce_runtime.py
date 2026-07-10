from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from xagent.web.models.task import Task, TaskStatus

from ..models.workforce import WorkforceRun
from .task_lease_service import (
    TaskLease,
    release_current_runner_task_lease,
    release_task_lease,
)
from .workforce_snapshot import build_agent_tool_overrides

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkforceTaskRuntime:
    workforce_run_id: int
    workforce_id: int
    snapshot: dict[str, Any]
    allowed_agent_ids: list[int]
    agent_tool_overrides: dict[int, dict[str, Any]]
    worker_tool_names: set[str]
    manager_system_prompt: str | None
    manager_agent_id: int | None
    enable_global_agent_tools: bool = False
    allow_cross_user_agent_ids: bool = True

    @property
    def agent_call_stack(self) -> list[int]:
        return [self.manager_agent_id] if self.manager_agent_id is not None else []


def extract_workforce_run_id(task: Any) -> int | None:
    agent_config = getattr(task, "agent_config", None)
    if not isinstance(agent_config, dict):
        return None
    workforce_run_id = agent_config.get("workforce_run_id")
    return workforce_run_id if isinstance(workforce_run_id, int) else None


def is_workforce_task(task: Any) -> bool:
    agent_config = getattr(task, "agent_config", None)
    return isinstance(agent_config, dict) and isinstance(
        agent_config.get("workforce_run_id"), int
    )


def resolve_workforce_task_runtime(
    db: Session,
    task: Any,
) -> WorkforceTaskRuntime | None:
    workforce_run_id = extract_workforce_run_id(task)
    if workforce_run_id is None:
        return None

    task_id = getattr(task, "id", None)
    user_id = getattr(task, "user_id", None)
    if task_id is None or user_id is None:
        return None

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(user_id),
        )
        .first()
    )
    if run is None or not isinstance(run.snapshot, dict):
        return None

    snapshot = run.snapshot
    workforce_data = snapshot.get("workforce")
    manager_data = snapshot.get("manager")
    workers_data = snapshot.get("workers")
    if not isinstance(workforce_data, dict) or not isinstance(manager_data, dict):
        return None
    if not isinstance(workers_data, list):
        return None

    allowed_agent_ids: list[int] = []
    for worker in workers_data:
        if not isinstance(worker, dict) or worker.get("enabled") is False:
            continue
        agent_id = worker.get("agent_id")
        if isinstance(agent_id, int):
            allowed_agent_ids.append(agent_id)

    if not allowed_agent_ids:
        return None

    allowed_agent_id_set = set(allowed_agent_ids)
    overrides = {
        agent_id: override
        for agent_id, override in build_agent_tool_overrides(
            snapshot, workforce_run_id=workforce_run_id
        ).items()
        if agent_id in allowed_agent_id_set
    }
    worker_tool_names = {
        str(override["tool_name"])
        for override in overrides.values()
        if isinstance(override.get("tool_name"), str)
    }
    workforce_id = workforce_data.get("id")
    manager_agent_id = manager_data.get("agent_id")
    manager_system_prompt = manager_data.get("runtime_prompt")

    return WorkforceTaskRuntime(
        workforce_run_id=workforce_run_id,
        workforce_id=int(workforce_id) if isinstance(workforce_id, int) else 0,
        snapshot=snapshot,
        allowed_agent_ids=allowed_agent_ids,
        agent_tool_overrides=overrides,
        worker_tool_names=worker_tool_names,
        manager_system_prompt=manager_system_prompt
        if isinstance(manager_system_prompt, str)
        else None,
        manager_agent_id=manager_agent_id
        if isinstance(manager_agent_id, int)
        else None,
    )


def _map_task_status(status: Any) -> str | None:
    if isinstance(status, str):
        try:
            status = TaskStatus(status)
        except ValueError:
            return None
    if status == TaskStatus.PENDING:
        return "pending"
    if status == TaskStatus.RUNNING:
        return "running"
    if status in {TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER}:
        return "paused"
    if status == TaskStatus.COMPLETED:
        return "completed"
    if status == TaskStatus.FAILED:
        return "failed"
    return None


def sync_workforce_run_status(
    db: Session, task: Any, status: Any | None = None
) -> bool:
    workforce_run_id = extract_workforce_run_id(task)
    mapped_status = _map_task_status(status if status is not None else task.status)
    if workforce_run_id is None or mapped_status is None:
        return False

    task_id = getattr(task, "id", None)
    user_id = getattr(task, "user_id", None)
    if task_id is None or user_id is None:
        return False

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(user_id),
        )
        .first()
    )
    if run is None:
        return False

    changed = False
    if run.status != mapped_status:
        setattr(run, "status", mapped_status)
        changed = True

    if mapped_status in {"completed", "failed", "cancelled"}:
        if run.completed_at is None:
            setattr(run, "completed_at", datetime.now(timezone.utc))
            changed = True
    elif run.completed_at is not None:
        setattr(run, "completed_at", None)
        changed = True

    return changed


def mark_workforce_task_status(
    db: Session,
    task: Task,
    status: TaskStatus,
    *,
    error_message: str | None = None,
    clear_output: bool = False,
) -> bool:
    """Update the task lifecycle source of truth and project it to WorkforceRun."""
    from .task_execution_controller import (
        apply_task_control_transition,
        control_state_for_status,
    )

    changed = False
    expected_control_state = control_state_for_status(status)
    if task.status != status or task.control_state != expected_control_state.value:
        apply_task_control_transition(
            task,
            expected_control_state,
            status=status,
        )
        changed = True
    if error_message is not None and task.error_message != error_message:
        setattr(task, "error_message", error_message)
        changed = True
    if clear_output and task.output is not None:
        setattr(task, "output", None)
        changed = True

    return sync_workforce_run_status(db, task, status) or changed


def _sync_workforce_run_status_for_task_id(
    db: Session,
    task_id: int,
    status: TaskStatus,
) -> bool:
    task = db.query(Task).filter(Task.id == int(task_id)).first()
    if task is None:
        return False
    changed = sync_workforce_run_status(db, task, status)
    if changed:
        db.commit()
    return changed


def release_task_lease_with_workforce_sync(
    db: Session,
    lease: TaskLease | None,
    *,
    status: TaskStatus,
) -> bool:
    released = release_task_lease(db, lease, status=status)
    if not released or lease is None:
        return released
    try:
        _sync_workforce_run_status_for_task_id(db, lease.task_id, status)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "Failed to sync workforce run status after task lease release",
            exc_info=True,
        )
    return released


def release_current_runner_task_lease_with_workforce_sync(
    db: Session,
    task_id: int,
    *,
    status: TaskStatus,
    runner_id: str | None = None,
    expected_run_id: str | None = None,
) -> bool:
    released = release_current_runner_task_lease(
        db,
        task_id,
        status=status,
        runner_id=runner_id,
        expected_run_id=expected_run_id,
    )
    if not released:
        return released
    try:
        _sync_workforce_run_status_for_task_id(db, task_id, status)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "Failed to sync workforce run status after current runner lease release",
            exc_info=True,
        )
    return released
