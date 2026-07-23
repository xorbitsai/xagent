from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session, selectinload

from xagent.web.models.task import Task, TaskStatus

from ..models.workforce import Workforce, WorkforceAgent, WorkforceRun
from .task_lease_service import (
    TaskLease,
    release_current_runner_task_lease,
    release_task_lease,
)
from .workforce_snapshot import (
    WORKFORCE_CONFIG_FINGERPRINT_VERSION,
    build_agent_tool_overrides,
    compute_live_workforce_config_fingerprint,
)

logger = logging.getLogger(__name__)

# Run statuses that still hold (or can reclaim) execution resources.
ACTIVE_WORKFORCE_RUN_STATUSES = frozenset({"pending", "running", "paused"})


class WorkforceTurnRejectedError(Exception):
    """A new turn on a workforce task must not start.

    Raised by :func:`ensure_workforce_turn_allowed` when the owning workforce
    was archived or its live config no longer matches the run's pinned
    fingerprint. The turn orchestrator maps this onto its transport-facing
    ``TaskTurnError`` with the same reason string.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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


def _load_workforce_for_fingerprint(db: Session, workforce_id: int) -> Workforce | None:
    return (
        db.query(Workforce)
        .options(
            selectinload(Workforce.manager_agent),
            selectinload(Workforce.workers).selectinload(WorkforceAgent.agent),
        )
        .filter(Workforce.id == int(workforce_id))
        .first()
    )


def ensure_workforce_turn_allowed(
    db: Session,
    *,
    task_id: int,
    task_owner_user_id: int,
    agent_config: dict[str, Any] | None = None,
) -> None:
    """Gate a new turn on a workforce task against the live workforce state.

    Called at the shared turn-entry point for every turn kind. CREATE turns
    are already validated upstream by ``validate_workforce_for_run``, but an
    archive can commit between ``create_workforce_run``'s commit and the
    claim — its cancellation sweep marks the run cancelled yet cannot stop a
    turn that never started — so this check is the last line before
    execution. Rejects with :class:`WorkforceTurnRejectedError` when:

    - ``workforce_archived``: the owning workforce was archived (or its row
      is gone). Archive terminates external exposure; long-lived sessions
      must not keep executing past it.
    - ``workforce_config_changed``: the live config no longer matches the
      fingerprint pinned in the run snapshot. The snapshot only freezes
      prompt-building data while worker execution re-reads live Agent rows,
      so a drifted config silently changes behavior mid-session; reject and
      require a fresh session instead.
    - ``workforce_run_not_found``: ``agent_config`` names a run that does
      not exist (or belongs to another task/user). The runtime resolver
      would silently degrade such a task to a bare manager agent with zero
      delegation ability; fail loudly instead of executing it.

    No-op for non-workforce tasks and for runs whose snapshot predates the
    fingerprint (backwards compatibility). Preview runs skip the fingerprint
    check: the builder edits config while previewing by design.

    ``agent_config`` may be passed by callers that already read the task row
    (the turn orchestrator reads it in its post-claim snapshot SELECT) to
    save a redundant round-trip; when omitted it is read here, scoped to the
    owner.
    """
    if agent_config is None:
        row = (
            db.query(Task.agent_config)
            .filter(Task.id == int(task_id), Task.user_id == int(task_owner_user_id))
            .first()
        )
        if row is None or not isinstance(row[0], dict):
            return
        agent_config = row[0]
    workforce_run_id = agent_config.get("workforce_run_id")
    if not isinstance(workforce_run_id, int):
        return

    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task_id),
            WorkforceRun.user_id == int(task_owner_user_id),
        )
        .first()
    )
    if run is None:
        raise WorkforceTurnRejectedError("workforce_run_not_found")

    workforce = _load_workforce_for_fingerprint(db, int(run.workforce_id))
    if workforce is None or workforce.status == "archived":
        raise WorkforceTurnRejectedError("workforce_archived")

    if bool(run.is_preview):
        return
    snapshot: dict[str, Any] = run.snapshot if isinstance(run.snapshot, dict) else {}
    pinned = snapshot.get("config_fingerprint")
    if not isinstance(pinned, str) or not pinned:
        return
    # Only compare fingerprints pinned by the CURRENT algorithm: a version
    # bump (algorithm change) would otherwise make every pre-existing run's
    # pinned value mismatch the freshly computed live one and spuriously
    # reject those sessions on deploy — the exact failure the fingerprint
    # exists to prevent. Runs pinned under an older version are exempt.
    if snapshot.get("config_fingerprint_version") != (
        WORKFORCE_CONFIG_FINGERPRINT_VERSION
    ):
        return
    live = compute_live_workforce_config_fingerprint(workforce)
    if live != pinned:
        raise WorkforceTurnRejectedError("workforce_config_changed")


@dataclass(frozen=True)
class WorkforceRunPauseTarget:
    """A previously RUNNING task that still needs a PAUSE after archive."""

    run_id: int
    task_id: int


def cancel_active_workforce_runs(
    db: Session,
    workforce_id: int,
) -> list[WorkforceRunPauseTarget]:
    """Mark every in-flight run of a workforce terminal ``cancelled``.

    Archiving only flips ``Workforce.status``; without this, in-flight runs
    keep executing because turn resolution never re-checks live workforce
    state. ``cancelled`` is non-overwritable in ``sync_workforce_run_status``,
    so a PAUSE landing later cannot flip the run back to ``paused``. New
    turns are rejected separately by ``ensure_workforce_turn_allowed``.

    Deliberately does NOT commit and does NOT touch the command transport:
    the caller commits the archive flip and these cancellations in one
    atomic transaction, then dispatches PAUSE to the returned targets via
    :func:`pause_workforce_tasks_after_archive`. (The durable enqueue commits
    internally, so calling it on this session mid-loop would leak a partial
    archive state.)
    """
    runs = (
        db.query(WorkforceRun)
        .options(selectinload(WorkforceRun.task))
        .filter(
            WorkforceRun.workforce_id == int(workforce_id),
            WorkforceRun.status.in_(ACTIVE_WORKFORCE_RUN_STATUSES),
        )
        .all()
    )

    pause_targets: list[WorkforceRunPauseTarget] = []
    for run in runs:
        task = run.task
        if task is not None and task.status == TaskStatus.RUNNING:
            pause_targets.append(
                WorkforceRunPauseTarget(run_id=int(run.id), task_id=int(task.id))
            )
        setattr(run, "status", "cancelled")
        if run.completed_at is None:
            setattr(run, "completed_at", datetime.now(timezone.utc))
    return pause_targets


async def pause_workforce_tasks_after_archive(
    pause_targets: list[WorkforceRunPauseTarget],
    *,
    workforce_id: int,
    actor_user_id: int | None,
) -> None:
    """Best-effort PAUSE dispatch for tasks left running by an archive.

    Runs AFTER the caller committed the archive/cancel transaction, on its
    own short-lived sessions, so the durable enqueue's internal commit can
    never leak a partial archive state. A failed pause is logged and skipped:
    the run is already terminal and the turn-entry guard blocks new turns,
    so the orphaned execution can only run its current turn to completion.
    """
    if not pause_targets:
        return

    from ..models.database import get_session_local
    from .task_command_transport import (
        TaskCommandKind,
        dispatch_task_command_promptly,
        enqueue_task_command,
    )

    SessionLocal = get_session_local()
    for target in pause_targets:
        try:
            with SessionLocal() as command_db:
                enqueued = enqueue_task_command(
                    command_db,
                    task_id=target.task_id,
                    actor_user_id=actor_user_id,
                    command_id=f"workforce-archive-{target.run_id}",
                    kind=TaskCommandKind.PAUSE,
                    payload={},
                )
            from ..api.websocket import execute_durable_task_command

            await dispatch_task_command_promptly(
                execute_durable_task_command,
                command_db_id=enqueued.command_id,
            )
        except Exception:
            logger.warning(
                "Failed to pause running task %s while archiving workforce %s",
                target.task_id,
                workforce_id,
                exc_info=True,
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

    # "cancelled" is terminal and only ever set explicitly (workforce
    # archive). A late task-status projection (e.g. the PAUSE issued during
    # archive landing as "paused") must not resurrect the run.
    if run.status == "cancelled":
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
