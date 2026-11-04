"""Single-task deletion shared by the Web UI and SDK surfaces.

The row purge, the external cleanup it owes, and the inline release are ordered
once here so the v1 delete cannot drift from ``DELETE /api/chat/task/{id}``.
Callers keep their own authorization and post-commit runtime work (WebSocket
detach / background cancellation); everything that touches the deleted task's
rows and owned resources lives in this module.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ...core.task_runtime import TaskRuntimeContext
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from .connector_runtime import drop_ephemeral_runtime_values_for_task
from .hot_path_cache import invalidate_task_cache
from .task_cleanup_obligations import (
    CleanupObligation,
    CleanupObligationStatus,
    RecordedCleanupObligation,
    captured_workspace_obligation,
    describe_cleanup_failure,
    extension_obligation,
    record_cleanup_obligations_no_commit,
    settle_cleanup_attempts_sync,
)
from .task_deletion import purge_task_rows
from .task_execution_controller import TaskControlState
from .task_retention import (
    retention_commands_clear_condition,
    retention_lease_clear_condition,
)
from .task_runtime import (
    TaskRuntimeExtensionError,
    delete_task_extensions,
    registered_task_extensions,
    task_extension_bindings_from_agent_config,
)
from .task_workspace_cleanup import (
    WorkspaceCleanupTarget,
    capture_workspace_cleanup_target_best_effort,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskDeleteSnapshot:
    """Detached inputs the deletion needs after the task row is read."""

    task_id: int
    task_title: str
    task_user_id: int
    task_source: str | None
    bound_extensions: tuple[str, ...]


@dataclass(frozen=True)
class CommittedTaskDeletion:
    """A committed row deletion plus the cleanup obligations it recorded."""

    snapshot: TaskDeleteSnapshot
    workspace_target: WorkspaceCleanupTarget | None
    recorded_workspace: RecordedCleanupObligation
    extensions_owed: tuple[CleanupObligation, ...]


@dataclass(frozen=True)
class TaskDeleteOutcome:
    """The external cleanup outcome both deletion endpoints report."""

    workspace_cleanup_pending: bool
    external_cleanup_pending: bool


class TaskExtensionCleanupRejected(RuntimeError):
    """A bound runtime extension could not release its task-owned state."""

    def __init__(self, cause: TaskRuntimeExtensionError) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.unreleased = cause.unreleased_extensions


class TaskDeleteBusy(RuntimeError):
    """The task became active after the pre-check but before the delete."""


_DELETE_BLOCKED_STATUSES = (TaskStatus.PENDING, TaskStatus.RUNNING)
_DELETE_BLOCKED_CONTROL_STATES = (
    TaskControlState.RUNNING.value,
    TaskControlState.PAUSE_REQUESTED.value,
    TaskControlState.RESUME_REQUESTED.value,
)


def task_is_active_for_delete(db: Session, task: Task) -> bool:
    """Whether deleting this task could race a live executor.

    Besides ``status`` and ``control_state``, a task is active while it holds
    an unexpired execution lease or an accepted command that has not finished:
    under shared execution a queued turn stages a command and leaves the
    status terminal, so the status columns alone would miss it.
    """

    if (
        task.status in _DELETE_BLOCKED_STATUSES
        or str(task.control_state or "") in _DELETE_BLOCKED_CONTROL_STATES
    ):
        return True
    quiet = (
        db.query(Task.id)
        .filter(
            Task.id == task.id,
            retention_lease_clear_condition(now=datetime.now(timezone.utc)),
            retention_commands_clear_condition(),
        )
        .first()
    )
    return quiet is None


def task_delete_snapshot(task: Task) -> TaskDeleteSnapshot:
    return TaskDeleteSnapshot(
        task_id=int(task.id),
        task_title=str(task.title),
        task_user_id=int(task.user_id),
        task_source=str(task.source) if task.source is not None else None,
        bound_extensions=task_extension_bindings_from_agent_config(task.agent_config),
    )


def _task_runtime_context(snapshot: TaskDeleteSnapshot) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        task_id=snapshot.task_id,
        user_id=snapshot.task_user_id,
        source=snapshot.task_source,
        session_factory=get_session_local(),
    )


def load_task_delete_snapshot_sync(
    *,
    task_id: int,
    requester_user_id: int,
    is_admin: bool,
) -> TaskDeleteSnapshot | None:
    """Load the JWT deletion snapshot inside an operation-local session."""

    session_factory = get_session_local()
    delete_db = session_factory()
    try:
        query = delete_db.query(Task).filter(Task.id == task_id)
        if not is_admin:
            query = query.filter(Task.user_id == requester_user_id)
        task = query.first()
        if task is None:
            return None
        return task_delete_snapshot(task)
    finally:
        delete_db.close()


def _extension_obligations(
    snapshot: TaskDeleteSnapshot,
    unreleased: tuple[str, ...],
    *,
    force: bool,
) -> tuple[CleanupObligation, ...]:
    registered = set(registered_task_extensions())
    owed: list[CleanupObligation] = []
    for name in unreleased:
        if name in registered:
            # Only force delete can reach here for a registered provider; it
            # accepted the leak, so record it for reconciliation and never retry.
            status = CleanupObligationStatus.ABANDONED
            reason: str | None = (
                "force delete accepted this runtime extension's state as "
                "leaked; it was not released"
            )
        else:
            status = CleanupObligationStatus.PENDING
            reason = None
        owed.append(
            extension_obligation(
                task_id=snapshot.task_id,
                user_id=snapshot.task_user_id,
                source=snapshot.task_source,
                extension=name,
                status=status,
                reason=reason,
            )
        )
    return tuple(owed)


def _delete_task_rows_sync(
    *,
    task_id: int,
    workspace: CleanupObligation,
    extensions: tuple[CleanupObligation, ...],
    refuse_if_active: bool,
) -> RecordedCleanupObligation | None:
    """Delete one task in an operation-local session.

    The obligations commit with the rows, so they are on record exactly when
    the rows are gone. Returns ``None`` when the task no longer exists.
    """

    session_factory = get_session_local()
    delete_db = session_factory()
    try:
        if refuse_if_active:
            # Re-read under the deletion transaction's row lock: the caller's
            # pre-check can be stale by the time cleanup work has finished.
            # ``FOR UPDATE`` is a no-op on SQLite, so there this re-check and
            # the purge are not atomic; the guarantee holds on PostgreSQL only.
            locked = (
                delete_db.query(Task)
                .filter(Task.id == task_id)
                .with_for_update()
                .first()
            )
            if locked is not None and task_is_active_for_delete(delete_db, locked):
                delete_db.rollback()
                raise TaskDeleteBusy(task_id)
        deleted = purge_task_rows(
            delete_db, task_id=task_id, detached_reason="task_deleted"
        )
        if not deleted:
            delete_db.rollback()
            return None
        [recorded] = record_cleanup_obligations_no_commit(
            delete_db, [workspace], inline_attempt=True
        )
        record_cleanup_obligations_no_commit(delete_db, extensions)
        delete_db.commit()
        return recorded
    except Exception:
        delete_db.rollback()
        raise
    finally:
        delete_db.close()


def _refuse_if_task_active_sync(task_id: int) -> None:
    session_factory = get_session_local()
    db = session_factory()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task is not None and task_is_active_for_delete(db, task):
            raise TaskDeleteBusy(task_id)
    finally:
        db.close()


async def commit_task_deletion(
    snapshot: TaskDeleteSnapshot,
    *,
    force: bool = False,
    refuse_if_active: bool = False,
) -> CommittedTaskDeletion | None:
    """Release provider state, then delete the rows and record what is owed.

    ``refuse_if_active`` adds the v1 busy guard: an unlocked check before the
    irreversible extension release, then a locked re-check inside the deletion
    transaction (PostgreSQL only; see ``_delete_task_rows_sync``). The Web UI
    delete leaves it off because it never had that guard.
    """

    if refuse_if_active:
        # Narrows the window but does not close it: a task that turns active
        # after this check and before the locked re-check still gets a 409
        # with its provider state already released. Releasing after the
        # commit instead would delete the task even when a provider fails,
        # which the Web UI delete (and the retry-safe contract) rules out.
        await asyncio.to_thread(_refuse_if_task_active_sync, snapshot.task_id)
    try:
        unreleased = await delete_task_extensions(
            _task_runtime_context(snapshot),
            bound_extensions=snapshot.bound_extensions,
            force=force,
        )
    except TaskRuntimeExtensionError as exc:
        logger.error(
            "Runtime extension cleanup failed; preserving task %s for retry",
            snapshot.task_id,
            exc_info=True,
        )
        raise TaskExtensionCleanupRejected(exc) from exc
    if unreleased:
        logger.error(
            "Deleting task %s with unreleased runtime extension state for %s",
            snapshot.task_id,
            ", ".join(unreleased),
        )

    workspace_target = await asyncio.to_thread(
        capture_workspace_cleanup_target_best_effort,
        snapshot.task_id,
        snapshot.task_user_id,
    )
    workspace_owed = captured_workspace_obligation(
        snapshot.task_id, snapshot.task_user_id, workspace_target
    )
    extensions_owed = _extension_obligations(snapshot, unreleased, force=force)

    recorded_workspace = await asyncio.to_thread(
        _delete_task_rows_sync,
        task_id=snapshot.task_id,
        workspace=workspace_owed,
        extensions=extensions_owed,
        refuse_if_active=refuse_if_active,
    )
    if recorded_workspace is None:
        return None
    drop_ephemeral_runtime_values_for_task(snapshot.task_id)
    invalidate_task_cache(snapshot.task_id)
    return CommittedTaskDeletion(
        snapshot=snapshot,
        workspace_target=workspace_target,
        recorded_workspace=recorded_workspace,
        extensions_owed=extensions_owed,
    )


async def finish_task_deletion(
    committed: CommittedTaskDeletion,
    *,
    request: Any = None,
) -> TaskDeleteOutcome:
    """Release the runtime/workspace and settle the recorded obligation."""

    from .agent_service_manager import get_agent_manager

    snapshot = committed.snapshot
    workspace_error: str | None = None
    try:
        manager = get_agent_manager(request)
        await asyncio.to_thread(
            manager.remove_agent,
            snapshot.task_id,
            snapshot.task_user_id,
            workspace_target=committed.workspace_target,
        )
    except Exception as exc:
        workspace_error = describe_cleanup_failure(exc)
        logger.error(
            "Task %s rows were deleted but its workspace cleanup failed; the "
            "retry driver will re-attempt it",
            snapshot.task_id,
            exc_info=True,
        )
    [workspace_status] = await asyncio.to_thread(
        settle_cleanup_attempts_sync,
        get_session_local(),
        [(committed.recorded_workspace, workspace_error)],
    )
    workspace_pending = workspace_status is not None
    if workspace_pending:
        logger.error(
            "Task %s was deleted with its workspace left on disk",
            snapshot.task_id,
        )
    return TaskDeleteOutcome(
        workspace_cleanup_pending=workspace_pending,
        external_cleanup_pending=workspace_pending or bool(committed.extensions_owed),
    )
