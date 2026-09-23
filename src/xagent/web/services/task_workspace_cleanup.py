"""Removing a task's workspace directory, independent of the agent cache.

Two halves on purpose, because they have to happen at different times:

``capture_workspace_cleanup_target`` reads everything that depends on the
task **row** -- today that is the execution scope, which decides the
workspace's base directory. ``remove_task_workspace`` then does the
filesystem work, and needs no row at all.

Deletion runs the two in that order around the row delete. Doing the whole
thing afterwards, the way the cache-eviction path used to, resolves the
scope of a task that no longer exists: the snapshot loader returns ``None``
for a missing row (``execution_scope_snapshot.load_task_execution_scope_
snapshot``), the segment list comes back empty, and a workspace written
under a scoped base is no longer named by any probed candidate. The
directory then survives the task that owned it, with nothing left to find
it by.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...config import get_uploads_dir
from ...core.execution_scope import (
    ExecutionScope,
    get_execution_scope,
    resolve_execution_scope_off_turn,
)
from .workspace_binding import canonical_workspace_base

logger = logging.getLogger(__name__)


def _task_workspace_id(task_id: int) -> str:
    """The workspace directory name one web task owns under a base dir."""

    return f"web_task_{task_id}"


@dataclass(frozen=True)
class WorkspaceCleanupTarget:
    """Everything needed to delete one task's workspace, row-independent.

    Frozen and made of plain values so it can be captured before the task
    row is deleted, handed across a thread boundary, and still name the same
    directory afterwards.
    """

    task_id: int
    owner_id: Optional[int]
    base_dirs: tuple[str, ...]


def capture_workspace_cleanup_target(
    task_id: int,
    owner_id: Optional[int] = None,
    *,
    prefer_active_scope: bool = True,
) -> WorkspaceCleanupTarget:
    """Resolve the candidate base directories while the task row still exists.

    Scoped workspace first (when a resolver maps this task to a scope), then
    the user-isolated one, then the legacy uploads-root fallback. One
    spelling per candidate: the uploads root is rejected at configuration
    time unless its two readings name the same directory (see
    ``config.get_uploads_dir``), so the canonical spelling and the raw one
    reach the same place and a second candidate would only re-probe what the
    first already probed. A tree written under a root spelling that
    configuration now refuses is not reachable from any spelling of the
    current value, so recovering it is an operator migration rather than a
    candidate this function can enumerate.

    Resolution failures propagate. Callers that capture *before* deleting
    decide for themselves whether a scope they cannot resolve should block
    the deletion; callers that capture at cleanup time keep the fail-closed
    behaviour their retry bookkeeping depends on.

    ``prefer_active_scope`` is the activated turn's scope, which is this
    task's answer only when the activated turn is this task's. A caller
    capturing for many tasks at once passes ``False``: one activated scope
    cannot be the answer for all of them, and the tasks it does not belong to
    would be looked for under a base directory that was never theirs.
    """

    scope = None
    if owner_id:
        # Contextvar-first for the same reason as get_agent_for_task:
        # cleanup inside an activated turn reuses the turn's resolution.
        scope = get_execution_scope() if prefer_active_scope else None
        if scope is None:
            # Off-turn: this runs when the agent is no longer in memory, so
            # there is no turn left to fail. An authority mismatch here would
            # abandon the directory instead of deleting it, and the resolver
            # has already given an authoritative answer to delete against --
            # so the off-turn helper takes that answer and warns. Every other
            # resolution failure still propagates.
            scope = resolve_execution_scope_off_turn(task_id)

    return _target_for(task_id, owner_id, scope)


def _target_for(
    task_id: int,
    owner_id: Optional[int],
    scope: Optional[ExecutionScope],
) -> WorkspaceCleanupTarget:
    """Assemble the candidate list for an already-decided scope."""

    base_dirs: list[str] = []
    if owner_id:
        segments = scope.workspace_segments if scope is not None else ()
        for base_dir in (
            canonical_workspace_base(owner_id, segments),
            canonical_workspace_base(owner_id),
        ):
            if base_dir not in base_dirs:
                base_dirs.append(base_dir)
    legacy_root = str(get_uploads_dir())
    if legacy_root not in base_dirs:
        base_dirs.append(legacy_root)

    return WorkspaceCleanupTarget(
        task_id=int(task_id),
        owner_id=owner_id,
        base_dirs=tuple(base_dirs),
    )


def unscoped_workspace_cleanup_target(
    task_id: int,
    owner_id: int,
) -> WorkspaceCleanupTarget:
    """The candidates that remain when the scope cannot be resolved.

    A caller whose capture failed still has somewhere to look: the owner's
    un-segmented root and the legacy uploads root. That is exactly what the
    task-level path falls back to when it re-captures after the row is gone,
    so a caller that captures up front degrades to the same set instead of
    abandoning the directory entirely.

    It cannot name a workspace written under a scope segment -- that is the
    leak the failed capture predicts, and why callers still report the task as
    pending cleanup rather than treating this as a full answer.
    """

    return _target_for(task_id, owner_id, None)


def capture_workspace_cleanup_target_best_effort(
    task_id: int,
    owner_id: Optional[int] = None,
    *,
    prefer_active_scope: bool = True,
) -> Optional[WorkspaceCleanupTarget]:
    """Capture for a caller that must not fail the deletion over a capture.

    Returns ``None`` when the scope will not resolve. Both deletion paths want
    this: the task-level one still has a post-deletion fallback that may find
    an unscoped workspace, and the account-level one must not lose every other
    task's directory to one unresolvable row. Neither can afford to refuse a
    deletion the caller already asked for.

    Logged at ERROR rather than WARNING because ``None`` predicts a leak: what
    follows can only look under the unscoped candidates, so a workspace
    written under a scope segment will not be found by anything afterwards.
    """

    try:
        return capture_workspace_cleanup_target(
            task_id, owner_id, prefer_active_scope=prefer_active_scope
        )
    except Exception:
        logger.error(
            "Could not capture the workspace cleanup target for task %s; a "
            "scoped workspace directory will not be found after deletion and "
            "needs manual reconciliation",
            task_id,
            exc_info=True,
        )
        return None


def remove_task_workspace(target: WorkspaceCleanupTarget) -> None:
    """Delete the captured workspace directory. Idempotent.

    Finding nothing is a success, not a failure: the agent-owned cleanup path
    may already have removed it, and a retry of an interrupted deletion must be
    able to complete. Callers learn that a removal failed from the exception,
    which is the only outcome any of them acts on.
    """

    from ...core.workspace import TaskWorkspace

    # Imported here rather than at module scope: the allowlist policy belongs
    # to the agent manager, which imports this module. A call-time import
    # keeps the dependency pointing one way without duplicating the policy.
    from .agent_service_manager import _build_allowed_external_dirs

    workspace_id = _task_workspace_id(target.task_id)

    # Build allowed external directories (user's upload directory for
    # knowledge base files). Use only_existing=True here because cleanup runs
    # against on-disk state.
    allowed_external_dirs = _build_allowed_external_dirs(
        target.owner_id, only_existing=True
    )

    for base_dir in target.base_dirs:
        # Probed before constructing: TaskWorkspace's constructor creates the
        # workspace tree, so building one per candidate would make every probe
        # succeed and delete a directory it had just created, leaving the
        # task's real workspace untouched.
        if not (Path(base_dir) / workspace_id).exists():
            continue

        workspace = TaskWorkspace(
            workspace_id, base_dir, allowed_external_dirs=allowed_external_dirs
        )
        workspace_path = str(workspace.workspace_dir)
        logger.info(
            "Found existing workspace directory for task %s (user %s): %s",
            target.task_id,
            target.owner_id,
            workspace_path,
        )
        workspace.cleanup()
        logger.info(
            "Cleaned up workspace directory for task %s (user %s): %s",
            target.task_id,
            target.owner_id,
            workspace_path,
        )
        return

    logger.info(
        "No workspace directory found for task %s (user %s)",
        target.task_id,
        target.owner_id,
    )
