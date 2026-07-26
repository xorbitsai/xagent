"""Persisted ExecutionScope snapshots for internally created tasks.

Workforce runs create fresh Task rows whose ids the embedding application's
scope resolver cannot map. The creating context's scope is persisted into
the task's ``agent_config`` JSON (``EXECUTION_SCOPE_AGENT_CONFIG_KEY``, no
schema migration) at creation; this module's loader reads it back, and the
core per-task resolution (:func:`xagent.core.execution_scope.
resolve_execution_scope`) prefers the snapshot over the resolver — so a
sub-task of a scoped parent executes fully scoped even after a process
restart.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...core.execution_scope import (
    ExecutionScope,
    execution_scope_from_agent_config,
    set_execution_scope_snapshot_loader,
)

logger = logging.getLogger(__name__)


def load_task_execution_scope_snapshot(task_id: str) -> Optional[ExecutionScope]:
    """Load the persisted scope snapshot for a task, or None.

    Non-integer task ids (nested in-process executions, builder chats)
    have no Task row and resolve to None. A malformed persisted snapshot
    raises (via ExecutionScope validation) rather than degrading to
    unscoped — degrading would silently merge namespaces.
    """
    try:
        task_key = int(task_id)
    except (TypeError, ValueError):
        return None

    from ..models.database import get_session_local
    from ..models.task import Task

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        row = db.query(Task.agent_config).filter(Task.id == task_key).first()

    if row is None:
        return None
    return execution_scope_from_agent_config(row[0])


def register_execution_scope_snapshot_loader() -> None:
    """Install the Task-table-backed snapshot loader (app startup)."""
    set_execution_scope_snapshot_loader(load_task_execution_scope_snapshot)
    logger.info("Execution scope snapshot loader registered")
