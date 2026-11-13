"""Content redaction for tasks whose row must survive (conversation deletion).

A host product lets its users delete their conversation history while the
usage it bills from stays put. That usage is computed from ``tasks`` rows --
the token counters and the rows themselves -- so deleting the task, the way
``task_delete`` does, is not an option. Redaction removes everything a task
*said or produced* and keeps what it *cost*; :func:`redact_task_rows` defines
the exact split.

Per task, in order:

1. One transaction locks the row, refuses a task that is not quiescent (the
   retention predicate: terminal status, no live lease, no command owed
   execution), deletes the content rows and uploaded files, blanks the
   content columns, and commits.
2. After the commit, the cached agent runtime is evicted and the workspace
   directory removed, the ephemeral connector values dropped and the read
   caches invalidated.

The workspace goes after the commit because a directory cannot be restored
by a rollback; it is found from the row, which still exists, so unlike task
deletion nothing needs capturing up front and no cleanup obligation is
recorded -- the obligation driver abandons any obligation whose task is live.
A failure there propagates instead, and the caller re-runs the redaction:
every step is idempotent, so a re-run finishes what an earlier one left.

No ``content redacted`` column is added. Readers already treat every removed
piece as a legitimate empty state (no transcript, no steps, no files, NULL
output), ``traces_expired_at`` tells trace readers the steps were removed,
and the host records the redaction on its own rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy.orm import Session

from ..models.task import Task
from .agent_service_manager import get_agent_manager
from .connector_runtime import drop_ephemeral_runtime_values_for_task
from .hot_path_cache import invalidate_task_cache
from .task_deletion import redact_task_rows
from .task_file_lifecycle import lock_attachment_task
from .task_retention import retention_quiescent_condition
from .task_workspace_cleanup import capture_workspace_cleanup_target


@dataclass(frozen=True)
class TaskContentRedactionReport:
    """Which of the requested tasks ended in which state.

    ``redacted`` includes tasks that were already redacted: the content is
    gone either way. ``busy`` tasks were left untouched and can be retried
    once they settle; ``missing`` ones do not exist.
    """

    redacted: tuple[int, ...] = ()
    busy: tuple[int, ...] = ()
    missing: tuple[int, ...] = ()


def _redact_rows_if_quiescent(
    db: Session, task_id: int, *, now: datetime
) -> int | Literal["busy", "missing"]:
    """Run one task's row redaction in its own transaction.

    Returns the owner id of the redacted task, ``"busy"`` when the task is
    executing or owes execution (nothing is changed), or ``"missing"``. The
    quiescence check reads the row under the same ``FOR UPDATE`` lock the
    redaction takes, so a command insert (whose foreign key takes ``FOR KEY
    SHARE``) cannot slip in between on PostgreSQL; ``FOR UPDATE`` is a no-op
    on SQLite, where the check and the redaction are not atomic.
    """

    committed = False
    try:
        task = lock_attachment_task(db, task_id, deleting=True)
        if task is None:
            return "missing"
        owner_id = int(task.user_id)
        quiescent = (
            db.query(Task.id)
            .filter(Task.id == task_id, retention_quiescent_condition(now=now))
            .first()
        )
        if quiescent is None:
            return "busy"
        redact_task_rows(db, task_id=task_id, now=now)
        db.commit()
        committed = True
        return owner_id
    finally:
        # Releases the row lock on every exit that did not commit.
        if not committed:
            db.rollback()


def _release_task_runtime(task_id: int, owner_id: int) -> None:
    """Drop the in-memory state and the workspace of a redacted task.

    ``remove_agent`` evicts this process's cached agent -- which still holds
    the conversation in memory -- and removes the workspace through it, or
    through the captured target when nothing is cached. A scope that will
    not resolve raises rather than degrading to the unscoped candidates: a
    redaction that cannot find the workspace must not report success.
    """

    # In-memory state first: a workspace failure below must not leave the
    # caches serving the content the commit already removed.
    drop_ephemeral_runtime_values_for_task(task_id)
    invalidate_task_cache(task_id)
    target = capture_workspace_cleanup_target(
        task_id, owner_id, prefer_active_scope=False
    )
    get_agent_manager().remove_agent(task_id, owner_id, workspace_target=target)


def redact_task_content(
    db: Session, task_ids: Sequence[int]
) -> TaskContentRedactionReport:
    """Delete the content of each task while keeping its row and usage.

    Owns its transactions: one per task, committed before that task's
    workspace is removed, so a busy or failing task never holds back the
    ones before it and no row lock is held across a filesystem walk. An
    exception propagates with the earlier tasks already redacted; re-running
    the same call is safe and completes the rest.

    Blocking (database, object storage and filesystem I/O); async callers
    run it in a worker thread.
    """

    now = datetime.now(timezone.utc)
    redacted: list[int] = []
    busy: list[int] = []
    missing: list[int] = []
    for task_id in dict.fromkeys(task_ids):
        match _redact_rows_if_quiescent(db, task_id, now=now):
            case "busy":
                busy.append(task_id)
            case "missing":
                missing.append(task_id)
            case int(owner_id):
                _release_task_runtime(task_id, owner_id)
                redacted.append(task_id)
    return TaskContentRedactionReport(
        redacted=tuple(redacted), busy=tuple(busy), missing=tuple(missing)
    )
