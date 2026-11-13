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
   content columns, and commits. The uploaded files' stored objects are
   deleted inside this transaction, so the row lock is held across that
   object-storage I/O (see ``task_deletion._delete_task_files`` for why);
   the task is quiescent, so nothing else is waiting on it.
2. After the commit, the read caches are invalidated, the ephemeral
   connector values dropped, the cached agent runtime evicted and the
   workspace directory removed -- unless a turn that started after the
   commit already owns the runtime (see :func:`_release_task_runtime`).

The workspace goes after the commit because a directory cannot be restored
by a rollback; it is found from the row, which still exists, so unlike task
deletion nothing needs capturing up front and no cleanup obligation is
recorded -- the obligation driver abandons any obligation whose task is live.
A failure there is reported per task in ``cleanup_failed`` instead, and the
caller re-runs the redaction for those: every step is idempotent, so a
re-run finishes what an earlier one left.

No ``content redacted`` column is added. Readers already treat every removed
piece as a legitimate empty state (no transcript, no steps, no files, NULL
output), ``traces_expired_at`` tells trace readers the steps were removed,
and the host records the redaction on its own rows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, cast

from sqlalchemy.orm import Session

from ..models.task import Task
from .agent_service_manager import get_agent_manager
from .connector_runtime import drop_ephemeral_runtime_values_for_task
from .hot_path_cache import invalidate_task_cache
from .task_deletion import redact_task_rows
from .task_file_lifecycle import lock_attachment_task
from .task_retention import retention_quiescent_condition
from .task_workspace_cleanup import capture_workspace_cleanup_target

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskContentRedactionReport:
    """Which of the requested tasks ended in which state.

    ``redacted`` includes tasks that were already redacted: the content is
    gone either way. ``cleanup_failed`` tasks had their rows redacted and
    committed, but removing the workspace or cached runtime raised; re-run
    the redaction for them. ``busy`` tasks were left untouched and can be
    retried once they settle; ``missing`` ones do not exist.
    """

    redacted: tuple[int, ...] = ()
    cleanup_failed: tuple[int, ...] = ()
    busy: tuple[int, ...] = ()
    missing: tuple[int, ...] = ()


@dataclass(frozen=True)
class _RedactedRuntime:
    """What the post-commit cleanup needs, read under the row lock.

    ``run_id`` and ``run_generation`` identify the runtime that existed when
    the rows were redacted, so the cleanup can tell it apart from one a later
    turn acquires after the commit.
    """

    owner_id: int
    run_id: str | None
    run_generation: int | None


def _redact_rows_if_quiescent(
    db: Session, task_id: int, *, now: datetime
) -> _RedactedRuntime | Literal["busy", "missing"]:
    """Run one task's row redaction in its own transaction.

    Returns the runtime identity of the redacted task, ``"busy"`` when the
    task is executing or owes execution (nothing is changed), or
    ``"missing"``. The quiescence check reads the row under the ``FOR
    UPDATE`` lock the redaction runs under, so a command insert (whose
    foreign key takes ``FOR KEY SHARE``) cannot slip in between on
    PostgreSQL; ``FOR UPDATE`` is a no-op on SQLite, where the check and the
    redaction are not atomic.
    """

    committed = False
    try:
        task = lock_attachment_task(db, task_id, deleting=True)
        if task is None:
            return "missing"
        quiescent = (
            db.query(Task.id)
            .filter(Task.id == task_id, retention_quiescent_condition(now=now))
            .first()
        )
        if quiescent is None:
            return "busy"
        runtime = _RedactedRuntime(
            owner_id=int(task.user_id),
            run_id=cast(str | None, task.run_id),
            run_generation=get_agent_manager().current_run_generation(task_id),
        )
        redact_task_rows(db, task, now=now)
        db.commit()
        committed = True
        return runtime
    finally:
        # Releases the row lock on every exit that did not commit.
        if not committed:
            db.rollback()


def _release_task_runtime(task_id: int, runtime: _RedactedRuntime) -> None:
    """Drop the in-memory state and the workspace of a redacted task.

    ``remove_agent`` evicts this process's cached agent -- which still holds
    the conversation in memory -- and removes the workspace through it, or
    through the captured target when nothing is cached. It is fenced on the
    run identity read under the lock: a turn that started after the commit
    (or is building its runtime now) owns the runtime and workspace, and
    ``remove_agent`` then leaves both alone rather than pull them out from
    under it. A scope that will not resolve raises rather than degrading to
    the unscoped candidates: a redaction that cannot find the workspace must
    not report success.
    """

    # In-memory state first: a workspace failure below must not leave the
    # caches serving the content the commit already removed.
    drop_ephemeral_runtime_values_for_task(task_id)
    invalidate_task_cache(task_id)
    target = capture_workspace_cleanup_target(
        task_id, runtime.owner_id, prefer_active_scope=False
    )
    get_agent_manager().remove_agent(
        task_id,
        runtime.owner_id,
        expected_run_id=runtime.run_id,
        expected_run_generation=runtime.run_generation,
        workspace_target=target,
    )


def redact_task_content(
    db: Session, task_ids: Sequence[int]
) -> TaskContentRedactionReport:
    """Delete the content of each task while keeping its row and usage.

    Owns its transactions: one per task, committed before that task's
    workspace is removed, so a busy task never holds back the ones before
    it and no row lock is held across the workspace walk. A failure in a
    task's database work propagates with the earlier tasks already redacted;
    a failure in its post-commit cleanup is logged and reported in
    ``cleanup_failed`` while the batch continues. Re-running the same call
    is safe either way and completes what is left.

    Blocking (database, object storage and filesystem I/O); async callers
    run it in a worker thread.
    """

    now = datetime.now(timezone.utc)
    redacted: list[int] = []
    cleanup_failed: list[int] = []
    busy: list[int] = []
    missing: list[int] = []
    for task_id in dict.fromkeys(task_ids):
        match _redact_rows_if_quiescent(db, task_id, now=now):
            case "busy":
                busy.append(task_id)
            case "missing":
                missing.append(task_id)
            case _RedactedRuntime() as runtime:
                try:
                    _release_task_runtime(task_id, runtime)
                except Exception:
                    logger.exception(
                        "Task %s content was redacted but its runtime or "
                        "workspace cleanup failed; re-run the redaction",
                        task_id,
                    )
                    cleanup_failed.append(task_id)
                else:
                    redacted.append(task_id)
    return TaskContentRedactionReport(
        redacted=tuple(redacted),
        cleanup_failed=tuple(cleanup_failed),
        busy=tuple(busy),
        missing=tuple(missing),
    )
