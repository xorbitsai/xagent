"""Content redaction for tasks whose row must survive (conversation deletion).

A host product lets its users delete their conversation history while the
usage it bills from stays put. That usage is computed from ``tasks`` rows --
the token counters and the rows themselves -- so deleting the task, the way
``task_delete`` does, is not an option. Redaction removes everything a task
*said or produced* and keeps what it *cost*; :func:`redact_task_rows` defines
the exact split.

Only settled tasks are redacted
-------------------------------
A task is redacted only when it is quiescent under the retention predicate:
status COMPLETED or FAILED, no live execution lease, no command owed
execution. Every other task is reported ``busy`` and left untouched. That
includes tasks that will **not** settle on their own: a PAUSED task, one
WAITING_FOR_USER, and a PENDING one with no command queued. Retrying those
loops forever; the host must cancel or otherwise settle them first, then
redact. Task deletion accepts those states because it removes the row and
with it every reader; redaction keeps the row, and a parked run's resume
state is exactly the content it removes.

Per task, in order
------------------
1. One transaction locks the row, refuses a task that is not quiescent,
   deletes the content rows and uploaded files, blanks the content columns,
   and commits. The uploaded files' stored objects are deleted inside this
   transaction, so the row lock is held across that object-storage I/O (see
   ``task_deletion.delete_task_files`` for why); the task is quiescent, so
   nothing else is waiting on it. A failure here rolls back that task's rows
   and reports it ``failed``; the stored files it had already deleted stay
   deleted, so a ``failed`` task is partially cleaned and must be re-run.
2. After the commit, the read caches are invalidated, the ephemeral
   connector values dropped, the cached agent runtime evicted and the
   workspace directory removed -- unless a turn that started after the
   commit already owns the runtime (see :func:`_release_task_runtime`).

Eviction is process-local. ``remove_agent`` drops the agent this process
has cached; a host running several agent worker processes must evict the
task's cached agent on every other one itself, or a follow-up turn there
can reuse the pre-redaction conversation the agent holds in memory and
write it back. Task deletion has the same limit.

The workspace goes after the commit because a directory cannot be restored
by a rollback; it is found from the row, which still exists, so unlike task
deletion nothing needs capturing up front and no cleanup obligation is
recorded -- the obligation driver abandons any obligation whose task is live.
A failed or skipped cleanup is reported per task in ``cleanup_failed``
instead, and the caller re-runs the redaction for those: every step is
idempotent, so a re-run finishes what an earlier one left.

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
from typing import Callable, Literal, TypeVar, cast

from sqlalchemy.orm import Session

from ..models.task import Task
from .agent_service_manager import get_agent_manager
from .connector_runtime import drop_ephemeral_runtime_values_for_task
from .hot_path_cache import invalidate_task_cache
from .task_deletion import delete_task_files, redact_task_rows
from .task_file_lifecycle import lock_attachment_task
from .task_retention import retention_quiescent_condition
from .task_workspace_cleanup import (
    WorkspaceCleanupTarget,
    capture_workspace_cleanup_target,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass(frozen=True)
class TaskContentRedactionReport:
    """Which of the requested tasks ended in which state.

    ``redacted`` includes tasks that were already redacted: the content is
    gone either way. ``cleanup_failed`` tasks had their rows redacted and
    committed, but removing the workspace or cached runtime raised, or was
    skipped because a later turn already owns the runtime; re-run the
    redaction for them once they settle; any other process's cached agent
    is never evicted here (see the module docstring). ``failed`` tasks raised
    in their row redaction and were rolled back, but the stored files deleted
    before the failure stay deleted: they are partially cleaned, their file
    rows may name missing bytes, and they must be re-run.
    ``busy`` tasks are not settled and were left untouched -- see the module
    docstring for the states the host must cancel first; ``missing`` ones do
    not exist.
    """

    redacted: tuple[int, ...] = ()
    cleanup_failed: tuple[int, ...] = ()
    failed: tuple[int, ...] = ()
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


def _run_on_quiescent_task(
    db: Session,
    task_id: int,
    action: Callable[[Task], _T],
    *,
    now: datetime,
) -> _T | Literal["busy", "missing"]:
    """Run ``action`` on one locked, settled task in its own transaction.

    Returns what ``action`` returned once its work is committed, ``"busy"``
    when the task is not settled (nothing is changed; PAUSED,
    WAITING_FOR_USER and a command-less PENDING task stay busy until the
    host settles them), or ``"missing"``. The quiescence check reads the row
    under the ``FOR UPDATE`` lock ``action`` runs under, so a command insert
    (whose foreign key takes ``FOR KEY SHARE``) cannot slip in between on
    PostgreSQL; ``FOR UPDATE`` is a no-op on SQLite, where the check and the
    action are not atomic. Raises whatever ``action`` raises, after rolling
    back.
    """

    committed = False
    try:
        task = lock_attachment_task(db, task_id, deleting=True)
        if task is None:
            return "missing"
        # The caller's session may already hold this row from before the
        # lock; the redaction rewrites ``agent_config`` from it and the
        # eviction fence reads ``run_id``, so both must be the locked values.
        db.refresh(task)
        quiescent = (
            db.query(Task.id)
            .filter(Task.id == task_id, retention_quiescent_condition(now=now))
            .first()
        )
        if quiescent is None:
            return "busy"
        result = action(task)
        db.commit()
        committed = True
        return result
    finally:
        # Releases the row lock on every exit that did not commit.
        if not committed:
            db.rollback()


def _redact_rows(db: Session, task: Task, *, now: datetime) -> _RedactedRuntime:
    """Redact one locked task's rows; return what its cleanup needs."""
    runtime = _RedactedRuntime(
        owner_id=int(task.user_id),
        run_id=cast(str | None, task.run_id),
        run_generation=get_agent_manager().current_run_generation(int(task.id)),
    )
    redact_task_rows(db, task, now=now)
    return runtime


def _release_task_runtime(task_id: int, runtime: _RedactedRuntime) -> bool:
    """Drop the in-memory state and the workspace of a redacted task.

    ``remove_agent`` evicts this process's cached agent -- which still holds
    the conversation in memory -- and removes the workspace through it, or
    through the captured target when nothing is cached. It is fenced on the
    run identity read under the lock: a turn that started after the commit
    (or is building its runtime now) owns the runtime and workspace, and
    ``remove_agent`` then leaves both alone rather than pull them out from
    under it. That skip returns ``False``: the workspace still holds
    pre-redaction files and nothing else removes them, so the caller must
    re-run the redaction once that turn settles.

    A scope that will not resolve raises rather than degrading to the
    unscoped candidates: a redaction that cannot find the workspace must not
    report success. The cached agent is evicted regardless, since it holds
    the conversation in memory.

    Process-local: another worker process's cached agent is the host's to
    evict (see the module docstring).
    """

    # In-memory state first: a workspace failure below must not leave the
    # caches serving the content the commit already removed.
    drop_ephemeral_runtime_values_for_task(task_id)
    invalidate_task_cache(task_id)
    manager = get_agent_manager()

    def evict(target: WorkspaceCleanupTarget | None) -> bool:
        return manager.remove_agent(
            task_id,
            runtime.owner_id,
            expected_run_id=runtime.run_id,
            expected_run_generation=runtime.run_generation,
            workspace_target=target,
        )

    try:
        target = capture_workspace_cleanup_target(
            task_id, runtime.owner_id, prefer_active_scope=False
        )
    except Exception:
        # The cached agent still holds the conversation: evict it anyway. A
        # cached agent removes its own workspace without a target; with none
        # cached the eviction re-captures and fails the same way. Either way
        # the task is reported ``cleanup_failed``.
        evict(None)
        raise
    return evict(target)


def redact_task_content(
    db: Session, task_ids: Sequence[int]
) -> TaskContentRedactionReport:
    """Delete the content of each task while keeping its row and usage.

    Owns its transactions: one per task, committed before that task's
    workspace is removed, so a busy or failing task never holds back the
    others and no row lock is held across the workspace walk. A task whose
    row redaction raises is rolled back and reported ``failed``; one whose
    post-commit cleanup raises or is skipped is reported ``cleanup_failed``;
    both are logged and the batch continues. Re-running the same call is
    safe and completes what is left.

    Only settled tasks are redacted: PAUSED, WAITING_FOR_USER and
    command-less PENDING tasks come back ``busy`` on every call until the
    host cancels or settles them (see the module docstring).

    ``db`` must be clean -- no pending changes and no open transaction the
    caller still needs: each task commits or rolls back this session, so
    caller work would be committed by the first redacted task or discarded
    by the first busy or missing one.

    Eviction of the cached agent is process-local; a host with several agent
    worker processes evicts it on the others itself.

    Blocking (database, object storage and filesystem I/O); async callers
    run it in a worker thread.
    """

    now = datetime.now(timezone.utc)
    redacted: list[int] = []
    cleanup_failed: list[int] = []
    failed: list[int] = []
    busy: list[int] = []
    missing: list[int] = []
    for task_id in dict.fromkeys(task_ids):
        try:
            outcome = _run_on_quiescent_task(
                db, task_id, lambda task: _redact_rows(db, task, now=now), now=now
            )
        except Exception:
            logger.exception(
                "Task %s content redaction failed and was rolled back", task_id
            )
            failed.append(task_id)
            continue
        match outcome:
            case "busy":
                busy.append(task_id)
            case "missing":
                missing.append(task_id)
            case _RedactedRuntime() as runtime:
                try:
                    released = _release_task_runtime(task_id, runtime)
                except Exception:
                    logger.exception(
                        "Task %s content was redacted but its runtime or "
                        "workspace cleanup failed; re-run the redaction",
                        task_id,
                    )
                    released = False
                if released:
                    redacted.append(task_id)
                else:
                    cleanup_failed.append(task_id)
    return TaskContentRedactionReport(
        redacted=tuple(redacted),
        cleanup_failed=tuple(cleanup_failed),
        failed=tuple(failed),
        busy=tuple(busy),
        missing=tuple(missing),
    )


@dataclass(frozen=True)
class TaskFileDeletionReport:
    """Which of the requested tasks ended in which state.

    ``deleted`` tasks have no uploaded files left bound to them (including
    tasks that had none). ``failed`` tasks raised and were rolled back, but
    the stored files deleted before the failure stay deleted: they are
    partially cleaned and must be re-run. ``busy`` and ``missing`` mean what
    they mean for :class:`TaskContentRedactionReport`.
    """

    deleted: tuple[int, ...] = ()
    failed: tuple[int, ...] = ()
    busy: tuple[int, ...] = ()
    missing: tuple[int, ...] = ()


def delete_task_content_files(
    db: Session, task_ids: Sequence[int]
) -> TaskFileDeletionReport:
    """Delete each task's uploaded files and keep everything else.

    The files-only sibling of :func:`redact_task_content`, with the same
    contract: one transaction per settled task, the same ``busy`` refusal
    (the host settles parked tasks first), per-task ``failed`` reporting,
    a clean ``db`` required, idempotent re-runs. The file semantics are
    :func:`task_deletion.delete_task_files`: a file a knowledge base still
    uses is detached rather than deleted, each deleted file is fenced
    against KB reuse, and a file another cleanup owner claimed is left to it.

    The transcript, trace, task columns, cached agent and workspace
    directory are untouched -- the task keeps talking about files it no
    longer has, which is what a files-only deletion asks for. Read caches
    are invalidated after each commit.

    Blocking (database, object storage and filesystem I/O); async callers
    run it in a worker thread.
    """

    now = datetime.now(timezone.utc)
    deleted: list[int] = []
    failed: list[int] = []
    busy: list[int] = []
    missing: list[int] = []
    for task_id in dict.fromkeys(task_ids):
        try:
            outcome = _run_on_quiescent_task(
                db, task_id, lambda task: delete_task_files(db, task), now=now
            )
        except Exception:
            logger.exception(
                "Task %s file deletion failed and was rolled back", task_id
            )
            failed.append(task_id)
            continue
        match outcome:
            case "busy":
                busy.append(task_id)
            case "missing":
                missing.append(task_id)
            case None:
                invalidate_task_cache(task_id)
                deleted.append(task_id)
    return TaskFileDeletionReport(
        deleted=tuple(deleted),
        failed=tuple(failed),
        busy=tuple(busy),
        missing=tuple(missing),
    )
