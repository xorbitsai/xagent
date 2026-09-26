"""The records the retention purge leaves so readers can say "expired" (#2565).

Conversation expiry deletes a task outright. Without a record, every reader
afterwards sees the same thing it sees for a task its owner deleted or one
that never existed: not-found. For an integration polling historical task ids
that is a silent 200 -> 404 flip, and for a trigger or workforce run it turns
a completed run into one whose task simply vanished.

This module writes those records, in the purge's own transaction and under its
row lock, and reads the tombstone back:

* :func:`record_task_expiry_no_commit` -- conversation expiry. One content-free
  :class:`ExpiredTaskTombstone`, plus ``task_expired_at`` on every trigger and
  workforce run that points at the task. Must run *before* the task row is
  deleted: the tombstone's columns are read from the row, and the runs are
  found by ``task_id``, which the delete SETs NULL.
* :func:`find_expired_task` -- the read primitive. Returns the tombstone only
  when no live task holds the id.

Only the retention purge calls the writer. User-initiated deletion goes
through ``purge_task_rows`` and writes nothing here, which is why none of this
lives in that shared helper: a task its owner deleted keeps answering
not-found.

Neither write may disturb its row's own clocks. ``TriggerRun.updated_at`` and
``WorkforceRun.last_activity_at`` both carry ``onupdate=func.now()``, and the
second is what the preview-run reaper reads to decide a run is stale -- so an
expiry write that let it fire would make an expired run look freshly active.
Both are pinned to their current value in the SET clause, the same idiom the
trace path uses for ``tasks.updated_at``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from ..models.expired_task import ExpiredTaskTombstone
from ..models.task import Task
from ..models.trigger import TriggerRun
from ..models.workforce import WorkforceRun
from .task_runtime import mcp_runtime_authorization_policy_required


def record_task_expiry_no_commit(db: Session, task_id: int, *, now: datetime) -> None:
    """Record that the retention purge is expiring ``task_id``'s conversation.

    Call under the purge's ``FOR UPDATE`` on the task row, before the row is
    deleted, in the same transaction. Does not commit: the records must commit
    or roll back together with the delete, or a reader could be told a task
    expired that still exists, or not be told about one that is gone.

    A tombstone already stored under this id is replaced rather than
    conflicted on. PostgreSQL never reuses a task id, so on the only dialect
    the purge runs on that cannot happen; if it ever did, the older tombstone
    would describe a different task, and the newer one is the one to keep.
    """
    row = db.execute(
        select(
            Task.user_id,
            Task.agent_id,
            Task.source,
            Task.is_visible,
            Task.agent_config,
            Task.created_at,
        ).where(Task.id == task_id)
    ).one()
    # ``workforce_runs.task_id`` is unique, so at most one run owns the task.
    workforce_id = db.execute(
        select(WorkforceRun.workforce_id).where(WorkforceRun.task_id == task_id)
    ).scalar_one_or_none()

    db.execute(
        delete(ExpiredTaskTombstone)
        .where(ExpiredTaskTombstone.task_id == task_id)
        .execution_options(synchronize_session=False)
    )
    db.add(
        ExpiredTaskTombstone(
            task_id=task_id,
            user_id=int(row.user_id),
            agent_id=row.agent_id,
            workforce_id=workforce_id,
            source=row.source,
            is_visible=bool(row.is_visible),
            # The same test the MCP runtime applies, so the boolean means what
            # Conversation Logs' SQL predicate means: only a literal ``True``.
            is_channel_plumbing=mcp_runtime_authorization_policy_required(
                row.agent_config
            ),
            task_created_at=row.created_at,
            expired_at=now,
        )
    )
    db.execute(
        update(TriggerRun)
        .where(TriggerRun.task_id == task_id)
        .values(task_expired_at=now, updated_at=TriggerRun.updated_at)
        .execution_options(synchronize_session=False)
    )
    db.execute(
        update(WorkforceRun)
        .where(WorkforceRun.task_id == task_id)
        .values(task_expired_at=now, last_activity_at=WorkforceRun.last_activity_at)
        .execution_options(synchronize_session=False)
    )
    # The purge deletes the task through the ORM, which flushes in its own
    # order; flush the tombstone now so it cannot be reordered past anything
    # the delete depends on.
    db.flush()


def find_expired_task(db: Session, task_id: int) -> ExpiredTaskTombstone | None:
    """The tombstone for ``task_id``, or ``None``.

    ``None`` whenever a live task holds the id, so a tombstone can never
    shadow a task that exists. PostgreSQL does not reuse ids, but SQLite
    hands the highest deleted id to the next insert, and this read must be
    right on both.

    This answers "did retention expire this id" and nothing else. Whether the
    caller may be *told* so is the caller's access predicate, applied to the
    tombstone's columns exactly as it would have been applied to the task's:
    a tombstone the caller could not have seen live must produce the same
    not-found a missing task does.
    """
    if db.execute(select(Task.id).where(Task.id == task_id)).first() is not None:
        return None
    return db.get(ExpiredTaskTombstone, task_id)
