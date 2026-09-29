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

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from ..models.expired_task import ExpiredTaskTombstone
from ..models.task import Task
from ..models.trigger import AgentTrigger, TriggerRun
from ..models.workforce import WorkforceRun
from .task_runtime import mcp_runtime_authorization_policy_required_clause


def _trigger_type_for_task(db: Session, task_id: int, agent_config: Any) -> str | None:
    """The task's raw trigger type: webhook/scheduled/gmail, not the UI source.

    Mirrors ``api/conversation_logs.py``'s ``_trigger_type_for_task`` rather
    than importing it: ``api/`` depends on ``services/``, never the other way
    around. The ``agent_config`` key wins first, the same precedence
    ``_conversation_source_query``'s ``coalesce`` uses there; the fallback
    reads ``AgentTrigger.type`` off the highest-id ``TriggerRun`` for this
    task, since ``trigger_runs.task_id`` is not unique and only the newest
    run's trigger is the live one.
    """
    config_type = (
        agent_config.get("trigger_type") if isinstance(agent_config, Mapping) else None
    )
    if config_type:
        return str(config_type)
    trigger_type = db.execute(
        select(AgentTrigger.type)
        .join(TriggerRun, TriggerRun.trigger_id == AgentTrigger.id)
        .where(TriggerRun.task_id == task_id)
        .order_by(TriggerRun.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    return str(trigger_type) if trigger_type else None


def record_task_expiry_no_commit(db: Session, task_id: int, *, now: datetime) -> None:
    """Record that the retention purge is expiring ``task_id``'s conversation.

    Call under the purge's ``FOR UPDATE`` on the task row, before the row is
    deleted, in the same transaction. Does not commit: the records must commit
    or roll back together with the delete, or a reader could be told a task
    expired that still exists, or not be told about one that is gone.

    A tombstone already stored under this id is replaced rather than
    conflicted on. SQLite reuses the highest deleted task id -- tests and any
    other SQLite caller of this writer can expire a task, create a new one at
    the same id, and expire that one too, which would hit this table's own
    primary key without this delete. PostgreSQL never reuses ids, so there the
    replace never fires; either way, the tombstone that survives describes
    whichever task most recently expired under this id, which is the one a
    reader asking about it now means.
    """
    row = db.execute(
        select(
            Task.user_id,
            Task.agent_id,
            Task.source,
            Task.is_visible,
            Task.agent_config,
            Task.created_at,
            # Projected as the same boolean expression Conversation Logs'
            # scope filter applies, and read out already coerced to a real
            # boolean (never ``NULL``) so the stored flag equals that live
            # predicate on every dialect by construction.
            mcp_runtime_authorization_policy_required_clause()
            .is_(True)
            .label("is_channel_plumbing"),
        ).where(Task.id == task_id)
    ).one()
    # ``workforce_runs.task_id`` is unique, so at most one run owns the task.
    workforce_id = db.execute(
        select(WorkforceRun.workforce_id).where(WorkforceRun.task_id == task_id)
    ).scalar_one_or_none()
    # Live, ``trigger_type`` is reached through ``TriggerRun.task_id``, which
    # the purge SETs NULL once the task row is deleted, so -- like
    # ``workforce_id`` above -- it must be resolved before that happens.
    trigger_type = _trigger_type_for_task(db, task_id, row.agent_config)

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
            trigger_type=trigger_type,
            is_visible=bool(row.is_visible),
            is_channel_plumbing=bool(row.is_channel_plumbing),
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
    # Explicit rather than left to the session's next autoflush: sessions run
    # with ``autoflush=False`` (``models/database.py``), so without this call
    # nothing here would reach the database until something else flushed. The
    # tombstone has no foreign key to ``tasks`` -- ordering against the
    # task delete is not the reason -- the reason is that a failure this
    # flush raises surfaces here, inside the purge's own ``try``, rather than
    # at some later, unrelated flush point; and any read of this row later in
    # this transaction sees it.
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
