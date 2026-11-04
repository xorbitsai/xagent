"""Shared fixtures for the retention purge's expired-task tombstone (#2565).

``test_conversation_logs.py`` and ``test_chat_task_expired.py`` both need to
insert a tombstone directly -- pinning a route's access predicate without
depending on the purge's eligibility rules -- and to run the real retention
purge end to end and assert the resulting ``410 task_expired`` body. This
module holds what both need so neither reimplements it, the same way
``monitor_daily_window_shared.py`` is shared by the SQLite and PostgreSQL
monitor-window suites.

Not a ``conftest.py``: these are plain functions and a value, imported by name
where they are used. Each route's plain-404 (not-found) assertion stays local
to its own test file -- the detail string differs per route, and that is the
one thing this module deliberately does not paper over.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from xagent.web.models.expired_task import ExpiredTaskTombstone
from xagent.web.models.task import Task
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services.task_retention_purge import RetentionPurgeAction, purge_task

from .conftest import _direct_db_session

#: The tombstone timestamp inserted tests pin their expiry to.
EXPIRED_AT = datetime(2026, 9, 1, 8, 30, 0, tzinfo=timezone.utc)


def insert_tombstone(
    *,
    task_id: int,
    user_id: int,
    source: str | None = "sdk",
    trigger_type: str | None = None,
    is_visible: bool = False,
    is_channel_plumbing: bool = False,
    agent_id: int | None = None,
    workforce_id: int | None = None,
    expired_at: datetime = EXPIRED_AT,
) -> None:
    """Insert a tombstone directly, bypassing the purge's own eligibility rules.

    Defaults describe the ordinary hidden-external case (a completed ``sdk``
    conversation, not channel plumbing); pass overrides for the cell under
    test.
    """
    db = _direct_db_session()
    try:
        db.add(
            ExpiredTaskTombstone(
                task_id=task_id,
                user_id=user_id,
                agent_id=agent_id,
                workforce_id=workforce_id,
                source=source,
                trigger_type=trigger_type,
                is_visible=is_visible,
                is_channel_plumbing=is_channel_plumbing,
                task_created_at=expired_at,
                expired_at=expired_at,
            )
        )
        db.commit()
    finally:
        db.close()


def assert_task_expired(
    response: Any, task_id: int, expired_at: datetime = EXPIRED_AT
) -> None:
    """Assert the shared ``410 task_expired`` body every in-scope route answers."""
    assert response.status_code == 410, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "task_expired"
    assert detail["task_id"] == task_id
    assert datetime.fromisoformat(detail["expired_at"]) == expired_at
    assert detail["message"]


def purge_conversation(task_id: int) -> datetime:
    """Run the real retention purge on ``task_id``; return its ``now``.

    Makes the task eligible first -- terminal, no lease, no owed command, and
    an activity anchor 400 days before ``now`` against a 365-day conversation
    window -- the same shape ``test_tasks.py``'s trace-expiry steps test and
    the v1 suite's own ``_purge_conversation`` use. The caller creates the
    task (and any trigger run or workforce run it wants captured into the
    tombstone) before calling this.
    """
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    now = base + timedelta(days=400)
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.last_activity_at = base
        task.lease_expires_at = None
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id
        ).delete(synchronize_session=False)
        db.commit()
        assert (
            purge_task(db, task_id, now=now, conversation_days=365, trace_days=90)
            is RetentionPurgeAction.PURGED_CONVERSATION
        )
    finally:
        db.close()
    return now
