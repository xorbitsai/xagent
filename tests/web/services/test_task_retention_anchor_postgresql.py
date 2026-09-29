"""The purge against a transcript writer that never touches the anchor (#2580).

A binary that predates #2571 inserts ``task_chat_messages`` rows without
writing ``tasks.last_activity_at``. During a rolling deploy it can do so after
the migration's forward-only backfill has already anchored the task, and after
a rollback past #2571 it does so indefinitely. Either way the stored anchor
ends up older than the task's newest message, and a purge that trusted it
would expire the conversation early.

The fix does not depend on every writer being current. The locked assessment
reads the newest message itself, and ``task_chat_messages.task_id`` is a
``NOT NULL`` foreign key to ``tasks.id`` -- the same fence
``test_task_retention_purge_postgresql.py`` establishes for commands: the
insert's ``FOR KEY SHARE`` on the parent row conflicts with the assessment's
``FOR UPDATE``, so no message can commit between the read and the delete.

Two sessions against a real PostgreSQL, because both halves are claims about
committed-read visibility and row locks. Without ``XAGENT_TEST_POSTGRES_URL``
the whole module skips.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.task_retention import (
    RetentionDisposition,
    assess_task_retention,
)
from xagent.web.services.task_retention_purge import (
    RetentionPurgeAction,
    purge_task,
    select_purge_candidates,
)

pytestmark = pytest.mark.postgresql

MIGRATION = "xagent.migrations.versions.20260922_task_last_activity_at"

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)
CONVERSATION_DAYS = 365
TRACE_DAYS = 90
LOCK_TIMEOUT_MS = 1000


@pytest.fixture
def sessions():
    with disposable_database_factory("retention_anchor") as make:
        engine = make("anchor")
        Base.metadata.create_all(engine)
        yield sa.orm.sessionmaker(bind=engine, autoflush=False)


def _insert_unanchored_message(
    db: Session, *, task_id: int, user_id: int, created_at: datetime
) -> None:
    """Persist a transcript row the way a pre-#2571 binary does: no anchor write."""
    db.add(
        TaskChatMessage(
            task_id=task_id,
            user_id=user_id,
            role="user",
            content="written by a binary that never touches the anchor",
            message_type="text",
            created_at=created_at,
        )
    )
    db.flush()


def _seed_backfilled_task(sessions: sessionmaker[Session]) -> tuple[int, int]:
    """A 400-day-old conversation, anchored by the real migration backfill."""
    with sessions() as db:
        user = User(username="anchor-drift", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="retention anchor drift fixture",
            status=TaskStatus.COMPLETED,
            last_activity_at=None,
        )
        task.created_at = NOW - timedelta(days=500)
        db.add(task)
        db.flush()
        _insert_unanchored_message(
            db,
            task_id=int(task.id),
            user_id=int(user.id),
            created_at=NOW - timedelta(days=400),
        )
        db.commit()
        task_id, user_id = int(task.id), int(user.id)

    migration = importlib.import_module(MIGRATION)
    anchor_sql = migration._anchor_expression(with_messages=True, with_created_at=True)
    engine = sessions.kw["bind"]
    with engine.begin() as connection:
        migration._backfill(connection, anchor_sql=anchor_sql)
    return task_id, user_id


def _stored_anchor(sessions: sessionmaker[Session], task_id: int) -> datetime:
    with sessions() as db:
        return db.execute(
            sa.select(Task.last_activity_at).where(Task.id == task_id)
        ).scalar_one()


def test_an_old_writer_after_the_backfill_does_not_get_the_task_purged(
    sessions,
) -> None:
    """#2580's acceptance scenario, end to end through ``purge_task``.

    The batch holding the task has committed, then an old-style writer commits
    a newer message. The stored anchor is asserted stale first -- without that
    the test would pass for the wrong reason -- and the purge must still keep
    the conversation, because by the task's newest message it is 10 days old.
    """
    task_id, user_id = _seed_backfilled_task(sessions)
    assert _stored_anchor(sessions, task_id) == NOW - timedelta(days=400)

    with sessions() as old_writer:
        _insert_unanchored_message(
            old_writer,
            task_id=task_id,
            user_id=user_id,
            created_at=NOW - timedelta(days=10),
        )
        old_writer.commit()
    assert _stored_anchor(sessions, task_id) == NOW - timedelta(days=400), (
        "the drift this test exists for did not happen"
    )

    with sessions() as purger:
        action = purge_task(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
    assert action is RetentionPurgeAction.SKIPPED_NOT_DUE

    with sessions() as db:
        assert (
            db.execute(
                sa.select(sa.func.count())
                .select_from(TaskChatMessage)
                .where(TaskChatMessage.task_id == task_id)
            ).scalar_one()
            == 2
        ), "the conversation must survive a purge measured from a stale anchor"


def test_the_assessment_lock_blocks_an_unanchored_message_insert(sessions) -> None:
    """No message can land between the anchor read and the delete.

    Otherwise the fix would only narrow the race: a message committed after
    the assessment read ``MAX(created_at)`` but before the purge deleted would
    be removed with a conversation it had just made recent.
    """
    task_id, user_id = _seed_backfilled_task(sessions)

    with sessions() as purger, sessions() as old_writer:
        assessment = assess_task_retention(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
        assert assessment.disposition is RetentionDisposition.CONVERSATION_EXPIRED, (
            "fixture must be expirable for the block to mean anything"
        )

        old_writer.execute(sa.text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT_MS}ms'"))
        with pytest.raises(DBAPIError) as excinfo:
            _insert_unanchored_message(
                old_writer,
                task_id=task_id,
                user_id=user_id,
                created_at=NOW - timedelta(days=1),
            )
        assert "lock timeout" in str(excinfo.value).lower()
        old_writer.rollback()
        purger.rollback()


def _message_count(sessions: sessionmaker[Session], task_id: int) -> int:
    with sessions() as db:
        return db.execute(
            sa.select(sa.func.count())
            .select_from(TaskChatMessage)
            .where(TaskChatMessage.task_id == task_id)
        ).scalar_one()


def test_an_unanchored_message_in_flight_makes_the_purge_skip_the_task(
    sessions,
) -> None:
    """The other ordering: the insert is open when the purge tries to lock.

    The insert's foreign-key check already holds ``FOR KEY SHARE`` on the task
    row, so the assessment's ``FOR UPDATE SKIP LOCKED`` is not granted and the
    task reads back as busy. The purge must neither wait on the writer nor
    delete a conversation from a snapshot that cannot see its newest message.
    """
    task_id, user_id = _seed_backfilled_task(sessions)

    with sessions() as old_writer, sessions() as purger:
        _insert_unanchored_message(
            old_writer,
            task_id=task_id,
            user_id=user_id,
            created_at=NOW - timedelta(days=1),
        )  # flushed, not committed

        action = purge_task(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
        assert action is RetentionPurgeAction.SKIPPED_BUSY

        old_writer.commit()

    assert _message_count(sessions, task_id) == 2


def test_a_drifted_task_keeps_its_conversation_across_sweeps(sessions) -> None:
    """The downgrade end to end, and what every later sweep does with it.

    By the stored anchor the task is 400 days old, so the scan admits it for
    conversation expiry. By its newest message it is 200 days old, which only
    the trace period has passed. The first sweep removes the trace and keeps
    the task; the stored anchor is not repaired, so later sweeps select it
    again and find nothing left to delete.
    """
    task_id, user_id = _seed_backfilled_task(sessions)
    with sessions() as db:
        _insert_unanchored_message(
            db,
            task_id=task_id,
            user_id=user_id,
            created_at=NOW - timedelta(days=200),
        )
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id=f"evt-{task_id}",
                event_type="agent_execution_checkpoint",
                timestamp=NOW - timedelta(days=200),
                data={},
            )
        )
        db.commit()

    actions = []
    for _sweep in range(2):
        with sessions() as db:
            candidates = select_purge_candidates(
                db,
                now=NOW,
                conversation_days=CONVERSATION_DAYS,
                trace_days=TRACE_DAYS,
                limit=10,
            )
        assert task_id in candidates
        with sessions() as db:
            actions.append(
                purge_task(
                    db,
                    task_id,
                    now=NOW,
                    conversation_days=CONVERSATION_DAYS,
                    trace_days=TRACE_DAYS,
                )
            )

    assert actions == [
        RetentionPurgeAction.PURGED_TRACES,
        RetentionPurgeAction.NOTHING_TO_PURGE,
    ]
    assert _message_count(sessions, task_id) == 2
    with sessions() as db:
        assert db.get(Task, task_id) is not None
        assert (
            db.execute(
                sa.select(sa.func.count())
                .select_from(TraceEvent)
                .where(TraceEvent.task_id == task_id)
            ).scalar_one()
            == 0
        )
