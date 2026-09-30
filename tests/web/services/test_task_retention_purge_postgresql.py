"""What fences the purge against a concurrent command insert (#2563).

Two sessions against a real PostgreSQL, because the claim under test is a
claim about PostgreSQL's row locks and nothing a single session can observe
would support it.

The claim: ``task_execution_commands.task_id`` is ``NOT NULL`` with a foreign
key to ``tasks.id``, so PostgreSQL validates every command insert by taking
``FOR KEY SHARE`` on the parent task row, and ``FOR KEY SHARE`` conflicts with
the ``FOR UPDATE`` that :func:`assess_task_retention` holds. If that holds,
the purge is fenced against *every* command producer -- including the three
(``api/websocket.py``, ``services/task_interaction_service.py``,
``services/workforce_runtime.py``) that #2562 could not verify take the task
row first -- without any of them being changed, because the fence is the
foreign key rather than the producer's own locking discipline.

These tests are why that is written as established rather than as reasoning
about a lock-compatibility matrix. They run in CI's PostgreSQL job; without
``XAGENT_TEST_POSTGRES_URL`` the whole module skips.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services.task_retention import (
    RetentionDisposition,
    assess_task_retention,
)
from xagent.web.services.task_retention_purge import (
    RetentionPurgeAction,
    _purge_trace_rows,
    purge_task,
)

pytestmark = pytest.mark.postgresql

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
CONVERSATION_DAYS = 365
TRACE_DAYS = 90

#: Long enough that a lock genuinely taken and released is not reported as
#: contention, short enough that a real block fails the test in a second
#: rather than hanging the suite.
LOCK_TIMEOUT_MS = 1000


@pytest.fixture
def sessions():
    with disposable_database_factory("retention_purge") as make:
        engine = make("lock")
        Base.metadata.create_all(engine)
        yield sa.orm.sessionmaker(bind=engine, autoflush=False)


def _seed_expired_task(sessions: sessionmaker[Session], *, days_old: int) -> int:
    with sessions() as db:
        user = User(username=f"purge-lock-{days_old}", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="retention purge lock fixture",
            status=TaskStatus.COMPLETED,
            last_activity_at=NOW - timedelta(days=days_old),
        )
        db.add(task)
        db.flush()
        event = TraceEvent(
            task_id=int(task.id),
            event_id=f"evt-{task.id}",
            event_type="agent_execution_checkpoint",
            timestamp=NOW,
            data={},
        )
        db.add(event)
        db.flush()
        # The pointer is the whole point of the ordering: it is a foreign key
        # into the rows both paths delete, so a seed without it would let a
        # reversed NULL-first pass.
        task.last_checkpoint_event_id = str(event.event_id)
        task.last_checkpoint_trace_event_id = int(event.id)
        db.commit()
        return int(task.id)


def _insert_command(db: Session, task_id: int, *, command_id: str = "cmd-1") -> None:
    """Insert a pending command the way every producer ultimately does.

    Deliberately not routed through ``stage_task_command``: what is under test
    is the foreign key's own locking, which is identical whichever module
    issues the INSERT. Going through the transport would test the transport.
    """
    db.add(
        TaskExecutionCommand(
            task_id=task_id,
            actor_user_id=None,
            command_id=command_id,
            kind="append",
            payload={},
            status="pending",
        )
    )
    db.flush()


def test_the_assessment_lock_blocks_a_concurrent_command_insert(sessions) -> None:
    """The fence itself: FOR UPDATE on the task row blocks the FK's FOR KEY SHARE."""
    task_id = _seed_expired_task(sessions, days_old=400)

    with sessions() as purger, sessions() as producer:
        assessment = assess_task_retention(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
        assert assessment.eligible, "fixture must be eligible for this to mean anything"

        producer.execute(sa.text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT_MS}ms'"))
        with pytest.raises((OperationalError, DBAPIError)) as excinfo:
            _insert_command(producer, task_id)
        # The timeout is what proves it blocked: an unfenced insert returns in
        # microseconds and never reaches the timeout at all.
        assert "lock timeout" in str(excinfo.value).lower()
        producer.rollback()
        purger.rollback()


def test_a_command_committed_first_makes_the_purge_skip_the_task(sessions) -> None:
    """Producer wins: the purge sees the pending command and leaves the task alone."""
    task_id = _seed_expired_task(sessions, days_old=400)

    with sessions() as producer:
        _insert_command(producer, task_id)
        producer.commit()

    with sessions() as purger:
        action = purge_task(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )

    assert action is RetentionPurgeAction.SKIPPED_BUSY
    with sessions() as db:
        assert (
            db.execute(
                sa.select(Task.id).where(Task.id == task_id)
            ).scalar_one_or_none()
            is not None
        )


def test_a_command_arriving_after_the_purge_fails_instead_of_vanishing(
    sessions,
) -> None:
    """Purge wins: the late command is rejected, never silently deleted.

    This is the half that matters for correctness. A command accepted against
    a task the purge has already removed must not commit -- the foreign key is
    what makes that a rejection rather than a row nobody will ever execute.
    """
    task_id = _seed_expired_task(sessions, days_old=400)

    with sessions() as purger:
        action = purge_task(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
    assert action is RetentionPurgeAction.PURGED_CONVERSATION

    with sessions() as producer:
        with pytest.raises(IntegrityError):
            _insert_command(producer, task_id)
        producer.rollback()


def test_the_purge_loses_to_a_producer_that_commits_while_it_waits(sessions) -> None:
    """The "purge loses cleanly" ordering #2563 asks for, with real overlap.

    The other race tests are sequential -- one session commits before the
    other starts. Here the producer's insert is *uncommitted* while the purge
    tries to assess, which is the ordering that actually happens under load:
    the purge must not see a half-written command, must not delete the task,
    and must not block forever waiting for a transaction it cannot influence.

    ``SKIP LOCKED`` is what makes the last part true. The producer's insert
    takes ``FOR KEY SHARE`` on the task row, so the purge's ``FOR UPDATE``
    cannot be granted; without ``SKIP LOCKED`` it would wait for the
    producer's transaction, and a sweep would stall behind any slow writer.
    """
    task_id = _seed_expired_task(sessions, days_old=400)

    with sessions() as producer, sessions() as purger:
        _insert_command(producer, task_id)  # held open, not committed

        action = purge_task(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
        # Skipped rather than blocked: the row is locked by the producer, and
        # a locked row reads back as absent, which is reported as not
        # eligible.
        assert action is RetentionPurgeAction.SKIPPED_BUSY

        producer.commit()

    with sessions() as db:
        assert (
            db.execute(
                sa.select(Task.id).where(Task.id == task_id)
            ).scalar_one_or_none()
            is not None
        ), "the task must survive a command that was in flight"
        assert (
            db.execute(
                sa.select(sa.func.count())
                .select_from(TaskExecutionCommand)
                .where(TaskExecutionCommand.task_id == task_id)
            ).scalar_one()
            == 1
        ), "the accepted command must survive too"


def test_two_concurrent_purges_of_one_task_do_not_both_delete_it(sessions) -> None:
    """What ``docs/deployment.md`` promises about multiple web replicas.

    Each replica runs its own loop, so the same task can be selected twice.
    The row lock is what makes that safe, and this is the test behind the
    claim: one purge deletes, the other finds nothing and says so rather than
    failing or double-counting.
    """
    task_id = _seed_expired_task(sessions, days_old=400)

    with sessions() as first, sessions() as second:
        first_action = purge_task(
            first,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
        second_action = purge_task(
            second,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )

    assert first_action is RetentionPurgeAction.PURGED_CONVERSATION
    assert second_action is RetentionPurgeAction.SKIPPED_BUSY
    with sessions() as db:
        assert (
            db.execute(
                sa.select(Task.id).where(Task.id == task_id)
            ).scalar_one_or_none()
            is None
        )


def test_trace_expiry_holds_the_lock_through_its_deletes(sessions) -> None:
    """The narrower path is fenced too, and through its writes, not just its read.

    Trace expiry keeps the task row, so nothing about its *outcome* would
    reveal a missing fence -- only the contention does. The deletes run here
    before the producer tries its insert, so what is pinned is that the lock
    still holds at the point the path has actually changed rows, rather than
    only at the assessment.
    """
    task_id = _seed_expired_task(sessions, days_old=100)

    with sessions() as purger, sessions() as producer:
        assessment = assess_task_retention(
            purger,
            task_id,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
        )
        assert assessment.disposition is RetentionDisposition.TRACE_EXPIRED
        _purge_trace_rows(purger, task_id)
        purger.flush()

        producer.execute(sa.text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT_MS}ms'"))
        with pytest.raises(OperationalError) as excinfo:
            _insert_command(producer, task_id)
        # Asserting the message, not just the exception type: IntegrityError is
        # itself a DBAPIError, so a NOT NULL or unique failure in the fixture
        # would otherwise read as "the insert was blocked".
        assert "lock timeout" in str(excinfo.value).lower()
        producer.rollback()
        purger.rollback()

    # And the rollback really did undo the deletes, so this test leaves the
    # trace behind rather than quietly depending on its own side effects.
    with sessions() as db:
        assert (
            db.execute(
                sa.select(sa.func.count())
                .select_from(TraceEvent)
                .where(TraceEvent.task_id == task_id)
            ).scalar_one()
            == 1
        )


def test_foreign_key_ordering_survives_both_paths_under_real_enforcement(
    sessions,
) -> None:
    """Neither path leaves a constraint violated on a server that enforces them.

    SQLite's coverage of the same orderings is contingent on a pragma this
    repo sets per engine; PostgreSQL has no such escape, which is what makes
    this the authoritative run of the ordering.
    """
    conversation_task = _seed_expired_task(sessions, days_old=400)
    trace_task = _seed_expired_task(sessions, days_old=100)

    for task_id, expected in (
        (conversation_task, RetentionPurgeAction.PURGED_CONVERSATION),
        (trace_task, RetentionPurgeAction.PURGED_TRACES),
    ):
        with sessions() as db:
            assert (
                purge_task(
                    db,
                    task_id,
                    now=NOW,
                    conversation_days=CONVERSATION_DAYS,
                    trace_days=TRACE_DAYS,
                )
                is expected
            )

    with sessions() as db:
        assert (
            db.execute(
                sa.select(Task.id).where(Task.id == conversation_task)
            ).scalar_one_or_none()
            is None
        )
        assert (
            db.execute(
                sa.select(Task.id).where(Task.id == trace_task)
            ).scalar_one_or_none()
            is not None
        )
        assert (
            db.execute(
                sa.select(sa.func.count())
                .select_from(TraceEvent)
                .where(TraceEvent.task_id == trace_task)
            ).scalar_one()
            == 0
        )
