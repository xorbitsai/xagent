"""Savepoint necessity for ``supersede_legacy_question_rows`` (PostgreSQL half).

The one cell of the eight (T-S-7c) that cannot be proven on SQLite. The
reason is a real, already-documented backend difference, not a hypothesis:
``test_task_status_storage_postgresql.py``'s module docstring records that
once a statement fails, "PostgreSQL marks the transaction failed and every
further statement on the same connection raises InFailedSqlTransaction
until a rollback". SQLite has no such behavior -- a failed statement there
does not poison the rest of the open transaction, so removing the helper's
``db.begin_nested()`` savepoint cannot be made to fail on SQLite: the
outer commit would still succeed, and the mutation this test exists to
catch would stay invisible.

This test provokes a *real* database-layer failure -- a CHECK constraint
violation on the exact UPDATE statement the helper issues, added via raw
DDL against this disposable database -- rather than a mocked exception, so
the transaction is actually left aborted by PostgreSQL itself when the
savepoint is missing.

Requires ``XAGENT_TEST_POSTGRES_URL`` pointed at a disposable database;
skipped otherwise. The fixture below runs ``Base.metadata.drop_all`` on
setup and teardown against whatever that URL names -- never point it at a
database you care about.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import ops_signals
from xagent.web.services.chat_history_service import supersede_legacy_question_rows


@pytest.fixture()
def db_session():
    """Isolated tables in a real, disposable PostgreSQL database."""
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.rollback()
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _clean_signal():
    ops_signals.clear_degradation(ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED)
    yield
    ops_signals.clear_degradation(ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED)


def _create_task_with_question_row(db) -> tuple[int, int]:
    user = User(username="supersede-pg", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    task = Task(
        user_id=int(user.id),
        title="Supersede PG task",
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    row = TaskChatMessage(
        task_id=int(task.id),
        user_id=int(user.id),
        role="assistant",
        content="A question",
        message_type="question",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return int(task.id), int(row.id)


@pytest.mark.postgresql
def test_supersede_savepoint_absorbs_a_real_check_violation_and_the_caller_still_commits(
    db_session,
) -> None:
    """A real CHECK constraint makes the helper's own UPDATE fail inside
    PostgreSQL. With the savepoint in place, the failure is contained: the
    helper degrades, the row is untouched, and -- the actual point of this
    test -- the caller's own finishing transaction, which still has more
    work queued in the same commit (exactly the real call-site shape:
    supersede runs alongside other writes, not alone), can still commit
    afterward.

    The trailing write matters for what this test actually proves: a bare
    ``db.commit()`` with nothing else pending would silently succeed even
    against an aborted PostgreSQL transaction -- PostgreSQL treats COMMIT
    on an already-aborted transaction as an implicit rollback rather than
    an error, so a caller with no other pending work would never notice
    the difference between the correct and the mutated helper. Queuing a
    second write reproduces the real failure surface: the flush that
    write requires is a real statement sent while the transaction is
    still aborted, which is exactly where a missing savepoint turns into
    ``InFailedSqlTransaction`` for the caller.
    """
    task_id, row_id = _create_task_with_question_row(db_session)
    user_id = db_session.query(Task).filter(Task.id == task_id).one().user_id

    # A real DB-layer failure on exactly the statement the helper issues:
    # forbid writing the literal this UPDATE writes.
    db_session.execute(
        text(
            "ALTER TABLE task_chat_messages "
            "ADD CONSTRAINT ck_probe_block_supersede "
            "CHECK (message_type <> 'question_superseded')"
        )
    )
    db_session.commit()

    result = supersede_legacy_question_rows(db_session, task_id=task_id)

    assert result == 0
    signal_name = ops_signals.CLARIFICATION_LEGACY_SUPERSEDE_FAILED
    assert signal_name in ops_signals.active_degradations()

    # The rest of the caller's finishing transaction: one more write
    # queued in the same commit, matching the real call-site shape.
    trailing_write = TaskChatMessage(
        task_id=task_id,
        user_id=int(user_id),
        role="assistant",
        content="final answer",
        message_type="assistant_message",
    )
    db_session.add(trailing_write)
    db_session.commit()

    row = db_session.query(TaskChatMessage).filter(TaskChatMessage.id == row_id).one()
    assert row.message_type == "question"
    assert trailing_write.id is not None
