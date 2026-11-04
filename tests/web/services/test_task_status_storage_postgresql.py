"""Task.status storage-semantics sentinels (PostgreSQL half).

Companion to test_task_status_storage.py. Fixture pattern copied from
test_runtime_key_transition_postgres.py:29-68 (skip-if-unset via
XAGENT_TEST_POSTGRES_URL). See that SQLite file's module docstring for the
storage-format background; this file exists because PostgreSQL and SQLite
fail differently for the same misuse:

- A raw text() SQL query with a value-cased literal WHERE clause matches
  zero rows on SQLite; on PostgreSQL the native ENUM type rejects the
  literal outright (DataError / InvalidTextRepresentation).
- Once that error happens, PostgreSQL marks the transaction failed and
  every further statement on the same connection raises
  InFailedSqlTransaction until a rollback -- each sentinel below that
  provokes a DB-layer error rolls back immediately afterward.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError

from tests.web.services.task_status_storage_shared import (
    RAW_VALUE_LITERAL_ZERO_MATCH_SQL,
    TASK_STATUS_STORAGE_NAMES,
    assert_cast_without_lower_silently_zero_matches,
    assert_orm_bind_rejects_raw_value_string,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus


@pytest.fixture()
def db_session():
    """Isolated Task rows in a real PostgreSQL test database."""
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
        db.close()
        Base.metadata.drop_all(bind=engine)


def _create_user(db) -> int:
    from xagent.web.models.user import User

    user = User(username="task-status-storage-pg", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return int(user.id)


def test_task_status_round_trip_postgresql(db_session) -> None:
    user_id = _create_user(db_session)
    for status, expected_name in TASK_STATUS_STORAGE_NAMES.items():
        task = Task(
            user_id=user_id,
            title=f"round-trip {status.name}",
            status=status,
        )
        db_session.add(task)
        db_session.commit()
        raw_value = db_session.execute(
            text("SELECT status FROM tasks WHERE id = :id"),
            {"id": task.id},
        ).scalar_one()
        assert raw_value == expected_name, (
            f"{status} stored as {raw_value!r}, expected member name {expected_name!r}"
        )


def test_pg_enum_reflects_exactly_the_taskstatus_members(db_session) -> None:
    """create_all's produced native ENUM must equal the model's member set.

    This is a structural sentinel, not a deployed-database check: it pins
    that a *fresh* create_all schema has no drift between the model's
    TaskStatus and the PostgreSQL ENUM type's labels. It cannot observe a
    deployed database whose ENUM type was created before a member was added
    to TaskStatus and never altered since; detecting that belongs to
    startup validation and is out of reach of any fixture-created schema.
    """
    rows = db_session.execute(
        text(
            "SELECT e.enumlabel FROM pg_type t "
            "JOIN pg_enum e ON t.oid = e.enumtypid "
            "WHERE t.typname = :typename "
            "ORDER BY e.enumsortorder"
        ),
        {"typename": "taskstatus"},
    ).fetchall()
    reflected = [row[0] for row in rows]
    assert set(reflected) == {status.name for status in TaskStatus}, (
        f"pg_enum reflection {reflected} does not match the TaskStatus "
        "member set -- create_all() drifted from the model"
    )


def test_wrong_case_literal_query_raises_on_postgresql(db_session) -> None:
    """The DB-layer sentinel's PostgreSQL half: DataError, not zero rows.

    The DataError leaves this session's transaction in a failed state, so
    it is rolled back before the positive control below runs; without that
    rollback every following statement raises InFailedSqlTransaction.
    """
    user_id = _create_user(db_session)
    task = Task(
        user_id=user_id,
        title="wrong-case literal pg",
        status=TaskStatus.WAITING_FOR_USER,
    )
    db_session.add(task)
    db_session.commit()

    with pytest.raises(DataError) as exc_info:
        db_session.execute(
            text(RAW_VALUE_LITERAL_ZERO_MATCH_SQL),
            {"value": TaskStatus.WAITING_FOR_USER.value},
        )
    assert type(exc_info.value.orig).__name__ == "InvalidTextRepresentation", (
        f"expected .orig to be InvalidTextRepresentation, got "
        f"{type(exc_info.value.orig)!r}"
    )
    db_session.rollback()

    correct_case_rows = db_session.execute(
        text(RAW_VALUE_LITERAL_ZERO_MATCH_SQL),
        {"value": TASK_STATUS_STORAGE_NAMES[TaskStatus.WAITING_FOR_USER]},
    ).fetchall()
    assert correct_case_rows == [(task.id,)], (
        "positive control failed: the correct-case member name should "
        f"match: {correct_case_rows}"
    )


def test_cast_without_lower_silently_zero_matches_postgresql(db_session) -> None:
    user_id = _create_user(db_session)
    task = Task(
        user_id=user_id,
        title="cast-without-lower pg",
        status=TaskStatus.RUNNING,
    )
    db_session.add(task)
    db_session.commit()
    assert_cast_without_lower_silently_zero_matches(
        db_session, stored_status=TaskStatus.RUNNING
    )


def test_orm_bind_rejects_raw_value_string_postgresql(db_session) -> None:
    """Bind-layer sentinel's PostgreSQL half -- same StatementError(LookupError)
    shape as SQLite (test_orm_bind_rejects_raw_value_string_sqlite), pinned
    symmetrically. This raises before a statement ever reaches the server,
    so unlike the DB-layer sentinel above, no rollback recovery is needed.
    """
    user_id = _create_user(db_session)
    task = Task(
        user_id=user_id, title="bind-layer sentinel pg", status=TaskStatus.RUNNING
    )
    db_session.add(task)
    db_session.commit()
    assert_orm_bind_rejects_raw_value_string(db_session, task.id)


def test_poison_write_orm_rejected_raw_sql_poisons_on_postgresql(db_session) -> None:
    """Poison-write half A (ORM rejected) on PostgreSQL.

    Half B (raw SQL still poisons a subsequent ORM read) does not have a
    PostgreSQL equivalent: PostgreSQL's native ENUM type rejects the raw
    write itself (test_wrong_case_literal_query_raises_on_postgresql above)
    -- there is no window where an invalid label reaches the column, so
    there is nothing left to poison a later ORM read. That asymmetry is why
    the two backends are pinned separately rather than by one shared test.
    """
    user_id = _create_user(db_session)
    task = Task(user_id=user_id, title="poison-write pg", status=TaskStatus.RUNNING)
    db_session.add(task)
    db_session.commit()
    assert_orm_bind_rejects_raw_value_string(db_session, task.id)
