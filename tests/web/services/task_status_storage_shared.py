"""Shared fixtures for the Task.status storage-semantics sentinel suite.

Split across test_task_status_storage.py (SQLite, always runs) and
test_task_status_storage_postgresql.py (PostgreSQL, skips without
XAGENT_TEST_POSTGRES_URL) so the PostgreSQL suite can use the
skip-if-unset fixture pattern in test_runtime_key_transition_postgres.py
without dragging a real Postgres dependency into every local test run. The
assertions below are written once here and called from both files so both
backends are pinned to the identical bind-layer exception shape,
StatementError wrapping LookupError.
"""

from __future__ import annotations

from sqlalchemy import text, update
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from xagent.web.models.task import Task, TaskStatus

# Primary sentinel target: sqlalchemy.Enum(TaskStatus) with no
# values_callable persists member *names*, not member values. This is a
# hand-written literal, deliberately not derived from the column at import
# time. This literal is the primary guard;
# test_storage_names_match_column_compilation is a drift check that keeps it
# honest against the real column, not the reverse -- deriving the expectation
# from the column would make the pin agree with any change made to it.
TASK_STATUS_STORAGE_NAMES: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "PENDING",
    TaskStatus.RUNNING: "RUNNING",
    TaskStatus.PAUSED: "PAUSED",
    TaskStatus.WAITING_FOR_USER: "WAITING_FOR_USER",
    TaskStatus.COMPLETED: "COMPLETED",
    TaskStatus.FAILED: "FAILED",
}

# A plausible near-miss of the existing LOWER(CAST(status AS VARCHAR)) form
# used by
# src/xagent/migrations/versions/20260711_add_task_execution_control_state.py:
# CAST without LOWER(). The migration's own form is safe today only because
# every current TaskStatus member's name.lower() equals its value --
# test_member_name_lower_equals_value in test_task_status_storage.py pins
# that invariant. A bare CAST comparison against the lowercase value is a
# silent trap for any future member where that coincidence breaks; it fails
# now for the same reason. Zero rows, no exception, on both backends.
CAST_WITHOUT_LOWER_ZERO_MATCH_SQL = (
    "SELECT id FROM tasks WHERE CAST(status AS VARCHAR) = :value"
)

RAW_VALUE_LITERAL_ZERO_MATCH_SQL = "SELECT id FROM tasks WHERE status = :value"


def assert_orm_bind_rejects_raw_value_string(db: Session, task_id: int) -> None:
    """Bind-layer sentinel: Enum(validate_strings=True) rejects a raw value.

    ``validate_strings=True`` only closes the ORM/Core bind path -- this
    pins exactly that path failing closed with a symmetric exception shape
    on both backends, not the raw-SQL bypass (that is a separate DB-layer
    sentinel, necessarily backend-specific because SQLite and PostgreSQL
    react differently to a raw write of an unknown label).
    """
    try:
        db.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(status=TaskStatus.WAITING_FOR_USER.value)
        )
        db.flush()
    except StatementError as error:
        assert isinstance(error.orig, LookupError), (
            f"expected StatementError wrapping LookupError, got {type(error.orig)}"
        )
    else:
        raise AssertionError(
            "ORM update with a raw enum-value string must raise StatementError "
            "(validate_strings=True); it silently succeeded instead"
        )
    finally:
        db.rollback()


def assert_cast_without_lower_silently_zero_matches(
    db: Session, *, stored_status: TaskStatus
) -> None:
    rows = db.execute(
        text(CAST_WITHOUT_LOWER_ZERO_MATCH_SQL),
        {"value": stored_status.value},
    ).fetchall()
    assert rows == [], (
        "CAST(status AS VARCHAR) compared against the enum value was "
        f"expected to silently zero-match (name stored, not value): {rows}"
    )
