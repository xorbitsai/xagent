"""Deletion-path tests for the exact-row checkpoint pointer's NULL-first
ordering.

Each fixture below builds its own dedicated engine rather than reusing the
shared ``tests/web/api/conftest.py`` engine. That shared engine already runs
with SQLite FK enforcement on (``init_db`` -> ``configure_db`` ->
``apply_sqlite_concurrency_pragmas``), so a mistake in a fixture that
touches it would widen blast radius across every one of the ~100 web-API
test files that share it; a dedicated engine keeps this module's fixtures
self-contained instead.

Three schema forms are covered, matching the dialect asymmetry the
migration documents: a freshly ``create_all``'d SQLite database (full
metadata, including the checkpoint pointer's FK) with FK enforcement turned
on for this engine's connections; a live PostgreSQL database (always FK
enforcing); and a SQLite database built through the Alembic migration
history, which has no DB-level FK for the checkpoint pointer column at all
(Alembic's SQLite batch mode cannot add one without a full table rebuild).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Query, Session, sessionmaker

from tests.web.services.checkpoint_anchor_shared import (
    CHECKPOINT_ANCHOR_FK_NAME as FK_NAME,
)
from tests.web.services.checkpoint_anchor_shared import (
    build_upgraded_sqlite_engine,
    reset_checkpoint_anchor_fk_create_rule,
)
from tests.web.services.task_interaction_schema_shared import (
    make_row as make_interaction_row,
)
from tests.web.services.task_interaction_schema_shared import (
    make_task,
    make_trace_event,
    make_user,
)
from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
from xagent.web.api.admin_users import _purge_user_task_rows
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task import TraceEvent as DatabaseTraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User
from xagent.web.services.task_deletion import purge_task_rows
from xagent.web.services.task_interaction_schema import (
    interaction_requests_table_exists,
)


def _seed_task_with_anchored_checkpoint(session: Session, *, username: str) -> int:
    """A task whose checkpoint pointer resolves to a real trace_events row."""
    user = User(username=username, password_hash="hash", is_admin=False)
    session.add(user)
    session.flush()
    task = Task(
        user_id=int(user.id),
        title="task",
        status=TaskStatus.PENDING,
    )
    session.add(task)
    session.flush()
    event_row = DatabaseTraceEvent(
        task_id=int(task.id),
        event_id="evt-1",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        data={"checkpoint_type": CHECKPOINT_TYPE, "snapshot": {"label": "x"}},
    )
    session.add(event_row)
    session.flush()
    task.last_checkpoint_event_id = event_row.event_id
    task.last_checkpoint_trace_event_id = event_row.id
    session.commit()
    return int(task.id)


def _seed_task_with_interaction_row(
    session: Session, *, username: str, status: str = "active"
) -> tuple[int, int]:
    """A task with one interaction row anchored to a real trace_events row.

    Independent of _seed_task_with_anchored_checkpoint above: these tests
    are about task_interaction_requests.resume_trace_event_id, not
    tasks.last_checkpoint_trace_event_id, so the task's own pointer is left
    unset.
    """
    user_id = make_user(session)
    task_id = make_task(session, user_id=user_id)
    anchor_id = make_trace_event(session, task_id=task_id)
    row = TaskInteractionRequest(
        **make_interaction_row(
            task_id=task_id, resume_trace_event_id=anchor_id, status=status
        )
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return task_id, int(row.id)


@contextmanager
def _recorded_statements(engine):
    """Record every statement this engine executes, in order.

    The alembic-upgraded form has no DB-level FK for the pointer column, so
    a reversed NULL-first cannot fail loudly here. Asserting the final
    persisted pointer is NULL catches *removing* the NULL-first step but not
    *moving* it after the trace_events delete, because the update still runs
    before the assertion either way. Statement order is the only direct
    evidence of the ordering itself.
    """
    seen: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        seen.append(" ".join(statement.split()))

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


def _index_of(seen, predicate, what):  # type: ignore[no-untyped-def]
    for index, statement in enumerate(seen):
        if predicate(statement):
            return index
    raise AssertionError(f"{what} never executed; recorded: {seen}")


def _assert_pointer_nulled_before_trace_events_deleted(seen) -> None:  # type: ignore[no-untyped-def]
    null_first = _index_of(
        seen,
        lambda s: s.startswith("UPDATE tasks")
        and "last_checkpoint_trace_event_id" in s,
        "pointer NULL update",
    )
    trace_delete = _index_of(
        seen,
        lambda s: s.startswith("DELETE FROM trace_events"),
        "trace_events delete",
    )
    assert null_first < trace_delete, seen


# ---------------------------------------------------------------------------
# Fixtures: three dedicated schema forms.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_fk_on_session(tmp_path):
    """Freshly create_all'd SQLite database, FK enforcement on for this
    engine's connections only (not the shared conftest engine)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'fk_on.db'}")

    @event.listens_for(engine, "connect")
    def _set_fk(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def _tables_excluding_interaction_requests() -> list:
    return [
        table
        for table in Base.metadata.sorted_tables
        if table.name != TaskInteractionRequest.__tablename__
    ]


@pytest.fixture
def sqlite_fk_on_session_without_interaction_table(tmp_path):
    """Same shape as sqlite_fk_on_session, minus task_interaction_requests
    -- reproduces a deployment upgraded to a revision before that table
    exists (see test_task_interaction_schema_gate.py for the same filter
    used against the presence predicate directly)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'fk_on_no_interaction.db'}")

    @event.listens_for(engine, "connect")
    def _set_fk(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(
        bind=engine, tables=_tables_excluding_interaction_requests()
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sqlite_upgraded_session(tmp_path):
    """A SQLite database shaped like the migration's ADD COLUMN path for a
    table that already existed -- the D1 asymmetry: no DB-level FK for the
    checkpoint pointer column on this form.

    ``tasks``/``trace_events`` are create_all-only in production (see
    fab71cf4b1ad_add_sdk_fields_to_tasks.py's guard) and never created by a
    migration, so running the real Alembic history against an empty
    database leaves them absent rather than reproducing this state. The
    actual rebuild trick lives in checkpoint_anchor_shared.py, shared with
    test_task_lease_recovery.py's sqlite_no_anchor_fk_session, which needs
    the identical shape.
    """
    engine = build_upgraded_sqlite_engine(tmp_path / "upgraded.db")
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


@pytest.fixture
def postgres_session():
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    yield session
    session.close()
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()


@pytest.fixture
def postgres_session_without_interaction_table():
    """Same shape as postgres_session, minus task_interaction_requests."""
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(
        bind=engine, tables=_tables_excluding_interaction_requests()
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    yield session
    session.close()
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()


# ---------------------------------------------------------------------------
# D1 asymmetry as a tested fact, not just a comment.
# ---------------------------------------------------------------------------


def test_upgraded_sqlite_has_no_db_level_anchor_fk(sqlite_upgraded_session) -> None:
    fk_names = {
        fk["name"]
        for fk in inspect(sqlite_upgraded_session.get_bind()).get_foreign_keys("tasks")
    }
    assert FK_NAME not in fk_names


def test_fresh_sqlite_has_the_db_level_anchor_fk(sqlite_fk_on_session) -> None:
    fk_names = {
        fk["name"]
        for fk in inspect(sqlite_fk_on_session.get_bind()).get_foreign_keys("tasks")
    }
    assert FK_NAME in fk_names


# ---------------------------------------------------------------------------
# purge_task_rows (single task).
# ---------------------------------------------------------------------------


def test_purge_task_rows_clears_the_anchor_under_fk_enforcement(
    sqlite_fk_on_session,
) -> None:
    session = sqlite_fk_on_session
    task_id = _seed_task_with_anchored_checkpoint(session, username="u1")

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert (
        session.query(DatabaseTraceEvent)
        .filter(DatabaseTraceEvent.task_id == task_id)
        .count()
        == 0
    )


@pytest.mark.postgresql
def test_purge_task_rows_clears_the_anchor_on_postgres(postgres_session) -> None:
    session = postgres_session
    task_id = _seed_task_with_anchored_checkpoint(session, username="u1")

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert (
        session.query(DatabaseTraceEvent)
        .filter(DatabaseTraceEvent.task_id == task_id)
        .count()
        == 0
    )


def test_purge_task_rows_nulls_pointer_before_the_task_delete_flushes(
    sqlite_upgraded_session,
) -> None:
    """The alembic-upgraded SQLite form has no DB-level FK to make a missing
    NULL-first fail loudly (see test_upgraded_sqlite_has_no_db_level_anchor_fk
    above), so this checks the fact directly instead of relying on an error.

    purge_task_rows's own trace_events delete and pointer-NULL update are
    both bulk statements that execute immediately, but its ``db.delete(task)``
    is a per-instance ORM delete that only flushes on the caller's next
    flush/commit -- and this session has autoflush disabled. So immediately
    after purge_task_rows() returns and before this test's own commit, the
    task row's pointer, as currently persisted, reveals whether NULL-first
    ran before the (already-executed) trace_events delete.
    """
    session = sqlite_upgraded_session
    task_id = _seed_task_with_anchored_checkpoint(session, username="u1")

    with _recorded_statements(session.get_bind()) as seen:
        assert purge_task_rows(session, task_id=task_id) is True
    _assert_pointer_nulled_before_trace_events_deleted(seen)

    pointer = session.execute(
        text("SELECT last_checkpoint_trace_event_id FROM tasks WHERE id = :id"),
        {"id": task_id},
    ).scalar_one()
    assert pointer is None
    session.commit()


# ---------------------------------------------------------------------------
# purge_task_rows: task_interaction_requests deletion (PR-C2b).
# ---------------------------------------------------------------------------


def _interaction_row_count(session: Session, task_id: int) -> int:
    return (
        session.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.task_id == task_id)
        .count()
    )


def test_purge_task_rows_deletes_an_active_interaction_row_sqlite(
    sqlite_fk_on_session,
) -> None:
    session = sqlite_fk_on_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-active-single", status="active"
    )

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


@pytest.mark.postgresql
def test_purge_task_rows_deletes_an_active_interaction_row_postgres(
    postgres_session,
) -> None:
    session = postgres_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-active-single", status="active"
    )

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


def test_purge_task_rows_deletes_a_terminal_interaction_row_sqlite(
    sqlite_fk_on_session,
) -> None:
    """Regression control for M4: without a direct row-count assertion, this
    test could stay green even if the interaction delete statement were
    removed entirely, because a terminal row's anchor is never SET NULL by
    the trace_events delete and so purge_task_rows would still return True
    and leave the task deleted."""
    session = sqlite_fk_on_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-terminal-single", status="terminated"
    )

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


@pytest.mark.postgresql
def test_purge_task_rows_deletes_a_terminal_interaction_row_postgres(
    postgres_session,
) -> None:
    session = postgres_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-terminal-single", status="terminated"
    )

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


def _assert_interaction_delete_between_pointer_update_and_trace_events_delete(
    seen,
) -> None:  # type: ignore[no-untyped-def]
    pointer_update = _index_of(
        seen,
        lambda s: s.startswith("UPDATE tasks")
        and "last_checkpoint_trace_event_id" in s,
        "pointer NULL update",
    )
    interaction_delete = _index_of(
        seen,
        lambda s: s.startswith("DELETE FROM task_interaction_requests"),
        "task_interaction_requests delete",
    )
    trace_delete = _index_of(
        seen,
        lambda s: s.startswith("DELETE FROM trace_events"),
        "trace_events delete",
    )
    assert pointer_update < interaction_delete < trace_delete, seen


def test_purge_task_rows_deletes_interaction_rows_before_trace_events_fk_on(
    sqlite_fk_on_session,
) -> None:
    session = sqlite_fk_on_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-order-single-fk-on", status="active"
    )

    with _recorded_statements(session.get_bind()) as seen:
        assert purge_task_rows(session, task_id=task_id) is True
    _assert_interaction_delete_between_pointer_update_and_trace_events_delete(seen)
    session.commit()


def test_purge_task_rows_deletes_interaction_rows_before_trace_events_upgraded(
    sqlite_upgraded_session,
) -> None:
    """The alembic-upgraded form is where this matters most: it has no
    DB-level FK for the checkpoint pointer column and (SQLite FK
    enforcement defaulting off for this engine) no enforced ON DELETE SET
    NULL for the interaction anchor either, so a reversed ordering would
    not fail loudly here. Statement order is the only direct evidence."""
    session = sqlite_upgraded_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-order-single-upgraded", status="active"
    )

    with _recorded_statements(session.get_bind()) as seen:
        assert purge_task_rows(session, task_id=task_id) is True
    _assert_interaction_delete_between_pointer_update_and_trace_events_delete(seen)
    session.commit()


@pytest.mark.postgresql
def test_purge_task_rows_deletes_interaction_rows_before_trace_events_postgres(
    postgres_session,
) -> None:
    session = postgres_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-order-single-postgres", status="active"
    )

    with _recorded_statements(session.get_bind()) as seen:
        assert purge_task_rows(session, task_id=task_id) is True
    _assert_interaction_delete_between_pointer_update_and_trace_events_delete(seen)
    session.commit()


# ---------------------------------------------------------------------------
# _purge_user_task_rows (bulk, admin path).
# ---------------------------------------------------------------------------


def test_purge_user_task_rows_clears_the_anchor_under_fk_enforcement(
    sqlite_fk_on_session,
) -> None:
    session = sqlite_fk_on_session
    task_id = _seed_task_with_anchored_checkpoint(session, username="u1")
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert (
        session.query(DatabaseTraceEvent)
        .filter(DatabaseTraceEvent.task_id == task_id)
        .count()
        == 0
    )


@pytest.mark.postgresql
def test_purge_user_task_rows_clears_the_anchor_on_postgres(postgres_session) -> None:
    session = postgres_session
    task_id = _seed_task_with_anchored_checkpoint(session, username="u1")
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert (
        session.query(DatabaseTraceEvent)
        .filter(DatabaseTraceEvent.task_id == task_id)
        .count()
        == 0
    )


def test_purge_user_task_rows_nulls_pointer_before_trace_events_are_gone(
    sqlite_upgraded_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same intent as the single-task test above, adapted for the bulk path:
    every step in _purge_user_task_rows (the pointer NULL-first, the
    trace_events delete, and the tasks delete) is an immediate bulk
    statement, so there is no ORM-deferred step to inspect afterward. This
    intercepts only the final tasks bulk delete -- turning it into a no-op
    -- so the intermediate state (trace_events already gone, tasks not yet
    deleted) is directly observable: pointer must already be NULL.
    """
    session = sqlite_upgraded_session
    task_id = _seed_task_with_anchored_checkpoint(session, username="u1")
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    original_delete = Query.delete

    def guarded_delete(self: Query, *args: object, **kwargs: object) -> int:
        descriptions = self.column_descriptions
        if descriptions and descriptions[0]["type"] is Task:
            return 0  # stand-in for the final tasks bulk delete
        return original_delete(self, *args, **kwargs)

    monkeypatch.setattr(Query, "delete", guarded_delete)

    with _recorded_statements(session.get_bind()) as seen:
        _purge_user_task_rows(session, user_id=user_id)
    _assert_pointer_nulled_before_trace_events_deleted(seen)

    pointer = session.execute(
        text("SELECT last_checkpoint_trace_event_id FROM tasks WHERE id = :id"),
        {"id": task_id},
    ).scalar_one()
    assert pointer is None
    remaining_events = session.execute(
        text("SELECT COUNT(*) FROM trace_events WHERE task_id = :id"),
        {"id": task_id},
    ).scalar_one()
    assert remaining_events == 0
    session.rollback()


# ---------------------------------------------------------------------------
# _purge_user_task_rows: task_interaction_requests deletion (PR-C2b).
# ---------------------------------------------------------------------------


def test_purge_user_task_rows_deletes_an_active_interaction_row_sqlite(
    sqlite_fk_on_session,
) -> None:
    session = sqlite_fk_on_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-active-bulk", status="active"
    )
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


@pytest.mark.postgresql
def test_purge_user_task_rows_deletes_an_active_interaction_row_postgres(
    postgres_session,
) -> None:
    session = postgres_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-active-bulk", status="active"
    )
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


def test_purge_user_task_rows_deletes_a_terminal_interaction_row_sqlite(
    sqlite_fk_on_session,
) -> None:
    """M4 regression control, bulk path: see the single-task terminal test
    above for why the count must be asserted directly."""
    session = sqlite_fk_on_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-terminal-bulk", status="terminated"
    )
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


@pytest.mark.postgresql
def test_purge_user_task_rows_deletes_a_terminal_interaction_row_postgres(
    postgres_session,
) -> None:
    session = postgres_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-terminal-bulk", status="terminated"
    )
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
    assert _interaction_row_count(session, task_id) == 0


def test_purge_user_task_rows_deletes_interaction_rows_before_trace_events_fk_on(
    sqlite_fk_on_session,
) -> None:
    session = sqlite_fk_on_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-order-bulk-fk-on", status="active"
    )
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    with _recorded_statements(session.get_bind()) as seen:
        _purge_user_task_rows(session, user_id=user_id)
    _assert_interaction_delete_between_pointer_update_and_trace_events_delete(seen)
    session.commit()


def test_purge_user_task_rows_deletes_interaction_rows_before_trace_events_upgraded(
    sqlite_upgraded_session,
) -> None:
    """Same rationale as the single-task upgraded-form test above: no
    DB-level enforcement makes a reversed ordering fail loudly here, so
    statement order is the only direct evidence."""
    session = sqlite_upgraded_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-order-bulk-upgraded", status="active"
    )
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    with _recorded_statements(session.get_bind()) as seen:
        _purge_user_task_rows(session, user_id=user_id)
    _assert_interaction_delete_between_pointer_update_and_trace_events_delete(seen)
    session.commit()


@pytest.mark.postgresql
def test_purge_user_task_rows_deletes_interaction_rows_before_trace_events_postgres(
    postgres_session,
) -> None:
    session = postgres_session
    task_id, _row_id = _seed_task_with_interaction_row(
        session, username="u-order-bulk-postgres", status="active"
    )
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    with _recorded_statements(session.get_bind()) as seen:
        _purge_user_task_rows(session, user_id=user_id)
    _assert_interaction_delete_between_pointer_update_and_trace_events_delete(seen)
    session.commit()


# ---------------------------------------------------------------------------
# Absent-table branch: both purge functions on a deployment upgraded to a
# revision before task_interaction_requests exists (PR-C2b, P4).
# ---------------------------------------------------------------------------


def test_purge_task_rows_succeeds_without_the_interaction_table_sqlite(
    sqlite_fk_on_session_without_interaction_table,
) -> None:
    session = sqlite_fk_on_session_without_interaction_table
    assert interaction_requests_table_exists(session) is False
    task_id = _seed_task_with_anchored_checkpoint(session, username="u-no-table-single")

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0


@pytest.mark.postgresql
def test_purge_task_rows_succeeds_without_the_interaction_table_postgres(
    postgres_session_without_interaction_table,
) -> None:
    session = postgres_session_without_interaction_table
    assert interaction_requests_table_exists(session) is False
    task_id = _seed_task_with_anchored_checkpoint(session, username="u-no-table-single")

    assert purge_task_rows(session, task_id=task_id) is True
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0


def test_purge_user_task_rows_succeeds_without_the_interaction_table_sqlite(
    sqlite_fk_on_session_without_interaction_table,
) -> None:
    session = sqlite_fk_on_session_without_interaction_table
    assert interaction_requests_table_exists(session) is False
    task_id = _seed_task_with_anchored_checkpoint(session, username="u-no-table-bulk")
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0


@pytest.mark.postgresql
def test_purge_user_task_rows_succeeds_without_the_interaction_table_postgres(
    postgres_session_without_interaction_table,
) -> None:
    session = postgres_session_without_interaction_table
    assert interaction_requests_table_exists(session) is False
    task_id = _seed_task_with_anchored_checkpoint(session, username="u-no-table-bulk")
    user_id = session.query(Task).filter(Task.id == task_id).one().user_id

    _purge_user_task_rows(session, user_id=user_id)
    session.commit()

    assert session.query(Task).filter(Task.id == task_id).count() == 0
