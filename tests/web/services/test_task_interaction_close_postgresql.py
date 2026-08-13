"""The legacy resume close's rowcount grid, re-run on PostgreSQL.

Companion to test_task_interaction_close.py, which carries the full suite
(rowcount grid, table-absent no-op, NOT EXISTS guard, and the staging
interaction to prove the close is a real behavior change) against SQLite.
Every CHECK and unique constraint the rowcount grid depends on is already
pinned on both backends by test_task_interaction_schema.py /
test_task_interaction_schema_postgresql.py, so this file re-runs only the
rowcount grid itself -- the one group whose statement targets a real table
shape worth confirming on the production backend. Everything else in
test_task_interaction_close.py (table-absent no-op, the NOT EXISTS guard,
statement sequencing) is dialect-independent control flow, already covered
there.

Fixture pattern: a disposable, uniquely-named schema inside whatever
database XAGENT_TEST_POSTGRES_URL names (CREATE SCHEMA / DROP SCHEMA
CASCADE), matching test_interaction_staging_postgresql.py's convention --
see that file's docstring for why a schema instead of the whole database.

Known coverage gap, not closed here: neither this file nor
test_task_interaction_close.py ever calls
close_legacy_resume_interaction_sync -- the one function that issues the
FOR NO KEY UPDATE / key_share=True lock read the ordering obligation in
test_interaction_close_lock_ordering.py depends on -- against a real
PostgreSQL connection. test_task_interaction_close.py calls it on SQLite,
where with_for_update's clause is silently dropped and the single-writer
lock serializes instead, so it never issues the real statement either. No
test anywhere in this repo, on this database, opens two sessions and
proves FOR NO KEY UPDATE actually holds the lock the ordering argument
claims (blocking a concurrent purge, not blocking a concurrent KEY SHARE
stager) -- every case here runs one session against one fixture at a
time. The lock-ordering guard is therefore statement-order-in-source-code
evidence only, not lock-behavior evidence.
"""

from __future__ import annotations

import os
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_interaction_schema_shared import (
    make_row,
    make_trace_event,
    row_state,
    seed_active_row,
    seed_task_with_run,
    task_marker,
)
from xagent.web.models.database import Base
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services.task_interaction_close import close_legacy_resume_interaction

pytestmark = pytest.mark.postgresql


@pytest.fixture()
def engine():
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    schema = "legacy_resume_interaction_close_" + uuid.uuid4().hex[:8]
    admin_engine = sa.create_engine(url)
    with admin_engine.begin() as conn:
        conn.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    admin_engine.dispose()

    eng = sa.create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    Base.metadata.create_all(bind=eng)
    yield eng
    eng.dispose()

    admin_engine = sa.create_engine(url)
    with admin_engine.begin() as conn:
        conn.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
    admin_engine.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db(session_factory):
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_close_retires_the_active_row_for_its_own_run(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    row_id = seed_active_row(db, task_id=task_id, run_id="run-a")

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 1
    row = row_state(db, row_id)
    assert row.status == "terminated"
    assert row.active_slot is None
    assert row.terminal_reason == "answered_via_legacy_resume"
    assert row.terminated_at is not None
    assert task_marker(db, task_id) is None


def test_close_is_a_no_op_replaying_an_already_terminated_row(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
        )
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    row_id = int(row.id)
    original_terminal_reason = row.terminal_reason

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    row = row_state(db, row_id)
    assert row.status == "terminated"
    assert row.terminal_reason == original_terminal_reason
    assert task_marker(db, task_id) is None


def test_close_is_a_no_op_with_no_interaction_rows_at_all(db) -> None:
    """Today's 100% case: the table has no production writer yet."""
    task_id = seed_task_with_run(db, run_id="run-a", marker=None)

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    assert task_marker(db, task_id) is None


def test_close_does_not_touch_a_different_runs_active_row(db) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=1)
    orphan_row_id = seed_active_row(db, task_id=task_id, run_id="run-b")

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    orphan = row_state(db, orphan_row_id)
    assert orphan.status == "active"
    assert task_marker(db, task_id) is None


def test_close_does_not_overwrite_a_row_already_recycled_by_another_terminal_reason(
    db,
) -> None:
    task_id = seed_task_with_run(db, run_id="run-a", marker=None)
    anchor_id = make_trace_event(db, task_id=task_id)
    row = TaskInteractionRequest(
        **make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            run_id="run-a",
            status="terminated",
            terminal_reason="run_superseded",
        )
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    row_id = int(row.id)

    rowcount = close_legacy_resume_interaction(db, task_id=task_id, run_id="run-a")
    db.commit()

    assert rowcount == 0
    assert row_state(db, row_id).terminal_reason == "run_superseded"
