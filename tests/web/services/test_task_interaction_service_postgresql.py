"""PostgreSQL-only coverage for ``materialize_compatibility_view``'s two
backend-sensitive facts: the schema-presence gate (A5-P1) and the
four-field active-row predicate's tiering (A5-P2), both first established
as ad hoc probes against a disposable database during this delivery's own
fact audit and pinned here as real, repeatable assertions.

Fixture pattern follows ``test_interaction_staging_postgresql.py``: a
disposable, uniquely-named schema inside whatever database
``XAGENT_TEST_POSTGRES_URL`` points at, created and dropped per test run,
rather than the shared database itself -- see that file's own docstring
for why (a shared ``alembic_version`` row contended across worktrees).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from itertools import count

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_interaction_schema_shared import make_task, make_user
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_service as svc
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD

pytestmark = pytest.mark.postgresql

_key_counter = count()


@pytest.fixture()
def engine():
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    schema = "svc_task_interaction_" + uuid.uuid4().hex[:8]
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
def db_session(session_factory):
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_trace_event(db, *, task_id: int, run_partition: str = "run-a") -> int:
    event = TraceEvent(
        task_id=task_id,
        event_id=f"pg-trace-event-{next(_key_counter)}",
        event_type=str(CHECKPOINT_EVENT_TYPE),
        timestamp=_now(),
        build_id=None,
        data={
            TASK_RUN_ID_TRACE_FIELD: run_partition,
            "checkpoint_type": "agent_execution_checkpoint",
            "execution_id": "exec-1",
        },
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return int(event.id)


def _make_active_row(
    db,
    *,
    task_id: int,
    run_id: str,
    resume_trace_event_id: int,
    resume_run_partition: str,
) -> TaskInteractionRequest:
    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=1,
        status="active",
        active_slot=1,
        origin="internal",
        request_payload={
            "message": "Which environment?",
            "interactions": [
                {"type": "text_input", "field": "env", "label": "Environment"}
            ],
        },
        request_idempotency_key=f"pg-key-{next(_key_counter)}",
        resume_trace_event_id=resume_trace_event_id,
        resume_event_id="resume-event-1",
        resume_execution_id="exec-1",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=resume_run_partition,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _make_answered_row(db, *, task_id: int, run_id: str) -> None:
    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=1,
        status="answered",
        active_slot=None,
        origin="internal",
        request_payload={"message": "old", "interactions": []},
        response_payload={"env": "prod"},
        request_idempotency_key=f"pg-answered-key-{next(_key_counter)}",
        resume_trace_event_id=None,
        resume_event_id="resume-event-2",
        resume_execution_id="exec-2",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=run_id,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
        responder_identity="user:1",
        responded_at=now,
    )
    db.add(row)
    db.commit()


# ---------------------------------------------------------------------------
# A5-P1: the schema-presence gate, both states.
# ---------------------------------------------------------------------------


def test_table_exists_gate_before_and_after_create_all(engine) -> None:
    # Raw DDL, not Base.metadata.drop_all(tables=[...]): SQLAlchemy's
    # table-scoped drop_all still walks metadata-registered PostgreSQL enum
    # types and tries to drop those too, which fails here with
    # DependentObjectsStillExist because other tables in this schema (e.g.
    # agents.status) depend on the same enum type. A plain DROP TABLE has no
    # such side effect.
    with engine.begin() as conn:
        conn.execute(sa.text("DROP TABLE task_interaction_requests CASCADE"))

    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        view = svc.materialize_compatibility_view(db, 1)
        assert view.tier == "legacy"
    finally:
        db.close()

    Base.metadata.create_all(bind=engine, tables=[TaskInteractionRequest.__table__])
    db = session_factory()
    try:
        view = svc.materialize_compatibility_view(db, 1)
        assert view.tier == "legacy"  # table exists but no active row either
    finally:
        db.close()


# ---------------------------------------------------------------------------
# A5-P2: the four-field predicate's per-task tiering, three devices in one
# schema -- a live active row, a stale-run active row, and an answered row.
# ---------------------------------------------------------------------------


def test_active_row_predicate_three_way_tiering(db_session) -> None:
    db = db_session

    # Task 1: run matches, active row visible.
    user_id = make_user(db)
    task1 = make_task(db, user_id=user_id)
    t1 = db.query(Task).filter(Task.id == task1).first()
    t1.run_id = "run-a"
    db.commit()
    trace1 = _make_trace_event(db, task_id=task1, run_partition="run-a")
    _make_active_row(
        db,
        task_id=task1,
        run_id="run-a",
        resume_trace_event_id=trace1,
        resume_run_partition="run-a",
    )

    # Task 2: active row exists but was staged under a stale run.
    task2 = make_task(db, user_id=user_id)
    t2 = db.query(Task).filter(Task.id == task2).first()
    t2.run_id = "run-b"
    db.commit()
    trace2 = _make_trace_event(db, task_id=task2, run_partition="run-old")
    _make_active_row(
        db,
        task_id=task2,
        run_id="run-old",
        resume_trace_event_id=trace2,
        resume_run_partition="run-old",
    )

    # Task 3: only an answered row -- not active at all.
    task3 = make_task(db, user_id=user_id)
    t3 = db.query(Task).filter(Task.id == task3).first()
    t3.run_id = "run-c"
    db.commit()
    _make_answered_row(db, task_id=task3, run_id="run-c")

    view1 = svc.materialize_compatibility_view(db, task1)
    assert view1.tier == "native"

    view2 = svc.materialize_compatibility_view(db, task2)
    assert view2.tier == "legacy"

    view3 = svc.materialize_compatibility_view(db, task3)
    assert view3.tier == "legacy"


# ---------------------------------------------------------------------------
# TaskStatusPredicate compile-time assertion. Necessarily green today, not
# aspirationally green: the active-row query this delivery ships never
# references Task.status at all (see _active_native_row_criteria's own
# docstring for why -- "is the task WAITING_FOR_USER" is a concern the
# future answer fence adds, not part of "which row is the live one"), so
# there is no TaskStatus literal for this assertion to ever have caught in
# this delivery. It stays here anyway, not deleted, as the tripwire for the
# change that does add a Task.status conjunct to a query built from this
# same predicate (the answer fence, or the write-side reclaim statement):
# when either lands, this assertion must go on compiling their query too,
# and it must keep passing only because that new conjunct goes through
# TaskStatusPredicate rather than a bare TaskStatus member-name string. A
# future author who adds a literal instead of using TaskStatusPredicate
# should see this assertion turn red, not stay silently green because
# nobody pointed it at the new query.
# ---------------------------------------------------------------------------


def test_active_row_query_compiles_with_zero_taskstatus_literals(engine) -> None:
    stmt = (
        sa.select(TaskInteractionRequest)
        .join(Task, Task.id == TaskInteractionRequest.task_id)
        .where(
            TaskInteractionRequest.task_id == 1,
            *svc._active_native_row_criteria(),
        )
    )
    compiled = str(stmt.compile(bind=engine, compile_kwargs={"literal_binds": True}))
    for member in TaskStatus:
        assert member.name not in compiled
        assert member.name.lower() not in compiled
