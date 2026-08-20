"""PostgreSQL-only coverage for ``task_interaction_service``, for everything
in this delivery that a SQLite-backed test cannot actually exercise:

- ``materialize_compatibility_view``'s two backend-sensitive facts, first
  established as ad hoc probes against a disposable database during this
  delivery's own fact audit: the schema-presence gate (A5-P1) and the
  four-field active-row predicate's tiering (A5-P2).
- The answer fence's guest entity-binding JSON lookup, which compiles to
  different SQL on PostgreSQL (``->>``) than on SQLite
  (``json_extract``) and needs a real PostgreSQL server to prove the two
  backends agree. (The TaskStatus-bind-parameter structural guard this
  bullet used to also cover needs no database at all and lives in the
  sibling SQLite-backed file instead.)
- ``respond()``'s six fence-miss (rowcount=0) classifications -- four of
  them from a non-active interaction row (already answered, or terminated
  for one of three reasons), the other two from a row that is still
  ``active`` but whose *task* has moved on -- and the three answered-row
  CHECK constraints, run against a real PostgreSQL server rather than as a
  second copy of the SQLite-backed cell tests -- the point here is the
  backend, not new behavior coverage.
- Four genuinely concurrent tests, each opening a second real connection:
  the ownership-change TOCTOU window that PostgreSQL's row lock closes
  (which SQLite's single-writer model cannot reproduce), two
  ``respond()`` calls serializing on the same ``tasks`` row, and two
  lock-order deadlock regressions (``respond()`` racing a staging
  reclaim, and ``respond()`` racing a concurrent purge).
- One sequential (not concurrent) test running both
  ``respond()``-vs-``purge_task_rows`` orderings one after another --
  each ordering runs to completion and commits before the next
  statement starts, so nothing here actually contends on a lock -- and
  confirming neither ordering leaves residue.
- Step 8's ``classify_task_command_conflict`` RACED_DUPLICATE branch,
  reached only on PostgreSQL (SQLite's single-writer lock closes the
  window before this call reaches its own insert): two tests pause
  ``respond()`` right after its own existing-row check finds nothing and
  let a second, independent writer commit first, one with a matching
  payload (replayed) and one with a different one (conflict, whole
  transaction rolled back).

Fixture pattern follows ``test_interaction_staging_postgresql.py``: a
disposable, uniquely-named schema inside whatever database
``XAGENT_TEST_POSTGRES_URL`` points at, created and dropped per test run,
rather than the shared database itself -- see that file's own docstring
for why (a shared ``alembic_version`` row contended across worktrees).
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from itertools import count

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import xagent.web.models.database as database_module
from tests.web.services.task_interaction_schema_shared import (
    assert_rejected,
    make_row,
    make_task,
    make_user,
)
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_read as read_surface
from xagent.web.services import task_interaction_service as svc
from xagent.web.services.chat_history_service import persist_assistant_message
from xagent.web.services.interaction_rollout import counters_snapshot
from xagent.web.services.ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    clear_degradation,
)
from xagent.web.services.task_command_transport import (
    TaskCommandKind,
    enqueue_task_command,
)
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD

pytestmark = pytest.mark.postgresql


@pytest.fixture(autouse=True)
def _clean_degradation_registry():
    """The anchor resolver registers process-global degradation signals on
    its failure paths; clear this module's two signals around every test so
    they cannot leak into tests that read the shared registry."""
    for signal in (CHECKPOINT_PK_ANCHOR_DANGLING, CHECKPOINT_LOAD_UNAVAILABLE):
        clear_degradation(signal)
    yield
    for signal in (CHECKPOINT_PK_ANCHOR_DANGLING, CHECKPOINT_LOAD_UNAVAILABLE):
        clear_degradation(signal)


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
# The PostgreSQL half of the "no db.rollback()" regression test: the anchor
# fetch's except clause is a whitelist of transient infrastructure failures
# (OperationalError, InterfaceError, DisconnectionError, TimeoutError), and
# deliberately does not roll back on any of them. The SQLite half of this
# same cell lives in test_task_interaction_service.py; this is the
# real-backend counterpart the design calls for explicitly, since PostgreSQL
# is where server-side transaction state after a DBAPI failure differs from
# SQLite's. This test drives the except clause with a synthetic,
# monkeypatched exception -- it verifies the resolver's own behavior (no
# rollback, no swallowed session state), not what a genuine DBAPI failure
# does to the underlying PostgreSQL transaction; see the test's own
# docstring.
# ---------------------------------------------------------------------------


def test_the_session_survives_a_failed_anchor_fetch_with_no_rollback_postgresql(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies the resolver's own behavior at the anchor fetch: it does
    not call db.rollback() and does not swallow the caller's session state
    when the fetch raises. Driven with a synthetic, monkeypatched exception
    raised in Python before any statement reaches the DBAPI connection, so
    this is not a claim about what a genuine DBAPI-level failure does to
    the underlying PostgreSQL transaction -- that survival differs by
    backend (SQLite keeps the caller's staged writes; PostgreSQL discards
    them at the next commit once the server-side transaction is
    invalidated) and is documented at the call site, not proven here."""
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    db.commit()

    trace_id = _make_trace_event(db, task_id=task_id, run_partition="run-a")
    _make_active_row(
        db,
        task_id=task_id,
        run_id="run-a",
        resume_trace_event_id=trace_id,
        resume_run_partition="run-a",
    )

    # The caller's own staged, uncommitted write -- simulates a caller
    # that has already modified something in this same session before
    # asking the read surface for the pending question.
    task.title = "staged-before-the-failing-read"

    from sqlalchemy.orm import Session as OrmSession

    original_get = OrmSession.get
    raise_once = {"armed": True}

    def _raising_get(self, model, pk, *args, **kwargs):
        if model is TraceEvent and raise_once["armed"]:
            raise_once["armed"] = False
            raise sa.exc.OperationalError(
                "SELECT 1", {}, Exception("simulated session failure")
            )
        return original_get(self, model, pk, *args, **kwargs)

    monkeypatch.setattr(OrmSession, "get", _raising_get)
    view = svc.materialize_compatibility_view(db, task_id)

    assert view.tier == "unanswerable"
    assert view.reason == "checkpoint_unavailable"

    # (a) the caller's own pending write is still staged and committable --
    # a db.rollback() in the except clause would have discarded it, and on
    # PostgreSQL a real DBAPI-level error left uncaught in an open
    # transaction poisons every later statement until a rollback happens,
    # so this also proves the OperationalError raised above never reached
    # the actual DBAPI connection.
    db.commit()
    reloaded = db.query(Task).filter(Task.id == task_id).first()
    assert reloaded.title == "staged-before-the-failing-read"

    # (b) the same session can still run a plain query...
    still_readable = (
        db.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.task_id == task_id)
        .first()
    )
    assert still_readable is not None

    # ...and complete a whole second response construction.
    second_view = svc.materialize_compatibility_view(db, task_id)
    assert second_view.tier == "native"


# ---------------------------------------------------------------------------
# The read surface adapter's own double-backend cells (A6/A9/A13 from the
# adapter's SQLite-backed test file): the table-existence gate a T1 cell
# depends on, the four-field active-row predicate a T2 cell depends on, and
# a malformed JSON payload a T3''' cell depends on. The remaining twelve
# adapter cells are backend-independent and covered once, on SQLite, in
# test_task_interaction_read.py.
# ---------------------------------------------------------------------------


def test_a6_marker_matches_no_active_row_reads_the_legacy_transcript_postgresql(
    db_session,
) -> None:
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    task.interaction_protocol_version = 1
    db.commit()
    db.refresh(task)
    persist_assistant_message(
        db,
        task_id,
        user_id,
        "A live question",
        message_type="question",
        interactions=[{"type": "text_input", "label": "Live"}],
    )

    question, interactions = read_surface.get_pending_interaction_question(db, task)

    assert question is not None
    assert question.startswith("A live question")
    assert interactions == [{"type": "text_input", "label": "Live"}]


def test_a9_native_projection_postgresql(db_session) -> None:
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    task.interaction_protocol_version = 1
    db.commit()
    db.refresh(task)

    trace_id = _make_trace_event(db, task_id=task_id, run_partition="run-a")
    _make_active_row(
        db,
        task_id=task_id,
        run_id="run-a",
        resume_trace_event_id=trace_id,
        resume_run_partition="run-a",
    )

    question, interactions = read_surface.get_pending_interaction_question(db, task)

    assert question == "Which environment?"
    assert interactions == [
        {
            "type": "text_input",
            "field": "env",
            "label": "Environment",
            "options": None,
            "placeholder": None,
            "multiline": False,
            "min": None,
            "max": None,
            "default_value": None,
            "accept": None,
            "multiple": False,
        }
    ]


def test_a13_unparseable_payload_drops_both_slots_postgresql(db_session) -> None:
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    task.interaction_protocol_version = 1
    db.commit()
    db.refresh(task)

    trace_id = _make_trace_event(db, task_id=task_id, run_partition="run-a")
    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id="run-a",
        kind="clarification",
        protocol_version=1,
        status="active",
        active_slot=1,
        origin="internal",
        request_payload={"not": "a valid v1 payload"},
        request_idempotency_key=f"pg-malformed-key-{next(_key_counter)}",
        resume_trace_event_id=trace_id,
        resume_event_id="resume-event-1",
        resume_execution_id="exec-1",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition="run-a",
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()

    question, interactions = read_surface.get_pending_interaction_question(db, task)

    assert (question, interactions) == (None, None)


# ---------------------------------------------------------------------------
# The TaskStatusPredicate structural assertion (zero TaskStatus bind
# parameters in the active-row query) needs no database at all -- it lives
# in the sibling SQLite-backed file (test_task_interaction_service.py) so
# it runs by default instead of skipping whenever XAGENT_TEST_POSTGRES_URL
# is unset.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The guest entity-binding JSON term compiles differently on
# each backend (``->>`` on PostgreSQL, ``json_extract`` on SQLite -- see
# ``_answer_fence_task_predicate``'s own docstring) and must still agree on
# every case. The SQLite half of this same assertion lives in
# ``test_task_interaction_service.py``.
# ---------------------------------------------------------------------------


def test_answer_fence_guest_entity_binding_matches_on_postgresql(db_session) -> None:
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.agent_config = {"auth_mode": "widget", "guest_id": "guest-abc"}
    task.status = TaskStatus.WAITING_FOR_USER
    db.commit()

    def matches(guest_id: str) -> bool:
        principal = svc.InteractionPrincipal(
            kind="guest",
            user_id=user_id,
            is_admin=False,
            auth_mode="widget",
            guest_id=guest_id,
        )
        terms = svc._answer_fence_task_predicate(principal)
        stmt = sa.select(
            sa.exists(
                sa.select(sa.literal(1))
                .select_from(Task)
                .where(Task.id == task_id, *terms)
            )
        )
        return bool(db.execute(stmt).scalar())

    assert matches("guest-abc") is True
    assert matches("guest-WRONG") is False


# ---------------------------------------------------------------------------
# The six rowcount=0 classification cells, exercised through a real
# svc.respond() call against PostgreSQL, plus the CHECK-guarded answered-row
# shape.
# ---------------------------------------------------------------------------


@pytest.fixture()
def _respond_pg(monkeypatch: pytest.MonkeyPatch, session_factory):
    import xagent.web.models.database as database_module

    monkeypatch.setattr(database_module, "get_session_local", lambda: session_factory)
    return session_factory


def _pg_waiting_task(
    session_factory,
) -> tuple[int, int]:
    db = session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.WAITING_FOR_USER
        task.control_state = "waiting_for_user"
        task.run_id = "run-a"
        task.state_version = 5
        db.commit()
        return user_id, task_id
    finally:
        db.close()


def _pg_active_row(session_factory, *, task_id: int) -> int:
    db = session_factory()
    try:
        trace_event_id = _make_trace_event(db, task_id=task_id, run_partition="run-a")
        row = _make_active_row(
            db,
            task_id=task_id,
            run_id="run-a",
            resume_trace_event_id=trace_event_id,
            resume_run_partition="run-a",
        )
        return int(row.id)
    finally:
        db.close()


def _pg_owning_principal(user_id: int) -> svc.InteractionPrincipal:
    """The one principal shape every ``respond()`` call in this file uses
    except the guest binding test above, which needs a distinct
    ``kind``/``auth_mode``/``guest_id`` shape of its own."""

    return svc.InteractionPrincipal(
        kind="user", user_id=user_id, is_admin=False, auth_mode=None
    )


def _pg_envelope(idempotency_key: str) -> svc.RespondEnvelope:
    """Every ``RespondEnvelope`` in this file carries the same
    ``kind``/``protocol_version``/``values`` and differs only by
    ``idempotency_key``."""

    return svc.RespondEnvelope(
        kind="clarification",
        protocol_version=1,
        values={"env": "prod"},
        idempotency_key=idempotency_key,
    )


def test_answer_fence_rowcount_across_six_fence_miss_states(_respond_pg) -> None:
    """Each of the six ways the fence UPDATE can land on rowcount=0,
    exercised sequentially (one ``svc.respond()`` call after another, no
    concurrent writer) against a real PostgreSQL server rather than SQLite --
    the point of running this here instead of in the SQLite-backed sibling
    file is the backend, not concurrency. Only cases 1-4 below put the
    interaction row itself in a non-active state (answered, or terminated
    for one of three reasons); cases 5 and 6 leave the row ``active`` and
    move the *task* on instead (off ``WAITING_FOR_USER``, and onto a
    different run), which is what still lands the fence on rowcount=0 for
    those two. Each case runs through ``svc.respond()`` end to end rather
    than the fence statement in isolation. Actual concurrent-writer
    coverage lives in the dedicated tests below this one
    (``test_concurrent_ownership_change_is_blocked...``,
    ``test_two_concurrent_respond_calls_serialize...``, and the
    deadlock/residue tests), which do open a second connection."""

    # 1. Already answered, fresh key -> Conflict(already_answered).
    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    db = _respond_pg()
    try:
        row = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == interaction_id)
            .first()
        )
        row.status = "answered"
        row.active_slot = None
        row.response_payload = {"env": "prod"}
        row.responded_at = _now()
        row.responder_identity = _pg_owning_principal(user_id).identity_string()
        db.commit()
    finally:
        db.close()
    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=_pg_owning_principal(user_id),
        envelope=_pg_envelope("pg-already-answered"),
    )
    assert outcome == svc.RespondConflict(reason="already_answered")

    # 2/3/4. Terminated, three reasons.
    for terminal_reason, expected in (
        ("deadline_elapsed", "expired"),
        ("run_superseded", "run_superseded"),
        ("answered_via_legacy_resume", "answered_via_chat"),
    ):
        user_id, task_id = _pg_waiting_task(_respond_pg)
        interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
        db = _respond_pg()
        try:
            row = (
                db.query(TaskInteractionRequest)
                .filter(TaskInteractionRequest.id == interaction_id)
                .first()
            )
            row.status = "terminated"
            row.active_slot = None
            row.terminal_reason = terminal_reason
            row.terminated_at = _now()
            db.commit()
        finally:
            db.close()
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_pg_owning_principal(user_id),
            envelope=_pg_envelope(f"pg-terminated-{terminal_reason}"),
        )
        assert outcome == svc.RespondStale(reason=expected)

    # 5. Still active, task not WAITING.
    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    db = _respond_pg()
    try:
        db.query(Task).filter(Task.id == task_id).update({"status": TaskStatus.RUNNING})
        db.commit()
    finally:
        db.close()
    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=_pg_owning_principal(user_id),
        envelope=_pg_envelope("pg-not-waiting"),
    )
    assert outcome == svc.RespondStale(reason="run_ended")

    # 6. Still active, waiting, but a foreign run. Both helpers build on
    # "run-a", so the task alone is advanced afterwards -- neither carries
    # a run_id parameter.
    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    db = _respond_pg()
    try:
        db.query(Task).filter(Task.id == task_id).update({"run_id": "run-current"})
        db.commit()
    finally:
        db.close()
    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=_pg_owning_principal(user_id),
        envelope=_pg_envelope("pg-foreign-run"),
    )
    assert outcome == svc.RespondStale(reason="foreign_run")


# ---------------------------------------------------------------------------
# The answered-row shape is CHECK-guarded, exact constraint names.
# ---------------------------------------------------------------------------


def test_answered_row_cannot_also_carry_terminated_at(db_session) -> None:
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=None,
        status="answered",
        terminated_at=_now(),
    )
    assert_rejected(db, row, "ck_task_interaction_requests_terminated_at_pairs_status")


def test_answered_row_requires_responder_identity(db_session) -> None:
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=None,
        status="answered",
        responder_identity=None,
    )
    assert_rejected(
        db, row, "ck_task_interaction_requests_responder_pairs_responded_at"
    )


def test_answered_row_cannot_keep_its_active_slot(db_session) -> None:
    db = db_session
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=None,
        status="answered",
        active_slot=1,
    )
    assert_rejected(db, row, "ck_task_interaction_requests_active_slot_pairs_status")


# ---------------------------------------------------------------------------
# Concurrency: two real connections, real locks. The reverse assertion that
# the ownership predicate is redundant on PostgreSQL because the row lock
# closes the ownership-change TOCTOU window (see
# ``_answer_fence_task_predicate``'s own docstring), two concurrent
# respond() calls serializing on the tasks row, and the two
# respond()-vs-purge interleavings.
# ---------------------------------------------------------------------------


def test_concurrent_ownership_change_is_blocked_until_respond_commits(
    _respond_pg, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    owner_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    setup_db = _respond_pg()
    try:
        intruder_id = make_user(setup_db)
    finally:
        setup_db.close()

    results: dict = {}

    # Sleep injected between step 2's lock and the fence write, standing in
    # for step 3's authorization work, by wrapping the fence statement
    # builder. Patched via monkeypatch (not a manual assign-then-restore)
    # so the module-global attribute is restored at fixture teardown no
    # matter what happens on the holder thread below -- a raise before its
    # own restore code ran would otherwise leave every later test in this
    # process patched.
    real_fence_stmt = svc._answer_fence_stmt

    # The holder signals here rather than letting the racer guess with a
    # wall-clock stagger. This function runs after respond()'s step 2 has
    # taken the tasks row lock and before its fence write, so setting the
    # event here is the exact moment the racer's UPDATE must start blocking
    # from. With a fixed 0.1s stagger instead, the test could fail in both
    # directions on a loaded runner: a holder more than 0.1s late to reach
    # step 2 lets the racer's UPDATE through unblocked, and a racer late to
    # issue its UPDATE measures less than the full hold.
    lock_taken = threading.Event()

    def _slow_fence_stmt(*args, **kwargs):
        lock_taken.set()
        time.sleep(1.0)
        return real_fence_stmt(*args, **kwargs)

    monkeypatch.setattr(svc, "_answer_fence_stmt", _slow_fence_stmt)

    def holder() -> None:
        principal = _pg_owning_principal(owner_id)
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_pg_envelope("pg-a5-holder"),
        )
        results["holder_outcome"] = outcome

    def racer() -> None:
        assert lock_taken.wait(timeout=30.0), "holder never reached the fence"
        t0 = time.monotonic()
        db = _respond_pg()
        try:
            db.query(Task).filter(Task.id == task_id).update({"user_id": intruder_id})
            db.commit()
        finally:
            db.close()
        results["racer_wait_s"] = time.monotonic() - t0

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=racer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert isinstance(results["holder_outcome"], svc.RespondAccepted), results
    assert results["racer_wait_s"] >= 0.8, results


def test_two_concurrent_respond_calls_serialize_on_the_tasks_row(_respond_pg) -> None:
    """Two real connections answering the same row with different
    idempotency keys must never both win: step 2's ``FOR NO KEY UPDATE``
    row lock blocks the loser at its own step 2 until the winner's entire
    transaction -- fence UPDATE, Task CAS, staged command, and commit --
    has gone through, so the loser's own fence UPDATE only ever runs
    against an already-answered row. That ordering makes the loser's path
    deterministic, not merely "some Stale or Conflict": the winner's CAS
    (``apply_task_control_transition``, step 7) never touches
    ``Task.status``, only ``state_version``/``control_state``, so the
    loser's own fence still finds ``Task.status`` at ``WAITING_FOR_USER``
    and only the interaction row's own active-row criteria fail; nothing
    in this test reclaims the interaction's resume anchor between the two
    calls, so the loser's step 5.5 still resolves it; and the winner's
    idempotency key is not the loser's own, so the loser's reread finds no
    command staged under its key. Every one of those is what the loser's
    fence-miss classification (step 6) needs to land on
    ``Conflict(already_answered)`` specifically. The assertion below still
    only checks that exactly one call lands outside ``Accepted``, without
    pinning that specific reason -- the invariant this test exists to pin
    is "never both win", not the loser's exact outcome."""

    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    principal = _pg_owning_principal(user_id)

    outcomes: list = []

    def call(idempotency_key: str) -> None:
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_pg_envelope(idempotency_key),
        )
        outcomes.append(outcome)

    t1 = threading.Thread(target=call, args=("pg-concurrent-1",))
    t2 = threading.Thread(target=call, args=("pg-concurrent-2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    accepted = [o for o in outcomes if isinstance(o, svc.RespondAccepted)]
    stale_or_conflict = [
        o for o in outcomes if isinstance(o, (svc.RespondStale, svc.RespondConflict))
    ]
    assert len(accepted) == 1, outcomes
    assert len(stale_or_conflict) == 1, outcomes


def test_respond_vs_purge_both_interleavings_leave_no_residue(_respond_pg) -> None:
    from xagent.web.services.task_deletion import purge_task_rows

    for order in ("purge_first", "respond_first"):
        user_id, task_id = _pg_waiting_task(_respond_pg)
        interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
        principal = _pg_owning_principal(user_id)

        if order == "purge_first":
            db = _respond_pg()
            try:
                purge_task_rows(db, task_id=task_id)
                db.commit()
            finally:
                db.close()
            outcome = svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_pg_envelope(f"pg-{order}"),
            )
            assert outcome == svc.RespondUnavailable(reason="task_missing")
        else:
            outcome = svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_pg_envelope(f"pg-{order}"),
            )
            assert isinstance(outcome, svc.RespondAccepted)
            db = _respond_pg()
            try:
                purge_task_rows(db, task_id=task_id)
                db.commit()
            finally:
                db.close()

        verify_db = _respond_pg()
        try:
            remaining_task = verify_db.query(Task).filter(Task.id == task_id).count()
            remaining_ir = (
                verify_db.query(TaskInteractionRequest)
                .filter(TaskInteractionRequest.task_id == task_id)
                .count()
            )
            assert remaining_task == 0, order
            assert remaining_ir == 0, order
        finally:
            verify_db.close()


# ---------------------------------------------------------------------------
# The lock-strength mutation's real witness: respond() racing a staging-side
# reclaim. The reclaim UPDATE takes the interaction row first and its
# replacement INSERT then needs a KEY SHARE lock on the tasks parent row --
# a `FOR UPDATE` step 2 (instead of `FOR NO KEY UPDATE`) closes a cycle with
# that INSERT and deadlocks; a `FOR NO KEY UPDATE` step 2 does not, because
# `NO KEY UPDATE` and `KEY SHARE` do not conflict. This does not go through
# svc.respond() for the staging side -- the interaction staging primitive's
# own reclaim/replace sequence is reproduced directly in SQL, since
# exercising real production code for both sides is not required to pin
# the lock mechanics this guards.
# ---------------------------------------------------------------------------


def test_respond_vs_staging_reclaim_does_not_deadlock(_respond_pg, engine) -> None:
    import time

    import psycopg2.extras

    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    principal = _pg_owning_principal(user_id)
    setup_db = _respond_pg()
    try:
        replacement_trace_event_id = _make_trace_event(setup_db, task_id=task_id)
    finally:
        setup_db.close()

    # Through the same engine's connection pool's raw DBAPI connection,
    # rather than a fresh psycopg2.connect(...) built from the URL, so the
    # search_path (the disposable per-test schema) carries over identically
    # to every other query in this test module.
    def _raw_conn():
        raw = engine.raw_connection()
        raw.autocommit = False
        return raw

    results: dict = {}

    def reclaim_then_insert() -> None:
        conn = _raw_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE task_interaction_requests SET "
                "status = 'terminated', terminal_reason = 'run_superseded', "
                "active_slot = NULL, terminated_at = now(), updated_at = now() "
                "WHERE task_id = %s AND active_slot IS NOT NULL",
                (task_id,),
            )
            update_issued.set()
            time.sleep(0.3)
            cur.execute(
                "INSERT INTO task_interaction_requests ("
                "task_id, run_id, kind, protocol_version, status, active_slot, "
                "origin, request_payload, request_idempotency_key, "
                "resume_trace_event_id, resume_event_id, resume_execution_id, "
                "resume_locator_format, resume_checkpoint_type, "
                "resume_run_partition, created_at, expires_at"
                ") VALUES (%s, %s, 'clarification', 1, 'active', 1, 'internal', "
                "%s, %s, %s, 'resume-event-x', 'exec-x', 'trace_event_pk_v1', "
                "'agent_execution_checkpoint', %s, now(), now() + interval '15 minutes')",
                (
                    task_id,
                    "run-a",
                    psycopg2.extras.Json({"message": "new", "interactions": []}),
                    "pg-reclaim-key",
                    replacement_trace_event_id,
                    "run-a",
                ),
            )
            conn.commit()
            results["reclaim_outcome"] = "ok"
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            results["reclaim_outcome"] = f"error: {exc}"
        finally:
            conn.close()

    def respond_call() -> None:
        update_issued.wait(timeout=5)
        try:
            outcome = svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_pg_envelope("pg-reclaim-race"),
            )
            results["respond_outcome"] = outcome
        except Exception as exc:  # noqa: BLE001
            results["respond_outcome"] = f"error: {exc}"

    # Deterministic ordering, not a fixed sleep: respond() must not start
    # until the reclaim's UPDATE has genuinely been issued (even though
    # still uncommitted) on the reclaim thread's own connection, or the two
    # threads' actual DB statement order -- the fact this test exists to
    # pin -- would be left to OS scheduling.
    update_issued = threading.Event()
    t1 = threading.Thread(target=reclaim_then_insert)
    t2 = threading.Thread(target=respond_call)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results.get("reclaim_outcome") == "ok", results
    # The reclaim's UPDATE lands, uncommitted, before respond()'s own fence
    # UPDATE reaches the same row; respond() blocks on it (a plain row lock,
    # not the tasks lock this test is about), resumes once the reclaim
    # commits, and correctly reports the row as superseded rather than
    # timing out or raising -- the interesting fact this test pins is that
    # neither thread deadlocks getting there, not which of them "wins".
    assert results.get("respond_outcome") == svc.RespondStale(
        reason="run_superseded"
    ), results


# ---------------------------------------------------------------------------
# The lock-order mutation's real witness: respond() genuinely racing
# purge_task_rows on two threads, not the sequential two-orderings test
# above (which never contends on a lock at all -- each ordering there runs
# to completion and commits before the next statement starts). purge's own
# lock order is tasks-first, same as respond()'s: purge's checkpoint-
# pointer UPDATE takes the tasks row before its interaction-row DELETE runs.
# If respond()'s own order were ever reversed, this is the concrete cycle
# that would deadlock -- reproduced here with purge's two statements issued
# directly, since exercising purge_task_rows itself on a second thread has
# no seam to inject the delay this test needs between its two statements.
# ---------------------------------------------------------------------------


def test_respond_vs_concurrent_purge_does_not_deadlock(_respond_pg, engine) -> None:
    import time

    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    principal = _pg_owning_principal(user_id)

    update_issued = threading.Event()
    results: dict = {}

    def purge_sequence() -> None:
        conn = engine.raw_connection()
        conn.autocommit = False
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE tasks SET last_checkpoint_event_id = NULL, "
                "last_checkpoint_trace_event_id = NULL WHERE id = %s",
                (task_id,),
            )
            update_issued.set()
            time.sleep(0.3)
            cur.execute(
                "DELETE FROM task_interaction_requests WHERE task_id = %s", (task_id,)
            )
            # Not a full purge_task_rows reproduction (trace_events and the
            # tasks row itself are left alone) -- only the two statements
            # whose relative order determines whether this races respond()
            # into a lock cycle: the checkpoint-pointer UPDATE on tasks
            # above, and this DELETE on the interaction row.
            conn.commit()
            results["purge_outcome"] = "ok"
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            results["purge_outcome"] = f"error: {exc}"
        finally:
            conn.close()

    def respond_call() -> None:
        update_issued.wait(timeout=5)
        try:
            outcome = svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_pg_envelope("pg-purge-race"),
            )
            results["respond_outcome"] = outcome
        except Exception as exc:  # noqa: BLE001
            results["respond_outcome"] = f"error: {exc}"

    t1 = threading.Thread(target=purge_sequence)
    t2 = threading.Thread(target=respond_call)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results.get("purge_outcome") == "ok", results
    # respond()'s own step 2 blocks on purge's tasks-row UPDATE (same lock
    # order, so this is ordinary serialization, not a cycle); by the time it
    # unblocks, the interaction row is already gone, so respond() correctly
    # reports that rather than deadlocking or hanging.
    assert results.get("respond_outcome") == svc.RespondUnavailable(
        reason="interaction_missing"
    ), results


# ---------------------------------------------------------------------------
# classify_task_command_conflict's IntegrityError branch (step 8), reached
# from svc.respond() when stage_task_command's own insert -- inside this
# call's own ``db.begin_nested()`` -- collides with the
# ``uq_task_command_identity`` unique constraint. See
# test_task_interaction_service.py's own section by this name for the full
# reachability audit of all three TaskCommandConflictKind values against
# the real code: UNRELATED and TASK_MISSING are both provably unreachable
# through respond() on either backend (actor_user_id is always the task's
# own owner, which the fence's ownership predicate requires and which
# cannot be deleted out from under a task row this transaction still
# references; the task row itself cannot disappear before this call's own
# insert either). RACED_DUPLICATE is the only reachable one, and only here
# on PostgreSQL: a second writer must commit a row for the same
# (task_id, command_id) strictly between stage_task_command's own
# existing-row check and its own insert flush, a window SQLite's single
# whole-database writer lock closes (respond() has already written the
# fence UPDATE and the CAS on its own connection by the time it reaches
# step 8, so that connection already holds SQLite's one writer lock).
# PostgreSQL's row lock from step 2 is scoped to the specific ``tasks`` row
# alone, so a second writer can complete an insert into the unrelated
# ``task_execution_commands`` table while this transaction is still open.
#
# Both of RACED_DUPLICATE's sub-branches are pinned below -- they produce
# different RespondOutcomes and only one touches the response-conflict
# counter (the same counter the idempotency_key_reused cells in the
# SQLite-backed file exercise through the earlier, pre-existing-command
# shortcut at step 5; this is the same counter's other, previously
# untested, wiring point at step 8):
#
# - payload_matches=True: the second writer's row carries the identical
#   actor/kind/payload this call would itself have staged. respond()
#   replays the winning row (``RespondReplayed``) rather than treating the
#   loss as a failure, and does not touch the counter.
# - payload_matches=False: the second writer's row carries different
#   ``values``, so it canonicalizes to a different payload.
#   classify_task_command_conflict still finds it (it does not compare
#   payloads to decide whether a row raced -- only to decide what to do
#   once one has), and respond() reports ``RespondConflict`` and does
#   increment the counter, rolling back the whole transaction -- including
#   the fence UPDATE and the CAS this call had already issued -- since none
#   of that write was this call's own to keep.
# ---------------------------------------------------------------------------


def _pg_conflict_counter() -> int:
    return counters_snapshot().get(svc.COUNTER_LIFECYCLE_RESPONSE_CONFLICT, 0)


def _pause_command_add_session_factory(
    base_session_factory, *, precheck_done, release_event
):
    """A session factory wrapping ``base_session_factory`` whose returned
    session pauses the instant something calls ``.add()`` with a
    ``TaskExecutionCommand`` -- the exact point stage_task_command reaches
    right after its own existing-row check has already found nothing and
    right before the insert flush this test needs a second writer to beat.
    Only ``get_session_local`` is repointed at this wrapper, and only after
    every setup call already used the plain ``_respond_pg`` factory
    directly, so only respond()'s own internally-created session is
    instrumented -- the racing writer below opens its own, unpatched
    session straight off the base factory."""

    def factory():
        db = base_session_factory()
        real_add = db.add

        def paused_add(instance):
            if isinstance(instance, TaskExecutionCommand):
                precheck_done.set()
                assert release_event.wait(timeout=10)
            real_add(instance)

        db.add = paused_add
        return db

    return factory


def test_respond_races_a_committed_duplicate_command_and_replays_on_matching_payload(
    _respond_pg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second, independent writer commits a TaskExecutionCommand row for
    the same (task_id, command_id) with the identical actor/kind/payload
    while respond() is paused right after its own stage_task_command's
    existing-row check already found nothing. respond()'s own insert then
    collides on the unique constraint; classify_task_command_conflict
    reports RACED_DUPLICATE with payload_matches=True, and respond() commits
    and replays the winning row instead of treating this as a failure."""

    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    principal = _pg_owning_principal(user_id)
    envelope = _pg_envelope("pg-raced-duplicate-match")
    matching_payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values=envelope.values
    )

    precheck_done = threading.Event()
    release_respond = threading.Event()
    results: dict = {}

    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: _pause_command_add_session_factory(
            _respond_pg, precheck_done=precheck_done, release_event=release_respond
        ),
    )

    def call_respond() -> None:
        results["outcome"] = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=envelope,
        )

    t_respond = threading.Thread(target=call_respond)
    t_respond.start()
    assert precheck_done.wait(timeout=10)

    db_racer = _respond_pg()
    try:
        winner = enqueue_task_command(
            db_racer,
            task_id=task_id,
            actor_user_id=principal.user_id,
            command_id=envelope.idempotency_key,
            kind=TaskCommandKind.RESUME,
            payload=matching_payload,
        )
    finally:
        db_racer.close()

    before_counter = _pg_conflict_counter()
    release_respond.set()
    t_respond.join(timeout=10)

    outcome = results["outcome"]
    assert isinstance(outcome, svc.RespondReplayed), outcome
    assert outcome.receipt.command_db_id == winner.command_id
    assert _pg_conflict_counter() == before_counter

    verify_db = _respond_pg()
    try:
        commands = (
            verify_db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .all()
        )
        assert len(commands) == 1, commands
        assert commands[0].id == winner.command_id
        assert commands[0].actor_user_id == principal.user_id

        row = (
            verify_db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == interaction_id)
            .first()
        )
        assert row.status == "answered"
        assert row.responded_at is not None

        task = verify_db.query(Task).filter(Task.id == task_id).first()
        assert task.state_version == 6
    finally:
        verify_db.close()


def test_respond_races_a_committed_duplicate_command_and_conflicts_on_mismatched_payload(
    _respond_pg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same race as above, except the second writer's row carries different
    answer values -- a different canonicalized payload.
    classify_task_command_conflict still classifies this as
    RACED_DUPLICATE (it finds the row purely by (task_id, command_id), not
    by payload), but with payload_matches=False; respond() reports
    RespondConflict, increments the response-conflict counter, and rolls
    back the whole transaction -- undoing its own fence UPDATE and CAS,
    since none of that write is this call's to keep once its own command
    lost the race."""

    user_id, task_id = _pg_waiting_task(_respond_pg)
    interaction_id = _pg_active_row(_respond_pg, task_id=task_id)
    principal = _pg_owning_principal(user_id)
    envelope = _pg_envelope("pg-raced-duplicate-mismatch")
    mismatched_payload = svc._respond_command_payload(
        interaction_id=interaction_id,
        principal=principal,
        values={"env": "a-different-answer-than-this-call-submits"},
    )

    precheck_done = threading.Event()
    release_respond = threading.Event()
    results: dict = {}

    monkeypatch.setattr(
        database_module,
        "get_session_local",
        lambda: _pause_command_add_session_factory(
            _respond_pg, precheck_done=precheck_done, release_event=release_respond
        ),
    )

    def call_respond() -> None:
        results["outcome"] = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=envelope,
        )

    t_respond = threading.Thread(target=call_respond)
    t_respond.start()
    assert precheck_done.wait(timeout=10)

    db_racer = _respond_pg()
    try:
        winner = enqueue_task_command(
            db_racer,
            task_id=task_id,
            actor_user_id=principal.user_id,
            command_id=envelope.idempotency_key,
            kind=TaskCommandKind.RESUME,
            payload=mismatched_payload,
        )
    finally:
        db_racer.close()

    before_counter = _pg_conflict_counter()
    release_respond.set()
    t_respond.join(timeout=10)

    outcome = results["outcome"]
    assert outcome == svc.RespondConflict(reason="idempotency_key_reused"), outcome
    assert _pg_conflict_counter() == before_counter + 1

    verify_db = _respond_pg()
    try:
        commands = (
            verify_db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .all()
        )
        assert len(commands) == 1, commands
        assert commands[0].id == winner.command_id
        assert commands[0].payload == mismatched_payload

        row = (
            verify_db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == interaction_id)
            .first()
        )
        assert row.status == "active"
        assert row.active_slot == 1
        assert row.responded_at is None

        task = verify_db.query(Task).filter(Task.id == task_id).first()
        assert task.state_version == 5
        assert task.control_state == "waiting_for_user"
    finally:
        verify_db.close()
