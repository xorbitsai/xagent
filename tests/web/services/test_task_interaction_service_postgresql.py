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
- The three answered-row CHECK constraints. (This build does not classify
  a rowcount=0 fence miss -- see ``RespondOutcomeUnknown``'s own
  docstring -- so the six non-active-row classification cells a more
  detailed build would need proven against a real PostgreSQL server are
  not delivered here either.)
- Four genuinely concurrent tests, each opening a second real connection:
  the ownership-change TOCTOU window that PostgreSQL's row lock closes
  (which SQLite's single-writer model cannot reproduce), two
  ``respond()`` calls serializing on the same ``tasks`` row, and two
  lock-order deadlock regressions (``respond()`` racing a staging
  reclaim, and ``respond()`` racing a concurrent purge) -- both
  deadlock regressions assert this build's own conservative outcome
  where their more detailed sibling would assert a specific reason.
- One sequential (not concurrent) test running both
  ``respond()``-vs-``purge_task_rows`` orderings one after another --
  each ordering runs to completion and commits before the next
  statement starts, so nothing here actually contends on a lock -- and
  confirming neither ordering leaves residue.

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

from tests.web.services.task_interaction_schema_shared import (
    assert_rejected,
    make_row,
    make_task,
    make_user,
)
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_service as svc
from xagent.web.services.ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    clear_degradation,
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
# Fixtures shared by every real svc.respond() call in this file, plus the
# CHECK-guarded answered-row shape. (The six rowcount=0 classification
# cells this build does not classify are not exercised here -- see this
# file's own module docstring.)
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
    idempotency keys must never both win: step 2's row lock serializes
    them, and the loser's fence UPDATE matches zero rows once the winner
    commits. This build does not classify why the loser's fence missed
    (see ``RespondOutcomeUnknown``'s own docstring) -- a more detailed
    build's loser would land on ``Stale`` or ``Conflict``, this build's
    lands on ``OutcomeUnknown`` -- so the loser is accepted here as any
    non-``Accepted`` outcome; the invariant this test exists to pin is
    "never both win", not which label the loser gets."""

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
        o
        for o in outcomes
        if isinstance(
            o, (svc.RespondStale, svc.RespondConflict, svc.RespondOutcomeUnknown)
        )
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
    # commits, and correctly reports the miss rather than timing out or
    # raising -- the interesting fact this test pins is that neither thread
    # deadlocks getting there, not which of them "wins". This build does
    # not classify why the fence missed (a more detailed build's sibling
    # asserts the specific ``Stale(run_superseded)`` this scenario produces
    # -- see ``RespondOutcomeUnknown``'s own docstring for why this build
    # reports the ambiguity instead).
    assert isinstance(results.get("respond_outcome"), svc.RespondOutcomeUnknown), (
        results
    )


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
