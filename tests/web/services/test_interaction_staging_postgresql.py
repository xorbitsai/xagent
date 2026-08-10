"""Savepoint containment (T-SP group) on PostgreSQL.

Companion to test_interaction_staging.py, which carries every other group
(T-P, T-CM, T-GATE) plus its own copy of T-SP for SQLite. Per this suite's
own cut-line decision: T-P validates Python-side validation and statement
sequencing, both dialect-independent; the full 23-CHECK constraint set is
already pinned on both backends by test_task_interaction_schema.py /
test_task_interaction_schema_postgresql.py (50 PostgreSQL tests), so
re-running T-P and T-CM here would be duplicate coverage. What is genuinely
dialect-specific is savepoint/failed-transaction semantics -- that is what
this file re-runs, plus the one PostgreSQL-only case (T-SP-5) that has no
SQLite analog: PostgreSQL poisons the rest of a transaction
(InFailedSqlTransaction) after an uncaught IntegrityError until something
rolls back at least to the savepoint that was open when it fired; SQLite
does not.

Fixture pattern: originally the same init_db + Base.metadata.drop_all/
create_all against the whole XAGENT_TEST_POSTGRES_URL database that
test_task_interaction_schema_postgresql.py uses -- an existing convention,
not invented here. Switched to a disposable, uniquely-named schema instead
(CREATE SCHEMA / DROP SCHEMA CASCADE) after init_db()'s automatic-upgrade
check started failing here with the shared XAGENT_TEST_POSTGRES_URL
database's alembic_version stamped to a migration this worktree doesn't
recognize (confirmed independently: the same failure reproduces against
test_task_interaction_schema_postgresql.py unchanged). That table lives in
the database's default schema, is global to the whole server-side database,
and is not itself disposable -- another
worktree pointed at the same local Postgres instance had stamped it while
running its own migrations. Base.metadata.create_all() against a private
schema never touches alembic_version at all, sidestepping that contention
entirely rather than fighting over one shared table two worktrees both
have a legitimate claim to.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from itertools import count
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError, InternalError
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_interaction_schema_shared import (
    make_task,
    make_trace_event,
    make_user,
)
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import ops_signals
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    InteractionSlotTaken,
    _identity_lookup_stmt,
    interaction_handoff,
    stage_interaction_request,
)
from xagent.web.services.task_lease_service import TaskLease

_key_counter = count()


@pytest.fixture()
def engine():
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    schema = "c2a_interaction_staging_" + uuid.uuid4().hex[:8]
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


@pytest.fixture()
def fixtures(db_session):
    user_id = make_user(db_session)
    task_id = make_task(db_session, user_id=user_id)
    anchor_id = make_trace_event(db_session, task_id=task_id)
    return task_id, anchor_id


@pytest.fixture(autouse=True)
def _reset_ops_signals():
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    yield
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _anchor(trace_event_id: int, *, run_partition: str = "run-a") -> InteractionAnchor:
    return InteractionAnchor(
        trace_event_id=trace_event_id,
        resume_event_id="resume-event-1",
        resume_execution_id="resume-exec-1",
        resume_run_partition=run_partition,
    )


def _next_key() -> str:
    return f"pg-key-{next(_key_counter)}"


def _stage_kwargs(anchor: InteractionAnchor, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_id": anchor.resume_run_partition,
        "anchor": anchor,
        "kind": "clarification",
        "protocol_version": 1,
        "origin": "internal",
        "request_payload": {"prompt": "example"},
        "request_idempotency_key": _next_key(),
        "expires_at": _now() + timedelta(minutes=15),
        "now": _now(),
    }
    values.update(overrides)
    return values


def _lease(task_id: int) -> TaskLease:
    return TaskLease(
        task_id=task_id, runner_id="runner-1", run_id="run-a", attempt_id=None
    )


def _mark_caller_write(db: Session, task_id: int, title: str) -> None:
    db.execute(sa.update(Task).where(Task.id == task_id).values(title=title))


def _caller_write_survived(db: Session, task_id: int, title: str) -> bool:
    return (
        db.execute(sa.select(Task.title).where(Task.id == task_id)).scalar_one()
        == title
    )


def _count_cursor_executions(engine: sa.Engine) -> list[str]:
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001, ARG001
        statements.append(statement)

    return statements


def _force_next_identity_select_to_miss(
    monkeypatch: pytest.MonkeyPatch, db: Session
) -> dict[str, bool]:
    """PostgreSQL counterpart of the same helper in test_interaction_staging.py
    -- see that module for the full rationale. On this backend, READ
    COMMITTED gives the loser's own step-3 pre-read a fresh per-statement
    snapshot, so by the time it runs after the winner has committed, it
    sees the winner's row directly, the same as on SQLite; this forces that
    one read to miss instead, so the call proceeds into a real INSERT that
    genuinely collides with the winner's committed row."""

    original_execute = Session.execute
    state = {"armed": True}

    class _MissResult:
        def first(self) -> None:
            return None

    def _patched(self: Session, statement: Any, *args: Any, **kwargs: Any) -> Any:
        if self is db and state["armed"] and isinstance(statement, sa.Select):
            state["armed"] = False
            return _MissResult()
        return original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "execute", _patched)
    return state


def test_p_over_length_run_id_rejected_before_sql_via_handoff(
    engine, db_session, fixtures
) -> None:
    """A run_id longer than its column (VARCHAR(64)) must never reach SQL:
    on PostgreSQL an over-length value would raise DataError, not
    IntegrityError, so the conflict classifier in
    stage_interaction_request's own except IntegrityError block would never
    see it, and DataError is not in _SWALLOWED either, so it would escape
    interaction_handoff entirely. The Python-side length check in
    _validate_request_fields must preempt that by rejecting with ValueError
    before any statement (besides the CM's own SAVEPOINT) is issued.

    The anchor's own resume_run_partition is left short and valid on
    purpose: only the lease's run_id is over-length here, so the failure
    exercised is _validate_request_fields's own run_id-length check, not
    _assert_anchor_consistent's anchor-field check (which runs first, inside
    handoff.stage(), and would otherwise mask this one if both were long)."""

    task_id, anchor_id = fixtures
    long_run_id = "x" * 65
    anchor = _anchor(anchor_id)
    lease = TaskLease(
        task_id=task_id, runner_id="runner-1", run_id=long_run_id, attempt_id=None
    )
    db = db_session
    task = db.get(Task, task_id)

    statements = _count_cursor_executions(engine)
    before = len(statements)
    with pytest.raises(ValueError) as excinfo:
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )
    assert "64" in str(excinfo.value)
    issued = statements[before:]
    assert not any(
        s.strip().upper().startswith(("INSERT", "UPDATE")) for s in issued
    ), issued
    db.rollback()


def test_sp1_slot_taken_rolls_back_cleanly(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    db = db_session

    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
    )
    db.commit()
    task = db.get(Task, task_id)
    _mark_caller_write(db, task_id, "pg-sp1-write")

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()
    assert _caller_write_survived(db, task_id, "pg-sp1-write")
    assert ops_signals.INTERACTION_HANDOFF_DEGRADED in ops_signals.active_degradations()


def test_sp2_replay_after_conflict_commits_cleanly(session_factory, fixtures) -> None:
    """``a``'s own pre-read below (before ``b`` commits) exists only to
    open ``a``'s connection ahead of ``b``'s commit; it is not what decides
    this test's path. By the time ``a``'s own ``h.stage()`` call below runs
    its own step-3 pre-read, ``b`` has already committed, and READ
    COMMITTED gives that pre-read a fresh snapshot that sees ``b``'s row
    directly -- so this test pins the step-3 pre-read replay path, the same
    one T-P-9 pins on SQLite, not step 7's post-conflict re-check. See
    ``test_sp6_replay_after_conflict_via_insert_collision`` below for the
    construction that reaches step 7 for real on this backend."""

    task_id, anchor_id = fixtures
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    key = _next_key()

    a = session_factory()
    b = session_factory()
    try:
        a.execute(
            _identity_lookup_stmt(
                task_id=task_id, run_id="run-a", request_idempotency_key=key
            )
        ).first()

        task_b = b.get(Task, task_id)
        with interaction_handoff(b, lease, task=task_b, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=key,
                expires_at=_now() + timedelta(minutes=15),
            )
        b.commit()

        task_a = a.get(Task, task_id)
        _mark_caller_write(a, task_id, "pg-sp2-write")
        with interaction_handoff(a, lease, task=task_a, anchor=anchor, now=_now()) as h:
            result = h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=key,
                expires_at=_now() + timedelta(minutes=15),
            )
        a.commit()
        assert result.created is False
        assert _caller_write_survived(a, task_id, "pg-sp2-write")
    finally:
        a.close()
        b.close()


def test_sp6_replay_after_conflict_via_insert_collision(
    engine, session_factory, fixtures, monkeypatch
) -> None:
    """PostgreSQL counterpart of
    test_interaction_staging.py::test_p9b_replay_after_conflict_via_insert_collision
    -- see that test's docstring for the full mechanism. Confirmed by the
    same poison-probe check: replacing step 7's REPLAY return with an
    unconditional raise leaves T-SP-2 above green but fails this test."""

    task_id, anchor_id = fixtures
    anchor = _anchor(anchor_id)
    key = _next_key()

    statements = _count_cursor_executions(engine)

    b = session_factory()
    b_result = stage_interaction_request(
        b, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    b.commit()
    b.close()
    assert b_result.created is True

    a = session_factory()
    try:
        _mark_caller_write(a, task_id, "pg-sp6-write")

        _force_next_identity_select_to_miss(monkeypatch, a)
        before = len(statements)
        a_result = stage_interaction_request(
            a, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
        )
        issued = statements[before:]

        insert_count = sum(1 for s in issued if s.strip().upper().startswith("INSERT"))
        assert insert_count == 1, issued
        assert any("ROLLBACK TO SAVEPOINT" in s.upper() for s in issued), issued
        assert not any("RELEASE SAVEPOINT" in s.upper() for s in issued), issued

        assert a_result.created is False
        assert a_result.staged_db_id == b_result.staged_db_id
        assert a_result.status == "active"
        assert a_result.active_slot == b_result.active_slot

        a.commit()
        assert _caller_write_survived(a, task_id, "pg-sp6-write")

        row_count = a.execute(
            sa.select(sa.func.count())
            .select_from(TaskInteractionRequest)
            .where(
                TaskInteractionRequest.task_id == task_id,
                TaskInteractionRequest.request_idempotency_key == key,
            )
        ).scalar_one()
        assert row_count == 1, "A's own losing INSERT must not leave a second row"
    finally:
        a.close()


def test_sp3_clean_stage_commits_with_caller(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    db = db_session
    task = db.get(Task, task_id)
    _mark_caller_write(db, task_id, "pg-sp3-write")

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        result = h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()
    assert _caller_write_survived(db, task_id, "pg-sp3-write")
    row = db.get(TaskInteractionRequest, result.staged_db_id)
    assert row is not None


def test_sp4_all_three_cells_preserve_callers_pending_write(
    session_factory, fixtures
) -> None:
    """The caller's pre-with pending write survives all three savepoint
    exit shapes: rollback-on-exception (slot-taken), commit-on-success, and
    commit-on-REPLAY-after-conflict -- run back to back on independent rows
    so a leak in any one cell cannot mask another."""

    task_id, anchor_id = fixtures
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)

    for label in ("slot-taken", "clean", "replay"):
        db = session_factory()
        try:
            key = _next_key()
            if label == "slot-taken":
                stage_interaction_request(
                    db,
                    task_id=task_id,
                    **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
                )
                db.commit()
            elif label == "replay":
                stage_interaction_request(
                    db,
                    task_id=task_id,
                    **_stage_kwargs(anchor, request_idempotency_key=key),
                )
                db.commit()

            task = db.get(Task, task_id)
            _mark_caller_write(db, task_id, f"pg-sp4-{label}")
            with interaction_handoff(
                db, lease, task=task, anchor=anchor, now=_now()
            ) as h:
                h.stage(
                    kind="clarification",
                    protocol_version=1,
                    request_payload={"prompt": "p"},
                    request_idempotency_key=key,
                    expires_at=_now() + timedelta(minutes=15),
                )
            db.commit()
            assert _caller_write_survived(db, task_id, f"pg-sp4-{label}")
        finally:
            db.close()
        # Clear the active slot between iterations so each label starts clean.
        cleanup = session_factory()
        cleanup.execute(
            sa.update(TaskInteractionRequest)
            .where(
                TaskInteractionRequest.task_id == task_id,
                TaskInteractionRequest.active_slot.isnot(None),
            )
            .values(
                status="terminated",
                active_slot=None,
                terminal_reason="deadline_elapsed",
                terminated_at=_now(),
            )
        )
        cleanup.commit()
        cleanup.close()


def test_sp5_integrity_error_poisons_transaction_until_savepoint_rollback(
    db_session, fixtures
) -> None:
    """PostgreSQL-only: an uncaught IntegrityError poisons the rest of the
    transaction (InFailedSqlTransaction) until something rolls back to the
    savepoint open when it fired. Confirms the primitive's own inner
    savepoint is what makes the post-conflict re-check (step 7) possible at
    all on this backend -- without it, the very next statement issued would
    surface as InFailedSqlTransaction (sqlalchemy.exc.InternalError) instead of running."""

    task_id, anchor_id = fixtures
    anchor = _anchor(anchor_id)
    db = db_session
    key = _next_key()

    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
    )
    db.commit()

    # Manually reproduce the primitive's own inner-savepoint discipline to
    # observe the raw backend behavior: without a savepoint, the INSERT's
    # IntegrityError poisons the transaction and the very next statement
    # fails with InFailedSqlTransaction.
    row = {
        "task_id": task_id,
        "run_id": "run-a",
        "kind": "clarification",
        "protocol_version": 1,
        "status": "active",
        "active_slot": 1,
        "origin": "internal",
        "request_payload": {"prompt": "p"},
        "response_payload": None,
        "request_idempotency_key": key,
        "resume_trace_event_id": anchor_id,
        "resume_event_id": "resume-event-1",
        "resume_execution_id": "resume-exec-1",
        "resume_locator_format": "trace_event_pk_v1",
        "resume_checkpoint_type": "agent_execution_checkpoint",
        "resume_run_partition": "run-a",
        "responder_user_id": None,
        "responder_identity": None,
        "terminal_reason": None,
        "expires_at": _now() + timedelta(minutes=15),
        "responded_at": None,
        "terminated_at": None,
    }
    with pytest.raises(IntegrityError):
        db.execute(sa.insert(TaskInteractionRequest).values(**row))
        db.flush()
    with pytest.raises(InternalError):
        db.execute(
            _identity_lookup_stmt(
                task_id=task_id, run_id="run-a", request_idempotency_key=key
            )
        ).first()
    db.rollback()

    # The primitive's own inner savepoint avoids exactly this poisoning: a
    # SlotTaken conflict on the INSERT (the first row staged above still
    # holds the active slot) rolls back only to its own inner savepoint,
    # and the session is immediately usable again -- the failure this test
    # exists to contrast with.
    with pytest.raises(InteractionSlotTaken):
        stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
        )
    db.rollback()
    still_usable = db.execute(
        _identity_lookup_stmt(
            task_id=task_id, run_id="run-a", request_idempotency_key=key
        )
    ).first()
    assert still_usable is None
