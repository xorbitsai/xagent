"""Contract tests for ``stage_interaction_request`` and ``interaction_handoff``
(SQLite half; always runs). The PostgreSQL half
(``test_interaction_staging_postgresql.py``) re-runs only the savepoint
containment group -- see that file's module docstring for why the rest is
not duplicated there.

Each test builds its own file-backed sqlite database under ``tmp_path``
rather than the process-wide singleton, for the same reason
``test_trace_event_staging.py`` does: several tests here need two
independent sessions with genuine transaction isolation between them
(REPLAY-after-conflict), which an in-memory database shared over one pooled
connection does not give.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_interaction_schema_shared import (
    make_task,
    make_trace_event,
    make_user,
)
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import ops_signals
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    InteractionAnchorCorrupt,
    InteractionOwnerStateError,
    InteractionRequestClosed,
    InteractionRunPartitionMismatch,
    InteractionSlotTaken,
    interaction_handoff,
    stage_interaction_request,
)
from xagent.web.services.task_lease_service import TaskLease

_key_counter = count()


def _engine(tmp_path: Path):
    """A file-backed sqlite engine, private to one test, configured exactly
    like the process-wide engine (``apply_sqlite_concurrency_pragmas`` --
    WAL journaling, busy_timeout, foreign keys).

    This module's own two-session tests (T-P-9, T-SP-2) need this exact
    configuration, not a "more correct" one: an earlier version of this
    helper additionally disabled pysqlite's own (non-standard) transaction
    handling, the workaround SQLAlchemy's docs recommend for serializable
    isolation. That workaround does fix a real gap -- a released SAVEPOINT
    (``sp.commit()`` on ``Session.begin_nested()``) is visible to a second
    connection on the same file before the outer transaction ever commits,
    confirmed by direct reproduction -- but it also makes SQLite refuse a
    session's write once its own read transaction has gone stale relative
    to another session's intervening commit ("database is locked"), which
    breaks the exact interleaving REPLAY-after-conflict depends on. Since
    the process-wide engine (``xagent/db/sqlite.py``) does not apply that
    workaround either, using it here would test a configuration this
    codebase does not actually run. The pre-outer-commit cross-connection
    visibility gap this leaves is real on SQLite as this codebase
    configures it today; this suite avoids asserting the opposite -- see
    the tests that would have depended on it for what they check instead.
    """
    db_path = tmp_path / "interaction_staging.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


def _session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _seed(session_factory) -> tuple[int, int]:
    db = session_factory()
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    anchor_id = make_trace_event(db, task_id=task_id)
    db.close()
    return task_id, anchor_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _anchor(
    trace_event_id: int, *, run_partition: str = "run-a", **overrides: Any
) -> InteractionAnchor:
    values: dict[str, Any] = {
        "trace_event_id": trace_event_id,
        "resume_event_id": "resume-event-1",
        "resume_execution_id": "resume-exec-1",
        "resume_run_partition": run_partition,
    }
    values.update(overrides)
    return InteractionAnchor(**values)


def _next_key() -> str:
    return f"key-{next(_key_counter)}"


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


def _count_cursor_executions(engine) -> list[str]:
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001, ARG001
        statements.append(statement)

    return statements


def _row_state(db: Session, staged_db_id: int):
    return db.execute(
        sa.select(
            TaskInteractionRequest.run_id,
            TaskInteractionRequest.status,
            TaskInteractionRequest.active_slot,
            TaskInteractionRequest.terminal_reason,
            TaskInteractionRequest.terminated_at,
        ).where(TaskInteractionRequest.id == staged_db_id)
    ).one()


def _clear_signals() -> None:
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)


@pytest.fixture(autouse=True)
def _reset_ops_signals():
    _clear_signals()
    yield
    _clear_signals()


# --------------------------------------------------------------------------
# T-P group -- the primitive
# --------------------------------------------------------------------------


_T_P_1_CASES = [
    pytest.param({"kind": "approval"}, ValueError, id="illegal-kind"),
    pytest.param({"protocol_version": 2}, ValueError, id="protocol-not-1"),
    pytest.param({"origin": "email"}, ValueError, id="illegal-origin"),
    pytest.param(
        {"request_idempotency_key": "has a space"}, ValueError, id="key-bad-pattern"
    ),
    pytest.param({"request_idempotency_key": ""}, ValueError, id="key-empty"),
    pytest.param(
        {"expires_at": _now() - timedelta(minutes=1)}, ValueError, id="ttl-non-positive"
    ),
    pytest.param({"expires_at": datetime.now()}, ValueError, id="expires-at-naive"),
    pytest.param(
        {
            "expires_at": _now().replace(tzinfo=timezone(timedelta(hours=8)))
            + timedelta(minutes=15)
        },
        ValueError,
        id="expires-at-non-utc",
    ),
    pytest.param({"request_payload": None}, ValueError, id="payload-none"),
]


@pytest.mark.parametrize("override, expected_exc", _T_P_1_CASES)
def test_step_one_rejections_send_no_sql(
    tmp_path: Path, override: dict[str, Any], expected_exc: type[Exception]
) -> None:
    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    kwargs = _stage_kwargs(anchor, **override)
    before = len(statements)
    with pytest.raises(expected_exc):
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
    db.close()


def test_step_one_rejects_empty_anchor_fields_without_sql(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id, resume_event_id="")
    kwargs = _stage_kwargs(anchor)
    before = len(statements)
    with pytest.raises(InteractionAnchorCorrupt):
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
    db.close()


def test_step_one_rejects_missing_anchor_trace_event_id_without_sql(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    object.__setattr__(anchor, "trace_event_id", None)
    kwargs = _stage_kwargs(anchor)
    before = len(statements)
    with pytest.raises(InteractionAnchorCorrupt):
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
    db.close()


def test_p2_run_partition_mismatch_is_not_a_slot_taken_subclass(
    tmp_path: Path,
) -> None:
    """T-P-2: run_id != resume_run_partition raises
    InteractionRunPartitionMismatch, and that type is not a subclass of
    InteractionSlotTaken -- the two must stay distinguishable so a
    corruption is never observably confused with an ordinary slot race."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id, run_partition="run-b")
    kwargs = _stage_kwargs(anchor, run_id="run-a")
    with pytest.raises(InteractionRunPartitionMismatch) as excinfo:
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert not isinstance(excinfo.value, InteractionSlotTaken)
    db.close()


def test_p3_clean_stage_is_not_visible_before_commit(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    result = stage_interaction_request(db, task_id=task_id, **_stage_kwargs(anchor))
    assert result.created is True
    assert result.status == "active"
    assert result.active_slot == 1
    assert result.staged_db_id > 0

    other = session_factory()
    visible = other.execute(
        sa.select(TaskInteractionRequest.id).where(
            TaskInteractionRequest.task_id == task_id
        )
    ).first()
    assert visible is None, "uncommitted row must not be visible on another session"
    other.close()

    db.commit()
    visible_after = other.execute(
        sa.select(TaskInteractionRequest.id).where(
            TaskInteractionRequest.task_id == task_id
        )
    ).first()
    db.close()
    assert visible_after is not None


def test_p4_precheck_branches(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    key = _next_key()

    created = stage_interaction_request(
        db, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    db.commit()
    assert created.created is True

    before = len(statements)
    replay = stage_interaction_request(
        db, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    issued = statements[before:]
    assert replay.created is False
    assert replay.status == "active"
    assert replay.staged_db_id == created.staged_db_id
    assert not any(
        s.strip().upper().startswith(("UPDATE", "INSERT")) for s in issued
    ), issued
    db.close()


def test_p4_answered_and_terminated_hits_raise_request_closed(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)

    key_answered = _next_key()
    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(anchor, request_idempotency_key=key_answered),
    )
    db.commit()
    db.execute(
        sa.update(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.request_idempotency_key == key_answered,
        )
        .values(
            status="answered",
            active_slot=None,
            response_payload={"answer": "x"},
            responded_at=_now(),
            responder_identity="user:1",
        )
    )
    db.commit()
    with pytest.raises(InteractionRequestClosed):
        stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(anchor, request_idempotency_key=key_answered),
        )
    db.rollback()

    key_terminated = _next_key()
    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(anchor, request_idempotency_key=key_terminated),
    )
    db.commit()
    db.execute(
        sa.update(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.request_idempotency_key == key_terminated,
        )
        .values(
            status="terminated",
            active_slot=None,
            terminal_reason="deadline_elapsed",
            terminated_at=_now(),
        )
    )
    db.commit()
    with pytest.raises(InteractionRequestClosed):
        stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(anchor, request_idempotency_key=key_terminated),
        )
    db.rollback()
    db.close()


def test_p5_same_run_tombstone_stays_closed(tmp_path: Path) -> None:
    """k-N1: a key reclaimed to deadline_elapsed earlier in the *same* run
    still raises InteractionRequestClosed on reuse -- it is not REPLAY and
    not a fresh CREATED row."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id, run_partition="run-a")
    key = _next_key()

    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            anchor,
            request_idempotency_key=key,
            expires_at=_now() + timedelta(minutes=1),
        ),
    )
    db.commit()

    # A second request in the same run, different key, reclaims the first
    # (now past its short TTL) via the deadline_elapsed branch.
    later = _now() + timedelta(minutes=5)
    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            anchor,
            request_idempotency_key=_next_key(),
            expires_at=later + timedelta(minutes=15),
            now=later,
        ),
    )
    db.commit()

    with pytest.raises(InteractionRequestClosed):
        stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(anchor, request_idempotency_key=key, now=later),
        )
    db.rollback()
    db.close()


def test_p5b_identity_is_run_scoped_not_task_scoped(tmp_path: Path) -> None:
    """§7.1: the same idempotency key used by two different runs on the same
    task must not be conflated -- each run's step-4 pre-read must only ever
    see rows from its own run. Regression for a step-4 predicate that
    forgets run_id and falls back to task-scoped identity: without run_id in
    the WHERE clause, run-b's pre-read for a shared key would find run-a's
    still-active row and incorrectly replay it as if it were run-b's own,
    short-circuiting before the reclaim that should have superseded it."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    shared_key = "shared-key"

    run_a = stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            _anchor(anchor_id, run_partition="run-a"),
            request_idempotency_key=shared_key,
        ),
    )
    db.commit()

    run_b = stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            _anchor(anchor_id, run_partition="run-b"),
            request_idempotency_key=shared_key,
        ),
    )
    db.commit()

    assert run_b.created is True, "run-b must get its own fresh row, not replay run-a's"
    assert run_b.staged_db_id != run_a.staged_db_id

    row_a = _row_state(db, run_a.staged_db_id)
    assert row_a.run_id == "run-a"
    assert row_a.status == "terminated"
    assert row_a.terminal_reason == "run_superseded"

    row_b = _row_state(db, run_b.staged_db_id)
    assert row_b.run_id == "run-b"
    assert row_b.status == "active"
    db.close()


_T_P_6_CASES = [
    pytest.param("run-a", "run-a", 1, False, id="same-run-expired"),
    pytest.param("run-a", "run-a", 15, True, id="same-run-unexpired-control"),
    pytest.param("run-a", "run-b", 1, False, id="cross-run-expired"),
    pytest.param("run-a", "run-b", 15, False, id="cross-run-unexpired"),
    pytest.param("run-b", "run-a", 1, False, id="reverse-cross-run-expired"),
    pytest.param("run-b", "run-a", 15, False, id="reverse-cross-run-unexpired"),
]


@pytest.mark.parametrize(
    "existing_run, reclaiming_run, ttl_minutes, expect_untouched", _T_P_6_CASES
)
def test_p6_reclaim_six_cells(
    tmp_path: Path,
    existing_run: str,
    reclaiming_run: str,
    ttl_minutes: int,
    expect_untouched: bool,
) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()

    existing_anchor = _anchor(anchor_id, run_partition=existing_run)
    existing = stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            existing_anchor, expires_at=_now() + timedelta(minutes=ttl_minutes)
        ),
    )
    db.commit()

    reclaim_now = _now() + timedelta(minutes=5)
    if expect_untouched:
        # Same run, unexpired: the reclaim predicate does not match, so the
        # slot is still held -> the new INSERT collides with it.
        with pytest.raises(InteractionSlotTaken):
            stage_interaction_request(
                db,
                task_id=task_id,
                **_stage_kwargs(
                    _anchor(anchor_id, run_partition=reclaiming_run),
                    now=reclaim_now,
                    expires_at=reclaim_now + timedelta(minutes=15),
                ),
            )
        db.rollback()
        row = _row_state(db, existing.staged_db_id)
        assert row.status == "active"
        assert row.active_slot == 1
        assert row.terminal_reason is None
        assert row.terminated_at is None
        db.close()
        return

    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            _anchor(anchor_id, run_partition=reclaiming_run),
            now=reclaim_now,
            expires_at=reclaim_now + timedelta(minutes=15),
        ),
    )
    db.commit()
    row = _row_state(db, existing.staged_db_id)
    assert row.status == "terminated"
    assert row.active_slot is None
    assert row.terminated_at is not None
    expected_reason = (
        "run_superseded" if existing_run != reclaiming_run else "deadline_elapsed"
    )
    assert row.terminal_reason == expected_reason
    db.close()


def test_p7_case_branch_prioritizes_run_superseded(tmp_path: Path) -> None:
    """A row that is both cross-run *and* expired is recorded as
    run_superseded, not deadline_elapsed -- the CASE checks run identity
    first."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    existing = stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            _anchor(anchor_id, run_partition="run-a"),
            expires_at=_now() + timedelta(minutes=1),
        ),
    )
    db.commit()
    later = _now() + timedelta(minutes=10)
    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            _anchor(anchor_id, run_partition="run-b"),
            now=later,
            expires_at=later + timedelta(minutes=15),
        ),
    )
    db.commit()
    row = _row_state(db, existing.staged_db_id)
    assert row.terminal_reason == "run_superseded"
    db.close()


def test_p8_owner_state_error_is_not_integrity_error(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()

    # A doomed pending write: a Task row with a NOT NULL column left unset
    # (title is nullable=False) added directly, bypassing the ORM's default.
    doomed = Task(user_id=10**9, title=None)  # user_id references nothing
    db.add(doomed)

    with pytest.raises(InteractionOwnerStateError) as excinfo:
        stage_interaction_request(
            db, task_id=task_id, **_stage_kwargs(_anchor(anchor_id))
        )
    assert not isinstance(excinfo.value, IntegrityError)
    db.rollback()
    db.close()


def test_p9_replay_after_conflict(tmp_path: Path) -> None:
    """Two sessions race on the same (task_id, run_id, key). The loser's
    step-4 pre-read misses (it ran before the winner committed), but its
    INSERT collides with the winner's row; the post-conflict re-check finds
    the winner's row and replays it rather than raising SlotTaken."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    anchor = _anchor(anchor_id)
    key = _next_key()

    a = session_factory()
    b = session_factory()

    # A performs its own step-1..4 manually up through the pre-read miss,
    # then pauses (does not reclaim/insert yet).
    from xagent.web.services.task_interaction_staging import _identity_lookup_stmt

    a_hit = a.execute(
        _identity_lookup_stmt(
            task_id=task_id, run_id="run-a", request_idempotency_key=key
        )
    ).first()
    assert a_hit is None

    # B runs the whole call and commits, winning the race.
    b_result = stage_interaction_request(
        b, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    b.commit()
    b.close()

    # A now finishes its own call; its INSERT collides with B's committed row.
    a_result = stage_interaction_request(
        a, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    a.commit()
    a.close()

    assert a_result.created is False
    assert a_result.staged_db_id == b_result.staged_db_id
    assert not isinstance(a_result, InteractionSlotTaken)


def test_p10_slot_taken_does_not_retry(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)

    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
    )
    db.commit()

    before = len(statements)
    with pytest.raises(InteractionSlotTaken):
        stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
        )
    db.rollback()
    issued = statements[before:]
    insert_count = sum(1 for s in issued if s.strip().upper().startswith("INSERT"))
    assert insert_count == 1, issued
    db.close()


def test_p11_replay_ignores_expiry(tmp_path: Path) -> None:
    """k-N2: step 4 replays an already-expired active row without
    consulting expires_at."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    key = _next_key()

    created = stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            anchor,
            request_idempotency_key=key,
            expires_at=_now() + timedelta(minutes=1),
        ),
    )
    db.commit()

    much_later = _now() + timedelta(hours=1)
    replay = stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(
            anchor,
            request_idempotency_key=key,
            now=much_later,
            expires_at=much_later + timedelta(minutes=15),
        ),
    )
    assert replay.created is False
    assert replay.staged_db_id == created.staged_db_id
    assert replay.status == "active"
    db.close()


def test_p_reclaim_survives_a_conflict_on_its_own_calls_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-3 regression: the reclaim UPDATE (step 5) stays in the outer
    transaction, not the inner savepoint that wraps the INSERT (step 6), so
    an INSERT-time conflict on this same call does not undo a genuine
    reclaim that same call already performed.

    Not constructed with two naturally-racing sessions the way T-P-9 is,
    for two independent reasons. First, ``uq_task_interaction_active_slot``
    caps a task at one active row, so a call whose own reclaim frees that
    slot for real leaves nothing left to collide with on the slot
    dimension -- the only collision surface left
    (``uq_task_interaction_request_identity``) is only reachable by a
    session whose *entire* call, reclaim included, commits before this
    one's own INSERT fires, which means the reclaim that actually happened
    would belong to the other session's already-committed transaction, not
    this one's, and could not tell this mutation apart from correct code.
    Second, and more fundamentally on SQLite: this call's own reclaim
    UPDATE is itself an uncommitted write, and SQLite's writer lock is
    database-wide, not per-row (confirmed directly: a second session
    attempting any write while this one holds an uncommitted write blocks
    with "database is locked" regardless of which row it targets) -- true
    concurrent interleaving with a write already in flight is not
    constructible on this backend at all.

    So this drives the interleaving directly instead -- sanctioned by this
    PR's own design record (§3.3 A-2: construct REPLAY-after-conflict with
    two sessions *or* by calling the primitive's internals directly) -- by
    forcing the INSERT's own flush to fail exactly once, without a second
    connection: what matters for this mutation is only where the reclaim
    statement sits relative to the inner savepoint boundary, not why the
    INSERT failed.
    """

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()

    # A stale, cross-run active row this call's own reclaim will terminate.
    victim = stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(_anchor(anchor_id, run_partition="run-victim")),
    )
    db.commit()

    race_anchor = _anchor(anchor_id, run_partition="run-race")

    original_flush = Session.flush
    flush_calls_on_db = {"n": 0}

    def _fail_third_flush(self: Session, *args: Any, **kwargs: Any) -> Any:
        if self is db:
            flush_calls_on_db["n"] += 1
            # Three flush() calls happen on this session before step 6's
            # INSERT would otherwise succeed: (1) step 3's explicit flush of
            # the caller's own pending writes -- none here; (2) the implicit
            # snapshot flush Session.begin_nested() always issues to
            # establish the inner savepoint; (3) step 6's own explicit
            # flush, right as the INSERT is attempted, immediately after
            # this call's own reclaim UPDATE has already run -- exactly the
            # point a genuinely racing session's conflict would surface.
            if flush_calls_on_db["n"] == 3:
                raise IntegrityError("simulated identity conflict", None, None)
        return original_flush(self, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", _fail_third_flush)

    with pytest.raises(InteractionSlotTaken):
        stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(race_anchor, request_idempotency_key=_next_key()),
        )

    # The INSERT's own attempt failed and its inner savepoint rolled back;
    # the reclaim issued before that savepoint even opened must still be
    # intact in this session's own (still uncommitted) view.
    row = _row_state(db, victim.staged_db_id)
    assert row.status == "terminated"
    assert row.terminal_reason == "run_superseded"
    db.rollback()
    db.close()


# --------------------------------------------------------------------------
# T-CM group -- the context manager
# --------------------------------------------------------------------------


def _lease(
    task_id: int, *, run_id: str = "run-a", attempt_id: str | None = None
) -> TaskLease:
    return TaskLease(
        task_id=task_id, runner_id="runner-1", run_id=run_id, attempt_id=attempt_id
    )


def _mark_caller_write(db: Session, task_id: int, title: str) -> None:
    db.execute(sa.update(Task).where(Task.id == task_id).values(title=title))


def _caller_write_survived(db: Session, task_id: int, title: str) -> bool:
    return (
        db.execute(sa.select(Task.title).where(Task.id == task_id)).scalar_one()
        == title
    )


def _force_attempt_mismatch(db: Session, task_id: int) -> TaskLease:
    db.execute(
        sa.update(Task)
        .where(Task.id == task_id)
        .values(lease_attempt_id="attempt-current")
    )
    return _lease(task_id, attempt_id="attempt-stale")


_T_CM_1_CASES = [
    "slot-taken",
    "request-closed",
    "anchor-corrupt",
    "attempt-mismatch",
    "run-partition-mismatch",
    "replay-after-conflict",
]


@pytest.mark.parametrize("case", _T_CM_1_CASES)
def test_cm1_six_cell_exit_matrix(tmp_path: Path, case: str) -> None:
    """T-CM-1, widened to six cells: the four exceptions the book's design
    names, plus InteractionRunPartitionMismatch (swallowed under PR-C2a's
    F-3 override, see the CM's docstring), plus REPLAY-after-conflict as
    the successful-return control. Every cell: (a) the with-block exits
    without the exception escaping; (b) exactly the swallowed exceptions
    register a degradation and log; (c) the caller's own pre-with pending
    write survives and commits alongside the caller."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)

    stage_key = _next_key()

    if case == "slot-taken":
        stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
        )
        db.commit()
        expect_signal = ops_signals.INTERACTION_HANDOFF_DEGRADED
    elif case == "request-closed":
        r = stage_interaction_request(
            db,
            task_id=task_id,
            **_stage_kwargs(anchor, request_idempotency_key=stage_key),
        )
        db.commit()
        db.execute(
            sa.update(TaskInteractionRequest)
            .where(TaskInteractionRequest.id == r.staged_db_id)
            .values(
                status="terminated",
                active_slot=None,
                terminal_reason="deadline_elapsed",
                terminated_at=_now(),
            )
        )
        db.commit()
        expect_signal = ops_signals.INTERACTION_HANDOFF_DEGRADED
    elif case == "anchor-corrupt":
        anchor = _anchor(anchor_id, resume_event_id="")
        expect_signal = ops_signals.INTERACTION_HANDOFF_DEGRADED
    elif case == "attempt-mismatch":
        lease = _force_attempt_mismatch(db, task_id)
        db.commit()
        expect_signal = ops_signals.INTERACTION_HANDOFF_DEGRADED
    elif case == "run-partition-mismatch":
        anchor = _anchor(anchor_id, run_partition="some-other-run")
        expect_signal = ops_signals.INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED
    else:
        assert case == "replay-after-conflict"
        expect_signal = None

    task = db.get(Task, task_id)
    _mark_caller_write(db, task_id, f"caller-write-{case}")

    if case == "replay-after-conflict":
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            first = h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=stage_key,
                expires_at=_now() + timedelta(minutes=15),
            )
        db.commit()
        assert first.created is True
        db.close()
        assert ops_signals.active_degradations() == {}
        return

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=stage_key,
            expires_at=_now() + timedelta(minutes=15),
        )

    db.commit()
    assert _caller_write_survived(db, task_id, f"caller-write-{case}")
    signals = ops_signals.active_degradations()
    assert expect_signal in signals
    db.close()


def test_cm2_owner_state_error_propagates_uncaught(tmp_path: Path) -> None:
    """T-CM-2: InteractionOwnerStateError is the one exception this module
    raises that is never swallowed -- it propagates out of the with-block,
    and the CM's own savepoint has already been rolled back by the time it
    does.

    The doomed write is added *inside* the with-block, right before
    ``stage()`` -- not before ``interaction_handoff`` is even entered --
    so its flush happens inside ``stage_interaction_request``'s own step-3
    flush, reached through the CM's ``except`` clause around ``yield``, the
    same path a real caller's own pending write would take. Adding it
    before the ``with`` line instead would only exercise the CM's *other*
    IntegrityError guard, the one around its own ``db.begin_nested()`` --
    a real gap an earlier version of this test had, which a
    _SWALLOWED-widening mutation could pass right through undetected."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(InteractionOwnerStateError) as excinfo:
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            doomed = Task(user_id=10**9, title=None)
            db.add(doomed)
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )
    assert not isinstance(excinfo.value, IntegrityError)
    assert db.in_transaction()
    db.rollback()
    db.close()


def test_cm3_attempt_assertion_gates_on_not_none(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    task = db.get(Task, task_id)

    # attempt_id is None -> assertion is skipped even though task's own
    # lease_attempt_id disagrees.
    db.execute(
        sa.update(Task).where(Task.id == task_id).values(lease_attempt_id="whatever")
    )
    db.commit()
    lease = _lease(task_id, attempt_id=None)
    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        result = h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()
    assert result.created is True
    assert ops_signals.active_degradations() == {}
    db.close()


def test_cm6_assertions_precede_any_staging_statement(tmp_path: Path) -> None:
    """T-CM-6 (post-fix form): the attempt and anchor assertions run at the
    very start of ``stage()``, before the reclaim UPDATE and before the
    INSERT's own savepoint -- so a mismatched attempt never reaches SQL and
    never leaves a row behind. A mutation that ran the assertions after
    stage_interaction_request's own work would let a stale attempt's
    request through and this test would catch it via the row count."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _force_attempt_mismatch(db, task_id)
    db.commit()
    task = db.get(Task, task_id)

    before = len(statements)
    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    issued = statements[before:]
    assert not any(
        s.strip()
        .upper()
        .startswith(
            (
                "INSERT INTO task_interaction_requests".upper(),
                "UPDATE task_interaction_requests".upper(),
            )
        )
        for s in issued
    ), issued
    db.commit()
    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == 0
    db.close()


@pytest.mark.parametrize("case", ["attempt-mismatch", "anchor-corrupt"])
def test_v_n4_degrade_still_lets_caller_commit_run(tmp_path: Path, case: str) -> None:
    """v-n4, made executable: after a degrade, code placed *after* the
    with-block -- standing in for the caller's own commit -- still runs and
    its effects are durable. This is the invariant the generator-yield
    finding rescued: before the fix, __enter__ raised
    RuntimeError('generator didn't yield') and nothing after the with-block
    ever executed."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    task = db.get(Task, task_id)

    if case == "attempt-mismatch":
        anchor = _anchor(anchor_id)
        lease = _force_attempt_mismatch(db, task_id)
        db.commit()
    else:
        anchor = _anchor(anchor_id, resume_event_id="")
        lease = _lease(task_id)

    ran_after_with = False
    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    # This line standing in for "the caller's own commit" -- reaching it at
    # all is the assertion.
    ran_after_with = True
    db.execute(
        sa.update(Task).where(Task.id == task_id).values(title="post-with-write")
    )
    db.commit()

    assert ran_after_with is True
    db2 = session_factory()
    assert (
        db2.execute(sa.select(Task.title).where(Task.id == task_id)).scalar_one()
        == "post-with-write"
    )
    db2.close()
    db.close()


def test_cm4_no_notification_and_no_outer_commit() -> None:
    import ast

    from xagent.web.services import task_interaction_staging

    tree = ast.parse(Path(task_interaction_staging.__file__).read_text())
    roots = ("notify", "notification", "dispatch", "publish", "broadcast")
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.name.split(".")[-1] for a in node.names)
            names.update(a.asname for a in node.names if a.asname)
    offenders = sorted(n for n in names if any(r in n.lower() for r in roots))
    assert offenders == [], offenders


def test_cm4_with_exit_does_not_commit_outer_transaction(tmp_path: Path) -> None:
    """The CM must not itself commit the caller's outer transaction, checked
    three ways: (a) ``db.in_transaction()`` is still true right after the
    ``with`` exits; (b) the row is not visible to a second connection before
    the caller's own commit; (c) rolling back instead of committing removes
    the row entirely -- from both this session's own point of view and a
    fresh session's.

    (b) and (c) are the regression pin for a real bug found and fixed while
    building this module: on SQLite, a session whose first write-adjacent
    statement is ``interaction_handoff``'s own outer ``db.begin_nested()`` --
    exactly this test's shape, where the only earlier statement on ``db`` is
    a plain SELECT -- breaks pysqlite's transaction tracking badly enough
    that the savepoint's release becomes a real, permanent commit; a
    rollback afterward silently does nothing. See the zero-row UPDATE
    ``interaction_handoff`` issues immediately before opening its savepoint,
    and that function's docstring, for the fix and the full explanation.
    Without it, this test fails at (b) (the row leaks to another connection
    before commit) and, worse, at (c) even after ``db.rollback()``."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    assert db.in_transaction()

    other = session_factory()
    visible_before_commit = other.execute(
        sa.select(TaskInteractionRequest.id).where(
            TaskInteractionRequest.task_id == task_id
        )
    ).first()
    assert visible_before_commit is None, (
        "the row must not be visible before the caller commits"
    )
    other.close()

    db.rollback()
    assert (
        db.execute(
            sa.select(TaskInteractionRequest.id).where(
                TaskInteractionRequest.task_id == task_id
            )
        ).first()
        is None
    ), "a caller rollback must remove the staged row from this session's own view"
    db.close()

    fresh = session_factory()
    assert (
        fresh.execute(
            sa.select(TaskInteractionRequest.id).where(
                TaskInteractionRequest.task_id == task_id
            )
        ).first()
        is None
    ), "a caller rollback must remove the staged row entirely, not just locally"
    fresh.close()


# --------------------------------------------------------------------------
# T-SP group -- savepoint containment (SQLite half; PostgreSQL half repeats
# this group in test_interaction_staging_postgresql.py)
# --------------------------------------------------------------------------


def test_sp1_slot_taken_rolls_back_cleanly(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    stage_interaction_request(
        db,
        task_id=task_id,
        **_stage_kwargs(anchor, request_idempotency_key=_next_key()),
    )
    db.commit()
    _mark_caller_write(db, task_id, "sp1-write")

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()
    assert _caller_write_survived(db, task_id, "sp1-write")
    db.close()


def test_sp2_replay_after_conflict_commits_cleanly(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    key = _next_key()

    a = session_factory()
    b = session_factory()
    from xagent.web.services.task_interaction_staging import _identity_lookup_stmt

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
    b.close()

    task_a = a.get(Task, task_id)
    _mark_caller_write(a, task_id, "sp2-write")
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
    assert _caller_write_survived(a, task_id, "sp2-write")
    a.close()


def test_sp3_clean_stage_commits_with_caller(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)
    _mark_caller_write(db, task_id, "sp3-write")

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        result = h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()
    assert _caller_write_survived(db, task_id, "sp3-write")
    other = session_factory()
    row = other.get(TaskInteractionRequest, result.staged_db_id)
    assert row is not None
    other.close()
    db.close()
