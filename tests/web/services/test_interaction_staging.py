"""Contract tests for ``stage_interaction_request`` (the T-P group).

The context manager (``interaction_handoff``) and its own tests (T-CM,
T-SP) land in a follow-up commit -- this file's own docstring will grow to
describe both once they do.

Each test builds its own file-backed sqlite database under ``tmp_path``
rather than the process-wide singleton, for the same reason
``test_trace_event_staging.py`` does: T-P-9 needs two independent sessions
with genuine transaction isolation between them (REPLAY-after-conflict),
which an in-memory database shared over one pooled connection does not
give.
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
    stage_interaction_request,
)

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
