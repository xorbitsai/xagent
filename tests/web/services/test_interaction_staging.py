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

import json
import logging
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, event
from sqlalchemy.exc import DataError, IntegrityError, StatementError
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
    InteractionHandoffMisuse,
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


def _force_next_identity_select_to_miss(
    monkeypatch: pytest.MonkeyPatch, db: Session
) -> dict[str, bool]:
    """Force the *next* ``SELECT`` issued on ``db`` to report no row, no
    matter what is actually committed, then let every later statement on
    ``db`` -- including any later ``SELECT`` -- run for real.

    Exists because two sessions racing the way this suite constructs a race
    (the winner commits and closes *before* the loser's own call even
    starts) does not reach step 6's post-conflict re-check on either
    backend: by the time the loser's own step-3 pre-read fires, the
    winner's row is already committed and visible to it (SQLite: a bare,
    non-transactional SELECT sees the latest commit; PostgreSQL READ
    COMMITTED takes a fresh per-statement snapshot), so the loser's call
    returns straight from step 3 and never reaches its own INSERT at all.
    That is a fact about this suite's sequential, same-process
    construction, not a general claim that step 6 is unreachable by
    natural interleaving: on PostgreSQL specifically, a genuinely
    concurrent INSERT that blocks on another session's still-uncommitted
    duplicate row -- true overlap, not this suite's commit-then-run
    ordering -- reaches step 6 naturally once the blocking transaction
    commits and the waiting session's own INSERT then fails with a real
    IntegrityError. Forcing one read to lie is what drives the same
    interleaving inside this suite's own sequential constructions: the
    call proceeds to the reclaim UPDATE and its own INSERT, collides with
    the winner's real, already-committed row on the database's own unique
    constraints, rolls back its own inner savepoint, and only then does
    step 6's second, *unpatched* identity SELECT run -- for real, against
    the real database -- and find the winner's row.

    Only the first ``SELECT`` on ``db`` after this is called is faked;
    every other statement (the reclaim ``UPDATE``, the ``INSERT``, step 6's
    own re-check ``SELECT``) goes through unpatched. Non-``SELECT``
    statements (a caller's own prior write, the reclaim ``UPDATE``) are
    never intercepted at all, regardless of ordering.
    """

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


def _defeat_json_probe_but_not_bind_time_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch ``json.dumps`` so this module's own pre-INSERT probe
    (``json.dumps(request_payload, allow_nan=False)`` in
    ``_validate_request_fields``) unconditionally reports success, while
    SQLAlchemy's own bind-time JSON serialization -- inside the INSERT's own
    flush, past that probe -- keeps failing for real on whatever
    non-serializable value the caller passed.

    An unconditional ``json.dumps = lambda *a, **k: "{}"`` does not achieve
    this: ``json`` is one process-wide module object, and both this
    module's probe (``import json`` in ``task_interaction_staging.py``) and
    SQLAlchemy's own JSON column type (``import json`` in
    ``sqlalchemy/sql/sqltypes.py``) hold the *same* object, confirmed
    directly -- patching ``dumps`` on it patches both callers identically,
    which fakes out the real bind-time serialization too and makes the
    INSERT succeed instead of failing, defeating the point of this
    construction entirely.

    The two call sites are distinguishable by their call signature, not by
    identity: this module's probe always passes ``allow_nan=False``: see
    ``_validate_request_fields``. SQLAlchemy's ``JSON._make_bind_processor``
    calls its serializer as ``json_serializer(value)`` -- one positional
    argument, no keywords at all (confirmed by reading
    ``sqlalchemy/sql/sqltypes.py`` directly). This patch keys off exactly
    that: a call carrying ``allow_nan`` in its keyword arguments is the
    probe and gets faked out; every other call is real serialization and
    runs the real ``json.dumps``.
    """

    real_dumps = json.dumps

    def _patched(*args: Any, **kwargs: Any) -> Any:
        if "allow_nan" in kwargs:
            return "{}"
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(json, "dumps", _patched)


def _fail_next_flush_with_data_error(
    monkeypatch: pytest.MonkeyPatch, db: Session
) -> None:
    """Force the *next* ``Session.flush()`` call on ``db`` to raise a real
    ``sqlalchemy.exc.DataError``, then let every later flush on ``db`` run
    for real.

    SQLite does not enforce ``VARCHAR`` length, so a genuine column-length
    ``DataError`` (the case (g) widens this module's two
    ``except (IntegrityError, DataError)`` guards for -- see
    ``task_interaction_staging.py``'s module docstring, "Three mechanisms,
    not one") cannot be constructed on this backend by driving real data
    through a real INSERT. Monkeypatching ``Session.flush`` itself is the
    least invasive substitute: it raises the real exception type from
    inside the real ``try`` block each guard wraps -- confirmed directly
    that ``Session.begin_nested()`` calls ``self.flush()`` while
    establishing its SAVEPOINT, so this same patch reaches both guards'
    call sites, not just ``stage_interaction_request``'s own explicit
    ``db.flush()``.
    """

    original_flush = Session.flush
    state = {"armed": True}

    def _patched(self: Session, *args: Any, **kwargs: Any) -> Any:
        if self is db and state["armed"]:
            state["armed"] = False
            raise DataError("simulated column-length violation", None, None)
        return original_flush(self, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", _patched)


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


# A payload json.dumps cannot walk without recursing forever -- built once,
# module level, since a circular structure cannot be expressed as a literal
# inline in a pytest.param call.
_circular_payload: dict[str, Any] = {}
_circular_payload["self"] = _circular_payload

_T_P_1_CASES = [
    pytest.param({"kind": "approval"}, ValueError, id="illegal-kind"),
    pytest.param({"kind": 123}, ValueError, id="kind-not-str"),
    pytest.param({"protocol_version": 2}, ValueError, id="protocol-not-1"),
    # 1.0 == 1 numerically, so the old `protocol_version != _PROTOCOL_VERSION`
    # check alone let a float silently through; the isinstance(int) guard
    # added ahead of it is what this case pins.
    pytest.param({"protocol_version": 1.0}, ValueError, id="protocol-version-float"),
    pytest.param(
        {"expires_at": "not-a-datetime"}, ValueError, id="expires-at-not-datetime"
    ),
    pytest.param({"expires_at": None}, ValueError, id="expires-at-none"),
    pytest.param({"now": "not-a-datetime"}, ValueError, id="now-not-datetime"),
    pytest.param({"now": None}, ValueError, id="now-none"),
    pytest.param({"origin": "email"}, ValueError, id="illegal-origin"),
    pytest.param({"origin": 123}, ValueError, id="origin-not-str"),
    pytest.param(
        {"request_idempotency_key": "has a space"}, ValueError, id="key-bad-pattern"
    ),
    pytest.param({"request_idempotency_key": ""}, ValueError, id="key-empty"),
    pytest.param({"request_idempotency_key": 123}, ValueError, id="key-not-str"),
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
    pytest.param(
        {"request_payload": {"when": datetime.now(timezone.utc)}},
        ValueError,
        id="payload-not-json-serializable-datetime",
    ),
    pytest.param(
        {"request_payload": _circular_payload},
        ValueError,
        id="payload-not-json-serializable-circular",
    ),
    pytest.param(
        {"request_payload": {"value": float("nan")}},
        ValueError,
        id="payload-not-json-serializable-nan",
    ),
    pytest.param({"run_id": ""}, ValueError, id="run-id-empty"),
    pytest.param({"run_id": "x" * 65}, ValueError, id="run-id-too-long"),
    # A list, not any non-str: it has a __len__ (so it would have survived
    # the length check added for run_id above without crashing there) and
    # will never == anchor.resume_run_partition (a str), so before the
    # isinstance guard this misclassified as InteractionRunPartitionMismatch
    # instead of failing loudly on the real problem -- a caller passing the
    # wrong type.
    pytest.param({"run_id": ["not", "a", "string"]}, ValueError, id="run-id-not-str"),
    # None is caught by the same isinstance(run_id, str) guard as the list
    # case above, not the `if not run_id` emptiness check right after it --
    # pinned separately since both branches raise ValueError here, and only
    # running this case proves which one actually fired.
    pytest.param({"run_id": None}, ValueError, id="run-id-none"),
    pytest.param({"now": datetime.now()}, ValueError, id="now-naive"),
    pytest.param(
        {"now": _now().replace(tzinfo=timezone(timedelta(hours=8)))},
        ValueError,
        id="now-non-utc",
    ),
]

_T_P_1_ANCHOR_CASES = [
    pytest.param(
        {"resume_execution_id": ""},
        {},
        InteractionAnchorCorrupt,
        id="anchor-execution-id-empty",
    ),
    pytest.param(
        {"resume_run_partition": ""},
        {"run_id": "run-a"},
        InteractionAnchorCorrupt,
        id="anchor-run-partition-empty",
    ),
    pytest.param(
        {"resume_locator_format": "wrong_format"},
        {},
        InteractionAnchorCorrupt,
        id="anchor-locator-format-wrong",
    ),
    pytest.param(
        {"resume_checkpoint_type": "wrong_type"},
        {},
        InteractionAnchorCorrupt,
        id="anchor-checkpoint-type-wrong",
    ),
    pytest.param(
        {"resume_event_id": "x" * 256},
        {},
        InteractionAnchorCorrupt,
        id="anchor-event-id-too-long",
    ),
    pytest.param(
        {"resume_execution_id": "x" * 256},
        {},
        InteractionAnchorCorrupt,
        id="anchor-execution-id-too-long",
    ),
    pytest.param(
        {"resume_run_partition": "x" * 65},
        {"run_id": "run-a"},
        InteractionAnchorCorrupt,
        id="anchor-run-partition-too-long",
    ),
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


def test_step_one_rejects_wrong_type_anchor_without_sql(tmp_path: Path) -> None:
    """An ``anchor`` that is not an ``InteractionAnchor`` at all (a caller
    passing a plain dict, say) must raise ``InteractionAnchorCorrupt`` from
    the isinstance guard at the top of ``_validate_anchor_fields``, not an
    ``AttributeError`` from the first ``anchor.<field>`` access that guard
    exists to preempt."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    kwargs = _stage_kwargs(_anchor(anchor_id))
    kwargs["anchor"] = {"trace_event_id": anchor_id}
    before = len(statements)
    with pytest.raises(InteractionAnchorCorrupt):
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
    db.close()


@pytest.mark.parametrize(
    "anchor_override, kwargs_override, expected_exc", _T_P_1_ANCHOR_CASES
)
def test_step_one_rejects_anchor_field_variants_without_sql(
    tmp_path: Path,
    anchor_override: dict[str, Any],
    kwargs_override: dict[str, Any],
    expected_exc: type[Exception],
) -> None:
    """The remaining ``_validate_anchor_fields`` branches
    ``test_step_one_rejects_empty_anchor_fields_without_sql`` and
    ``test_step_one_rejects_missing_anchor_trace_event_id_without_sql`` don't
    already cover: an empty ``resume_execution_id`` or
    ``resume_run_partition``, and a wrong-vocabulary ``resume_locator_format``
    or ``resume_checkpoint_type``. The run-partition case also overrides
    ``run_id`` so the top-level ``run_id must not be empty`` check
    (``_validate_request_fields``, checked before ``_validate_anchor_fields``
    is ever called) doesn't fire first and mask the anchor check this case
    means to exercise -- ``_stage_kwargs`` defaults ``run_id`` from
    ``anchor.resume_run_partition``, so leaving it alone here would make
    both empty together."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id, **anchor_override)
    kwargs = _stage_kwargs(anchor, **kwargs_override)
    before = len(statements)
    with pytest.raises(expected_exc):
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
    db.close()


_T_P_TASK_ID_CASES = [
    pytest.param(True, id="bool-true"),
    pytest.param("7", id="string-digit"),
    pytest.param(5.9, id="float"),
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
]


@pytest.mark.parametrize("bad_task_id", _T_P_TASK_ID_CASES)
def test_step_one_rejects_invalid_task_id_without_sql(
    tmp_path: Path, bad_task_id: Any
) -> None:
    """``task_id`` gets a proper identity guard, not a silent ``int()``
    coercion: ``bool`` is a subclass of ``int`` in Python and must be
    rejected explicitly (``isinstance(x, bool)`` before ``isinstance(x,
    int)``), a numeric string or a float must be rejected rather than
    coerced, and zero/negative values must be rejected outright. All five
    cases raise before any SQL is issued, the same as every other step-1
    rejection."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    kwargs = _stage_kwargs(anchor)
    before = len(statements)
    with pytest.raises(ValueError):
        stage_interaction_request(db, task_id=bad_task_id, **kwargs)
    assert statements[before:] == []
    db.close()


def test_p2_run_partition_mismatch_is_not_a_slot_taken_subclass(
    tmp_path: Path,
) -> None:
    """T-P-2: run_id != resume_run_partition raises
    InteractionRunPartitionMismatch, and that type is not a subclass of
    InteractionSlotTaken -- the two must stay distinguishable so a
    corruption is never observably confused with an ordinary slot race.
    Also, like every other step-1 rejection, this must be raised in plain
    Python before any SQL is issued."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id, run_partition="run-b")
    kwargs = _stage_kwargs(anchor, run_id="run-a")
    before = len(statements)
    with pytest.raises(InteractionRunPartitionMismatch) as excinfo:
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
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


def test_p3b_stale_caller_clock_does_not_misclassify_as_slot_taken(
    tmp_path: Path,
) -> None:
    """A caller clock running 20 minutes behind wall-clock time must not
    make a legitimate 15-minute TTL misclassify as InteractionSlotTaken.

    Before created_at was bound to the caller's own now (see the model's
    "Clock source" docstring on ck_task_interaction_requests_expiry_after_creation),
    created_at came from server_default=func.now() -- real wall-clock time --
    while expires_at was computed from the caller's stale now. A caller
    running behind produces an expires_at that lands *before* the server's
    real created_at, which the database's CHECK rejects as an
    IntegrityError on the INSERT. Because this table was otherwise empty,
    step 6's post-conflict re-check found no identity row at this call's own
    identity and misclassified that CHECK violation as InteractionSlotTaken
    -- the false diagnosis this test pins: a stale-but-internally-consistent
    caller clock silently swallowed the caller's turn instead of staging its
    request. Binding created_at to the same caller now used for expires_at
    retires this on both backends: both operands now come from one clock, so
    a stale caller now is accepted exactly like a fresh one."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)

    stale_now = _now() - timedelta(minutes=20)
    kwargs = _stage_kwargs(
        anchor, now=stale_now, expires_at=stale_now + timedelta(minutes=15)
    )
    result = stage_interaction_request(db, task_id=task_id, **kwargs)
    assert result.created is True
    assert result.status == "active"
    db.commit()
    db.close()


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
    """The same idempotency key used by two different runs on the same
    task must not be conflated -- each run's step-3 pre-read must only ever
    see rows from its own run. Regression for a step-3 predicate that
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

    with pytest.raises(InteractionOwnerStateError):
        stage_interaction_request(
            db, task_id=task_id, **_stage_kwargs(_anchor(anchor_id))
        )
    db.rollback()
    db.close()


def test_p8b_inner_savepoint_cleans_up_on_non_integrity_error_during_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inner savepoint opened around the INSERT (``inner =
    db.begin_nested()``) must unwind on a bind-time failure inside flush,
    not only on ``IntegrityError`` -- the one type the ``except`` clause
    right after it names.

    An earlier version of this test simulated the failure with a mocked
    ``Session.flush`` raising a plain ``TypeError``. That mock left
    ``inner.is_active`` True the whole time, so it passed for the wrong
    reason -- it never exercised what a real bind-time failure actually
    does to the savepoint. The real failure is worse: SQLAlchemy's JSON
    column type serializes ``request_payload`` at bind time, inside the
    INSERT's own flush -- past this module's pre-INSERT
    ``json.dumps(..., allow_nan=False)`` probe -- and when that
    serialization fails, SQLAlchemy marks the open SAVEPOINT inactive
    *before* the ``StatementError`` it raises ever surfaces here (measured
    directly). By the time control reaches the ``finally``,
    ``inner.is_active`` is already False even though the SAVEPOINT itself is
    still open in the database, waiting to be unwound -- an ``if
    inner.is_active: inner.rollback()`` guard reads that False and skips the
    call entirely, leaking the savepoint open. Only an unconditional
    ``rollback()`` (guarded solely against an already-*closed* transaction,
    via ``ResourceClosedError``) actually pops it.

    To reach that path without this module's own pre-INSERT probe catching
    the payload first, the probe is defeated here (see
    ``_defeat_json_probe_but_not_bind_time_serialization`` for why an
    unconditional ``json.dumps`` patch would silently defang the real
    serialization too, and how this one avoids that) so real serialization
    is left free to fail for real on the non-serializable ``datetime`` in
    the payload below."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)

    _defeat_json_probe_but_not_bind_time_serialization(monkeypatch)

    kwargs = _stage_kwargs(anchor, request_payload={"bad": datetime.now(timezone.utc)})

    with pytest.raises(StatementError):
        stage_interaction_request(db, task_id=task_id, **kwargs)

    assert db.in_nested_transaction() is False

    # The session must accept the next statement -- proving the savepoint
    # was actually unwound, not merely marked inactive and left open
    # underneath (which would instead surface as PendingRollbackError here).
    db.execute(sa.select(sa.literal(1)))
    db.rollback()
    db.close()


def test_p8c_data_error_from_flush_is_owner_state_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case (g): ``stage_interaction_request``'s own step-2 flush guard
    (``except (IntegrityError, DataError)``) must convert a ``DataError``
    the same way it already converts an ``IntegrityError`` -- both are the
    caller's own pending write failing to flush, not a conflict on this
    call's own INSERT. See ``_fail_next_flush_with_data_error`` for why a
    real ``DataError`` has to be driven in via monkeypatch on this backend."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)

    _fail_next_flush_with_data_error(monkeypatch, db)

    with pytest.raises(InteractionOwnerStateError, match="before interaction staging"):
        stage_interaction_request(db, task_id=task_id, **_stage_kwargs(anchor))

    db.rollback()
    db.close()


@pytest.mark.parametrize("raiser", ["logger_error", "register_degradation"])
def test_cm10_logger_error_raising_still_rolls_savepoint_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raiser: str
) -> None:
    """If ``logger.error`` or ``register_degradation`` itself raises inside
    ``interaction_handoff``'s swallowed-exception handler (a misconfigured
    logging handler, or a bug in the signal registry, for instance), the
    savepoint rollback must still happen -- it lives in that handler's own
    ``finally``, not as a bare statement after either call that only runs if
    both of them succeed. Before the fix this pins, a raise from
    ``logger.error`` skipped the rollback entirely, leaking the outer
    savepoint open on top of replacing the swallowed exception with
    whatever the logging call raised. Both calls are wrapped in the same
    ``try``: a handler failure -- from either one -- escapes uncaught by
    design (it is not one of the six swallowed types), but must never leak
    the savepoint it was about to roll back."""

    from xagent.web.services import task_interaction_staging as staging_module

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id, resume_event_id="")  # triggers InteractionAnchorCorrupt
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("handoff signal handler failure")

    if raiser == "logger_error":
        monkeypatch.setattr(staging_module.logger, "error", _boom)
    else:
        assert raiser == "register_degradation"
        monkeypatch.setattr(staging_module, "register_degradation", _boom)

    with pytest.raises(RuntimeError, match="handoff signal handler failure"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )

    assert db.in_nested_transaction() is False
    db.rollback()
    db.close()


def test_p9_replay_after_conflict(tmp_path: Path) -> None:
    """Two sessions race on the same (task_id, run_id, key). ``a``'s own
    manual pre-read (below, before ``b`` commits) misses, as intended -- but
    that is not the read that decides this test's path. By the time ``a``'s
    own call to ``stage_interaction_request`` runs its *own* step-3
    pre-read, ``b`` has already committed, and a fresh SELECT on SQLite sees
    a just-committed row immediately (no persistent snapshot outside an
    explicit transaction) -- so ``a``'s call returns straight from step 3's
    hit, exactly like a same-key replay with no race at all. It never
    reaches its own INSERT, and therefore never reaches step 6's
    post-conflict re-check. This test pins that pre-read replay path (a
    real, separate contract this module makes) and that the loser's own
    call never inserts a duplicate row -- not step 6. See
    ``test_p9b_replay_after_conflict_via_insert_collision`` below for the
    dedicated test that reaches step 6 for real, by forcing the second
    call's own pre-read to miss despite the row already being committed."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    anchor = _anchor(anchor_id)
    key = _next_key()

    a = session_factory()
    b = session_factory()

    # A performs its own step-1..3 manually up through the pre-read miss,
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

    # A now finishes its own call. B has already committed, so A's own
    # step-3 pre-read hits B's row directly here -- no INSERT is attempted
    # on this path (see test_p9b for the construction that forces one).
    a_result = stage_interaction_request(
        a, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    a.commit()
    a.close()

    assert a_result.created is False
    assert a_result.staged_db_id == b_result.staged_db_id

    # a_result is a StagedInteractionRequest by construction -- it is never
    # an InteractionSlotTaken instance, so asserting that in isolation pins
    # nothing about this test's outcome. What actually needs pinning: A's
    # own call, having replayed B's row from its step-3 pre-read, must not
    # have inserted a second row of its own alongside B's.
    verify = session_factory()
    row_count = verify.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.request_idempotency_key == key,
        )
    ).scalar_one()
    verify.close()
    assert row_count == 1, "the loser's own replayed call must not leave a second row"


def test_p9b_replay_after_conflict_via_insert_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dedicated test for step 6's post-conflict re-check itself --
    T-P-9 above pins the pre-read replay path (step 3), which is what every
    naturally-racing construction on this codebase's two backends actually
    exercises; it does not reach step 6 (confirmed: replacing step 6's
    REPLAY return with an unconditional raise leaves T-P-9, T-SP-2, and
    T-CM-1's ``replay-after-conflict`` cell all green).

    Reaching step 6 for real requires the loser's own step-3 pre-read to
    miss despite the winner's row already being committed -- an
    interleaving that does not occur naturally on either backend this suite
    runs against, so it is driven directly via
    ``_force_next_identity_select_to_miss`` instead. B stages and commits
    for real first. A's own pre-read is then forced to report a miss, so
    A's call proceeds to the reclaim UPDATE and its own INSERT, which
    collides for real with B's already-committed row (both
    ``uq_task_interaction_active_slot`` and
    ``uq_task_interaction_request_identity`` are live collision surfaces
    here, since A and B share both ``task_id`` and ``run_id``). That
    collision rolls back A's own inner savepoint, and step 6's own
    re-check -- an unpatched, real SELECT by the time it runs -- finds B's
    row and replays it.
    """

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    anchor = _anchor(anchor_id)
    key = _next_key()

    b = session_factory()
    b_result = stage_interaction_request(
        b, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    b.commit()
    b.close()
    assert b_result.created is True

    a = session_factory()
    _mark_caller_write(a, task_id, "p9b-write")

    _force_next_identity_select_to_miss(monkeypatch, a)
    before = len(statements)
    a_result = stage_interaction_request(
        a, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    issued = statements[before:]

    # The INSERT was genuinely attempted and genuinely collided.
    insert_count = sum(1 for s in issued if s.strip().upper().startswith("INSERT"))
    assert insert_count == 1, issued
    # ...and its own inner savepoint rolled back rather than committed --
    # this is what makes step 6's re-check possible at all (see the
    # module's own docstring on why a shared savepoint breaks this).
    assert any("ROLLBACK TO SAVEPOINT" in s.upper() for s in issued), issued
    assert not any("RELEASE SAVEPOINT" in s.upper() for s in issued), issued

    # Step 6's own re-check ran and replayed B's row -- not a fresh insert
    # of A's own, and not InteractionSlotTaken.
    assert a_result.created is False
    assert a_result.staged_db_id == b_result.staged_db_id
    assert a_result.status == "active"
    assert a_result.active_slot == b_result.active_slot

    a.commit()
    assert _caller_write_survived(a, task_id, "p9b-write")

    verify = session_factory()
    row_count = verify.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.request_idempotency_key == key,
        )
    ).scalar_one()
    verify.close()
    assert row_count == 1, "A's own losing INSERT must not leave a second row"
    a.close()


def test_p9c_post_conflict_recheck_reclassifies_terminal_identity_as_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-4 regression: when step 6's post-conflict re-check finds the
    identity row it collided with in a *terminal* state, it must raise
    ``InteractionRequestClosed`` -- the same classification step 3's own
    pre-read gives that state -- not ``InteractionSlotTaken``. This is the
    cell ``test_p9b_replay_after_conflict_via_insert_collision`` above does
    not cover: that test's collision lands on a still-*active* identity row
    (a REPLAY); this one lands on an already-*closed* one, which is a fact
    about that row's own lifecycle, not an active-slot race.

    B stages and commits a row at ``(task_id, run-a, key)``, then that row
    is terminated directly, in one statement that respects
    ``ck_..._terminated_at_pairs_status``'s paired-column shape (``status``,
    ``terminated_at``, ``terminal_reason`` set together with
    ``active_slot=NULL``) -- the same shape the module's own reclaim UPDATE
    uses. The identity row still exists, just closed, so
    ``uq_task_interaction_request_identity`` still blocks a second INSERT
    at the same ``(task_id, run_id, key)`` -- unlike
    ``uq_task_interaction_active_slot``, which B's terminated row no longer
    holds. A's own pre-read is then forced to miss (the same
    ``_force_next_identity_select_to_miss`` machinery ``test_p9b`` uses), so
    A's call proceeds into the reclaim UPDATE and its own INSERT, which
    collides for real -- on the identity unique, not the active-slot one.
    That collision rolls back A's own inner savepoint, and step 6's own
    re-check -- an unpatched, real SELECT by the time it runs -- finds B's
    now-terminal row and reclassifies the conflict accordingly."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    anchor = _anchor(anchor_id)
    key = _next_key()

    b = session_factory()
    b_result = stage_interaction_request(
        b, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
    )
    b.commit()
    assert b_result.created is True

    b.execute(
        sa.update(TaskInteractionRequest)
        .where(TaskInteractionRequest.id == b_result.staged_db_id)
        .values(
            status="terminated",
            active_slot=None,
            terminal_reason="deadline_elapsed",
            terminated_at=_now(),
        )
    )
    b.commit()
    b.close()

    a = session_factory()
    _force_next_identity_select_to_miss(monkeypatch, a)
    before = len(statements)
    with pytest.raises(InteractionRequestClosed) as excinfo:
        stage_interaction_request(
            a, task_id=task_id, **_stage_kwargs(anchor, request_idempotency_key=key)
        )
    issued = statements[before:]
    assert not isinstance(excinfo.value, InteractionSlotTaken)

    # The INSERT was genuinely attempted and genuinely collided -- on the
    # identity unique, since B's row holds no active slot to collide on.
    insert_count = sum(1 for s in issued if s.strip().upper().startswith("INSERT"))
    assert insert_count == 1, issued
    assert any("ROLLBACK TO SAVEPOINT" in s.upper() for s in issued), issued
    assert not any("RELEASE SAVEPOINT" in s.upper() for s in issued), issued

    a.rollback()
    a.close()

    verify = session_factory()
    row_count = verify.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(
            TaskInteractionRequest.task_id == task_id,
            TaskInteractionRequest.request_idempotency_key == key,
        )
    ).scalar_one()
    verify.close()
    assert row_count == 1, "A's own losing INSERT must not leave a second row"


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
    """k-N2: step 3 replays an already-expired active row without
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
    """Regression pin: the reclaim UPDATE (step 4) stays in the outer
    transaction, not the inner savepoint that wraps the INSERT (step 5), so
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

    So this drives the interleaving directly instead -- constructing
    REPLAY-after-conflict via two racing sessions or by calling the
    primitive's internals directly are both valid, and this uses the
    latter -- by forcing the INSERT's own flush to fail exactly once,
    without a second connection: what matters for this mutation is only
    where the reclaim statement sits relative to the inner savepoint
    boundary, not why the INSERT failed.

    The dirty-view read right after the raise (below) only proves this
    session's own uncommitted state -- it cannot tell a real, durable
    reclaim apart from one this session merely staged in memory and would
    lose on rollback; a mutation that dropped the reclaim from the outer
    transaction entirely, but happened to leave this session's own
    in-memory identity map looking reclaimed, could still pass that read.
    Committing and re-reading from a fresh session afterward is what
    actually proves the reclaim survived: durable on disk, not just visible
    to the session that wrote it.
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
            # Three flush() calls happen on this session before step 5's
            # INSERT would otherwise succeed: (1) step 2's explicit flush of
            # the caller's own pending writes -- none here; (2) the implicit
            # snapshot flush Session.begin_nested() always issues to
            # establish the inner savepoint; (3) step 5's own explicit
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

    db.commit()
    other = session_factory()
    durable_row = _row_state(other, victim.staged_db_id)
    assert durable_row.status == "terminated"
    assert durable_row.terminal_reason == "run_superseded"
    other.close()
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
    "origin-unknown",
    "replay-after-conflict",
]

# F-3: expected TaskInteractionRequest row count for this task after each
# degrading cell exits. slot-taken and request-closed each pre-stage one row
# before the handoff runs (see the case setup below), so zero rows would be
# wrong for those two -- the pre-staged row must survive untouched and no
# second row must have been added. The other four cells fail before any
# staging SQL is issued (validation, or the attempt/anchor/origin assertions
# in stage(), all run before stage_interaction_request's own first
# statement), so zero is the only row count consistent with that ordering.
# replay-after-conflict is not a degrading cell (see the early `return`
# above) and is not in this table.
_T_CM_1_ROW_COUNT_AFTER_DEGRADE = {
    "slot-taken": 1,
    "request-closed": 1,
    "anchor-corrupt": 0,
    "attempt-mismatch": 0,
    "run-partition-mismatch": 0,
    "origin-unknown": 0,
}

# F-10: the exact set of keys interaction_handoff's swallowed-exception
# handler passes as `extra` to logger.error (task_interaction_staging.py,
# the "interaction handoff degraded" log line). Pinned here, not re-derived,
# so a key added to or dropped from that call site's `extra` dict without a
# matching test update fails loudly instead of silently.
_DEGRADATION_LOG_EXTRA_KEYS = {
    "task_id",
    "lease_run_id",
    "lease_attempt_id",
    "anchor_run_partition",
    "exception_type",
    "degradation_signal",
}


@pytest.mark.parametrize("case", _T_CM_1_CASES)
def test_cm1_seven_cell_exit_matrix(
    tmp_path: Path, case: str, caplog: pytest.LogCaptureFixture
) -> None:
    """T-CM-1, widened to seven cells: the six swallowed types
    (InteractionSlotTaken, InteractionRequestClosed, InteractionAnchorCorrupt,
    InteractionAttemptMismatch, InteractionRunPartitionMismatch -- swallowed
    under this module's F-3 override, see the CM's docstring -- and
    InteractionOriginUnknown), plus REPLAY-after-conflict as the
    successful-return control. Every cell: (a) the with-block exits
    without the exception escaping; (b) exactly the swallowed exceptions
    register a degradation and log; (c) the caller's own pre-with pending
    write survives and commits alongside the caller.

    The ``replay-after-conflict`` cell's own construction (below) pins the
    step-3 pre-read replay path through the full context manager -- the
    same path T-P-9 pins at the primitive level, not step 6's
    post-conflict re-check: by the time this cell's own ``h.stage()`` call
    runs its step-3 pre-read, the winner session has already committed, and
    that pre-read hits directly (see T-P-9's docstring for why). Step 6 is
    reached and pinned separately, by
    ``test_p9b_replay_after_conflict_via_insert_collision`` at the
    primitive level -- confirmed by the same poison-probe check T-P-9's
    docstring describes: this cell stays green with step 6's REPLAY return
    replaced by an unconditional raise."""

    caplog.set_level(
        logging.ERROR, logger="xagent.web.services.task_interaction_staging"
    )
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
    elif case == "origin-unknown":
        db.execute(sa.update(Task).where(Task.id == task_id).values(source="web"))
        db.commit()
        expect_signal = ops_signals.INTERACTION_HANDOFF_DEGRADED
    else:
        assert case == "replay-after-conflict"
        expect_signal = None

    if case == "replay-after-conflict":
        # A REPLAY-after-conflict via the step-3 pre-read, not a clean
        # insert and not step 6 (see this test's own docstring): db's first
        # statement on this session has to happen before the winner's
        # commit, purely so this session's own connection exists before
        # that commit -- db's own later pre-read inside h.stage() still
        # hits the winner's row directly once it runs, since that commit
        # has already happened by then. Same construction as T-P-9 / T-SP-2.
        from xagent.web.services.task_interaction_staging import (
            _identity_lookup_stmt,
        )

        db.execute(
            _identity_lookup_stmt(
                task_id=task_id,
                run_id=lease.run_id,
                request_idempotency_key=stage_key,
            )
        ).first()

        winner = session_factory()
        winner_task = winner.get(Task, task_id)
        with interaction_handoff(
            winner, lease, task=winner_task, anchor=anchor, now=_now()
        ) as h:
            winner_result = h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=stage_key,
                expires_at=_now() + timedelta(minutes=15),
            )
        winner.commit()
        winner.close()
        assert winner_result.created is True

        task = db.get(Task, task_id)
        _mark_caller_write(db, task_id, f"caller-write-{case}")
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            first = h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=stage_key,
                expires_at=_now() + timedelta(minutes=15),
            )
        db.commit()
        assert first.created is False
        assert first.staged_db_id == winner_result.staged_db_id
        assert _caller_write_survived(db, task_id, f"caller-write-{case}")
        db.close()
        assert ops_signals.active_degradations() == {}
        return

    task = db.get(Task, task_id)
    _mark_caller_write(db, task_id, f"caller-write-{case}")

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
    # An exact-set comparison, not `expect_signal in signals`, but scoped to
    # this module's own signal names: the module-global registry
    # (ops_signals._signals) can carry residue from other tests that share
    # the same process, and a global equality check would be flaky against
    # that residue, not against this test's own behavior.
    interaction_signals = {name for name in signals if name.startswith("interaction_")}
    assert interaction_signals == {expect_signal}

    # F-3: table state after degrade. slot-taken and request-closed each
    # pre-staged one row before the handoff ran; the other four cells fail
    # before any staging SQL is issued. Either way, the handoff's own
    # savepoint rollback must leave exactly this many rows behind -- not
    # more (a leaked INSERT) and not fewer (a rollback that undid a
    # pre-existing row it never should have touched).
    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == _T_CM_1_ROW_COUNT_AFTER_DEGRADE[case]

    # The degradation log record's `extra` payload is exactly the six keys
    # the swallow handler emits -- pinned by diffing this record's
    # attributes against a control record's, rather than merely checking
    # the six are present, so a key added to that call site without a
    # matching test update is caught too. The control record is emitted
    # through the same logger and captured by the same caplog handler, so
    # every environment-dependent built-in attribute (which varies across
    # Python versions and capture setups) appears on both sides and cancels
    # out of the diff; a hand-constructed bare LogRecord does not have that
    # property.
    degraded_records = [
        r for r in caplog.records if r.message == "interaction handoff degraded"
    ]
    assert len(degraded_records) == 1, degraded_records
    record = degraded_records[0]
    logging.getLogger("xagent.web.services.task_interaction_staging").error(
        "caplog baseline probe"
    )
    baseline = next(r for r in caplog.records if r.message == "caplog baseline probe")
    extra_keys = set(vars(record)) - set(vars(baseline))
    assert extra_keys == _DEGRADATION_LOG_EXTRA_KEYS
    assert record.task_id == task_id
    assert record.degradation_signal == expect_signal

    db.close()


def test_cm2_owner_state_error_propagates_uncaught(tmp_path: Path) -> None:
    """T-CM-2: InteractionOwnerStateError is the one exception this module
    raises that is never swallowed -- it propagates out of the with-block,
    and the CM's own savepoint has already been rolled back by the time it
    does.

    The doomed write is added *inside* the with-block, right before
    ``stage()`` -- not before ``interaction_handoff`` is even entered --
    so its flush happens inside ``stage_interaction_request``'s own step-2
    flush, reached through the CM's ``except`` clause around ``yield``, the
    same path a real caller's own pending write would take. Adding it
    before the ``with`` line instead would only exercise the CM's *other*
    IntegrityError guard, the one around its own ``db.begin_nested()`` --
    covered separately by
    ``test_cm2e_owner_state_error_from_begin_nested_propagates_uncaught``
    below."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(InteractionOwnerStateError):
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
    assert db.in_transaction()
    db.rollback()
    db.close()


def test_cm2e_owner_state_error_from_begin_nested_propagates_uncaught(
    tmp_path: Path,
) -> None:
    """T-CM-2's own *other* IntegrityError guard: a doomed write added
    *before* ``interaction_handoff`` is even entered is caught by
    ``Session.begin_nested()``'s own mandatory pre-savepoint flush --
    ``interaction_handoff``'s own ``except`` around that call, not
    ``stage_interaction_request``'s step-2 flush T-CM-2 above exercises.
    Both guards raise ``InteractionOwnerStateError``; only the message
    differs (see the two raise sites in ``task_interaction_staging.py``).
    Because the failure fires from ``__enter__``, the ``with`` block's body
    never runs -- ``stage()`` is never reached."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    doomed = Task(user_id=10**9, title=None)
    db.add(doomed)

    with pytest.raises(InteractionOwnerStateError, match="while opening"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()):
            pass

    db.rollback()
    # The session must accept the next statement -- proving the failure
    # left it usable after a full rollback, not merely marked but still
    # poisoned.
    assert db.execute(sa.select(sa.literal(1))).scalar_one() == 1
    db.close()


def test_cm2f_data_error_from_begin_nested_is_owner_state_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case (g)'s other guard: ``interaction_handoff``'s own
    ``except (IntegrityError, DataError)`` around ``db.begin_nested()`` must
    also convert a ``DataError`` -- the counterpart to
    ``test_p8c_data_error_from_flush_is_owner_state_error`` above, which
    pins the same conversion at ``stage_interaction_request``'s own step-2
    flush. See ``_fail_next_flush_with_data_error`` for why a real
    ``DataError`` has to be driven in via monkeypatch on this backend, and
    ``test_cm2e`` above for why the ``with`` block's body never runs when
    this guard is the one that fires."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    _fail_next_flush_with_data_error(monkeypatch, db)

    with pytest.raises(InteractionOwnerStateError, match="while opening"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()):
            pass

    db.rollback()
    db.close()


def test_cm2b_non_serializable_payload_propagates_through_handoff(
    tmp_path: Path,
) -> None:
    """A request_payload the JSON-serialization probe rejects raises
    ValueError from stage_interaction_request's own validation block, before
    any staging SQL is issued -- and, called through interaction_handoff,
    that ValueError is not one of the six swallowed types, so it propagates
    out of the with-block uncaught, the same as InteractionOwnerStateError
    and InteractionHandoffMisuse. A caller must see this as a loud
    programming-error failure, not a silently degraded turn."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(ValueError, match="not JSON-serializable"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"self": _circular_payload},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )

    assert ops_signals.active_degradations() == {}
    assert db.in_transaction()
    db.rollback()
    db.close()


def test_cm2c_statement_error_at_bind_time_propagates_uncaught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike ``test_cm2b`` above (a ``ValueError`` this module's own
    pre-INSERT probe catches in plain Python), a bind-time
    ``StatementError`` -- the probe defeated the same way
    ``test_p8b_inner_savepoint_cleans_up_on_non_integrity_error_during_insert``
    defeats it -- is not one of the six swallowed types either, so it must
    propagate out of the with-block the same way, through the CM's
    ``except BaseException`` branch, with the outer savepoint actually
    unwound behind it (not merely marked inactive) and no degradation
    signal registered."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    _defeat_json_probe_but_not_bind_time_serialization(monkeypatch)

    with pytest.raises(StatementError):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"bad": datetime.now(timezone.utc)},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )

    assert db.in_nested_transaction() is False
    assert ops_signals.active_degradations() == {}
    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == 0
    db.rollback()
    db.close()


def test_cm2d_bare_runtime_error_from_with_body_propagates_and_unwinds(
    tmp_path: Path,
) -> None:
    """A caller bug inside the with-block that is none of the six swallowed
    types, ``InteractionOwnerStateError``, or ``InteractionHandoffMisuse``
    must still propagate uncaught, through the CM's ``except BaseException``
    branch -- and that branch's own savepoint rollback must actually unwind
    it, discarding the row ``stage()`` already staged, rather than leaking
    it open the way an ``is_active``-guarded rollback would on a
    flush-deactivated savepoint."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(RuntimeError, match="caller bug inside the with block"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )
            raise RuntimeError("caller bug inside the with block")

    assert db.in_nested_transaction() is False
    assert ops_signals.active_degradations() == {}
    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == 0
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


def test_cm3b_zero_stage_calls_is_legal(tmp_path: Path) -> None:
    """F-11: zero stage() calls is legal (see _InteractionHandoff's own
    docstring, task_interaction_staging.py :1038-1042) -- a caller may open
    interaction_handoff, decide not to ask, and exit normally with nothing
    staged."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()):
        _mark_caller_write(db, task_id, "caller-write-zero-stage")

    db.commit()

    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == 0
    assert ops_signals.active_degradations() == {}
    assert _caller_write_survived(db, task_id, "caller-write-zero-stage")
    db.close()


def test_cm5_in_block_write_is_discarded_with_the_degraded_row(
    tmp_path: Path,
) -> None:
    """Pins the module docstring's containment boundary: a write the caller
    issues *inside* the ``with`` block shares this function's own savepoint
    with the interaction row itself, so a degrade's ``ROLLBACK TO
    SAVEPOINT`` discards both together -- unlike a write flushed *before*
    the block opens (see the other ``test_cm*`` cases, which all mark their
    caller write ahead of ``with`` and assert it survives). This is exactly
    why the caller contract restricts the ``with`` body to handoff
    operations alone: a write placed inside it is not merely at risk of
    being rolled back on a degrade, it *will* be, indistinguishably from
    the interaction row it was staged alongside."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _force_attempt_mismatch(db, task_id)
    db.commit()
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        _mark_caller_write(db, task_id, "in-block-write")
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()

    assert ops_signals.INTERACTION_HANDOFF_DEGRADED in ops_signals.active_degradations()
    assert not _caller_write_survived(db, task_id, "in-block-write")
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


def test_cm7_second_stage_in_one_handoff_is_rejected(tmp_path: Path) -> None:
    """T-CM-7: ``stage()`` may be called at most once per handoff. A second
    call inside the same ``with`` block raises ``InteractionHandoffMisuse``,
    which is not one of the five swallowed types -- it propagates through
    the CM's ``except BaseException`` branch, which rolls back the *outer*
    savepoint before re-raising. That rollback discards the first call's
    already-inner-committed row along with it: the whole handoff is loud
    (the exception is visible to the caller) and leaves nothing behind,
    never a silent ``created=True`` receipt for the first call while the
    second is dropped on the floor."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(InteractionHandoffMisuse):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "first"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "second"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )

    # Not swallowed: no degradation signal, same as InteractionOwnerStateError.
    assert ops_signals.active_degradations() == {}
    assert db.in_transaction()
    db.commit()
    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == 0, "the outer savepoint rollback must discard the first row too"
    db.close()


def test_cm7b_lease_task_mismatch_raises_before_any_staging_statement(
    tmp_path: Path,
) -> None:
    """``stage()`` must reject a lease that names a different task than the
    one this handoff was constructed with -- a caller bug distinct from
    ``InteractionAttemptMismatch`` (same task, stale attempt): here the
    lease and the task disagree about which row is even in play. Raised as
    a plain ``ValueError``, not swallowed, before the attempt or anchor
    assertions or any staging SQL against ``task_interaction_requests``."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id + 1)
    task = db.get(Task, task_id)

    before = len(statements)
    with pytest.raises(ValueError, match="lease names task"):
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
        s.strip().upper().startswith("INSERT INTO task_interaction_requests".upper())
        for s in issued
    ), issued
    assert ops_signals.active_degradations() == {}
    db.rollback()
    db.close()


@pytest.mark.parametrize("caller_action", ["rollback", "commit"])
def test_cm8_degradation_registers_even_after_caller_deactivates_savepoint(
    tmp_path: Path, caller_action: str
) -> None:
    """T-CM-8: a caller that violates the "no I/O in between" contract by
    calling ``db.rollback()`` or ``db.commit()`` from inside the with-block
    (both end the whole transaction, deactivating this CM's own outer
    savepoint along with it) must not turn a swallowed exception into an
    unrelated crash, and must not lose the degradation signal either.

    Registration and logging now run before the guarded rollback
    (``if savepoint.is_active: savepoint.rollback()``): the savepoint this
    handler would otherwise try to roll back a second time is already
    inactive by the time ``h.stage()`` raises, so the guard skips it, the
    swallowed exception still does not escape, and the signal is still
    registered -- the ordering this test pins is what makes that possible;
    an unguarded ``savepoint.rollback()`` here would raise
    ``ResourceClosedError`` in its place, both replacing the swallowed
    exception and skipping the degradation signal."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _force_attempt_mismatch(db, task_id)
    db.commit()
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        if caller_action == "rollback":
            db.rollback()
        else:
            db.commit()
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )

    signals = ops_signals.active_degradations()
    assert ops_signals.INTERACTION_HANDOFF_DEGRADED in signals
    db.close()


@pytest.mark.parametrize("caller_action", ["rollback", "commit"])
def test_cm8b_normal_exit_after_caller_deactivates_savepoint_raises_misuse(
    tmp_path: Path, caller_action: str
) -> None:
    """T-CM-8's success-path twin: ``h.stage()`` succeeds first, then the
    caller violates the "no I/O in between" contract by calling
    ``db.rollback()`` or ``db.commit()`` from inside the with-block. Both
    end the whole transaction, deactivating this context manager's own
    outer savepoint the same way T-CM-8 does for the swallowed-exception
    path -- but here the block's body raises nothing, so the generator
    resumes at the normal-exit branch, where the savepoint no longer exists
    to commit. Raising ``InteractionHandoffMisuse`` there instead of
    silently skipping the now-impossible commit is what this commit fixes:
    silently skipping would report success for a row whose containment is
    already gone (see that exception's own docstring)."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(InteractionHandoffMisuse):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )
            if caller_action == "rollback":
                db.rollback()
            else:
                db.commit()

    # Not swallowed: no degradation signal, same as InteractionOwnerStateError.
    assert ops_signals.active_degradations() == {}
    db.close()


@pytest.mark.parametrize("caller_action", ["rollback", "commit"])
def test_cm8c_normal_exit_after_deactivation_without_a_stage_call_names_no_row(
    tmp_path: Path, caller_action: str
) -> None:
    """T-CM-8b's zero-``stage()``-call twin: calling ``h.stage()`` zero
    times is legal on its own (a caller may decide not to ask), but a
    caller that never calls it and still commits or rolls back from inside
    the ``with`` block deactivates this context manager's own outer
    savepoint the same way T-CM-8b's does -- the misuse detection does not
    require a prior ``stage()`` call to fire. The two variants must not
    share a message that claims a staged row exists when
    ``handoff._staged_row`` is still ``False``: this pins the "no
    interaction row was staged" wording for that case, while T-CM-8b pins
    the other."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(InteractionHandoffMisuse, match="no interaction row was staged"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()):
            if caller_action == "rollback":
                db.rollback()
            else:
                db.commit()

    assert ops_signals.active_degradations() == {}
    db.close()


def test_cm8d_stage_call_that_raises_before_staging_names_no_row(
    tmp_path: Path,
) -> None:
    """The case the ``_staged`` / ``_staged_row`` split exists for:
    ``h.stage()`` is called (so ``_staged`` -- the one-call-per-handoff
    reentry counter -- is ``True`` from the first line of ``stage()``,
    before any of its own checks run), but this call's ``lease`` has no
    ``run_id``, so ``stage()`` raises ``ValueError`` before it ever calls
    ``stage_interaction_request`` -- no row was staged, so ``_staged_row``
    stays ``False``. The caller here catches that ``ValueError`` itself,
    inside the ``with`` block, and then commits from inside it -- the same
    no-I/O-in-between violation T-CM-8b/T-CM-8c construct, deactivating
    this handoff's own savepoint. The misuse message that follows must
    describe this the same way T-CM-8c's zero-call case does ("no
    interaction row was staged"), not T-CM-8b's ("the staged interaction
    row no longer exists") -- ``_staged`` alone cannot tell these two cases
    apart, because it is already ``True`` in both by the time the ``with``
    block exits."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = TaskLease(
        task_id=task_id, runner_id="runner-1", run_id=None, attempt_id=None
    )
    task = db.get(Task, task_id)

    with pytest.raises(InteractionHandoffMisuse, match="no interaction row was staged"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            try:
                h.stage(
                    kind="clarification",
                    protocol_version=1,
                    request_payload={"prompt": "p"},
                    request_idempotency_key=_next_key(),
                    expires_at=_now() + timedelta(minutes=15),
                )
            except ValueError:
                pass
            db.commit()

    assert ops_signals.active_degradations() == {}
    db.close()


def test_cm9_out_of_vocabulary_task_source_degrades(tmp_path: Path) -> None:
    """T-CM-9: ``origin`` is a frozen copy of ``task.source`` -- an audit
    column recording which surface raised the interaction. A ``task.source``
    outside the frozen origin vocabulary (drift: e.g. a value some other,
    newer code path started writing after this vocabulary was fixed) must
    not be silently coerced to ``"internal"``, which would write a false
    provenance. It degrades instead: no row is written, no exception
    escapes, and the shared handoff signal is registered -- the same
    treatment as the other four originally-swallowed types."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)

    db.execute(sa.update(Task).where(Task.id == task_id).values(source="web"))
    db.commit()
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()

    assert ops_signals.INTERACTION_HANDOFF_DEGRADED in ops_signals.active_degradations()
    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == 0
    db.close()


def test_cm9b_vocabulary_binding_is_the_origin_set_not_the_gating_set() -> None:
    """The staging validation must bind to the model's six-member origin
    vocabulary (INTERACTION_ORIGIN_VOCABULARY), not the rollout gate's
    seven-member gating vocabulary (INTERACTION_GATING_SOURCES), whose extra
    synthetic "channel" key exists only as an operator gating token. Binding
    to the wrong set would let "channel" pass Python-side validation, hit
    the database CHECK, and come back misclassified as a slot conflict.
    Identity, not equality: an equal-but-distinct copy would already be a
    second source."""

    from xagent.web.models.task_interaction import INTERACTION_ORIGIN_VOCABULARY
    from xagent.web.services import task_interaction_staging
    from xagent.web.services.interaction_rollout import INTERACTION_GATING_SOURCES

    assert task_interaction_staging._ORIGIN_VOCABULARY is INTERACTION_ORIGIN_VOCABULARY
    assert task_interaction_staging._ORIGIN_VOCABULARY != INTERACTION_GATING_SOURCES
    assert "channel" in INTERACTION_GATING_SOURCES
    assert "channel" not in task_interaction_staging._ORIGIN_VOCABULARY


def test_cm9c_gating_only_channel_source_degrades(tmp_path: Path) -> None:
    """The gating-only "channel" token, read back as a ``task.source``, must
    take the unknown-origin degrade path exactly like any other
    out-of-vocabulary source: no row, no escaping exception, the shared
    handoff signal registered. This is the behavioral half of the binding
    pin above -- if validation ever bound to INTERACTION_GATING_SOURCES,
    "channel" would pass Python-side validation and reach the database
    CHECK instead."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)

    db.execute(sa.update(Task).where(Task.id == task_id).values(source="channel"))
    db.commit()
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()

    degradations = ops_signals.active_degradations()
    assert ops_signals.INTERACTION_HANDOFF_DEGRADED in degradations
    # The degrade must be the unknown-origin classification made BEFORE any
    # SQL. If validation were bound to the gating set instead, "channel"
    # would reach the database CHECK and come back reclassified as a slot
    # conflict -- same signal, same zero rows, different exception type --
    # which is exactly the failure this assertion distinguishes.
    assert (
        "InteractionOriginUnknown"
        in degradations[ops_signals.INTERACTION_HANDOFF_DEGRADED]
    )
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

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    # Code placed here, standing in for "the caller's own commit" -- the
    # real assertion is that it runs at all and its write is durable,
    # checked below via a fresh session, not via a boolean flag that would
    # only prove this line was reached, not that its effect persisted.
    db.execute(
        sa.update(Task).where(Task.id == task_id).values(title="post-with-write")
    )
    db.commit()

    db2 = session_factory()
    assert (
        db2.execute(sa.select(Task.title).where(Task.id == task_id)).scalar_one()
        == "post-with-write"
    )
    db2.close()
    db.close()


def test_cm4_no_notification() -> None:
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
    # Exact-set, not `in`, scoped to this module's own signal names -- same
    # idiom as T-CM-1 (test_cm1_seven_cell_exit_matrix): the module-global
    # registry can carry residue from other tests sharing the same process,
    # and a bare `in` check would not catch a second, unexpected signal
    # firing alongside the expected one.
    signals = ops_signals.active_degradations()
    interaction_signals = {name for name in signals if name.startswith("interaction_")}
    assert interaction_signals == {ops_signals.INTERACTION_HANDOFF_DEGRADED}
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


@pytest.mark.parametrize("bad_id", _T_P_TASK_ID_CASES)
def test_step_one_rejects_non_integer_anchor_row_ids_without_sql(
    tmp_path: Path, bad_id: Any
) -> None:
    """The anchor's row id is a database identity: a bool coerces to a valid
    foreign key pointing at a different trace row, so it is rejected before
    any SQL, on the same footing as ``task_id``. Classified as anchor
    corruption, consistent with the ``None`` case."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(bad_id)
    kwargs = _stage_kwargs(anchor)
    before = len(statements)
    with pytest.raises(InteractionAnchorCorrupt):
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
    db.close()


def test_step_one_rejects_boolean_protocol_version_without_sql(
    tmp_path: Path,
) -> None:
    """``True == 1`` in Python, so the equality check alone would admit a
    bool; the strict guard keeps the vocabulary genuinely integer-valued."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    kwargs = _stage_kwargs(anchor, protocol_version=True)
    before = len(statements)
    with pytest.raises(ValueError):
        stage_interaction_request(db, task_id=task_id, **kwargs)
    assert statements[before:] == []
    db.close()


def test_cm_swallows_subclasses_of_swallowed_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The except clause matches subclasses, so the signal lookup must too:
    a subclass of a swallowed type must degrade with its parent's signal
    rather than escaping as a KeyError from inside the handler."""

    from xagent.web.services import task_interaction_staging as _module

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    task = db.get(Task, task_id)
    lease = _lease(task_id)
    _mark_caller_write(db, task_id, "cm-subclass-write")

    class _SubSlotTaken(InteractionSlotTaken):
        pass

    def _raise_subclass(*args: Any, **kwargs: Any) -> None:
        raise _SubSlotTaken("subclassed slot conflict")

    monkeypatch.setattr(_module, "stage_interaction_request", _raise_subclass)

    with interaction_handoff(
        db, lease, task=task, anchor=_anchor(anchor_id), now=_now()
    ) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()
    # A real durability assertion, not a boolean flag: the with-block must
    # actually exit and let the caller's own commit run, and that commit's
    # effect (the caller write marked above) must survive it -- the same
    # caller-write-survival pattern every other T-CM cell in this suite
    # uses to prove the same thing.
    assert _caller_write_survived(db, task_id, "cm-subclass-write")
    assert ops_signals.INTERACTION_HANDOFF_DEGRADED in ops_signals.active_degradations()
    db.close()


def test_every_swallowed_type_has_a_degradation_signal() -> None:
    """Structural pin for the fallback in the CM's signal lookup: the
    fallback exists so a mapping gap degrades generically instead of
    crashing, and this test exists so such a gap cannot ship unnoticed."""

    from xagent.web.services.task_interaction_staging import (
        _DEGRADATION_SIGNALS,
        _SWALLOWED,
    )

    unmapped = [
        cls.__name__
        for cls in _SWALLOWED
        if not any(issubclass(cls, mapped) for mapped in _DEGRADATION_SIGNALS)
    ]
    assert unmapped == []


def test_cm11_swallow_handler_survives_a_non_object_anchor(tmp_path: Path) -> None:
    """Regression pin for ``_safe()``. A caller passing a dict as ``anchor``
    (not an ``InteractionAnchor`` at all -- the same shape
    ``test_step_one_rejects_wrong_type_anchor_without_sql`` uses against
    ``stage_interaction_request`` directly) makes ``_validate_anchor_fields``
    raise ``InteractionAnchorCorrupt`` from its own ``isinstance`` guard,
    before ever touching ``anchor.resume_run_partition``. But the *swallow
    handler* in ``interaction_handoff`` used to read that exact attribute
    straight off the same object while building its log line -- a dict has
    no ``resume_run_partition``, so that read crashed with ``AttributeError``
    and replaced the swallowed ``InteractionAnchorCorrupt`` instead of
    degrading. This pins that a dict anchor degrades the same way every
    other ``InteractionAnchorCorrupt`` does: no exception escapes, the
    shared signal registers, and no row is written."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    lease = _lease(task_id)
    task = db.get(Task, task_id)
    bad_anchor = {"trace_event_id": anchor_id}

    with interaction_handoff(db, lease, task=task, anchor=bad_anchor, now=_now()) as h:
        h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()

    assert ops_signals.INTERACTION_HANDOFF_DEGRADED in ops_signals.active_degradations()
    row_count = db.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    assert row_count == 0
    db.close()


def test_safe_returns_none_for_detached_expired_instance(tmp_path: Path) -> None:
    """``_safe()``'s own docstring names ``DetachedInstanceError``
    specifically, not just ``AttributeError``, as a case it must swallow --
    ``test_cm11_swallow_handler_survives_a_non_object_anchor`` above only
    exercises the ``AttributeError`` path (a dict has no such attribute at
    all). This pins the other named path directly: an ORM instance that is
    both expired (its column values must be reloaded from the database
    before a read can return) and detached (there is no session left to do
    that reload with) raises ``DetachedInstanceError`` -- a
    ``SQLAlchemyError`` subclass, not ``AttributeError`` -- from the
    attribute access itself, so a bare ``getattr(obj, name, None)`` would
    not catch it and would crash the logging ``_safe()`` exists to
    protect."""

    from xagent.web.services.task_interaction_staging import _safe

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, _anchor_id = _seed(session_factory)
    db = session_factory()
    task = db.get(Task, task_id)
    db.commit()
    db.expire_all()
    db.expunge(task)

    assert _safe(task, "title") is None
    db.close()


def test_cm12_lease_without_run_id_is_rejected_in_stage(tmp_path: Path) -> None:
    """A ``TaskLease`` with no ``run_id`` at all (the pre-attempt-column
    rolling-restart window and the ``_task_lease_snapshot`` ambient sentinel
    can both produce one, and here it is built directly with both
    ``run_id`` and ``attempt_id`` set to ``None``) cannot stage anything:
    ``stage()``'s own ``ValueError`` for this ("lease has no run_id; cannot
    stage an interaction request without one") is a programming-error
    signal, not one of the six swallowed types, so it propagates out of the
    with-block uncaught instead of degrading -- there is no
    ``anchor.resume_run_partition`` for a ``None`` ``run_id`` to even be
    compared against."""

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = TaskLease(
        task_id=task_id, runner_id="runner-1", run_id=None, attempt_id=None
    )
    task = db.get(Task, task_id)

    with pytest.raises(ValueError, match="lease has no run_id"):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(
                kind="clarification",
                protocol_version=1,
                request_payload={"prompt": "p"},
                request_idempotency_key=_next_key(),
                expires_at=_now() + timedelta(minutes=15),
            )

    assert ops_signals.active_degradations() == {}
    assert db.in_transaction()
    db.rollback()

    verify = session_factory()
    row_count = verify.execute(
        sa.select(sa.func.count())
        .select_from(TaskInteractionRequest)
        .where(TaskInteractionRequest.task_id == task_id)
    ).scalar_one()
    verify.close()
    assert row_count == 0
    db.close()


def test_handoff_never_writes_any_task_column(tmp_path: Path) -> None:
    """The module docstring's own invariant: neither entry point writes any
    *data* column of ``tasks``. The SQLite-only savepoint guard
    (``interaction_handoff``'s zero-row ``UPDATE tasks SET id = id WHERE id
    = -1``) is written with ``sa.text()`` specifically so its SET clause
    names exactly one column, a self-assignment matching zero rows --
    ``sa.update(Task)`` with no ``.values()`` at all would instead compile a
    SET clause naming every column SQLAlchemy maps on ``Task`` (lease-fencing
    columns included), and even ``.values(id=-1)`` alone still picks up
    ``updated_at`` from its ``onupdate=func.now()``.

    A before/after column snapshot alone cannot catch every way this
    invariant could break: an UPDATE whose SET clause names every column on
    ``tasks`` but whose WHERE clause matches zero rows (``WHERE id = -1``,
    for instance) would leave the snapshot byte-identical even though the
    statement itself was wrong. The statements this handoff actually issues
    are the real subject, so a ``before_cursor_execute`` listener
    (``_count_cursor_executions``) captures every one of them, and this
    asserts the only UPDATE that ever targets ``tasks`` is the guard's own
    single-column self-assignment -- not merely that its (zero) rowcount
    happened to leave no trace. The column snapshot is kept as a secondary,
    redundant check."""

    engine = _engine(tmp_path)
    statements = _count_cursor_executions(engine)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    before = {
        column.name: getattr(task, column.name) for column in Task.__table__.columns
    }
    before_statement_count = len(statements)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        staged = h.stage(
            kind="clarification",
            protocol_version=1,
            request_payload={"prompt": "p"},
            request_idempotency_key=_next_key(),
            expires_at=_now() + timedelta(minutes=15),
        )
    db.commit()
    assert staged.created is True

    issued = statements[before_statement_count:]
    task_updates = [
        statement
        for statement in issued
        if statement.split(None, 2)[:2] == ["UPDATE", Task.__tablename__]
    ]
    guard_sql = f"UPDATE {Task.__tablename__} SET id = id WHERE id = -1"
    assert task_updates == [guard_sql], task_updates
    for statement in task_updates:
        set_clause = statement.upper().split(" SET ", 1)[1].split(" WHERE ")[0]
        for column in (
            "runner_id",
            "lease_attempt_id",
            "run_id",
            "status",
            "lease_expires_at",
        ):
            assert column.upper() not in set_clause, (column, statement)

    db.close()

    fresh = session_factory()
    fresh_task = fresh.get(Task, task_id)
    after = {
        column.name: getattr(fresh_task, column.name)
        for column in Task.__table__.columns
    }
    fresh.close()

    assert after == before
