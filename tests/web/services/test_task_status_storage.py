"""Task.status storage-semantics sentinels (SQLite half).

sqlalchemy.Enum(TaskStatus) with no values_callable persists the enum
member *name* (e.g. "WAITING_FOR_USER"), not its value
("waiting_for_user"). An ORM- or Core-built comparison or write against a
raw value string fails at bind time with StatementError wrapping
LookupError, symmetrically on both backends (pinned by Sentinel 3 below).
Zero-rows-on-SQLite/raises-on-PostgreSQL still happens, but only for raw
text() SQL that bypasses that bind layer: SQLite silently matches zero
rows (see test_wrong_case_literal_query_zero_hits_sqlite below) and
PostgreSQL raises DataError instead (pinned in the PostgreSQL half of this
suite, test_task_status_storage_postgresql.py).

task_status_predicate (xagent.web.models.task) is the one typed entry point
for every SQL predicate/write against this column. These tests pin: (1)
the storage format itself, so a future accidental switch to storing values
is caught; (2) that task_status_predicate compiles to exactly what the raw
comparisons it replaced compiled to; (3) that a bypass of the binding is
caught by a bind-layer sentinel; (4) that a raw-SQL bypass (which the
column's validate_strings=True cannot see) is caught by a DB-layer
sentinel.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from sqlalchemy import Column, Enum, Integer, case, func, or_, text, update
from sqlalchemy.orm import declarative_base

from tests.web.services.task_status_storage_shared import (
    RAW_VALUE_LITERAL_ZERO_MATCH_SQL,
    TASK_STATUS_STORAGE_NAMES,
    assert_cast_without_lower_silently_zero_matches,
    assert_orm_bind_rejects_raw_value_string,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus, task_status_predicate
from xagent.web.services import task_lease_service
from xagent.web.services.task_execution_controller import control_state_for_status
from xagent.web.services.task_lease_service import (
    lease_checkpoint_event_id_case,
    lease_run_id_case,
    lease_state_version_case,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'task-status-storage.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_user(db) -> int:
    from xagent.web.models.user import User

    user = User(username="task-status-storage", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return int(user.id)


def _compiled(expr) -> str:
    return str(expr.compile(compile_kwargs={"literal_binds": True}))


# --------------------------------------------------------------------------
# Sentinel 1 (test domain): hand-written round-trip vs. drift check
# --------------------------------------------------------------------------


def test_task_status_round_trip_sqlite(db_session) -> None:
    """Every TaskStatus member persists as its name, not its value."""
    user_id = _create_user(db_session)
    for status, expected_name in TASK_STATUS_STORAGE_NAMES.items():
        task = Task(
            user_id=user_id,
            title=f"round-trip {status.name}",
            status=status,
        )
        db_session.add(task)
        db_session.commit()
        raw_value = db_session.execute(
            text("SELECT status FROM tasks WHERE id = :id"),
            {"id": task.id},
        ).scalar_one()
        assert raw_value == expected_name, (
            f"{status} stored as {raw_value!r}, expected member name {expected_name!r}"
        )


def test_storage_names_match_column_compilation() -> None:
    """Drift check: the hand-written pin still matches what the column
    actually compiles/binds, using only public SQLAlchemy type API
    (type.enums, bind_processor(dialect)) -- not a re-derivation used as
    the primary guard (that is TASK_STATUS_STORAGE_NAMES itself; see
    task_status_storage_shared.py for why the roles are not reversed).
    """
    column_type = Task.__table__.c.status.type
    assert set(column_type.enums) == {s.name for s in TaskStatus}, (
        "Enum(TaskStatus) DDL labels drifted from the TaskStatus members"
    )

    from sqlalchemy.dialects import postgresql, sqlite

    for dialect in (sqlite.dialect(), postgresql.dialect()):
        bind_processor = column_type.bind_processor(dialect)
        for status, expected_name in TASK_STATUS_STORAGE_NAMES.items():
            bound = bind_processor(status) if bind_processor else status
            assert bound == expected_name, (
                f"{dialect.name} bind_processor({status}) = {bound!r}, "
                f"expected {expected_name!r}"
            )


def test_member_name_lower_equals_value() -> None:
    """The control-state migration's LOWER(CAST(status AS VARCHAR)) form
    (src/xagent/migrations/versions/20260711_add_task_execution_control_state.py,
    lines 67-73) compares the column against each member's lowercased
    *value* -- e.g. "running" -- which only matches the stored uppercase
    *name* because name.lower() happens to equal value for every current
    member. This pin fails the day a future member breaks that coincidence,
    naming the mismatched member so the migration gets fixed instead of the
    assertion.
    """
    mismatched = [
        (status.name, status.name.lower(), status.value)
        for status in TaskStatus
        if status.name.lower() != status.value
    ]
    assert mismatched == [], (
        "TaskStatus member(s) with name.lower() != value: "
        f"{mismatched} -- this breaks the LOWER(CAST(status AS VARCHAR)) "
        "comparison in "
        "src/xagent/migrations/versions/20260711_add_task_execution_control_state.py"
    )


# --------------------------------------------------------------------------
# Sentinel 2 (DB layer, raw text()): wrong-case / cast-form literals
# --------------------------------------------------------------------------


def test_wrong_case_literal_query_zero_hits_sqlite(db_session) -> None:
    """A raw value-cased literal WHERE clause silently matches zero rows."""
    user_id = _create_user(db_session)
    task = Task(
        user_id=user_id,
        title="wrong-case literal",
        status=TaskStatus.WAITING_FOR_USER,
    )
    db_session.add(task)
    db_session.commit()

    wrong_case_rows = db_session.execute(
        text(RAW_VALUE_LITERAL_ZERO_MATCH_SQL),
        {"value": TaskStatus.WAITING_FOR_USER.value},
    ).fetchall()
    assert wrong_case_rows == [], (
        "a lowercase value literal must not match the stored uppercase "
        f"member name: {wrong_case_rows}"
    )

    correct_case_rows = db_session.execute(
        text(RAW_VALUE_LITERAL_ZERO_MATCH_SQL),
        {"value": TASK_STATUS_STORAGE_NAMES[TaskStatus.WAITING_FOR_USER]},
    ).fetchall()
    assert correct_case_rows == [(task.id,)], (
        "positive control failed: the correct-case member name should "
        f"match: {correct_case_rows}"
    )


def test_cast_without_lower_silently_zero_matches_sqlite(db_session) -> None:
    user_id = _create_user(db_session)
    task = Task(
        user_id=user_id,
        title="cast-without-lower",
        status=TaskStatus.RUNNING,
    )
    db_session.add(task)
    db_session.commit()
    assert_cast_without_lower_silently_zero_matches(
        db_session, stored_status=TaskStatus.RUNNING
    )


# --------------------------------------------------------------------------
# Sentinel 3 (bind layer, ORM/Core): validate_strings=True fails closed
# --------------------------------------------------------------------------


def test_orm_bind_rejects_raw_value_string_sqlite(db_session) -> None:
    user_id = _create_user(db_session)
    task = Task(user_id=user_id, title="bind-layer sentinel", status=TaskStatus.RUNNING)
    db_session.add(task)
    db_session.commit()
    assert_orm_bind_rejects_raw_value_string(db_session, task.id)


def test_poison_write_orm_rejected_raw_sql_still_poisons_sqlite(db_session) -> None:
    """Poison-write, both halves.

    Half A: the ORM write path is rejected (validate_strings=True) --
    reuses the bind-layer sentinel above on the real Task model.
    Half B: validate_strings only closes the ORM/Core bind path. A raw
    text() UPDATE still writes an unknown label straight into the column,
    and a subsequent ORM read of that row raises LookupError while trying
    to construct the TaskStatus member. Both halves are asserted here so
    the guarantee is not overstated as "the column cannot hold a bad value".
    """
    user_id = _create_user(db_session)
    task = Task(user_id=user_id, title="poison-write", status=TaskStatus.RUNNING)
    db_session.add(task)
    db_session.commit()
    task_id = task.id

    assert_orm_bind_rejects_raw_value_string(db_session, task_id)

    db_session.execute(
        text("UPDATE tasks SET status = :value WHERE id = :id"),
        {"value": TaskStatus.WAITING_FOR_USER.value, "id": task_id},
    )
    db_session.commit()

    raw_value = db_session.execute(
        text("SELECT status FROM tasks WHERE id = :id"), {"id": task_id}
    ).scalar_one()
    assert raw_value == TaskStatus.WAITING_FOR_USER.value, (
        f"raw SQL must actually have written the poisoned value -- got {raw_value!r}"
    )

    db_session.expire_all()
    with pytest.raises(LookupError):
        db_session.get(Task, task_id).status  # noqa: B018 -- triggers the ORM load


def test_poison_write_control_throwaway_model_without_validate_strings(
    tmp_path,
) -> None:
    """Control for the poison-write test above.

    Once Task.status has validate_strings=True, the "before" behavior can
    no longer be demonstrated on the real Task column. A throwaway model
    uses the pre-fix declaration -- Enum(TaskStatus), no validate_strings --
    and three assertions bracket what the fix changed:

    1. The ORM write of the lowercase enum *value* succeeds, explicitly
       asserted rather than left to an uncaught exception, so a future
       SQLAlchemy version that starts validating turns this assertion red
       instead of silently making the control vacuous.
    2. The raw stored value is the poisoned lowercase string
       ("waiting_for_user"), not the member name.
    3. A subsequent ORM read of that row raises LookupError -- the
       identical failure the real Task.status column now prevents at
       write time, so this pair of tests brackets the fix.
    """
    ThrowawayBase = declarative_base()

    class _LegacyStyleThrowawayTask(ThrowawayBase):  # type: ignore[misc, valid-type]
        __tablename__ = "throwaway_legacy_style_tasks"
        id = Column(Integer, primary_key=True)
        status = Column(Enum(TaskStatus))

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as OrmSession

    engine = create_engine(f"sqlite:///{tmp_path / 'throwaway.db'}")
    ThrowawayBase.metadata.create_all(engine)
    try:
        with OrmSession(engine) as db:
            db.add(_LegacyStyleThrowawayTask(id=1, status="RUNNING"))
            db.commit()
            # No validate_strings on this throwaway column -- the write
            # below must succeed where the real Task.status column (with
            # validate_strings=True) raises StatementError.
            try:
                db.execute(
                    update(_LegacyStyleThrowawayTask)
                    .where(_LegacyStyleThrowawayTask.id == 1)
                    .values(status=TaskStatus.WAITING_FOR_USER.value)
                )
                db.commit()
            except Exception as exc:
                pytest.fail(
                    "control write must succeed without validate_strings -- "
                    "if this raises, the control no longer demonstrates the "
                    f"pre-fix behavior it exists to bracket: {exc!r}"
                )
            stored = db.execute(
                text("SELECT status FROM throwaway_legacy_style_tasks WHERE id = 1")
            ).scalar_one()
            assert stored == TaskStatus.WAITING_FOR_USER.value, (
                "control case must show the pre-fix write succeeding "
                f"silently with the poisoned lowercase value; got {stored!r}"
            )
            db.expire_all()
            with pytest.raises(LookupError):
                db.get(_LegacyStyleThrowawayTask, 1).status  # noqa: B018
    finally:
        ThrowawayBase.metadata.drop_all(engine)


# --------------------------------------------------------------------------
# Predicate <-> raw-comparison compiled-SQL equivalence, one per real site
# --------------------------------------------------------------------------
# Mutation check: flipping task_status_predicate.eq to return
# `Task.status == status.value` (lowercase) turns
# test_predicate_eq_matches_raw_comparison and
# test_where_eq_matches_raw_comparison red, because the compiled literal no
# longer matches the raw-literal reference on the right-hand side. The
# round-trip sentinels are unaffected by that mutation: they write through
# the ORM and never call the binding, so equivalence coverage here is what
# catches an eq regression, not the round trip.


def test_predicate_eq_matches_raw_comparison() -> None:
    raw = Task.status == TaskStatus.RUNNING
    bound = task_status_predicate.eq(TaskStatus.RUNNING)
    assert _compiled(raw) == _compiled(bound) == "tasks.status = 'RUNNING'"


def test_predicate_ne_matches_raw_comparison() -> None:
    raw = Task.status != TaskStatus.RUNNING
    bound = task_status_predicate.ne(TaskStatus.RUNNING)
    assert _compiled(raw) == _compiled(bound) == "tasks.status != 'RUNNING'"


def test_predicate_ne_accepts_a_runtime_status_variable() -> None:
    """The ne() form must work with a runtime TaskStatus, not just a
    literal constant -- release_task_lease_no_commit and
    release_current_runner_task_lease both call ne(status) where status is
    a caller-supplied parameter.
    """
    for status in TaskStatus:
        raw = Task.status != status
        bound = task_status_predicate.ne(status)
        assert _compiled(raw) == _compiled(bound)


def test_predicate_in_single_member_collapses_to_eq() -> None:
    raw = Task.status == TaskStatus.RUNNING
    bound = task_status_predicate.in_([TaskStatus.RUNNING])
    assert _compiled(raw) == _compiled(bound)


def test_predicate_in_multi_member_uses_in_clause() -> None:
    raw = Task.status.in_([TaskStatus.RUNNING, TaskStatus.PENDING])
    bound = task_status_predicate.in_([TaskStatus.RUNNING, TaskStatus.PENDING])
    assert (
        _compiled(raw) == _compiled(bound) == "tasks.status IN ('RUNNING', 'PENDING')"
    )


def test_predicate_not_in_single_member_collapses_to_ne() -> None:
    raw = Task.status != TaskStatus.RUNNING
    bound = task_status_predicate.not_in([TaskStatus.RUNNING])
    assert _compiled(raw) == _compiled(bound)


def test_predicate_not_in_multi_member_uses_not_in_clause() -> None:
    raw = Task.status.notin_([TaskStatus.RUNNING, TaskStatus.PENDING])
    bound = task_status_predicate.not_in([TaskStatus.RUNNING, TaskStatus.PENDING])
    assert _compiled(raw) == _compiled(bound)


def test_predicate_in_empty_raises() -> None:
    with pytest.raises(ValueError):
        task_status_predicate.in_([])


def test_predicate_not_in_empty_raises() -> None:
    with pytest.raises(ValueError):
        task_status_predicate.not_in([])


@pytest.mark.parametrize(
    "method,args",
    [
        ("eq", ("running",)),
        ("ne", ("running",)),
        ("in_", (["running"],)),
        ("not_in", (["running"],)),
        ("value", ("running",)),
    ],
)
def test_predicate_rejects_non_task_status_input(method, args) -> None:
    with pytest.raises(TypeError):
        getattr(task_status_predicate, method)(*args)


def test_predicate_value_passes_through_a_valid_status() -> None:
    assert task_status_predicate.value(TaskStatus.FAILED) is TaskStatus.FAILED


def test_predicate_in_rejects_none_with_a_descriptive_error() -> None:
    with pytest.raises(TypeError, match="wrap a single member in a list"):
        task_status_predicate.in_(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="wrap a single member in a list"):
        task_status_predicate.not_in(None)  # type: ignore[arg-type]


# --- Full-statement equivalence for the 3 UPDATE SET-case builders --------
# lease_run_id_case / lease_state_version_case / lease_checkpoint_event_id_case
# are imported straight from task_lease_service.py -- these tests are a
# deliberate second consumer of the production expression, not a hand-copied
# reimplementation of it. Each diffs the complete compiled UPDATE, not an
# isolated boolean expression, since the case()'s THEN/ELSE wiring is part of
# what "equivalent" means here.


def test_lease_run_id_case_matches_raw_comparison() -> None:
    """acquire_task_lease_no_commit's run_id SET case."""
    candidate_run_id = "candidate-run-id"
    stmt_raw = (
        update(Task)
        .where(Task.id == 1)
        .values(
            run_id=case(
                (Task.status != TaskStatus.RUNNING, candidate_run_id),
                else_=func.coalesce(Task.run_id, candidate_run_id),
            )
        )
    )
    stmt_bound = (
        update(Task)
        .where(Task.id == 1)
        .values(run_id=lease_run_id_case(candidate_run_id))
    )
    assert _compiled(stmt_raw) == _compiled(stmt_bound)


@pytest.mark.parametrize(
    "status",
    [TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.FAILED, TaskStatus.COMPLETED],
)
def test_lease_state_version_case_matches_raw_comparison(status) -> None:
    """The state_version SET case shared by acquire_task_lease_no_commit and
    both release sites (release_task_lease_no_commit,
    release_current_runner_task_lease) -- one row per (status, control_state)
    pair actually reachable in production: RUNNING for the acquire site,
    PAUSED/FAILED/COMPLETED for the two release sites.
    """
    control_state = control_state_for_status(status).value
    current_version = func.coalesce(Task.state_version, 0)
    stmt_raw = (
        update(Task)
        .where(Task.id == 1)
        .values(
            state_version=case(
                (
                    or_(
                        Task.status != status,
                        Task.control_state != control_state,
                    ),
                    current_version + 1,
                ),
                else_=current_version,
            )
        )
    )
    stmt_bound = (
        update(Task)
        .where(Task.id == 1)
        .values(
            state_version=lease_state_version_case(
                status, control_state, current_version
            )
        )
    )
    assert _compiled(stmt_raw) == _compiled(stmt_bound)


def test_lease_checkpoint_event_id_case_matches_raw_comparison() -> None:
    """acquire_task_lease_no_commit's last_checkpoint_event_id SET case."""
    stmt_raw = (
        update(Task)
        .where(Task.id == 1)
        .values(
            last_checkpoint_event_id=case(
                (
                    or_(Task.status != TaskStatus.RUNNING, Task.run_id.is_(None)),
                    None,
                ),
                else_=Task.last_checkpoint_event_id,
            )
        )
    )
    stmt_bound = (
        update(Task)
        .where(Task.id == 1)
        .values(last_checkpoint_event_id=lease_checkpoint_event_id_case())
    )
    assert _compiled(stmt_raw) == _compiled(stmt_bound)


_LEASE_CASE_BUILDER_NAMES = {
    "lease_run_id_case",
    "lease_state_version_case",
    "lease_checkpoint_event_id_case",
    "lease_checkpoint_trace_event_id_case",
}


def _called_name(func: ast.expr) -> str | None:
    """The callable name for a ``Call.func`` node: a bare ``case`` name, or
    the attribute tail of a qualified call such as ``sa.case``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _case_call_names(tree: ast.AST) -> set[str]:
    """``case`` plus any module-local alias bound by a
    ``from sqlalchemy[.x] import case as X`` import in the given tree."""
    names = {"case"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "sqlalchemy"
        ):
            for alias in node.names:
                if alias.name == "case":
                    names.add(alias.asname or alias.name)
    return names


def _stray_task_status_case_calls(
    source: str, builder_names: set[str]
) -> list[tuple[str | None, int]]:
    """``case()`` calls whose subtree references ``Task.status``, outside
    the given builder functions. Scoped to Task.status specifically -- a
    case() built on an unrelated column is not this binding's concern and
    must not be flagged.

    The callee is matched by name tail (via ``_called_name``), so both a
    qualified call (``sa.case``) and a module-local
    ``from sqlalchemy import case as X`` alias (resolved by
    ``_case_call_names`` from imports in this same tree) are recognised,
    not just a bare ``case`` reference. A hit is cleared when *any*
    enclosing function scope -- not just the innermost -- is one of the
    given builder names, so a helper nested inside a builder is not
    flagged. Accepted false-positive surface: attribute-tail matching means
    an unrelated ``anything.case(...)`` call is matched by name too; it
    only gets reported if `_references_task_status` also fires for it,
    which requires an unrelated ``.case()`` method that also touches
    ``Task.status`` -- a shape that would deserve a look anyway.
    """
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _enclosing_function_names(node: ast.AST) -> list[str]:
        names: list[str] = []
        current = parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(current.name)
            current = parents.get(current)
        return names

    def _references_task_status(node: ast.Call) -> bool:
        return any(
            isinstance(sub, ast.Attribute)
            and sub.attr == "status"
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "Task"
            for sub in ast.walk(node)
        )

    case_names = _case_call_names(tree)
    hits: list[tuple[str | None, int]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _called_name(node.func) in case_names):
            continue
        if not _references_task_status(node):
            continue
        enclosing = _enclosing_function_names(node)
        if builder_names.intersection(enclosing):
            continue
        hits.append((enclosing[0] if enclosing else None, node.lineno))
    return hits


def test_lease_case_builders_are_the_only_case_calls_in_the_lease_service() -> None:
    """Without this, someone can inline a fifth Task.status-referencing
    case() back into a statement and the imported-builder tests above stay
    green while production drifts -- the same failure mode this recoupling
    closes, one level up. Scoped to case() calls that reference
    Task.status: a future case() built on an unrelated column is not this
    binding's concern and must not fail this test (see the negative-control
    test below).
    """
    source = Path(task_lease_service.__file__).read_text()
    stray = _stray_task_status_case_calls(source, _LEASE_CASE_BUILDER_NAMES)
    assert stray == [], (
        f"case() referencing Task.status called outside the lease builders: {stray}"
    )


def test_lease_case_ban_ignores_case_calls_on_unrelated_columns() -> None:
    """Negative control: a case() outside the lease builders that never
    touches Task.status must not be flagged by the check above.
    """
    fixture = """
from sqlalchemy import case

def some_unrelated_helper():
    return case((Task.runner_id == "x", 1), else_=0)
"""
    stray = _stray_task_status_case_calls(fixture, _LEASE_CASE_BUILDER_NAMES)
    assert stray == [], stray


@pytest.mark.parametrize(
    "case_id,fixture",
    [
        (
            "qualified",
            """
import sqlalchemy as sa

def h():
    return sa.case((Task.status != "running", 1), else_=0)
""",
        ),
        (
            "aliased",
            """
from sqlalchemy import case as sql_case

def h():
    return sql_case((Task.status != "running", 1), else_=0)
""",
        ),
    ],
    ids=["qualified", "aliased"],
)
def test_lease_case_ban_flags_qualified_and_aliased_case_calls(
    case_id: str, fixture: str
) -> None:
    """A qualified ``sa.case(...)`` and a
    ``from sqlalchemy import case as sql_case`` alias must both be
    recognised by the lease case() ban -- matching the bare ``case`` name
    alone would still miss the aliased form.
    """
    stray = _stray_task_status_case_calls(fixture, _LEASE_CASE_BUILDER_NAMES)
    assert stray == [("h", 5)], (case_id, stray)


def test_lease_case_ban_allows_helpers_nested_in_the_lease_builders() -> None:
    """A helper function nested *inside* one of the lease builders
    must not be flagged -- this requires checking every enclosing scope,
    not just the innermost one.
    """
    fixture = """
from sqlalchemy import case

def lease_run_id_case(candidate_run_id):
    def _helper():
        return case((Task.status != "RUNNING", 1), else_=0)
    return _helper()
"""
    stray = _stray_task_status_case_calls(fixture, _LEASE_CASE_BUILDER_NAMES)
    assert stray == [], stray


def test_lease_case_ban_flags_helpers_nested_outside_the_lease_builders() -> None:
    """A helper nested inside a *non*-builder function must still be
    flagged -- without this, "check all enclosing scopes" could be
    implemented as "never flag anything nested" and stay green.
    """
    fixture = """
from sqlalchemy import case

def some_helper():
    def _inner():
        return case((Task.status != "RUNNING", 1), else_=0)
    return _inner()
"""
    stray = _stray_task_status_case_calls(fixture, _LEASE_CASE_BUILDER_NAMES)
    assert stray == [("_inner", 6)], stray


# --- WHERE-clause equivalence for the 6 WHERE sites (constants, not case) -


def test_where_eq_matches_raw_comparison() -> None:
    """The four production WHERE sites that compare
    Task.status == TaskStatus.RUNNING --
    _expired_task_lease_candidates_query, recover_expired_task_lease_no_commit,
    _refresh_task_lease_no_commit, and fail_and_release_task_lease_no_commit
    (all in task_lease_service.py) -- are, verbatim, the same
    task_status_predicate.eq(TaskStatus.RUNNING) call; there is no per-site
    expression left to differentiate once that is true.
    """
    raw = Task.status == TaskStatus.RUNNING
    bound = task_status_predicate.eq(TaskStatus.RUNNING)
    assert _compiled(raw) == _compiled(bound)


def test_where_ne_site_or_clause_matches_raw_comparison() -> None:
    """acquire_task_lease_no_commit -- the ne() inside the or_() that gates
    whether the lease can be taken.
    """
    stmt_raw = update(Task).where(
        Task.id == 1,
        or_(
            Task.status != TaskStatus.RUNNING,
            Task.runner_id == "runner",
        ),
    )
    stmt_bound = update(Task).where(
        Task.id == 1,
        or_(
            task_status_predicate.ne(TaskStatus.RUNNING),
            Task.runner_id == "runner",
        ),
    )
    assert _compiled(stmt_raw) == _compiled(stmt_bound)


def test_where_ne_new_run_site_matches_raw_comparison() -> None:
    """acquire_task_lease_no_commit -- the standalone new_run guard,
    appended as its own .where() on an existing statement rather than
    nested in an or_(). Different composition from the site above, so it
    gets its own full-statement diff instead of being assumed equivalent by
    shape.
    """
    stmt_raw = update(Task).where(Task.id == 1)
    stmt_raw = stmt_raw.where(Task.status != TaskStatus.RUNNING)
    stmt_bound = update(Task).where(Task.id == 1)
    stmt_bound = stmt_bound.where(task_status_predicate.ne(TaskStatus.RUNNING))
    assert _compiled(stmt_raw) == _compiled(stmt_bound)


def test_monitor_active_agents_in_site_matches_raw_comparison() -> None:
    """monitor.py get_dashboard_stats -- the active_agents query."""
    raw = Task.status.in_(["RUNNING", "PENDING"])
    bound = task_status_predicate.in_([TaskStatus.RUNNING, TaskStatus.PENDING])
    assert (
        _compiled(raw) == _compiled(bound) == "tasks.status IN ('RUNNING', 'PENDING')"
    )


def test_in_rejects_a_bare_member_with_a_descriptive_error() -> None:
    with pytest.raises(TypeError, match="wrap a single member in a list"):
        task_status_predicate.in_(TaskStatus.RUNNING)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="wrap a single member in a list"):
        task_status_predicate.not_in("RUNNING")  # type: ignore[arg-type]
