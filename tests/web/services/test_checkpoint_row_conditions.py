"""Unit cells for ``failed_checkpoint_row_conditions`` and
``is_missing_run_partition_only`` (``trace_event_staging.py``), the one
definition of "is this trace_events row a legitimate checkpoint for this
task, run and execution" that both by-primary-key resolvers read.

The predicate reads three row attributes and a payload dict and touches no
database, so these cells build a plain stand-in row rather than a session.
The two resolvers' own suites cover what each does with the answer; this
file covers the answer itself.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xagent
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD
from xagent.web.services.trace_event_staging import (
    CHECKPOINT_ROW_BUILD_SCOPE,
    CHECKPOINT_ROW_CHECKPOINT_TYPE,
    CHECKPOINT_ROW_EVENT_TYPE,
    CHECKPOINT_ROW_EXECUTION_IDENTITY,
    CHECKPOINT_ROW_RUN_PARTITION,
    CHECKPOINT_ROW_TASK_OWNERSHIP,
    failed_checkpoint_row_conditions,
    is_missing_run_partition_only,
)

_TASK_ID = 42
_RUN_ID = "run-a"
_EXECUTION_ID = "42"


@dataclass
class _StandInRow:
    """The three attributes ``failed_checkpoint_row_conditions`` reads off a
    ``trace_events`` row. Not a real ``DatabaseTraceEvent`` -- the predicate
    never touches anything else about the row."""

    task_id: int
    event_type: str
    build_id: str | None


def _row(**overrides: Any) -> _StandInRow:
    defaults: dict[str, Any] = {
        "task_id": _TASK_ID,
        "event_type": "system_update_general",
        "build_id": None,
    }
    defaults.update(overrides)
    return _StandInRow(**defaults)


def _data(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "checkpoint_type": "agent_execution_checkpoint",
        TASK_RUN_ID_TRACE_FIELD: _RUN_ID,
        "execution_id": _EXECUTION_ID,
    }
    defaults.update(overrides)
    return defaults


def _failed(row: _StandInRow, data: dict[str, Any], *, run_id: str | None = _RUN_ID):
    return failed_checkpoint_row_conditions(
        row,
        data,
        task_id=_TASK_ID,
        run_id=run_id,
        execution_id=_EXECUTION_ID,
    )


def test_all_six_conditions_pass() -> None:
    assert _failed(_row(), _data()) == frozenset()


def test_wrong_task_id_fails_ownership() -> None:
    row = _row(task_id=_TASK_ID + 1)
    assert _failed(row, _data()) == {CHECKPOINT_ROW_TASK_OWNERSHIP}


def test_wrong_event_type_fails_event_type() -> None:
    row = _row(event_type="agent_progress")
    assert _failed(row, _data()) == {CHECKPOINT_ROW_EVENT_TYPE}


def test_build_scoped_row_fails_build_scope() -> None:
    row = _row(build_id="agent_1_x")
    assert _failed(row, _data()) == {CHECKPOINT_ROW_BUILD_SCOPE}


def test_unreadable_checkpoint_type_fails_checkpoint_type() -> None:
    data = _data(checkpoint_type="something_else")
    assert _failed(_row(), data) == {CHECKPOINT_ROW_CHECKPOINT_TYPE}


def test_wrong_run_partition_value_fails_run_partition() -> None:
    data = _data(**{TASK_RUN_ID_TRACE_FIELD: "run-b"})
    assert _failed(_row(), data) == {CHECKPOINT_ROW_RUN_PARTITION}


def test_missing_run_partition_field_fails_run_partition() -> None:
    data = _data()
    del data[TASK_RUN_ID_TRACE_FIELD]
    assert _failed(_row(), data) == {CHECKPOINT_ROW_RUN_PARTITION}


def test_mismatched_execution_id_fails_execution_identity() -> None:
    data = _data(execution_id="not-the-task")
    assert _failed(_row(), data) == {CHECKPOINT_ROW_EXECUTION_IDENTITY}


def test_absent_execution_identity_is_lenient() -> None:
    """A legacy row carrying no identity field at all passes on purpose --
    this leniency is part of the condition, not a gap in it."""

    data = _data()
    del data["execution_id"]
    assert _failed(_row(), data) == frozenset()


def test_null_run_id_matched_by_absent_run_field_passes() -> None:
    """The root-checkpoint read path can genuinely have no run id yet: a
    null ``run_id`` is a legitimate partition, matched by a row whose own
    run field is also absent."""

    data = _data()
    del data[TASK_RUN_ID_TRACE_FIELD]
    assert _failed(_row(), data, run_id=None) == frozenset()


def test_null_run_id_not_matched_by_present_run_field() -> None:
    data = _data()
    assert _failed(_row(), data, run_id=None) == {CHECKPOINT_ROW_RUN_PARTITION}


def test_non_string_run_field_matches_through_string_coercion() -> None:
    """A run field stored as a non-string JSON scalar answers here the way it
    answers in SQL, where ``checkpoint_run_partition_filter`` reads the same
    field through ``.as_string()``. Nothing writes such a row today -- the
    writer always stores a str -- so this pins the two forms agreeing, not a
    shape in production."""

    data = _data(**{TASK_RUN_ID_TRACE_FIELD: 7})
    assert _failed(_row(), data, run_id="7") == frozenset()


def test_non_string_run_field_with_wrong_value_still_fails_run_partition() -> None:
    """The coercion above narrows nothing: a non-string run field that does
    not name this run still fails the partition condition."""

    data = _data(**{TASK_RUN_ID_TRACE_FIELD: 7})
    assert _failed(_row(), data) == {CHECKPOINT_ROW_RUN_PARTITION}


def test_two_conditions_can_fail_at_once() -> None:
    row = _row(task_id=_TASK_ID + 1)
    data = _data()
    del data[TASK_RUN_ID_TRACE_FIELD]
    assert _failed(row, data) == {
        CHECKPOINT_ROW_TASK_OWNERSHIP,
        CHECKPOINT_ROW_RUN_PARTITION,
    }


def test_only_run_partition_absent_is_true_for_the_absent_shape() -> None:
    data = _data()
    del data[TASK_RUN_ID_TRACE_FIELD]
    failed = _failed(_row(), data)
    assert is_missing_run_partition_only(failed, data) is True


def test_only_run_partition_absent_is_false_when_value_is_merely_wrong() -> None:
    data = _data(**{TASK_RUN_ID_TRACE_FIELD: "run-b"})
    failed = _failed(_row(), data)
    assert is_missing_run_partition_only(failed, data) is False


def test_only_run_partition_absent_is_false_when_something_else_also_fails() -> None:
    row = _row(task_id=_TASK_ID + 1)
    data = _data()
    del data[TASK_RUN_ID_TRACE_FIELD]
    failed = _failed(row, data)
    assert is_missing_run_partition_only(failed, data) is False


def test_explicit_json_null_run_field_reads_as_absent() -> None:
    """An explicitly stored JSON ``null`` is the same answer as a missing
    key here, and the docstring says so. Pinned rather than left implicit:
    a future tightening to ``not in row_data`` would move this row into the
    corrupt branch, which is a decision to make deliberately, not by
    accident."""

    data = _data(**{TASK_RUN_ID_TRACE_FIELD: None})
    failed = _failed(_row(), data)
    assert failed == {CHECKPOINT_ROW_RUN_PARTITION}
    assert is_missing_run_partition_only(failed, data) is True


_SHARED_PREDICATE = "failed_checkpoint_row_conditions"
_BY_PK_RESOLVERS = (
    ("xagent/web/api/trace_handlers.py", "_load_pk_anchored_checkpoint"),
    ("xagent/web/services/task_interaction_anchor.py", "resolve_interaction_anchor"),
)


def _imports(source: str, name: str) -> bool:
    """Whether the module imports ``name`` anywhere -- at the top or inside a
    function, since a caller may import through a function-level import to
    avoid a module cycle."""

    return any(
        isinstance(node, ast.ImportFrom)
        and any(alias.name == name for alias in node.names)
        for node in ast.walk(ast.parse(source))
    )


def _resolver_def(source: str, resolver: str) -> ast.AST:
    """The ``def`` node for ``resolver``, wherever it sits -- module level or
    inside a class body."""

    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == resolver
        ):
            return node
    raise AssertionError(f"no def named {resolver} in this module")


def _calls_within(node: ast.AST, name: str) -> bool:
    """Whether ``name`` is called anywhere inside ``node``'s own subtree.

    Scoped to the subtree on purpose: a call somewhere else in the same file
    says nothing about whether *this* resolver still reads the shared
    definition.
    """

    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False


def test_both_by_pk_resolvers_read_the_shared_predicate() -> None:
    """Neither by-primary-key resolver may go back to a private inlined copy
    of the row-validity conditions.

    This replaces the two AST pins that used to compare the two hand-copied
    disjunctions operand by operand. Those pins had a real blind spot -- they
    saw the operands but not how ``partition_matches`` was computed above
    them, which is where the two copies actually differed -- and they became
    meaningless once there was one definition rather than two. What still
    needs guarding is not that the copies agree but that no copy comes back,
    so this asserts reachability of the shared definition, not its text.

    The call check is scoped to each resolver's own ``def`` node rather than
    to the file. File scope would pass on a module that inlines a private
    disjunction inside the resolver while some unrelated call to the shared
    predicate survives elsewhere in the same file -- which is the whole shape
    this test exists to catch.

    Not extended to lease recovery's own consumer
    (``task_lease_service._candidate_row_failures``): its behaviour is pinned
    directly by ``test_task_lease_recovery.py``, which is the stronger check
    of the two where it is available.
    """

    root = Path(next(iter(xagent.__path__))).parent
    for relative, resolver in _BY_PK_RESOLVERS:
        path = root / relative
        assert path.is_file(), path
        source = path.read_text(encoding="utf-8")
        assert _imports(source, _SHARED_PREDICATE), (
            f"{relative} no longer imports {_SHARED_PREDICATE}"
        )
        assert _calls_within(_resolver_def(source, resolver), _SHARED_PREDICATE), (
            f"{relative}::{resolver} no longer calls {_SHARED_PREDICATE}"
        )
