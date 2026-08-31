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

from dataclasses import dataclass
from typing import Any

from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD
from xagent.web.services.trace_event_staging import (
    CHECKPOINT_ROW_BUILD_SCOPE,
    CHECKPOINT_ROW_CHECKPOINT_TYPE,
    CHECKPOINT_ROW_CONDITIONS,
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


def test_row_condition_constants_are_registered() -> None:
    """Cell 13: guards against adding a condition constant without also
    listing it in ``CHECKPOINT_ROW_CONDITIONS``."""

    assert set(CHECKPOINT_ROW_CONDITIONS) == {
        CHECKPOINT_ROW_TASK_OWNERSHIP,
        CHECKPOINT_ROW_EVENT_TYPE,
        CHECKPOINT_ROW_BUILD_SCOPE,
        CHECKPOINT_ROW_CHECKPOINT_TYPE,
        CHECKPOINT_ROW_RUN_PARTITION,
        CHECKPOINT_ROW_EXECUTION_IDENTITY,
    }


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
