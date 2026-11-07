"""The two settlement functions store the structured result of a run in
``tasks.structured_output_record`` when, and only when, they settle it as
COMPLETED."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services.structured_output_projection import build_record
from xagent.web.services.task_execution import (
    _finalize_resumed_task,
    _finalize_task_execution_result_isolated,
    _PreparedTaskFileOutputs,
)
from xagent.web.services.task_lease_service import TaskLease, acquire_task_lease

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture

NO_OUTPUTS = _PreparedTaskFileOutputs((), (), ())
PREVIOUS = {"run_id": "an-earlier-run", "status": "ready", "output": {}, "reason": None}
READY = {"status": "ready", "output": {"b": 1, "a": 2}, "reason": None}
INCOMPLETE = {
    "status": "incomplete",
    "output": None,
    "reason": {
        "code": "information_missing",
        "missing_fields": ["/name"],
        "message": "free text that is never stored",
    },
}


@pytest.fixture(autouse=True)
def _late_bound_sessions(monkeypatch):
    from xagent.web.models import database

    monkeypatch.setattr(
        "xagent.web.services.task_execution.get_session_local",
        lambda: database.get_session_local(),
    )


def _finalize_new_turn(factory, tid: int, lease: TaskLease, result: dict[str, Any]):
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    return _finalize_task_execution_result_isolated(
        task_id=tid,
        task_user_id=user_id,
        pre_run_status=TaskStatus.RUNNING,
        result=result,
        expected_run_id=lease.run_id,
        task_lease=lease,
        resolved_scope_segments=(),
        prepared_outputs=NO_OUTPUTS,
    )


def _finalize_resume(factory, tid: int, lease: TaskLease, result: dict[str, Any]):
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    return _finalize_resumed_task(
        tid,
        status=str(result.get("status", "completed")),
        success=bool(result.get("success", False)),
        output=str(result.get("output", "")),
        task_owner_user_id=user_id,
        result=result,
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
    )


FINALIZERS: dict[str, Callable[..., Any]] = {
    "new_turn": _finalize_new_turn,
    "resume": _finalize_resume,
}


@pytest.fixture(params=sorted(FINALIZERS))
def settle(request, canonical):
    """(read_record, run_id, finalize) for a task running under a fresh lease."""
    factory, tid = canonical
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    assert lease is not None
    finalize = FINALIZERS[request.param]

    def set_record(value: Any) -> None:
        with factory() as db:
            db.get(Task, tid).structured_output_record = value
            db.commit()

    def read() -> Any:
        with factory() as db:
            return db.get(Task, tid).structured_output_record

    def run(result: dict[str, Any]) -> None:
        finalize(factory, tid, lease, result)

    def set_status(status: TaskStatus) -> None:
        with factory() as db:
            db.get(Task, tid).status = status
            db.commit()

    return lease.run_id, read, set_record, set_status, run


def completed(**extra: Any) -> dict[str, Any]:
    return {"status": "completed", "success": True, "output": "answer", **extra}


def test_a_completed_run_stores_status_output_and_reason_under_its_run_id(settle):
    run_id, read, _set_record, _set_status, run = settle

    run(completed(structured_output=READY))

    assert read() == {
        "run_id": run_id,
        "status": "ready",
        "output": {"b": 1, "a": 2},
        "reason": None,
    }
    assert list(read()["output"]) == ["b", "a"]


def test_the_stored_reason_keeps_only_the_code_and_the_nonempty_field_lists(settle):
    run_id, read, _set_record, _set_status, run = settle

    run(completed(structured_output=INCOMPLETE))

    assert read() == {
        "run_id": run_id,
        "status": "incomplete",
        "output": None,
        "reason": {"code": "information_missing", "missing_fields": ["/name"]},
    }


def test_a_completed_run_without_a_structured_output_clears_the_column(settle):
    """The same run first stored a value; the next settlement overwrites it."""
    run_id, read, set_record, _set_status, run = settle
    set_record({**PREVIOUS, "run_id": run_id})

    run(completed())

    assert read() is None


def test_a_completed_run_replaces_a_record_with_the_new_one(settle):
    run_id, read, set_record, _set_status, run = settle
    set_record({**PREVIOUS, "run_id": run_id})

    run(completed(structured_output=INCOMPLETE))

    assert read()["status"] == "incomplete"


@pytest.mark.parametrize(
    "result",
    [
        {"status": "failed", "success": False, "error": "boom"},
        {"status": "waiting_for_user", "success": False, "output": ""},
        {"status": "interrupted", "success": False, "output": ""},
    ],
    ids=["failed", "waiting", "interrupted"],
)
def test_a_run_that_does_not_complete_leaves_the_column_alone(settle, result):
    _run_id, read, set_record, _set_status, run = settle
    set_record(PREVIOUS)

    run({**result, "structured_output": READY})

    assert read() == PREVIOUS


@pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.COMPLETED])
def test_a_late_result_for_a_task_settled_elsewhere_leaves_the_column_alone(
    settle, status
):
    """A cancelled task is already FAILED when the in-flight result arrives."""
    _run_id, read, set_record, set_status, run = settle
    set_record(PREVIOUS)
    set_status(status)

    run(completed(structured_output=READY))

    assert read() == PREVIOUS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (
            "not a mapping",
            {
                "run_id": "run-1",
                "status": "failed",
                "output": None,
                "reason": {"code": "internal_error"},
            },
        ),
    ],
)
def test_build_record_stores_none_for_none_and_a_failed_record_for_a_non_mapping(
    value, expected
):
    assert build_record("run-1", value) == expected


def test_build_record_keeps_the_status_of_a_mapping():
    record = build_record("run-1", {"status": "ready", "output": {}, "reason": None})

    assert record["status"] == "ready"
