"""The ``execution_settled`` fact of a run does not carry the structured
result; ``tasks.structured_output_record`` is its only copy."""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest
import sqlalchemy as sa

from tests.web.services import test_structured_output_wiring as wiring
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent.structured_output import STRUCTURED_OUTPUT_RESULT_KEY
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import task_event_context_service, task_execution
from xagent.web.services.task_execution import (
    _finalize_resumed_task,
    _finalize_task_execution_result_isolated,
    _PreparedTaskFileOutputs,
)
from xagent.web.services.task_lease_service import TaskLease, acquire_task_lease

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture

_settlement_result = lambda result: task_execution._settlement_result(result)  # noqa: E731
NO_OUTPUTS = _PreparedTaskFileOutputs((), (), ())
READY = {"status": "ready", "output": {"name": "Bo"}, "reason": None}


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
def run_once(request, canonical):
    """A function that settles one more run of the task and returns the factory
    and the settlement facts written so far."""
    factory, tid = canonical
    finalize = FINALIZERS[request.param]

    def run(result: dict[str, Any]) -> None:
        with factory() as db:
            lease = acquire_task_lease(db, tid, new_run=True)
        assert lease is not None
        finalize(factory, tid, lease, result)

    def settled() -> list[TaskExecutionEvent]:
        with factory() as db:
            rows = db.scalars(
                sa.select(TaskExecutionEvent)
                .where(
                    TaskExecutionEvent.task_id == tid,
                    TaskExecutionEvent.kind == "execution_settled",
                )
                .order_by(TaskExecutionEvent.sequence)
            ).all()
            db.expunge_all()
            return list(rows)

    def record() -> Any:
        with factory() as db:
            return db.get(Task, tid).structured_output_record

    return factory, tid, run, settled, record


def completed(**extra: Any) -> dict[str, Any]:
    return {"status": "completed", "success": True, "output": "answer", **extra}


def test_the_settlement_fact_of_a_structured_run_has_no_structured_result(run_once):
    _factory, _tid, run, settled, record = run_once
    result = completed(**{STRUCTURED_OUTPUT_RESULT_KEY: READY})
    original = copy.deepcopy(result)

    run(result)

    assert record()["status"] == "ready"
    (fact,) = settled()
    stored = fact.payload["result"]
    assert STRUCTURED_OUTPUT_RESULT_KEY not in stored
    assert stored == {
        k: v for k, v in original.items() if k != STRUCTURED_OUTPUT_RESULT_KEY
    }
    assert result == original


def test_the_settlement_fact_of_a_run_without_a_format_is_unchanged(run_once):
    _factory, _tid, run, settled, record = run_once
    result = completed()

    run(result)

    assert record() is None
    (fact,) = settled()
    assert fact.payload == {"status": "completed", "result": result}


def test_the_read_back_settlements_of_two_structured_runs_carry_no_result(run_once):
    factory, tid, run, _settled, _record = run_once

    run(completed(**{STRUCTURED_OUTPUT_RESULT_KEY: READY}))
    run(completed(**{STRUCTURED_OUTPUT_RESULT_KEY: READY}))

    with factory() as db:
        events = task_event_context_service._root_events(
            db,
            tid,
            after=0,
            through=10**9,
            kinds=frozenset({"execution_settled"}),
        )
    assert len(events) == 2
    for event in events:
        assert STRUCTURED_OUTPUT_RESULT_KEY not in event.payload["result"]


def test_settlement_result_returns_the_same_object_when_there_is_no_key():
    result = completed()

    assert _settlement_result(result) is result


def test_settlement_result_drops_the_key_from_a_copy():
    result = completed(**{STRUCTURED_OUTPUT_RESULT_KEY: READY})

    stripped = _settlement_result(result)

    assert stripped is not result
    assert STRUCTURED_OUTPUT_RESULT_KEY not in stripped
    assert stripped == completed()
    assert result[STRUCTURED_OUTPUT_RESULT_KEY] is READY


# --- a result the settlement must still be able to write ---------------------

_LIST_SCHEMA = {
    "type": "object",
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    "required": ["names"],
    "additionalProperties": False,
}


def _tuple_argument_call() -> dict[str, Any]:
    """A tool call whose arguments are a dict holding a tuple with a NUL."""
    return {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "submit_result",
                    "arguments": {"names": ("a\u0000",)},
                },
            }
        ],
    }


@pytest.mark.parametrize("path", ["new_turn", "resume"])
async def test_a_tuple_value_in_the_arguments_does_not_break_the_settlement(
    canonical, monkeypatch, path
) -> None:
    # ``canonical`` runs this on SQLite and on PostgreSQL. A jsonb column on
    # PostgreSQL, such as the payload of the settlement fact, refuses a NUL
    # and SQLite does not, so a NUL from these arguments that reached such a
    # write would break the settlement only on PostgreSQL.
    factory, _ = canonical
    config = wiring._config(wiring._stored(schema=copy.deepcopy(_LIST_SCHEMA)))
    llm = wiring.ScriptedModel([_tuple_argument_call() for _ in range(3)])
    run = wiring._run_new_turn if path == "new_turn" else wiring._run_resume

    with factory() as db:
        outcome = await run(db, monkeypatch, config, llm)
        task_id = outcome[0]
        status, record = wiring._read_task(db, task_id)
    assert status == TaskStatus.COMPLETED
    assert record["status"] == "failed"
    assert record["reason"]["code"] == "schema_mismatch"
    assert llm.calls == 3
