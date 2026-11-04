"""A resumed run's outcome reaches the trigger run it belongs to.

A user pause or a wait for the user leaves the task's ``TriggerRun`` RUNNING;
the run that later resumes it is settled by ``_finalize_resumed_task``, which
must end that trigger run the way ``finish_turn`` ends a new run's. Runs on
SQLite and, when ``XAGENT_TEST_POSTGRES_URL`` is set, on PostgreSQL through
the shared ``canonical`` fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.models.agent import Agent
from xagent.web.models.task import Task
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.services.task_execution import (
    _finalize_resumed_task,
    _PreparedTaskFileOutputs,
)
from xagent.web.services.task_lease_service import TaskLease, acquire_task_lease

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture

NO_OUTPUTS = _PreparedTaskFileOutputs((), (), ())


@pytest.fixture(autouse=True)
def _late_bound_sessions(monkeypatch):
    """``task_execution`` binds the session factory at import; follow the
    one the ``canonical`` fixture installs instead."""
    from xagent.web.models import database

    monkeypatch.setattr(
        "xagent.web.services.task_execution.get_session_local",
        lambda: database.get_session_local(),
    )


def _trigger_run(factory, tid: int, status: TriggerRunStatus) -> int:
    with factory() as db:
        task = db.get(Task, tid)
        task.source = "trigger"
        agent = Agent(user_id=task.user_id, name=f"Resume agent {tid}")
        db.add(agent)
        db.flush()
        trigger = AgentTrigger(
            user_id=task.user_id,
            agent_id=agent.id,
            type=TriggerType.SCHEDULED.value,
            name=f"Resume trigger {tid}",
            config={},
        )
        db.add(trigger)
        db.flush()
        run = TriggerRun(
            trigger_id=trigger.id,
            task_id=tid,
            status=status.value,
            idempotency_key=f"resume-projection-{tid}",
        )
        db.add(run)
        db.commit()
        return int(run.id)


def _resume(factory, tid: int) -> TaskLease:
    with factory() as db:
        lease = acquire_task_lease(db, tid, new_run=True)
    assert lease is not None
    return lease


def _finalize(factory, tid: int, lease: TaskLease, **outcome: Any) -> dict[str, Any]:
    with factory() as db:
        user_id = int(db.get(Task, tid).user_id)
    return _finalize_resumed_task(
        tid,
        task_owner_user_id=user_id,
        task_lease=lease,
        prepared_outputs=NO_OUTPUTS,
        **outcome,
    )


def _trigger_run_state(factory, run_id: int) -> tuple[str, str | None, bool]:
    with factory() as db:
        run = db.get(TriggerRun, run_id)
        return run.status, run.error_message, run.finished_at is not None


def test_completed_resume_completes_the_trigger_run(canonical):
    factory, tid = canonical
    run_id = _trigger_run(factory, tid, TriggerRunStatus.RUNNING)
    lease = _resume(factory, tid)

    finalized = _finalize(
        factory,
        tid,
        lease,
        status="completed",
        success=True,
        output="the answer",
        result={"status": "completed", "success": True, "output": "the answer"},
    )

    assert finalized["final_status"] == "completed"
    assert _trigger_run_state(factory, run_id) == (
        TriggerRunStatus.COMPLETED.value,
        None,
        True,
    )


def test_failed_resume_fails_the_trigger_run(canonical):
    factory, tid = canonical
    run_id = _trigger_run(factory, tid, TriggerRunStatus.RUNNING)
    lease = _resume(factory, tid)

    finalized = _finalize(
        factory,
        tid,
        lease,
        status="failed",
        success=False,
        output="tool exploded",
        result={"status": "failed", "success": False, "error": "tool exploded"},
    )

    assert finalized["final_status"] == "failed"
    with factory() as db:
        task_error = db.get(Task, tid).error_message
    status, error, finished = _trigger_run_state(factory, run_id)
    assert (status, finished) == (TriggerRunStatus.FAILED.value, True)
    assert error == task_error and error


@pytest.mark.parametrize(
    ("status", "final_status"),
    [("waiting_for_user", "waiting_for_user"), ("interrupted", "paused")],
)
def test_resume_that_stops_again_keeps_the_trigger_run_running(
    canonical, status, final_status
):
    factory, tid = canonical
    run_id = _trigger_run(factory, tid, TriggerRunStatus.RUNNING)
    lease = _resume(factory, tid)

    finalized = _finalize(
        factory,
        tid,
        lease,
        status=status,
        success=False,
        output="",
        result={"status": status, "success": False},
    )

    assert finalized["final_status"] == final_status
    assert _trigger_run_state(factory, run_id) == (
        TriggerRunStatus.RUNNING.value,
        None,
        False,
    )


def test_settled_trigger_run_is_left_alone(canonical):
    """A trigger run another path already settled keeps its outcome."""
    factory, tid = canonical
    run_id = _trigger_run(factory, tid, TriggerRunStatus.FAILED)
    lease = _resume(factory, tid)

    _finalize(
        factory,
        tid,
        lease,
        status="completed",
        success=True,
        output="the answer",
        result={"status": "completed", "success": True, "output": "the answer"},
    )

    assert _trigger_run_state(factory, run_id) == (
        TriggerRunStatus.FAILED.value,
        None,
        False,
    )
