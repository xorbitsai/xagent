"""The trigger-run start bookkeeping must not overwrite a parked projection (#2177).

``_start_prepared_trigger_run_id`` schedules the turn through ``begin_turn`` and
only then calls ``_mark_trigger_run_started``. The worker can park the task inside
that window: ``finish_turn`` then projects the run to ``paused`` and releases the
lease. A start writer that stamps ``running`` unconditionally puts the run back to
``running`` -- and the parked task has no lease and no later transition left to
repair it, so #2177 stays reproducible through this ordering.

This file reproduces that interleaving with the production start path and a real
database (``_with_session`` resolves the application's session factory, so the
rows have to live there rather than in a test-private engine).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.shared.db_teardown import drop_all_tables
from xagent.web.models.agent import Agent
from xagent.web.models.database import get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.services import triggers as triggers_module
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    sync_trigger_run_status,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'trigger-start-park.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        drop_all_tables(get_engine())


def _prepare_run(db) -> tuple[AgentTrigger, TriggerRun]:
    """Create one prepared trigger run plus its pending task, through production code."""

    user = User(username="start-park-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    agent = Agent(user_id=int(user.id), name="start-park agent")
    db.add(agent)
    db.commit()
    trigger = AgentTrigger(
        user_id=int(user.id),
        agent_id=int(agent.id),
        type=TriggerType.WEBHOOK.value,
        name="start-park trigger",
        config={},
    )
    db.add(trigger)
    db.commit()

    run, created = triggers_module.prepare_trigger_run(
        db,
        trigger=trigger,
        event_payload={"subject": "park during start"},
        source_event_id="start-park-event",
    )
    assert created is True
    assert run.task_id is not None
    return trigger, run


def _park_the_task(db, task_id: int) -> None:
    """Park the task and project it the way the park path does."""

    task = db.query(Task).filter(Task.id == task_id).one()
    task.status = TaskStatus.WAITING_FOR_USER
    task.control_state = "waiting_for_user"
    db.add(task)
    db.commit()
    db.refresh(task)
    assert sync_trigger_run_status(db, task, TaskStatus.WAITING_FOR_USER) is True
    db.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_for_completion", [False, True])
async def test_start_bookkeeping_does_not_overwrite_a_parked_run(
    db_session, monkeypatch, wait_for_completion
):
    """REPRO (#2177): the worker parks during begin_turn, before start bookkeeping."""

    trigger, run = _prepare_run(db_session)
    run_id = int(run.id)
    task_id = int(run.task_id)

    async def park_during_begin_turn(**_kwargs):
        # The worker runs to a stop before _mark_trigger_run_started is reached.
        _park_the_task(db_session, task_id)
        # A finished future keeps the wait_for_completion path deterministic:
        # the caller awaits it instead of waiting on the still-parked task.
        completed = asyncio.get_running_loop().create_future()
        completed.set_result(None)
        return SimpleNamespace(background_task=completed, task_id=task_id, run_id=None)

    monkeypatch.setattr(
        triggers_module.TaskTurnOrchestrator, "begin_turn", park_during_begin_turn
    )

    started = await triggers_module.start_prepared_trigger_run(
        db_session, run=run, wait_for_completion=wait_for_completion
    )
    assert started is True

    db_session.expire_all()
    stored_run = db_session.get(TriggerRun, run_id)
    assert stored_run.status == TriggerRunStatus.PAUSED.value, (
        "start bookkeeping overwrote the parked projection: the run reads as "
        "'running' again, and its parked task has no lease or later transition "
        "left to repair it (#2177)"
    )
    assert stored_run.finished_at is None

    # The bookkeeping the call exists for still happened.
    db_session.refresh(trigger)
    assert trigger.last_run_at is not None


def test_start_again_verdict_does_not_drag_an_advanced_run_back(db_session):
    """REPRO (#2177): the other start writer is a read-then-check.

    ``_mark_trigger_run_running_if_task_running`` reads the task, then writes the
    run unconditionally. The task check says "a turn owns this task", but the park
    can land between that read and the write, so the run write needs its own
    condition -- otherwise it resurrects a parked run.
    """

    trigger, run = _prepare_run(db_session)
    run_id = int(run.id)
    task_id = int(run.task_id)

    task = db_session.query(Task).filter(Task.id == task_id).one()
    task.status = TaskStatus.RUNNING
    db_session.add(task)
    parked = db_session.get(TriggerRun, run_id)
    parked.status = TriggerRunStatus.PAUSED.value
    db_session.add(parked)
    db_session.commit()

    marked = triggers_module._mark_trigger_run_running_if_task_running(run_id, task_id)

    assert marked is True, (
        "the caller uses this verdict to decide not to fail the run, so a running "
        "task must keep reporting True"
    )
    db_session.expire_all()
    stored = db_session.get(TriggerRun, run_id)
    assert stored.status == TriggerRunStatus.PAUSED.value, (
        "the start-again write dragged a parked run back to 'running' (#2177)"
    )


@pytest.mark.asyncio
async def test_rejected_start_does_not_fail_a_parked_run(db_session, monkeypatch):
    """REPRO (#2177): a rejected start must not terminalize a parked run.

    When ``begin_turn`` rejects the turn and the task is not running, the caller
    marks the run failed. If the task parked in the meantime, that failure is a
    terminal lie: the projection never re-selects terminal rows, so a later
    resume could not correct it.
    """

    trigger, run = _prepare_run(db_session)
    run_id = int(run.id)
    task_id = int(run.task_id)

    async def park_then_reject(**_kwargs):
        _park_the_task(db_session, task_id)
        raise TaskTurnError("task is owned by another turn")

    monkeypatch.setattr(
        triggers_module.TaskTurnOrchestrator, "begin_turn", park_then_reject
    )

    started = await triggers_module.start_prepared_trigger_run(db_session, run=run)
    assert started is False

    db_session.expire_all()
    stored = db_session.get(TriggerRun, run_id)
    assert stored.status == TriggerRunStatus.PAUSED.value, (
        "a rejected start marked the parked run failed; terminal rows are never "
        "re-selected, so a later resume could not correct it (#2177)"
    )
    assert stored.finished_at is None
