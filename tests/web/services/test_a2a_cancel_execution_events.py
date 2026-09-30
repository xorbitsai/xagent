"""A2A cancellation records its outcome with the state it settles."""

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models.agent import Agent
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import a2a_task_cancel, task_execution
from xagent.web.services import task_execution_event_writer as writer
from xagent.web.services.task_lease_service import acquire_task_lease
from xagent.web.services.task_orchestrator import settle_task_lease_isolated

engine = engine_fixture
task_id = task_id_fixture


@pytest.fixture
def a2a_task(engine, task_id, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    factory = sessionmaker(engine)
    monkeypatch.setattr(a2a_task_cancel, "get_session_local", lambda: factory)
    monkeypatch.setattr("xagent.web.models.database.get_session_local", lambda: factory)
    manager = task_execution.BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    with factory() as db:
        task = db.get(Task, task_id)
        agent = Agent(user_id=task.user_id, name="A2A cancel")
        db.add(agent)
        db.flush()
        task.agent_id = agent.id
        task.source = "a2a"
        task.conversation_storage_version = 2
        task.status = TaskStatus.PAUSED
        task.control_state = "paused"
        task.run_id = "cancel-run"
        task.state_version = 3
        task.output = "prior output"
        task.agent_config = {"kept": "value"}
        db.commit()
        target = dict(
            task_id=task_id,
            agent_id=agent.id,
            expected_run_id=task.run_id,
            expected_state_version=task.state_version,
        )
    return SimpleNamespace(factory=factory, target=target, manager=manager)


def outcomes(db, task_id):
    return list(
        db.scalars(
            select(TaskExecutionEvent).where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.kind == "execution_settled",
            )
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER])
@pytest.mark.parametrize("version", [1, 2])
async def test_idle_cancel_commits_one_result_and_replays(a2a_task, status, version):
    ctx = a2a_task
    task_id = ctx.target["task_id"]
    with ctx.factory() as db:
        task = db.get(Task, task_id)
        task.status = status
        task.control_state = status.value
        task.conversation_storage_version = version
        db.commit()
    for _ in range(2):
        snapshot = await a2a_task_cancel.cancel_a2a_task(**ctx.target)
        assert snapshot.status == TaskStatus.FAILED
        assert snapshot.agent_config["a2a_state"] == "TASK_STATE_CANCELED"
    with ctx.factory() as db:
        task = db.get(Task, task_id)
        assert task.state_version == 4
        assert task.output is None
        assert task.agent_config == {
            "kept": "value",
            "a2a_state": "TASK_STATE_CANCELED",
        }
        rows = outcomes(db, task_id)
        if version == 1:
            assert rows == []
        else:
            assert len(rows) == 1
            assert rows[0].run_id == "cancel-run"
            assert rows[0].payload == {
                "status": "failed",
                "result": {
                    "status": "cancelled",
                    "error": "Task canceled by A2A client.",
                },
            }


@pytest.mark.asyncio
async def test_cancel_fact_failure_rolls_back_state_and_retry_succeeds(
    a2a_task, monkeypatch
):
    ctx = a2a_task
    append = writer.append_task_execution_event_no_commit

    def fail_after_append(*args, **kwargs):
        append(*args, **kwargs)
        raise OSError("cancel fact unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(
            writer, "append_task_execution_event_no_commit", fail_after_append
        )
        with pytest.raises(OSError, match="cancel fact unavailable"):
            await a2a_task_cancel.cancel_a2a_task(**ctx.target)
    with ctx.factory() as db:
        task = db.get(Task, ctx.target["task_id"])
        assert task.status == TaskStatus.PAUSED
        assert task.control_state == "paused"
        assert task.state_version == 3
        assert task.output == "prior output"
        assert task.agent_config == {"kept": "value"}
        assert task.conversation_event_sequence == 0
        assert outcomes(db, task.id) == []
    await a2a_task_cancel.cancel_a2a_task(**ctx.target)
    with ctx.factory() as db:
        assert len(outcomes(db, ctx.target["task_id"])) == 1


@pytest.mark.asyncio
async def test_live_cancel_keeps_execution_finalizers_result(a2a_task):
    ctx = a2a_task
    with ctx.factory() as db:
        lease = acquire_task_lease(db, ctx.target["task_id"], new_run=True)
        task = db.get(Task, lease.task_id)
        target = {
            **ctx.target,
            "expected_run_id": task.run_id,
            "expected_state_version": task.state_version,
        }
    started = asyncio.Event()

    async def execute():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert await asyncio.to_thread(
                settle_task_lease_isolated,
                lease,
                error_message="resume execution cancelled",
            )
            raise

    running = asyncio.create_task(execute())
    ctx.manager.register_task(lease.task_id, running)
    try:
        await started.wait()
        snapshot = await a2a_task_cancel.cancel_a2a_task(**target)
        assert snapshot.agent_config["a2a_state"] == "TASK_STATE_CANCELED"
        await a2a_task_cancel.cancel_a2a_task(**target)
        with ctx.factory() as db:
            rows = outcomes(db, lease.task_id)
            assert len(rows) == 1
            assert rows[0].payload == {
                "status": "failed",
                "result": {"error": "resume execution cancelled"},
            }
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
