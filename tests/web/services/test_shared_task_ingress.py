"""Non-channel ingress crosses START and keeps its response contract."""

import asyncio
from unittest.mock import Mock

import pytest

from xagent.web.models.agent import Agent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_event_bridge, task_orchestrator, task_start
from xagent.web.services.task_completion import TaskRunChanged, _is_run_finished


@pytest.fixture
def ingress(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(
        task_orchestrator,
        "_schedule_bg",
        Mock(side_effect=AssertionError("ingress scheduled execution")),
    )
    init_db(db_url=f"sqlite:///{tmp_path / 'ingress.db'}")
    with get_session_local()() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        agent = Agent(user_id=owner.id, name="shared")
        db.add(agent)
        db.commit()
        ids = owner.id, agent.id
    yield ids
    Base.metadata.drop_all(bind=get_engine())


@pytest.mark.asyncio
async def test_sdk_create_and_append_return_durable_acceptance(ingress):
    owner, agent = ingress
    first = await task_start.create_sdk_task(
        agent_id=agent,
        task_owner_user_id=owner,
        actor_user_id=owner,
        message="first",
        file_ids=(),
        connector_runtime_context=(),
        timezone="Asia/Taipei",
    )
    with get_session_local()() as db:
        task = db.get(Task, first.task_id)
        command = db.query(TaskExecutionCommand).one()
        assert task.runner_id is None
        assert command.target_run_id == first.run_id
        assert command.payload["timezone"] == "Asia/Taipei"
        assert command.status == "pending"
        task.status = TaskStatus.COMPLETED
        task.control_state = "completed"
        db.commit()
    second = await task_start.append_sdk_turn(
        task_id=first.task_id,
        scope=task_start.SdkTaskScope(agent_id=agent, workforce_id=None),
        actor_user_id=owner,
        request_agent_id=agent,
        request_workforce_id=None,
        message="second",
        file_ids=(),
        connector_runtime_context=(),
    )
    assert first.run_id != second.run_id
    with get_session_local()() as db:
        commands = (
            db.query(TaskExecutionCommand).order_by(TaskExecutionCommand.id).all()
        )
        assert [c.payload["kind"] for c in commands] == ["create", "append"]
        assert commands[1].target_run_id == second.run_id
        assert db.get(Task, first.task_id).runner_id is None
        assert (
            db.query(TaskChatMessage)
            .filter_by(task_id=first.task_id, role="user")
            .count()
            == 2
        )


@pytest.mark.asyncio
async def test_a2a_create_commits_start_without_local_execution(ingress):
    owner, agent = ingress
    result = await task_start.start_a2a_turn(
        agent_id=agent,
        task_owner_user_id=owner,
        agent_execution_mode="balanced",
        text="hello",
        message_id="message-1",
        context_id=None,
        task_id=None,
    )
    with get_session_local()() as db:
        task = db.get(Task, result.id)
        command = db.query(TaskExecutionCommand).one()
        assert task.source == "a2a"
        assert task.runner_id is None
        assert command.target_run_id == result.run_id
        assert command.payload["kind"] == "create"
        assert (
            db.query(TaskChatMessage).filter_by(task_id=task.id, role="user").count()
            == 1
        )


@pytest.mark.asyncio
async def test_legacy_existing_execution_waits_for_exact_durable_run(ingress):
    owner, _ = ingress
    with get_session_local()() as db:
        task = Task(user_id=owner, title="legacy", status=TaskStatus.PENDING)
        db.add(task)
        db.commit()
        task_id = task.id
    pending = asyncio.create_task(
        task_start.execute_existing_task(
            task_id=task_id,
            task_owner_user_id=owner,
            task_source="internal",
            task_description="saved",
            context={},
            actor_user_id=owner,
        )
    )
    try:

        async def accepted():
            while True:
                if pending.done():
                    pending.result()
                with get_session_local()() as db:
                    command = (
                        db.query(TaskExecutionCommand)
                        .filter_by(task_id=task_id)
                        .first()
                    )
                    if command is not None:
                        return command.target_run_id
                await asyncio.sleep(0.01)

        run_id = await asyncio.wait_for(accepted(), 5)
        assert not pending.done()
        with get_session_local()() as db:
            task = db.get(Task, task_id)
            assert task.run_id == run_id
            assert task.runner_id is None
            assert db.query(TaskChatMessage).count() == 0
            task.status = TaskStatus.COMPLETED
            task.control_state = "completed"
            db.commit()
        await asyncio.wait_for(pending, 5)
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
    ],
)
def test_completion_wait_requires_release_and_rejects_replacement(ingress, status):
    owner, _ = ingress
    with get_session_local()() as db:
        task = Task(
            user_id=owner,
            title="wait",
            run_id="run-1",
            status=status,
            runner_id="worker",
        )
        db.add(task)
        db.commit()
        task_id = task.id
    assert not _is_run_finished(task_id, "run-1")
    with get_session_local()() as db:
        db.get(Task, task_id).runner_id = None
        db.commit()
    assert _is_run_finished(task_id, "run-1")
    with pytest.raises(TaskRunChanged):
        _is_run_finished(task_id, "run-2")
