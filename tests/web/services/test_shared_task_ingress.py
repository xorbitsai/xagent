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
        command.status = "completed"
        task.run_id = first.run_id
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
        assert result.run_id is None
        assert command.target_run_id is not None
        assert task.status == TaskStatus.PENDING
        assert command.payload["kind"] == "create"
        assert (
            db.query(TaskChatMessage).filter_by(task_id=task.id, role="user").count()
            == 1
        )


@pytest.mark.asyncio
async def test_legacy_existing_execution_returns_after_durable_acceptance(ingress):
    owner, _ = ingress
    with get_session_local()() as db:
        task = Task(user_id=owner, title="legacy", status=TaskStatus.PENDING)
        db.add(task)
        db.commit()
        task_id = task.id
    await asyncio.wait_for(
        task_start.execute_existing_task(
            task_id=task_id,
            task_owner_user_id=owner,
            task_source="internal",
            task_description="saved",
            context={},
            actor_user_id=owner,
        ),
        5,
    )
    with get_session_local()() as db:
        command = db.query(TaskExecutionCommand).filter_by(task_id=task_id).one()
        assert command.target_run_id is not None
        assert command.status == "pending"
        task = db.get(Task, task_id)
        assert task.run_id is None
        assert task.runner_id is None
        assert db.query(TaskChatMessage).count() == 0


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


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
    ],
)
def test_recovery_releases_expired_owner_without_mutating_business_status(
    ingress, status
):
    from datetime import datetime, timedelta, timezone

    owner, _ = ingress
    with get_session_local()() as db:
        task = Task(
            user_id=owner,
            title="dead owner",
            run_id="run-1",
            status=status,
            runner_id="worker",
            lease_attempt_id="attempt",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
        db.add(task)
        db.commit()
        task_id = task.id
    assert not _is_run_finished(task_id, "run-1")
    with get_session_local()() as db:
        db.get(Task, task_id).lease_expires_at = datetime.now(timezone.utc) - timedelta(
            minutes=1
        )
        db.commit()
    with pytest.raises(TaskRunChanged):
        _is_run_finished(task_id, "previous-run")
    with get_session_local()() as db:
        assert db.get(Task, task_id).runner_id == "worker"
    assert not _is_run_finished(task_id, "run-1")
    from xagent.web.services.task_lease_recovery import recover_expired_idle_task_leases

    assert recover_expired_idle_task_leases(batch_size=10) == 1
    assert _is_run_finished(task_id, "run-1")
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        assert task.runner_id is None
        assert task.lease_attempt_id is None
        assert task.status == status


@pytest.mark.asyncio
async def test_sdk_append_records_current_actor_after_owner_transfer(
    ingress, monkeypatch
):
    from xagent.web.services import task_start_consumer
    from xagent.web.services.task_command_transport import claim_task_command

    owner, agent_id = ingress
    first = await task_start.create_sdk_task(
        agent_id=agent_id,
        task_owner_user_id=owner,
        actor_user_id=owner,
        message="first",
        timezone=None,
        file_ids=(),
        connector_runtime_context=(),
    )
    with get_session_local()() as db:
        actor = User(username="new-owner", password_hash="unused")
        db.add(actor)
        db.flush()
        actor_id, actor_subject = actor.id, actor.actor_subject
        db.get(Agent, agent_id).user_id = actor_id
        task = db.get(Task, first.task_id)
        task.status = TaskStatus.COMPLETED
        task.control_state = "completed"
        db.query(TaskExecutionCommand).update({"status": "completed"})
        db.commit()
    second = await task_start.append_sdk_turn(
        task_id=first.task_id,
        scope=task_start.SdkTaskScope(agent_id=agent_id, workforce_id=None),
        actor_user_id=actor_id,
        request_agent_id=agent_id,
        request_workforce_id=None,
        message="second",
        file_ids=(),
        connector_runtime_context=(),
    )
    with get_session_local()() as db:
        row = (
            db.query(TaskExecutionCommand).filter_by(target_run_id=second.run_id).one()
        )
        assert row.actor_user_id == actor_id
        assert row.actor_subject == actor_subject
        assert row.task_owner_user_id == owner
        command = claim_task_command(db, runner_id="worker", command_db_id=row.id)
    from xagent.web.services.task_coordinator_service import (
        acquire_task_lease_no_commit,
    )

    with get_session_local()() as db, db.begin():
        owner_lease = acquire_task_lease_no_commit(
            db, first.task_id, runner_id="worker"
        )
    handoff = task_start_consumer._commit_handoff(command, owner_lease)
    assert handoff.task_owner_user_id == owner
    assert handoff.claimed.task_lease.run_id == second.run_id


@pytest.mark.asyncio
async def test_trigger_batch_continues_after_run_replacement(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from xagent.web.services import quota_hooks, task_completion, triggers

    monkeypatch.setattr(triggers, "_get_pending_trigger_run_ids", lambda limit: [1, 2])
    monkeypatch.setattr(
        triggers,
        "_load_prepared_trigger_start",
        lambda run_id: SimpleNamespace(
            run_id=run_id,
            task_id=run_id,
            trigger_id=run_id,
            trigger_type="scheduled",
            test=False,
            task_owner_user_id=1,
            prompt="run",
        ),
    )
    begin = AsyncMock(
        side_effect=[
            SimpleNamespace(task_id=1, run_id="old", background_task=None),
            SimpleNamespace(task_id=2, run_id="current", background_task=None),
        ]
    )
    monkeypatch.setattr(triggers.TaskTurnOrchestrator, "begin_turn", begin)
    monkeypatch.setattr(triggers, "_mark_trigger_run_started", Mock())
    monkeypatch.setattr(quota_hooks, "record_trigger", Mock())
    finish = Mock()
    fail = Mock()
    monkeypatch.setattr(triggers, "_finish_trigger_run_after_task", finish)
    monkeypatch.setattr(triggers, "_mark_trigger_run_failed_by_id", fail)
    monkeypatch.setattr(
        task_completion,
        "wait_for_task_run",
        AsyncMock(side_effect=[TaskRunChanged(), None]),
    )
    assert (
        await triggers.dispatch_pending_trigger_runs(Mock(), wait_for_completion=True)
        == 1
    )
    assert begin.await_count == 2
    fail.assert_not_called()
    assert finish.call_count == 1
    assert finish.call_args.args[0].run_id == 2
    assert finish.call_args.args[1] == "current"
