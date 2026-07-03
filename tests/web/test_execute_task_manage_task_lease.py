"""Tests for ``AgentServiceManager.execute_task`` lease delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.workforce_runtime import sync_workforce_run_status


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'execute_task_lease.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


class _FakeAgentService:
    async def execute_task(self, **_kwargs):
        return {"success": True}


@pytest.mark.asyncio
async def test_execute_task_acquires_and_releases_lease_when_manage_true(
    db_session,
) -> None:
    user = User(username="lease-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    fake_lease = TaskLease(task_id=int(task.id), runner_id="test-runner")
    manager = AgentServiceManager()

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease",
            return_value=fake_lease,
        ) as mock_acquire,
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status",
            return_value=False,
        ) as mock_sync,
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=True,
        )

    assert result["success"] is True
    mock_acquire.assert_called_once()
    mock_release.assert_called_once()
    mock_sync.assert_called_once()


@pytest.mark.asyncio
async def test_execute_task_skips_lease_but_syncs_running_when_manage_false(
    db_session,
) -> None:
    user = User(username="lease-user2", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="lease test",
        description="test",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.commit()

    manager = AgentServiceManager()

    with (
        patch(
            "xagent.web.api.chat.acquire_task_lease",
        ) as mock_acquire,
        patch(
            "xagent.web.api.chat.release_task_lease_with_workforce_sync",
        ) as mock_release,
        patch(
            "xagent.web.api.chat.run_task_lease_heartbeat",
            new=AsyncMock(),
        ),
        patch(
            "xagent.web.api.chat.stop_task_lease_heartbeat",
            new=AsyncMock(),
        ) as mock_stop_hb,
        patch.object(
            manager, "_acquire_sandbox_task", new=AsyncMock(return_value=None)
        ),
        patch.object(manager, "_release_sandbox_task", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.sync_workforce_run_status",
            return_value=False,
        ) as mock_sync,
        patch(
            "xagent.web.tracking.task_tracker.TaskTracker",
            side_effect=RuntimeError("skip tracking in unit test"),
        ),
    ):
        result = await manager.execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    assert result["success"] is True
    mock_acquire.assert_not_called()
    mock_release.assert_not_called()
    mock_sync.assert_called_once()
    mock_stop_hb.assert_awaited_once_with(None, None)


def test_sync_workforce_run_status_running_is_idempotent(db_session) -> None:
    """Repeat RUNNING sync is a no-op when WorkforceRun is already running."""
    user = User(username="sync-user", password_hash="hash", is_admin=False)
    db_session.add(user)
    db_session.flush()
    manager = Agent(
        user_id=user.id,
        name="Manager",
        description="desc",
        instructions="instr",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db_session.add(manager)
    db_session.flush()
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Team",
        description="desc",
        manager_agent_id=manager.id,
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    task = Task(
        user_id=user.id,
        title="sync test",
        description="test",
        status=TaskStatus.RUNNING,
        agent_id=manager.id,
        agent_config={},
        execution_mode="auto",
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": run.id}
    db_session.commit()

    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is False
    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is False
    db_session.refresh(run)
    assert run.status == "running"
    assert run.completed_at is None
