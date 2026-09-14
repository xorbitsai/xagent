"""Start transactions belong to the runner, including cancellation handoff."""

import asyncio
import subprocess
import sys
import textwrap
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.web.pool_contention_shared import GUARD_TIMEOUT
from xagent.web.models.task import TaskStatus
from xagent.web.services import task_start


def test_task_starts_execute_without_api_routes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import importlib.abc
            import sys

            class RejectRoutes(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "xagent.web.api" or fullname.startswith("xagent.web.api."):
                        raise AssertionError(f"Runner imported API: {fullname}")

            sys.meta_path.insert(0, RejectRoutes())
            import asyncio
            from tempfile import TemporaryDirectory
            from unittest.mock import patch
            from xagent.web.models.database import Base, configure_db, get_engine, get_session_local
            from xagent.web.models.user import User
            from xagent.web.models.agent import Agent
            from xagent.web.models.task import Task, TaskStatus
            from xagent.web.models.chat_message import TaskChatMessage
            from xagent.web.services import task_start, task_orchestrator

            async def main():
                with TemporaryDirectory() as directory:
                    configure_db(f"sqlite:///{directory}/starts.db")
                    Base.metadata.create_all(get_engine())
                    with get_session_local()() as db:
                        user = User(username="start-owner", password_hash="unused")
                        db.add(user)
                        db.flush()
                        owner = user.id
                        agent = Agent(user_id=owner, name="start-agent")
                        db.add(agent)
                        db.flush()
                        agent_id = agent.id
                        legacy = Task(user_id=owner, title="legacy", description="saved task", source="internal", status=TaskStatus.PENDING)
                        db.add(legacy)
                        db.flush()
                        legacy_id = legacy.id
                        db.commit()

                    background = []
                    def schedule(**kwargs):
                        handle = asyncio.create_task(asyncio.sleep(0))
                        background.append(handle)
                        return handle

                    with patch.object(task_orchestrator, "_schedule_bg", side_effect=schedule):
                        created = await task_start.create_sdk_task(
                            agent_id=agent_id, task_owner_user_id=owner, actor_user_id=owner,
                            message="first", file_ids=(), connector_runtime_context=(), timezone="Asia/Taipei",
                        )
                        assert created.status == TaskStatus.RUNNING
                        assert not hasattr(created, "background_task")
                        assert created.accepted_at is not None
                        with get_session_local()() as db:
                            row = db.get(Task, created.task_id)
                            assert row.run_id == created.run_id
                            assert row.state_version == created.state_version
                            row.status = TaskStatus.COMPLETED
                            db.commit()
                        appended = await task_start.append_sdk_turn(
                            task_id=created.task_id,
                            scope=task_start.SdkTaskScope(agent_id=agent_id, workforce_id=None),
                            actor_user_id=owner, request_agent_id=agent_id, request_workforce_id=None,
                            message="second", file_ids=(), connector_runtime_context=(),
                        )
                        assert appended.run_id
                        assert appended.run_id != created.run_id
                        with get_session_local()() as db:
                            row = db.get(Task, appended.task_id)
                            assert row.run_id == appended.run_id
                            assert row.state_version == appended.state_version
                        assert not hasattr(appended, "background_task")
                        a2a = await task_start.start_a2a_turn(
                            agent_id=agent_id, task_owner_user_id=owner, agent_execution_mode="balanced",
                            text="a2a first", message_id="message-1", context_id=None, task_id=None,
                        )
                        legacy_handle_index = len(background)
                        await task_start.execute_existing_task(
                            task_id=legacy_id, task_owner_user_id=owner, task_source="internal",
                            task_description="saved task", context={}, actor_user_id=owner,
                        )
                        assert background[legacy_handle_index].done()
                        await asyncio.gather(*background)
                    with get_session_local()() as db:
                        sdk_messages = db.query(TaskChatMessage).filter_by(task_id=created.task_id, role="user").order_by(TaskChatMessage.id).all()
                        assert [message.content for message in sdk_messages] == ["first", "second"]
                        assert db.query(TaskChatMessage).filter_by(task_id=a2a.id, role="user").count() == 1
                        assert db.query(TaskChatMessage).filter_by(task_id=legacy_id, role="user").count() == 0
                    assert not any(name == "xagent.web.api" or name.startswith("xagent.web.api.") for name in sys.modules)

            asyncio.run(main())
        """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["create", "append"])
async def test_sdk_start_cancellation_after_claim_still_schedules(monkeypatch, kind):
    claimed = threading.Event()
    release = threading.Event()
    prepared = SimpleNamespace(
        task_id=91,
        agent_id=7,
        task_owner_user_id=3,
        created_at=datetime.now(timezone.utc),
        payload=SimpleNamespace(turn_id="turn-1"),
        claimed_turn=Mock(),
    )

    def prepare(**kwargs):
        claimed.set()
        assert release.wait(timeout=5)
        return prepared

    started = SimpleNamespace(
        status=TaskStatus.RUNNING,
        updated_at=datetime.now(timezone.utc),
        run_id="run-1",
        state_version=2,
        control_state="running",
    )
    schedule = AsyncMock(return_value=started)
    if kind == "create":
        monkeypatch.setattr(task_start, "_prepare_created_task_isolated", prepare)
        monkeypatch.setattr(
            task_start.TaskTurnOrchestrator, "schedule_claimed_create_turn", schedule
        )
        operation = asyncio.create_task(
            task_start.create_sdk_task(
                agent_id=7,
                task_owner_user_id=3,
                actor_user_id=3,
                message="first",
                file_ids=(),
                connector_runtime_context=(),
                timezone=None,
            )
        )
    else:
        monkeypatch.setattr(task_start, "_prepare_append_turn_isolated", prepare)
        monkeypatch.setattr(
            task_start.TaskTurnOrchestrator, "ensure_no_background_turn", Mock()
        )
        monkeypatch.setattr(
            task_start.TaskTurnOrchestrator, "schedule_claimed_turn", schedule
        )
        operation = asyncio.create_task(
            task_start.append_sdk_turn(
                task_id=91,
                scope=task_start.SdkTaskScope(agent_id=7, workforce_id=None),
                actor_user_id=3,
                request_agent_id=7,
                request_workforce_id=None,
                message="next",
                file_ids=(),
                connector_runtime_context=(),
            )
        )
    try:
        assert await asyncio.to_thread(claimed.wait, 5)
        operation.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, timeout=5)
        schedule.assert_awaited_once()
        assert schedule.await_args.kwargs["claimed"] is prepared.claimed_turn
    finally:
        release.set()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_legacy_execute_waits_for_background_completion(monkeypatch):
    scheduled = asyncio.Event()
    background = asyncio.get_running_loop().create_future()

    async def schedule(**kwargs):
        scheduled.set()
        return background

    monkeypatch.setattr(
        task_start.TaskTurnOrchestrator, "schedule_existing_task_execution", schedule
    )
    execution = asyncio.create_task(
        task_start.execute_existing_task(
            task_id=91,
            task_owner_user_id=3,
            task_source="internal",
            task_description="saved task",
            context={},
            actor_user_id=3,
        )
    )
    try:
        await asyncio.wait_for(scheduled.wait(), timeout=GUARD_TIMEOUT)
        assert not execution.done()
        background.set_result(None)
        await asyncio.wait_for(execution, timeout=GUARD_TIMEOUT)
    finally:
        if not background.done():
            background.set_result(None)
        await asyncio.gather(execution, return_exceptions=True)
