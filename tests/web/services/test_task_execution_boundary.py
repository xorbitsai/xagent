"""The task runtime can load and publish events without a Web API host."""

import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock, Mock, call

import pytest

from xagent.web.services import task_events


def test_execution_services_and_tracer_load_without_api_routes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.abc
                import sys

                class RejectRoutes(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "xagent.web.api" or fullname.startswith("xagent.web.api."):
                            raise AssertionError(f"Execution imported API route: {fullname}")

                sys.meta_path.insert(0, RejectRoutes())
                from xagent.web.services import agent_service_manager, task_execution, task_orchestrator, task_resume
                from xagent.web.services.external_task_cancel import _broadcast_external_cancel_terminal_event
                import asyncio
                from unittest.mock import AsyncMock, patch
                from xagent.web.services import task_events
                sink = AsyncMock()
                task_events.set_task_event_sink(sink)
                event = {"type": "task_error", "task_id": 1}
                with patch.object(task_execution, "create_terminal_task_error_event", return_value=event):
                    asyncio.run(_broadcast_external_cancel_terminal_event(1))
                sink.assert_awaited_once_with(event, 1)
                from xagent.web.tracing import create_task_tracer
                create_task_tracer(1, user_id=1)
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_event_delivery_uses_the_host_sink(monkeypatch) -> None:
    monkeypatch.setattr(task_events, "_task_event_sink", None)
    monkeypatch.setattr(task_events, "_warned_missing_sink", False)
    counter = Mock()
    monkeypatch.setattr(task_events, "increment_counter", counter)
    event = {"type": "task_completed", "task_id": 42}

    sink = AsyncMock()
    task_events.set_task_event_sink(sink)
    await task_events.publish_task_event(event, 42)
    sink.assert_awaited_once_with(event, 42)

    sink.side_effect = RuntimeError("delivery failed")
    with pytest.raises(RuntimeError, match="delivery failed"):
        await task_events.publish_task_event(event, 42)
    counter.assert_not_called()


@pytest.mark.asyncio
async def test_missing_sink_counts_every_event_and_warns_once_until_registered(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(task_events, "_task_event_sink", None)
    monkeypatch.setattr(task_events, "_warned_missing_sink", False)
    counter = Mock()
    monkeypatch.setattr(task_events, "increment_counter", counter)
    event = {"type": "task_error", "message": "private event content"}

    await task_events.publish_task_event(event, 42)
    task_events.set_task_event_sink(None)
    await task_events.publish_task_event(event, 42)
    warnings = [r for r in caplog.records if r.name == task_events.__name__]
    assert len(warnings) == 1
    assert warnings[0].levelname == "WARNING"
    assert "no host event sink" in warnings[0].message
    assert event["message"] not in caplog.text

    sink = AsyncMock()
    task_events.set_task_event_sink(sink)
    await task_events.publish_task_event(event, 42)
    sink.assert_awaited_once_with(event, 42)
    assert counter.call_count == 2

    task_events.set_task_event_sink(None)
    await task_events.publish_task_event(event, 42)
    warnings = [r for r in caplog.records if r.name == task_events.__name__]
    assert len(warnings) == 2
    assert (
        counter.call_args_list
        == [call("xagent.task_events.dropped", attributes={"outcome": "no_sink"})] * 3
    )


def test_web_host_registers_event_delivery_on_import() -> None:
    # A fresh process exercises registration, not an adapter installed by a
    # previous test or an importlib.reload that leaves old module state behind.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import asyncio
            from unittest.mock import AsyncMock, patch
            from xagent.web.services import task_events
            assert task_events._task_event_sink is None
            from xagent.web.api import websocket
            event = {"type": "task_completed", "task_id": 42}
            sink = AsyncMock()
            with patch.object(websocket.manager, "broadcast_to_task", sink):
                asyncio.run(task_events.publish_task_event(event, 42))
            sink.assert_awaited_once_with(event, 42)
        """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_control_and_reply_commands_execute_without_api_routes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
            import importlib.abc
            import sys


            class RejectRoutes(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "xagent.web.api" or fullname.startswith("xagent.web.api."):
                        raise AssertionError(f"Execution imported API route: {fullname}")


            sys.meta_path.insert(0, RejectRoutes())
            import asyncio
            from datetime import datetime, timedelta, timezone
            from tempfile import TemporaryDirectory
            from unittest.mock import AsyncMock, patch
            from xagent.web.models.database import Base, configure_db, get_engine, get_session_local
            from xagent.web.models.user import User
            from xagent.web.models.agent import Agent
            from xagent.web.models.task import Task, TaskStatus
            from xagent.web.models.chat_message import TaskChatMessage
            from xagent.web.services.task_command_execution import execute_durable_task_command
            from xagent.web.services.task_command_transport import (
                ClaimedTaskCommand,
                TaskCommandKind,
            )
            from xagent.web.services import agent_service_manager
            from xagent.core.agent.runner import UserMessageInjectionOutcome
            from xagent.web.services import task_execution, task_resume
            from xagent.web.services.task_lease_service import (
                stop_task_lease_heartbeat, release_task_lease_no_commit,
            )


            async def main():
                with TemporaryDirectory() as directory:
                    configure_db(f"sqlite:///{directory}/commands.db")
                    Base.metadata.create_all(get_engine())
                    with get_session_local()() as db:
                        user = User(username="runner-test", password_hash="unused")
                        db.add(user)
                        db.flush()
                        uid = user.id
                        db.add(Agent(id=1, user_id=uid, name="cancel-agent"))
                        db.flush()
                        tasks = [
                            Task(
                                user_id=uid,
                                title=kind,
                                description=kind,
                                status=TaskStatus.RUNNING,
                                control_state="running",
                                run_id=kind,
                                state_version=0,
                                agent_config={},
                            )
                            for kind in ["message", "pause", "resume", "cancel"]
                        ]
                        tasks[2].runner_id = "other-runner"
                        tasks[2].lease_expires_at = datetime.now(timezone.utc) + timedelta(
                            minutes=5
                        )
                        tasks[3].source = "a2a"
                        tasks[3].agent_id = 1
                        db.add_all(tasks)
                        db.flush()
                        ids = [t.id for t in tasks]
                        db.add(
                            TaskChatMessage(
                                user_id=uid,
                                task_id=ids[0],
                                role="user",
                                content="hello",
                                message_type="user",
                                turn_id="message-command",
                                delivery_status="completed",
                            )
                        )
                        db.commit()
                    agent = AsyncMock()
                    agent.pause_execution.return_value = True
                    with patch.object(agent_service_manager, "get_agent_manager") as manager:
                        manager.return_value.get_agent_for_task = AsyncMock(return_value=agent)
                        for index, kind in enumerate(TaskCommandKind):
                            # Enum includes only the four durable command kinds.
                            task_index = ["message", "pause", "resume", "cancel"].index(kind.value)
                            payload = (
                                {"client_message_id": "message-command", "message": "hello"}
                                if kind.value == "message"
                                else {}
                            )
                            if kind.value == "cancel":
                                payload = {"agent_id": 1, "target_state_version": 0}
                            command = ClaimedTaskCommand(
                                id=index + 1,
                                task_id=ids[task_index],
                                actor_user_id=uid,
                                command_id=f"{kind.value}-command",
                                kind=kind,
                                payload=payload,
                                target_run_id=kind.value,
                                attempt_count=1,
                            )
                            result = await execute_durable_task_command(command)
                            assert result["kind"] == kind.value, result
                            if kind.value == "resume":
                                assert result["resume_outcome"] == "already_in_progress", result
                        agent.pause_execution.assert_awaited_once()
                        manager.return_value.get_agent_for_task.assert_awaited_once()
                    with get_session_local()() as db:
                        assert db.get(Task, ids[1]).control_state == "pause_requested"
                        canceled = db.get(Task, ids[3])
                        assert canceled.status == TaskStatus.FAILED
                        assert canceled.agent_config["a2a_state"] == "TASK_STATE_CANCELED"
                        assert canceled.state_version == 1
                        assert db.query(TaskChatMessage).filter_by(task_id=ids[0]).count() == 1
                    # Both reply paths run their real claims, writes and scheduling
                    # without loading HTTP routes. Only the Agent and execution
                    # leaf are replaced, so no model or external tools are needed.
                    with get_session_local()() as db:
                        replies = [
                            Task(
                                user_id=uid, agent_id=1, source=source,
                                title="reply", status=TaskStatus.WAITING_FOR_USER,
                                control_state="waiting_for_user", run_id=f"reply-{source}",
                            )
                            for source in ["a2a", "sdk"]
                        ]
                        db.add_all(replies)
                        db.flush()
                        reply_ids = [task.id for task in replies]
                        db.commit()

                    async def finish_resume(**kwargs):
                        await stop_task_lease_heartbeat(
                            kwargs["preacquired_heartbeat_task"],
                            kwargs["preacquired_heartbeat_stop"],
                        )
                        with get_session_local()() as db:
                            assert release_task_lease_no_commit(
                                db, kwargs["preacquired_lease"], status=TaskStatus.COMPLETED,
                            )
                            db.commit()

                    agent.post_user_message.return_value = UserMessageInjectionOutcome.POSTED_FRESH
                    with (
                        patch.object(agent_service_manager, "get_agent_manager") as manager,
                        patch.object(task_execution, "execute_resume_background", side_effect=finish_resume),
                    ):
                        manager.return_value.get_agent_for_task = AsyncMock(return_value=agent)
                        assert await task_resume.resume_a2a_task(
                            agent_id=1, task_owner_user_id=uid, task_id=reply_ids[0],
                            previous_run_id="reply-a2a", resumable_status=TaskStatus.WAITING_FOR_USER,
                            text="A2A answer", message_id="answer-1",
                        )
                        a2a_resume = task_execution.background_task_manager.resume_tasks[reply_ids[0]]
                        result = await task_resume.resume_task_reply(task_resume.TaskReplyInput(
                            task_id=reply_ids[1], agent_id=1, task_owner_user_id=uid,
                            run_id="reply-sdk", status=TaskStatus.WAITING_FOR_USER, text="SDK answer",
                        ))
                        assert result.run_id == "reply-sdk"
                        assert result.control_state == "running"
                        sdk_resume = task_execution.background_task_manager.resume_tasks[reply_ids[1]]
                        await asyncio.gather(a2a_resume, sdk_resume)
                    with get_session_local()() as db:
                        for tid, answer in zip(reply_ids, ["A2A answer", "SDK answer"]):
                            row = db.get(Task, tid)
                            assert row.input == answer
                            assert row.status == TaskStatus.COMPLETED
                            assert row.runner_id is None
                    assert not any(
                        name == "xagent.web.api" or name.startswith("xagent.web.api.")
                        for name in sys.modules
                    )


            asyncio.run(main())
            """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
