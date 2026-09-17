"""A startup deferral must not be converted to a private error reply."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.web.models.task import TaskStatus
from xagent.web.services import agent_service_manager
from xagent.web.services import task_command_execution as commands
from xagent.web.services import task_setup_snapshot
from xagent.web.services.task_command_transport import TaskCommandDeferred


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_result", [False, True])
async def test_startup_pause_defers_without_error_and_success_uses_events(
    monkeypatch, pause_result
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    task = SimpleNamespace(user_id=1, run_id="run", status=TaskStatus.RUNNING)
    monkeypatch.setattr(
        task_setup_snapshot,
        "load_task_setup_snapshot_sync",
        lambda *a, **k: SimpleNamespace(task=task, runtime_user=object()),
    )
    monkeypatch.setattr(commands, "resolve_execution_scope_off_turn", lambda *a: None)
    service = SimpleNamespace(pause_execution=AsyncMock(return_value=pause_result))
    monkeypatch.setattr(
        agent_service_manager,
        "get_agent_manager",
        lambda: SimpleNamespace(get_agent_for_task=AsyncMock(return_value=service)),
    )
    pending = asyncio.create_task(asyncio.Event().wait())
    monkeypatch.setattr(
        commands.task_execution_service,
        "background_task_manager",
        SimpleNamespace(running_tasks={1: pending}),
    )
    monkeypatch.setattr(
        commands, "_apply_pause_requested_isolated", lambda *a, **k: True
    )
    monkeypatch.setattr(commands, "_mark_task_pause_accepted", lambda *a: None)
    publish = AsyncMock()
    monkeypatch.setattr(commands, "publish_task_event", publish)
    reply = AsyncMock()
    message = {"user": SimpleNamespace(id=1, is_admin=False)}
    try:
        if pause_result:
            await commands.pause_task(reply, 1, message)
            publish.assert_awaited_once()
        else:
            with pytest.raises(TaskCommandDeferred, match="still starting"):
                await commands.pause_task(reply, 1, message)
            publish.assert_not_awaited()
        reply.assert_not_awaited()
        assert "_durable_command_error" not in message
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
