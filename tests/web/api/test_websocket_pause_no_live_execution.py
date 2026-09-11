"""What the pause handler reports when no live execution can be interrupted.

``AgentService.pause_execution`` is stateless, so it answers "is a run live
right now" honestly and returns ``False`` for an already-paused task as well as
for one that is simply not running. Those two cases are very different to a
user, so the handler distinguishes them from the persisted task status.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import _make_command_reply
from xagent.web.models.task import TaskStatus
from xagent.web.services import agent_service_manager as agent_runtime_service
from xagent.web.services import task_command_execution as command_execution_service
from xagent.web.services import task_execution as task_execution_service
from xagent.web.services import task_setup_snapshot as snapshot_module


async def _run_pause_with_no_live_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: TaskStatus,
) -> list[str]:
    """Drive the pause handler for a task whose run cannot be interrupted."""

    task_id = 41
    owner_id = 7
    actor = SimpleNamespace(id=owner_id, is_admin=False)
    runtime_user = SimpleNamespace(id=owner_id, is_admin=False)
    snapshot = SimpleNamespace(
        task=SimpleNamespace(user_id=owner_id, status=status, run_id="run-1"),
        runtime_user=runtime_user,
    )
    reported: list[str] = []

    async def read_error_payload(
        resolved_task_id: int, message: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        assert resolved_task_id == task_id
        reported.append(message)
        return {"type": "error", "message": message}

    agent_service = MagicMock()
    agent_service.pause_execution = AsyncMock(return_value=False)
    agent_manager = MagicMock()
    agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()

    monkeypatch.setattr(
        snapshot_module,
        "load_task_setup_snapshot_sync",
        lambda *args, **kwargs: snapshot,
    )
    monkeypatch.setattr(
        command_execution_service,
        "resolve_execution_scope_off_turn",
        lambda _task_id: None,
    )
    monkeypatch.setattr(
        command_execution_service,
        "_read_task_error_payload_offloop",
        read_error_payload,
    )
    monkeypatch.setattr(
        agent_runtime_service, "get_agent_manager", lambda: agent_manager
    )
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    try:
        await command_execution_service.pause_task(
            _make_command_reply(MagicMock()), task_id, {"user": actor}
        )
    finally:
        task_execution_service._clear_task_pause_accepted(task_id)

    connection_manager.broadcast_to_task.assert_not_awaited()
    return reported


@pytest.mark.asyncio
async def test_pause_on_an_already_paused_task_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reported = await _run_pause_with_no_live_execution(
        monkeypatch, status=TaskStatus.PAUSED
    )

    assert reported == ["Task is already paused"]


@pytest.mark.asyncio
async def test_pause_on_a_task_with_no_live_run_reports_that(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reported = await _run_pause_with_no_live_execution(
        monkeypatch, status=TaskStatus.RUNNING
    )

    assert reported == ["No live execution found to pause"]
