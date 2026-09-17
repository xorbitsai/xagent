"""Role boundaries and shared web admission order."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from xagent.web import app as app_module
from xagent.web.api.websocket import manager
from xagent.web.services import task_command_transport as transport
from xagent.web.services import (
    task_coordinator_runtime,
    task_event_bridge,
    task_execution,
)


@pytest.mark.asyncio
async def test_web_role_cannot_claim_or_promptly_execute(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "web")
    db, execute = Mock(), AsyncMock()
    assert transport.claim_task_command(db, command_db_id=1) is None
    await transport.dispatch_task_command_promptly(execute, command_db_id=1)
    assert transport.start_task_command_dispatcher(execute) is None
    assert app_module.start_task_lease_recovery_task(FastAPI()) is None
    db.assert_not_called()
    assert db.mock_calls == []
    execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [None, "bridge", "dispatcher"])
async def test_shared_web_starts_bridge_before_consumers_and_unwinds_failure(
    monkeypatch, fails
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    app = FastAPI()
    events = []
    monkeypatch.setattr(app_module, "_trigger_dispatcher_task", None)
    monkeypatch.setattr(app_module, "_task_command_dispatcher_task", None)

    async def start_bridge(**kwargs):
        events.append("bridge")
        assert kwargs["deliver"] == manager.deliver_shared_event
        if fails == "bridge":
            raise ConnectionError("bridge")

    def start_dispatcher(execute):
        events.append("dispatcher")
        if fails == "dispatcher":
            raise RuntimeError("dispatcher")

    monkeypatch.setattr(task_event_bridge, "start_task_event_bridge", start_bridge)
    monkeypatch.setattr(
        task_event_bridge,
        "stop_task_event_bridge",
        AsyncMock(side_effect=lambda: events.append("bridge_stop")),
    )
    monkeypatch.setattr(
        manager, "start_stream_reconciliation", lambda: events.append("reconcile")
    )
    monkeypatch.setattr(
        manager,
        "stop_stream_reconciliation",
        AsyncMock(side_effect=lambda: events.append("reconcile_stop")),
    )
    monkeypatch.setattr(
        task_execution,
        "background_task_manager",
        SimpleNamespace(
            start_accepting=lambda: events.append("admit"),
            shutdown=AsyncMock(side_effect=lambda: events.append("execution_stop")),
        ),
    )
    monkeypatch.setattr(
        app_module,
        "start_trigger_dispatcher_task",
        lambda app: events.append("triggers"),
    )
    monkeypatch.setattr(
        app_module,
        "start_task_lease_recovery_task",
        lambda app: events.append("recovery"),
    )
    monkeypatch.setattr(
        app_module,
        "stop_task_lease_recovery_task",
        AsyncMock(side_effect=lambda app: events.append("recovery_stop")),
    )
    monkeypatch.setattr(transport, "start_task_command_dispatcher", start_dispatcher)
    monkeypatch.setattr(
        transport,
        "stop_task_command_dispatcher",
        AsyncMock(side_effect=lambda: events.append("claims_stop")),
    )
    monkeypatch.setattr(
        task_coordinator_runtime,
        "close_task_coordinators",
        AsyncMock(side_effect=lambda: events.append("owner_stop")),
    )
    if fails:
        with pytest.raises((ConnectionError, RuntimeError), match=fails):
            await app_module._start_shared_task_runtime(app)
    else:
        await app_module._start_shared_task_runtime(app)
    if fails == "bridge":
        assert events == ["bridge"]
    else:
        assert events[:6] == [
            "bridge",
            "reconcile",
            "admit",
            "triggers",
            "recovery",
            "dispatcher",
        ]
        if fails:
            assert events[6:] == [
                "claims_stop",
                "recovery_stop",
                "owner_stop",
                "execution_stop",
                "reconcile_stop",
                "bridge_stop",
            ]
