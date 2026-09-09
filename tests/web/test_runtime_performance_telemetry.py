from __future__ import annotations

import pytest
from fastapi import FastAPI

from xagent.web import app as app_module


def test_runtime_performance_monitor_stays_off_without_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_app = FastAPI()
    monkeypatch.setattr(
        app_module,
        "initialize_runtime_performance_telemetry",
        lambda: False,
    )

    app_module.start_runtime_performance_monitor(test_app)

    assert test_app.state.runtime_performance_task is None


@pytest.mark.asyncio
async def test_runtime_performance_monitor_registers_gauges_once_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_app = FastAPI()
    registered: list[str] = []
    shutdown_calls: list[bool] = []
    monkeypatch.setattr(
        app_module,
        "initialize_runtime_performance_telemetry",
        lambda: True,
    )
    monkeypatch.setattr(
        app_module,
        "register_observable_gauge",
        lambda metric, *_args, **_kwargs: registered.append(metric),
    )
    monkeypatch.setattr(
        app_module,
        "shutdown_runtime_performance_telemetry",
        lambda: shutdown_calls.append(True),
    )

    app_module.start_runtime_performance_monitor(test_app)
    first_task = test_app.state.runtime_performance_task
    app_module.start_runtime_performance_monitor(test_app)

    assert test_app.state.runtime_performance_task is first_task
    assert registered == [
        "xagent.agent_tasks.running",
        "xagent.agent_tasks.resuming",
        "xagent.websocket.connections",
    ]

    await app_module.stop_runtime_performance_monitor(test_app)
    assert test_app.state.runtime_performance_task is None
    assert first_task.cancelled()
    assert shutdown_calls == [True]
