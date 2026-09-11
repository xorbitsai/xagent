from __future__ import annotations

import pytest
from fastapi import FastAPI

from xagent.web import app as app_module


@pytest.fixture
def metric_reader(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from xagent.core import runtime_performance as metrics

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    monkeypatch.setattr(
        metrics,
        "runtime_performance",
        metrics.RuntimePerformanceTelemetry(provider.get_meter("test")),
    )
    yield reader
    provider.shutdown()


def collected_metrics(reader):
    data = reader.get_metrics_data()
    assert data is not None
    return {
        m.name: m
        for r in data.resource_metrics
        for s in r.scope_metrics
        for m in s.metrics
    }


@pytest.mark.asyncio
async def test_login_rejection_emits_metrics(monkeypatch, metric_reader):
    from fastapi import HTTPException

    from xagent.web.api import auth

    monkeypatch.setattr(auth, "get_user_by_login_identifier", lambda *_: None)
    with pytest.raises(HTTPException) as error:
        await auth.login(
            auth.LoginRequest(username="unknown", password="test"), db=None
        )
    assert error.value.status_code == 401
    metrics = collected_metrics(metric_reader)
    point = metrics["xagent.auth.login.requests"].data.data_points[0]
    assert point.value == 1
    assert point.attributes["outcome"] == "rejected"
    assert metrics["xagent.auth.login.total.duration"].data.data_points[0].count == 1


@pytest.mark.asyncio
async def test_broadcast_records_one_payload_for_multiple_connections(
    metric_reader, caplog
):
    from unittest.mock import AsyncMock

    from xagent.web.api.websocket import ConnectionManager

    manager = ConnectionManager()
    sockets = [AsyncMock(), AsyncMock()]
    for socket in sockets:
        manager.register_connection(socket, 42)
    await manager.broadcast_to_task({"type": "diagnostic"}, 42)
    metrics = collected_metrics(metric_reader)
    assert metrics["xagent.websocket.payload.size"].data.data_points[0].count == 1
    assert metrics["xagent.websocket.messages.sent"].data.data_points[0].value == 2
    assert metrics["xagent.websocket.broadcast.fanout"].data.data_points[0].sum == 2
    with pytest.raises(TypeError):
        await manager.broadcast_to_task({"type": "diagnostic", "bad": object()}, 42)
    assert manager.connections_for_task(42) == sockets
    assert "Failed to serialize WebSocket broadcast" in caplog.text


@pytest.mark.asyncio
async def test_database_trace_handler_emits_write_outcome(monkeypatch, metric_reader):
    from xagent.core.agent.trace import TASK_START_GENERAL, TraceEvent
    from xagent.web.services.trace_handlers import DatabaseTraceHandler

    handler = DatabaseTraceHandler(42)
    saved = []
    monkeypatch.setattr(handler, "_sync_save_to_database", saved.append)
    event = TraceEvent(event_type=TASK_START_GENERAL, task_id="42", data={})
    await handler.handle_event(event)
    assert saved == [event]
    point = collected_metrics(metric_reader)[
        "xagent.trace.database.writes"
    ].data.data_points[0]
    assert point.value == 1
    assert point.attributes["outcome"] == "succeeded"


@pytest.mark.asyncio
async def test_websocket_trace_handler_emits_metrics(monkeypatch, metric_reader):
    from unittest.mock import AsyncMock

    from xagent.core.agent.trace import TASK_START_GENERAL, TraceEvent
    from xagent.web.services import task_event_trace_handler as ws_trace_handlers

    handler = ws_trace_handlers.TaskEventTraceHandler(42)
    handler._task_description_loaded = True
    monkeypatch.setattr(handler, "_has_prior_user_message_turn", lambda *_: False)
    broadcast = AsyncMock()
    monkeypatch.setattr(ws_trace_handlers, "publish_task_event", broadcast)
    await handler.handle_event(
        TraceEvent(event_type=TASK_START_GENERAL, task_id="42", data={})
    )
    broadcast.assert_awaited_once()
    metrics = collected_metrics(metric_reader)
    assert metrics["xagent.websocket.trace.events"].data.data_points[0].value == 1
    assert (
        metrics["xagent.websocket.trace_serialization.duration"]
        .data.data_points[0]
        .count
        == 1
    )


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
