from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import xagent.core.runtime_performance as runtime_metrics
from xagent.core.agent.trace import TASK_START_GENERAL, TraceEvent, TraceHandler, Tracer
from xagent.core.runtime_performance import (
    METER_NAME,
    RuntimePerformanceTelemetry,
    _histogram_views,
    monitor_event_loop_lag,
    run_in_thread_with_telemetry,
    runtime_performance,
)


@pytest.mark.parametrize(
    "endpoint",
    [
        "localhost:4318",
        "http://localhost:4318?secret=value",
        "http://localhost:4318/#fragment",
    ],
)
def test_invalid_endpoint_disables_telemetry(monkeypatch, caplog, endpoint):
    monkeypatch.setenv("XAGENT_RUNTIME_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("XAGENT_OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", endpoint)
    assert runtime_metrics.initialize_runtime_performance_telemetry() is False
    assert "runtime telemetry is disabled" in caplog.text
    assert endpoint not in caplog.text


def test_meter_failure_does_not_mask_business_result_or_error(monkeypatch):
    meter = MagicMock()
    meter.create_counter.side_effect = RuntimeError("broken meter")
    meter.create_histogram.return_value.record.side_effect = RuntimeError(
        "broken histogram"
    )
    monkeypatch.setattr(
        runtime_metrics, "runtime_performance", RuntimePerformanceTelemetry(meter)
    )
    runtime_metrics.increment_counter("xagent.test")
    with runtime_metrics.observe_duration("xagent.test.duration"):
        result = 42
    assert result == 42
    with pytest.raises(ValueError, match="business error"):
        with runtime_metrics.observe_duration("xagent.test.duration"):
            raise ValueError("business error")


@pytest.fixture
def otel_backend() -> Iterator[
    tuple[RuntimePerformanceTelemetry, InMemoryMetricReader]
]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(
        metric_readers=[reader],
        views=_histogram_views(),
        shutdown_on_exit=False,
    )
    telemetry = RuntimePerformanceTelemetry(provider.get_meter(METER_NAME))
    try:
        yield telemetry, reader
    finally:
        provider.shutdown()


def _metrics(reader: InMemoryMetricReader) -> dict[str, Any]:
    data = reader.get_metrics_data()
    assert data is not None
    return {
        metric.name: metric
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def test_records_counter_and_explicit_bucket_histogram_in_otel(
    otel_backend: tuple[RuntimePerformanceTelemetry, InMemoryMetricReader],
) -> None:
    telemetry, reader = otel_backend

    telemetry.increment(
        "xagent.trace.events",
        attributes={"event.type": "task_start_general"},
    )
    telemetry.observe(
        "xagent.trace.dispatch.duration",
        6.0,
        unit="ms",
    )

    metrics = _metrics(reader)
    counter_point = metrics["xagent.trace.events"].data.data_points[0]
    histogram_point = metrics["xagent.trace.dispatch.duration"].data.data_points[0]
    assert counter_point.value == 1
    assert counter_point.attributes == {"event.type": "task_start_general"}
    assert histogram_point.count == 1
    assert histogram_point.sum == 6.0
    assert 10.0 in histogram_point.explicit_bounds
    assert (
        histogram_point.bucket_counts[histogram_point.explicit_bounds.index(10.0)] == 1
    )


def test_attribute_allowlist_and_cardinality_guard_exclude_sensitive_values(
    otel_backend: tuple[RuntimePerformanceTelemetry, InMemoryMetricReader],
) -> None:
    telemetry, reader = otel_backend

    for index in range(100):
        telemetry.increment(
            "xagent.auth.login.requests",
            attributes={
                "outcome": f"outcome-{index}",
                "task_id": "sensitive-task-id",
            },
        )

    metric = _metrics(reader)["xagent.auth.login.requests"]
    points = metric.data.data_points
    by_outcome = {point.attributes["outcome"]: point.value for point in points}
    assert len(points) == 65
    assert by_outcome["other"] == 36
    assert "sensitive-task-id" not in str(metric)
    assert all("task_id" not in point.attributes for point in points)


def test_observable_gauge_reads_current_value(
    otel_backend: tuple[RuntimePerformanceTelemetry, InMemoryMetricReader],
) -> None:
    telemetry, reader = otel_backend
    current = {"value": 2}
    telemetry.register_observable_gauge(
        "xagent.agent_tasks.running",
        lambda: current["value"],
        unit="{task}",
        description="Current locally running agent tasks",
    )

    first_point = _metrics(reader)["xagent.agent_tasks.running"].data.data_points[0]
    assert first_point.value == 2

    current["value"] = 5
    second_point = _metrics(reader)["xagent.agent_tasks.running"].data.data_points[0]
    assert second_point.value == 5


def test_otlp_http_exporter_posts_metrics_to_configured_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str, bytes]] = []
    received = threading.Event()

    class CollectorHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append((self.path, self.headers.get("Content-Type", ""), body))
            self.send_response(200)
            self.end_headers()
            received.set()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), CollectorHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/metrics"
    monkeypatch.setattr(
        runtime_metrics,
        "get_runtime_telemetry_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        runtime_metrics,
        "get_otel_metrics_endpoint",
        lambda: endpoint,
    )
    monkeypatch.setattr(
        runtime_metrics,
        "get_otel_export_interval_milliseconds",
        lambda: 50,
    )
    monkeypatch.setattr(
        runtime_metrics,
        "get_otel_service_name",
        lambda: "xagent-test",
    )

    runtime_metrics.shutdown_runtime_performance_telemetry()
    try:
        assert runtime_metrics.initialize_runtime_performance_telemetry() is True
        runtime_metrics.increment_counter("xagent.trace.events")
        assert received.wait(timeout=2.0)

        path, content_type, body = requests[0]
        export_request = ExportMetricsServiceRequest.FromString(body)
        metric_names = {
            metric.name
            for resource_metrics in export_request.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }
        resource_attributes = {
            attribute.key: attribute.value.string_value
            for resource_metrics in export_request.resource_metrics
            for attribute in resource_metrics.resource.attributes
        }
        assert path == "/v1/metrics"
        assert content_type == "application/x-protobuf"
        assert "xagent.trace.events" in metric_names
        assert resource_attributes["service.name"] == "xagent-test"
    finally:
        runtime_metrics.shutdown_runtime_performance_telemetry()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_thread_helper_exports_queue_execution_and_total_histograms() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    runtime_performance.bind_meter(provider.get_meter(METER_NAME))
    try:
        result = await run_in_thread_with_telemetry("unit_test", lambda: 42)

        assert result == 42
        metrics = _metrics(reader)
        for metric_name in (
            "xagent.thread_pool.queue_wait.duration",
            "xagent.thread_pool.execution.duration",
            "xagent.thread_pool.total.duration",
        ):
            point = metrics[metric_name].data.data_points[0]
            assert point.count == 1
            assert point.attributes == {"operation": "unit_test"}
    finally:
        runtime_performance.disable()
        provider.shutdown()


@pytest.mark.asyncio
async def test_event_loop_monitor_and_tracer_export_without_payload_content() -> None:
    class RecordingHandler(TraceHandler):
        async def handle_event(self, event: TraceEvent) -> None:
            await asyncio.sleep(0)

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    runtime_performance.bind_meter(provider.get_meter(METER_NAME))
    monitor = asyncio.create_task(monitor_event_loop_lag(0.001))
    try:
        tracer = Tracer()
        tracer.add_handler(RecordingHandler())
        await tracer.trace_event(
            TASK_START_GENERAL,
            task_id="sensitive-task-id",
            data={"prompt": "sensitive prompt body"},
        )
        await asyncio.sleep(0.01)
        monitor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await monitor

        metrics = _metrics(reader)
        assert "xagent.event_loop.lag" in metrics
        assert "xagent.trace.events" in metrics
        assert "xagent.trace.handler.duration" in metrics
        serialized_metrics = str(metrics)
        assert "sensitive-task-id" not in serialized_metrics
        assert "sensitive prompt body" not in serialized_metrics
    finally:
        if not monitor.done():
            monitor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await monitor
        runtime_performance.disable()
        provider.shutdown()
