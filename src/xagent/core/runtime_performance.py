"""OpenTelemetry metrics for runtime performance hot paths.

The module owns a private ``MeterProvider`` when OTLP export is enabled.  With
no exporter configured it keeps a no-op meter, so instrumentation remains safe
for library and local-development use.  Only bounded operational attributes are
accepted; task/user identifiers, SQL, prompts, and payload content are never
recorded.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import socket
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from typing import Any, TypeVar

from opentelemetry.metrics import Meter, NoOpMeterProvider, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import (
    ExplicitBucketHistogramAggregation,
    View,
)
from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID, SERVICE_NAME, Resource

from ..config import (
    get_otel_export_interval_milliseconds,
    get_otel_metrics_endpoint,
    get_otel_service_name,
    get_runtime_telemetry_enabled,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

METER_NAME = "xagent.runtime"
_SHUTDOWN_TIMEOUT_MILLISECONDS = 5_000
_MAX_ATTRIBUTE_VALUES_PER_METRIC = 64
_OVERFLOW_ATTRIBUTE_VALUE = "other"
_ALLOWED_ATTRIBUTE_KEYS = frozenset(
    {
        "error.type",
        "event.type",
        "handler",
        "operation",
        "outcome",
    }
)

_DURATION_BUCKETS_MS = (
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    30_000.0,
    60_000.0,
)
_BYTE_BUCKETS = (
    256.0,
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
)
_COUNT_BUCKETS = (
    0.0,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    512.0,
    1_024.0,
)

_DURATION_METRICS = frozenset(
    {
        "xagent.auth.login.password_verify.duration",
        "xagent.auth.login.token_creation.duration",
        "xagent.auth.login.total.duration",
        "xagent.auth.login.sync_lookup.duration",
        "xagent.auth.login.sync_commit.duration",
        "xagent.event_loop.lag",
        "xagent.thread_pool.execution.duration",
        "xagent.thread_pool.queue_wait.duration",
        "xagent.thread_pool.total.duration",
        "xagent.trace.database.commit.duration",
        "xagent.trace.database.serialization.duration",
        "xagent.trace.dispatch.duration",
        "xagent.trace.handler.duration",
        "xagent.websocket.broadcast.duration",
        "xagent.websocket.trace_handler.duration",
        "xagent.websocket.trace_serialization.duration",
    }
)
_BYTE_METRICS = frozenset(
    {
        "xagent.trace.payload.size",
        "xagent.websocket.payload.size",
    }
)
_COUNT_METRICS = frozenset({"xagent.websocket.broadcast.fanout"})


class RuntimePerformanceTelemetry:
    """Small instrumentation facade backed by an OpenTelemetry ``Meter``."""

    def __init__(self, meter: Meter | None = None) -> None:
        self._lock = threading.Lock()
        self._meter = meter or _no_op_meter()
        self._enabled = meter is not None
        self._counters: dict[str, Any] = {}
        self._histograms: dict[tuple[str, str], Any] = {}
        self._observable_gauges: dict[str, Any] = {}
        self._observable_gauge_definitions: dict[
            str, tuple[Callable[[], float], str, str]
        ] = {}
        self._attribute_values: dict[tuple[str, str], set[str]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind_meter(self, meter: Meter) -> None:
        """Replace the no-op meter and recreate registered instruments."""

        with self._lock:
            self._meter = meter
            self._enabled = True
            self._counters.clear()
            self._histograms.clear()
            self._observable_gauges.clear()
            self._attribute_values.clear()
            definitions = list(self._observable_gauge_definitions.items())
            for metric, (callback, unit, description) in definitions:
                self._create_observable_gauge_locked(
                    metric,
                    callback,
                    unit=unit,
                    description=description,
                )

    def disable(self) -> None:
        """Return to no-op instrumentation after the owned provider stops."""

        with self._lock:
            self._meter = _no_op_meter()
            self._enabled = False
            self._counters.clear()
            self._histograms.clear()
            self._observable_gauges.clear()
            self._attribute_values.clear()

    def increment(
        self,
        metric: str,
        *,
        amount: int = 1,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        if not self._enabled:
            return
        bounded_attributes = self._bounded_attributes(metric, attributes)
        with self._lock:
            counter = self._counters.get(metric)
            if counter is None:
                counter = self._meter.create_counter(metric)
                self._counters[metric] = counter
        counter.add(amount, bounded_attributes)

    def observe(
        self,
        metric: str,
        value: float,
        *,
        unit: str,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        if not self._enabled:
            return
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0:
            return
        bounded_attributes = self._bounded_attributes(metric, attributes)
        key = (metric, unit)
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = self._meter.create_histogram(metric, unit=unit)
                self._histograms[key] = histogram
        histogram.record(numeric_value, bounded_attributes)

    def register_observable_gauge(
        self,
        metric: str,
        callback: Callable[[], float],
        *,
        unit: str,
        description: str,
    ) -> None:
        with self._lock:
            self._observable_gauge_definitions[metric] = (
                callback,
                unit,
                description,
            )
            if self._enabled and metric not in self._observable_gauges:
                self._create_observable_gauge_locked(
                    metric,
                    callback,
                    unit=unit,
                    description=description,
                )

    def _create_observable_gauge_locked(
        self,
        metric: str,
        callback: Callable[[], float],
        *,
        unit: str,
        description: str,
    ) -> None:
        def observe(_options: Any) -> list[Observation]:
            try:
                value = float(callback())
            except Exception:
                return []
            if not math.isfinite(value):
                return []
            return [Observation(value)]

        self._observable_gauges[metric] = self._meter.create_observable_gauge(
            metric,
            callbacks=[observe],
            unit=unit,
            description=description,
        )

    def _bounded_attributes(
        self,
        metric: str,
        attributes: Mapping[str, str] | None,
    ) -> dict[str, str] | None:
        if not attributes:
            return None
        bounded: dict[str, str] = {}
        with self._lock:
            for key, raw_value in attributes.items():
                if key not in _ALLOWED_ATTRIBUTE_KEYS:
                    continue
                value = str(raw_value)
                seen = self._attribute_values.setdefault((metric, key), set())
                if value not in seen and len(seen) >= _MAX_ATTRIBUTE_VALUES_PER_METRIC:
                    value = _OVERFLOW_ATTRIBUTE_VALUE
                seen.add(value)
                bounded[key] = value
        return bounded or None


def _no_op_meter() -> Meter:
    return NoOpMeterProvider().get_meter(METER_NAME)


def _histogram_views() -> list[View]:
    views = [
        View(
            instrument_name=metric,
            aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS_MS),
        )
        for metric in sorted(_DURATION_METRICS)
    ]
    views.extend(
        View(
            instrument_name=metric,
            aggregation=ExplicitBucketHistogramAggregation(_BYTE_BUCKETS),
        )
        for metric in sorted(_BYTE_METRICS)
    )
    views.extend(
        View(
            instrument_name=metric,
            aggregation=ExplicitBucketHistogramAggregation(_COUNT_BUCKETS),
        )
        for metric in sorted(_COUNT_METRICS)
    )
    return views


runtime_performance = RuntimePerformanceTelemetry()
_provider_lock = threading.Lock()
_meter_provider: MeterProvider | None = None


def initialize_runtime_performance_telemetry() -> bool:
    """Configure OTLP/HTTP metrics export when explicitly enabled."""

    global _meter_provider
    try:
        if not get_runtime_telemetry_enabled():
            return False
        endpoint = get_otel_metrics_endpoint()
    except ValueError:
        # Do not echo an endpoint that may contain credentials or query data.
        logger.warning("Invalid OTLP metrics endpoint; runtime telemetry is disabled")
        return False
    if endpoint is None:
        logger.warning(
            "Runtime telemetry is enabled but no OTLP metrics endpoint is configured"
        )
        return False

    with _provider_lock:
        if _meter_provider is not None:
            return True

        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=get_otel_export_interval_milliseconds(),
        )
        resource = Resource.create(
            {
                SERVICE_NAME: get_otel_service_name(),
                SERVICE_INSTANCE_ID: f"{socket.gethostname()}:{os.getpid()}",
            }
        )
        provider = MeterProvider(
            metric_readers=[reader],
            resource=resource,
            shutdown_on_exit=False,
            views=_histogram_views(),
        )
        runtime_performance.bind_meter(provider.get_meter(METER_NAME))
        _meter_provider = provider
    logger.info("Runtime performance telemetry OTLP export is enabled")
    return True


def shutdown_runtime_performance_telemetry() -> None:
    """Flush and stop the owned provider without affecting application shutdown."""

    global _meter_provider
    with _provider_lock:
        provider = _meter_provider
        _meter_provider = None
        runtime_performance.disable()
    if provider is None:
        return
    try:
        provider.shutdown(timeout_millis=_SHUTDOWN_TIMEOUT_MILLISECONDS)
    except Exception:
        logger.warning("Could not shut down runtime telemetry", exc_info=True)


def increment_counter(
    metric: str,
    *,
    amount: int = 1,
    attributes: Mapping[str, str] | None = None,
) -> None:
    try:
        runtime_performance.increment(metric, amount=amount, attributes=attributes)
    except Exception:
        # Observability must never change the behavior of the measured path.
        return


def observe_value(
    metric: str,
    value: float,
    *,
    unit: str,
    attributes: Mapping[str, str] | None = None,
) -> None:
    try:
        runtime_performance.observe(
            metric,
            value,
            unit=unit,
            attributes=attributes,
        )
    except Exception:
        return


def register_observable_gauge(
    metric: str,
    callback: Callable[[], float],
    *,
    unit: str,
    description: str,
) -> None:
    try:
        runtime_performance.register_observable_gauge(
            metric,
            callback,
            unit=unit,
            description=description,
        )
    except Exception:
        return


@contextmanager
def observe_duration(
    metric: str,
    *,
    attributes: Mapping[str, str] | None = None,
) -> Generator[None, None, None]:
    if not runtime_performance.enabled:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        observe_value(
            metric,
            (time.perf_counter() - started) * 1_000.0,
            unit="ms",
            attributes=attributes,
        )


async def run_in_thread_with_telemetry(
    operation: str,
    function: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Run synchronous work in asyncio's executor and time queue vs execution."""

    if not runtime_performance.enabled:
        return await asyncio.to_thread(function, *args, **kwargs)

    submitted_at = time.perf_counter()
    attributes = {"operation": operation}

    def invoke() -> _T:
        started_at = time.perf_counter()
        observe_value(
            "xagent.thread_pool.queue_wait.duration",
            (started_at - submitted_at) * 1_000.0,
            unit="ms",
            attributes=attributes,
        )
        try:
            return function(*args, **kwargs)
        finally:
            observe_value(
                "xagent.thread_pool.execution.duration",
                (time.perf_counter() - started_at) * 1_000.0,
                unit="ms",
                attributes=attributes,
            )

    try:
        return await asyncio.to_thread(invoke)
    finally:
        observe_value(
            "xagent.thread_pool.total.duration",
            (time.perf_counter() - submitted_at) * 1_000.0,
            unit="ms",
            attributes=attributes,
        )


async def monitor_event_loop_lag(interval_seconds: float = 0.5) -> None:
    """Continuously sample event-loop scheduling delay until cancelled."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    loop = asyncio.get_running_loop()
    expected_at = loop.time() + interval_seconds
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            observed_at = loop.time()
            lag_ms = max(0.0, observed_at - expected_at) * 1_000.0
            observe_value("xagent.event_loop.lag", lag_ms, unit="ms")
            increment_counter("xagent.event_loop.samples")
            expected_at = observed_at + interval_seconds
    except asyncio.CancelledError:
        raise


def start_event_loop_lag_monitor(
    interval_seconds: float = 0.5,
) -> asyncio.Task[None]:
    monitor = monitor_event_loop_lag(interval_seconds)
    try:
        return asyncio.create_task(
            monitor,
            name="runtime-performance-event-loop-lag",
        )
    except Exception:
        monitor.close()
        raise


async def stop_event_loop_lag_monitor(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task
