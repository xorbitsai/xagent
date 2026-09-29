"""Pacing at the durable command boundary, independent of active capacity."""

import asyncio
import multiprocessing
import os

from sqlalchemy import event, literal, text
from sqlalchemy.engine import make_url

from tests.web.services.test_task_execution_admission import (
    Execution,
    _dispatch_process,
)
from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import (
    enqueue,
)
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.web.services import task_admission_pacing as pacing
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission as admission

host = host_fixture
engine = engine_fixture


async def test_start_burst_and_refill_have_no_window_rollover_double_burst(
    host, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr(pacing, "database_time", lambda: literal(now[0]))
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "paced",
            10,
            20,
            pacing=pacing.StartupPacing(interval_seconds=1, burst=2, lane="batch"),
        )
    )
    commands = [enqueue(host) for _ in range(5)]
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    assert await transport.dispatch_one_task_command(execute)
    assert not await transport.dispatch_one_task_command(execute)
    now[0] = 100.999
    assert not await transport.dispatch_one_task_command(execute)
    now[0] = 101.0
    assert await transport.dispatch_one_task_command(execute)
    assert not await transport.dispatch_one_task_command(execute)
    now[0] = 102.0
    assert await transport.dispatch_one_task_command(execute)
    assert execute.started == [command.command_id for command in commands[:4]]
    assert transport.load_task_command(commands[-1].command_id).attempt_count == 0


async def test_batch_start_pacing_cannot_spend_interactive_allowance(host, monkeypatch):
    monkeypatch.setattr(pacing, "database_time", lambda: literal(100.0))
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "batch",
            5,
            20,
            pacing=pacing.StartupPacing(interval_seconds=30, burst=1, lane="batch"),
        )
    )
    batch = [enqueue(host) for _ in range(3)]
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "interactive",
            1,
            5,
            pacing=pacing.StartupPacing(
                interval_seconds=1, burst=1, lane="interactive"
            ),
        )
    )
    interactive = enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    assert await transport.dispatch_one_task_command(execute)
    assert execute.started == [batch[0].command_id, interactive.command_id]
    assert not await transport.dispatch_one_task_command(execute)


async def test_idle_refill_is_bounded_and_backward_clock_does_not_create_allowance(
    host, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr(pacing, "database_time", lambda: literal(now[0]))
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "idle",
            10,
            20,
            pacing=pacing.StartupPacing(interval_seconds=1, burst=2),
        )
    )
    for _ in range(6):
        enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    assert await transport.dispatch_one_task_command(execute)
    now[0] = 99.0
    assert not await transport.dispatch_one_task_command(execute)
    now[0] = 1000.0
    assert await transport.dispatch_one_task_command(execute)
    assert await transport.dispatch_one_task_command(execute)
    assert not await transport.dispatch_one_task_command(execute)
    assert len(execute.started) == 4


async def test_live_guidance_does_not_consume_another_start(host, monkeypatch):
    from xagent.web.models.task_command import TaskExecutionCommand

    now = [100.0]
    monkeypatch.setattr(pacing, "database_time", lambda: literal(now[0]))
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "guidance",
            5,
            20,
            pacing=pacing.StartupPacing(interval_seconds=1, burst=1),
        )
    )
    first = enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, first.command_id).task_id
    message = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.MESSAGE)

    async def inject(command):
        assert command.id == message.command_id
        return {}

    assert await transport.dispatch_one_task_command(inject)
    next_turn = enqueue(host)
    now[0] = 101.0
    assert await transport.dispatch_one_task_command(execute)
    assert execute.started == [first.command_id, next_turn.command_id]


async def test_guidance_that_becomes_a_new_turn_waits_for_startup_allowance(
    host, monkeypatch
):
    from unittest.mock import AsyncMock

    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.services import task_command_execution
    from xagent.web.services.task_command_execution import execute_durable_task_command

    now = [100.0]
    monkeypatch.setattr(pacing, "database_time", lambda: literal(now[0]))
    monkeypatch.setattr(task_command_execution, "publish_task_event", AsyncMock())
    monkeypatch.setattr(
        task_command_execution, "get_session_local", lambda: host.sessions
    )
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "guidance-race",
            2,
            10,
            pacing=pacing.StartupPacing(interval_seconds=60, burst=1),
        )
    )
    first = enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    with host.sessions() as db:
        task_id = db.get(TaskExecutionCommand, first.command_id).task_id
        message = transport.enqueue_task_command(
            db,
            task_id=task_id,
            actor_user_id=host.user,
            command_id="next-message",
            kind=transport.TaskCommandKind.MESSAGE,
            payload={"message": "Continue", "client_message_id": "next-message"},
        )

    async def settle_before_routing(command):
        execute.finish.set()
        execute.cleanup.set()
        await execute.terminal.wait()
        return await execute_durable_task_command(command)

    assert await transport.dispatch_one_task_command(settle_before_routing)
    waiting = transport.load_task_command(message.command_id)
    assert waiting.status == "pending"
    assert (waiting.failure_count, waiting.defer_count) == (0, 0)
    assert not await transport.dispatch_one_task_command(execute)
    now[0] = 160.0
    # The speculative guidance claim remains a monotonic fencing attempt.
    from tests.web.services.test_task_execution_admission import dispatch_next

    await dispatch_next(execute)
    assert execute.started == [first.command_id, message.command_id]


async def test_snapshot_distinguishes_pacing_from_capacity_and_bounds_scope(
    host, monkeypatch
):
    import pytest

    from xagent.web.services import task_admission_observation as observation

    monkeypatch.setattr(pacing, "database_time", lambda: literal(100.0))
    monkeypatch.setattr(observation, "database_time", lambda: literal(100.0))
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "observed",
            2,
            20,
            pacing=pacing.StartupPacing(interval_seconds=10, burst=1, lane="batch"),
        )
    )
    enqueue(host)
    enqueue(host)
    execute = Execution(host)
    assert await transport.dispatch_one_task_command(execute)
    with host.sessions() as db:
        row = observation.read_admission_snapshot(db, ["observed"])[0]
        assert (row.active, row.pending, row.capacity, row.max_pending) == (1, 1, 2, 20)
        assert row.lane == "batch"
        assert row.delay_reason == "startup_pacing"
        assert row.startup_delay_seconds == 10.0
        assert observation.read_admission_snapshot(db, ["foreign"]) == []
        with pytest.raises(ValueError, match="32"):
            observation.read_admission_snapshot(db, [str(i) for i in range(33)])
    monkeypatch.setattr(pacing, "database_time", lambda: literal(110.0))
    monkeypatch.setattr(observation, "database_time", lambda: literal(110.0))
    assert await transport.dispatch_one_task_command(execute)
    enqueue(host)
    with host.sessions() as db:
        row = observation.read_admission_snapshot(db, ["observed"])[0]
        assert (row.active, row.pending, row.delay_reason) == (2, 1, "capacity")


async def test_metrics_count_committed_claims_and_refusals_without_tenant_labels(
    host, monkeypatch
):
    import pytest
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from xagent.core.runtime_performance import RuntimePerformanceTelemetry
    from xagent.web.services import task_admission_observation as observation

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    monkeypatch.setattr(
        observation,
        "runtime_performance",
        RuntimePerformanceTelemetry(provider.get_meter("admission-test")),
    )
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "team:private-id:batch",
            1,
            1,
            pacing=pacing.StartupPacing(interval_seconds=1, burst=1, lane="batch"),
        )
    )

    def set_timezone(connection, record, proxy):
        if host.engine.dialect.name == "postgresql":
            with connection.cursor() as cursor:
                cursor.execute("SET TIME ZONE 'Asia/Shanghai'")

    event.listen(host.engine, "checkout", set_timezone)
    try:
        from datetime import timezone

        from xagent.web.models.task_command import TaskExecutionCommand

        accepted = enqueue(host)
        with host.sessions() as db:
            created = db.get(TaskExecutionCommand, accepted.command_id).created_at
            instant = (
                created if created.tzinfo else created.replace(tzinfo=timezone.utc)
            ).timestamp()
            monkeypatch.setattr(
                observation, "database_time", lambda: literal(instant + 30)
            )
            snapshot = observation.read_admission_snapshot(
                db, ["team:private-id:batch"]
            )[0]
            assert snapshot.oldest_pending_seconds == pytest.approx(30)
        with pytest.raises(admission.AdmissionQueueFull):
            enqueue(host)
        execute = Execution(host)
        assert await transport.dispatch_one_task_command(execute)
        metrics = {
            metric.name: metric
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        assert metrics["task.admission.claims"].data.data_points[0].value == 1
        assert metrics["task.admission.queue_full"].data.data_points[0].value == 1
        assert metrics["task.admission.initial_wait"].data.data_points[0].count == 1
        assert metrics["task.admission.initial_wait"].data.data_points[
            0
        ].sum == pytest.approx(30)
        for metric in metrics.values():
            for point in metric.data.data_points:
                assert set(point.attributes) <= {"operation", "outcome"}
                assert point.attributes["operation"] == "batch"
                assert "private-id" not in str(point.attributes)
    finally:
        event.remove(host.engine, "checkout", set_timezone)
        provider.shutdown()


async def test_three_processes_share_one_startup_allowance(host):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy(
            "shared",
            10,
            20,
            pacing=pacing.StartupPacing(interval_seconds=60, burst=2, lane="batch"),
        )
    )
    accepted = [enqueue(host).command_id for _ in range(6)]
    if host.engine.dialect.name == "postgresql":
        with host.engine.connect() as connection:
            database = connection.scalar(text("SELECT current_database()"))
        url = (
            make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
            .set(database=database)
            .render_as_string(hide_password=False)
        )
    else:
        url = str(host.engine.url)
    context = multiprocessing.get_context("spawn")
    barrier, release, results = context.Barrier(3), context.Event(), context.Queue()
    workers = [
        context.Process(target=_dispatch_process, args=(url, barrier, release, results))
        for _ in range(3)
    ]
    started, scanned = [], []
    try:
        for worker in workers:
            worker.start()
        while len(scanned) < 3:
            kind, value = await asyncio.to_thread(results.get, True, 60)
            assert kind != "error", value
            (started if kind == "started" else scanned).append(value)
        assert sorted(started) == accepted[:2]
        assert all(
            transport.load_task_command(command).attempt_count == 0
            for command in accepted[2:]
        )
    finally:
        release.set()
        for worker in workers:
            await asyncio.to_thread(worker.join, 15)
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
        results.close()
    assert all(worker.exitcode == 0 for worker in workers)
