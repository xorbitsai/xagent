"""Commit-failure signals for V2 execution-event writes.

One ERROR log line per failed write locates the task; a bounded-attribute
counter gives the rate per site and failure type. Metrics are asserted as
exported, so an attribute outside the exporter's allowed keys fails here.
"""

from __future__ import annotations

import logging

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.web.services.test_outbound_input_commit_retry import (
    accept_input as accept_input_fixture,
)
from tests.web.services.test_outbound_input_commit_retry import (
    any_commit,
)
from tests.web.services.test_outbound_input_commit_retry import (
    outbound as outbound_fixture,
)
from tests.web.services.test_outbound_input_commit_retry import (
    question,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from tests.web.services.test_trace_fact_commit_retry import (
    fail_commits,
    holds_fact,
)
from tests.web.services.test_trace_fact_commit_retry import (
    immediate_retries as immediate_retries_fixture,
)
from tests.web.services.test_trace_fact_commit_retry import (
    task_start,
    tool_end,
    tool_start,
)
from xagent.core.agent.checkpoint import ExecutionEventPersistenceError
from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import execution_event_signals, task_execution
from xagent.web.services.db_write_retry import database_sqlstate
from xagent.web.services.execution_event_signals import report_commit_failure
from xagent.web.services.task_lease_service import (
    TaskLeaseLostError,
    acquire_task_lease,
    bind_task_lease_context,
)
from xagent.web.services.trace_handlers import (
    CommitState,
    DatabaseTraceHandler,
    ToolAttemptAlreadyStartedError,
)
from xagent.web.tracing import ExecutionEventTraceAdapter

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
immediate_retries = immediate_retries_fixture
outbound = outbound_fixture
accept_input = accept_input_fixture

SENTINEL = "T3A-SENTINEL"


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


def exported(reader, name: str) -> dict[tuple[tuple[str, str], ...], int | float]:
    """Exported data points of one metric, keyed by their sorted attributes."""
    data = reader.get_metrics_data()
    points: dict[tuple[tuple[str, str], ...], int | float] = {}
    if data is None:
        return points
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    points[tuple(sorted(point.attributes.items()))] = point.value
    return points


def failures(reader) -> dict[tuple[tuple[str, str], ...], int | float]:
    return exported(reader, "xagent.execution_event.commit.failures")


def key(**attributes: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(attributes.items()))


def error_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def failure_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("execution_event_commit_failed ")
    ]


def fields(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split()[1:])


def operational(message: str = "server closed the connection unexpectedly"):
    return lambda: OperationalError(
        "COMMIT", {}, Exception(message), connection_invalidated=True
    )


# ---------------------------------------------------------------- trace_fact


@pytest.mark.asyncio
async def test_exhausted_trace_fact_failure_is_reported_once(
    canonical, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    fail_commits(monkeypatch, holds_fact(), committed=False, times=10)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError):
            await ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id))
    (line,) = failure_lines(caplog)
    assert line == (
        f"execution_event_commit_failed task_id={task_id} run_id=- storage=v2 "
        "component=execution-events site=trace_fact error_type=OperationalError "
        "sqlstate=- retried=True"
    )
    # The one ERROR for this V2 failure is the structured line.
    assert [r.getMessage() for r in error_records(caplog)] == [line]
    assert "Required trace event persistence failed" not in caplog.text
    assert failures(metric_reader) == {
        key(site="trace_fact", **{"error.type": "OperationalError"}, storage="v2"): 1
    }


@pytest.mark.asyncio
async def test_trace_fact_failure_carries_the_bound_run_id(
    canonical, monkeypatch, caplog
):
    factory, task_id = canonical
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
    fail_commits(monkeypatch, holds_fact(), committed=False, times=10)
    with bind_task_lease_context(lease):
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ExecutionEventPersistenceError):
                await ExecutionEventTraceAdapter(task_id).commit_event(
                    task_start(task_id)
                )
    (line,) = failure_lines(caplog)
    assert fields(line)["run_id"] == lease.run_id


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["conflict", "integrity"])
async def test_non_retryable_trace_fact_failure_is_reported_once(
    canonical, monkeypatch, caplog, metric_reader, kind
):
    factory, task_id = canonical
    adapter = ExecutionEventTraceAdapter(task_id)
    event = task_start(task_id, input="hello")
    if kind == "conflict":
        await adapter.commit_event(event)
        event.data = {"input": "changed"}
        expected = "ExecutionEventConflict"
    else:
        fail_commits(
            monkeypatch,
            holds_fact(),
            committed=False,
            error=lambda: IntegrityError("INSERT", {}, Exception("duplicate key")),
        )
        expected = "IntegrityError"
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError):
            await adapter.commit_event(event)
    (line,) = failure_lines(caplog)
    assert fields(line)["error_type"] == expected
    assert fields(line)["retried"] == "False"
    assert fields(line)["site"] == "trace_fact"
    assert [r.getMessage() for r in error_records(caplog)] == [line]
    assert failures(metric_reader) == {
        key(site="trace_fact", **{"error.type": expected}, storage="v2"): 1
    }


@pytest.mark.asyncio
async def test_recovered_trace_fact_write_reports_nothing(
    canonical, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    fail_commits(monkeypatch, holds_fact(), committed=True)
    with caplog.at_level(logging.DEBUG):
        await ExecutionEventTraceAdapter(task_id).commit_event(task_start(task_id))
    assert failure_lines(caplog) == []
    assert error_records(caplog) == []
    assert failures(metric_reader) == {}


@pytest.mark.asyncio
async def test_tool_start_guard_failure_names_the_attempt(
    canonical, caplog, metric_reader
):
    factory, task_id = canonical
    adapter = ExecutionEventTraceAdapter(task_id)
    first = tool_start(task_id, "attempt-guard")
    first.require_persisted = True
    await adapter._save_to_database(first, commit_state=CommitState())
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError) as caught:
            await adapter.commit_event(tool_start(task_id, "attempt-guard"))
    cause = caught.value.__cause__
    assert isinstance(cause, ToolAttemptAlreadyStartedError)
    assert isinstance(cause, RuntimeError)
    assert cause.tool_attempt_id == "attempt-guard"
    assert "already started" in str(cause)
    (line,) = failure_lines(caplog)
    assert fields(line)["error_type"] == "ToolAttemptAlreadyStartedError"
    assert fields(line)["tool_attempt_id"] == "attempt-guard"
    assert [r.getMessage() for r in error_records(caplog)] == [line]
    assert failures(metric_reader) == {
        key(
            site="trace_fact",
            **{"error.type": "ToolAttemptAlreadyStartedError"},
            storage="v2",
        ): 1
    }


@pytest.mark.asyncio
async def test_tool_event_failure_carries_the_attempt_id_for_any_error(
    canonical, monkeypatch, caplog
):
    factory, task_id = canonical
    fail_commits(monkeypatch, holds_fact(), committed=False, times=10)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError):
            await ExecutionEventTraceAdapter(task_id).commit_event(
                tool_end(task_id, "attempt-end")
            )
    (line,) = failure_lines(caplog)
    assert fields(line)["tool_attempt_id"] == "attempt-end"
    assert fields(line)["error_type"] == "OperationalError"


# ------------------------------------------------------------------ outbound


@pytest.mark.asyncio
async def test_exhausted_outbound_write_is_reported_once(
    outbound, monkeypatch, caplog, metric_reader
):
    factory, task_id, published, attempts = outbound
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)
    fail_commits(monkeypatch, holds_fact(), committed=False, times=10)
    with bind_task_lease_context(lease):
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ExecutionEventPersistenceError):
                await task_execution.make_agent_outbound_handler(
                    task_id, authoritative=True
                )(question())
    assert len(attempts) == 3
    (line,) = failure_lines(caplog)
    parsed = fields(line)
    assert parsed["task_id"] == str(task_id)
    assert parsed["run_id"] == lease.run_id
    assert parsed["site"] == "outbound"
    assert parsed["retried"] == "True"
    assert parsed["error_type"] == "OperationalError"
    assert failures(metric_reader) == {
        key(site="outbound", **{"error.type": "OperationalError"}, storage="v2"): 1
    }


@pytest.mark.asyncio
async def test_non_retryable_outbound_write_reports_retried_false(
    outbound, monkeypatch, caplog, metric_reader
):
    factory, task_id, published, attempts = outbound
    fail_commits(
        monkeypatch,
        holds_fact(),
        committed=False,
        error=lambda: IntegrityError("INSERT", {}, Exception("duplicate key")),
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError):
            await task_execution.make_agent_outbound_handler(
                task_id, authoritative=True
            )(question())
    (line,) = failure_lines(caplog)
    assert fields(line)["retried"] == "False"
    assert fields(line)["run_id"] == "-"
    assert fields(line)["error_type"] == "IntegrityError"


@pytest.mark.asyncio
async def test_outbound_lease_loss_reports_the_lease_error_type(
    outbound, monkeypatch, caplog, metric_reader
):
    factory, task_id, published, attempts = outbound
    with factory() as db:
        lease = acquire_task_lease(db, task_id, new_run=True)

    def replace_lease():
        with factory() as db:
            db.get(Task, task_id).run_id = "replacement-run"
            db.commit()

    fail_commits(monkeypatch, holds_fact(), committed=False, between=replace_lease)
    with bind_task_lease_context(lease):
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ExecutionEventPersistenceError) as caught:
                await task_execution.make_agent_outbound_handler(
                    task_id, authoritative=True
                )(question())
    assert isinstance(caught.value.__cause__, TaskLeaseLostError)
    (line,) = failure_lines(caplog)
    assert fields(line)["error_type"] == "TaskLeaseLostError"
    # The first attempt failed retryably and the rerun lost the lease.
    assert fields(line)["retried"] == "True"


@pytest.mark.asyncio
async def test_recovered_outbound_write_reports_nothing(
    outbound, monkeypatch, caplog, metric_reader
):
    factory, task_id, published, attempts = outbound
    fail_commits(monkeypatch, holds_fact(), committed=True)
    with caplog.at_level(logging.DEBUG):
        await task_execution.make_agent_outbound_handler(task_id, authoritative=True)(
            question()
        )
    assert len(attempts) == 2
    assert failure_lines(caplog) == []
    assert failures(metric_reader) == {}


# --------------------------------------------------------------- input_accept


@pytest.mark.asyncio
async def test_exhausted_input_accept_is_reported_once(
    accept_input, monkeypatch, caplog, metric_reader
):
    factory, task_id, run, attempts = accept_input
    fail_commits(monkeypatch, any_commit, committed=False, times=10)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(OperationalError):
            await run()
    (line,) = failure_lines(caplog)
    parsed = fields(line)
    assert parsed["site"] == "input_accept"
    assert parsed["retried"] == "True"
    assert parsed["error_type"] == "OperationalError"
    assert parsed["task_id"] == str(task_id)
    assert parsed["run_id"] != "-"
    assert failures(metric_reader) == {
        key(site="input_accept", **{"error.type": "OperationalError"}, storage="v2"): 1
    }


@pytest.mark.asyncio
async def test_recovered_input_accept_reports_nothing(
    accept_input, monkeypatch, caplog, metric_reader
):
    factory, task_id, run, attempts = accept_input
    fail_commits(monkeypatch, any_commit, committed=True)
    with caplog.at_level(logging.DEBUG):
        await run()
    assert len(attempts) == 2
    assert failure_lines(caplog) == []
    assert failures(metric_reader) == {}


# ---------------------------------------------------------- terminal_failure


@pytest.mark.asyncio
async def test_terminal_failure_commit_failure_is_reported(
    canonical, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.RUNNING
        task.runner_id = None
        task.run_id = "run-terminal"
        db.commit()
    fail_commits(monkeypatch, any_commit, committed=False)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError, match="Terminal failure"):
            task_execution._terminal_task_error_payload(
                task_id, "boom", expected_run_id="run-terminal"
            )
    (line,) = failure_lines(caplog)
    parsed = fields(line)
    assert parsed["site"] == "terminal_failure"
    assert parsed["run_id"] == "run-terminal"
    assert parsed["retried"] == "False"
    assert parsed["error_type"] == "OperationalError"
    assert failures(metric_reader) == {
        key(
            site="terminal_failure",
            **{"error.type": "OperationalError"},
            storage="v2",
        ): 1
    }


@pytest.mark.asyncio
async def test_terminal_failure_read_after_commit_is_not_a_commit_failure(
    canonical, monkeypatch, caplog, metric_reader
):
    """The fact committed; failing to read the reply loses no write."""
    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        task.status = TaskStatus.RUNNING
        task.runner_id = None
        task.run_id = "run-terminal"
        db.commit()

    def unreadable(*_args, **_kwargs):
        raise OperationalError("SELECT", {}, Exception("connection lost"))

    monkeypatch.setattr(task_execution, "_task_error_payload", unreadable)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError):
            task_execution._terminal_task_error_payload(
                task_id, "boom", expected_run_id="run-terminal"
            )
    with factory() as db:
        assert db.get(Task, task_id).status == TaskStatus.FAILED
    assert failure_lines(caplog) == []
    assert failures(metric_reader) == {}


# ------------------------------------------------------------ writes counter


def general_event(task_id: int) -> TraceEvent:
    return TraceEvent(
        TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL),
        task_id=str(task_id),
        data={"input": "hello"},
    )


@pytest.mark.asyncio
async def test_trace_writes_counter_carries_the_storage_version(
    canonical, metric_reader
):
    factory, task_id = canonical
    await ExecutionEventTraceAdapter(task_id).commit_event(general_event(task_id))
    assert exported(metric_reader, "xagent.trace.database.writes") == {
        key(outcome="succeeded", storage="v2"): 1
    }
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = 1
        db.commit()
    await DatabaseTraceHandler(task_id).handle_event(general_event(task_id))
    assert exported(metric_reader, "xagent.trace.database.writes") == {
        key(outcome="succeeded", storage="v2"): 1,
        key(outcome="succeeded", storage="v1"): 1,
    }


@pytest.mark.asyncio
async def test_trace_writes_counter_storage_on_failed_outcomes(
    canonical, monkeypatch, metric_reader
):
    factory, task_id = canonical
    adapter = ExecutionEventTraceAdapter(task_id)
    event = task_start(task_id, input="hello")
    await adapter.commit_event(event)
    event.data = {"input": "changed"}
    with pytest.raises(ExecutionEventPersistenceError):
        await adapter.commit_event(event)
    fail_commits(monkeypatch, holds_fact(), committed=False, times=10)
    with pytest.raises(ExecutionEventPersistenceError):
        await adapter.commit_event(task_start(task_id, other=1))
    assert set(exported(metric_reader, "xagent.trace.database.writes")) == {
        key(outcome="succeeded", storage="v2"),
        key(outcome="failed", storage="v2"),
        key(outcome="retryable_failed", storage="v2"),
    }


@pytest.mark.asyncio
async def test_v1_required_failure_keeps_its_own_error_line(
    canonical, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    with factory() as db:
        task = db.get(Task, task_id)
        task.conversation_storage_version = 1
        db.commit()
    fail_commits(
        monkeypatch,
        lambda db: True,
        committed=False,
        error=lambda: ValueError("bad"),
    )
    event = general_event(task_id)
    event.require_persisted = True
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError):
            await DatabaseTraceHandler(task_id)._save_to_database(event)
    assert "Required trace event persistence failed" in caplog.text
    assert failure_lines(caplog) == []
    assert failures(metric_reader) == {}
    assert exported(metric_reader, "xagent.trace.database.writes") == {
        key(outcome="failed", storage="v1"): 1
    }


# ------------------------------------------------------- privacy / robustness


@pytest.mark.asyncio
async def test_no_payload_or_exception_text_reaches_execution_event_logs(
    canonical, monkeypatch, caplog
):
    factory, task_id = canonical
    fail_commits(
        monkeypatch,
        holds_fact(),
        committed=False,
        times=10,
        error=operational(f"{SENTINEL} in exception"),
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExecutionEventPersistenceError):
            await ExecutionEventTraceAdapter(task_id).commit_event(
                task_start(task_id, secret=SENTINEL)
            )
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("execution_event_")
    ]
    assert len(lines) == 1
    assert all(SENTINEL not in line for line in lines)
    assert all(r.exc_info is None for r in caplog.records if r.levelno >= 40)


def test_report_never_raises_when_the_counter_fails(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("counter down")

    monkeypatch.setattr(execution_event_signals, "increment_counter", boom)
    with caplog.at_level(logging.DEBUG):
        report_commit_failure(
            "outbound", ValueError("x"), task_id=1, run_id=None, retried=False
        )


def test_report_never_raises_when_the_logger_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("logger down")

    monkeypatch.setattr(execution_event_signals.logger, "error", boom)
    report_commit_failure(
        "outbound", ValueError("x"), task_id=1, run_id="r", retried=True
    )


def test_error_type_skips_the_persistence_wrapper_and_finds_the_sqlstate(
    caplog, metric_reader
):
    class Driver(Exception):
        sqlstate = "40P01"

    root = OperationalError("COMMIT", {}, Driver("deadlock"))
    wrapped = ExecutionEventPersistenceError("Conversation event commit failed")
    wrapped.__cause__ = root
    assert database_sqlstate(wrapped) == "40P01"
    assert database_sqlstate(ValueError("x")) is None
    with caplog.at_level(logging.DEBUG):
        report_commit_failure(
            "trace_fact", wrapped, task_id=7, run_id="r1", retried=False
        )
        report_commit_failure(
            "trace_fact",
            ExecutionEventPersistenceError("bare"),
            task_id=7,
            run_id=None,
            retried=False,
        )
    first, second = failure_lines(caplog)
    assert fields(first)["error_type"] == "OperationalError"
    assert fields(first)["sqlstate"] == "40P01"
    assert "tool_attempt_id" not in fields(first)
    assert fields(second)["error_type"] == "ExecutionEventPersistenceError"
    assert fields(second)["sqlstate"] == "-"
    assert failures(metric_reader) == {
        key(site="trace_fact", **{"error.type": "OperationalError"}, storage="v2"): 1,
        key(
            site="trace_fact",
            **{"error.type": "ExecutionEventPersistenceError"},
            storage="v2",
        ): 1,
    }
