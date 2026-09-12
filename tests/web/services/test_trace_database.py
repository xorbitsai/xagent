"""Admission and lifetime contracts independent of PostgreSQL availability."""

import asyncio
import threading
from contextvars import ContextVar
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import QueuePool

from xagent.web.services import trace_database, trace_handlers
from xagent.web.services.trace_database import TraceDatabaseRuntime


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


@pytest.mark.asyncio
async def test_admission_precedes_thread_submission_and_keeps_context(monkeypatch):
    runtime = TraceDatabaseRuntime(None, use_async=False, limit=2)
    release = threading.Event()
    started = []
    context = ContextVar("test_trace_context", default="missing")
    token = context.set("lease-context")

    def write():
        started.append(context.get())
        assert release.wait(3)

    tasks = [asyncio.create_task(runtime.run(write, lambda db: None)) for _ in range(6)]
    try:
        await wait_until(lambda: len(started) == 2)
        await asyncio.sleep(0.01)
        assert len(started) == 2
        # API work on the default pool still has a worker available.
        assert await asyncio.wait_for(asyncio.to_thread(lambda: 42), 1) == 42
    finally:
        release.set()
        await asyncio.gather(*tasks)
        context.reset(token)
        await runtime.close()
    assert started == ["lease-context"] * 6


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_queued", [False, True])
async def test_handler_cancellation_drains_before_releasing_admission(
    monkeypatch, cancel_queued
):
    from xagent.core.agent.trace import TASK_START_GENERAL, TraceEvent

    runtime = TraceDatabaseRuntime(None, use_async=False, limit=1)
    monkeypatch.setattr(trace_handlers, "get_trace_database_runtime", lambda: runtime)
    release = threading.Event()
    started = []
    handler = trace_handlers.DatabaseTraceHandler(42)

    def write(event):
        started.append(event.id)
        assert release.wait(3)

    monkeypatch.setattr(handler, "_sync_save_to_database", write)
    events = [
        TraceEvent(TASK_START_GENERAL, task_id="42", require_persisted=True)
        for _ in range(2)
    ]
    callers = [
        asyncio.create_task(handler._save_to_database(event)) for event in events
    ]
    try:
        await wait_until(lambda: len(runtime._operations) == 2 and len(started) == 1)
        cancelled = callers[int(cancel_queued)]
        cancelled.cancel()
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.sleep(0.01)
        assert not cancelled.done()
        assert len(started) == 1
    finally:
        release.set()
        outcomes = await asyncio.gather(*callers, return_exceptions=True)
        await runtime.close()
    assert isinstance(outcomes[int(cancel_queued)], asyncio.CancelledError)
    assert started == [event.id for event in events]


@pytest.mark.asyncio
async def test_close_drains_accepted_writes_and_rejects_new_work():
    runtime = TraceDatabaseRuntime(None, use_async=False, limit=1)
    release = threading.Event()
    started = threading.Event()

    def write():
        started.set()
        assert release.wait(3)

    work = asyncio.create_task(runtime.run(write, lambda db: None))
    await wait_until(started.is_set)
    close = asyncio.create_task(runtime.close())
    await wait_until(lambda: runtime._closing)
    try:
        with pytest.raises(RuntimeError, match="closing"):
            await runtime.run(lambda: None, lambda db: None)
        close.cancel()
        await asyncio.sleep(0.01)
        assert not close.done()
    finally:
        release.set()
        await work
        with pytest.raises(asyncio.CancelledError):
            await close
        await runtime.close()


@pytest.mark.asyncio
async def test_failed_write_returns_permit():
    runtime = TraceDatabaseRuntime(None, use_async=False, limit=1)

    def fail():
        raise ValueError("write failed")

    with pytest.raises(ValueError, match="write failed"):
        await runtime.run(fail, lambda db: None)
    await runtime.run(lambda: None, lambda db: None)
    await runtime.close()


def test_sync_shared_pool_headroom_and_async_backend_validation():
    engine = create_engine("sqlite://", poolclass=QueuePool, pool_size=3)
    try:
        assert TraceDatabaseRuntime(engine, use_async=False, limit=8).limit == 1
        memory_runtime = TraceDatabaseRuntime(engine, use_async=True, limit=2)
        assert memory_runtime.engine is None
        assert memory_runtime.limit == 1
        with pytest.raises(ValueError, match="positive"):
            TraceDatabaseRuntime(None, use_async=False, limit=0)
    finally:
        engine.dispose()


def test_runtime_is_owned_by_loop(monkeypatch):
    monkeypatch.setenv("XAGENT_ASYNC_TRACE_DB_ENABLED", "false")
    runtimes = []

    async def use():
        runtime = trace_database.get_trace_database_runtime()
        assert trace_database.get_trace_database_runtime() is runtime
        runtimes.append(runtime)
        await trace_database.close_trace_database_runtime()
        await trace_database.close_trace_database_runtime()

    asyncio.run(use())
    asyncio.run(use())
    assert runtimes[0] is not runtimes[1]


@pytest.mark.asyncio
async def test_async_config_preserves_url_options_and_bounds_pool(monkeypatch):
    source = Mock()
    source.dialect.name = "postgresql"
    source.url = make_url("postgresql://test@localhost/db?sslmode=require")
    source.get_execution_options.return_value = {"postgresql_readonly": True}
    engine = Mock(dispose=AsyncMock())
    create = Mock(return_value=engine)
    monkeypatch.setattr(trace_database, "create_async_engine", create)
    monkeypatch.setattr(trace_database, "get_engine", lambda: source)
    monkeypatch.setenv("XAGENT_ASYNC_TRACE_DB_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TRACE_DB_MAX_INFLIGHT", "6")
    runtime = trace_database.get_trace_database_runtime()
    assert runtime.limit == 6
    url = create.call_args.args[0]
    assert url.drivername == "postgresql+psycopg"
    assert url.query == source.url.query
    kwargs = create.call_args.kwargs
    assert kwargs["pool_size"] == 6
    assert kwargs["max_overflow"] == 0
    assert kwargs["hide_parameters"] is True
    assert kwargs["execution_options"] == {"postgresql_readonly": True}
    await trace_database.close_trace_database_runtime()
    engine.dispose.assert_awaited_once()


def test_missing_async_driver_fails_explicitly(monkeypatch):
    source = Mock()
    source.dialect.name = "postgresql"
    monkeypatch.setattr(
        trace_database,
        "create_async_engine",
        Mock(side_effect=ModuleNotFoundError("psycopg")),
    )
    with pytest.raises(RuntimeError, match="postgresql extra"):
        TraceDatabaseRuntime(source, use_async=True, limit=4)
