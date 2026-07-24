"""Tests for the shared synchronous database worker boundary."""

from __future__ import annotations

import asyncio
import threading

import pytest
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from xagent.web.services.db_runtime import (
    await_task_settlement,
    is_database_pool_timeout,
    run_db_io_cancellation_safe,
)


async def _wait_for_thread_event(event: threading.Event) -> None:
    async with asyncio.timeout(1):
        while not event.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_run_db_io_offloads_operation_from_event_loop() -> None:
    loop_thread_id = threading.get_ident()
    operation_thread_ids: list[int] = []

    def operation() -> str:
        operation_thread_ids.append(threading.get_ident())
        return "done"

    result = await run_db_io_cancellation_safe(operation)

    assert result == "done"
    assert len(operation_thread_ids) == 1
    assert operation_thread_ids[0] != loop_thread_id


@pytest.mark.asyncio
async def test_run_db_io_drains_worker_before_propagating_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def operation() -> str:
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        return "done"

    caller = asyncio.create_task(run_db_io_cancellation_safe(operation))
    await _wait_for_thread_event(started)

    caller.cancel()
    await asyncio.sleep(0.02)

    assert not caller.done()
    assert not finished.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=1)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_run_db_io_preserves_worker_error_as_cancellation_cause() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    worker_error = RuntimeError("worker failed after caller cancellation")

    def operation() -> None:
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        raise worker_error

    caller = asyncio.create_task(run_db_io_cancellation_safe(operation))
    await _wait_for_thread_event(started)

    caller.cancel()
    await asyncio.sleep(0.02)
    assert not caller.done()

    release.set()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(caller, timeout=1)

    assert finished.is_set()
    assert exc_info.value.__cause__ is worker_error


@pytest.mark.asyncio
async def test_await_task_settlement_returns_late_result_and_cancellation() -> None:
    release = asyncio.Event()

    async def operation() -> str:
        await release.wait()
        return "settled"

    child = asyncio.create_task(operation())
    waiter = asyncio.create_task(await_task_settlement(child))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    result, cancellation = await asyncio.wait_for(waiter, timeout=1)

    assert result == "settled"
    assert isinstance(cancellation, asyncio.CancelledError)


def test_pool_timeout_classifier_walks_exception_chain() -> None:
    timeout = SQLAlchemyTimeoutError("pool checkout timed out")
    wrapped = RuntimeError("database operation failed")
    wrapped.__cause__ = timeout

    assert is_database_pool_timeout(wrapped) is True
    assert is_database_pool_timeout(RuntimeError("different failure")) is False
