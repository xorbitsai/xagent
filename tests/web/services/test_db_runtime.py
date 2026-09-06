"""Tests for the shared synchronous database worker boundary."""

from __future__ import annotations

import asyncio
import threading

import pytest
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from xagent.web.services.db_runtime import (
    await_task_settlement,
    drain_async_task_cancellation_safe,
    is_database_pool_timeout,
    propagate_deferred_cancellation,
    run_db_io_cancellation_safe,
)


async def _wait_for_thread_event(event: threading.Event) -> None:
    async with asyncio.timeout(1):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_propagate_deferred_cancellation_without_capture_preserves_flow() -> None:
    with propagate_deferred_cancellation(None):
        pass


def test_propagate_deferred_cancellation_without_capture_preserves_error() -> None:
    late_error = RuntimeError("late durable work failed")

    with pytest.raises(RuntimeError) as exc_info:
        with propagate_deferred_cancellation(None):
            raise late_error

    assert exc_info.value is late_error


def test_propagate_deferred_cancellation_restores_capture_on_normal_exit() -> None:
    cancellation = asyncio.CancelledError("caller cancelled")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        with propagate_deferred_cancellation(cancellation):
            pass

    assert exc_info.value is cancellation


def test_propagate_deferred_cancellation_preserves_late_error_as_cause() -> None:
    cancellation = asyncio.CancelledError("caller cancelled")
    late_error = RuntimeError("late durable work failed")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        with propagate_deferred_cancellation(cancellation):
            raise late_error

    assert exc_info.value is cancellation
    assert exc_info.value.__cause__ is late_error


def test_propagate_deferred_cancellation_preserves_later_cancellation_as_cause() -> (
    None
):
    cancellation = asyncio.CancelledError("caller cancelled")
    later_cancellation = asyncio.CancelledError("cancelled again")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        with propagate_deferred_cancellation(cancellation):
            raise later_cancellation

    assert exc_info.value is cancellation
    assert exc_info.value.__cause__ is later_cancellation


@pytest.mark.parametrize(
    "process_control_error",
    [
        SystemExit("shutdown"),
        KeyboardInterrupt("interrupt"),
        GeneratorExit("generator closed"),
    ],
)
def test_propagate_deferred_cancellation_does_not_mask_process_control_error(
    process_control_error: BaseException,
) -> None:
    cancellation = asyncio.CancelledError("caller cancelled")

    with pytest.raises(type(process_control_error)) as exc_info:
        with propagate_deferred_cancellation(cancellation):
            raise process_control_error

    assert exc_info.value is process_control_error


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


@pytest.mark.asyncio
async def test_await_task_settlement_preserves_child_process_control_after_caller_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw child control signal must not be rewritten as caller cancellation."""

    class WorkerShutdown(BaseException):
        pass

    entered = asyncio.Event()
    cancellation_processed = asyncio.Event()
    shield = asyncio.shield

    def observe_shield(task):
        if task is child:
            if entered.is_set():
                cancellation_processed.set()
            else:
                entered.set()
        return shield(task)

    monkeypatch.setattr(asyncio, "shield", observe_shield)
    child: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    waiter = asyncio.create_task(await_task_settlement(child))  # type: ignore[arg-type]
    try:
        await asyncio.wait_for(entered.wait(), timeout=30)
        waiter.cancel()
        await asyncio.wait_for(cancellation_processed.wait(), timeout=30)
        shutdown = WorkerShutdown("controlled test shutdown")
        child.set_exception(shutdown)
        with pytest.raises(WorkerShutdown) as exc_info:
            await asyncio.wait_for(waiter, timeout=30)
        assert exc_info.value is shutdown
    finally:
        if not child.done():
            child.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_drain_async_task_propagates_cancellation_after_child_settles() -> None:
    release = asyncio.Event()
    finished = asyncio.Event()

    async def operation() -> str:
        await release.wait()
        finished.set()
        return "settled"

    child = asyncio.create_task(operation())
    waiter = asyncio.create_task(drain_async_task_cancellation_safe(child))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=1)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_drain_async_task_preserves_late_child_error_as_cancellation_cause() -> (
    None
):
    release = asyncio.Event()
    child_error = RuntimeError("cleanup failed after caller cancellation")

    async def operation() -> None:
        await release.wait()
        raise child_error

    child = asyncio.create_task(operation())
    waiter = asyncio.create_task(drain_async_task_cancellation_safe(child))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(waiter, timeout=1)

    assert exc_info.value.__cause__ is child_error


@pytest.mark.asyncio
async def test_drain_async_task_drains_after_repeated_cancellation() -> None:
    release = asyncio.Event()
    finished = asyncio.Event()

    async def operation() -> None:
        await release.wait()
        finished.set()

    child = asyncio.create_task(operation())
    waiter = asyncio.create_task(drain_async_task_cancellation_safe(child))
    await asyncio.sleep(0)

    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=1)
    assert finished.is_set()


def test_pool_timeout_classifier_walks_exception_chain() -> None:
    timeout = SQLAlchemyTimeoutError("pool checkout timed out")
    wrapped = RuntimeError("database operation failed")
    wrapped.__cause__ = timeout

    assert is_database_pool_timeout(wrapped) is True
    assert is_database_pool_timeout(RuntimeError("different failure")) is False


def test_pool_timeout_classifier_walks_context_only_chain() -> None:
    timeout = SQLAlchemyTimeoutError("pool checkout timed out")
    wrapped = RuntimeError("database operation failed")
    wrapped.__context__ = timeout

    assert is_database_pool_timeout(wrapped) is True


def test_pool_timeout_classifier_stops_at_cause_context_cycle_without_timeout() -> None:
    first = RuntimeError("first failure")
    second = RuntimeError("second failure")
    first.__cause__ = second
    second.__context__ = first

    assert is_database_pool_timeout(first) is False
