"""Async boundaries for synchronous database work."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

_T = TypeVar("_T")


def is_process_control_exception(error: BaseException) -> bool:
    """Return whether ``error`` must bypass operational error handling."""

    return not isinstance(error, (Exception, asyncio.CancelledError))


def caller_is_cancelling() -> bool:
    """Return whether the running task itself has a pending cancellation.

    A ``CancelledError`` surfacing from an awaited task that someone else
    cancelled leaves this ``False``; only a cancel aimed at the caller sets it.
    """

    current = asyncio.current_task()
    return current is not None and current.cancelling() > 0


def is_database_pool_timeout(error: BaseException) -> bool:
    """Return whether an exception chain represents pool checkout exhaustion."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SQLAlchemyTimeoutError):
            return True
        current = current.__cause__ or current.__context__
    return False


async def await_task_settlement(
    task: asyncio.Task[_T],
) -> tuple[_T, asyncio.CancelledError | None]:
    """Wait for ``task`` to settle while recording caller cancellation.

    The returned cancellation lets resource owners inspect a late result and
    perform compensation before cancellation is propagated. The task's own
    cancellation and errors still propagate immediately when the caller was
    not cancelled.
    """
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            caller_is_cancelling = current is not None and current.cancelling() > 0
            if not caller_is_cancelling:
                raise
            if cancellation is None:
                cancellation = exc
            if not task.done():
                continue
            try:
                result = task.result()
            except BaseException as task_error:
                if is_process_control_exception(task_error):
                    raise
                raise cancellation from task_error
        except BaseException as task_error:
            if is_process_control_exception(task_error):
                raise
            if cancellation is not None:
                raise cancellation from task_error
            raise
        return result, cancellation


@contextmanager
def propagate_deferred_cancellation(
    cancellation: asyncio.CancelledError | None,
) -> Iterator[None]:
    """Restore captured caller cancellation on every scope exit.

    Resource owners may need to finish durable work and best-effort follow-up
    after :func:`await_task_settlement` records cancellation. Once that work
    starts, neither an early return nor a later operational exception may
    silently replace the caller's cancellation. Process-control exceptions
    such as ``SystemExit`` and ``KeyboardInterrupt`` still propagate unchanged.
    """

    try:
        yield
    except asyncio.CancelledError as error:
        if cancellation is not None:
            raise cancellation from error
        raise
    except Exception as error:
        if cancellation is not None:
            raise cancellation from error
        raise
    if cancellation is not None:
        raise cancellation


async def run_db_io_cancellation_safe(operation: Callable[[], _T]) -> _T:
    """Run blocking database work without abandoning it on cancellation.

    ``operation`` must create, use, and close its own SQLAlchemy Session in the
    worker thread and return only detached data. If the awaiting coroutine is
    cancelled after the worker starts, the worker is drained before that
    cancellation is propagated so a late transaction cannot race cleanup.
    """
    worker = asyncio.get_running_loop().create_task(asyncio.to_thread(operation))
    result, cancellation = await await_task_settlement(worker)
    if cancellation is not None:
        raise cancellation
    return result


def task_returned(task: asyncio.Task[Any]) -> bool:
    """Whether a finished ``task`` returned, rather than raised or was cancelled.

    For deciding on a drained settlement's own outcome. A caller cancelled
    while it ran gets ``CancelledError`` from the wait either way
    (:func:`await_task_settlement`), so how the wait ended cannot tell a
    committed release from a failed one; the task can.
    """
    return task.done() and not task.cancelled() and task.exception() is None


async def drain_async_task_cancellation_safe(task: asyncio.Task[_T]) -> _T:
    """Drain an owned asynchronous task before propagating cancellation."""
    result, cancellation = await await_task_settlement(task)
    if cancellation is not None:
        raise cancellation
    return result


async def cancel_and_drain_async_task(task: asyncio.Task[_T]) -> None:
    """Cancel an owned task and drain all of its asynchronous cleanup."""

    if not task.done():
        task.cancel()

    async def drain() -> None:
        await asyncio.gather(task, return_exceptions=True)

    cleanup = asyncio.create_task(drain())
    await drain_async_task_cancellation_safe(cleanup)
