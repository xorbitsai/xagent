"""Async boundaries for synchronous database work."""

from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

_T = TypeVar("_T")


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
                raise cancellation from task_error
        except BaseException as task_error:
            if cancellation is not None:
                raise cancellation from task_error
            raise
        return result, cancellation


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
