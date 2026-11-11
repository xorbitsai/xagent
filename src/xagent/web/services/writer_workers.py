"""Track database-writing worker threads that outlive their awaiting task.

``asyncio.to_thread`` hands work to a thread; cancelling the coroutine that
awaits it ends the await but not the thread, which can still commit. A host
that retires this process's writers (``app.retire_writers``) must know when
those threads are really done, so the background writers that run a batch in a
thread go through :func:`run_tracked`, and the retire waits on
:func:`drain_workers` before it reports success.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from ...core.utils.blocking import run_in_default_executor, set_blocking_tracker
from .release_hold import wait_until

T = TypeVar("T")

_lock = threading.Lock()
_active = 0


async def run_tracked(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """``asyncio.to_thread``, counted until the thread itself returns.

    Cancelling the await before the thread started drops the work and the
    count; once it started the thread keeps its count until it finishes.
    """
    global _active
    state = {"started": False, "abandoned": False}

    @functools.wraps(fn)
    def runner() -> T:
        global _active
        with _lock:
            if state["abandoned"]:
                return None  # type: ignore[return-value]
            state["started"] = True
        try:
            return fn(*args, **kwargs)
        finally:
            with _lock:
                _active -= 1

    with _lock:
        _active += 1
    try:
        return await asyncio.to_thread(runner)
    finally:
        with _lock:
            if not state["started"]:
                state["abandoned"] = True
                _active -= 1


def start_tracked_thread(
    target: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    name: str | None = None,
    daemon: bool = True,
) -> threading.Thread:
    """Start a plain thread that stays counted until ``target`` returns.

    For a worker that is joined with a timeout and so can outlive its caller.
    The count is taken before the thread starts, so a drain never sees a
    started thread as absent.
    """
    global _active

    def runner() -> None:
        global _active
        try:
            target(*args)
        finally:
            with _lock:
                _active -= 1

    thread = threading.Thread(target=runner, name=name, daemon=daemon)
    with _lock:
        _active += 1
    try:
        thread.start()
    except BaseException:
        with _lock:
            _active -= 1
        raise
    return thread


_tracking_scope: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "writer_workers_tracking_scope", default=False
)


@contextmanager
def tracked_blocking_scope() -> Iterator[None]:
    """Count the core ``run_blocking`` threads started under this context."""
    token = _tracking_scope.set(True)
    try:
        yield
    finally:
        _tracking_scope.reset(token)


async def _scoped_blocking_tracker(call: Callable[[], Any]) -> Any:
    if _tracking_scope.get():
        return await run_tracked(call)
    return await run_in_default_executor(call)


def install_blocking_tracker() -> None:
    """Route ``xagent.core`` blocking calls through this module; idempotent."""
    set_blocking_tracker(_scoped_blocking_tracker)


install_blocking_tracker()


def active_workers() -> int:
    with _lock:
        return _active


async def drain_workers(timeout: float) -> bool:
    """Wait for every tracked thread to return; ``False`` if some did not.

    ``<= 0``, not ``== 0``: a count driven below zero (a test reset racing a
    live thread) must not make a drain wait for the full timeout.
    """
    return await wait_until(lambda: active_workers() <= 0, timeout)
