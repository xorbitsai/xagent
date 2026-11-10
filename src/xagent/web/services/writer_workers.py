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
import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar

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


def active_workers() -> int:
    with _lock:
        return _active


async def drain_workers(timeout: float) -> bool:
    """Wait for every tracked thread to return; ``False`` if some did not."""
    return await wait_until(lambda: not active_workers(), timeout)
