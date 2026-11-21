"""Ownership of MCP listing work that can outlive its requesting coroutine."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, Callable


_active_lifetimes: set[MCPLoadLifetime] = set()


class MCPLoadLifetime:
    """Return admission only after the caller and all registered work finish.

    There is no waiter task or timeout-based release: a still-live transport
    keeps its slot, even if its reaper cannot reclaim it. This fails closed
    rather than admitting an unbounded succession of abandoned loads.
    """

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._tasks: set[asyncio.Task[Any]] = set()
        self._finished = False
        _active_lifetimes.add(self)

    def track(self, task: asyncio.Task[Any]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()
        self._maybe_release()

    def pending_tasks(self) -> set[asyncio.Task[Any]]:
        return {
            task
            for task in self._tasks
            if task is not asyncio.current_task() and not task.done()
        }

    def finish(self) -> None:
        self._finished = True
        self._maybe_release()

    def _maybe_release(self) -> None:
        if self._finished and not self._tasks:
            self._finished = False
            _active_lifetimes.discard(self)
            self._release()


current_load_lifetime: ContextVar[MCPLoadLifetime | None] = ContextVar(
    "mcp_load_lifetime", default=None
)


def track_mcp_load_task(task: asyncio.Task[Any]) -> None:
    lifetime = current_load_lifetime.get()
    if lifetime is not None:
        lifetime.track(task)
