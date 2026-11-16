"""Cancellation-proof await for blocking container work."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


async def await_shielded(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coro`` to completion even under repeated cancellation.

    For a Docker call that runs under a per-name lock and may outlive its
    awaiter: ``asyncio.to_thread`` cancellation stops waiting, it does not stop
    the thread. The lock's ``finally`` releases as soon as a ``CancelledError``
    propagates out of the ``async with`` body, so a bare
    ``await asyncio.to_thread(container.stop)`` frees same-name exclusion while
    the daemon call is still running. A same-name task that then acquires the
    lock can reuse a container that the late stop (or remove) then terminates,
    or observe a transient "already exists" error.

    Mirrors the shield-loop in
    ``TaskTurnOrchestrator.begin_turn`` (web/services/task_orchestrator.py):
    wrap the coroutine in its own task and await it under ``asyncio.shield`` in
    a loop, so any number of cancellations delivered to *this* await is
    absorbed without cancelling the underlying task. Once the task settles, the
    original cancellation is re-raised regardless of how the task itself
    settled (the task's own exception, if any, is logged but not raised) so the
    caller's cancellation is still honored; if no cancellation occurred, the
    task's result or exception is returned/propagated unchanged.
    """
    task = asyncio.ensure_future(coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task_error = task.exception()
            if task_error is not None:
                logger.error(
                    "Container operation failed while a cancelled caller was"
                    " waiting for it to settle",
                    exc_info=(
                        type(task_error),
                        task_error,
                        task_error.__traceback__,
                    ),
                )
        raise
