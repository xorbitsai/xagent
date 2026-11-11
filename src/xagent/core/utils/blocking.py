"""Run blocking work off the event loop through one replaceable seam.

``asyncio.to_thread`` hands work to a thread that a cancelled awaiter cannot
stop. A host that has to know when such threads are really done (the web
layer's ``retire_writers``) cannot see calls made from ``xagent.core``, which
must not import the web package. Core code that starts blocking work which can
write shared state therefore calls :func:`run_blocking`, and the host installs
a tracker with :func:`set_blocking_tracker`.

Without a tracker (any use of core outside the web server) this behaves like
``asyncio.to_thread``: the work runs in the loop's default executor under a
copy of the caller's context variables.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

# A tracker receives the already context-bound, argument-free call and returns
# its result; it decides whether and how the thread is counted.
BlockingTracker = Callable[[Callable[[], Any]], Awaitable[Any]]

_tracker: BlockingTracker | None = None


def set_blocking_tracker(tracker: BlockingTracker | None) -> BlockingTracker | None:
    """Install ``tracker`` (or ``None`` to clear it); returns the previous one.

    Installing the same tracker again is a no-op, so a host may call this on
    every import or startup.
    """
    global _tracker
    previous, _tracker = _tracker, tracker
    return previous


def get_blocking_tracker() -> BlockingTracker | None:
    return _tracker


async def run_in_default_executor(call: Callable[[], T]) -> T:
    """Run an argument-free ``call`` in the loop's default executor."""
    return await asyncio.get_running_loop().run_in_executor(None, call)


async def run_blocking(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """``asyncio.to_thread``, routed through the installed tracker if any."""
    call = functools.partial(
        contextvars.copy_context().run, functools.partial(fn, *args, **kwargs)
    )
    tracker = _tracker
    if tracker is None:
        return await run_in_default_executor(call)
    result: T = await tracker(call)
    return result
