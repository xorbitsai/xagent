"""Retain asynchronous ownership cleanup until it finishes."""

import asyncio
from collections.abc import Coroutine
from typing import Any

_PENDING_CLOSES: set[asyncio.Task[None]] = set()


def finish_async_cleanup(cleanup: Coroutine[Any, Any, None]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(cleanup)
    else:
        task = loop.create_task(cleanup)
        _PENDING_CLOSES.add(task)
        task.add_done_callback(_PENDING_CLOSES.discard)
