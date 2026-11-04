"""Bounds for tests that wait on real database-backed progress.

A wait whose condition is reached only after a real SQLite/PostgreSQL commit
is a hang detector, not a latency assertion. CI runners occasionally stall a
single fsync for seconds, so a 1-3s bound fails healthy code. Await the
observed condition and bound it with ``DB_PROGRESS_TIMEOUT``; a passing run
still returns as soon as the condition holds.

Keep short windows only for negative probes ("X has not happened yet"), where
the window is the assertion itself and a longer one would only slow the test.
"""

from __future__ import annotations

import asyncio
from typing import Callable

DB_PROGRESS_TIMEOUT = 30.0


async def eventually(
    predicate: Callable[[], object],
    *,
    timeout: float = DB_PROGRESS_TIMEOUT,
    interval: float = 0.01,
) -> None:
    """Poll ``predicate`` on the event loop until it is truthy."""

    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(interval)
