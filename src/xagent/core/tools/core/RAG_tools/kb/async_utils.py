"""Async helpers shared by the KB compatibility facades."""

from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Await ``value`` when the store returned a coroutine, pass it through otherwise."""
    if inspect.isawaitable(value):
        return await value
    return value
