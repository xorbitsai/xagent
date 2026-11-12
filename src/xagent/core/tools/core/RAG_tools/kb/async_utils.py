"""Async helpers shared by the KB compatibility facades."""

from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Await ``value`` when the store returned a coroutine, pass it through otherwise."""
    if inspect.isawaitable(value):
        return await value
    return value


def has_async_method(store: object, name: str) -> bool:
    """Return True only for a real coroutine method.

    A ``MagicMock`` answers every attribute with a callable, so a plain
    ``callable(getattr(...))`` check would make the facades await a non-awaitable
    mock value. Real duck-typed stores expose ``async def`` methods.
    """
    method = getattr(store, name, None)
    return callable(method) and inspect.iscoroutinefunction(method)
