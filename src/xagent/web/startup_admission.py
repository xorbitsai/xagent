"""Host-provided gates that must pass before XAgent admits runtime work."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import Protocol

from fastapi import FastAPI

_CALLBACKS_STATE_KEY = "_host_startup_admission_callbacks"


class HostStartupAdmissionCallback(Protocol):
    """Host check that returns on admission and raises on rejection.

    Either ``async def`` (runs on the event loop: it must only await) or a
    plain ``def`` (runs in a worker thread, so blocking database or file work
    never stalls the loop that answers lifecycle requests during a standby
    activation).
    """

    def __call__(self) -> Awaitable[None] | None: ...


def register_host_startup_admission(
    app: FastAPI,
    callback: HostStartupAdmissionCallback,
) -> None:
    """Register a callback to run before this app starts runtime work.

    Hosts must register callbacks before application startup. Callbacks run
    sequentially in registration order on every application lifespan. Any
    exception is terminal for that startup and is propagated unchanged.

    The immutable tuple is stored on the application instance rather than in
    module state, so separate app instances and tests cannot leak callbacks
    into one another.
    """
    callbacks = _get_callbacks(app)
    setattr(app.state, _CALLBACKS_STATE_KEY, (*callbacks, callback))


async def run_host_startup_admissions(app: FastAPI) -> None:
    """Run a stable snapshot of this app's registered admission callbacks."""
    for callback in _get_callbacks(app):
        if _is_async_callback(callback):
            result = callback()
        else:
            from .services.writer_workers import run_tracked_to_completion

            result = await run_tracked_to_completion(callback)
        if inspect.isawaitable(result):
            await result


def _is_async_callback(callback: HostStartupAdmissionCallback) -> bool:
    return asyncio.iscoroutinefunction(callback) or asyncio.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


def _get_callbacks(app: FastAPI) -> tuple[HostStartupAdmissionCallback, ...]:
    callbacks = getattr(app.state, _CALLBACKS_STATE_KEY, ())
    return tuple(callbacks)


__all__ = [
    "HostStartupAdmissionCallback",
    "register_host_startup_admission",
    "run_host_startup_admissions",
]
