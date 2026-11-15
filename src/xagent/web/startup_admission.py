"""Host-provided gates that must pass before XAgent admits runtime work."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from fastapi import FastAPI

_CALLBACKS_STATE_KEY = "_host_startup_admission_callbacks"


class HostStartupAdmissionCallback(Protocol):
    """Host check that returns on admission and raises on rejection.

    Invoked on the event loop, then awaited if it returned an awaitable: an
    ``async def``, or a plain callable that builds a coroutine and may touch
    the running loop while doing so.
    """

    def __call__(self) -> Awaitable[None] | None: ...


class _ThreadedAdmission:
    """A plain blocking callback the host registered to run in a worker thread."""

    def __init__(self, callback: Callable[[], Any]) -> None:
        self.callback = callback


def register_host_startup_admission(
    app: FastAPI,
    callback: HostStartupAdmissionCallback | Callable[[], Any],
    *,
    run_in_thread: bool = False,
) -> None:
    """Register a callback to run before this app starts runtime work.

    Hosts must register callbacks before application startup. Callbacks run
    sequentially in registration order on every application lifespan. Any
    exception is terminal for that startup and is propagated unchanged.

    By default the callback is invoked on the event loop, exactly as it always
    was. A host whose check does blocking database or file work passes
    ``run_in_thread=True`` with a plain ``def`` that returns ``None``: it then
    runs in a tracked worker thread (a cancel waits for it), so it cannot stall
    the loop that answers lifecycle requests during a standby activation. It
    must not touch the event loop, and returning an awaitable is rejected.

    The immutable tuple is stored on the application instance rather than in
    module state, so separate app instances and tests cannot leak callbacks
    into one another.
    """
    entry: object = _ThreadedAdmission(callback) if run_in_thread else callback
    callbacks = _get_callbacks(app)
    setattr(app.state, _CALLBACKS_STATE_KEY, (*callbacks, entry))


async def run_host_startup_admissions(app: FastAPI) -> None:
    """Run a stable snapshot of this app's registered admission callbacks."""
    for callback in _get_callbacks(app):
        if isinstance(callback, _ThreadedAdmission):
            from .services.writer_workers import run_tracked_to_completion

            result = await run_tracked_to_completion(callback.callback)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError(
                    "a host startup admission registered with run_in_thread=True "
                    "must be a plain function that returns None, not an awaitable"
                )
            continue
        result = callback()
        if inspect.isawaitable(result):
            await result


def _get_callbacks(
    app: FastAPI,
) -> tuple[HostStartupAdmissionCallback | _ThreadedAdmission, ...]:
    callbacks = getattr(app.state, _CALLBACKS_STATE_KEY, ())
    return tuple(callbacks)


__all__ = [
    "HostStartupAdmissionCallback",
    "register_host_startup_admission",
    "run_host_startup_admissions",
]
