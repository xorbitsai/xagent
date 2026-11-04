"""Host-provided gates that must pass before XAgent admits runtime work."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI

_CALLBACKS_STATE_KEY = "_host_startup_admission_callbacks"


class HostStartupAdmissionCallback(Protocol):
    """Async host check that returns on admission and raises on rejection."""

    async def __call__(self) -> None: ...


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
        await callback()


def _get_callbacks(app: FastAPI) -> tuple[HostStartupAdmissionCallback, ...]:
    callbacks = getattr(app.state, _CALLBACKS_STATE_KEY, ())
    return tuple(callbacks)


__all__ = [
    "HostStartupAdmissionCallback",
    "register_host_startup_admission",
    "run_host_startup_admissions",
]
