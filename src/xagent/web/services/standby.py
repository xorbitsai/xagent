"""Standby start mode: a warmed process that touches no shared state until activated.

A planned release can start the candidate process next to the serving one. In
standby it loads and warms everything that only reads (imports, connection
pools, skill and template catalogs), but it must not write the database, claim
commands, recover leases or run any singleton loop while the old process still
serves. ``XAGENT_STANDBY`` selects the mode; unset keeps today's behavior.

Two pieces live here and nowhere else, so every host asks the same questions:

- :func:`standby_requested` -- was this process asked to start in standby?
- :data:`standby` -- the controller. Startup code registers each *deferred*
  step (a write, a claim path, a singleton) with :meth:`StandbyController.defer`
  instead of running it; :meth:`StandbyController.activate` runs them in
  registration order, so activation reproduces the order of a normal start.

The module imports only ``config`` (itself a leaf) because ``release_hold``
imports it at import time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum

from ...config import get_standby_enabled

logger = logging.getLogger(__name__)


def standby_requested() -> bool:
    """True when ``XAGENT_STANDBY`` asks for standby; a bad value is a startup error."""
    return get_standby_enabled()


class StandbyPhase(str, Enum):
    NORMAL = "normal"  # standby was not requested; nothing is deferred
    STANDBY = "standby"  # warmed, no writes, no singletons
    ACTIVATING = "activating"
    ACTIVE = "active"  # same state as a normal start
    FAILED = "failed"  # activation raised; the process must not serve


class StandbyActivationError(RuntimeError):
    """Activation failed or was refused; the process stays closed."""


DeferredStep = Callable[[], Awaitable[None]]
FailureHook = Callable[[], Awaitable[None]]


class StandbyController:
    """Ordered ledger of the startup steps a standby process postponed."""

    def __init__(self) -> None:
        self._phase = StandbyPhase.NORMAL
        self._steps: list[tuple[str, DeferredStep]] = []
        self._failure_hooks: list[FailureHook] = []
        self._activated_at: float | None = None
        self._lock: asyncio.Lock | None = None

    @property
    def phase(self) -> StandbyPhase:
        return self._phase

    @property
    def standby(self) -> bool:
        """True until activation completes; the guard startup code branches on."""
        return self._phase in (
            StandbyPhase.STANDBY,
            StandbyPhase.ACTIVATING,
            StandbyPhase.FAILED,
        )

    @property
    def activated_at(self) -> float | None:
        """Wall-clock time (``time.time()``) activation began; ``None`` before it."""
        return self._activated_at

    def begin(self) -> None:
        """Enter standby at the start of startup; idempotent while in standby."""
        if self._phase is StandbyPhase.NORMAL:
            self._phase = StandbyPhase.STANDBY
            self._steps.clear()
            self._failure_hooks = []

    def defer(self, name: str, step: DeferredStep) -> None:
        """Postpone ``step`` to activation. Only valid while in standby."""
        if self._phase is not StandbyPhase.STANDBY:
            raise StandbyActivationError(
                f"cannot defer {name!r} while the phase is {self._phase.value}"
            )
        self._steps.append((name, step))

    @property
    def deferred(self) -> list[str]:
        return [name for name, _ in self._steps]

    def on_failure(self, callback: FailureHook) -> None:
        """Register what must run when activation fails after steps already ran.

        A step such as "open admission" has side effects the failure leaves in
        place (an open hold, started loops); the host undoes them here. Hooks
        run in registration order, each guarded, after the phase is ``FAILED``.
        """
        self._failure_hooks.append(callback)

    async def activate(self) -> None:
        """Run every deferred step in order; idempotent once active.

        A failure leaves the phase ``FAILED`` and runs the failure hooks, so a
        half-activated process never serves. A later call after a failure is
        refused rather than retried, because steps that already ran (a
        migration check, a started loop) are not safe to run twice.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._phase is StandbyPhase.ACTIVE:
                return
            if self._phase is not StandbyPhase.STANDBY:
                raise StandbyActivationError(
                    f"cannot activate while the phase is {self._phase.value}"
                )
            self._phase = StandbyPhase.ACTIVATING
            # Taken before the first step: singletons started by a step
            # (schedule cutoffs, "process started at" bounds) must see the
            # activation instant, never the earlier import time.
            self._activated_at = time.time()
            steps = list(self._steps)
            hooks = self._failure_hooks
            try:
                for name, step in steps:
                    logger.info("Standby activation: %s", name)
                    await step()
            except BaseException:
                if self._phase is StandbyPhase.ACTIVATING:
                    self._activated_at = None
                    self._phase = StandbyPhase.FAILED
                logger.exception("Standby activation failed; staying closed")
                for hook in list(hooks):
                    try:
                        await hook()
                    except Exception:
                        logger.exception("Standby failure hook failed")
                raise
            # A shutdown that reset the ledger mid-activation owns the phase.
            if self._phase is StandbyPhase.ACTIVATING:
                self._steps.clear()
                self._phase = StandbyPhase.ACTIVE
                logger.info("Standby activation complete")

    def snapshot(self) -> dict[str, object]:
        return {
            "phase": self._phase.value,
            "deferred": self.deferred,
            "activated_at": self._activated_at,
        }

    def reset(self) -> None:
        """Back to a normal process; for tests and a fresh ASGI lifespan."""
        self._phase = StandbyPhase.NORMAL
        self._steps.clear()
        self._failure_hooks = []
        self._activated_at = None
        self._lock = None


standby = StandbyController()
