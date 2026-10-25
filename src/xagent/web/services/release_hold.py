"""Opt-in admission hold and responsibility ledger for a planned process restart.

A host that replaces this process on purpose (for example the SaaS thin
ingress Gate) needs two answers the execution code already knows but does not
publish:

- **May new execution start here right now?** ``ReleaseHold.closed`` is that
  switch. Entry points that would open a *new* execution call
  :meth:`ReleaseHold.admit` (or :meth:`ReleaseHold.try_admit`) before any side
  effect; while the hold is closed they are refused with :class:`ExecutionHeld`
  -- or, where the entry has a durable pending form, they leave that pending
  record for the next process instead. The command dispatcher reads the same
  switch through its cap (``DispatchCapSnapshot.held``), so durable commands
  that would open an execution stay ``pending`` while controls and input for
  already-active tasks keep flowing.
- **Has everything already accepted here settled?** Executions are owned by
  ``BackgroundTaskManager`` once registered. Between an entry accepting work
  and that registration there is a hand-off window (database claims, setup,
  scheduling) in which neither side would report the work. Each admitted or
  tracked *responsibility* stays in this ledger until the code that holds it
  returns, and that code returns only after its child is registered, so the
  responsibility moves without an instant where both counts are zero.
  :func:`execution_status` combines this ledger with the manager's own tables.

Nothing here cancels, times out or queues work. Default state is open with an
empty ledger, so a process that never closes the hold behaves exactly as
before -- the ledger only adds counting around existing code paths.

Constraints:

- Event-loop thread only. Every caller runs on the loop that owns the tracked
  executions (async handlers, ASGI middleware, dispatcher workers).
- One process. A pooled deployment would need every process to report; the
  host must refuse such a topology rather than read one process as idle.
- Stdlib only at import time: ``task_execution`` imports this module, so it
  must stay a leaf to avoid import cycles.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Responsibility:
    """One accepted unit of work this process still owes, by kind and identity."""

    kind: str
    identity: str
    started_at: float


class HoldToken:
    """A ledger entry's handle; releasing it is idempotent.

    Usable as a context manager so the entry leaves the ledger on every exit
    path, including cancellation and exceptions.
    """

    __slots__ = ("_hold", "_key")

    def __init__(self, hold: ReleaseHold, key: int) -> None:
        self._hold = hold
        self._key: int | None = key

    def release(self) -> None:
        key, self._key = self._key, None
        if key is not None:
            self._hold._responsibilities.pop(key, None)

    def __enter__(self) -> HoldToken:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class ReleaseHold:
    """Process-wide execution admission switch plus hand-off ledger."""

    def __init__(self) -> None:
        self._closed = False
        self._reason: str | None = None
        self._responsibilities: dict[int, Responsibility] = {}
        self._keys = itertools.count()
        # Refusals since the hold last closed, by entry kind; diagnostics only.
        self._refused: Counter[str] = Counter()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self, reason: str) -> None:
        """Refuse new execution from now on; idempotent."""
        if not self._closed:
            self._refused.clear()
            logger.info("Execution admission held (%s)", reason)
        self._closed = True
        self._reason = reason

    def reopen(self) -> None:
        """Admit new execution again on this same process; idempotent."""
        if self._closed:
            logger.info(
                "Execution admission reopened (%s); refused while held: %s",
                self._reason,
                dict(self._refused),
            )
        self._closed = False
        self._reason = None

    def _register(self, kind: str, identity: str) -> HoldToken:
        key = next(self._keys)
        self._responsibilities[key] = Responsibility(
            kind=kind, identity=identity, started_at=time.monotonic()
        )
        return HoldToken(self, key)

    def try_admit(self, kind: str, identity: str = "") -> HoldToken | None:
        """Register new execution responsibility, or ``None`` while held.

        The check and the registration happen in one synchronous step, so a
        hold that closes concurrently either sees this entry in the ledger or
        this call sees the hold -- never neither.
        """
        if self._closed:
            self._refused[kind] += 1
            logger.info(
                "Refused new %s execution %s: admission held (%s)",
                kind,
                identity,
                self._reason,
            )
            return None
        return self._register(kind, identity)

    @contextmanager
    def admit(self, kind: str, identity: str = "") -> Iterator[None]:
        """Hold new-execution responsibility for the block, or raise.

        Raises :class:`ExecutionHeld` (a retryable "not accepted") before the
        block runs, so a refused caller has had no side effect.
        """
        token = self.try_admit(kind, identity)
        if token is None:
            from .task_execution_admission import ExecutionHeld

            raise ExecutionHeld(kind)
        with token:
            yield

    def track(self, kind: str, identity: str = "") -> HoldToken:
        """Register responsibility that is allowed even while held.

        For work that does not open a new execution but must still finish
        before the process stops: a dispatch attempt that may apply input to
        an active task, an in-progress trigger tick, a control's outer task.
        """
        return self._register(kind, identity)

    def adopt(self, kind: str, identity: str, task: asyncio.Future[Any]) -> None:
        """Count ``task`` until it finishes in any way (outer tasks)."""
        token = self.track(kind, identity)
        task.add_done_callback(lambda _done: token.release())

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "closed": self._closed,
            "reason": self._reason,
            "refused": dict(self._refused),
            "responsibilities": [
                {
                    "kind": item.kind,
                    "identity": item.identity,
                    "age_seconds": round(now - item.started_at, 3),
                }
                for item in self._responsibilities.values()
            ],
        }


release_hold = ReleaseHold()


def execution_status() -> dict[str, Any]:
    """What a host needs to tell whether core execution here has settled.

    ``busy`` is true while any of these is non-zero: a hand-off responsibility
    in the ledger, a live execution handle, a resume coordinator or its
    pre-registration reservation, a dispatcher slot reservation, or a detached
    prompt dispatch. Each component is reported so a waiting release can log
    *why* it waits.
    """
    from .task_command_transport import prompt_dispatch_count
    from .task_execution import background_task_manager

    ledger = release_hold.snapshot()
    counts = {
        **background_task_manager.release_counts(),
        "prompt_dispatches": prompt_dispatch_count(),
        "responsibilities": len(ledger["responsibilities"]),
    }
    return {
        "busy": any(counts.values()),
        "counts": counts,
        "hold": ledger,
    }
