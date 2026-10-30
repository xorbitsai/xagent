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

A pooled host (one web process plus shared-execution workers, each with its
own interpreter and so its own hold) needs two more answers from every worker:

- **Will this worker start any further command attempt?**
  :meth:`ReleaseHold.seal` closes the hold *and* refuses every new command
  attempt -- controls included -- at the dispatcher entry
  (:meth:`ReleaseHold.try_begin_attempt`). It never cancels an attempt that
  was already counted; those settle and leave the ledger, and only a later
  status read can prove that they did.
- **Did everything that left the counts actually settle?** Counts reach zero
  both when an owner released its task lease and when it gave up and left
  the lease to TTL recovery. :meth:`ReleaseHold.record_unsettled` marks the
  second case: ``unsettled_total`` only grows for the life of the process
  (``reopen`` does not clear it), with a bounded list of the latest entries
  for diagnostics. A host must treat ``unsettled_total > 0`` as "cannot prove
  settled"; this module never decides that for it, and ``busy`` keeps its
  meaning.

Nothing here cancels, times out or queues work. Default state is open with an
empty ledger, so a process that never closes the hold behaves exactly as
before -- the ledger only adds counting around existing code paths.

Constraints:

- Event-loop thread only. Every caller runs on the loop that owns the tracked
  executions (async handlers, ASGI middleware, dispatcher workers).
- One process. Each interpreter has its own hold; a pooled host must collect
  every process's ``execution_status`` and treat a missing report as busy,
  never read one process as the whole pool.
- Stdlib only at import time: ``task_execution`` imports this module, so it
  must stay a leaf to avoid import cycles.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import Counter, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# How many unsettled entries are kept for diagnostics. ``unsettled_total`` keeps
# counting past it; dropping old entries never lowers the blocking count.
UNSETTLED_DIAGNOSTICS_LIMIT = 50


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
        self._sealed = False
        # Sticky for the life of the process: the blocking evidence.
        self._unsettled_total = 0
        self._unsettled: deque[dict[str, str]] = deque(
            maxlen=UNSETTLED_DIAGNOSTICS_LIMIT
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def unsettled_total(self) -> int:
        return self._unsettled_total

    def close(self, reason: str) -> None:
        """Refuse new execution from now on; idempotent."""
        if not self._closed:
            self._refused.clear()
            logger.info("Execution admission held (%s)", reason)
        self._closed = True
        self._reason = reason

    def seal(self, reason: str) -> None:
        """Close, and refuse every new command attempt from now on; idempotent.

        Synchronous on the event-loop thread, so an attempt either registered
        before this call (and stays counted until it returns) or is refused by
        :meth:`try_begin_attempt` after it. Only :meth:`reopen` lifts a seal.

        Pending commands stay in the database for the successor: a task owner
        on this process no longer keeps its lease for them and releases on its
        next idle check (``TaskCoordinator._release_if_idle``).
        """
        self.close(reason)
        if not self._sealed:
            logger.info("Command attempts sealed (%s)", reason)
        self._sealed = True

    def reopen(self) -> None:
        """Admit new execution again on this same process; idempotent.

        Lifts a seal too. ``unsettled_total`` is deliberately kept: reopening
        proves nothing about work that was left to recovery.
        """
        if self._closed:
            logger.info(
                "Execution admission reopened (%s); refused while held: %s",
                self._reason,
                dict(self._refused),
            )
        self._closed = False
        self._sealed = False
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

    def try_begin_attempt(self, kind: str, identity: str = "") -> HoldToken | None:
        """Register one command attempt, or ``None`` once sealed.

        Unlike :meth:`try_admit` this passes while merely closed: a closed
        worker still applies controls and input for active tasks. The token
        covers the whole attempt -- candidate query, ownership wait, apply --
        so a status read during any of it reports busy. Refusals are counted
        but not logged: idle dispatcher workers poll every half second.
        """
        if self._sealed:
            self._refused[kind] += 1
            return None
        return self._register(kind, identity)

    def record_unsettled(self, kind: str, identity: str, reason: str) -> None:
        """Note responsibility that ended here without proof that it settled.

        For ownership left to TTL recovery: an unconfirmed lease release, a
        claim left by an interrupted command, a settlement that could not
        commit. Not a failure report and not a recovery -- it only says this
        process cannot vouch that the work settled. Never called on a
        proven-successful path.
        """
        self._unsettled_total += 1
        self._unsettled.append({"kind": kind, "identity": identity, "reason": reason})
        logger.warning(
            "Unsettled %s %s left for recovery (%s); unsettled_total=%d",
            kind,
            identity,
            reason,
            self._unsettled_total,
        )

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
            "sealed": self._sealed,
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
            "unsettled_total": self._unsettled_total,
            "unsettled": list(self._unsettled),
        }


release_hold = ReleaseHold()


def execution_status() -> dict[str, Any]:
    """What a host needs to tell whether core execution here has settled.

    ``busy`` is true while any of these is non-zero: a hand-off responsibility
    in the ledger, a live execution handle, a resume coordinator or its
    pre-registration reservation, a dispatcher slot reservation, a detached
    prompt dispatch, or a shared-execution task coordinator on this loop
    (always zero in combined mode). Each component is reported so a waiting
    release can log *why* it waits.

    ``sealed`` and ``unsettled_total`` (with the bounded ``unsettled`` list)
    are reported beside ``busy`` and never change it; a pooled host blocks on
    ``unsettled_total > 0`` itself.
    """
    from .task_command_transport import prompt_dispatch_count
    from .task_coordinator_runtime import live_coordinator_count
    from .task_execution import background_task_manager

    ledger = release_hold.snapshot()
    counts = {
        **background_task_manager.release_counts(),
        "prompt_dispatches": prompt_dispatch_count(),
        "responsibilities": len(ledger["responsibilities"]),
        "task_coordinators": live_coordinator_count(),
    }
    return {
        "busy": any(counts.values()),
        "counts": counts,
        "hold": ledger,
        "sealed": ledger["sealed"],
        "unsettled_total": ledger["unsettled_total"],
        "unsettled": ledger["unsettled"],
    }
