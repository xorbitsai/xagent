"""Input that arrives while an execution is running.

An execution owner may offer a user message to a running execution. The
message waits in a mailbox until the execution reaches a safe point, where the
owner-supplied admission decides whether the message may enter the context.
Messages that did not reach a safe point are handed back to their admission
when the execution ends, or earlier, when the execution moves on to a pattern
that does not take live input.

The mailbox is deliberately synchronous: callers and the execution loop share
one event loop, so "check closed, then append" and "close, then take the
remainder" cannot interleave.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class LiveInputClaim(str, Enum):
    """What an admission reports after trying to claim a message."""

    CLAIMED = "claimed"  # The caller recorded the claim; the message may be injected.
    TAKEN = "taken"  # The turn id belongs to another delivery; do not inject.
    UNAVAILABLE = "unavailable"  # Not claimed; nothing durable was left behind.
    UNKNOWN = "unknown"  # The claim outcome cannot be established.


class LiveInputResult(str, Enum):
    """The final outcome reported to an admission, exactly once per message."""

    APPLIED = "applied"
    RETURNED = "returned"
    TAKEN = "taken"
    REFUSED = "refused"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class LiveInputAdmission(Protocol):
    """Owner-supplied hooks that decide and record what happens to a message."""

    async def claim(self) -> LiveInputClaim: ...

    async def finish(self, result: LiveInputResult) -> None: ...


@dataclass(frozen=True)
class LiveUserInput:
    """One message offered to a running execution."""

    turn_id: str
    execution_message: str
    display_message: str
    admission: LiveInputAdmission


class LiveInputOffer(str, Enum):
    """The synchronous answer to offering a message to a running execution."""

    QUEUED = "queued"
    ALREADY_QUEUED = "already_queued"
    ALREADY_ACCEPTED = "already_accepted"
    CONFLICT = "conflict"
    NO_LIVE_RUN = "no_live_run"
    UNSUPPORTED = "unsupported"


class LiveInputMailbox:
    """Ordered, closable queue of messages waiting for a safe point.

    A message stays known by its turn id from the moment it is queued until
    the execution has settled it: first while it waits, then, once taken,
    while it is being admitted. An offer of a turn id the mailbox still knows
    is answered from that entry instead of being queued again.
    """

    def __init__(self) -> None:
        self._items: deque[LiveUserInput] = deque()
        # Taken messages that have not been settled yet, by turn id.
        self._taken: dict[str, LiveUserInput] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def offer(self, item: LiveUserInput) -> LiveInputOffer:
        if self._closed:
            return LiveInputOffer.NO_LIVE_RUN
        known = self._taken.get(item.turn_id)
        if known is None:
            known = next(
                (queued for queued in self._items if queued.turn_id == item.turn_id),
                None,
            )
        if known is not None:
            if known.execution_message == item.execution_message:
                return LiveInputOffer.ALREADY_QUEUED
            return LiveInputOffer.CONFLICT
        self._items.append(item)
        return LiveInputOffer.QUEUED

    def take_next(self) -> LiveUserInput | None:
        if not self._items:
            return None
        item = self._items.popleft()
        self._taken[item.turn_id] = item
        return item

    def settle(self, item: LiveUserInput) -> None:
        """Forget a taken ``item`` once the execution is done admitting it."""
        if self._taken.get(item.turn_id) is item:
            del self._taken[item.turn_id]

    def pending(self) -> bool:
        return bool(self._items)

    def close(self) -> list[LiveUserInput]:
        self._closed = True
        remaining = list(self._items)
        self._items.clear()
        return remaining
