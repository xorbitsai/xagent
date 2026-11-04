"""Per-key async mutex registry with safe entry eviction.

Sandbox lifecycle code needs one mutex per sandbox name, drawn from an
unbounded namespace (names such as ``ssh::{task_id}``), so entries must be
evicted once nobody holds or waits on them or the registry grows forever.
Doing that correctly is the whole difficulty: a naive "pop when released"
scheme lets a waiter that is already blocked on the popped lock run
concurrently with a newcomer that installed a fresh lock under the same key.

This module owns that primitive once. It lives in the sandbox package
because that is the lower of the two layers that need it -- the web layer
already imports from ``xagent.sandbox``, while the sandbox layer must never
import from ``xagent.web``.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class KeyedLockEntry:
    """One key's mutex plus the number of tasks holding or waiting on it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    waiters: int = 0


class KeyedLockRegistry:
    """A dict of per-key async mutexes that evicts unreferenced entries.

    Waiter bookkeeping is deliberately synchronous and unguarded: every step
    is a single dict get/set/pop or an int increment/decrement with no
    ``await`` in between, so nothing else can interleave on a single-threaded
    event loop, and a guarding lock would only add another await point for a
    cancellation to land on mid-rollback.

    Eviction is identity-checked against the entry the caller actually used,
    so a concurrent waiter that already installed a fresh entry for the same
    key is never evicted out from under it.
    """

    def __init__(self) -> None:
        self._entries: dict[str, KeyedLockEntry] = {}

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> KeyedLockEntry:
        return self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> Optional[KeyedLockEntry]:
        """Return ``key``'s entry, or ``None`` when no task holds or waits on it.

        Lets a caller assert that some task currently holds a key's lock.
        ``asyncio.Lock`` has no owner concept, so this can only show that the
        lock is held, never by whom.
        """
        return self._entries.get(key)

    @asynccontextmanager
    async def locked(self, key: str) -> AsyncIterator[None]:
        """Hold ``key``'s mutex for the duration of the body.

        If the acquire is cancelled, the waiter count is rolled back in the
        same ``except BaseException`` step, so a cancelled waiter leaks
        neither the count nor a now-unreferenced entry.
        """
        entry = self._entries.get(key)
        if entry is None:
            entry = KeyedLockEntry()
            self._entries[key] = entry
        entry.waiters += 1

        try:
            await entry.lock.acquire()
        except BaseException:
            entry.waiters -= 1
            self._drop_if_unused(key, entry)
            raise

        try:
            yield
        finally:
            entry.lock.release()
            entry.waiters -= 1
            self._drop_if_unused(key, entry)

    def _drop_if_unused(self, key: str, entry: KeyedLockEntry) -> None:
        """Evict ``key``'s entry once it has no holder and no waiter left.

        Called only from the synchronous bookkeeping steps in ``locked``,
        with no ``await`` between the waiter-count update and this call, so
        nothing else can interleave and observe an inconsistent count.
        """
        if entry.waiters > 0:
            return
        if self._entries.get(key) is entry:
            self._entries.pop(key, None)
