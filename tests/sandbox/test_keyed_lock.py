"""
Unit tests for ``KeyedLockRegistry``, the shared per-key async mutex.

Tested directly rather than only through ``DockerSandboxService._named_lock``,
because the registry is now a reusable primitive with more than one intended
consumer: its invariants have to hold on their own terms, not just as observed
through one caller.

The invariant that matters is entry identity across release-and-requeue. A
registry that pops an entry the moment its holder releases lets a waiter
already blocked on that entry run concurrently with a newcomer that installed
a fresh one under the same key -- two tasks holding "the same" lock.
"""

from __future__ import annotations

import asyncio

import pytest

from xagent.sandbox.keyed_lock import KeyedLockEntry, KeyedLockRegistry


class TestKeyedLockMutualExclusion:
    @pytest.mark.asyncio
    async def test_second_caller_waits_for_the_first(self):
        registry = KeyedLockRegistry()
        order: list[str] = []
        first_holds = asyncio.Event()
        release_first = asyncio.Event()

        async def first() -> None:
            async with registry.locked("k"):
                order.append("first-in")
                first_holds.set()
                await release_first.wait()
                order.append("first-out")

        async def second() -> None:
            async with registry.locked("k"):
                order.append("second-in")

        task_first = asyncio.create_task(first())
        await first_holds.wait()
        task_second = asyncio.create_task(second())
        await asyncio.sleep(0)

        assert order == ["first-in"], "second must not enter while first holds"

        release_first.set()
        await asyncio.gather(task_first, task_second)
        assert order == ["first-in", "first-out", "second-in"]

    @pytest.mark.asyncio
    async def test_different_keys_do_not_block_each_other(self):
        registry = KeyedLockRegistry()
        a_holds = asyncio.Event()
        release_a = asyncio.Event()
        b_entered = asyncio.Event()

        async def hold_a() -> None:
            async with registry.locked("a"):
                a_holds.set()
                await release_a.wait()

        async def hold_b() -> None:
            async with registry.locked("b"):
                b_entered.set()

        task_a = asyncio.create_task(hold_a())
        await a_holds.wait()
        task_b = asyncio.create_task(hold_b())
        await asyncio.wait_for(b_entered.wait(), timeout=1)

        release_a.set()
        await asyncio.gather(task_a, task_b)

    @pytest.mark.asyncio
    async def test_stays_exclusive_across_release_and_requeue(self):
        """The entry a waiter is queued on must survive its holder's release."""
        registry = KeyedLockRegistry()
        concurrent = 0
        max_concurrent = 0

        entered_first = asyncio.Event()
        release_first = asyncio.Event()
        entered_second = asyncio.Event()
        release_second = asyncio.Event()

        async def holder(entered: asyncio.Event, release: asyncio.Event) -> None:
            nonlocal concurrent, max_concurrent
            async with registry.locked("k"):
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
                entered.set()
                await release.wait()
                concurrent -= 1

        task_first = asyncio.create_task(holder(entered_first, release_first))
        await entered_first.wait()

        task_second = asyncio.create_task(holder(entered_second, release_second))
        await asyncio.sleep(0)
        entry_while_first_holds = registry["k"]
        assert entry_while_first_holds.waiters == 2

        release_first.set()
        await task_first
        await entered_second.wait()
        assert registry["k"] is entry_while_first_holds, (
            "the waiting task must end up holding the same entry it queued on"
        )

        release_second.set()
        await task_second
        assert max_concurrent == 1


class TestKeyedLockEviction:
    @pytest.mark.asyncio
    async def test_entry_is_evicted_once_unused(self):
        registry = KeyedLockRegistry()
        async with registry.locked("k"):
            assert "k" in registry
        assert "k" not in registry
        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_entry_is_retained_while_a_waiter_is_queued(self):
        registry = KeyedLockRegistry()
        entered = asyncio.Event()
        release = asyncio.Event()
        waiter_entered = asyncio.Event()
        release_waiter = asyncio.Event()

        async def holder(e: asyncio.Event, r: asyncio.Event) -> None:
            async with registry.locked("k"):
                e.set()
                await r.wait()

        task_holder = asyncio.create_task(holder(entered, release))
        await entered.wait()
        task_waiter = asyncio.create_task(holder(waiter_entered, release_waiter))
        await asyncio.sleep(0)
        assert registry["k"].waiters == 2

        release.set()
        await task_holder
        assert "k" in registry, "the queued waiter still references this entry"

        release_waiter.set()
        await task_waiter
        assert "k" not in registry

    @pytest.mark.asyncio
    async def test_eviction_is_identity_checked(self):
        """A stale entry must never evict a fresh one installed under the key."""
        registry = KeyedLockRegistry()
        async with registry.locked("k"):
            stale = registry["k"]
        assert "k" not in registry

        # Install a fresh entry, then ask the registry to drop the stale one.
        async with registry.locked("k"):
            fresh = registry["k"]
            assert fresh is not stale
            registry._drop_if_unused("k", stale)
            assert registry["k"] is fresh, (
                "dropping a stale entry must not evict the live one"
            )

    @pytest.mark.asyncio
    async def test_body_raising_still_releases_and_evicts(self):
        registry = KeyedLockRegistry()
        with pytest.raises(ValueError):
            async with registry.locked("k"):
                raise ValueError("boom")
        assert "k" not in registry, "a raising body must not leak the entry"

        # The lock must be reusable, i.e. actually released.
        async with registry.locked("k"):
            pass


class TestKeyedLockCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_waiter_rolls_back_its_count_and_leaks_nothing(self):
        registry = KeyedLockRegistry()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with registry.locked("k"):
                entered.set()
                await release.wait()

        async def waiter() -> None:
            async with registry.locked("k"):
                raise AssertionError("a cancelled waiter must never enter the body")

        task_holder = asyncio.create_task(holder())
        await entered.wait()
        task_waiter = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert registry["k"].waiters == 2

        task_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_waiter
        assert registry["k"].waiters == 1

        release.set()
        await task_holder
        assert "k" not in registry

    @pytest.mark.asyncio
    async def test_cancelling_the_only_waiter_evicts_the_entry(self):
        """No holder to clean up after it, so the rollback must evict itself."""
        registry = KeyedLockRegistry()
        started = asyncio.Event()

        async def waiter() -> None:
            started.set()
            async with registry.locked("k"):
                await asyncio.sleep(3600)

        task = asyncio.create_task(waiter())
        await started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "k" not in registry


class TestKeyedLockRegistryMapping:
    def test_get_returns_none_for_an_unknown_key(self):
        assert KeyedLockRegistry().get("nope") is None

    @pytest.mark.asyncio
    async def test_get_returns_the_live_entry_while_held(self):
        registry = KeyedLockRegistry()
        async with registry.locked("k"):
            entry = registry.get("k")
            assert isinstance(entry, KeyedLockEntry)
            assert entry.lock.locked()
            assert entry.waiters == 1
