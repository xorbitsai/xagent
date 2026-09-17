"""
Unit tests for DockerSandboxService's per-name lifecycle lock (_named_lock)
and control-object construction (_get_live_control). Pure asyncio tests
against a minimal fake Docker client; no Docker daemon required.
"""

from __future__ import annotations

import ast
import asyncio
import threading
import time
from pathlib import Path

import pytest

import xagent.sandbox.docker_sandbox as docker_sandbox_module
from xagent.sandbox.docker_sandbox import (
    LABEL_SANDBOX_NAME,
    DockerSandboxService,
    MemDockerStore,
    _SandboxControl,
)


class _FakeContainerCollection:
    """Minimal Docker container collection stub: always reports no containers."""

    def list(self, *args, **kwargs):
        return []


class _FakeDockerClient:
    """Minimal Docker client stub sufficient for service construction and delete()."""

    def __init__(self) -> None:
        self.containers = _FakeContainerCollection()

    def ping(self):
        return True


def _make_service() -> DockerSandboxService:
    return DockerSandboxService(
        MemDockerStore(), namespace="test", client=_FakeDockerClient()
    )


class _StoppableContainer:
    """Container stub whose ``stop()`` can be held open from the test.

    ``stop()`` runs on a worker thread (both callers reach it through
    ``asyncio.to_thread``), so the occupancy counter is guarded by a real
    ``threading.Lock`` and the gate is a ``threading.Event``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.labels = {LABEL_SANDBOX_NAME: name}
        self.attrs: dict = {
            "Config": {"Image": "busybox:latest", "WorkingDir": "/home"},
            "HostConfig": {},
            "State": {"Status": "running"},
            "NetworkSettings": {"Networks": {"bridge": {}}},
            "Created": "2026-01-01T00:00:00Z",
        }
        self.stop_calls = 0
        self.start_calls = 0
        self.max_concurrent_stops = 0
        self.stop_entered = threading.Event()
        self.release_stop = threading.Event()
        self._occupancy = 0
        self._guard = threading.Lock()

    def reload(self):
        return None

    def start(self):
        self.start_calls += 1

    def stop(self):
        with self._guard:
            self.stop_calls += 1
            self._occupancy += 1
            self.max_concurrent_stops = max(self.max_concurrent_stops, self._occupancy)
        self.stop_entered.set()
        if not self.release_stop.wait(timeout=10):
            raise AssertionError("stop() gate was never released")
        with self._guard:
            self._occupancy -= 1


class _FakeClientWithContainer:
    """Docker client stub that always resolves one managed container."""

    def __init__(self, container: _StoppableContainer) -> None:
        self.containers = _SingleContainerCollection(container)

    def ping(self):
        return True


class _SingleContainerCollection:
    def __init__(self, container: _StoppableContainer) -> None:
        self._container = container

    def list(self, *args, **kwargs):
        return [self._container]


async def _await_flag(flag: threading.Event, timeout: float = 5.0) -> None:
    """Await a thread-set flag from the event loop without blocking it."""
    deadline = time.monotonic() + timeout
    while not flag.is_set():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the worker thread")
        await asyncio.sleep(0.01)


class TestHandleStopSharesTheServiceLifecycleLock:
    """Pin that ``DockerSandbox.stop()`` and the service's own lifecycle
    methods are mutually exclusive for one container name.

    ``stop()`` used to take only the sandbox's own ``exclusive_access``
    barrier, which drains in-flight ``operation()`` work and tracks no
    exclusive holder -- and neither ``container.stop()`` nor
    ``container.start()`` registers as an ``operation()``, so a stop could
    interleave with ``stop_existing``/``start_existing``/``delete``/
    ``create_snapshot`` for the same name.
    """

    @pytest.mark.asyncio
    async def test_handles_are_built_with_the_services_lock_registry(self):
        container = _StoppableContainer("wired")
        service = DockerSandboxService(
            MemDockerStore(),
            namespace="test",
            client=_FakeClientWithContainer(container),
        )

        from_start_existing = await service.start_existing("wired")
        from_get_or_create = await service.get_or_create("wired")

        # Same registry object, not a private copy: sharing the registry is
        # the whole mechanism behind the mutual exclusion below.
        assert from_start_existing._locks is service._locks
        assert from_get_or_create._locks is service._locks

    @pytest.mark.asyncio
    async def test_stop_and_stop_existing_cannot_overlap_for_one_name(self):
        name = "contended"
        container = _StoppableContainer(name)
        service = DockerSandboxService(
            MemDockerStore(),
            namespace="test",
            client=_FakeClientWithContainer(container),
        )
        handle = await service.start_existing(name)

        # The handle's stop() enters container.stop() and parks there.
        stop_task = asyncio.create_task(handle.stop())
        await _await_flag(container.stop_entered)
        assert service._locks[name].lock.locked()
        assert service._locks[name].waiters == 1

        # A concurrent lifecycle transition for the same name must queue on
        # the *same* lock entry rather than proceeding in parallel.
        stop_existing_task = asyncio.create_task(service.stop_existing(name))
        for _ in range(5):
            await asyncio.sleep(0)
        assert service._locks[name].waiters == 2, (
            "stop_existing() must queue on the same keyed lock entry stop() holds"
        )
        assert not stop_existing_task.done()
        assert container.stop_calls == 1, (
            "stop_existing() must not reach container.stop() while stop() holds the lock"
        )

        container.release_stop.set()
        await asyncio.wait_for(stop_task, timeout=5)
        await asyncio.wait_for(stop_existing_task, timeout=5)

        assert container.stop_calls == 2
        assert container.max_concurrent_stops == 1, (
            "the two stop paths overlapped inside the critical section"
        )
        assert name not in service._locks

    @pytest.mark.asyncio
    async def test_lock_holding_paths_do_not_self_deadlock(self):
        """Regression pin for the fix's own failure mode.

        The keyed lock is not reentrant, so a lock-holding path that reached
        ``DockerSandbox.stop()`` would deadlock on itself. ``stop_existing()``
        stops its container through the raw ``Container.stop`` API for exactly
        that reason; this asserts the real call completes rather than hanging.
        """
        name = "no-deadlock"
        container = _StoppableContainer(name)
        container.release_stop.set()
        service = DockerSandboxService(
            MemDockerStore(),
            namespace="test",
            client=_FakeClientWithContainer(container),
        )

        handle = await service.start_existing(name)
        await asyncio.wait_for(service.stop_existing(name), timeout=5)
        await asyncio.wait_for(handle.stop(), timeout=5)

        assert container.stop_calls == 2
        assert name not in service._locks


class TestNamedLockIdentityAndMutualExclusion:
    """Pin the split-brain fix: a waiter queued behind a holder must land on
    the same lock entry the holder used, not a freshly-constructed one."""

    @pytest.mark.asyncio
    async def test_named_lock_stays_mutually_exclusive_across_release_and_requeue(
        self,
    ):
        service = _make_service()
        concurrent_holders = 0
        max_concurrent = 0

        entered_b = asyncio.Event()
        release_b = asyncio.Event()
        entered_a = asyncio.Event()
        release_a = asyncio.Event()
        entered_c = asyncio.Event()
        release_c = asyncio.Event()

        async def holder(entered: asyncio.Event, release: asyncio.Event) -> None:
            nonlocal concurrent_holders, max_concurrent
            async with service._named_lock("shared-name"):
                concurrent_holders += 1
                max_concurrent = max(max_concurrent, concurrent_holders)
                entered.set()
                await release.wait()
                concurrent_holders -= 1

        # B takes the lock first.
        task_b = asyncio.create_task(holder(entered_b, release_b))
        await entered_b.wait()

        # A queues behind B while B still holds it.
        task_a = asyncio.create_task(holder(entered_a, release_a))
        await asyncio.sleep(0)
        entry_while_b_holds = service._locks["shared-name"]
        assert entry_while_b_holds.waiters == 2

        # B finishes; the entry must survive (A is still waiting on it) so
        # that A ends up acquiring the SAME entry rather than racing a
        # newly-constructed one against a delete()/other holder.
        release_b.set()
        await task_b
        await entered_a.wait()
        assert service._locks["shared-name"] is entry_while_b_holds

        # A now holds it; C attempts to acquire concurrently and must queue
        # behind A rather than proceeding on an independent lock instance.
        task_c = asyncio.create_task(holder(entered_c, release_c))
        await asyncio.sleep(0)
        assert concurrent_holders == 1
        assert not entered_c.is_set()

        release_a.set()
        await task_a
        await entered_c.wait()

        release_c.set()
        await task_c

        assert max_concurrent == 1
        assert "shared-name" not in service._locks


class TestNamedLockWaiterRecycling:
    """Pin the entry-recycling contract: only evict when unused."""

    @pytest.mark.asyncio
    async def test_entry_recycled_when_no_waiters_remain(self):
        service = _make_service()
        async with service._named_lock("solo"):
            assert "solo" in service._locks
        assert "solo" not in service._locks

    @pytest.mark.asyncio
    async def test_entry_retained_while_a_waiter_is_pending(self):
        service = _make_service()
        entered_holder = asyncio.Event()
        release_holder = asyncio.Event()
        entered_waiter = asyncio.Event()
        release_waiter = asyncio.Event()

        async def holder(entered: asyncio.Event, release: asyncio.Event) -> None:
            async with service._named_lock("busy"):
                entered.set()
                await release.wait()

        task_holder = asyncio.create_task(holder(entered_holder, release_holder))
        await entered_holder.wait()

        task_waiter = asyncio.create_task(holder(entered_waiter, release_waiter))
        await asyncio.sleep(0)
        assert service._locks["busy"].waiters == 2

        # Releasing the holder must not drop the entry: the waiter is still
        # queued on it.
        release_holder.set()
        await task_holder
        assert "busy" in service._locks

        release_waiter.set()
        await task_waiter
        assert "busy" not in service._locks


class TestNamedLockCancellationSafety:
    """Pin that a cancelled waiter rolls back its waiter count and leaks nothing."""

    @pytest.mark.asyncio
    async def test_cancel_while_waiting_rolls_back_waiter_count_and_entry(self):
        service = _make_service()
        entered_holder = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with service._named_lock("cancel-me"):
                entered_holder.set()
                await release_holder.wait()

        task_holder = asyncio.create_task(holder())
        await entered_holder.wait()

        async def waiter() -> None:
            async with service._named_lock("cancel-me"):
                raise AssertionError("cancelled waiter must never enter the body")

        task_waiter = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert service._locks["cancel-me"].waiters == 2

        task_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_waiter

        # Cancellation rolled the waiter count back; only the holder remains.
        assert service._locks["cancel-me"].waiters == 1

        release_holder.set()
        await task_holder

        # Once the holder also finishes, the entry must not have leaked.
        assert "cancel-me" not in service._locks

    @pytest.mark.asyncio
    async def test_repeated_cancel_of_a_queued_waiter_rolls_back_cleanly(self):
        # Cancelling a queued waiter must roll back its waiter count and, once
        # the holder leaves, evict the entry. Calling cancel() a second time
        # before the task is resumed must not double-decrement the count.
        #
        # Note what this can and cannot reach: the rollback in _named_lock's
        # `except BaseException` branch contains no `await`, so a second
        # cancellation cannot interleave *inside* it -- asyncio delivers at
        # most one CancelledError to a task that has not resumed yet, and the
        # synchronous rollback is indivisible on this event loop. The case of
        # a cancellation arriving while an in-flight operation still holds the
        # lock is a different mechanism (_await_shielded) and is covered in
        # test_docker_lifecycle_api.py's repeated-cancellation test.
        service = _make_service()
        entered_holder = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder() -> None:
            async with service._named_lock("double-cancel"):
                entered_holder.set()
                await release_holder.wait()

        task_holder = asyncio.create_task(holder())
        await entered_holder.wait()

        async def waiter() -> None:
            async with service._named_lock("double-cancel"):
                raise AssertionError("cancelled waiter must never enter the body")

        task_waiter = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert service._locks["double-cancel"].waiters == 2

        task_waiter.cancel()
        task_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_waiter

        assert service._locks["double-cancel"].waiters == 1

        release_holder.set()
        await task_holder

        assert "double-cancel" not in service._locks


class TestGetLiveControl:
    """Pin _get_live_control as the deleted-aware construction point.

    _get_live_control() asserts that _named_lock(name) is held, so every
    call below happens inside that context, matching its real call sites.
    """

    @pytest.mark.asyncio
    async def test_reuses_the_same_live_control_instance(self):
        service = _make_service()
        async with service._named_lock("box"):
            first = service._get_live_control("box")
            second = service._get_live_control("box")
        assert second is first

    @pytest.mark.asyncio
    async def test_replaces_a_deleted_control_with_a_fresh_instance(self):
        service = _make_service()
        async with service._named_lock("box"):
            first = service._get_live_control("box")
            first.deleted = True

            replaced = service._get_live_control("box")

        assert replaced is not first
        assert replaced.deleted is False
        assert service._controls["box"] is replaced

    def test_asserts_when_called_without_holding_the_named_lock(self):
        service = _make_service()
        with pytest.raises(AssertionError):
            service._get_live_control("unlocked")


class TestDeleteIdentityCheckedPop:
    """Pin delete()'s identity-checked pop: a replaced control must survive."""

    @pytest.mark.asyncio
    async def test_delete_does_not_evict_a_control_installed_after_its_lookup(
        self, monkeypatch
    ):
        service = _make_service()
        old_control = service._get_control("box")
        new_control = _SandboxControl(name="box")

        real_find_container = service._find_container

        async def find_container_and_swap(name: str):
            # Simulate another in-flight path installing a fresh control
            # object for this name between delete()'s control lookup and its
            # cleanup pop.
            service._controls[name] = new_control
            return await real_find_container(name)

        monkeypatch.setattr(service, "_find_container", find_container_and_swap)

        await service.delete("box")

        assert old_control is not new_control
        assert service._controls.get("box") is new_control


class TestSandboxControlSingleConstructionPoint:
    """Source-level pin: only _get_control and _get_live_control may
    construct a _SandboxControl. A new inline construction elsewhere is a
    regression of the single-construction-point contract.

    Two sanctioned construction points: _get_control (legacy paths,
    deleted-preserving) and _get_live_control (lock-held paths,
    deleted-replacing); no inline construction elsewhere.
    """

    def test_sandbox_control_constructed_in_exactly_two_places(self):
        """Resolved over the AST, not raw text.

        A substring count also matches occurrences in comments and
        docstrings and breaks on reformatting, neither of which has anything
        to do with the contract. What matters is which *functions* contain a
        real call to the constructor.
        """
        source = Path(docker_sandbox_module.__file__).read_text()
        tree = ast.parse(source)

        constructing_functions: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_SandboxControl"
                ):
                    constructing_functions.append(node.name)

        assert sorted(constructing_functions) == ["_get_control", "_get_live_control"]
