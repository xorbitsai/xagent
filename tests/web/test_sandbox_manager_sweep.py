"""Tests for the idle sandbox TTL sweep in SandboxManager.

The sweep runs against the ``SandboxService`` abstraction with a fake
service and a controllable clock — no Docker-only assumptions.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import xagent.web.sandbox_manager as sandbox_manager_module
from tests.web.sandbox_fakes import FakeSandboxService
from xagent.web.sandbox_manager import SandboxManager

TTL = 100.0


class _FakeClock:
    """Deterministic replacement for the ``time`` module in the manager."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _listed(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, state="stopped")


@pytest.fixture
def clock(monkeypatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(sandbox_manager_module, "time", fake)
    return fake


def _make_manager(listed_names: list[str] | None = None) -> SandboxManager:
    service = FakeSandboxService()
    service.list_sandboxes = AsyncMock(
        return_value=[_listed(name) for name in listed_names or []]
    )
    service.delete = AsyncMock()
    return SandboxManager(service)


@pytest.mark.asyncio
async def test_idle_sandbox_past_ttl_is_deleted_with_workers(clock) -> None:
    manager = _make_manager(["user::7", "user::7::worker::0", "user::7::worker::1"])
    manager._lease_providers["user::7"] = MagicMock()

    clock.advance(TTL + 1)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert reclaimed == ["user::7"]
    deleted = {call.args[0] for call in manager._service.delete.await_args_list}
    assert deleted == {"user::7", "user::7::worker::0", "user::7::worker::1"}
    assert "user::7" not in manager._lease_providers


@pytest.mark.asyncio
async def test_active_sandbox_is_never_deleted_regardless_of_idle_time(clock) -> None:
    manager = _make_manager(["user::7"])
    manager._lease_providers["user::7"] = MagicMock()
    assert await manager.attach("user", "7")

    clock.advance(TTL * 100)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert reclaimed == []
    manager._service.delete.assert_not_awaited()
    assert "user::7" in manager._lease_providers


@pytest.mark.asyncio
async def test_sandbox_within_ttl_is_kept(clock) -> None:
    manager = _make_manager(["user::7"])
    manager._lease_providers["user::7"] = MagicMock()
    assert await manager.attach("user", "7")
    assert await manager.release("user", "7")

    clock.advance(TTL / 2)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert reclaimed == []
    manager._service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_attach_during_sweep_prevents_deletion(clock) -> None:
    """A sweep firing while a task concurrently attaches must not delete."""
    manager = _make_manager()
    manager._lease_providers["user::7"] = MagicMock()
    list_started = asyncio.Event()
    list_release = asyncio.Event()

    async def slow_list_sandboxes() -> list:
        list_started.set()
        await list_release.wait()
        return [_listed("user::7")]

    manager._service.list_sandboxes.side_effect = slow_list_sandboxes

    clock.advance(TTL + 1)
    sweep_task = asyncio.create_task(manager.sweep_idle_sandboxes(TTL))
    await list_started.wait()

    # The attach lands while the sweep is already past its candidate
    # collection; the per-key re-check must observe it.
    assert await manager.attach("user", "7")
    list_release.set()
    reclaimed = await sweep_task

    assert reclaimed == []
    manager._service.delete.assert_not_awaited()
    assert "user::7" in manager._lease_providers


@pytest.mark.asyncio
async def test_fresh_provider_fetch_resets_idle_clock(clock) -> None:
    """get_or_create_lease_provider bumps activity, protecting the
    create-to-attach window from a concurrent sweep."""
    manager = _make_manager(["user::7"])
    manager._create_lease_provider_locked = AsyncMock(return_value=MagicMock())

    clock.advance(TTL + 1)
    await manager.get_or_create_lease_provider("user", "7")
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert reclaimed == []
    manager._service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_restart_containers_swept_one_ttl_after_startup(clock) -> None:
    """Containers discovered with no recorded activity count as idle since
    manager startup and get exactly one TTL grace period."""
    manager = _make_manager(["user::9", "task::12"])

    clock.advance(TTL - 1)
    assert await manager.sweep_idle_sandboxes(TTL) == []

    clock.advance(2)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert sorted(reclaimed) == ["task::12", "user::9"]


@pytest.mark.asyncio
async def test_swept_sandbox_is_recreated_cleanly_on_next_use(clock) -> None:
    """After reclamation the config cache is evicted so the next use
    recreates the sandbox without a config-equivalence error."""
    manager = _make_manager(["user::7"])
    manager._cache["user::7"] = MagicMock()
    manager._config_cache["user::7"] = MagicMock()
    providers = [MagicMock(), MagicMock()]
    manager._create_lease_provider_locked = AsyncMock(side_effect=providers)

    first = await manager.get_or_create_lease_provider("user", "7")
    clock.advance(TTL + 1)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)
    second = await manager.get_or_create_lease_provider("user", "7")

    assert reclaimed == ["user::7"]
    assert "user::7" not in manager._cache
    assert "user::7" not in manager._config_cache
    assert first is providers[0]
    assert second is providers[1]


@pytest.mark.asyncio
async def test_unmanaged_names_are_ignored(clock) -> None:
    manager = _make_manager(["__warmup__", "not-a-managed-name"])

    clock.advance(TTL + 1)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert reclaimed == []
    manager._service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_failure_degrades_to_cached_candidates(clock) -> None:
    manager = _make_manager()
    manager._service.list_sandboxes.side_effect = RuntimeError("daemon down")
    manager._cache["user::7"] = MagicMock()

    clock.advance(TTL + 1)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert reclaimed == ["user::7"]


@pytest.mark.asyncio
async def test_sweep_loop_exits_immediately_when_ttl_unset(monkeypatch) -> None:
    """TTL unset (default) means no sweep task activity at all."""
    monkeypatch.delenv("XAGENT_SANDBOX_IDLE_TTL", raising=False)
    manager = _make_manager()

    await asyncio.wait_for(manager.run_idle_sweep_loop(), timeout=30)

    manager._service.list_sandboxes.assert_not_awaited()
    manager._service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_loop_runs_periodically_and_survives_errors(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_SANDBOX_IDLE_TTL", "100")
    monkeypatch.setenv("XAGENT_SANDBOX_SWEEP_INTERVAL", "0.01")
    manager = _make_manager()
    sweeps = asyncio.Event()
    calls = []

    async def fake_sweep(ttl: float) -> list[str]:
        calls.append(ttl)
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        sweeps.set()
        return []

    manager.sweep_idle_sandboxes = fake_sweep

    loop_task = asyncio.create_task(manager.run_idle_sweep_loop())
    try:
        await asyncio.wait_for(sweeps.wait(), timeout=1.0)
    finally:
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

    assert calls[0] == 100.0
    assert len(calls) >= 2


@pytest.mark.asyncio
async def test_reconcile_route_idle_sandbox_is_swept(clock, monkeypatch) -> None:
    """The idle sweep also reclaims containers created via the
    spec-reconciliation route (``FakeSandboxService(runtime_spec_supported=
    True)``), not only the legacy ``get_or_create()`` route the rest of this
    module drives."""
    monkeypatch.setattr(
        sandbox_manager_module,
        "build_code_mount_volumes",
        lambda: [("/repo/src", "/app/src", "ro")],
    )
    service = FakeSandboxService(runtime_spec_supported=True)
    manager = SandboxManager(service)
    await manager.get_or_create_sandbox("task", "7")
    assert "task::7" in service._containers

    clock.advance(TTL + 1)
    reclaimed = await manager.sweep_idle_sandboxes(TTL)

    assert reclaimed == ["task::7"]
    assert "task::7" in service.deleted
    assert "task::7" not in service._containers


@pytest.mark.asyncio
async def test_concurrent_recreate_during_sweep_waits_for_lifecycle_lock(
    clock,
) -> None:
    """A same-key get_or_create_lease_provider racing an in-flight sweep
    deletion must block on the per-key lifecycle lock (the sweep's
    exclusion mechanism, unlike capacity eviction's gate) and build a
    fresh provider only after the reclamation finished."""
    manager = _make_manager(["user::7"])
    delete_started = asyncio.Event()
    delete_release = asyncio.Event()

    async def slow_delete(name: str) -> None:
        delete_started.set()
        await delete_release.wait()

    manager._service.delete.side_effect = slow_delete
    providers = [MagicMock(), MagicMock()]
    manager._create_lease_provider_locked = AsyncMock(side_effect=providers)

    clock.advance(TTL + 1)
    sweep_task = asyncio.create_task(manager.sweep_idle_sandboxes(TTL))
    await delete_started.wait()

    racer = asyncio.create_task(manager.get_or_create_lease_provider("user", "7"))
    for _ in range(10):
        await asyncio.sleep(0)
    # Blocked on the lifecycle lock held by the sweep across pop-and-delete.
    assert not racer.done()

    delete_release.set()
    reclaimed = await sweep_task
    provider = await racer

    assert reclaimed == ["user::7"]
    assert provider is providers[0]
    manager._create_lease_provider_locked.assert_awaited_once()
    assert manager._lease_providers["user::7"] is provider
