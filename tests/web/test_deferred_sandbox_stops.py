"""Containers the budgeted startup quiesce skips are still stopped, once."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from xagent.sandbox.base import SandboxNotFoundError, SandboxRecoveryRequiredError
from xagent.web import app as app_module
from xagent.web.sandbox_manager import SandboxManager, bounded_quiesce


class _Service:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.stopped: list[str] = []
        self.events: list[str] = []
        self.fail: set[str] = set()
        self.missing: set[str] = set()

    async def supports_runtime_spec(self) -> bool:
        return True

    async def list_sandboxes(self) -> list[Any]:
        return [SimpleNamespace(name=n, state="running") for n in self.names]

    async def stop_existing(self, name: str, *, timeout: float) -> None:
        if name in self.missing:
            raise SandboxNotFoundError(name)
        if name in self.fail:
            raise RuntimeError("daemon down")
        self.stopped.append(name)
        self.events.append(f"stop:{name}")

    async def inspect(self, name: str) -> None:
        self.events.append(f"inspect:{name}")


NAMES = ["task::1", "task::2"]


async def _defer_all(service: _Service) -> SandboxManager:
    manager = SandboxManager(service)  # type: ignore[arg-type]
    with bounded_quiesce(0.0):  # budget already spent
        await manager._quiesce()
    return manager


@pytest.fixture(autouse=True)
async def _reset_tasks():
    async def reset() -> None:
        for attr in ("_sandbox_deferred_stops_task", "_sandbox_warmup_task"):
            task = getattr(app_module, attr)
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            setattr(app_module, attr, None)

    await reset()  # another module's test may have left a task behind
    yield
    await reset()


@pytest.mark.asyncio
async def test_spent_budget_defers_and_logs_names_at_warning(caplog) -> None:
    service = _Service(NAMES)
    with caplog.at_level(logging.WARNING):
        manager = await _defer_all(service)

    assert service.stopped == []
    assert manager._deferred_stops == set(NAMES)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "task::1" in r.getMessage() and "task::2" in r.getMessage() for r in warnings
    )


@pytest.mark.asyncio
async def test_drain_stops_every_deferred_container_once() -> None:
    service = _Service(NAMES)
    manager = await _defer_all(service)

    await manager.drain_deferred_stops()
    await manager.drain_deferred_stops()

    assert sorted(service.stopped) == NAMES
    assert manager._deferred_stops == set()


@pytest.mark.asyncio
async def test_reuse_stops_a_deferred_container_before_inspecting_it() -> None:
    service = _Service(NAMES)
    manager = await _defer_all(service)
    manager._reconcile_existing = AsyncMock(return_value="sandbox")  # type: ignore[method-assign]
    manager._reconcile_create = AsyncMock(return_value="sandbox")  # type: ignore[method-assign]
    service.events.clear()

    await manager._reconcile_sandbox(
        "task",
        "1",
        "task::1",
        "task::1",
        None,
        None,  # type: ignore[arg-type]
    )

    assert service.events[:2] == ["stop:task::1", "inspect:task::1"]
    await manager.drain_deferred_stops()
    assert service.stopped == ["task::1", "task::2"]  # task::1 not stopped twice


@pytest.mark.asyncio
async def test_failed_stop_blocks_reuse_and_requeues_the_name() -> None:
    service = _Service(NAMES)
    manager = await _defer_all(service)
    service.fail.add("task::1")

    with pytest.raises(SandboxRecoveryRequiredError):
        await manager._reconcile_sandbox(
            "task",
            "1",
            "task::1",
            "task::1",
            None,
            None,  # type: ignore[arg-type]
        )
    assert "task::1" in manager._deferred_stops

    await manager.drain_deferred_stops()  # background: logs, does not raise
    assert "task::1" in manager._deferred_stops
    assert service.stopped == ["task::2"]


@pytest.mark.asyncio
async def test_a_vanished_container_is_dropped_so_a_recreated_one_is_safe() -> None:
    service = _Service(NAMES)
    manager = await _defer_all(service)
    service.missing.add("task::1")

    await manager.drain_deferred_stops()

    assert "task::1" not in manager._deferred_stops
    assert service.stopped == ["task::2"]


@pytest.mark.asyncio
async def test_sandbox_start_hands_deferred_stops_to_a_background_task(
    monkeypatch,
) -> None:
    monkeypatch.setattr("xagent.config.get_sandbox_idle_ttl", lambda: None)
    monkeypatch.setenv("XAGENT_SANDBOX_STARTUP_CLEANUP_TIMEOUT_SECONDS", "0.0001")
    service = _Service(NAMES)

    async def slow_list() -> list[Any]:
        await asyncio.sleep(0.01)  # the budget is spent before the first stop
        return [SimpleNamespace(name=n, state="running") for n in NAMES]

    service.list_sandboxes = slow_list  # type: ignore[method-assign]
    manager = SandboxManager(service)  # type: ignore[arg-type]
    manager.warmup = AsyncMock()  # type: ignore[method-assign]

    await asyncio.wait_for(app_module._start_sandbox_runtime(manager), 5)

    task = app_module._sandbox_deferred_stops_task
    assert task is not None
    assert task.get_name() == app_module.SANDBOX_DEFERRED_STOPS_TASK_NAME
    await asyncio.wait_for(task, 5)
    assert sorted(service.stopped) == NAMES


@pytest.mark.asyncio
async def test_sandbox_start_schedules_nothing_when_nothing_was_deferred(
    monkeypatch,
) -> None:
    monkeypatch.setattr("xagent.config.get_sandbox_idle_ttl", lambda: None)
    service = _Service(NAMES)
    manager = SandboxManager(service)  # type: ignore[arg-type]
    manager.warmup = AsyncMock()  # type: ignore[method-assign]

    await asyncio.wait_for(app_module._start_sandbox_runtime(manager), 5)

    assert app_module._sandbox_deferred_stops_task is None
    assert sorted(service.stopped) == NAMES  # default budget: stopped inline


@pytest.mark.asyncio
async def test_malformed_timeout_fails_boot_before_standby_begins(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_STANDBY", "true")
    monkeypatch.setenv("XAGENT_STANDBY_STEP_TIMEOUT_SECONDS", "soon")
    begun: list[bool] = []
    monkeypatch.setattr(app_module.standby, "begin", lambda: begun.append(True))

    with pytest.raises(ValueError, match="XAGENT_STANDBY_STEP_TIMEOUT_SECONDS"):
        await app_module.startup_event()

    assert begun == []


@pytest.mark.asyncio
async def test_cancelled_reuse_stop_keeps_the_name_queued() -> None:
    service = _Service(NAMES)
    manager = await _defer_all(service)

    async def cancelled(name: str, *, timeout: float) -> None:
        raise asyncio.CancelledError

    service.stop_existing = cancelled  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await manager._reconcile_sandbox(
            "task",
            "1",
            "task::1",
            "task::1",
            None,
            None,  # type: ignore[arg-type]
        )
    assert "task::1" in manager._deferred_stops


@pytest.mark.asyncio
async def test_drain_stops_an_unparsable_name_and_logs_it(caplog) -> None:
    service = _Service(["__warmup__"])
    manager = await _defer_all(service)
    assert manager._deferred_stops == {"__warmup__"}

    with caplog.at_level(logging.WARNING):
        await manager.drain_deferred_stops()

    assert service.stopped == ["__warmup__"]
    assert manager._deferred_stops == set()
    assert any("__warmup__" in r.getMessage() for r in caplog.records)
