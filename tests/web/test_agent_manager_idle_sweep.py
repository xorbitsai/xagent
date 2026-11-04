"""Idle sweep of cached AgentService runtimes in AgentServiceManager.

The manager keeps one runtime per task for as long as the process lives unless
something drops it. These tests pin the rules by which an idle runtime is
dropped, what is never dropped, and what bookkeeping goes with it. Time comes
from the manager's injectable ``_monotonic`` clock; nothing sleeps.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.web.test_agent_manager_reconstruction import _build_reconstruction_snapshot
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services import agent_service_manager as manager_module
from xagent.web.services.agent_service_manager import AgentServiceManager

MODULE_LOGGER = "xagent.web.services.agent_service_manager"

# Per-task dictionaries removed together with a dropped runtime.
CLEARED_MAPS = (
    "_agents",
    "_agent_owner_ids",
    "_agent_sandbox_keys",
    "_agent_sandbox_providers",
    "_agent_scope_fingerprints",
    "_agent_model_overrides",
    "_agent_run_ids",
    "_agent_run_generations",
    "_agent_evicted_scope_fingerprints",
    "_agent_last_active_at",
)

IDLE_RUNTIME_SECONDS = manager_module._IDLE_RUNTIME_MAX_AGE_SECONDS
RESUMABLE_SECONDS = manager_module._IDLE_RESUMABLE_MAX_AGE_SECONDS
SOFT_CAP = manager_module._IDLE_RUNTIME_SOFT_CAP
SOFT_CAP_FLOOR = manager_module._IDLE_RUNTIME_SOFT_CAP_MIN_AGE_SECONDS

NOT_RUNNING = None
RUNNING = {"status": "running", "is_running": True, "is_resumable": False}


def resumable(state: str) -> dict[str, Any]:
    return {"status": state, "is_running": False, "is_resumable": True}


class FakeRuntime:
    """The one AgentService method the sweep reads."""

    def __init__(self, status: dict[str, Any] | None = NOT_RUNNING) -> None:
        self.status = status

    def get_execution_status(self, execution_id: str) -> dict[str, Any] | None:
        if self.status == "raises":
            raise RuntimeError("status unavailable")
        return self.status  # type: ignore[return-value]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def manager(clock: Clock, monkeypatch: pytest.MonkeyPatch):
    mgr = AgentServiceManager()
    mgr._monotonic = clock
    yield mgr
    task = mgr._idle_sweep_task
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def cache(
    mgr: AgentServiceManager,
    task_id: int,
    runtime: FakeRuntime | None = None,
    *,
    acquired_at: float | None = None,
) -> FakeRuntime:
    """Seed every per-task map the way an acquisition leaves them."""
    runtime = runtime or FakeRuntime()
    mgr._agents[task_id] = runtime  # type: ignore[assignment]
    mgr._agent_owner_ids[task_id] = 1
    mgr._agent_sandbox_keys[task_id] = "user::1"
    mgr._agent_sandbox_providers[task_id] = object()
    mgr._agent_scope_fingerprints[task_id] = None
    mgr._agent_model_overrides[task_id] = (runtime, object())  # type: ignore[assignment]
    mgr._agent_run_ids[task_id] = f"run-{task_id}"
    mgr._agent_run_generations[task_id] = task_id
    mgr._agent_evicted_scope_fingerprints[task_id] = MagicMock()
    mgr._agent_last_active_at[task_id] = (
        mgr._monotonic() if acquired_at is None else acquired_at
    )
    mgr._agent_build_locks[task_id] = asyncio.Lock()
    return runtime


def assert_fully_forgotten(mgr: AgentServiceManager, task_id: int) -> None:
    for name in CLEARED_MAPS:
        assert task_id not in getattr(mgr, name), name


def eviction_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if "Evicted idle cached AgentService" in r.message
    ]


def notify_on_periodic_sweep(
    mgr: AgentServiceManager, monkeypatch: pytest.MonkeyPatch, event: asyncio.Event
) -> None:
    """Set ``event`` after every sweep the periodic loop completes."""
    real_sweep = mgr._sweep_idle_runtimes

    def sweep(**kwargs: Any) -> None:
        real_sweep(**kwargs)
        if kwargs.get("reason") == "periodic":
            event.set()

    monkeypatch.setattr(mgr, "_sweep_idle_runtimes", sweep)


@pytest.fixture
def make_trigger(manager: AgentServiceManager, monkeypatch: pytest.MonkeyPatch):
    """Ways to make the manager sweep once: insert, settled hook, periodic."""
    state = {"periodic_started": False}
    periodic_swept = asyncio.Event()

    async def insert() -> None:
        manager._sweep_idle_runtimes(reason="insert", just_inserted=999)

    async def settled_hook() -> None:
        manager.note_runtime_settled(999)

    async def periodic() -> None:
        if not state["periodic_started"]:
            monkeypatch.setattr(manager_module, "_global_agent_manager", manager)
            monkeypatch.setattr(manager_module, "_IDLE_SWEEP_INTERVAL_SECONDS", 0)
            notify_on_periodic_sweep(manager, monkeypatch, periodic_swept)
            manager._ensure_idle_sweep_task()
            state["periodic_started"] = True
        # Wait for the next full periodic sweep rather than for a duration.
        periodic_swept.clear()
        await asyncio.wait_for(periodic_swept.wait(), timeout=5)

    return {"insert": insert, "settled_hook": settled_hook, "periodic": periodic}


@pytest.mark.parametrize("trigger", ["insert", "settled_hook", "periodic"])
async def test_idle_runtime_evicted_after_min_age(
    manager, clock, make_trigger, caplog, trigger
) -> None:
    cache(manager, 1)
    sweep = make_trigger[trigger]

    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        if trigger == "settled_hook":
            # The idle window counts from the settled turn, not the acquisition.
            clock.now += 200
            manager.note_runtime_settled(1)
            assert 1 in manager._agents
        clock.now += IDLE_RUNTIME_SECONDS - 1
        await sweep()
        assert 1 in manager._agents
        assert eviction_records(caplog) == []

        clock.now += 1
        await sweep()

    assert_fully_forgotten(manager, 1)
    assert len(eviction_records(caplog)) == 1


@pytest.mark.parametrize(
    "kind",
    ["running", "status_raises", "build_lock_held", "actor_marked", "just_inserted"],
)
async def test_protected_runtime_never_evicted(manager, clock, kind) -> None:
    status = {"running": RUNNING, "status_raises": "raises"}.get(kind, NOT_RUNNING)
    cache(manager, 1, FakeRuntime(status))
    just_inserted = 1 if kind == "just_inserted" else None
    if kind == "build_lock_held":
        await manager._agent_build_locks[1].acquire()
    if kind == "actor_marked":
        manager._mcp_actor_policies[1] = MagicMock()

    clock.now += 1e6
    manager._sweep_idle_runtimes(reason="test", just_inserted=just_inserted)

    assert 1 in manager._agents
    assert 1 in manager._agent_run_generations


@pytest.mark.parametrize("state", ["interrupted", "waiting_for_user"])
async def test_resumable_runtime_evicted_only_after_long_age(
    manager, clock, state
) -> None:
    cache(manager, 1, FakeRuntime(resumable(state)))

    clock.now += RESUMABLE_SECONDS - 1
    manager._sweep_idle_runtimes(reason="test")
    assert 1 in manager._agents

    clock.now += 1
    manager._sweep_idle_runtimes(reason="test")
    assert_fully_forgotten(manager, 1)


async def acquire(mgr: AgentServiceManager, task_id: int, runtime: FakeRuntime):
    """Run the real acquisition wrapper around a stubbed build."""

    async def build(*_args: Any, **_kwargs: Any) -> FakeRuntime:
        mgr._agents[task_id] = runtime  # type: ignore[assignment]
        return runtime

    with patch.object(mgr, "_get_agent_for_task_unlocked", side_effect=build):
        return await mgr.get_agent_for_task(task_id)


async def test_acquisition_refreshes_idle_clock(manager, clock) -> None:
    runtime = FakeRuntime()
    await acquire(manager, 1, runtime)

    clock.now += 100
    await acquire(manager, 1, runtime)
    clock.now += 100
    manager._sweep_idle_runtimes(reason="test")
    assert 1 in manager._agents

    clock.now += 20
    manager._sweep_idle_runtimes(reason="test")
    assert 1 not in manager._agents


async def test_fresh_acquisition_before_execution_registers_is_kept(
    manager, clock
) -> None:
    """An acquired runtime whose execution has not registered yet survives."""
    await acquire(manager, 1, FakeRuntime(NOT_RUNNING))

    clock.now += IDLE_RUNTIME_SECONDS - 1
    manager._sweep_idle_runtimes(reason="test")

    assert 1 in manager._agents


@pytest.mark.parametrize("running_count", [0, 5])
async def test_soft_cap_evicts_oldest_idle_first(manager, clock, running_count) -> None:
    total = SOFT_CAP + 10
    for task_id in range(total):
        status = RUNNING if task_id < running_count else NOT_RUNNING
        cache(
            manager,
            task_id,
            FakeRuntime(status),
            acquired_at=clock.now + task_id * 0.001,
        )
    clock.now += SOFT_CAP_FLOOR + 10

    manager._sweep_idle_runtimes(reason="test")

    assert len(manager._agents) == SOFT_CAP + running_count
    # Running runtimes are the oldest acquisitions yet always survive; the
    # earliest idle ones go next.
    assert set(range(running_count)) <= set(manager._agents)
    assert set(range(total)) - set(manager._agents) == set(range(running_count, 10))


async def test_soft_cap_spares_runtimes_younger_than_its_floor(manager, clock) -> None:
    total = SOFT_CAP + 72
    for task_id in range(total):
        cache(
            manager,
            task_id,
            acquired_at=clock.now - 10 + task_id * 0.001,
        )

    manager._sweep_idle_runtimes(reason="test")
    assert len(manager._agents) == total

    clock.now += SOFT_CAP_FLOOR
    manager._sweep_idle_runtimes(reason="test")

    assert len(manager._agents) == SOFT_CAP
    assert set(manager._agents) == set(range(total - SOFT_CAP, total))


class ExecutingRuntime(FakeRuntime):
    """Stands in for AgentService under the real ``manager.execute_task``."""

    async def execute_task(self, **_kwargs: Any) -> dict[str, Any]:
        return {"success": True}


async def test_execution_end_restarts_idle_window(manager, clock) -> None:
    runtime = cache(manager, 1, ExecutingRuntime())
    clock.now += 500

    await manager.execute_task(
        agent_service=runtime,  # type: ignore[arg-type]
        task="run",
        task_id="1",
        manage_task_lease=False,
    )

    # The run ended but nobody has settled it yet; its old timestamp must not
    # make the runtime look long idle.
    manager._sweep_idle_runtimes(reason="test")
    assert 1 in manager._agents

    clock.now += IDLE_RUNTIME_SECONDS - 1
    manager._sweep_idle_runtimes(reason="test")
    assert 1 in manager._agents

    clock.now += 1
    manager._sweep_idle_runtimes(reason="test")
    assert 1 not in manager._agents


async def test_sweep_refreshes_a_running_runtime(manager, clock) -> None:
    runtime = cache(manager, 1, FakeRuntime(RUNNING))
    clock.now += 500

    manager._sweep_idle_runtimes(reason="test")
    assert 1 in manager._agents
    assert manager._agent_last_active_at[1] == clock.now

    runtime.status = NOT_RUNNING
    clock.now += IDLE_RUNTIME_SECONDS - 1
    manager._sweep_idle_runtimes(reason="test")
    assert 1 in manager._agents

    clock.now += 1
    manager._sweep_idle_runtimes(reason="test")
    assert 1 not in manager._agents


async def test_remove_agent_drops_last_active_entry(manager) -> None:
    runtime = cache(manager, 1)
    runtime.workspace = None  # type: ignore[attr-defined]
    runtime.cleanup_workspace = MagicMock()  # type: ignore[attr-defined]

    manager.remove_agent(1)

    runtime.cleanup_workspace.assert_called_once()  # type: ignore[attr-defined]
    assert 1 not in manager._agent_last_active_at


@pytest.mark.parametrize("status", [NOT_RUNNING, resumable("interrupted")])
async def test_settling_a_long_turn_restarts_the_idle_window(
    manager, clock, status
) -> None:
    await acquire(manager, 1, FakeRuntime(status))
    clock.now += 500
    resumable_turn = status is not NOT_RUNNING

    manager.note_runtime_settled(1)
    assert 1 in manager._agents

    if not resumable_turn:
        clock.now += IDLE_RUNTIME_SECONDS - 1
        manager._sweep_idle_runtimes(reason="test")
        assert 1 in manager._agents
        clock.now += 1
        manager._sweep_idle_runtimes(reason="test")
        assert 1 not in manager._agents
    else:
        clock.now += RESUMABLE_SECONDS - 1
        manager._sweep_idle_runtimes(reason="test")
        assert 1 in manager._agents
        clock.now += 1
        manager._sweep_idle_runtimes(reason="test")
        assert 1 not in manager._agents


async def test_settling_an_uncached_task_does_not_create_an_entry(manager) -> None:
    manager.note_runtime_settled(1)

    assert manager._agent_last_active_at == {}


async def test_hook_refuses_off_loop_thread(manager, clock) -> None:
    cache(manager, 1)
    clock.now += 1e6

    with pytest.raises(RuntimeError):
        await asyncio.to_thread(manager.note_runtime_settled, 1)

    assert 1 in manager._agents


def test_every_per_task_map_is_classified() -> None:
    mgr = AgentServiceManager()
    cleared = set(CLEARED_MAPS)
    conditionally_cleared = {"_agent_build_locks"}
    kept = {"_agent_cleanup_owner_ids"}
    untouched = {"_mcp_actor_policies"}

    dict_attributes = {
        name for name, value in vars(mgr).items() if isinstance(value, dict)
    }

    assert dict_attributes == cleared | conditionally_cleared | kept | untouched


async def test_sweep_keeps_cleanup_owner_and_actor_policy(manager, clock) -> None:
    cache(manager, 1)
    manager._agent_cleanup_owner_ids[1] = 7
    cache(manager, 2)
    manager._mcp_actor_policies[2] = MagicMock()

    clock.now += 1e6
    manager._sweep_idle_runtimes(reason="test")

    assert_fully_forgotten(manager, 1)
    assert manager._agent_cleanup_owner_ids == {1: 7}
    assert 2 in manager._agents and 2 in manager._mcp_actor_policies


@pytest.mark.parametrize(
    ("case", "expect_pruned"),
    [
        ("idle_unlocked_no_waiters", True),
        ("locked", False),
        ("has_waiter", False),
        ("younger_than_min_age", False),
        ("task_still_cached", False),
    ],
)
async def test_build_lock_pruned_only_when_idle_unlocked_and_unwaited(
    manager, clock, case, expect_pruned
) -> None:
    lock = asyncio.Lock()
    manager._agent_build_locks[1] = lock
    manager._agent_last_active_at[1] = clock.now - IDLE_RUNTIME_SECONDS
    waiter = None
    if case == "locked":
        await lock.acquire()
    elif case == "has_waiter":
        await lock.acquire()

        async def queue_behind_lock() -> None:
            async with lock:
                pass

        waiter = asyncio.ensure_future(queue_behind_lock())
        await asyncio.sleep(0)
        # Released but not yet taken over by the queued waiter: unlocked, yet
        # replacing the lock now would let a second builder bypass it.
        lock.release()
        assert not lock.locked()
    elif case == "younger_than_min_age":
        manager._agent_last_active_at[1] = clock.now - IDLE_RUNTIME_SECONDS + 1
    elif case == "task_still_cached":
        manager._agents[1] = FakeRuntime()  # type: ignore[assignment]

    manager._prune_build_lock(1)

    assert (1 not in manager._agent_build_locks) is expect_pruned
    if waiter is not None:
        await waiter


async def test_periodic_sweep_starts_lazily_and_stops_with_manager(
    manager, clock, monkeypatch
) -> None:
    monkeypatch.setattr(manager_module, "_global_agent_manager", manager)
    monkeypatch.setattr(manager_module, "_IDLE_SWEEP_INTERVAL_SECONDS", 0)
    periodic_swept = asyncio.Event()
    notify_on_periodic_sweep(manager, monkeypatch, periodic_swept)
    assert manager._idle_sweep_task is None

    await acquire(manager, 1, FakeRuntime())
    task = manager._idle_sweep_task
    assert task is not None and not task.done()

    # A second acquisition reuses the same loop.
    await acquire(manager, 2, FakeRuntime())
    assert manager._idle_sweep_task is task

    # The loop sweeps on its own.
    clock.now += IDLE_RUNTIME_SECONDS
    periodic_swept.clear()
    await asyncio.wait_for(periodic_swept.wait(), timeout=5)
    assert manager._agents == {}

    monkeypatch.setattr(manager_module, "_global_agent_manager", object())
    await asyncio.wait_for(task, timeout=5)


async def test_periodic_sweep_survives_a_failing_sweep(
    manager, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(manager_module, "_global_agent_manager", manager)
    monkeypatch.setattr(manager_module, "_IDLE_SWEEP_INTERVAL_SECONDS", 0)
    calls = 0
    second_call = asyncio.Event()

    def failing_sweep(**_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            second_call.set()
        raise RuntimeError("sweep broke")

    monkeypatch.setattr(manager, "_sweep_idle_runtimes", failing_sweep)

    with caplog.at_level(logging.WARNING, logger=MODULE_LOGGER):
        manager._ensure_idle_sweep_task()
        await asyncio.wait_for(second_call.wait(), timeout=5)

    assert calls >= 2
    assert manager._idle_sweep_task is not None and not manager._idle_sweep_task.done()
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_sandbox_eviction_then_sweep_cleans_side_maps(manager, clock) -> None:
    for task_id in (1, 2):
        cache(manager, task_id)

    manager._evict_agents_for_sandbox("user::1")
    assert manager._agents == {}
    for name in (
        "_agent_run_ids",
        "_agent_run_generations",
        "_agent_last_active_at",
    ):
        assert set(getattr(manager, name)) == {1, 2}, name
    assert set(manager._agent_evicted_scope_fingerprints) == {1, 2}

    clock.now += IDLE_RUNTIME_SECONDS - 1
    manager._sweep_idle_runtimes(reason="test")
    assert set(manager._agent_run_ids) == {1, 2}

    clock.now += 1
    manager._sweep_idle_runtimes(reason="test")
    for task_id in (1, 2):
        assert_fully_forgotten(manager, task_id)
    assert manager._agent_build_locks == {}


@pytest.fixture
def _legacy_task_selection(monkeypatch):
    from xagent.web.services.trace_handlers import DatabaseTraceHandler

    monkeypatch.setattr(
        "xagent.web.tracing.task_database_handler", DatabaseTraceHandler
    )


def _snapshot_for_task_1(trace_events: list[TraceEvent] | None = None):
    task = Task(
        id=1,
        user_id=1,
        title="Task",
        description="",
        status=TaskStatus.PENDING,
        agent_type="standard",
    )
    user = User(id=1, username="owner", password_hash="hash", is_admin=False)
    return _build_reconstruction_snapshot(
        task, user, task_llm=MagicMock(), trace_events=trace_events
    )


async def test_building_a_runtime_sweeps_idle_ones(
    manager, clock, _legacy_task_selection
) -> None:
    cache(manager, 99)
    clock.now += IDLE_RUNTIME_SECONDS

    with (
        patch("xagent.web.services.agent_service_manager.AgentService") as service_cls,
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.services.agent_service_manager.get_memory_store",
            return_value=MagicMock(),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
    ):
        service = MagicMock()
        service.get_execution_status.return_value = NOT_RUNNING
        service_cls.return_value = service
        await manager.get_agent_for_task(
            1,
            MagicMock(),
            user=User(id=1, username="owner", password_hash="hash", is_admin=False),
            task_setup_snapshot=_snapshot_for_task_1(),
            task_owner_user_id=1,
            resolved_execution_scope=None,
        )

    assert 1 in manager._agents
    assert_fully_forgotten(manager, 99)


async def test_rebuilding_from_history_sweeps_idle_ones(
    manager, clock, _legacy_task_selection
) -> None:
    cache(manager, 99)
    clock.now += IDLE_RUNTIME_SECONDS
    snapshot = _snapshot_for_task_1(
        [
            TraceEvent(
                id=1,
                task_id=1,
                event_id="event1",
                event_type="task_start_general",
                timestamp=datetime.now(),
                step_id=None,
                parent_event_id=None,
                data={"goal": "goal"},
            )
        ]
    )

    with (
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch("xagent.web.services.agent_service_manager.AgentService") as service_cls,
    ):
        service = MagicMock()
        service.reconstruct_from_history = AsyncMock()
        service_cls.return_value = service
        await manager._reconstruct_agent_from_history(
            1, None, task_setup_snapshot=snapshot
        )

    assert 1 in manager._agents
    assert_fully_forgotten(manager, 99)
