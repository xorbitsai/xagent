"""Standby activation stays bounded when many processes activate on one host."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from xagent.web import app as app_module
from xagent.web.services import task_event_bridge as bridge_module
from xagent.web.services.standby import (
    StandbyActivationError,
    StandbyController,
    StandbyPhase,
    StandbyStepTimeoutError,
)
from xagent.web.services.writer_workers import run_tracked_to_completion

_WAIT = 10.0


# --- per-step timeout -------------------------------------------------------


@pytest.mark.asyncio
async def test_step_timeout_fails_activation_naming_the_step(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_STANDBY_STEP_TIMEOUT_SECONDS", "0.1")
    controller = StandbyController()
    ran: list[str] = []
    hooks: list[str] = []

    async def fast() -> None:
        ran.append("fast")

    async def stuck() -> None:
        await asyncio.sleep(_WAIT)

    async def never() -> None:
        ran.append("never")

    async def hook() -> None:
        hooks.append("hook")

    controller.begin()
    controller.defer("fast step", fast)
    controller.defer("stuck step", stuck)
    controller.defer("after step", never)
    controller.on_failure(hook)

    with pytest.raises(StandbyStepTimeoutError, match="stuck step") as info:
        await controller.activate()

    assert isinstance(info.value, StandbyActivationError)
    assert info.value.step == "stuck step"
    assert ran == ["fast"]
    assert hooks == ["hook"]
    assert controller.phase is StandbyPhase.FAILED


@pytest.mark.asyncio
async def test_step_timeout_joins_a_tracked_worker_instead_of_abandoning_it(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XAGENT_STANDBY_STEP_TIMEOUT_SECONDS", "0.1")
    controller = StandbyController()
    started, release, finished = (threading.Event() for _ in range(3))
    hook_saw_finished: list[bool] = []

    def worker() -> None:
        started.set()
        release.wait(_WAIT)
        finished.set()

    async def step() -> None:
        await run_tracked_to_completion(worker)

    async def hook() -> None:
        hook_saw_finished.append(finished.is_set())

    controller.begin()
    controller.defer("blocking step", step)
    controller.on_failure(hook)

    activation = asyncio.create_task(controller.activate())
    await asyncio.get_running_loop().run_in_executor(None, started.wait, _WAIT)
    await asyncio.sleep(0.3)  # past the bound: the cancel is now waiting on the thread
    assert not activation.done()
    assert not finished.is_set()

    release.set()
    with pytest.raises(StandbyStepTimeoutError, match="blocking step"):
        await asyncio.wait_for(activation, _WAIT)
    assert hook_saw_finished == [True]


@pytest.mark.asyncio
async def test_a_timeout_raised_inside_a_step_is_not_reported_as_the_bound() -> None:
    controller = StandbyController()

    async def step() -> None:
        raise TimeoutError("upstream call timed out")

    controller.begin()
    controller.defer("calls upstream", step)
    with pytest.raises(TimeoutError, match="upstream call") as info:
        await controller.activate()
    assert not isinstance(info.value, StandbyStepTimeoutError)


# --- sandbox cleanup bound and background warmup ----------------------------


class _Manager:
    def __init__(self, *, cleanup_hangs: bool = False) -> None:
        self.cleanup_hangs = cleanup_hangs
        self.warmup_started = asyncio.Event()
        self.warmup_release = asyncio.Event()
        self.warmup_finished = False

    async def cleanup(self) -> None:
        if self.cleanup_hangs:
            await asyncio.sleep(_WAIT)

    async def warmup(self) -> None:
        self.warmup_started.set()
        await self.warmup_release.wait()
        self.warmup_finished = True

    async def run_idle_sweep_loop(self) -> None:  # pragma: no cover - not scheduled
        await asyncio.sleep(_WAIT)


@pytest.fixture
def no_idle_sweep(monkeypatch) -> None:
    monkeypatch.setattr("xagent.config.get_sandbox_idle_ttl", lambda: None)


@pytest.fixture(autouse=True)
def _reset_warmup_task():
    yield
    for attr in ("_sandbox_warmup_task", "_sandbox_deferred_stops_task"):
        task = getattr(app_module, attr)
        if task is not None and not task.done():
            task.cancel()
        setattr(app_module, attr, None)


@pytest.mark.asyncio
async def test_warmup_is_not_awaited_inside_the_activation_step(no_idle_sweep) -> None:
    manager = _Manager()
    await asyncio.wait_for(app_module._start_sandbox_runtime(manager), 2)

    # The step returned while the warmup is still running.
    assert not manager.warmup_finished
    task = app_module._sandbox_warmup_task
    assert task is not None and not task.done()

    await asyncio.wait_for(manager.warmup_started.wait(), 2)
    manager.warmup_release.set()
    await asyncio.wait_for(task, 2)
    assert manager.warmup_finished


def _running(name: str):
    from xagent.sandbox.base import SandboxConfig, SandboxInfo, SandboxTemplate

    return SandboxInfo(
        name=name,
        state="running",
        template=SandboxTemplate(type="image", image="img:v1"),
        config=SandboxConfig(),
    )


class _StopService:
    """Reconciling backend whose first stop outlives the cleanup budget."""

    def __init__(self, names: list[str], *, reconciles: bool = True) -> None:
        self.names = names
        self.reconciles = reconciles
        self.issued: list[str] = []
        self.finished: list[str] = []
        self.deleted: list[str] = []

    async def supports_runtime_spec(self) -> bool:
        return self.reconciles

    async def list_sandboxes(self):
        return [_running(n) for n in self.names]

    async def stop_existing(self, name: str, *, timeout: float) -> None:
        self.issued.append(name)
        await asyncio.sleep(0.3)  # a stop the budget must not cut short
        self.finished.append(name)

    async def delete(self, name: str) -> None:
        self.deleted.append(name)


@pytest.mark.asyncio
async def test_cleanup_budget_stops_issuing_stops_but_never_cuts_an_issued_one(
    monkeypatch, no_idle_sweep
) -> None:
    from xagent.web.sandbox_manager import SandboxManager

    monkeypatch.setenv("XAGENT_SANDBOX_STARTUP_CLEANUP_TIMEOUT_SECONDS", "0.1")
    service = _StopService(["a", "b", "c"])
    manager = SandboxManager(service)  # type: ignore[arg-type]
    manager.warmup = AsyncMock()  # type: ignore[method-assign]

    await asyncio.wait_for(app_module._start_sandbox_runtime(manager), 5)

    # The first stop was issued before the budget ran out and ran to completion;
    # nothing was cancelled mid-stop, and the startup issued no further stop.
    assert service.issued[0] == "a"
    assert service.finished == ["a"]
    # The skipped containers are handed to the background task, not dropped.
    deferred_task = app_module._sandbox_deferred_stops_task
    assert deferred_task is not None
    await asyncio.wait_for(deferred_task, 5)
    assert service.finished == ["a", "b", "c"]
    task = app_module._sandbox_warmup_task
    assert task is not None  # the step went on to schedule the warmup
    await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_legacy_cleanup_ignores_the_budget_so_required_work_is_not_skipped(
    monkeypatch, no_idle_sweep
) -> None:
    from xagent.web.sandbox_manager import SandboxManager, bounded_quiesce

    service = _StopService(["a", "b"], reconciles=False)
    manager = SandboxManager(service)  # type: ignore[arg-type]
    manager._get_sandbox_image_and_config = lambda: ("img:v1", None)  # type: ignore[method-assign]

    async def stop_existing(name: str, *, timeout: float) -> None:
        service.issued.append(name)

    service.stop_existing = stop_existing  # type: ignore[method-assign]

    with bounded_quiesce(0.0):  # budget already spent
        await manager.cleanup()

    assert service.issued == ["a", "b"]


@pytest.mark.asyncio
async def test_cancelled_quiesce_still_resets_process_local_state() -> None:
    from xagent.web.sandbox_manager import SandboxManager

    class _Service:
        async def list_sandboxes(self) -> list[Any]:
            await asyncio.sleep(_WAIT)
            return []

    manager = SandboxManager(_Service())  # type: ignore[arg-type]
    manager._cache["stale"] = object()  # type: ignore[index]
    manager._activity["stale"] = 1.0  # type: ignore[index]

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(manager._quiesce(), 0.1)

    assert not manager._cache
    assert not manager._activity


@pytest.mark.asyncio
async def test_list_sandboxes_reads_the_store_off_the_event_loop(monkeypatch) -> None:
    from xagent.sandbox import docker_sandbox as module

    loop_thread = threading.get_ident()
    read_threads: list[int] = []

    class _Store:
        def get_info(self, name: str) -> None:
            read_threads.append(threading.get_ident())

    service = object.__new__(module.DockerSandboxService)
    service._namespace = None
    service._store = _Store()
    service._client = SimpleNamespace(
        containers=SimpleNamespace(list=lambda **_kw: [object(), object()])
    )
    monkeypatch.setattr(
        module,
        "_parse_container_config",
        lambda _c: SimpleNamespace(name="sb"),
    )
    monkeypatch.setattr(module, "_merge_info", lambda runtime, _stored: runtime)

    result = await service.list_sandboxes()

    assert len(result) == 2
    assert len(read_threads) == 2
    assert loop_thread not in read_threads


# --- task event bridge Redis connect ----------------------------------------


class _FakePubSub:
    def __init__(self, owner: _FakeRedis) -> None:
        self._owner = owner
        self._channels: tuple[str, ...] = ()

    async def __aenter__(self) -> _FakePubSub:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def subscribe(self, *channels: str) -> None:
        self._owner.attempts += 1
        if self._owner.attempts <= self._owner.failures:
            raise ConnectionError("redis slow to accept")
        self._channels = channels

    async def listen(self):
        for channel in self._channels:
            yield {"type": "subscribe", "channel": channel}
        await asyncio.sleep(_WAIT)


class _FakeRedis:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts = 0

    def pubsub(self) -> _FakePubSub:
        return _FakePubSub(self)

    async def aclose(self) -> None:
        return None


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    return bridge_module.TaskEventBridge()


@pytest.mark.asyncio
async def test_bridge_start_retries_a_slow_redis_then_succeeds(bridge) -> None:
    bridge.redis = _FakeRedis(failures=3)
    bridge._status = AsyncMock()  # type: ignore[method-assign]

    await asyncio.wait_for(bridge.start(), 8)

    assert bridge.ready.is_set()
    assert bridge.redis.attempts == 4
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_start_fails_after_the_bound_when_redis_stays_down(
    monkeypatch, bridge
) -> None:
    monkeypatch.setenv("XAGENT_TASK_EVENT_BRIDGE_START_TIMEOUT_SECONDS", "1")
    bridge.redis = _FakeRedis(failures=10**6)
    bridge._status = AsyncMock()  # type: ignore[method-assign]

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(bridge.start(), 5)

    # The bridge's own bound fired (1 s), not the outer guard (5 s).
    assert time.monotonic() - started < 3
    assert bridge.redis.attempts >= 2  # it retried inside the bound
    assert not bridge.ready.is_set()


# --- controller -> Docker: an issued stop outlives no lifecycle guard -------


class _BlockingStopContainer:
    """Container whose ``stop`` blocks in its worker thread until released."""

    def __init__(self) -> None:
        self.name = "c"
        self.attrs: dict[str, Any] = {"State": {"Status": "running"}}
        self.labels: dict[str, str] = {}
        self.stop_started = threading.Event()
        self.release_stop = threading.Event()
        self.stop_finished = threading.Event()

    def reload(self) -> None:
        pass

    def stop(self, **_: Any) -> None:
        self.stop_started.set()
        self.release_stop.wait(_WAIT)
        self.stop_finished.set()


def _docker_service(container: Any) -> Any:
    from xagent.sandbox.docker_sandbox import DockerSandboxService, MemDockerStore

    client = SimpleNamespace(
        containers=SimpleNamespace(list=lambda *a, **k: [container]),
        ping=lambda: True,
    )
    return DockerSandboxService(
        MemDockerStore(), namespace="test", client=client, network="net"
    )


@pytest.mark.asyncio
async def test_step_timeout_keeps_same_name_exclusion_until_the_stop_settles(
    monkeypatch,
) -> None:
    """The step bound may fire before the Docker stop grace period ends.

    Cancelling the awaiter of ``to_thread(container.stop)`` must not free the
    per-name lock while the daemon call is still running, or a same-name task
    could reuse the container and then have the late stop terminate it.
    """
    monkeypatch.setenv("XAGENT_STANDBY_STEP_TIMEOUT_SECONDS", "0.1")
    container = _BlockingStopContainer()
    service = _docker_service(container)
    controller = StandbyController()
    reused_while_stop_running: list[bool] = []

    async def stop_step() -> None:
        await service.stop_existing("sbx", timeout=30)

    async def same_name_reuse() -> None:
        async with service._named_lock("sbx"):
            reused_while_stop_running.append(not container.stop_finished.is_set())

    controller.begin()
    controller.defer("cleanup", stop_step)

    activation = asyncio.create_task(controller.activate())
    assert await asyncio.to_thread(container.stop_started.wait, _WAIT)
    await asyncio.sleep(0.4)  # well past the 0.1 s step bound
    reuse = asyncio.create_task(same_name_reuse())
    await asyncio.sleep(0.2)
    # The step bound has fired, but activation has not returned and the
    # same-name task is still excluded because the stop is still in flight.
    assert not activation.done()
    assert not reuse.done()

    container.release_stop.set()
    with pytest.raises(StandbyStepTimeoutError, match="cleanup"):
        await activation
    await reuse
    assert reused_while_stop_running == [False]
    assert controller.phase is StandbyPhase.FAILED


@pytest.mark.asyncio
async def test_cancelled_warmup_removes_its_reserved_container_before_returning() -> (
    None
):
    from xagent.web.sandbox_manager import SandboxManager

    events: list[str] = []
    create_started = asyncio.Event()
    release_create = threading.Event()
    release_delete = threading.Event()
    delete_started = asyncio.Event()

    class _Service:
        async def get_or_create(self, name: str, **_: Any) -> Any:
            def create() -> None:
                release_create.wait(_WAIT)
                events.append("created")

            create_started.set()
            from xagent.sandbox.settle import await_shielded

            await await_shielded(asyncio.to_thread(create))
            raise AssertionError("not reached")

        async def delete(self, name: str) -> None:
            delete_started.set()
            await asyncio.to_thread(release_delete.wait, _WAIT)
            events.append(f"delete:{name}")

    manager = object.__new__(SandboxManager)
    manager._service = _Service()
    task = asyncio.create_task(manager.warmup())
    await create_started.wait()
    task.cancel()
    await asyncio.sleep(0.1)
    assert not task.done()  # still waiting for the issued create
    release_create.set()
    await delete_started.wait()
    task.cancel()  # a second cancel while the removal is in flight
    await asyncio.sleep(0.1)
    assert not task.done()
    release_delete.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["created", "delete:__warmup__"]
