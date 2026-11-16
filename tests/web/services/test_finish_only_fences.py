"""Finish-only mode fences work admitted before it, at the point of its side effect.

Each test holds the real code path paused across ``begin_finishing`` and checks
what happens when it resumes.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from tests.web.test_channel_generation_fence import _arm
from xagent.web import app as app_module
from xagent.web.api import websocket as ws_api
from xagent.web.models import database
from xagent.web.models.task import Task
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_event_bridge
from xagent.web.services.release_hold import (
    finishing_status,
    finishing_status_async,
    release_hold,
)

engine = engine_fixture
task_id = task_id_fixture

KINDS = ["telegram", "feishu", "slack"]


@pytest.fixture(autouse=True)
def _fresh_hold(monkeypatch):
    monkeypatch.setattr(release_hold, "_closed", False)
    monkeypatch.setattr(release_hold, "_sealed", False)
    monkeypatch.setattr(release_hold, "_finishing", False)
    monkeypatch.setattr(release_hold_module, "_finishing_counters", {})


# -- 1. a channel sync paused across the transition opens nothing ------------


class _FakeBot:
    created: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _FakeBot.created.append("bot")
        self.polling_task = None

    async def start(self) -> None:
        await asyncio.sleep(60)

    async def stop(self) -> None:
        return None


def _arm_real_registration(monkeypatch, kind):
    """The real ``_start_bot_for_*`` with a fake bot instance behind it."""
    manager, _started, mod = _arm(monkeypatch, kind)
    for attr in ("_start_bot_for_token", "_start_bot_for_appid"):
        manager.__dict__.pop(attr, None)
    _FakeBot.created = []
    monkeypatch.setattr(mod, f"{kind.capitalize()}BotInstance", _FakeBot)
    return manager, mod


async def _paused_sync(monkeypatch, kind, *, finish: bool):
    manager, mod = _arm_real_registration(monkeypatch, kind)
    gate = asyncio.Event()
    load = mod.load_active_channel_configs

    async def slow(**kwargs):
        await gate.wait()
        return await load(**kwargs)

    monkeypatch.setattr(mod, "load_active_channel_configs", slow)
    sync = asyncio.create_task(manager._sync_bots_async())
    await asyncio.sleep(0.05)  # past the guards, waiting on configuration I/O
    if finish:
        release_hold.begin_finishing()
        await manager.stop()
    gate.set()
    await sync
    for bot in list(manager.bots.values()):
        if bot.polling_task is not None:
            bot.polling_task.cancel()
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_sync_paused_across_begin_finishing_opens_nothing(
    monkeypatch, kind
) -> None:
    manager = await _paused_sync(monkeypatch, kind, finish=True)
    assert manager.bots == {} and _FakeBot.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_the_same_paused_sync_opens_the_bot_when_not_finishing(
    monkeypatch, kind
) -> None:
    manager = await _paused_sync(monkeypatch, kind, finish=False)
    assert len(manager.bots) == 1 and _FakeBot.created == ["bot"]


# -- 2. idle counts everything admitted before the fence ----------------------


class _Socket:
    def __init__(self) -> None:
        self.state = SimpleNamespace()

    async def send_json(self, *_a: Any, **_k: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_idle_waits_for_a_missing_task_request_still_in_setup(
    monkeypatch,
) -> None:
    in_setup, proceed = asyncio.Event(), asyncio.Event()

    async def no_row(**_kwargs: Any) -> None:
        return None

    async def setup(*_a: Any, **_k: Any) -> None:
        in_setup.set()
        await proceed.wait()

    monkeypatch.setattr(ws_api, "_enqueue_websocket_task_command", no_row)
    monkeypatch.setattr(
        ws_api.command_execution_service, "handle_missing_task_message", setup
    )
    handler = asyncio.create_task(ws_api.handle_chat_message(_Socket(), 1, {}))
    await asyncio.wait_for(in_setup.wait(), 5)
    release_hold.begin_finishing()

    status = await finishing_status_async()
    assert status["responsibilities"] == 1 and status["socket_writes"] == 1
    assert status["idle"] is False

    proceed.set()
    await asyncio.wait_for(handler, 5)
    assert (await finishing_status_async())["idle"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["trigger_run", "ws_preview", "trigger_tick"])
async def test_idle_counts_every_admitted_ledger_kind(kind) -> None:
    token = release_hold.try_admit(kind, "x")
    assert token is not None
    release_hold.begin_finishing()
    assert (await finishing_status_async())["idle"] is False
    token.release()
    assert (await finishing_status_async())["idle"] is True


@pytest.mark.asyncio
async def test_idle_ignores_the_dispatchers_own_polling_tokens() -> None:
    for kind in ("command_attempt", "command_dispatch"):
        assert release_hold.try_begin_attempt(kind, "n") is not None
    release_hold.begin_finishing()
    assert (await finishing_status_async())["idle"] is True


@pytest.mark.asyncio
async def test_idle_counts_socket_writes_and_prompt_dispatches(monkeypatch) -> None:
    release_hold.begin_finishing()
    write = release_hold.acquire_socket_write(None)
    status = await finishing_status_async()
    assert status["socket_writes"] == 1 and status["idle"] is False
    write.release()
    from xagent.web.services import task_command_transport as transport

    monkeypatch.setattr(transport, "prompt_dispatch_count", lambda: 1)
    status = await finishing_status_async()
    assert status["prompt_dispatches"] == 1 and status["idle"] is False


# -- 3. the reply probe stays off the event loop; execution_status is sync -----


@pytest.mark.asyncio
async def test_reply_probe_runs_off_the_loop_and_never_stalls_it(monkeypatch) -> None:
    release_hold.begin_finishing()
    seen: dict[str, Any] = {}

    def slow_probe() -> int:
        seen["main"] = threading.current_thread() is threading.main_thread()
        time.sleep(0.4)
        return 0

    monkeypatch.setattr(release_hold_module, "_pending_reply_count", slow_probe)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    before = ticks
    status = await finishing_status_async()
    task.cancel()
    assert seen["main"] is False
    assert ticks - before > 10
    assert status["idle"] is True


def _forbid_queries(monkeypatch) -> None:
    def forbidden() -> int:
        raise AssertionError("no database access here")

    monkeypatch.setattr(release_hold_module, "_pending_reply_count", forbidden)
    monkeypatch.setattr(
        database,
        "get_session_local",
        lambda: (_ for _ in ()).throw(AssertionError("no session")),
    )


def test_execution_status_is_sync_query_free_and_never_idle(monkeypatch) -> None:
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    release_hold.begin_finishing()
    _forbid_queries(monkeypatch)
    assert not inspect.iscoroutinefunction(release_hold_module.execution_status)
    status = release_hold_module.execution_status()
    assert not inspect.isawaitable(status)
    finishing = status["finishing"]
    assert finishing["active"] is True
    assert finishing["pending_replies"] is None and finishing["idle"] is False
    assert {"owned_runs", "responsibilities", "socket_writes"} <= set(finishing)
    # The keys execution_status had before finish-only mode are all still there.
    assert {"busy", "counts", "hold", "sealed", "unsettled_total"} <= set(status)


@pytest.mark.asyncio
async def test_neither_status_function_queries_outside_finish_only(
    monkeypatch,
) -> None:
    monkeypatch.setattr(task_event_bridge, "_bridge", SimpleNamespace(host_id="H"))
    _forbid_queries(monkeypatch)
    off = {"active": False, "pending_replies": 0, "idle": False}
    assert finishing_status() == off
    assert await finishing_status_async() == off
    assert release_hold_module.execution_status()["finishing"] == off


# -- 4. a non-shared topology is refused before anything changes --------------


@pytest.mark.asyncio
async def test_begin_finishing_refuses_non_shared_before_raising_the_fence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app_module, "get_shared_task_execution_enabled", lambda: False)
    calls: list[str] = []

    async def loops(**_k: Any) -> None:
        calls.append("loops")

    async def channels(**_k: Any) -> None:
        calls.append("channels")

    monkeypatch.setattr(app_module, "_stop_runtime_loops", loops)
    monkeypatch.setattr(app_module, "_stop_chat_channels", channels)
    with pytest.raises(RuntimeError, match="shared task execution"):
        await app_module.begin_finishing()
    assert calls == []
    assert not release_hold.finishing and not release_hold.closed
    assert not release_hold.sealed


# -- 5. no new task ownership once finishing -----------------------------------


@pytest.fixture
def factory(engine, monkeypatch):
    result = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "_SessionLocal", result)
    return result


def _runner(factory, task_id) -> str | None:
    with factory() as db:
        return db.get(Task, task_id).runner_id


@pytest.mark.asyncio
async def test_ensure_creates_no_coordinator_while_finishing(
    factory, task_id, monkeypatch
) -> None:
    acquisitions: list[int] = []
    acquire = runtime.TaskCoordinator._acquire

    def counting_acquire(self):
        acquisitions.append(self.task_id)
        return acquire(self)

    monkeypatch.setattr(runtime.TaskCoordinator, "_acquire", counting_acquire)
    registry = runtime.TaskCoordinatorRegistry()
    release_hold.begin_finishing()
    try:
        assert await asyncio.wait_for(registry.ensure(task_id), 10) is None
        assert registry._coordinators == {}
    finally:
        await registry.close()
    # Not even a lease taken and handed back: nothing was acquired at all.
    assert acquisitions == []
    assert _runner(factory, task_id) is None


@pytest.mark.asyncio
async def test_ensure_started_before_the_fence_does_not_keep_a_late_lease(
    factory, task_id, monkeypatch
) -> None:
    registry = runtime.TaskCoordinatorRegistry()
    entered, proceed = threading.Event(), threading.Event()
    acquire = runtime.TaskCoordinator._acquire

    def paused_acquire(self):
        entered.set()
        assert proceed.wait(10)
        return acquire(self)

    monkeypatch.setattr(runtime.TaskCoordinator, "_acquire", paused_acquire)
    ensuring = asyncio.create_task(registry.ensure(task_id))
    try:
        for _ in range(500):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        release_hold.begin_finishing()
        proceed.set()
        assert await asyncio.wait_for(ensuring, 10) is None
        for _ in range(500):
            if not registry._coordinators:
                break
            await asyncio.sleep(0.01)
        assert registry._coordinators == {}
        assert _runner(factory, task_id) is None
    finally:
        proceed.set()
        await registry.close()


@pytest.mark.asyncio
async def test_a_task_owned_before_the_fence_keeps_its_coordinator(
    factory, task_id
) -> None:
    registry = runtime.TaskCoordinatorRegistry()
    try:
        owned = await asyncio.wait_for(registry.ensure(task_id), 10)
        assert owned is not None
        release_hold.begin_finishing()
        assert await asyncio.wait_for(registry.ensure(task_id), 10) is owned
    finally:
        await registry.close()
