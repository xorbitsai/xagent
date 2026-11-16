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


# -- round 2: the fence sits at every connect point ---------------------------


def _set_finishing_during(awaitable_factory):
    async def inner(*_a: Any, **_k: Any) -> Any:
        release_hold.begin_finishing()
        return awaitable_factory()

    return inner


@pytest.mark.asyncio
async def test_slack_manual_bot_does_not_connect_when_the_fence_falls_during_auth(
    monkeypatch,
) -> None:
    from xagent.web.channels.slack import bot as mod

    built: list[Any] = []
    monkeypatch.setattr(mod, "SocketModeClient", lambda **kw: built.append(kw))
    bot = object.__new__(mod.SlackBotInstance)
    bot.app_token, bot.instance_id, bot.bot_user_id = "xapp", "i", None
    bot._accepting, bot._run_forever, bot.socket_client = True, asyncio.Event(), None

    async def auth_test() -> dict[str, str]:
        release_hold.begin_finishing()  # the fence falls while auth is awaited
        return {"user_id": "U"}

    bot.web_client = SimpleNamespace(auth_test=auth_test)
    await bot.start()
    assert built == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_slack_oauth_gateway_never_connects_once_finishing(
    monkeypatch, fail_first
) -> None:
    from xagent.web.channels.slack import bot as mod

    built: list[Any] = []

    class _Client:
        def __init__(self, **kw: Any) -> None:
            built.append(kw)
            self.socket_mode_request_listeners: list[Any] = []

        async def connect(self) -> None:
            if fail_first:
                release_hold.begin_finishing()  # fence falls between retries
                raise RuntimeError("connect failed")
            raise AssertionError("must not connect")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(mod, "SocketModeClient", _Client)

    async def no_sleep(self: Any) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(mod._RetryBackoff, "sleep", no_sleep)
    gateway = mod.SlackOAuthSocketGateway("xapp", lambda _p: None)
    if not fail_first:
        release_hold.begin_finishing()
    await asyncio.wait_for(gateway.start(), 2)
    assert len(built) == (1 if fail_first else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("installation_mode", ["manual", "oauth"])
async def test_slack_sync_paused_across_begin_finishing_opens_nothing(
    monkeypatch, installation_mode
) -> None:
    from xagent.web.channels.slack import bot as mod

    manager = mod.SlackChannelManager()
    gateways: list[Any] = []
    bots: list[Any] = []

    class _Gateway:
        def __init__(self, **kw: Any) -> None:
            gateways.append(kw)
            self.app_token = kw["app_token"]

        async def start(self) -> None:
            await asyncio.sleep(60)

        async def stop(self) -> None:
            return None

    class _Bot:
        def __init__(self, **kw: Any) -> None:
            bots.append(kw)
            self.polling_task = None
            self.channel_id = kw.get("channel_id")
            self.bot_token = kw.get("bot_token")
            self.bot_user_id = kw.get("bot_user_id")

        async def start(self) -> None:
            await asyncio.sleep(60)

        async def stop(self) -> None:
            return None

    class _Cfg:
        channel_id, channel_name = 7, "c"

        def config_value(self, key: str) -> Any:
            return {
                "installation_mode": installation_mode,
                "team_id": "T1",
                "bot_token": "xoxb",
                "app_token": "xapp",
                "bot_user_id": "U1",
            }.get(key)

    gate = asyncio.Event()

    async def load(**_kw: Any) -> list[Any]:
        await gate.wait()
        return [_Cfg()]

    async def allowed(*_a: Any) -> bool:
        return True

    monkeypatch.setattr(mod, "SlackOAuthSocketGateway", _Gateway)
    monkeypatch.setattr(mod, "SlackBotInstance", _Bot)
    monkeypatch.setattr(mod, "load_active_channel_configs", load)
    monkeypatch.setattr(mod, "tick_allowed", allowed)
    monkeypatch.setattr(mod, "get_slack_app_token", lambda: "xapp")
    monkeypatch.setattr(mod, "get_channel_ingress_enabled", lambda: True)
    sync = asyncio.create_task(manager._sync_bots_async())
    await asyncio.sleep(0.05)
    release_hold.begin_finishing()
    await manager.stop()
    gate.set()
    await sync
    assert gateways == [] and bots == []
    assert manager.oauth_bots == {} and manager.bots == {}
    assert manager.oauth_gateway is None


@pytest.mark.asyncio
async def test_feishu_bot_neither_connects_nor_reconnects_once_finishing(
    monkeypatch,
) -> None:
    from xagent.web.channels.feishu import bot as mod

    calls: list[str] = []

    class _Ws:
        _auto_reconnect = True

        def __init__(self, *a: Any, **k: Any) -> None:
            calls.append("built")

        async def _connect(self) -> None:
            calls.append("connect")
            release_hold.begin_finishing()  # fence falls during the connect
            raise RuntimeError("connect failed")

        async def _disconnect(self) -> None:
            calls.append("disconnect")

        async def _reconnect(self) -> None:
            calls.append("reconnect")

    monkeypatch.setattr(mod.lark.ws, "Client", _Ws)
    bot = object.__new__(mod.FeishuBotInstance)
    bot.app_id, bot.app_secret, bot.instance_id = "a", "s", "i"
    bot._accepting, bot.ws_client = True, None
    await bot.start()
    assert "reconnect" not in calls
    calls.clear()  # now finishing before the start: nothing is even built
    await bot.start()
    assert calls == []


@pytest.mark.asyncio
async def test_telegram_bot_stops_before_polling_when_the_fence_falls(
    monkeypatch,
) -> None:
    from xagent.web.channels.telegram import bot as mod

    calls: list[str] = []

    async def delete_webhook(**_kw: Any) -> None:
        calls.append("delete_webhook")
        release_hold.begin_finishing()  # fence falls during the first await

    async def set_my_commands(_c: Any) -> None:
        calls.append("set_my_commands")

    async def start_polling(*_a: Any, **_k: Any) -> None:
        calls.append("start_polling")

    bot = object.__new__(mod.TelegramBotInstance)
    bot._accepting, bot.instance_id, bot.bot_commands = True, "i", []
    bot.bot = SimpleNamespace(
        delete_webhook=delete_webhook, set_my_commands=set_my_commands
    )
    bot.dp = SimpleNamespace(start_polling=start_polling)
    await bot.start()
    assert "start_polling" not in calls
    calls.clear()
    await bot.start()
    assert calls == []


def test_every_connect_and_registration_call_is_preceded_by_the_finishing_check() -> (
    None
):
    """A new opener without the fence fails here, not in review.

    Inventories every call that opens an intake connection and every place a
    bot or gateway is registered in the three channel modules, and requires a
    ``release_hold.finishing`` test earlier in the same function.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "src/xagent/web/channels"
    connect_attrs = {"connect", "_connect", "_reconnect", "start_polling"}
    connect_names = {"SocketModeClient"}
    registries = {"bots", "oauth_bots", "oauth_gateway"}
    found: list[str] = []
    unfenced: list[str] = []
    for path in sorted(root.glob("*/bot.py")):
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            checks = [
                n.lineno
                for n in ast.walk(func)
                if isinstance(n, ast.If)
                and "release_hold.finishing" in ast.unparse(n.test)
            ]
            for node in ast.walk(func):
                what = None
                if isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Attribute) and f.attr in connect_attrs:
                        what = f.attr
                    elif isinstance(f, ast.Name) and f.id in connect_names:
                        what = f.id
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        base = t.value if isinstance(t, ast.Subscript) else t
                        if (
                            isinstance(base, ast.Attribute)
                            and base.attr in registries
                            and not (
                                isinstance(node.value, ast.Constant)
                                and node.value.value is None
                            )
                        ):
                            what = f"register {base.attr}"
                if what is None:
                    continue
                label = f"{path.parent.name}:{func.name}:{what}"
                found.append(label)
                if not any(line < node.lineno for line in checks):
                    unfenced.append(label)
    # The inventory is not empty: a refactor that renames these must update it.
    assert len(found) >= 12, found
    assert unfenced == [], unfenced


# -- round 2: idle counts tracked writer threads ------------------------------


@pytest.mark.asyncio
async def test_idle_waits_for_a_tracked_writer_thread(monkeypatch) -> None:
    from xagent.web.services import writer_workers

    started, release = threading.Event(), threading.Event()

    def blocking(**_kw: Any) -> None:
        started.set()
        release.wait(10)

    monkeypatch.setattr(ws_api, "_enqueue_websocket_task_command_sync", blocking)
    monkeypatch.setattr(ws_api, "get_shared_task_execution_enabled", lambda: False)
    release_hold.begin_finishing()
    # A pause is exempt from the write fence, so it holds no socket-write token:
    # only the tracked thread tells that a row is still being written.
    task = asyncio.create_task(
        ws_api._enqueue_websocket_task_command(
            task_id=1,
            message_data={"user": SimpleNamespace(id=1, is_admin=True)},
            kind=ws_api.TaskCommandKind.PAUSE,
        )
    )
    await asyncio.to_thread(started.wait, 5)
    status = await finishing_status_async()
    assert writer_workers.active_workers() == 1
    assert status["tracked_workers"] == 1 and status["socket_writes"] == 0
    assert status["idle"] is False
    release.set()
    await asyncio.wait_for(task, 5)
    assert (await finishing_status_async())["idle"] is True


# -- round 2: admissions while the reply probe runs ---------------------------


async def _with_probe_paused(monkeypatch, during):
    proceed, in_probe = threading.Event(), threading.Event()

    def slow() -> int:
        in_probe.set()
        proceed.wait(10)
        return 0

    monkeypatch.setattr(release_hold_module, "_pending_reply_count", slow)
    status = asyncio.create_task(finishing_status_async())
    await asyncio.to_thread(in_probe.wait, 5)
    during()
    proceed.set()
    return await asyncio.wait_for(status, 5)


@pytest.mark.asyncio
async def test_idle_is_false_when_work_is_admitted_while_the_probe_runs(
    monkeypatch,
) -> None:
    release_hold.begin_finishing()
    held: list[Any] = []
    # A steering socket write is still admitted after the fence (try_admit is
    # refused by then), so it is what can appear while the probe runs.
    status = await _with_probe_paused(
        monkeypatch, lambda: held.append(release_hold.acquire_socket_write(None))
    )
    assert status["socket_writes"] == 1 and status["idle"] is False
    held[0].release()


@pytest.mark.asyncio
async def test_idle_is_false_when_work_was_held_before_the_probe_and_ended_during_it(
    monkeypatch,
) -> None:
    token = release_hold.try_admit("trigger_run", "x")
    assert token is not None
    release_hold.begin_finishing()
    status = await _with_probe_paused(monkeypatch, token.release)
    # Zero afterwards, but the earlier read saw work: no stable zero observed.
    assert status["responsibilities"] == 0 and status["idle"] is False


# -- round 2: host counters are validated -------------------------------------


@pytest.mark.parametrize(
    "name",
    ["active", "idle", "owned_runs", "tracked_workers", "pending_replies"],
)
def test_reserved_counter_names_are_refused(name) -> None:
    with pytest.raises(ValueError):
        release_hold_module.register_finishing_counter(name, lambda: 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [-1, 0.5, True, None, "3"])
async def test_a_counter_that_is_not_a_count_is_unreadable_never_idle(bad) -> None:
    release_hold_module.register_finishing_counter("turns", lambda: bad)
    release_hold.begin_finishing()
    status = await finishing_status_async()
    assert status["turns"] is None and status["idle"] is False


@pytest.mark.asyncio
async def test_a_negative_counter_cannot_cancel_a_built_in_count() -> None:
    release_hold_module.register_finishing_counter("turns", lambda: -1)
    token = release_hold.try_admit("trigger_run", "x")
    assert token is not None
    release_hold.begin_finishing()
    assert (await finishing_status_async())["idle"] is False
    token.release()
