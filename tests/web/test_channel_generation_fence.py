"""Chat intake belongs to the serving generation only.

``start_allowed`` lets a start through when the record cannot be read, so the
real fence for the chat channels has to live in the shared manager sync that
every start path goes through: the manager's own loop, channel create/update
and the Slack OAuth/deactivation sync. An unreadable record keeps the channels
dormant; they open only once this generation is confirmed as the serving one.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.web.channels.feishu import bot as feishu_bot
from xagent.web.channels.slack import bot as slack_bot
from xagent.web.channels.telegram import bot as telegram_bot
from xagent.web.services import generation
from xagent.web.services.channel_runtime import ChannelConfigSnapshot


def _snapshot(items: tuple[tuple[str, str], ...]) -> ChannelConfigSnapshot:
    return ChannelConfigSnapshot(channel_id=1, channel_name="c", config_items=items)


def _arm(monkeypatch: pytest.MonkeyPatch, kind: str) -> tuple[Any, list[str], Any]:
    """Return (manager, started, api_sync) with one active channel configured."""
    started: list[str] = []

    async def record(*_a: Any, **_k: Any) -> None:
        started.append(kind)

    if kind == "telegram":
        mod, manager = telegram_bot, telegram_bot.TelegramChannelManager()
        items = (("bot_token", "tg-token"),)
        manager._start_bot_for_token = record  # type: ignore[method-assign]
    elif kind == "feishu":
        mod, manager = feishu_bot, feishu_bot.FeishuChannelManager()
        items = (("app_id", "app"), ("app_secret", "secret"))
        manager._start_bot_for_appid = record  # type: ignore[method-assign]
    else:
        mod, manager = slack_bot, slack_bot.SlackChannelManager()
        items = (("bot_token", "xoxb"), ("app_token", "xapp"))
        manager._start_bot_for_token = record  # type: ignore[method-assign]

    async def load(**_k: Any) -> tuple[ChannelConfigSnapshot, ...]:
        return (_snapshot(items),)

    monkeypatch.setattr(mod, "load_active_channel_configs", load)
    monkeypatch.setattr(mod, "get_channel_ingress_enabled", lambda: True)
    return manager, started, mod


KINDS = ["telegram", "feishu", "slack"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_sync_keeps_channels_dormant_until_serving_is_confirmed(
    monkeypatch, kind
) -> None:
    """Startup read error, then recovery with another generation serving."""
    monkeypatch.setenv(generation.GENERATION_ENV, "blue")
    manager, started, _ = _arm(monkeypatch, kind)
    record: dict[str, Any] = {"value": RuntimeError("db blip")}

    def read() -> str | None:
        if isinstance(record["value"], Exception):
            raise record["value"]
        return record["value"]

    monkeypatch.setattr(generation, "serving_generation", read)

    # The record is unreadable: fail closed, nothing opens.
    await manager._sync_bots_async()
    assert started == []
    # Recovered, and green serves: still nothing opens for blue.
    record["value"] = "green"
    await manager._sync_bots_async()
    assert started == []
    # Blue becomes the serving generation: now the sync opens the channel.
    record["value"] = "blue"
    await manager._sync_bots_async()
    assert started == [kind]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_direct_sync_from_the_api_respects_the_record(monkeypatch, kind) -> None:
    """Channel create/update and Slack OAuth call the sync helpers directly."""
    from xagent.web.api import channel as channel_api

    monkeypatch.setenv(generation.GENERATION_ENV, "blue")
    _, started, _ = _arm(monkeypatch, kind)
    # The helpers resolve the process-wide manager; arm that one.
    mod = {"telegram": telegram_bot, "feishu": feishu_bot, "slack": slack_bot}[kind]
    getter = getattr(mod, f"get_{kind}_channel")
    manager = getter()
    for attr in ("_start_bot_for_token", "_start_bot_for_appid"):
        if hasattr(manager, attr):

            async def record(*_a: Any, **_k: Any) -> None:
                started.append(kind)

            monkeypatch.setattr(manager, attr, record)
    monkeypatch.setattr(generation, "serving_generation", lambda: "green")
    trigger = getattr(channel_api, f"trigger_{kind}_sync")
    await trigger()
    assert started == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a_generation_that_stops_serving_closes_its_running_bots(
    monkeypatch, kind
) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "blue")
    manager, started, _ = _arm(monkeypatch, kind)
    stopped: list[str] = []

    async def stop() -> None:
        stopped.append(kind)

    manager.stop = stop  # type: ignore[method-assign]
    serving = {"value": "blue"}
    monkeypatch.setattr(generation, "serving_generation", lambda: serving["value"])
    await manager._sync_bots_async()
    assert started == [kind] and stopped == []
    # A read blip keeps what is running: it only refuses to open anything new.
    monkeypatch.setattr(
        generation,
        "serving_generation",
        lambda: (_ for _ in ()).throw(RuntimeError("blip")),
    )
    await manager._sync_bots_async()
    assert stopped == []
    monkeypatch.setattr(generation, "serving_generation", lambda: serving["value"])
    serving["value"] = "green"
    await manager._sync_bots_async()
    assert stopped == [kind]
    assert started == [kind]  # nothing new opened


def _forbid_record_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> str | None:
        raise AssertionError("generations are off: the record must not be read")

    monkeypatch.setattr(generation, "serving_generation", forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_generations_off_sync_reads_and_logs_nothing_new(
    monkeypatch, caplog, kind
) -> None:
    monkeypatch.delenv(generation.GENERATION_ENV, raising=False)
    manager, started, _ = _arm(monkeypatch, kind)
    _forbid_record_reads(monkeypatch)
    with caplog.at_level("DEBUG", logger=generation.logger.name):
        await manager._sync_bots_async()
    assert started == [kind]
    assert caplog.records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_generations_off_start_returns_after_one_sync_without_shared_mode(
    monkeypatch, kind
) -> None:
    monkeypatch.delenv(generation.GENERATION_ENV, raising=False)
    manager, started, mod = _arm(monkeypatch, kind)
    monkeypatch.setattr(mod, "get_shared_task_execution_enabled", lambda: False)
    _forbid_record_reads(monkeypatch)
    await asyncio.wait_for(manager.start(), timeout=5)
    assert started == [kind]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_manager_loop_keeps_retrying_until_serving_is_confirmed(
    monkeypatch, kind
) -> None:
    monkeypatch.setenv(generation.GENERATION_ENV, "blue")
    manager, started, mod = _arm(monkeypatch, kind)
    monkeypatch.setattr(mod, "get_shared_task_execution_enabled", lambda: False)
    reads = iter([RuntimeError("blip"), "green", "blue"])

    def read() -> str | None:
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(generation, "serving_generation", read)
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        # Only the manager's 30s retry interval is replaced; the test's own
        # yields keep their real behaviour.
        if delay == 30:
            sleeps.append(delay)
            delay = 0
        await real_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    task = asyncio.create_task(manager.start())
    for _ in range(200):
        if started:
            break
        await real_sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert started == [kind]
    assert sleeps and set(sleeps) == {30}
