"""Concurrent channel-manager syncs must not interleave.

Channel CRUD endpoints schedule ``_sync_bots_async`` as a background task
(``trigger_telegram_sync`` / ``trigger_feishu_sync``), so two syncs can
overlap. The real suspension point is inside ``_stop_bot_for_token`` /
``_stop_bot_for_appid``, which awaits the bot's shutdown drain and only then
removes it from ``self.bots``.

Without serialization that window loses a channel: sync A decides to stop a
token that has disappeared from the database and suspends in the drain; sync B
runs, sees the token still present in ``self.bots`` (A has not removed it yet)
and therefore does not start it, even though the database now lists it as
active again; A resumes and completes the removal. The token ends up neither
in ``self.bots`` nor running, and nothing starts it until the next sync.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.web.channels.feishu.bot import FeishuChannelManager
from xagent.web.channels.telegram.bot import TelegramChannelManager
from xagent.web.services.channel_runtime import ChannelConfigSnapshot

# Bound every wait so a regression that stops a sync from reaching the drain
# fails the test instead of hanging the CI job (no global pytest timeout is
# configured for this project).
_TEST_TIMEOUT_SECONDS = 5.0


@pytest.mark.asyncio
async def test_telegram_sync_does_not_interleave_with_a_pending_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TelegramChannelManager()
    token = "tg-token"
    # A bot is running for a token the database no longer lists.
    manager.bots[token] = object()  # type: ignore[assignment]

    active_tokens: list[str] = []
    drain_entered = asyncio.Event()
    release_drain = asyncio.Event()

    async def load_configs(**_kwargs: Any) -> tuple[ChannelConfigSnapshot, ...]:
        return tuple(
            ChannelConfigSnapshot(
                channel_id=1,
                channel_name="Telegram Bot",
                config_items=(("bot_token", value),),
            )
            for value in active_tokens
        )

    async def stop_bot(stopped_token: str) -> None:
        # Mirrors the real method: suspend on the shutdown drain, then remove.
        drain_entered.set()
        await release_drain.wait()
        manager.bots.pop(stopped_token, None)

    started: list[str] = []

    async def start_bot(started_token: str, *_args: Any, **_kwargs: Any) -> None:
        if started_token in manager.bots:
            return
        started.append(started_token)
        manager.bots[started_token] = object()  # type: ignore[assignment]

    manager._stop_bot_for_token = stop_bot  # type: ignore[method-assign]
    manager._start_bot_for_token = start_bot  # type: ignore[method-assign]
    monkeypatch.setattr(
        "xagent.web.channels.telegram.bot.load_active_channel_configs",
        load_configs,
    )

    # Sync A: token absent from the DB, so it stops the running bot.
    first = asyncio.create_task(manager._sync_bots_async())
    await asyncio.wait_for(drain_entered.wait(), timeout=_TEST_TIMEOUT_SECONDS)

    # The token is re-enabled while A is suspended mid-stop.
    active_tokens.append(token)
    second = asyncio.create_task(manager._sync_bots_async())
    await asyncio.sleep(0)
    # Pin the intent rather than relying on one scheduler step being enough:
    # B must be parked on the lock, not past it.
    assert manager._sync_lock.locked()

    release_drain.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=_TEST_TIMEOUT_SECONDS)

    # B must observe the completed stop and start the re-enabled token.
    assert started == [token]
    assert token in manager.bots


@pytest.mark.asyncio
async def test_feishu_sync_does_not_interleave_with_a_pending_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = FeishuChannelManager()
    app_id = "cli_123"
    manager.bots[app_id] = object()  # type: ignore[assignment]

    active_ids: list[str] = []
    drain_entered = asyncio.Event()
    release_drain = asyncio.Event()

    async def load_configs(**_kwargs: Any) -> tuple[ChannelConfigSnapshot, ...]:
        return tuple(
            ChannelConfigSnapshot(
                channel_id=2,
                channel_name="Feishu Bot",
                config_items=(("app_id", value), ("app_secret", "secret")),
            )
            for value in active_ids
        )

    async def stop_bot(stopped_id: str) -> None:
        drain_entered.set()
        await release_drain.wait()
        manager.bots.pop(stopped_id, None)

    started: list[str] = []

    async def start_bot(started_id: str, *_args: Any, **_kwargs: Any) -> None:
        if started_id in manager.bots:
            return
        started.append(started_id)
        manager.bots[started_id] = object()  # type: ignore[assignment]

    manager._stop_bot_for_appid = stop_bot  # type: ignore[method-assign]
    manager._start_bot_for_appid = start_bot  # type: ignore[method-assign]
    monkeypatch.setattr(
        "xagent.web.channels.feishu.bot.load_active_channel_configs",
        load_configs,
    )

    first = asyncio.create_task(manager._sync_bots_async())
    await asyncio.wait_for(drain_entered.wait(), timeout=_TEST_TIMEOUT_SECONDS)

    active_ids.append(app_id)
    second = asyncio.create_task(manager._sync_bots_async())
    await asyncio.sleep(0)
    assert manager._sync_lock.locked()

    release_drain.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=_TEST_TIMEOUT_SECONDS)

    assert started == [app_id]
    assert app_id in manager.bots
