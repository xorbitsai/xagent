"""Finish-only mode ends chat intake; a phase-1 retire is left exactly as it was.

``begin_finishing`` stops the channels, but channel create/update (and the
Slack OAuth sync) call the managers' ``_sync_bots_async`` directly, which would
open them again. The sync therefore opens nothing while the hold is finishing.
It is keyed on ``finishing`` only: the phase-1 retire (closed / sealed hold)
keeps behaving as it did.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.test_channel_generation_fence import _arm
from xagent.web.api import channel as channel_api
from xagent.web.services import generation
from xagent.web.services import release_hold as release_hold_module
from xagent.web.services.release_hold import release_hold

KINDS = ["telegram", "feishu", "slack"]


@pytest.fixture(autouse=True)
def _fresh_hold(monkeypatch):
    monkeypatch.delenv(generation.GENERATION_ENV, raising=False)
    monkeypatch.setattr(release_hold, "_closed", False)
    monkeypatch.setattr(release_hold, "_sealed", False)
    monkeypatch.setattr(release_hold, "_finishing", False)
    monkeypatch.setattr(release_hold_module, "_finishing_counters", {})


def _arm_process_manager(monkeypatch, kind: str) -> list[str]:
    """Arm the process-wide manager that the CRUD helpers resolve."""
    _, started, mod = _arm(monkeypatch, kind)
    manager = getattr(mod, f"get_{kind}_channel")()

    async def record(*_a: Any, **_k: Any) -> None:
        started.append(kind)

    for attr in ("_start_bot_for_token", "_start_bot_for_appid"):
        if hasattr(manager, attr):
            monkeypatch.setattr(manager, attr, record)
    return started


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_direct_sync_opens_nothing_while_finishing(monkeypatch, kind) -> None:
    started = _arm_process_manager(monkeypatch, kind)
    release_hold.begin_finishing()
    await getattr(channel_api, f"trigger_{kind}_sync")()
    assert started == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_direct_sync_is_unchanged_when_not_finishing(monkeypatch, kind) -> None:
    started = _arm_process_manager(monkeypatch, kind)
    await getattr(channel_api, f"trigger_{kind}_sync")()
    assert started == [kind]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("state", ["_closed", "_sealed"])
async def test_a_phase_one_retire_does_not_change_channel_sync(
    monkeypatch, kind, state
) -> None:
    """closed / sealed is what retire_writers sets: sync behaves as before."""
    started = _arm_process_manager(monkeypatch, kind)
    monkeypatch.setattr(release_hold, state, True)
    assert not release_hold.finishing
    await getattr(channel_api, f"trigger_{kind}_sync")()
    assert started == [kind]
