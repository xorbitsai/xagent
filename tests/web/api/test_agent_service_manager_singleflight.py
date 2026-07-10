"""Concurrency regression tests for task-scoped AgentService construction."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from xagent.web.api.chat import AgentServiceManager


@pytest.mark.asyncio
async def test_get_agent_for_task_serializes_builds_for_same_task() -> None:
    manager = AgentServiceManager()
    active_builds = 0
    max_active_builds = 0

    async def _build(*args, **kwargs):
        nonlocal active_builds, max_active_builds
        active_builds += 1
        max_active_builds = max(max_active_builds, active_builds)
        await asyncio.sleep(0)
        active_builds -= 1
        return object()

    manager._get_agent_for_task_unlocked = AsyncMock(side_effect=_build)

    await asyncio.gather(
        manager.get_agent_for_task(42),
        manager.get_agent_for_task(42),
    )

    assert max_active_builds == 1
    assert manager._get_agent_for_task_unlocked.await_count == 2


@pytest.mark.asyncio
async def test_get_agent_for_task_allows_different_tasks_to_build_concurrently() -> (
    None
):
    manager = AgentServiceManager()
    both_started = asyncio.Event()
    started: set[int] = set()

    async def _build(task_id: int, **kwargs):
        started.add(task_id)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return object()

    manager._get_agent_for_task_unlocked = AsyncMock(side_effect=_build)

    await asyncio.gather(
        manager.get_agent_for_task(42),
        manager.get_agent_for_task(43),
    )

    assert started == {42, 43}
