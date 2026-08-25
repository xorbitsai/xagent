"""Unit tests for AgentServiceManager.invalidate_cached_agents_for_owner.

A cached AgentService bakes its system prompt in at construction time and
the per-turn cache-hit path only re-checks owner/scope invariants, not
preferences (see the end-to-end regression in
test_agent_manager_reconstruction.py::test_voice_change_invalidates_cached_agent_on_next_turn).
This method is the PATCH /api/auth/me/preferences endpoint's only way to
reach an already-warm task's cache; these tests cover its dict bookkeeping,
and the in-flight-execution/concurrent-build guards added after review
found the first version could orphan a live execution or lose a race
against a concurrent build, in isolation.
"""

import asyncio

import pytest

from xagent.web.api.chat import AgentServiceManager


class _CachedAgent:
    """Minimal stand-in for a cached AgentService: only what
    invalidate_cached_agents_for_owner and cleanup paths touch."""

    def __init__(self, execution_status=None, raise_on_cleanup=False):
        self._execution_status = execution_status
        self._raise_on_cleanup = raise_on_cleanup

    def get_execution_status(self, execution_id: str):
        return self._execution_status

    def cleanup_workspace(self) -> None:
        if self._raise_on_cleanup:
            raise AssertionError("workspace must not be cleaned up")


@pytest.mark.asyncio
async def test_invalidate_evicts_only_the_matching_owners_tasks() -> None:
    manager = AgentServiceManager()
    manager._agents[1] = _CachedAgent()
    manager._agents[2] = _CachedAgent()
    manager._agent_owner_ids[1] = 7
    manager._agent_owner_ids[2] = 8
    manager._agent_sandbox_keys[1] = "user:7"
    manager._agent_sandbox_keys[2] = "user:8"
    manager._agent_sandbox_providers[1] = object()
    manager._agent_sandbox_providers[2] = object()
    manager._agent_scope_fingerprints[1] = None
    manager._agent_scope_fingerprints[2] = None

    await manager.invalidate_cached_agents_for_owner(7)

    assert 1 not in manager._agents
    assert 1 not in manager._agent_owner_ids
    assert 1 not in manager._agent_sandbox_keys
    assert 1 not in manager._agent_sandbox_providers
    assert 1 not in manager._agent_scope_fingerprints
    # A different owner's cached task must survive untouched.
    assert 2 in manager._agents
    assert manager._agent_owner_ids[2] == 8


@pytest.mark.asyncio
async def test_invalidate_does_not_touch_workspace() -> None:
    """Unlike remove_agent, this must not call cleanup_workspace - the
    same owner's on-disk data must survive the rebuild (mirrors the
    scope-fingerprint-mismatch eviction's own workspace-preserving
    behavior in _get_agent_for_task_unlocked)."""
    manager = AgentServiceManager()
    manager._agents[1] = _CachedAgent(raise_on_cleanup=True)
    manager._agent_owner_ids[1] = 7

    await manager.invalidate_cached_agents_for_owner(7)

    assert 1 not in manager._agents


@pytest.mark.asyncio
async def test_invalidate_is_a_noop_for_an_owner_with_no_cached_tasks() -> None:
    manager = AgentServiceManager()
    manager._agents[1] = _CachedAgent()
    manager._agent_owner_ids[1] = 7

    await manager.invalidate_cached_agents_for_owner(999)

    assert 1 in manager._agents
    assert manager._agent_owner_ids[1] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "execution_status",
    [
        {"is_running": True, "is_resumable": False},
        {"is_running": False, "is_resumable": True},
    ],
)
async def test_invalidate_defers_eviction_for_an_in_flight_task(
    execution_status,
) -> None:
    """Popping a task's AgentService mid-execution orphans that execution:
    the next live-control call (stop/interrupt/message) would build a
    *new* AgentService with an empty execution registry, disconnected
    from the real run. A running or paused/waiting-for-user execution
    must defer eviction, not force it."""
    manager = AgentServiceManager()
    manager._agents[1] = _CachedAgent(execution_status=execution_status)
    manager._agent_owner_ids[1] = 7

    await manager.invalidate_cached_agents_for_owner(7)

    assert 1 in manager._agents
    assert manager._agent_owner_ids[1] == 7


@pytest.mark.asyncio
async def test_invalidate_evicts_a_task_with_a_completed_execution() -> None:
    manager = AgentServiceManager()
    manager._agents[1] = _CachedAgent(
        execution_status={"is_running": False, "is_resumable": False}
    )
    manager._agent_owner_ids[1] = 7

    await manager.invalidate_cached_agents_for_owner(7)

    assert 1 not in manager._agents


@pytest.mark.asyncio
async def test_invalidate_defers_rather_than_blocks_on_an_in_flight_build() -> None:
    """get_agent_for_task holds _agent_build_locks[task_id] for the
    duration of a build (sandbox startup, remote MCP init - multi-second
    work). A same-moment invalidation must not block waiting its turn on
    that lock: with several of an owner's tasks mid-build at once, that
    would serialize a single preference-change PATCH behind all of them
    with no timeout. It must defer this task_id instead - the same
    tolerance already used for an in-flight execution - and return
    promptly; a later trigger (another preference change, a scope/owner
    mismatch) still catches the stale-voice result once the build
    finishes and gets cached."""
    manager = AgentServiceManager()
    manager._agent_owner_ids[1] = 7
    build_lock = asyncio.Lock()
    manager._agent_build_locks[1] = build_lock

    build_started = asyncio.Event()
    release_build = asyncio.Event()

    async def fake_build() -> None:
        async with build_lock:
            build_started.set()
            await release_build.wait()
            # The build "finishes" by publishing its (now-stale-voice)
            # result while still holding the lock, exactly like
            # get_agent_for_task does before releasing it.
            manager._agents[1] = _CachedAgent()

    build_task = asyncio.create_task(fake_build())
    await asyncio.wait_for(build_started.wait(), timeout=2)

    # Invalidation must return promptly - not block on the build's lock -
    # even though the build hasn't finished yet.
    await asyncio.wait_for(manager.invalidate_cached_agents_for_owner(7), timeout=1)
    assert 1 not in manager._agents

    release_build.set()
    await asyncio.wait_for(build_task, timeout=2)

    # The build's (stale-voice) result is cached now that it's finished -
    # this same invalidation call already returned and did not evict it,
    # by design; a later invalidation call is what would catch it.
    assert 1 in manager._agents


@pytest.mark.asyncio
async def test_invalidate_does_not_serialize_behind_several_concurrent_builds() -> None:
    """Regression for the exact reported failure mode: a user with
    several tasks mid-build at once must not have a single preferences
    PATCH's cache invalidation block behind all of them, one after
    another, with no timeout."""
    manager = AgentServiceManager()
    build_locks = {}
    release_builds = []
    build_tasks = []
    for task_id in (1, 2, 3):
        manager._agent_owner_ids[task_id] = 7
        lock = asyncio.Lock()
        build_locks[task_id] = lock
        manager._agent_build_locks[task_id] = lock
        started = asyncio.Event()
        release = asyncio.Event()
        release_builds.append(release)

        async def fake_build(lock=lock, started=started, release=release) -> None:
            async with lock:
                started.set()
                await release.wait()

        build_tasks.append(asyncio.create_task(fake_build()))
        await asyncio.wait_for(started.wait(), timeout=2)

    # All three builds are still in flight (none released). Invalidation
    # must still return well within a single build's own timeout budget,
    # not accumulate a wait per task.
    await asyncio.wait_for(manager.invalidate_cached_agents_for_owner(7), timeout=1)

    for release in release_builds:
        release.set()
    await asyncio.wait_for(asyncio.gather(*build_tasks), timeout=2)
