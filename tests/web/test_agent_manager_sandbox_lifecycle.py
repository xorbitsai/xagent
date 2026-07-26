"""Tests for task-scoped sandbox lifecycle handling in AgentServiceManager.

The active-task ref-count and per-key lifecycle locks live in
``SandboxManager`` (attach/release); these tests cover the web layer's
integration with that API: exactly-once worker cleanup and AgentService
eviction when the last task releases a sandbox.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from xagent.core.execution_scope import ExecutionScope
from xagent.web.api.chat import AgentServiceManager
from xagent.web.sandbox_manager import SandboxCapacityError, SandboxManager


class _FakeAgentService:
    def __init__(
        self,
        *,
        result: dict | None = None,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.result = result or {"success": True}
        self.started = started
        self.release = release

    async def execute_task(self, **_kwargs):
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return self.result


def _make_sandbox_manager() -> SandboxManager:
    service = AsyncMock()
    service.list_sandboxes = AsyncMock(return_value=[])
    service.delete = AsyncMock()
    return SandboxManager(service)


async def _seed_attached_provider(
    sandbox_mgr: SandboxManager, owner_id: int, provider: object
) -> None:
    """Cache a lease provider and attach one active task for an owner."""
    sandbox_mgr._lease_providers[f"user::{owner_id}"] = provider
    assert await sandbox_mgr.attach("user", str(owner_id))


@pytest.fixture
def sandbox_mgr(monkeypatch) -> SandboxManager:
    manager = _make_sandbox_manager()
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: manager,
    )
    return manager


@pytest.mark.asyncio
async def test_worker_cleanup_does_not_block_other_users(sandbox_mgr) -> None:
    """Slow worker cleanup for one user must not serialize other users."""
    manager = AgentServiceManager()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def slow_list_sandboxes() -> list:
        cleanup_started.set()
        await cleanup_release.wait()
        return []

    sandbox_mgr._service.list_sandboxes.side_effect = slow_list_sandboxes
    await _seed_attached_provider(sandbox_mgr, 7, AsyncMock())
    sandbox_mgr._lease_providers["user::8"] = AsyncMock()
    manager._agent_owner_ids[2] = 8
    manager._agent_sandbox_keys[2] = "user:8"

    release_task = asyncio.create_task(manager._release_sandbox_task("user:7"))
    await cleanup_started.wait()

    try:
        sandbox_key = await asyncio.wait_for(
            manager._acquire_sandbox_task("2"),
            timeout=0.05,
        )
    finally:
        cleanup_release.set()
        await release_task

    assert sandbox_key == "user:8"
    assert sandbox_mgr.ref_count("user", "8") == 1


@pytest.mark.asyncio
async def test_worker_cleanup_removes_sandbox_lifecycle_lock_entry(
    sandbox_mgr,
) -> None:
    """Lifecycle lock entries should not leak after a provider is removed."""
    manager = AgentServiceManager()
    await _seed_attached_provider(sandbox_mgr, 7, AsyncMock())

    await manager._release_sandbox_task("user:7")

    assert sandbox_mgr._lifecycle_locks == {}
    assert "user::7" not in sandbox_mgr._lease_providers


@pytest.mark.asyncio
async def test_worker_cleanup_blocks_same_user_provider_recreate(sandbox_mgr) -> None:
    """The same user must not recreate same-named workers during cleanup."""
    manager = AgentServiceManager()
    new_provider = AsyncMock()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    create_called = asyncio.Event()

    async def slow_list_sandboxes() -> list:
        cleanup_started.set()
        await cleanup_release.wait()
        return []

    async def create_provider(*_args, **_kwargs):
        create_called.set()
        return new_provider

    sandbox_mgr._service.list_sandboxes.side_effect = slow_list_sandboxes
    sandbox_mgr.create_lease_provider = AsyncMock(side_effect=create_provider)
    await _seed_attached_provider(sandbox_mgr, 7, AsyncMock())

    release_task = asyncio.create_task(manager._release_sandbox_task("user:7"))
    await cleanup_started.wait()

    create_task = asyncio.create_task(
        manager._get_or_create_task_sandbox(
            task_id=3,
            workspace_owner_id=7,
            workspace_config={},
        )
    )

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(create_called.wait(), timeout=0.05)

        cleanup_release.set()
        sandbox = await asyncio.wait_for(create_task, timeout=0.5)
        await release_task
    finally:
        cleanup_release.set()
        await asyncio.gather(release_task, create_task, return_exceptions=True)

    assert sandbox is new_provider
    sandbox_mgr.create_lease_provider.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_task_releases_sandbox_workers_after_task(sandbox_mgr) -> None:
    """Terminal task execution should release cached worker sandboxes."""
    manager = AgentServiceManager()
    agent_service = _FakeAgentService()
    sandbox_mgr._lease_providers["user::7"] = AsyncMock()
    sandbox_mgr.delete_worker_sandboxes = AsyncMock()
    manager._agents[1] = agent_service
    manager._agent_owner_ids[1] = 7
    manager._agent_sandbox_keys[1] = "user:7"

    result = await manager.execute_task(
        agent_service=agent_service,
        task="run",
        task_id="1",
        manage_task_lease=False,
    )

    assert result == {"success": True}
    assert "user::7" not in sandbox_mgr._lease_providers
    sandbox_mgr.delete_worker_sandboxes.assert_awaited_once_with("user", "7")


@pytest.mark.asyncio
async def test_execute_task_evicts_agents_for_released_sandbox_provider(
    sandbox_mgr,
) -> None:
    """Cached agents must not retain a provider after its sandbox is released."""
    manager = AgentServiceManager()
    agent_service = _FakeAgentService()
    other_agent_service = _FakeAgentService()
    sandbox_mgr._lease_providers["user::7"] = AsyncMock()
    manager._agents[1] = agent_service
    manager._agents[2] = other_agent_service
    manager._agent_owner_ids[1] = 7
    manager._agent_owner_ids[2] = 7
    manager._agent_sandbox_keys[1] = "user:7"
    manager._agent_sandbox_keys[2] = "user:7"

    await manager.execute_task(
        agent_service=agent_service,
        task="run",
        task_id="1",
        manage_task_lease=False,
    )

    assert "user::7" not in sandbox_mgr._lease_providers
    assert 1 not in manager._agents
    assert 2 not in manager._agents
    assert 1 not in manager._agent_owner_ids
    assert 2 not in manager._agent_owner_ids
    assert 1 not in manager._agent_sandbox_keys
    assert 2 not in manager._agent_sandbox_keys


@pytest.mark.asyncio
async def test_execute_task_keeps_workers_until_last_same_user_task_finishes(
    sandbox_mgr,
) -> None:
    """One task finishing must not delete workers still shared by another task."""
    manager = AgentServiceManager()
    first_agent_service = _FakeAgentService()
    second_agent_service = _FakeAgentService()
    sandbox_mgr._lease_providers["user::7"] = AsyncMock()
    sandbox_mgr.delete_worker_sandboxes = AsyncMock()
    manager._agents[1] = first_agent_service
    manager._agents[2] = second_agent_service
    manager._agent_owner_ids[1] = 7
    manager._agent_owner_ids[2] = 7
    manager._agent_sandbox_keys[1] = "user:7"
    manager._agent_sandbox_keys[2] = "user:7"

    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    first_agent_service.started = first_started
    first_agent_service.release = first_release
    second_agent_service.started = second_started
    second_agent_service.release = second_release

    first_task = asyncio.create_task(
        manager.execute_task(
            agent_service=first_agent_service,
            task="first",
            task_id="1",
            manage_task_lease=False,
        )
    )
    await first_started.wait()

    second_task = asyncio.create_task(
        manager.execute_task(
            agent_service=second_agent_service,
            task="second",
            task_id="2",
            manage_task_lease=False,
        )
    )
    await second_started.wait()

    first_release.set()
    await first_task
    assert "user::7" in sandbox_mgr._lease_providers
    assert 1 in manager._agents
    assert 2 in manager._agents
    sandbox_mgr.delete_worker_sandboxes.assert_not_awaited()

    second_release.set()
    await second_task
    assert "user::7" not in sandbox_mgr._lease_providers
    assert 1 not in manager._agents
    assert 2 not in manager._agents
    sandbox_mgr.delete_worker_sandboxes.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_with_reclaimed_provider_raises_and_evicts_agent(
    sandbox_mgr,
) -> None:
    """An agent built with a provider that was since reclaimed must fail
    clearly (and drop its stale cache) instead of running against deleted
    containers."""
    manager = AgentServiceManager()
    manager._agents[1] = AsyncMock()
    manager._agent_owner_ids[1] = 7
    manager._agent_sandbox_keys[1] = "user:7"

    with pytest.raises(RuntimeError, match="reclaimed before"):
        await manager._acquire_sandbox_task("1")

    assert 1 not in manager._agents
    assert 1 not in manager._agent_owner_ids
    assert 1 not in manager._agent_sandbox_keys
    assert sandbox_mgr.ref_count("user", "7") == 0


@pytest.mark.asyncio
async def test_acquire_without_recorded_key_returns_none(
    sandbox_mgr,
) -> None:
    """A task with no recorded sandbox key (e.g. local execution) returns
    None, never raises. The key is never re-derived from the owner id —
    the owner-only reconstruction fallback was removed with #757 scoping,
    since it would silently miss a scope-suffixed sandbox."""
    manager = AgentServiceManager()
    manager._agent_owner_ids[1] = 7

    assert await manager._acquire_sandbox_task("1") is None
    assert sandbox_mgr.ref_count("user", "7") == 0


@pytest.mark.asyncio
async def test_capacity_error_rejects_task_by_default(sandbox_mgr, monkeypatch) -> None:
    """Capacity exhaustion must reject the task, not silently run locally."""
    monkeypatch.delenv("XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY", raising=False)
    manager = AgentServiceManager()
    sandbox_mgr.get_or_create_lease_provider = AsyncMock(
        side_effect=SandboxCapacityError(cap=2, in_use=2)
    )

    with pytest.raises(SandboxCapacityError):
        await manager._get_or_create_task_sandbox(
            task_id=1,
            workspace_owner_id=7,
            workspace_config={},
        )

    assert 1 not in manager._agent_sandbox_keys


@pytest.mark.asyncio
async def test_capacity_error_falls_back_locally_when_enabled(
    sandbox_mgr, monkeypatch
) -> None:
    """The explicit opt-in restores local fallback under capacity pressure."""
    monkeypatch.setenv("XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY", "true")
    manager = AgentServiceManager()
    sandbox_mgr.get_or_create_lease_provider = AsyncMock(
        side_effect=SandboxCapacityError(cap=2, in_use=2)
    )

    sandbox = await manager._get_or_create_task_sandbox(
        task_id=1,
        workspace_owner_id=7,
        workspace_config={},
    )

    assert sandbox is None
    assert 1 not in manager._agent_sandbox_keys


@pytest.mark.asyncio
async def test_sandbox_unavailability_keeps_local_fallback(
    sandbox_mgr, monkeypatch
) -> None:
    """Non-capacity sandbox failures keep today's local-execution fallback."""
    monkeypatch.delenv("XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY", raising=False)
    manager = AgentServiceManager()
    sandbox_mgr.get_or_create_lease_provider = AsyncMock(
        side_effect=RuntimeError("docker daemon unreachable")
    )

    sandbox = await manager._get_or_create_task_sandbox(
        task_id=1,
        workspace_owner_id=7,
        workspace_config={},
    )

    assert sandbox is None
    assert 1 not in manager._agent_sandbox_keys


_SCOPE = ExecutionScope(
    sandbox_key_suffix="client-3",
    workspace_segments=("clients", "3", "end_users", "7"),
)

# A scope that sets no ``sandbox_key_suffix`` resolves to the same
# ``user:{owner}`` lifecycle key as unscoped execution, so it has no container
# of its own to protect and must keep the unscoped fallback behavior.
_SUFFIXLESS_SCOPE = ExecutionScope(
    workspace_segments=("clients", "3", "end_users", "7"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback_flag, side_effect, expected_exc",
    [
        pytest.param(
            "true",
            SandboxCapacityError(cap=2, in_use=2),
            SandboxCapacityError,
            id="capacity-with-flag-on",
        ),
        pytest.param(
            None,
            RuntimeError("docker daemon unreachable"),
            RuntimeError,
            id="non-capacity-outage",
        ),
    ],
)
async def test_suffix_scoped_execution_fails_closed(
    sandbox_mgr, monkeypatch, fallback_flag, side_effect, expected_exc
) -> None:
    """A suffix-scoped task never takes the local fallback.

    Local fallback would run an isolated/untrusted scoped workload outside its
    container, so both capacity exhaustion (even with the opt-in flag on) and a
    non-capacity sandbox outage must fail closed rather than downgrade to
    local execution.
    """
    if fallback_flag is None:
        monkeypatch.delenv(
            "XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY", raising=False
        )
    else:
        monkeypatch.setenv(
            "XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY", fallback_flag
        )
    manager = AgentServiceManager()
    sandbox_mgr.get_or_create_lease_provider = AsyncMock(side_effect=side_effect)

    with pytest.raises(expected_exc):
        await manager._get_or_create_task_sandbox(
            task_id=1,
            workspace_owner_id=7,
            workspace_config={},
            scope=_SCOPE,
        )

    assert 1 not in manager._agent_sandbox_keys


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback_flag, side_effect",
    [
        pytest.param(
            "true", SandboxCapacityError(cap=2, in_use=2), id="capacity-with-flag-on"
        ),
        pytest.param(
            None, RuntimeError("docker daemon unreachable"), id="non-capacity-outage"
        ),
    ],
)
async def test_suffixless_scope_keeps_unscoped_fallback(
    sandbox_mgr, monkeypatch, fallback_flag, side_effect
) -> None:
    """A scope without a suffix behaves exactly like unscoped execution.

    It shares the ``user:{owner}`` container, so there is no scope-owned
    isolation to protect: capacity fallback (when the flag is on) and the
    non-capacity local fallback both remain available. Gating fail-closed on
    ``sandbox_key_suffix`` — not scope-presence — is what keeps this path from
    becoming an availability regression.
    """
    if fallback_flag is None:
        monkeypatch.delenv(
            "XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY", raising=False
        )
    else:
        monkeypatch.setenv(
            "XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY", fallback_flag
        )
    manager = AgentServiceManager()
    sandbox_mgr.get_or_create_lease_provider = AsyncMock(side_effect=side_effect)

    sandbox = await manager._get_or_create_task_sandbox(
        task_id=1,
        workspace_owner_id=7,
        workspace_config={},
        scope=_SUFFIXLESS_SCOPE,
    )

    assert sandbox is None
    assert 1 not in manager._agent_sandbox_keys
