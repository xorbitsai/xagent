"""Slice 2 of #757: AgentService cache eviction on scope-fingerprint mismatch.

``get_agent_for_task`` resolves the ExecutionScope per call (same place the
owner is resolved) and compares its fingerprint against the one the cached
instance was built under. These tests drive the real ``get_agent_for_task``
through the resolver path and pin:

* a scope change between turns evicts and rebuilds (never silently reuses
  the old scope's namespace), without destroying the same-owner workspace,
* an A -> B -> A flap is logged as a probable resolver bug,
* unscoped behavior is unchanged (cached instance reused, no eviction),
* the resolved scope reaches sandbox acquisition: the rebuilt agent records
  a scope-suffixed sandbox key.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import (
    ExecutionScope,
    ExecutionScopeAuthorityError,
    ExecutionScopeContext,
    scope_fingerprint,
    set_execution_scope_snapshot_loader,
)
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import AgentServiceManager
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)

SCOPE_A = ExecutionScope(sandbox_key_suffix="tenant-a")
SCOPE_B = ExecutionScope(sandbox_key_suffix="tenant-b")


def _make_user() -> User:
    return User(id=1, username="scope-fp-user", password_hash="hash", is_admin=False)


def _make_task_row() -> Task:
    return Task(
        id=42,
        user_id=1,
        title="scope-fp task",
        description="x",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )


def _build_snapshot() -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="scope-fp-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )


def _build_db_mock(task_row: Task) -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = task_row
    return db


def _common_patches(
    manager: AgentServiceManager, *, sandbox_manager: Any = None
) -> list[Any]:
    # Environments without API keys have no default LLM; the build must get
    # past LLM resolution to reach the cache/sandbox logic under test.
    manager._default_llm = MagicMock()
    return [
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=sandbox_manager,
        ),
        patch("xagent.web.services.agent_service_manager.AgentService"),
    ]


async def _call(manager: AgentServiceManager, **kwargs: Any) -> None:
    try:
        await manager.get_agent_for_task(task_id=42, **kwargs)
    except Exception:
        # Downstream stubs may raise after the cache decision under test;
        # the assertions below inspect the cache maps directly.
        pass


@pytest.mark.asyncio
async def test_scope_change_between_turns_evicts_and_rebuilds() -> None:
    register_scope_resolver(lambda task_id: SCOPE_B)
    manager = AgentServiceManager()
    stale_agent = MagicMock()
    manager._agents[42] = stale_agent
    manager._agent_owner_ids[42] = 1
    manager._agent_sandbox_keys[42] = "user:1:tenant-a"
    manager._agent_scope_fingerprints[42] = scope_fingerprint(SCOPE_A)

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        await _call(
            manager,
            db=_build_db_mock(_make_task_row()),
            user=_make_user(),
            task_setup_snapshot=_build_snapshot(),
        )

    # The stale-scope instance is gone and the rebuild recorded the new
    # fingerprint; turn 2 must not execute in scope A's namespace.
    assert manager._agents.get(42) is not stale_agent
    assert manager._agent_scope_fingerprints.get(42) == scope_fingerprint(SCOPE_B)
    # Same owner: the workspace survives a scope reassignment.
    stale_agent.cleanup_workspace.assert_not_called()


@pytest.mark.asyncio
async def test_isolate_external_dirs_only_change_evicts_and_rebuilds() -> None:
    """(#296) isolate_external_dirs is baked into the
    cached AgentService's allowed_external_dirs at build time, not read
    fresh -- an isolate_external_dirs-only change (same sandbox suffix,
    workspace segments, mount, memory dimensions) must still evict the
    cache, or the stale allowed-dirs list keeps being enforced."""
    scope_shared = ExecutionScope(sandbox_key_suffix="tenant-a")
    scope_isolated = ExecutionScope(
        sandbox_key_suffix="tenant-a", isolate_external_dirs=True
    )
    assert scope_fingerprint(scope_shared) != scope_fingerprint(scope_isolated)

    register_scope_resolver(
        lambda task_id: scope_isolated,
    )
    manager = AgentServiceManager()
    stale_agent = MagicMock()
    manager._agents[42] = stale_agent
    manager._agent_owner_ids[42] = 1
    manager._agent_sandbox_keys[42] = "user:1:tenant-a"
    manager._agent_scope_fingerprints[42] = scope_fingerprint(scope_shared)

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        await _call(
            manager,
            db=_build_db_mock(_make_task_row()),
            user=_make_user(),
            task_setup_snapshot=_build_snapshot(),
        )

    assert manager._agents.get(42) is not stale_agent
    assert manager._agent_scope_fingerprints.get(42) == scope_fingerprint(
        scope_isolated
    )


@pytest.mark.asyncio
async def test_scope_flap_is_logged_as_probable_resolver_bug(caplog) -> None:
    """A -> B -> A: the resolver returning the fingerprint that a previous
    rebuild evicted means it flaps between values — every turn would evict
    and rebuild, silently defeating the cache."""
    register_scope_resolver(lambda task_id: SCOPE_A)
    manager = AgentServiceManager()
    manager._agents[42] = MagicMock()
    manager._agent_owner_ids[42] = 1
    manager._agent_scope_fingerprints[42] = scope_fingerprint(SCOPE_B)
    # A previous scope-mismatch rebuild evicted fingerprint A.
    manager._agent_evicted_scope_fingerprints[42] = deque(
        [scope_fingerprint(SCOPE_A)], maxlen=4
    )

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        with caplog.at_level(logging.WARNING):
            await _call(
                manager,
                db=_build_db_mock(_make_task_row()),
                user=_make_user(),
                task_setup_snapshot=_build_snapshot(),
            )

    assert any("probable resolver bug" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_multi_scope_cycle_is_also_flagged_as_resolver_bug(caplog) -> None:
    """A resolver cycling through 3+ scopes (A -> B -> C -> A -> ...)
    defeats the cache every turn just like a period-2 flap; the bounded
    recently-evicted memory must flag it too (Roger's approval-round
    finding on #789)."""
    cycle = [SCOPE_A, SCOPE_B, ExecutionScope(sandbox_key_suffix="tenant-c")]
    turn = {"n": 0}

    def cycling_resolver(task_id):
        return cycle[turn["n"] % len(cycle)]

    register_scope_resolver(cycling_resolver)
    manager = AgentServiceManager()

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        with caplog.at_level(logging.WARNING):
            for _ in range(4):  # A, B, C, then back to A
                await _call(
                    manager,
                    db=_build_db_mock(_make_task_row()),
                    user=_make_user(),
                    task_setup_snapshot=_build_snapshot(),
                )
                turn["n"] += 1

    assert any("probable resolver bug" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_activated_turn_scope_is_reused_without_re_resolving() -> None:
    """One turn resolves once: inside an activated turn context,
    get_agent_for_task consumes the contextvar scope instead of a fresh
    loader/resolver round-trip (Roger's approval-round finding on #789)."""
    resolver_calls: list[str] = []

    def resolver(task_id):
        resolver_calls.append(task_id)
        return SCOPE_A

    register_scope_resolver(resolver)
    manager = AgentServiceManager()

    with ExitStack() as stack:
        for p in _common_patches(manager):
            stack.enter_context(p)
        with ExecutionScopeContext(SCOPE_A):  # the turn's activation
            await _call(
                manager,
                db=_build_db_mock(_make_task_row()),
                user=_make_user(),
                task_setup_snapshot=_build_snapshot(),
            )

    assert resolver_calls == []
    assert manager._agent_scope_fingerprints.get(42) == scope_fingerprint(SCOPE_A)


@pytest.mark.asyncio
async def test_unscoped_cached_agent_is_reused_without_eviction() -> None:
    """No resolver -> fingerprint None on both sides -> today's behavior:
    the cached instance is returned untouched."""
    manager = AgentServiceManager()
    cached_agent = MagicMock()
    manager._agents[42] = cached_agent
    manager._agent_owner_ids[42] = 1
    manager._agent_scope_fingerprints[42] = None

    result = await manager.get_agent_for_task(
        task_id=42,
        db=_build_db_mock(_make_task_row()),
        user=_make_user(),
        task_setup_snapshot=_build_snapshot(),
    )

    assert result is cached_agent
    cached_agent.cleanup_workspace.assert_not_called()


@pytest.mark.asyncio
async def test_stable_scope_does_not_evict_between_turns() -> None:
    """An idempotent resolver returning an equal scope every turn keeps the
    cache warm — equality is by fingerprint value, not object identity."""
    register_scope_resolver(
        lambda task_id: ExecutionScope(sandbox_key_suffix="tenant-a"),
    )
    manager = AgentServiceManager()
    cached_agent = MagicMock()
    manager._agents[42] = cached_agent
    manager._agent_owner_ids[42] = 1
    manager._agent_scope_fingerprints[42] = scope_fingerprint(
        ExecutionScope(sandbox_key_suffix="tenant-a")
    )

    result = await manager.get_agent_for_task(
        task_id=42,
        db=_build_db_mock(_make_task_row()),
        user=_make_user(),
        task_setup_snapshot=_build_snapshot(),
    )

    assert result is cached_agent


class _ScopeTrackingToolConfig:
    """Minimal ``WebToolConfig`` stand-in: records every scope it is handed
    without exercising the real swap/cache-drop logic (that logic is pinned
    separately in ``tests/web/tools/test_web_tool_config_session_factory.py``).
    """

    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.set_calls: list[ExecutionScope] = []

    def set_execution_scope(self, scope: ExecutionScope) -> bool:
        self.set_calls.append(scope)
        self.scope = scope
        return True


class _CachedAgentWithToolConfig:
    def __init__(self, tool_config: _ScopeTrackingToolConfig) -> None:
        self.tool_config = tool_config
        self.invalidate_calls = 0

    def invalidate_tools(self) -> None:
        self.invalidate_calls += 1

    def cleanup_workspace(self) -> None:
        pass


@pytest.mark.asyncio
async def test_cached_agent_service_is_resynced_with_the_turn_execution_scope() -> None:
    """A same-fingerprint scope change keeps the cached AgentService (no
    rebuild), but the resolved scope for THIS turn must still reach the
    cached tool config -- otherwise a resolver that hands back a scope
    object carrying turn-varying data outside the namespace fingerprint
    (here: ``strict_memory_isolation``, which ``scope_fingerprint`` doesn't
    cover) would leave the OAuth resolver hook reading the first turn's
    scope object forever.

    Also covers the ``connector_runtime_turn_id=None`` case: unlike
    ``_sync_connector_runtime_turn``, the execution-scope resync must not be
    gated on a turn id being present.
    """
    tool_config = _ScopeTrackingToolConfig(SCOPE_A)
    cached_agent = _CachedAgentWithToolConfig(tool_config)
    manager = AgentServiceManager()
    manager._agents[42] = cached_agent
    manager._agent_owner_ids[42] = 1
    manager._agent_scope_fingerprints[42] = scope_fingerprint(SCOPE_A)

    scope_b = ExecutionScope(
        sandbox_key_suffix="tenant-a", strict_memory_isolation=True
    )
    assert scope_fingerprint(scope_b) == scope_fingerprint(SCOPE_A)
    assert scope_b != SCOPE_A

    result = await manager.get_agent_for_task(
        task_id=42,
        db=_build_db_mock(_make_task_row()),
        user=_make_user(),
        task_setup_snapshot=_build_snapshot(),
        connector_runtime_turn_id=None,
        resolved_execution_scope=scope_b,
    )

    assert result is cached_agent
    assert tool_config.scope is scope_b
    assert tool_config.set_calls == [scope_b]
    assert cached_agent.invalidate_calls == 1


@pytest.mark.asyncio
async def test_resolver_scope_reaches_sandbox_key_on_build() -> None:
    """End-to-end through the resolver path: a fresh build under a scoped
    resolver acquires the scoped container family and records the
    scope-suffixed key for execution-time attach."""
    register_scope_resolver(lambda task_id: SCOPE_A)
    manager = AgentServiceManager()
    fake_sandbox_manager = MagicMock()
    fake_sandbox_manager.get_or_create_lease_provider = AsyncMock(
        return_value=AsyncMock()
    )

    with ExitStack() as stack:
        for p in _common_patches(manager, sandbox_manager=fake_sandbox_manager):
            stack.enter_context(p)
        await _call(
            manager,
            db=_build_db_mock(_make_task_row()),
            user=_make_user(),
            task_setup_snapshot=_build_snapshot(),
        )

    fake_sandbox_manager.get_or_create_lease_provider.assert_awaited_once()
    lifecycle_args = fake_sandbox_manager.get_or_create_lease_provider.await_args.args
    assert lifecycle_args == ("user", "1:tenant-a")
    assert manager._agent_sandbox_keys.get(42) == "user:1:tenant-a"
    assert manager._agent_scope_fingerprints.get(42) == scope_fingerprint(SCOPE_A)


@pytest.mark.asyncio
async def test_resolver_scope_reaches_workspace_paths_on_build() -> None:
    """Slice 3: the resolved scope's workspace_segments reach the agent
    build — the AgentService is constructed with a scoped
    workspace_base_dir and carries the segments, and two scopes produce
    disjoint base dirs. Driven through the resolver path."""
    from xagent.config import get_uploads_dir
    from xagent.core.workspace import scoped_user_root

    scope = ExecutionScope(workspace_segments=("tenant-a",))
    register_scope_resolver(lambda task_id: scope)
    manager = AgentServiceManager()

    with ExitStack() as stack:
        # _common_patches's last entry patches AgentService anonymously;
        # patch it here instead to keep a handle on the mock's call kwargs.
        for p in _common_patches(manager)[:-1]:
            stack.enter_context(p)
        agent_service_mock = stack.enter_context(
            patch("xagent.web.services.agent_service_manager.AgentService")
        )
        await _call(
            manager,
            db=_build_db_mock(_make_task_row()),
            user=_make_user(),
            task_setup_snapshot=_build_snapshot(),
        )

    kwargs = agent_service_mock.call_args.kwargs
    expected_base = str(scoped_user_root(get_uploads_dir(), 1, ("tenant-a",)))
    assert kwargs["workspace_base_dir"] == expected_base
    assert kwargs["scope_segments"] == ("tenant-a",)
    # Default sharing: the user-level upload dir stays in the allowlist.
    assert (
        str(scoped_user_root(get_uploads_dir(), 1)) in kwargs["allowed_external_dirs"]
    )


@pytest.mark.asyncio
async def test_unscoped_build_uses_legacy_workspace_base_dir() -> None:
    """No resolver -> AgentService gets the byte-identical legacy base dir
    and empty scope segments."""
    from xagent.config import get_uploads_dir
    from xagent.core.workspace import scoped_user_root

    manager = AgentServiceManager()
    manager._default_llm = MagicMock()
    with ExitStack() as stack:
        for p in _common_patches(manager)[:-1]:
            stack.enter_context(p)
        agent_service_mock = stack.enter_context(
            patch("xagent.web.services.agent_service_manager.AgentService")
        )
        await _call(
            manager,
            db=_build_db_mock(_make_task_row()),
            user=_make_user(),
            task_setup_snapshot=_build_snapshot(),
        )

    kwargs = agent_service_mock.call_args.kwargs
    assert kwargs["workspace_base_dir"] == str(scoped_user_root(get_uploads_dir(), 1))
    assert kwargs["scope_segments"] == ()


@pytest.mark.asyncio
async def test_unscoped_build_records_legacy_key() -> None:
    """No resolver -> the build records the byte-identical legacy key."""
    manager = AgentServiceManager()
    fake_sandbox_manager = MagicMock()
    fake_sandbox_manager.get_or_create_lease_provider = AsyncMock(
        return_value=AsyncMock()
    )

    with ExitStack() as stack:
        for p in _common_patches(manager, sandbox_manager=fake_sandbox_manager):
            stack.enter_context(p)
        await _call(
            manager,
            db=_build_db_mock(_make_task_row()),
            user=_make_user(),
            task_setup_snapshot=_build_snapshot(),
        )

    lifecycle_args = fake_sandbox_manager.get_or_create_lease_provider.await_args.args
    assert lifecycle_args == ("user", "1")
    assert manager._agent_sandbox_keys.get(42) == "user:1"
    assert manager._agent_scope_fingerprints.get(42) is None


@pytest.mark.asyncio
async def test_turn_face_authority_mismatch_fails_the_turn() -> None:
    """``get_agent_for_task`` is a turn-face consumer of
    ``resolve_execution_scope``: unlike the off-turn consumers
    (``ManagedFileRef``'s construction, the pause/resume handlers), a
    namespace-affecting authority mismatch here must propagate and fail the
    turn rather than being downgraded to a warning.
    ``websocket._scope_segments_for_task`` is off-turn and still resolves
    fail-closed, so what decides this is whether a namespace is being chosen
    for new bytes, not whether a turn is running."""
    register_scope_resolver(
        lambda task_id: ExecutionScope(sandbox_key_suffix="from-resolver"),
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(sandbox_key_suffix="from-snapshot")
    )
    manager = AgentServiceManager()
    manager._default_llm = MagicMock()

    with pytest.raises(ExecutionScopeAuthorityError):
        await manager.get_agent_for_task(
            task_id=42,
            db=None,
            user=_make_user(),
            task_owner_user_id=1,
            task_setup_snapshot=_build_snapshot(),
        )
