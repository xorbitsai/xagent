"""Slice 2 of #757: scope-aware sandbox lifecycle keys.

Covers the single make/parse helper pair for sandbox lifecycle keys, scoped
sandbox acquisition (``user:{owner}:{suffix}``), worker-scope inheritance,
scoped keys in the eviction sweep, the removed owner-key attach fallback,
and the AgentService cache's scope-fingerprint eviction (including the
A -> B -> A flap warning) driven through the resolver path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import xagent.web.sandbox_manager as sandbox_manager_module
from tests.web.sandbox_fakes import FakeSandboxService
from xagent.core.execution_scope import (
    ExecutionScope,
    InvalidScopeComponentError,
    set_execution_scope_resolver,
)
from xagent.sandbox.base import SandboxMountIntent
from xagent.web.api.chat import AgentServiceManager
from xagent.web.sandbox_keys import (
    make_user_lifecycle_id,
    make_user_sandbox_key,
    parse_user_lifecycle_id,
    parse_user_sandbox_key,
)
from xagent.web.sandbox_manager import SandboxManager


@pytest.fixture(autouse=True)
def _clear_resolver():
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


class TestSandboxKeyHelpers:
    def test_unscoped_key_is_byte_identical_to_legacy_format(self):
        assert make_user_sandbox_key(7) == "user:7"
        assert make_user_lifecycle_id(7) == "7"

    def test_scoped_key_appends_suffix(self):
        assert make_user_sandbox_key(7, "tenant-a") == "user:7:tenant-a"
        assert make_user_lifecycle_id(7, "tenant-a") == "7:tenant-a"

    def test_round_trip(self):
        assert parse_user_sandbox_key(make_user_sandbox_key(7)) == (7, None)
        assert parse_user_sandbox_key(make_user_sandbox_key(7, "t-a")) == (7, "t-a")
        assert parse_user_lifecycle_id(make_user_lifecycle_id(7, "t-a")) == (7, "t-a")
        assert parse_user_lifecycle_id(make_user_lifecycle_id(7)) == (7, None)

    def test_make_rejects_invalid_suffix(self):
        with pytest.raises(InvalidScopeComponentError):
            make_user_sandbox_key(7, "a:b")
        with pytest.raises(InvalidScopeComponentError):
            make_user_lifecycle_id(7, "")

    @pytest.mark.parametrize(
        "key",
        ["user7", "task:7", "user:x", "user:7:", "user:", ""],
    )
    def test_parse_key_rejects_malformed(self, key):
        with pytest.raises(ValueError):
            parse_user_sandbox_key(key)

    @pytest.mark.parametrize("lifecycle_id", ["x", "7:", ""])
    def test_parse_lifecycle_id_rejects_malformed(self, lifecycle_id):
        with pytest.raises(ValueError):
            parse_user_lifecycle_id(lifecycle_id)

    def test_only_owner_segment_is_int_parsed(self):
        """A scoped key parses instead of raising — the historic bug was
        ``int("7:suffix")`` blowing up single-split parse sites."""
        owner, suffix = parse_user_sandbox_key("user:7:suffix_with_9")
        assert owner == 7
        assert suffix == "suffix_with_9"


def _make_sandbox_manager() -> SandboxManager:
    service = FakeSandboxService()
    service.list_sandboxes = AsyncMock(return_value=[])
    service.delete = AsyncMock()
    return SandboxManager(service)


@pytest.fixture
def sandbox_mgr(monkeypatch) -> SandboxManager:
    manager = _make_sandbox_manager()
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: manager,
    )
    return manager


class TestScopedSandboxAcquisition:
    @pytest.mark.asyncio
    async def test_unscoped_acquisition_is_unchanged(self, sandbox_mgr) -> None:
        manager = AgentServiceManager()
        provider = AsyncMock()
        sandbox_mgr.get_or_create_lease_provider = AsyncMock(return_value=provider)

        sandbox = await manager._get_or_create_task_sandbox(
            task_id=1, workspace_owner_id=7, mount_intent=None
        )

        assert sandbox is provider
        sandbox_mgr.get_or_create_lease_provider.assert_awaited_once_with(
            "user", "7", mount_intent=None, prepare_root=None
        )
        assert manager._agent_sandbox_keys[1] == "user:7"

    @pytest.mark.asyncio
    async def test_scoped_acquisition_composes_suffixed_key(self, sandbox_mgr) -> None:
        manager = AgentServiceManager()
        provider = AsyncMock()
        sandbox_mgr.get_or_create_lease_provider = AsyncMock(return_value=provider)
        scope = ExecutionScope(sandbox_key_suffix="tenant-a")

        sandbox = await manager._get_or_create_task_sandbox(
            task_id=1, workspace_owner_id=7, mount_intent=None, scope=scope
        )

        assert sandbox is provider
        sandbox_mgr.get_or_create_lease_provider.assert_awaited_once_with(
            "user", "7:tenant-a", mount_intent=None, prepare_root=None
        )
        assert manager._agent_sandbox_keys[1] == "user:7:tenant-a"

    @pytest.mark.asyncio
    async def test_scope_without_suffix_stays_unscoped(self, sandbox_mgr) -> None:
        """Fields are consumed independently: a scope carrying only other
        fields must not change the sandbox key."""
        manager = AgentServiceManager()
        sandbox_mgr.get_or_create_lease_provider = AsyncMock(return_value=AsyncMock())
        scope = ExecutionScope(workspace_segments=("proj",))

        await manager._get_or_create_task_sandbox(
            task_id=1, workspace_owner_id=7, mount_intent=None, scope=scope
        )

        sandbox_mgr.get_or_create_lease_provider.assert_awaited_once_with(
            "user", "7", mount_intent=None, prepare_root=None
        )
        assert manager._agent_sandbox_keys[1] == "user:7"

    @pytest.mark.asyncio
    async def test_two_scopes_under_one_user_get_disjoint_keys(
        self, sandbox_mgr
    ) -> None:
        manager = AgentServiceManager()
        sandbox_mgr.get_or_create_lease_provider = AsyncMock(return_value=AsyncMock())

        await manager._get_or_create_task_sandbox(
            task_id=1,
            workspace_owner_id=7,
            mount_intent=None,
            scope=ExecutionScope(sandbox_key_suffix="tenant-a"),
        )
        await manager._get_or_create_task_sandbox(
            task_id=2,
            workspace_owner_id=7,
            mount_intent=None,
            scope=ExecutionScope(sandbox_key_suffix="tenant-b"),
        )

        lifecycle_ids = [
            call.args[1]
            for call in sandbox_mgr.get_or_create_lease_provider.await_args_list
        ]
        assert lifecycle_ids == ["7:tenant-a", "7:tenant-b"]
        assert manager._agent_sandbox_keys[1] != manager._agent_sandbox_keys[2]


class TestOwnerFallbackRemoved:
    @pytest.mark.asyncio
    async def test_no_recorded_key_never_attaches_even_with_live_provider(
        self, sandbox_mgr
    ) -> None:
        """The owner-only key reconstruction fallback is gone: a task with
        no recorded sandbox key must not attach to the owner's provider —
        under a scoped key the reconstructed ``user:{owner}`` would target
        the wrong container family."""
        manager = AgentServiceManager()
        sandbox_mgr._lease_providers["user::7"] = AsyncMock()
        manager._agents[1] = AsyncMock()
        manager._agent_owner_ids[1] = 7  # owner known, but no recorded key

        assert await manager._acquire_sandbox_task("1") is None
        assert sandbox_mgr.ref_count("user", "7") == 0
        # No eviction either: the agent runs locally, as it was built.
        assert 1 in manager._agents

    @pytest.mark.asyncio
    async def test_scoped_recorded_key_attaches_to_scoped_lifecycle(
        self, sandbox_mgr
    ) -> None:
        manager = AgentServiceManager()
        sandbox_mgr._lease_providers["user::7:tenant-a"] = AsyncMock()
        manager._agent_sandbox_keys[1] = "user:7:tenant-a"

        key = await manager._acquire_sandbox_task("1")

        assert key == "user:7:tenant-a"
        assert sandbox_mgr.ref_count("user", "7:tenant-a") == 1
        assert sandbox_mgr.ref_count("user", "7") == 0

    @pytest.mark.asyncio
    async def test_release_of_scoped_key_spares_unscoped_agents(
        self, sandbox_mgr
    ) -> None:
        """Scoped and unscoped containers under one owner are disjoint
        namespaces: releasing one family must not evict the other's cached
        agents."""
        manager = AgentServiceManager()
        sandbox_mgr._lease_providers["user::7:tenant-a"] = AsyncMock()
        sandbox_mgr._lease_providers["user::7"] = AsyncMock()
        assert await sandbox_mgr.attach("user", "7:tenant-a")
        manager._agents[1] = AsyncMock()
        manager._agent_sandbox_keys[1] = "user:7:tenant-a"
        manager._agents[2] = AsyncMock()
        manager._agent_sandbox_keys[2] = "user:7"
        manager._agents[3] = AsyncMock()
        manager._agent_owner_ids[3] = 7  # local execution, no key

        await manager._release_sandbox_task("user:7:tenant-a")

        assert 1 not in manager._agents
        assert 2 in manager._agents  # unscoped family untouched
        assert 3 in manager._agents  # keyless (local) agent untouched
        assert "user::7" in sandbox_mgr._lease_providers


TTL = 100.0


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _listed(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, state="stopped")


@pytest.fixture
def clock(monkeypatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(sandbox_manager_module, "time", fake)
    return fake


class TestSandboxManagerScopedKeys:
    @pytest.mark.asyncio
    async def test_two_scopes_create_two_distinct_containers(self) -> None:
        manager = _make_sandbox_manager()
        created: list[str] = []

        async def fake_get_or_create(name, **_kwargs):
            created.append(name)
            return MagicMock()

        manager._service.get_or_create = AsyncMock(side_effect=fake_get_or_create)

        await manager.get_or_create_lease_provider("user", "7:tenant-a")
        await manager.get_or_create_lease_provider("user", "7:tenant-b")
        await manager.get_or_create_lease_provider("user", "7")

        assert created == ["user::7:tenant-a", "user::7:tenant-b", "user::7"]

    @pytest.mark.asyncio
    async def test_worker_sandboxes_inherit_parent_scope(self) -> None:
        """Workers derive their lifecycle id from the provider's, so they
        land in the parent scope's container family and their activity maps
        back to the scoped primary."""
        manager = _make_sandbox_manager()
        created: list[str] = []

        async def fake_get_or_create(name, **_kwargs):
            created.append(name)
            return MagicMock()

        manager._service.get_or_create = AsyncMock(side_effect=fake_get_or_create)

        provider = await manager.get_or_create_lease_provider("user", "7:tenant-a")
        worker = await provider.get_worker_sandbox(0)

        assert worker is not None
        assert created == ["user::7:tenant-a", "user::7:tenant-a::worker::0"]
        assert (
            SandboxManager._base_sandbox_name("user", "7:tenant-a::worker::0")
            == "user::7:tenant-a"
        )

    @pytest.mark.asyncio
    async def test_idle_sweep_reclaims_scoped_sandbox(self, clock) -> None:
        """A scoped key is an ordinary lifecycle key to the sweep — the
        historic single-split-then-int parse would have raised on it and
        silently skipped reclamation."""
        manager = _make_sandbox_manager()
        manager._service.list_sandboxes = AsyncMock(
            return_value=[
                _listed("user::7:tenant-a"),
                _listed("user::7:tenant-a::worker::0"),
            ]
        )
        manager._lease_providers["user::7:tenant-a"] = MagicMock()

        clock.advance(TTL + 1)
        reclaimed = await manager.sweep_idle_sandboxes(TTL)

        assert reclaimed == ["user::7:tenant-a"]
        deleted = {call.args[0] for call in manager._service.delete.await_args_list}
        assert deleted == {"user::7:tenant-a", "user::7:tenant-a::worker::0"}
        assert "user::7:tenant-a" not in manager._lease_providers

    @pytest.mark.asyncio
    async def test_active_scoped_sandbox_survives_sweep(self, clock) -> None:
        manager = _make_sandbox_manager()
        manager._service.list_sandboxes = AsyncMock(
            return_value=[_listed("user::7:tenant-a")]
        )
        manager._lease_providers["user::7:tenant-a"] = MagicMock()
        assert await manager.attach("user", "7:tenant-a")

        clock.advance(TTL * 10)
        reclaimed = await manager.sweep_idle_sandboxes(TTL)

        assert reclaimed == []
        manager._service.delete.assert_not_awaited()

    def test_default_workspace_mount_uses_owner_for_scoped_id(
        self, tmp_path, monkeypatch
    ) -> None:
        """The default user upload mount stays user-level for scoped
        lifecycle ids — ``user_7``, never ``user_7:tenant-a``."""
        monkeypatch.setattr(sandbox_manager_module, "get_uploads_dir", lambda: tmp_path)

        paths = SandboxManager._workspace_mount_paths("user", "7:tenant-a", None)
        assert paths == [(tmp_path / "user_7", True)]

        worker_paths = SandboxManager._workspace_mount_paths(
            "user", "7:tenant-a::worker::0", None
        )
        assert worker_paths == [(tmp_path / "user_7", True)]

        unscoped = SandboxManager._workspace_mount_paths("user", "7", None)
        assert unscoped == [(tmp_path / "user_7", True)]

    def test_ca_prefix_mount_is_shared_across_end_users(self, tmp_path) -> None:
        """#79-01: two end users of one client application supply a
        mount intent whose ``mount_root`` is the CA-level mount prefix, so
        both produce the identical mount root — the prerequisite for sharing
        one container without tripping config-equivalence."""
        ca_root = tmp_path / "user_5" / "clients" / "3"
        intent_eu7 = SandboxMountIntent(mount_root=str(ca_root))
        intent_eu8 = SandboxMountIntent(mount_root=str(ca_root))

        paths_eu7 = SandboxManager._workspace_mount_paths(
            "user", "5:client-3", intent_eu7
        )
        paths_eu8 = SandboxManager._workspace_mount_paths(
            "user", "5:client-3", intent_eu8
        )
        assert paths_eu7 == paths_eu8 == [(ca_root, True)]
