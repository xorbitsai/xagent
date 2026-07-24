from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.config import (
    MCPToolLoadSummary,
    MCPUnavailableSummary,
    RequiredMCPUnavailableError,
)
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)


class NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


class RefreshingToolConfig:
    _workspace_config = None

    def __init__(self, allowed_tools: list[str] | None = None) -> None:
        self.refresh_count = 0
        self.allowed_tools = allowed_tools

    def refresh_user_tool_overrides(self) -> None:
        self.refresh_count += 1

    def get_user_tool_overrides(self) -> dict[str, dict[str, bool]]:
        return {"disabled": {"enabled": self.refresh_count < 2}}

    def get_allowed_tools(self) -> list[str] | None:
        return self.allowed_tools


class AsyncRefreshingToolConfig(RefreshingToolConfig):
    def __init__(self) -> None:
        super().__init__()
        self.async_refresh_count = 0
        self.sync_refresh_count = 0

    async def refresh_runtime_policy(self) -> None:
        self.async_refresh_count += 1

    def refresh_user_tool_overrides(self) -> None:
        self.sync_refresh_count += 1
        raise AssertionError("sync policy refresh ran on the event loop")

    def get_user_tool_overrides(self) -> dict[str, dict[str, bool]]:
        return {"disabled": {"enabled": self.async_refresh_count < 2}}


class AllowedToolsConfig:
    _workspace_config = None

    def __init__(self, allowed_tools: list[str] | None = None) -> None:
        self.allowed_tools = allowed_tools

    def get_user_tool_overrides(self) -> dict[str, dict[str, bool]]:
        return {}

    def get_allowed_tools(self) -> list[str] | None:
        return self.allowed_tools


class SnapshotRefreshingToolConfig(AllowedToolsConfig):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_prepared = False
        self.release_count = 0

    async def refresh_runtime_policy(self) -> None:
        self.snapshot_prepared = True

    def release_prepared_factory_runtime(self) -> None:
        self.snapshot_prepared = False
        self.release_count += 1


class DelegationRuntimeConfig:
    _workspace_config = None

    def __init__(self) -> None:
        self.parent_task_id = "parent-task-1"
        self.parent_tracer = object()

    def get_user_tool_overrides(self) -> dict[str, dict[str, bool]]:
        return {}

    def get_allowed_tools(self) -> list[str] | None:
        return None

    def get_allowed_agent_ids(self) -> list[int] | None:
        return None

    def get_agent_tool_overrides(self) -> dict[int, dict[str, Any]]:
        return {}

    def get_enable_global_agent_tools(self) -> bool:
        return True

    def get_allow_cross_user_agent_ids(self) -> bool:
        return False

    def get_parent_task_id(self) -> str:
        return self.parent_task_id

    def get_parent_tracer(self) -> object:
        return self.parent_tracer

    def get_agent_call_stack(self) -> list[int]:
        return []


@pytest.mark.asyncio
async def test_agent_service_refreshes_initialized_tools_when_policy_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = RefreshingToolConfig()
    tool_sets: list[list[Any]] = [
        [NamedTool("allowed"), NamedTool("disabled")],
        [NamedTool("allowed")],
    ]

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        return tool_sets.pop(0)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="tool-refresh-test",
        id="tool-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    assert {tool.name for tool in service.tools} == {"allowed", "disabled"}

    await service._ensure_tools_initialized()
    assert {tool.name for tool in service.tools} == {"allowed"}
    assert tool_config.refresh_count == 2


@pytest.mark.asyncio
async def test_agent_service_awaits_async_runtime_policy_refresh(monkeypatch) -> None:
    tool_config = AsyncRefreshingToolConfig()
    tool_sets: list[list[Any]] = [
        [NamedTool("allowed"), NamedTool("disabled")],
        [NamedTool("allowed")],
    ]

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        return tool_sets.pop(0)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )
    service = AgentService(
        name="async-tool-refresh-test",
        id="async-tool-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    await service._ensure_tools_initialized()

    assert tool_config.async_refresh_count == 2
    assert tool_config.sync_refresh_count == 0
    assert {tool.name for tool in service.tools} == {"allowed"}


@pytest.mark.asyncio
async def test_unchanged_policy_releases_prepared_factory_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = SnapshotRefreshingToolConfig()

    async def unexpected_rebuild(config: Any) -> list[Any]:
        raise AssertionError(f"unexpected tool rebuild for {config!r}")

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        unexpected_rebuild,
    )
    service = AgentService(
        name="unchanged-tool-policy-test",
        id="unchanged-tool-policy-test",
        tools=[NamedTool("existing")],
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()

    assert tool_config.release_count == 1
    assert not tool_config.snapshot_prepared
    assert [tool.name for tool in service.tools] == ["existing"]


@pytest.mark.asyncio
async def test_web_tool_policy_refresh_only_prepares_full_runtime_for_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import QueuePool

    from xagent.core.tools.adapters.vibe.factory import ToolFactory
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
    from xagent.web.models.user import User
    from xagent.web.services.tool_credentials import (
        set_user_tool_allowlist_hook,
        set_user_tool_overrides_hook,
    )
    from xagent.web.tools import config as web_tool_config_module
    from xagent.web.tools.config import WebToolConfig

    engine = create_engine(
        f"sqlite:///{tmp_path / 'agent-tool-policy.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
        connect_args={"check_same_thread": False},
    )
    User.__table__.create(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with session_factory() as db:
        user = User(username="policy-user", password_hash="hash", is_admin=False)
        db.add(user)
        db.commit()
        user_id = int(user.id)

    policy = {"enabled": True}
    hook_calls = 0

    def load_overrides(_db, _user):
        nonlocal hook_calls
        hook_calls += 1
        return {"calculator": {"enabled": policy["enabled"]}}

    set_user_tool_overrides_hook(load_overrides)
    set_user_tool_allowlist_hook(lambda _db, _user: None)

    full_load_calls = 0
    real_full_loader = web_tool_config_module._load_tool_factory_runtime_snapshot

    def count_full_load(*args, **kwargs):
        nonlocal full_load_calls
        full_load_calls += 1
        return real_full_loader(*args, **kwargs)

    rebuild_calls = 0

    async def build_tools(config, apply_user_override_filter=True):
        nonlocal rebuild_calls
        rebuild_calls += 1
        return [NamedTool("calculator")]

    monkeypatch.setattr(
        web_tool_config_module,
        "_load_tool_factory_runtime_snapshot",
        count_full_load,
    )
    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)

    config = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=user_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=[]),
    )
    config._cached_tool_overrides = {"calculator": {"enabled": True}}
    config._cached_tool_allowlist = None
    config._tool_allowlist_cached = True
    service = AgentService(
        name="web-policy-refresh-test",
        id="web-policy-refresh-test",
        tools=[NamedTool("calculator")],
        tool_config=config,
        enable_workspace=False,
    )

    try:
        await service._ensure_tools_initialized()
        assert hook_calls == 1
        assert full_load_calls == 0
        assert rebuild_calls == 0

        policy["enabled"] = False
        await service._ensure_tools_initialized()
        assert hook_calls == 2
        assert full_load_calls == 1
        assert rebuild_calls == 1

        await service._ensure_tools_initialized()
        assert hook_calls == 3
        assert full_load_calls == 1
        assert rebuild_calls == 1
    finally:
        config.close()
        set_user_tool_overrides_hook(None)
        set_user_tool_allowlist_hook(None)
        engine.dispose()


@pytest.mark.asyncio
async def test_agent_service_rebuild_reuses_mcp_summary_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SummaryObservingConfig(RefreshingToolConfig):
        def __init__(self) -> None:
            super().__init__()
            self.summaries: list[MCPToolLoadSummary] = []

        async def emit_mcp_load_summary(self, summary: MCPToolLoadSummary) -> None:
            self.summaries.append(summary)

    tool_config = SummaryObservingConfig()
    observed_config_ids: list[int] = []

    async def create_all_tools(config: Any) -> list[Any]:
        observed_config_ids.append(id(config))
        await config.emit_mcp_load_summary(MCPToolLoadSummary())
        return [NamedTool("allowed")]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )
    service = AgentService(
        name="mcp-summary-observer-refresh-test",
        id="mcp-summary-observer-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    await service._ensure_tools_initialized()

    assert observed_config_ids == [id(tool_config), id(tool_config)]
    assert tool_config.summaries == [MCPToolLoadSummary(), MCPToolLoadSummary()]


@pytest.mark.asyncio
async def test_agent_service_refreshes_when_allowed_tools_changes_from_all_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = AllowedToolsConfig(allowed_tools=None)

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        return [NamedTool("allowed"), NamedTool("disabled")]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="allowed-tools-refresh-test",
        id="allowed-tools-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    assert {tool.name for tool in service.tools} == {"allowed", "disabled"}

    tool_config.allowed_tools = []
    await service._ensure_tools_initialized()
    assert service.tools == []


@pytest.mark.asyncio
async def test_agent_service_refreshes_when_delegation_parent_context_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = DelegationRuntimeConfig()
    tool_sets: list[list[Any]] = [
        [NamedTool("initial")],
        [NamedTool("new-task")],
        [NamedTool("new-tracer")],
    ]

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        return tool_sets.pop(0)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="delegation-runtime-refresh-test",
        id="delegation-runtime-refresh-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    await service._ensure_tools_initialized()
    assert [tool.name for tool in service.tools] == ["initial"]

    tool_config.parent_task_id = "parent-task-2"
    await service._ensure_tools_initialized()
    assert [tool.name for tool in service.tools] == ["new-task"]

    tool_config.parent_tracer = object()
    await service._ensure_tools_initialized()
    assert [tool.name for tool in service.tools] == ["new-tracer"]


@pytest.mark.asyncio
async def test_agent_service_preserves_connector_runtime_initialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = AllowedToolsConfig()
    runtime_error = ConnectorRuntimeError(
        ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
        "Connector runtime context is unavailable.",
        details={"reason": "runtime_view_resolution_failed"},
        status_code=503,
    )

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        raise runtime_error

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )

    service = AgentService(
        name="runtime-error-test",
        id="runtime-error-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        await service._ensure_tools_initialized()

    assert exc_info.value is runtime_error
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_agent_service_preserves_required_mcp_initialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_config = AllowedToolsConfig()
    required_error = RequiredMCPUnavailableError(
        [MCPUnavailableSummary.from_values("Gmail", "oauth_token_required")]
    )

    async def create_all_tools(config: Any) -> list[Any]:
        assert config is tool_config
        raise required_error

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )
    service = AgentService(
        name="required-mcp-error-test",
        id="required-mcp-error-test",
        tool_config=tool_config,
        enable_workspace=False,
    )

    with pytest.raises(RequiredMCPUnavailableError) as exc_info:
        await service._ensure_tools_initialized()

    assert exc_info.value is required_error
