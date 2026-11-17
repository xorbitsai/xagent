"""A rebuild shortly after a build reuses the loaded MCP tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.core.execution_scope import ExecutionScope
from xagent.core.tools.adapters.vibe import mcp_tools
from xagent.core.tools.adapters.vibe.config import (
    MCPFailurePolicy,
    MCPToolLoadSummary,
    RequiredMCPUnavailableError,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_adapter import UnavailableMCPTool


class _LoadedTool:
    source_server = "gmail"


class _Config:
    def __init__(
        self,
        servers: list[dict[str, Any]],
        sandbox: Any = None,
        policy: MCPFailurePolicy = MCPFailurePolicy.BEST_EFFORT,
    ) -> None:
        self.servers = servers
        self.sandbox = sandbox
        self.policy = policy
        self.identities: dict[str, Any] = {}
        self.summaries: list[MCPToolLoadSummary] = []

    def get_tool_selection_spec(self) -> None:
        return None

    async def get_mcp_server_configs(self) -> list[dict[str, Any]]:
        return self.servers

    def get_sandbox(self) -> Any:
        return self.sandbox

    def get_mcp_failure_policy(self) -> MCPFailurePolicy:
        return self.policy

    def get_actor_mcp_stdio_session_identities(self) -> dict[str, Any]:
        return self.identities

    async def emit_mcp_load_summary(self, summary: MCPToolLoadSummary) -> None:
        self.summaries.append(summary)


class _Loader:
    """Replaces the per-server loader; ``fail`` makes the next load partial."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    async def __call__(self, mcp_configs, **kwargs):
        self.calls += 1
        tools: list[Any] = [_LoadedTool()]
        if self.fail:
            tools.append(
                UnavailableMCPTool(
                    server_name="Gmail", server_id=1, reason="oauth_token_required"
                )
            )
        return tools


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> _Loader:
    fake = _Loader()
    monkeypatch.setattr(
        ToolFactory, "_create_mcp_tools_from_configs", staticmethod(fake)
    )
    return fake


SERVERS = [{"name": "Gmail", "transport": "streamable_http", "config": {"url": "u"}}]


@pytest.mark.asyncio
async def test_rebuild_reuses_load_but_still_reports_a_summary(loader) -> None:
    config = _Config(SERVERS)

    first = await mcp_tools.create_mcp_tools(config)
    second = await mcp_tools.create_mcp_tools(config)

    assert loader.calls == 1
    assert len(config.summaries) == 2
    assert [type(t) for t in second] == [type(t) for t in first]


@pytest.mark.asyncio
async def test_load_with_a_failed_server_is_not_reused(loader) -> None:
    config = _Config(SERVERS)
    loader.fail = True
    await mcp_tools.create_mcp_tools(config)

    loader.fail = False
    await mcp_tools.create_mcp_tools(config)
    await mcp_tools.create_mcp_tools(config)

    # The failed load is retried; the clean one after it is reused.
    assert loader.calls == 2


@pytest.mark.asyncio
async def test_required_policy_does_not_replay_a_stale_failure(loader) -> None:
    config = _Config(SERVERS, policy=MCPFailurePolicy.STRICT)
    loader.fail = True
    with pytest.raises(RequiredMCPUnavailableError):
        await mcp_tools.create_mcp_tools(config)

    loader.fail = False
    assert await mcp_tools.create_mcp_tools(config)
    assert loader.calls == 2


@pytest.mark.asyncio
async def test_changed_actor_identity_value_reloads(loader) -> None:
    config = _Config(SERVERS)
    config.identities = {"chrome": SimpleNamespace(key=("turn-1",))}
    await mcp_tools.create_mcp_tools(config)

    config.identities = {"chrome": SimpleNamespace(key=("turn-2",))}
    await mcp_tools.create_mcp_tools(config)

    assert loader.calls == 2


@pytest.mark.asyncio
async def test_callable_in_server_config_does_not_defeat_reuse(loader) -> None:
    config = _Config(
        [
            {
                **SERVERS[0],
                "config": {"url": "u", "_connector_runtime_refresh": lambda: 1},
            }
        ]
    )
    await mcp_tools.create_mcp_tools(config)

    config.servers = [
        {**SERVERS[0], "config": {"url": "u", "_connector_runtime_refresh": lambda: 2}}
    ]
    await mcp_tools.create_mcp_tools(config)

    assert loader.calls == 1


@pytest.mark.asyncio
async def test_unhashable_config_degrades_to_a_plain_load(loader) -> None:
    config = _Config([{**SERVERS[0], "config": {"url": "u", 1: "non-string key"}}])

    tools = await mcp_tools.create_mcp_tools(config)
    await mcp_tools.create_mcp_tools(config)

    assert tools
    assert loader.calls == 2


@pytest.mark.asyncio
async def test_changed_server_config_reloads(loader) -> None:
    config = _Config(SERVERS)
    await mcp_tools.create_mcp_tools(config)

    config.servers = [{**SERVERS[0], "config": {"url": "other"}}]
    await mcp_tools.create_mcp_tools(config)

    assert loader.calls == 2


@pytest.mark.asyncio
async def test_different_sandbox_reloads(loader) -> None:
    config = _Config(SERVERS, sandbox=object())
    await mcp_tools.create_mcp_tools(config)

    config.sandbox = object()
    await mcp_tools.create_mcp_tools(config)

    assert loader.calls == 2


@pytest.mark.asyncio
async def test_expired_window_reloads_and_drops_the_stale_entry(
    loader, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(mcp_tools, "_clock", lambda: now[0])
    config = _Config(SERVERS)
    await mcp_tools.create_mcp_tools(config)

    now[0] += mcp_tools._MCP_LOAD_REUSE_SECONDS + 1
    loader.fail = True  # the reload is partial, so nothing replaces the entry
    await mcp_tools.create_mcp_tools(config)

    assert loader.calls == 2
    assert getattr(config, mcp_tools._MCP_LOAD_CACHE_ATTR) is None


@pytest.mark.asyncio
async def test_discarding_the_entry_reloads(loader) -> None:
    config = _Config(SERVERS)
    await mcp_tools.create_mcp_tools(config)

    mcp_tools.discard_mcp_load_reuse(config)
    await mcp_tools.create_mcp_tools(config)

    assert loader.calls == 2


def _web_config():
    from xagent.web.tools.config import WebToolConfig

    return WebToolConfig(
        db=MagicMock(), request=None, user_id=1, include_mcp_tools=True
    )


@pytest.mark.parametrize(
    "fence",
    [
        lambda c: c.set_connector_runtime_turn_id("turn-2"),
        lambda c: c.set_mcp_actor_execution_identity(object()),
        lambda c: c.set_execution_scope(ExecutionScope(sandbox_key_suffix="other")),
    ],
    ids=["connector_turn", "actor_identity", "execution_scope"],
)
def test_host_fences_discard_the_reuse_entry(fence) -> None:
    config = _web_config()
    mcp_tools._remember_mcp_load(config, "key", [], MCPToolLoadSummary())
    assert getattr(config, mcp_tools._MCP_LOAD_CACHE_ATTR)

    assert fence(config) is True

    assert not getattr(config, mcp_tools._MCP_LOAD_CACHE_ATTR, None)
