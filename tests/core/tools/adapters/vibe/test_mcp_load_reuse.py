"""A rebuild shortly after a build reuses the loaded MCP tools."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.adapters.vibe import mcp_tools
from xagent.core.tools.adapters.vibe.config import MCPFailurePolicy, MCPToolLoadSummary
from xagent.core.tools.adapters.vibe.factory import ToolFactory


class _Config:
    def __init__(self, servers: list[dict[str, Any]], sandbox: Any = None) -> None:
        self.servers = servers
        self.sandbox = sandbox
        self.summaries: list[MCPToolLoadSummary] = []

    def get_tool_selection_spec(self) -> None:
        return None

    async def get_mcp_server_configs(self) -> list[dict[str, Any]]:
        return self.servers

    def get_sandbox(self) -> Any:
        return self.sandbox

    def get_mcp_failure_policy(self) -> MCPFailurePolicy:
        return MCPFailurePolicy.BEST_EFFORT

    async def emit_mcp_load_summary(self, summary: MCPToolLoadSummary) -> None:
        self.summaries.append(summary)


@pytest.fixture
def load_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, Any]]]:
    calls: list[list[dict[str, Any]]] = []

    async def fake_create(mcp_configs, **kwargs):
        calls.append(mcp_configs)
        return []

    monkeypatch.setattr(
        ToolFactory, "_create_mcp_tools_from_configs", staticmethod(fake_create)
    )
    return calls


SERVERS = [{"name": "Gmail", "transport": "streamable_http", "config": {"url": "u"}}]


@pytest.mark.asyncio
async def test_rebuild_reuses_load_but_still_reports_a_summary(load_calls) -> None:
    config = _Config(SERVERS)

    await mcp_tools.create_mcp_tools(config)
    await mcp_tools.create_mcp_tools(config)

    assert len(load_calls) == 1
    assert len(config.summaries) == 2


@pytest.mark.asyncio
async def test_changed_server_config_reloads(load_calls) -> None:
    config = _Config(SERVERS)
    await mcp_tools.create_mcp_tools(config)

    config.servers = [{**SERVERS[0], "config": {"url": "other"}}]
    await mcp_tools.create_mcp_tools(config)

    assert len(load_calls) == 2


@pytest.mark.asyncio
async def test_different_sandbox_reloads(load_calls) -> None:
    config = _Config(SERVERS, sandbox=object())
    await mcp_tools.create_mcp_tools(config)

    config.sandbox = object()
    await mcp_tools.create_mcp_tools(config)

    assert len(load_calls) == 2


@pytest.mark.asyncio
async def test_expired_window_reloads(
    load_calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(mcp_tools.time, "monotonic", lambda: now[0])
    config = _Config(SERVERS)
    await mcp_tools.create_mcp_tools(config)

    now[0] += mcp_tools._MCP_LOAD_REUSE_SECONDS + 1
    await mcp_tools.create_mcp_tools(config)

    assert len(load_calls) == 2
