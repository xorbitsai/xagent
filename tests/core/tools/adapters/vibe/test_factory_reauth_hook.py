"""Tests for the ``MCPReauthorizationRequired`` hook in
``ToolRegistry.create_registered_tools``.

Background:
    When a registered tool creator raises ``MCPReauthorizationRequired``
    during a real agent run, the per-creator loop must not let it abort
    the other creators (existing per-creator isolation must be preserved),
    but it also must not be silently swallowed like a generic exception --
    the config gets a chance (via an optional ``on_mcp_reauthorization_required``
    hook) to flag the connection so the UI can prompt the user to reconnect.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolRegistry
from xagent.core.tools.core.mcp.oauth.errors import MCPReauthorizationRequired


def _mock_tool(name: str, category: str):
    """Minimal mock tool with the ``.metadata.category`` shape that
    ``ToolRegistry._sort_tools_by_category`` inspects."""
    tool = MagicMock()
    tool.name = name
    tool.metadata = MagicMock()
    tool.metadata.category = category
    return tool


@pytest.fixture
def isolated_registry():
    """Snapshot and restore ``ToolRegistry._tool_creators`` so in-place
    mutations here don't leak into other test modules that depend on the
    production creator list."""
    saved = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    ToolRegistry._tool_creators = []
    ToolRegistry._modules_imported = True
    try:
        yield ToolRegistry
    finally:
        ToolRegistry._tool_creators = saved
        ToolRegistry._modules_imported = saved_imported


class _FakeConfigNoHook:
    """Config without ``on_mcp_reauthorization_required`` -- the generic
    case that must keep working safely."""

    def get_tool_selection_spec(self):
        return None


class _FakeConfigWithHook(_FakeConfigNoHook):
    def __init__(self):
        self.hook = AsyncMock()

    async def on_mcp_reauthorization_required(self, mcpserver_id):
        await self.hook(mcpserver_id)


async def test_reauth_error_does_not_abort_other_creators(isolated_registry):
    """A creator raising MCPReauthorizationRequired must not prevent other
    registered creators' tools from being returned."""
    basic_tool = _mock_tool("basic_tool", "basic")
    ok_tool = AsyncMock(return_value=[basic_tool])
    ok_tool.__name__ = "ok_creator"

    async def failing_creator(config):
        raise MCPReauthorizationRequired("notion", 7)

    failing_creator.__name__ = "failing_creator"

    isolated_registry.register(ok_tool, categories={"basic"})
    isolated_registry.register(failing_creator, categories={"mcp"})

    tools = await isolated_registry.create_registered_tools(_FakeConfigNoHook())

    assert ok_tool.await_count == 1
    assert tools == [basic_tool]


async def test_reauth_error_invokes_hook_with_mcpserver_id(isolated_registry):
    """When the config implements the hook, it is awaited with the
    ``mcpserver_id`` carried on the exception."""

    async def failing_creator(config):
        raise MCPReauthorizationRequired("notion", 7)

    failing_creator.__name__ = "failing_creator"
    isolated_registry.register(failing_creator, categories={"mcp"})

    config = _FakeConfigWithHook()
    await isolated_registry.create_registered_tools(config)

    config.hook.assert_awaited_once_with(7)


async def test_reauth_error_without_hook_does_not_crash(isolated_registry):
    """A config that doesn't implement the hook (the plain/generic case)
    must not raise -- the exception is logged and swallowed just like the
    hook-present case, minus the hook call."""

    async def failing_creator(config):
        raise MCPReauthorizationRequired("notion", 7)

    failing_creator.__name__ = "failing_creator"
    isolated_registry.register(failing_creator, categories={"mcp"})

    tools = await isolated_registry.create_registered_tools(_FakeConfigNoHook())

    assert tools == []
