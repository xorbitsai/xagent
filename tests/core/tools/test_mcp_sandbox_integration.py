"""Tests for sandbox-aware MCP tool loading."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from xagent.core.tools.adapters.vibe.mcp_adapter import load_mcp_tools_as_agent_tools
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
    list_tools_in_sandbox,
    should_sandbox_mcp_connection,
)
from xagent.core.tools.core.mcp.sessions import Connection


class TestShouldSandboxMcpConnection:
    """Tests for MCP sandbox classification."""

    @pytest.mark.parametrize(
        ("connection", "expected"),
        [
            ({"transport": "stdio", "command": "npx", "args": []}, True),
            ({"transport": "stdio", "command": "uvx", "args": []}, True),
            ({"transport": "stdio", "command": "/usr/bin/npx", "args": []}, True),
            ({"transport": "stdio", "command": "python", "args": []}, False),
            ({"transport": "sse", "url": "http://localhost"}, False),
            ({"transport": "streamable_http", "url": "http://localhost"}, False),
        ],
    )
    def test_classification(self, connection, expected):
        assert should_sandbox_mcp_connection(connection) is expected


class TestListToolsInSandbox:
    """Tests for sandbox-side MCP list_tools helper."""

    @pytest.mark.asyncio
    async def test_reads_result_file_and_builds_tools(self):
        sandbox = AsyncMock()
        sandbox.name = "test-sandbox"

        json_payload = '[{"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{}}}]'

        # ensure_requirements: write_file + pip install
        # list_tools_in_sandbox: mcp_runner exec, rm cleanup
        pip_result = MagicMock(exit_code=0, stderr="")
        runner_result = MagicMock(exit_code=0, stderr="", error_message=None)
        rm_result = MagicMock(exit_code=0)
        sandbox.exec.side_effect = [pip_result, runner_result, rm_result]
        sandbox.read_file.return_value = json_payload

        tools = await list_tools_in_sandbox(
            sandbox,
            {"transport": "stdio", "command": "npx", "args": ["demo"]},
        )

        assert len(tools) == 1
        assert tools[0].name == "echo"


class TestLoadMcpToolsAsAgentTools:
    """Tests for host-side MCP tool loading split."""

    @pytest.mark.asyncio
    async def test_sandboxed_stdio_server_uses_sandbox_path(self):
        connection: Connection = {
            "transport": "stdio",
            "command": "npx",
            "args": ["demo"],
        }
        sandbox = MagicMock(name="sandbox")
        mcp_tool = MCPTool(
            name="echo",
            description="Echo",
            inputSchema={"type": "object", "properties": {}},
        )
        wrapped_tool = MagicMock()

        with (
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.list_tools_in_sandbox",
                new=AsyncMock(return_value=[mcp_tool]),
            ) as mock_list,
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.create_sandboxed_tool",
                new=AsyncMock(return_value=wrapped_tool),
            ) as mock_wrap,
            patch(
                "xagent.core.tools.adapters.vibe.mcp_adapter._load_direct_mcp_tools",
                new=AsyncMock(),
            ) as mock_direct,
        ):
            tools, reauth_failures = await load_mcp_tools_as_agent_tools(
                {"demo": connection},
                sandbox=sandbox,
            )

        assert tools == [wrapped_tool]
        assert reauth_failures == []
        mock_list.assert_awaited_once_with(sandbox, connection)
        mock_wrap.assert_awaited_once()
        mock_direct.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_sandbox_connection_uses_direct_path(self):
        connection: Connection = {
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
        }
        direct_tool = MagicMock()

        with (
            patch(
                "xagent.core.tools.adapters.vibe.mcp_adapter._load_direct_mcp_tools",
                new=AsyncMock(return_value=[direct_tool]),
            ) as mock_direct,
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.load_sandboxed_mcp_tools",
                new=AsyncMock(),
            ) as mock_sandboxed,
        ):
            tools, reauth_failures = await load_mcp_tools_as_agent_tools(
                {"demo": connection},
                sandbox=MagicMock(),
            )

        assert tools == [direct_tool]
        assert reauth_failures == []
        mock_direct.assert_awaited_once()
        mock_sandboxed.assert_not_awaited()
