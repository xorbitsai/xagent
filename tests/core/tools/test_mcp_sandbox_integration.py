"""Tests for sandbox-aware MCP tool loading."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
    list_tools_in_sandbox,
    should_sandbox_mcp_connection,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SANDBOX_BASE_DEPENDENCIES,
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
    async def test_requirements_preserve_mcp_without_reinstalling_uv(self):
        sandbox = AsyncMock()
        sandbox.exec.side_effect = [
            MagicMock(exit_code=0, stderr="", error_message=None),
            MagicMock(exit_code=0),
        ]
        sandbox.read_file.return_value = "[]"

        with patch(
            "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.SandboxDependencyManager.ensure_requirements",
            new=AsyncMock(),
        ) as mock_ensure_requirements:
            await list_tools_in_sandbox(
                sandbox,
                {"transport": "stdio", "command": "uvx", "args": ["demo"]},
            )

        requirements = mock_ensure_requirements.await_args.args[1]
        assert requirements == [*SANDBOX_BASE_DEPENDENCIES, "mcp>=1.12.4,<2"]

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
            result = await load_mcp_tools_as_agent_tools(
                {"demo": connection},
                sandbox=sandbox,
            )

        assert result.tools == (wrapped_tool,)
        assert result.loaded_servers == ("demo",)
        assert result.failures == ()
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
                new=AsyncMock(
                    return_value=MCPLoadResult(
                        tools=(direct_tool,), loaded_servers=("demo",), failures=()
                    )
                ),
            ) as mock_direct,
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.load_sandboxed_mcp_tools",
                new=AsyncMock(),
            ) as mock_sandboxed,
        ):
            result = await load_mcp_tools_as_agent_tools(
                {"demo": connection},
                sandbox=MagicMock(),
            )

        assert result.tools == (direct_tool,)
        assert result.loaded_servers == ("demo",)
        assert result.failures == ()
        mock_direct.assert_awaited_once()
        mock_sandboxed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sandbox_list_failure_is_preserved_without_secret(self, caplog):
        # Pinned below default: the sandboxed-load path also emits a DEBUG
        # traceback (test_sandbox_list_failure_debug_traceback_is_opt_in
        # below), which would otherwise make this secret-safety assertion
        # fail under an ambient DEBUG log level (e.g. `pytest -o
        # log_level=DEBUG`) for a reason unrelated to what it actually
        # guards: the always-on ERROR log staying class-name-only.
        caplog.set_level("WARNING")
        connection: Connection = {
            "transport": "stdio",
            "command": "npx",
            "args": ["demo"],
        }

        with patch(
            "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.list_tools_in_sandbox",
            new=AsyncMock(
                side_effect=RuntimeError("Bearer planted-sandbox-list-secret")
            ),
        ):
            result = await load_mcp_tools_as_agent_tools(
                {"demo": connection}, sandbox=MagicMock()
            )

        assert result.tools == ()
        assert result.loaded_servers == ()
        assert len(result.failures) == 1
        assert result.failures[0].phase is MCPFailurePhase.SANDBOX_LIST_TOOLS
        assert result.failures[0].error_type == "RuntimeError"
        assert "planted-sandbox-list-secret" not in repr(result)
        assert "planted-sandbox-list-secret" not in caplog.text

    @pytest.mark.asyncio
    async def test_sandbox_list_failure_debug_traceback_is_opt_in(self, caplog):
        # Companion to the test above: pins the other half of the two-layer
        # contract -- opting into DEBUG surfaces the full traceback, so
        # reproducing a failure with XAGENT_LOG_LEVEL=DEBUG (or --debug)
        # actually captures it rather than silently getting nothing more
        # than the always-on ERROR log already gives.
        caplog.set_level("DEBUG")
        connection: Connection = {
            "transport": "stdio",
            "command": "npx",
            "args": ["demo"],
        }

        with patch(
            "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.list_tools_in_sandbox",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await load_mcp_tools_as_agent_tools(
                {"demo": connection}, sandbox=MagicMock()
            )

        assert "RuntimeError: boom" in caplog.text

    @pytest.mark.asyncio
    async def test_sandbox_wrap_failure_preserves_other_wrapped_tools(self):
        connection: Connection = {
            "transport": "stdio",
            "command": "npx",
            "args": ["demo"],
        }
        sandbox = MagicMock(name="sandbox")
        mcp_tools = [
            MCPTool(
                name="healthy",
                description="Healthy",
                inputSchema={"type": "object", "properties": {}},
            ),
            MCPTool(
                name="broken",
                description="Broken",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        wrapped_tool = MagicMock()

        with (
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.list_tools_in_sandbox",
                new=AsyncMock(return_value=mcp_tools),
            ),
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.create_sandboxed_tool",
                new=AsyncMock(
                    side_effect=[
                        wrapped_tool,
                        RuntimeError("Bearer planted-sandbox-wrap-secret"),
                    ]
                ),
            ),
        ):
            result = await load_mcp_tools_as_agent_tools(
                {"demo": connection}, sandbox=sandbox
            )

        assert result.tools == (wrapped_tool,)
        assert result.loaded_servers == ("demo",)
        assert len(result.failures) == 1
        assert result.failures[0].phase is MCPFailurePhase.SANDBOX_TOOL_WRAP
        assert result.failures[0].error_type == "RuntimeError"
        assert "planted-sandbox-wrap-secret" not in repr(result)

    @pytest.mark.asyncio
    async def test_sandbox_adapter_failure_is_preserved(self):
        connection: Connection = {
            "transport": "stdio",
            "command": "npx",
            "args": ["demo"],
        }
        mcp_tool = MCPTool(
            name="broken",
            description="Broken",
            inputSchema={"type": "object", "properties": {}},
        )

        with (
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.list_tools_in_sandbox",
                new=AsyncMock(return_value=[mcp_tool]),
            ),
            patch(
                "xagent.core.tools.adapters.vibe.mcp_adapter._build_mcp_tool_adapter",
                side_effect=ValueError("Bearer planted-builder-secret"),
            ),
        ):
            result = await load_mcp_tools_as_agent_tools(
                {"demo": connection}, sandbox=MagicMock()
            )

        assert result.tools == ()
        assert result.loaded_servers == ()
        assert len(result.failures) == 1
        assert result.failures[0].phase is MCPFailurePhase.ADAPTER_CONSTRUCTION
        assert result.failures[0].error_type == "ValueError"
        assert "planted-builder-secret" not in repr(result)

    @pytest.mark.asyncio
    async def test_sandbox_adapter_and_wrap_failures_are_both_preserved(self):
        connection: Connection = {
            "transport": "stdio",
            "command": "npx",
            "args": ["demo"],
        }
        mcp_tools = [
            MCPTool(
                name="broken-adapter",
                description="Broken adapter",
                inputSchema={"type": "object", "properties": {}},
            ),
            MCPTool(
                name="broken-wrap",
                description="Broken wrap",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        adapter = MagicMock()

        with (
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.list_tools_in_sandbox",
                new=AsyncMock(return_value=mcp_tools),
            ),
            patch(
                "xagent.core.tools.adapters.vibe.mcp_adapter._build_mcp_tool_adapter",
                side_effect=[
                    ValueError("Bearer planted-combined-adapter-secret"),
                    adapter,
                ],
            ),
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.create_sandboxed_tool",
                new=AsyncMock(
                    side_effect=RuntimeError("Bearer planted-combined-wrap-secret")
                ),
            ),
        ):
            result = await load_mcp_tools_as_agent_tools(
                {"demo": connection}, sandbox=MagicMock()
            )

        assert result.tools == ()
        assert result.loaded_servers == ()
        assert [failure.phase for failure in result.failures] == [
            MCPFailurePhase.ADAPTER_CONSTRUCTION,
            MCPFailurePhase.SANDBOX_TOOL_WRAP,
        ]
        assert [failure.error_type for failure in result.failures] == [
            "ValueError",
            "RuntimeError",
        ]
        assert "planted-combined-adapter-secret" not in repr(result)
        assert "planted-combined-wrap-secret" not in repr(result)

    @pytest.mark.asyncio
    async def test_sandbox_server_with_no_tools_is_reported(self):
        connection: Connection = {
            "transport": "stdio",
            "command": "npx",
            "args": ["demo"],
        }

        with patch(
            "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper.list_tools_in_sandbox",
            new=AsyncMock(return_value=[]),
        ):
            result = await load_mcp_tools_as_agent_tools(
                {"demo": connection}, sandbox=MagicMock()
            )

        assert result.tools == ()
        assert result.loaded_servers == ()
        assert len(result.failures) == 1
        assert result.failures[0].phase is MCPFailurePhase.NO_TOOLS_RETURNED
        assert result.failures[0].error_type is None
