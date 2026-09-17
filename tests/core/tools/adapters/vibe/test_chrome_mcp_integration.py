from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from mcp.types import Tool as MCPTool

from xagent.core.tools.adapters.vibe import mcp_adapter
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    ChromeExecutionMCPToolAdapter,
    MCPLoadResult,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_DEVTOOLS_PACKAGE,
    ChromeExecutionSessionPool,
    ChromeSessionContractError,
)
from xagent.web.services.actor_mcp_runtime import (
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
)
from xagent.web.services.chrome_mcp_runtime import (
    consume_chrome_actor_stdio_session,
)
from xagent.web.services.mcp_runtime import MCPActorExecutionIdentity


def _identity(*, turn_id: str = "turn-one", attempt: str = "attempt-one"):
    return ActorMCPStdioSessionIdentity(
        execution=MCPActorExecutionIdentity(
            task_id=7,
            run_id="run-one",
            turn_id=turn_id,
            lease_attempt_id=attempt,
        ),
        connection=ActorMCPStdioConnectionIdentity(
            user_id=11,
            resource_owner_key="toby:owner-secret",
            app_id="chrome-devtools",
            catalog_app_generation=UUID("11111111-1111-4111-8111-111111111111"),
            lifecycle_generation=UUID("22222222-2222-4222-8222-222222222222"),
        ),
    )


def _connection():
    return {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "--prefer-offline",
            CHROME_DEVTOOLS_PACKAGE,
            "--headless",
            "--isolated",
        ],
        "env": {"XAGENT_MCP_CALLER_ID": "11"},
    }


def _tool():
    return MCPTool(
        name="navigate_page",
        description="Navigate",
        inputSchema={"type": "object", "properties": {}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox", [None, object()])
async def test_host_consumer_strips_identity_and_env_before_child_serializer(
    monkeypatch, sandbox
):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    serialized_connections = []

    async def list_tools(_sandbox, connection):
        serialized_connections.append(connection)
        json.dumps(connection)
        return [_tool()]

    monkeypatch.setattr(mcp_adapter, "list_tools_in_sandbox", list_tools)
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )

    tools = await consume_chrome_actor_stdio_session(
        server_name="chrome-devtools",
        connection=_connection(),
        session_identity=_identity(),
        sandbox=sandbox,
    )

    assert len(tools) == 1
    assert isinstance(tools[0], ChromeExecutionMCPToolAdapter)
    assert tools[0].connection["env"] == {}
    assert serialized_connections[0]["env"] == {
        "NPM_CONFIG_CACHE": "/opt/npm-cache",
        "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
    }
    assert "actor_stdio_session_identity" not in repr(serialized_connections)
    assert "toby:owner-secret" not in repr(serialized_connections)


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, object()])
async def test_wrong_host_identity_fails_before_any_child_loader(monkeypatch, identity):
    child_loader = AsyncMock()
    generic_loader = AsyncMock()
    monkeypatch.setattr(mcp_adapter, "list_tools_in_sandbox", child_loader)
    monkeypatch.setattr(mcp_adapter, "load_sandboxed_mcp_tools", generic_loader)

    with pytest.raises(ChromeSessionContractError):
        await consume_chrome_actor_stdio_session(
            server_name="chrome-devtools",
            connection=_connection(),
            session_identity=identity,
            sandbox=None,
        )

    child_loader.assert_not_awaited()
    generic_loader.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_loader_does_not_route_by_chrome_server_name(monkeypatch):
    direct = AsyncMock(
        return_value=MCPLoadResult(
            tools=(object(),), loaded_servers=("chrome-devtools",), failures=()
        )
    )
    dedicated = AsyncMock()
    monkeypatch.setattr(mcp_adapter, "_load_direct_mcp_tools", direct)
    monkeypatch.setattr(mcp_adapter, "load_execution_scoped_chrome_tools", dedicated)

    result = await load_mcp_tools_as_agent_tools({"chrome-devtools": _connection()})

    assert len(result.tools) == 1
    direct.assert_awaited_once()
    dedicated.assert_not_awaited()


@pytest.mark.asyncio
async def test_chrome_adapter_reuses_scope_and_validates_daemon_result(monkeypatch):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    pool.invoke_tool.return_value = {
        "content": [{"type": "text", "text": "same browser"}],
        "structuredContent": {"page": 2},
        "isError": False,
    }
    monkeypatch.setattr(
        mcp_adapter, "list_tools_in_sandbox", AsyncMock(return_value=[_tool()])
    )
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )
    tools = await consume_chrome_actor_stdio_session(
        server_name="chrome-devtools",
        connection=_connection(),
        session_identity=_identity(),
        sandbox=None,
    )
    tool = tools[0]

    first = await tool._execute_mcp_call(tool.connection, {}, {})
    second = await tool._execute_mcp_call(tool.connection, {}, {})

    assert first == second
    assert first["structured_content"] == {"page": 2}
    assert pool.invoke_tool.await_count == 2
    assert (
        pool.invoke_tool.await_args_list[0].args[0]
        == pool.invoke_tool.await_args_list[1].args[0]
    )


@pytest.mark.asyncio
async def test_invalid_daemon_result_closes_scope_without_per_call_fallback(
    monkeypatch,
):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    pool.invoke_tool.return_value = {"content": [], "isError": "false"}
    direct_session = AsyncMock()
    monkeypatch.setattr(mcp_adapter, "create_session", direct_session)
    monkeypatch.setattr(
        mcp_adapter, "list_tools_in_sandbox", AsyncMock(return_value=[_tool()])
    )
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )
    tools = await consume_chrome_actor_stdio_session(
        server_name="chrome-devtools",
        connection=_connection(),
        session_identity=_identity(),
        sandbox=None,
    )

    with pytest.raises(ChromeSessionContractError):
        await tools[0]._execute_mcp_call(tools[0].connection, {}, {})

    pool.close_shielded.assert_awaited_once()
    direct_session.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_teardown_does_not_abort_runner_cleanup(monkeypatch):
    pool = AsyncMock(spec=ChromeExecutionSessionPool)
    pool.get_or_create.return_value = SimpleNamespace(sandbox=object())
    pool.close_shielded.side_effect = asyncio.CancelledError
    monkeypatch.setattr(
        mcp_adapter, "list_tools_in_sandbox", AsyncMock(return_value=[_tool()])
    )
    monkeypatch.setattr(
        "xagent.web.services.chrome_mcp_runtime.get_chrome_execution_session_pool",
        lambda: pool,
    )
    tools = await consume_chrome_actor_stdio_session(
        server_name="chrome-devtools",
        connection=_connection(),
        session_identity=_identity(),
        sandbox=None,
    )

    await tools[0].teardown()

    pool.close_shielded.assert_awaited_once()
