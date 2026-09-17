"""End-to-end pins for the placeholder tool an unavailable MCP server yields.

These tests run the whole production chain in one go: a config built by the
real ``WebToolConfig`` builders, handed to the real ``ToolFactory`` entry point
that assembles MCP tools, then the resulting placeholder tool actually invoked.
``XAGENT_USER_ID`` is explicitly removed from the environment, because no
server process writes it -- so the caller identity a placeholder would see in
production is always absent. A placeholder that consults a caller identity here
denies every caller and reports "Access denied" instead of the outage, which is
the failure these tests exist to catch.

The narrower unit tests elsewhere use hand-written config dictionaries; these
start from the real builders so a change to what those builders emit cannot
silently stop reaching the placeholder.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
    UnavailableMCPTool,
)
from xagent.web.tools.config import WebToolConfig

OWNER_USER_ID = 42


def _assert_reports_outage(result: Any, *, expected_message: str) -> None:
    """The placeholder answered with its outage, not with a denial.

    The denial check comes first so that a regression fails on the thing this
    test is about, rather than on a missing key of the outage result.
    """
    assert "Access denied" not in repr(result)
    text = result["content"][0]["text"]
    assert expected_message in text
    assert result["is_error"] is True
    assert result["error"] == expected_message


@pytest.mark.asyncio
async def test_config_load_failure_placeholder_reports_outage_without_a_caller_id(
    monkeypatch: pytest.MonkeyPatch,
    unavailable_mcp_server: SimpleNamespace,
) -> None:
    """Real ``_build_unavailable_mcp_config`` output -> factory -> invocation.

    This is the branch that runs when the config for a selected server cannot
    be built at all (for example a transport the runtime cannot serve).
    """
    monkeypatch.delenv("XAGENT_USER_ID", raising=False)

    cfg = WebToolConfig(db=None, request=None, user_id=OWNER_USER_ID)
    config = cfg._build_unavailable_mcp_config(
        server=unavailable_mcp_server, reason="config_load_failed"
    )

    tools = await ToolFactory._create_mcp_tools_from_configs([config])

    assert [tool.name for tool in tools] == ["mcp_example_1_unavailable"]
    placeholder = tools[0]
    assert isinstance(placeholder, UnavailableMCPTool)
    for result in (placeholder.run_json_sync({}), await placeholder.run_json_async({})):
        _assert_reports_outage(result, expected_message="MCP server is unavailable.")
        assert result["reason"] == "config_load_failed"


@pytest.mark.asyncio
async def test_handshake_failure_placeholder_reports_outage_without_a_caller_id(
    monkeypatch: pytest.MonkeyPatch,
    stdio_mcp_server: SimpleNamespace,
) -> None:
    """Real executable config -> failed handshake -> factory -> invocation.

    The config here is the one ``_build_mcp_server_config`` emits for a server
    that is configured correctly; the server itself fails to initialize. This
    is the most common way a placeholder reaches a user. That this config
    carries no caller allow-list for the factory to pass along is pinned
    separately by ``test_executable_mcp_config_does_not_carry_allow_users`` in
    ``test_webtoolconfig_identity.py``; here the assertion is on the outcome,
    so a regression fails on the answer the caller gets.
    """
    monkeypatch.delenv("XAGENT_USER_ID", raising=False)

    cfg = WebToolConfig(db=None, request=None, user_id=OWNER_USER_ID)
    config = await cfg._build_mcp_server_config(
        server=stdio_mcp_server,
        user_env_by_id={},
        shared_env_by_id={},
        env_source_by_id={},
    )

    async def failing_loader(connections, **kwargs):
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name="Test Stdio Server",
                    phase=MCPFailurePhase.INITIALIZE,
                    error_type="RuntimeError",
                    attempts=3,
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        failing_loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs([config])

    assert [tool.name for tool in tools] == ["mcp_test_stdio_server_5_unavailable"]
    placeholder = tools[0]
    assert isinstance(placeholder, UnavailableMCPTool)
    for result in (placeholder.run_json_sync({}), await placeholder.run_json_async({})):
        _assert_reports_outage(
            result, expected_message="MCP server initialization failed."
        )
        assert result["reason"] == "initialize"
