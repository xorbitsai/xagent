"""A stdio MCP connector subprocess and the parent process's own
``OutputFilteredToolWrapper`` each independently bound tool output length
(also field count / recursion depth): the child reads
``XAGENT_TOOL_MAX_OUTPUT_LENGTH`` etc. from its OWN environment to build a
bounded, valid-JSON result; the parent reads the same-named env var (or a
per-config override) from ITS OWN environment to decide where to blindly
character-slice that result. ``_create_stdio_session`` launches the child
with a minimal, explicitly-built env (credentials + caller id only), never
inheriting the parent's environment, so the two budgets can silently
disagree -- and when the parent's budget is smaller, its blind slice can cut
the child's already-valid JSON mid-structure.

``create_mcp_tools`` closes this by mirroring the parent's own effective
budget into every stdio config's env before tools are built from it, so
both sides always agree regardless of which env var / config override
produced the parent's number.
"""

import pytest

from xagent.config import (
    TOOL_MAX_FIELD_COUNT,
    TOOL_MAX_OUTPUT_LENGTH,
    TOOL_MAX_RECURSION_DEPTH,
)
from xagent.core.tools.adapters.vibe.config import MCPFailurePolicy
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools


class _FakeConfig:
    def __init__(self, mcp_configs):
        self._mcp_configs = mcp_configs

    def get_tool_selection_spec(self):
        return None

    async def get_mcp_server_configs(self):
        return self._mcp_configs

    def get_mcp_failure_policy(self):
        return MCPFailurePolicy.BEST_EFFORT

    def get_sandbox(self):
        return None

    def get_max_output_length(self):
        return 12345

    def get_max_field_count(self):
        return 99

    def get_max_recursion_depth(self):
        return 7


@pytest.mark.asyncio
async def test_stdio_configs_receive_the_parents_effective_output_limits(monkeypatch):
    captured: dict = {}

    async def fake_create(mcp_configs, sandbox=None):
        captured["mcp_configs"] = mcp_configs
        return []

    monkeypatch.setattr(ToolFactory, "_create_mcp_tools_from_configs", staticmethod(fake_create))

    config = _FakeConfig(
        [
            {
                "name": "jira",
                "transport": "stdio",
                "config": {"command": "python", "args": ["-m", "probe"]},
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    env = cfg["config"]["env"]
    assert env[TOOL_MAX_OUTPUT_LENGTH] == "12345"
    assert env[TOOL_MAX_FIELD_COUNT] == "99"
    assert env[TOOL_MAX_RECURSION_DEPTH] == "7"


@pytest.mark.asyncio
async def test_existing_stdio_env_entries_are_preserved(monkeypatch):
    """Credentials and caller-id env already placed on the config (e.g. by
    the actor-owned/team-owned loaders) must survive untouched, and an
    already-present output-limit value on the config -- from some future
    per-server override -- is never clobbered."""
    captured: dict = {}

    async def fake_create(mcp_configs, sandbox=None):
        captured["mcp_configs"] = mcp_configs
        return []

    monkeypatch.setattr(ToolFactory, "_create_mcp_tools_from_configs", staticmethod(fake_create))

    config = _FakeConfig(
        [
            {
                "name": "jira",
                "transport": "stdio",
                "config": {
                    "command": "python",
                    "args": [],
                    "env": {
                        "XAGENT_MCP_CALLER_ID": "user-1",
                        TOOL_MAX_OUTPUT_LENGTH: "preset-value",
                    },
                },
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    env = cfg["config"]["env"]
    assert env["XAGENT_MCP_CALLER_ID"] == "user-1"
    assert env[TOOL_MAX_OUTPUT_LENGTH] == "preset-value"
    assert env[TOOL_MAX_FIELD_COUNT] == "99"


@pytest.mark.asyncio
async def test_non_stdio_configs_are_left_untouched(monkeypatch):
    captured: dict = {}

    async def fake_create(mcp_configs, sandbox=None):
        captured["mcp_configs"] = mcp_configs
        return []

    monkeypatch.setattr(ToolFactory, "_create_mcp_tools_from_configs", staticmethod(fake_create))

    config = _FakeConfig(
        [
            {
                "name": "http-server",
                "transport": "streamable_http",
                "config": {"url": "https://example.com/mcp"},
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    assert "env" not in cfg["config"]


@pytest.mark.asyncio
async def test_unavailable_stdio_placeholder_is_left_untouched(monkeypatch):
    captured: dict = {}

    async def fake_create(mcp_configs, sandbox=None):
        captured["mcp_configs"] = mcp_configs
        return []

    monkeypatch.setattr(ToolFactory, "_create_mcp_tools_from_configs", staticmethod(fake_create))

    config = _FakeConfig(
        [
            {
                "name": "broken",
                "transport": "stdio",
                "config": {"unavailable": True, "reason": "oauth_token_required"},
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    assert "env" not in cfg["config"]
