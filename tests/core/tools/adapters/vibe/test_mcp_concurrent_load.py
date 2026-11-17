"""MCP servers load concurrently within one tool build, with stable ordering."""

from __future__ import annotations

import asyncio

import pytest

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    load_mcp_tools_as_agent_tools,
)


def _connections(names: list[str]) -> dict[str, dict[str, str]]:
    return {n: {"transport": "streamable_http", "url": f"http://{n}"} for n in names}


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_servers_load_concurrently_up_to_the_cap(monkeypatch):
    in_flight = 0
    peak = 0

    async def fake_load_direct(server_name, connection, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return MCPLoadResult(
            tools=(object(),), loaded_servers=(server_name,), failures=()
        )

    monkeypatch.setattr(mcp_adapter_module, "_load_direct_mcp_tools", fake_load_direct)
    cap = mcp_adapter_module._MCP_SERVER_LOAD_CONCURRENCY

    result = await load_mcp_tools_as_agent_tools(
        _connections([f"s{i}" for i in range(cap * 2)])
    )

    assert len(result.tools) == cap * 2
    assert 1 < peak <= cap


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_results_keep_connection_order_when_later_servers_finish_first(
    monkeypatch,
):
    tools = {name: object() for name in ("a", "b", "c")}
    delays = {"a": 0.15, "b": 0.05, "c": 0.0}

    async def fake_load_direct(server_name, connection, **kwargs):
        await asyncio.sleep(delays[server_name])
        if server_name == "b":
            raise RuntimeError("boom")
        return MCPLoadResult(
            tools=(tools[server_name],), loaded_servers=(server_name,), failures=()
        )

    monkeypatch.setattr(mcp_adapter_module, "_load_direct_mcp_tools", fake_load_direct)

    result = await load_mcp_tools_as_agent_tools(_connections(["a", "b", "c"]))

    assert [t.target for t in result.tools] == [tools["a"], tools["c"]]
    assert result.loaded_servers == ("a", "c")
    assert [f.server_name for f in result.failures] == ["b"]
    assert result.failures[0].phase is MCPFailurePhase.SESSION_START
