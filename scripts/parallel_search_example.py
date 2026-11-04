"""Run an opt-in Parallel search through Xagent's agent MCP tools."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import yaml

from xagent.core.tools.adapters.vibe.mcp_adapter import load_mcp_tools_as_agent_tools
from xagent.core.tools.core.mcp.data_config import MCPServerConfig
from xagent.core.tools.core.mcp.sessions import Connection


async def search(
    config_path: Path, query: str, fetch_url: str | None = None
) -> dict[str, Any]:
    """Load a standalone config and execute the discovered agent tools.

    This does not initialize the database or read or write saved connectors.
    Search and fetch share one free-tier session identifier.
    """
    config = MCPServerConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    connection = cast(Connection, config.to_connection())
    loaded = await load_mcp_tools_as_agent_tools({config.name: connection})
    if loaded.failures:
        phases = ", ".join(failure.phase.value for failure in loaded.failures)
        raise RuntimeError(f"Could not load MCP tools: {phases}")

    # Use the names exposed by the maintained loader, including its server prefix.
    tools = {tool.name: tool for tool in loaded.tools}
    session_id = uuid4().hex
    output = {}
    calls: list[tuple[str, dict[str, Any]]] = [
        (
            "web_search",
            {
                "objective": query,
                "search_queries": [query],
                "session_id": session_id,
            },
        )
    ]
    if fetch_url:
        calls.append(
            (
                "web_fetch",
                {
                    "urls": [fetch_url],
                    "objective": query[:200],
                    "session_id": session_id,
                },
            )
        )
    for name, arguments in calls:
        tool = tools.get(f"mcp_{config.name}_{name}")
        if tool is None:
            raise RuntimeError(f"MCP server did not expose {name}")
        result = await tool.run_json_async(arguments)
        if result.get("is_error") or result.get("success") is False:
            raise RuntimeError(f"{name} failed: {tool.return_value_as_string(result)}")
        output[name] = result
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--fetch-url", help="Also fetch excerpts from this URL")
    args = parser.parse_args()
    result = asyncio.run(search(args.config, args.query, args.fetch_url))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
