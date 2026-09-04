"""List MCP tools from inside sandbox using a stable Python entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any, cast

# WARNING: This file runs as a standalone script in the sandbox, not a module.
# Absolute imports only — relative imports are unavailable (no package context).
from xagent.core.tools.adapters.vibe.sandboxed_tool.runner_utils import (
    ensure_user_bin_in_path,
)
from xagent.core.tools.core.mcp.sessions import Connection
from xagent.core.tools.core.mcp.tools import (
    SANDBOX_RAW_ANNOTATIONS_KEY,
    load_mcp_tools,
    raw_annotations_for,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sandboxed MCP tool listing."""
    parser = argparse.ArgumentParser(description="List MCP tools in sandbox")
    parser.add_argument("--connection-b64", required=True)
    parser.add_argument("--result-file", required=True)
    return parser.parse_args()


def _load_connection(connection_b64: str) -> Connection:
    """Decode a base64-encoded connection config."""
    connection_json = base64.b64decode(connection_b64).decode("utf-8")
    return cast(Connection, json.loads(connection_json))


async def _list_tools(connection: Connection) -> list[dict[str, Any]]:
    """List and serialize MCP tools for JSON output.

    ``model_dump`` alone would lose the wire types of the ``annotations``
    values: the SDK's models declare them ``bool | None`` under non-strict
    validation, so a server that sent ``1`` or ``"true"`` is already
    indistinguishable from one that sent ``true`` by the time a tool is
    dumped here. The captured mapping is re-attached under a private key so
    the host side can classify the declaration honestly -- without it, every
    sandboxed connector would look like it made whatever claim the coercion
    produced.
    """
    tools = await load_mcp_tools(None, connection=connection)
    serialized: list[dict[str, Any]] = []
    for tool in tools:
        item = cast(dict[str, Any], tool.model_dump(mode="json"))
        raw = raw_annotations_for(tool)
        if raw is not None:
            item[SANDBOX_RAW_ANNOTATIONS_KEY] = raw
        serialized.append(item)
    return serialized


def main() -> None:
    """CLI entrypoint for sandboxed MCP tool listing."""
    ensure_user_bin_in_path()
    try:
        parsed = _parse_args()
        connection = _load_connection(parsed.connection_b64)
    except Exception as e:
        print(f"Sandbox mcp config error: {e}")
        raise

    result = asyncio.run(_list_tools(connection))
    Path(parsed.result_file).write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
