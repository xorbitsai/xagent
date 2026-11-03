"""Fetch a release calendar and latest releases through FXMacroData MCP."""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import yaml

from xagent.core.tools.adapters.vibe.mcp_adapter import load_mcp_tools_as_agent_tools
from xagent.core.tools.core.mcp.data_config import MCPServerConfig
from xagent.core.tools.core.mcp.sessions import Connection

API_KEY_ENV = "FXMACRODATA_API_KEY"
CURRENCY_PATTERN = re.compile(r"[A-Za-z]{3}")
# Printable ASCII without spaces: anything else is not a valid header value.
API_KEY_PATTERN = re.compile(r"[!-~]+")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
NOTICE_FIELDS = ("freemium_delay", "freemium_window")
TOOL_NAMES = ("release_calendar", "latest_announcements")


def read_api_key() -> str | None:
    """Return the optional API key from the environment, validated."""
    value = os.environ.get(API_KEY_ENV, "").strip()
    if not value:
        return None
    if not API_KEY_PATTERN.fullmatch(value):
        raise ValueError(f"{API_KEY_ENV} contains characters not allowed in a header")
    return value


def normalize_currency(value: str) -> str:
    code = value.strip()
    if not CURRENCY_PATTERN.fullmatch(code):
        raise ValueError("currency must be a 3-letter code such as USD")
    return code.lower()


def load_config(config_path: Path, api_key: str | None) -> MCPServerConfig:
    """Validate the YAML and add the bearer header only when a key is set."""
    config = MCPServerConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    url = urlsplit(config.url or "")
    if url.scheme != "https" and url.hostname not in LOOPBACK_HOSTS:
        raise ValueError("the MCP URL must use https")
    if api_key is None:
        return config
    headers = dict(config.headers or {})
    headers["Authorization"] = f"Bearer {api_key}"
    return config.model_copy(update={"headers": headers})


def redact(text: str, api_key: str | None) -> str:
    return text.replace(api_key, "[redacted]") if api_key else text


def parse_json_text(text: str) -> Any:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, (dict, list)) else None


def collect_notices(value: Any) -> list[str]:
    """Find free-tier delay and history-window messages anywhere in a result."""
    if isinstance(value, str):
        return collect_notices(parse_json_text(value))
    if isinstance(value, list):
        return [notice for item in value for notice in collect_notices(item)]
    if not isinstance(value, dict):
        return []
    notices = [
        value[field]["message"]
        for field in NOTICE_FIELDS
        if isinstance(value.get(field), dict)
        and isinstance(value[field].get("message"), str)
    ]
    return notices + [
        notice for item in value.values() for notice in collect_notices(item)
    ]


async def load_tools(config: MCPServerConfig) -> dict[str, Any]:
    connection = cast(Connection, config.to_connection())
    loaded = await load_mcp_tools_as_agent_tools({config.name: connection})
    if loaded.failures:
        phases = ", ".join(failure.phase.value for failure in loaded.failures)
        raise RuntimeError(f"Could not load MCP tools: {phases}")
    return {tool.name: tool for tool in loaded.tools}


async def call_tool(
    tools: dict[str, Any], server: str, name: str, currency: str, api_key: str | None
) -> dict[str, Any]:
    tool = tools.get(f"mcp_{server}_{name}")
    if tool is None:
        raise RuntimeError(f"MCP server did not expose {name}")
    result = await tool.run_json_async({"currency": currency})
    failed = not isinstance(result, dict) or result.get("is_error")
    if failed or result.get("success") is False:
        message = redact(tool.return_value_as_string(result), api_key)
        raise RuntimeError(f"{name} failed: {message}")
    return result


async def fetch(config_path: Path, currency: str = "USD") -> dict[str, Any]:
    """Call the release calendar and latest releases for one currency.

    Without FXMACRODATA_API_KEY the server answers for USD only and delays
    releases by 15 minutes. The delay message is returned under "notices" so
    an agent does not present delayed data as current.
    """
    api_key = read_api_key()
    code = normalize_currency(currency)
    config = load_config(config_path, api_key)
    tools = await load_tools(config)
    output: dict[str, Any] = {}
    for name in TOOL_NAMES:
        output[name] = await call_tool(tools, config.name, name, code, api_key)
    output["notices"] = list(dict.fromkeys(collect_notices(output)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--currency", default="USD")
    args = parser.parse_args()
    try:
        api_key = read_api_key()
        result = asyncio.run(fetch(args.config, args.currency))
    except Exception as error:  # report every failure without the key
        raw_key = os.environ.get(API_KEY_ENV, "").strip() or None
        text = redact(f"{type(error).__name__}: {error}", raw_key)
        print(text, file=sys.stderr)
        raise SystemExit(1) from None
    print(redact(json.dumps(result, ensure_ascii=False, indent=2), api_key))


if __name__ == "__main__":
    main()
