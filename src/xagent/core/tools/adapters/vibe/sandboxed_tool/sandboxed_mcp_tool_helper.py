"""Sandbox helpers for MCP tool registration and wrapping."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import posixpath
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from mcp.types import Tool as MCPTool

from ......sandbox.base import Sandbox
from ....core.mcp.sessions import Connection
from ....core.mcp.tools import (
    SANDBOX_RAW_ANNOTATIONS_KEY,
    attach_raw_annotations,
)
from ..base import AbstractBaseTool
from .sandbox_config import SandboxConfig, set_instance_sandbox_config
from .sandboxed_tool_wrapper import (
    SANDBOX_BASE_DEPENDENCIES,
    SANDBOX_SRC_ROOT,
    SandboxDependencyManager,
    _SandboxLeaseProviderLike,
    create_sandboxed_tool,
    resolve_primary_sandbox,
)

logger = logging.getLogger(__name__)

_MCP_RUNNER_PATH = (
    f"{SANDBOX_SRC_ROOT}/xagent/core/tools/adapters/vibe/sandboxed_tool/mcp_runner.py"
)

_MCP_SANDBOX_COMMANDS = {"npx", "uvx"}
_MCP_SANDBOX_TIMEOUT_SECONDS = 60

# Upper-bounded to match this repo's own top-level "mcp" pin (pyproject.toml,
# both [project].dependencies and [dependency-groups].sandbox). That pin is
# resolved once into uv.lock at image build time and stays fixed, but this
# string is installed live into the sandbox on every fresh container -- kept
# in sync here rather than relying on pyproject.toml alone because a lock
# drift there wouldn't be the only way this specific live-install path could
# still float past its floor. mcp 2.0 renamed/removed
# mcp.client.streamable_http's streamablehttp_client, which sessions.py
# imports -- confirmed in production as the root cause of chrome-devtools'
# (and every other sandboxed connector's) sandbox_list_tools failures once a
# fresh sandbox picked up 2.0 from PyPI while the backend stayed on the 1.x
# line.
_MCP_SANDBOX_EXTRA_PACKAGES = ["mcp>=1.12.4,<2"]
_MCP_SANDBOX_ENV = ["XAGENT_USER_ID"]
_MCP_SANDBOX_CONFIG = SandboxConfig(
    packages=tuple(_MCP_SANDBOX_EXTRA_PACKAGES),
    env_vars=tuple(_MCP_SANDBOX_ENV),
)

_SANDBOX_MCP_DEPENDENCIES = SANDBOX_BASE_DEPENDENCIES + _MCP_SANDBOX_EXTRA_PACKAGES


@dataclass(frozen=True)
class SandboxedMCPLoadResult:
    """Internal sandbox load outcome with public-safe failure metadata."""

    tools: tuple[AbstractBaseTool, ...]
    adapter_error_types: tuple[str, ...]
    wrap_error_types: tuple[str, ...]


def should_sandbox_mcp_connection(connection: Connection) -> bool:
    """Return whether the MCP connection should run inside sandbox."""
    if connection.get("transport") != "stdio":
        return False

    command = connection.get("command")
    if not isinstance(command, str) or not command.strip():
        return False

    return posixpath.basename(command) in _MCP_SANDBOX_COMMANDS


def _serialize_connection(connection: Connection) -> str:
    """Serialize a connection dict for sandbox transport."""
    return base64.b64encode(
        json.dumps(connection, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")


async def list_tools_in_sandbox(
    sandbox: Sandbox | _SandboxLeaseProviderLike,
    connection: Connection,
    *,
    timeout_seconds: int = _MCP_SANDBOX_TIMEOUT_SECONDS,
) -> list[MCPTool]:
    """List MCP tools by creating the MCP session inside sandbox."""
    result_file = f"/tmp/xagent_mcp_tools_{uuid.uuid4().hex}.json"
    connection_b64 = _serialize_connection(connection)

    # `sandbox` may be a SandboxLeaseProvider (no exec/read_file/name of its
    # own) rather than a real Sandbox -- unwrap to the primary sandbox, since
    # this one-shot metadata call doesn't need per-call worker leasing.
    target_sandbox = resolve_primary_sandbox(sandbox)

    await SandboxDependencyManager.ensure_requirements(
        target_sandbox, _SANDBOX_MCP_DEPENDENCIES
    )

    try:
        try:
            result = await asyncio.wait_for(
                target_sandbox.exec(
                    "python",
                    _MCP_RUNNER_PATH,
                    "--connection-b64",
                    connection_b64,
                    "--result-file",
                    result_file,
                    env={"PYTHONPATH": SANDBOX_SRC_ROOT},
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP list_tools timed out after {timeout_seconds} seconds"
            ) from exc

        if result.exit_code != 0:
            error_msg = result.stderr or result.error_message or "Unknown error"
            raise RuntimeError(f"Sandbox MCP list_tools failed: {error_msg}")

        try:
            output = await target_sandbox.read_file(result_file)
        except FileNotFoundError:
            logger.warning("MCP list_tools result file not found: %s", result_file)
            return []

        output = output.strip()
        if not output:
            logger.warning(
                "MCP list_tools result file is empty after successful exit: %s",
                result_file,
            )
            return []

        try:
            tool_data = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse MCP list_tools output from %s (%s)",
                result_file,
                type(e).__name__,
            )
            raise RuntimeError(f"Failed to parse MCP list_tools output: {e}") from e

        # The private annotations key the in-sandbox runner attached is
        # stripped before validation (``Tool`` would otherwise reject it or
        # carry it as extra) and re-attached afterwards, so a sandboxed tool
        # classifies from the same wire evidence a directly loaded one does.
        tools: list[MCPTool] = []
        for item in tool_data:
            raw = None
            if isinstance(item, dict):
                payload = {
                    k: v for k, v in item.items() if k != SANDBOX_RAW_ANNOTATIONS_KEY
                }
                candidate = item.get(SANDBOX_RAW_ANNOTATIONS_KEY)
                raw = candidate if isinstance(candidate, dict) else None
            else:
                payload = item
            tool = MCPTool.model_validate(payload)
            attach_raw_annotations(tool, raw)
            tools.append(tool)
        return tools
    finally:
        try:
            await target_sandbox.exec("rm", "-f", result_file)
        except Exception:
            pass


async def load_sandboxed_mcp_tools(
    connection: Connection,
    sandbox: Sandbox | _SandboxLeaseProviderLike,
    tool_builder: Callable[[MCPTool], AbstractBaseTool],
) -> SandboxedMCPLoadResult:
    """Load MCP tool metadata in sandbox and wrap built tools for sandboxed calls."""
    mcp_tools = await list_tools_in_sandbox(sandbox, connection)
    wrapped_tools: list[AbstractBaseTool] = []
    adapter_error_types: list[str] = []
    wrap_error_types: list[str] = []

    for mcp_tool in mcp_tools:
        try:
            tool = tool_builder(mcp_tool)
        except Exception as e:
            logger.warning(
                "Failed to build sandboxed MCP tool '%s' (%s)",
                mcp_tool.name,
                type(e).__name__,
            )
            adapter_error_types.append(type(e).__name__)
            continue

        try:
            set_instance_sandbox_config(tool, _MCP_SANDBOX_CONFIG)
            wrapped_tools.append(await create_sandboxed_tool(tool, sandbox))
        except Exception as e:
            logger.warning(
                "Failed to wrap sandboxed MCP tool '%s' (%s)",
                mcp_tool.name,
                type(e).__name__,
            )
            wrap_error_types.append(type(e).__name__)
            continue

    return SandboxedMCPLoadResult(
        tools=tuple(wrapped_tools),
        adapter_error_types=tuple(adapter_error_types),
        wrap_error_types=tuple(wrap_error_types),
    )
