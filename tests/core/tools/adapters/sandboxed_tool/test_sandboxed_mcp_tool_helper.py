"""Regression tests for list_tools_in_sandbox's SandboxLeaseProvider handling.

Uses plain mocks instead of boxlite: these exercise only the
Sandbox-vs-SandboxLeaseProvider unwrap logic, not real sandbox execution.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
    list_tools_in_sandbox,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxDependencyManager,
)


@pytest.fixture(autouse=True)
def _reset_dependency_manager():
    """SandboxDependencyManager tracks installed requirements in a class-level
    dict, shared process-wide -- reset it so one test's mock sandbox can't be
    mistaken for "already installed" state left behind by another."""
    SandboxDependencyManager.reset()
    yield
    SandboxDependencyManager.reset()


def _make_exec_result(exit_code: int = 0, stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.exit_code = exit_code
    result.stderr = stderr
    result.error_message = None
    return result


def _make_plain_sandbox(tool_names: list[str]) -> MagicMock:
    sandbox = MagicMock()
    sandbox.exec = AsyncMock(return_value=_make_exec_result())
    sandbox.read_file = AsyncMock(
        return_value=json.dumps(
            [{"name": name, "inputSchema": {"type": "object"}} for name in tool_names]
        )
    )
    sandbox.write_file = AsyncMock()
    return sandbox


class _FakeSandboxLeaseProvider:
    """Mirrors the real SandboxLeaseProvider shape _is_sandbox_lease_provider checks for."""

    def __init__(self, primary_sandbox) -> None:
        self.primary_sandbox = primary_sandbox

    def lease(self, *, concurrency_safe: bool):  # pragma: no cover - not exercised here
        raise NotImplementedError


@pytest.mark.asyncio
async def test_list_tools_in_sandbox_with_plain_sandbox():
    sandbox = _make_plain_sandbox(["some_tool"])
    connection = {"transport": "stdio", "command": "npx", "args": ["-y", "pkg"]}

    tools = await list_tools_in_sandbox(sandbox, connection)

    assert [t.name for t in tools] == ["some_tool"]
    sandbox.exec.assert_awaited()
    sandbox.read_file.assert_awaited()


@pytest.mark.asyncio
async def test_list_tools_in_sandbox_unwraps_lease_provider():
    """A SandboxLeaseProvider has no .name/.exec/.read_file of its own --
    list_tools_in_sandbox must use its .primary_sandbox instead of blowing
    up with AttributeError (the bug reproduced against stage: 'SandboxLeaseProvider'
    object has no attribute 'name')."""
    primary_sandbox = _make_plain_sandbox(["xero_tool"])
    lease_provider = _FakeSandboxLeaseProvider(primary_sandbox)
    connection = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@xeroapi/xero-mcp-server@latest"],
    }

    tools = await list_tools_in_sandbox(lease_provider, connection)

    assert [t.name for t in tools] == ["xero_tool"]
    primary_sandbox.exec.assert_awaited()
    primary_sandbox.read_file.assert_awaited()
