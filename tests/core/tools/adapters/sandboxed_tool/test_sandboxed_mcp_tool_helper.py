"""Regression tests for list_tools_in_sandbox's SandboxLeaseProvider handling.

Uses plain mocks instead of boxlite: these exercise only the
Sandbox-vs-SandboxLeaseProvider unwrap logic, not real sandbox execution.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
    list_tools_in_sandbox,
    load_sandboxed_mcp_tools,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxDependencyManager,
    resolve_primary_sandbox,
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
    sandbox.name = "mock_sandbox"
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


@pytest.mark.asyncio
async def test_load_sandboxed_mcp_tools_passes_original_provider_not_primary():
    """list_tools_in_sandbox resolves the primary sandbox internally for its
    one-shot metadata call, but load_sandboxed_mcp_tools must still hand
    create_sandboxed_tool the *original* provider -- not that resolved
    primary -- so each wrapped tool keeps per-call worker-slot leasing. A
    future refactor that started passing the primary everywhere would break
    concurrency isolation for real tool calls while leaving this untested."""
    primary_sandbox = _make_plain_sandbox(["xero_tool"])
    lease_provider = _FakeSandboxLeaseProvider(primary_sandbox)
    connection = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@xeroapi/xero-mcp-server@latest"],
    }
    fake_wrapped_tool = MagicMock()

    with patch(
        "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper."
        "create_sandboxed_tool",
        new=AsyncMock(return_value=fake_wrapped_tool),
    ) as mock_create:
        result = await load_sandboxed_mcp_tools(
            connection, lease_provider, lambda mcp_tool: MagicMock()
        )

    assert result.tools == (fake_wrapped_tool,)
    mock_create.assert_awaited_once()
    _called_tool, called_sandbox = mock_create.await_args.args
    assert called_sandbox is lease_provider
    assert called_sandbox is not primary_sandbox


def test_resolve_primary_sandbox_rejects_none():
    with pytest.raises(ValueError, match="sandbox cannot be None"):
        resolve_primary_sandbox(None)
