"""Unit tests for SandboxedToolWrapper sandbox lease selection."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)


@dataclass
class FakeExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class FakeSandbox:
    def __init__(self, name: str) -> None:
        self.name = name
        self.exec_calls: list[tuple[str, tuple[str, ...]]] = []
        self.files: dict[str, str] = {}

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        assert overwrite or remote_path not in self.files
        self.files[remote_path] = content

    async def exec(self, command: str, *args: str, env: dict[str, str] | None = None):
        self.exec_calls.append((command, args))
        if command == "cat":
            return FakeExecResult(stdout=f'{{"sandbox": "{self.name}"}}')
        return FakeExecResult()


class FakeLease:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self._sandbox = sandbox

    async def __aenter__(self) -> FakeSandbox:
        return self._sandbox

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


class FakeLeaseProvider:
    def __init__(self, *, primary: FakeSandbox, worker: FakeSandbox) -> None:
        self.primary = primary
        self.worker = worker
        self.lease_calls: list[bool] = []

    def lease(self, *, concurrency_safe: bool) -> FakeLease:
        self.lease_calls.append(concurrency_safe)
        return FakeLease(self.worker if concurrency_safe else self.primary)


@sandbox_config()
class UnsafeSandboxTool(FakeBaseTool):
    @property
    def name(self) -> str:
        return "unsafe_sandbox_tool"


@sandbox_config()
class SafeSandboxTool(FakeBaseTool):
    concurrency_safe = True

    @property
    def name(self) -> str:
        return "safe_sandbox_tool"


@pytest.mark.asyncio
async def test_safe_tool_executes_through_worker_sandbox() -> None:
    primary = FakeSandbox("primary")
    worker = FakeSandbox("worker")
    provider = FakeLeaseProvider(primary=primary, worker=worker)
    wrapper = SandboxedToolWrapper(SafeSandboxTool(), provider)

    with patch(
        "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper."
        "SandboxDependencyManager.ensure_requirements",
        new=AsyncMock(),
    ) as ensure_requirements:
        result = await wrapper.run_json_async({"code": "print('ok')"})

    assert result == {"sandbox": "worker"}
    assert provider.lease_calls == [True]
    ensure_requirements.assert_awaited_once()
    assert ensure_requirements.await_args.args[0] is worker
    assert primary.exec_calls == []
    assert [call[0] for call in worker.exec_calls] == ["python", "cat", "rm"]


@pytest.mark.asyncio
async def test_unsafe_tool_executes_through_primary_sandbox() -> None:
    primary = FakeSandbox("primary")
    worker = FakeSandbox("worker")
    provider = FakeLeaseProvider(primary=primary, worker=worker)
    wrapper = SandboxedToolWrapper(UnsafeSandboxTool(), provider)

    with patch(
        "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper."
        "SandboxDependencyManager.ensure_requirements",
        new=AsyncMock(),
    ) as ensure_requirements:
        result = await wrapper.run_json_async({"code": "print('ok')"})

    assert result == {"sandbox": "primary"}
    assert provider.lease_calls == [False]
    ensure_requirements.assert_awaited_once()
    assert ensure_requirements.await_args.args[0] is primary
    assert [call[0] for call in primary.exec_calls] == ["python", "cat", "rm"]
    assert worker.exec_calls == []
