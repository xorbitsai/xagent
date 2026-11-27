"""Tests for runner stdio redirection that prevents orphaned children from
holding the docker exec stream open (issue #3257).

The wrapper redirects the runner's stdout/stderr to files inside the sandbox
so that a child left behind by a finished runner writes to files rather than
to the exec stream, letting the per-sandbox exec_lock release promptly.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

# The SandboxedToolWrapper __init__ imports ..mcp_adapter which pulls in
# FastMCP, whose pydantic dependency may be incompatible in some local
# environments.  Inject a lightweight stub so the wrapper can be constructed.
_mcp_stub = types.ModuleType("xagent.core.tools.adapters.vibe.mcp_adapter")
_mcp_stub.MCPToolAdapter = type("MCPToolAdapter", (), {})
sys.modules.setdefault("xagent.core.tools.adapters.vibe.mcp_adapter", _mcp_stub)

from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (  # noqa: E402
    SandboxedToolWrapper,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (  # noqa: E402
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.python_executor import (  # noqa: E402
    PythonExecutorToolForBasic,
)

_ENSURE_REQUIREMENTS = (
    "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper."
    "SandboxDependencyManager.ensure_requirements"
)


@dataclass
class FakeExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class RedirectRecordingSandbox:
    """Records exec calls and simulates file-based stdio redirect.

    Recognises ``sh -c`` tool commands, extracts the redirect targets, and
    serves their contents on subsequent ``cat`` calls.
    """

    def __init__(self, tool_exit_code: int = 0, tool_stderr: str = "") -> None:
        self.name = "redirect-recording"
        self.tool_exit_code = tool_exit_code
        self.tool_stderr = tool_stderr
        self.commands: list[list[str]] = []
        self.files: dict[str, str] = {}
        self._stdout_file: str = ""
        self._stderr_file: str = ""

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        self.files[remote_path] = content

    async def exec(self, command: str, *args: str, env=None, **_: object):
        cmd = [command, *args]
        self.commands.append(cmd)

        if command == "cat":
            path = args[0] if args else ""
            if path.endswith(".json"):
                return FakeExecResult(
                    stdout='{"success": true, "output": "ok", "error": "", "return_code": 0}'
                )
            # stdout/stderr log files
            return FakeExecResult(stdout=self.files.get(path, ""))

        if command == "rm":
            return FakeExecResult()

        if command == "sh" and len(args) >= 2 and args[0] == "-c":
            shell_script = args[1]
            # Extract redirect targets from the shell command
            # Pattern: exec ... >file 2>file
            import re

            stderr_match = re.search(r"2>(\S+)", shell_script)
            stdout_match = re.search(r">(?!2)(\S+)", shell_script)
            if stdout_match:
                self._stdout_file = stdout_match.group(1)
            if stderr_match:
                self._stderr_file = stderr_match.group(1)
                self.files[self._stderr_file] = self.tool_stderr
            return FakeExecResult(
                exit_code=self.tool_exit_code,
                stderr=self.tool_stderr,
            )

        return FakeExecResult(exit_code=self.tool_exit_code)


@sandbox_config(exec_timeout=lambda: 30)
class BoundedTool(PythonExecutorToolForBasic):
    @property
    def name(self) -> str:
        return "bounded_stdio_tool"


@sandbox_config()
class UnboundedTool(PythonExecutorToolForBasic):
    @property
    def name(self) -> str:
        return "unbounded_stdio_tool"


async def _run(wrapper: SandboxedToolWrapper, code: str = "print('ok')"):
    with patch(_ENSURE_REQUIREMENTS, new=AsyncMock()):
        return await wrapper.run_json_async({"code": code})


@pytest.mark.asyncio
async def test_tool_command_wraps_runner_in_shell_redirect():
    """The runner command is wrapped in ``sh -c 'exec ... >file 2>file'``
    so orphaned children write to files, not to the docker exec stream."""
    sandbox = RedirectRecordingSandbox()
    wrapper = SandboxedToolWrapper(BoundedTool(None), sandbox)

    await _run(wrapper)

    tool_commands = [
        c for c in sandbox.commands if c[0] == "sh" and len(c) > 1 and c[1] == "-c"
    ]
    assert len(tool_commands) >= 1, "Expected at least one sh -c command"
    shell_cmd = tool_commands[0][2]
    assert "exec" in shell_cmd
    assert ">" in shell_cmd
    assert "2>" in shell_cmd
    # The redirect targets should be .log files
    assert ".log" in shell_cmd


@pytest.mark.asyncio
async def test_stdio_redirect_files_are_cleaned_up():
    """The stdout/stderr log files must be included in cleanup."""
    sandbox = RedirectRecordingSandbox()
    wrapper = SandboxedToolWrapper(BoundedTool(None), sandbox)

    await _run(wrapper)

    rm_commands = [c for c in sandbox.commands if c[0] == "rm"]
    assert len(rm_commands) >= 1
    rm_paths = rm_commands[-1][2:]  # skip "rm", "-f"
    # At least one path ends with .log
    assert any(p.endswith(".log") for p in rm_paths), (
        f"Cleanup should include .log files, got: {rm_paths}"
    )


@pytest.mark.asyncio
async def test_runner_stderr_is_read_from_file_not_exec_stream():
    """When the runner exits non-zero, its stderr is read from the redirect
    file, not from the exec result's stderr."""
    sandbox = RedirectRecordingSandbox(tool_exit_code=1, tool_stderr="boom")
    wrapper = SandboxedToolWrapper(BoundedTool(None), sandbox)

    with pytest.raises(RuntimeError, match="boom"):
        await _run(wrapper)

    # Verify the stderr file was read via cat
    cat_commands = [c for c in sandbox.commands if c[0] == "cat"]
    cat_paths = [c[1] for c in cat_commands if len(c) > 1]
    assert any(p.endswith(".log") for p in cat_paths), (
        "Expected cat calls for .log files"
    )


@pytest.mark.asyncio
async def test_unbounded_tool_also_gets_stdio_redirect():
    """Even without a timeout, the runner's stdio is redirected to files."""
    sandbox = RedirectRecordingSandbox()
    wrapper = SandboxedToolWrapper(UnboundedTool(None), sandbox)

    await _run(wrapper)

    tool_commands = [
        c for c in sandbox.commands if c[0] == "sh" and len(c) > 1 and c[1] == "-c"
    ]
    assert len(tool_commands) >= 1
    shell_cmd = tool_commands[0][2]
    assert "exec" in shell_cmd
    assert ">" in shell_cmd
    assert "2>" in shell_cmd
    # No timeout prefix when unbounded
    assert "timeout" not in shell_cmd


@pytest.mark.asyncio
async def test_timeout_wrapped_outside_shell_redirect():
    """The timeout command is inside the shell redirect, so the process
    group kill still works and the redirect applies to all output."""
    sandbox = RedirectRecordingSandbox()
    wrapper = SandboxedToolWrapper(BoundedTool(None), sandbox)

    await _run(wrapper)

    tool_commands = [
        c for c in sandbox.commands if c[0] == "sh" and len(c) > 1 and c[1] == "-c"
    ]
    shell_cmd = tool_commands[0][2]
    # timeout should be inside the exec redirect
    assert "timeout" in shell_cmd
    assert "-s" in shell_cmd
    assert "KILL" in shell_cmd
    # The exec prefix means sh is replaced by timeout
    assert shell_cmd.startswith("exec timeout")
