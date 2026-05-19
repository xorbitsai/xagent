"""Test that SandboxedToolWrapper registers files produced by the sandbox.

Files written inside the sandbox appear on the host via the mounted workspace
volume, but the sandbox process cannot reach the host database, so any
``workspace.auto_register_files()`` call made inside the sandbox is silently
dropped. The wrapper must register newly-created workspace files on the host
side around each ``sandbox.exec`` call so they receive valid ``file_id``s.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)
from xagent.core.workspace import TaskWorkspace


@dataclass
class FakeExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class _Args(BaseModel):
    payload: str = Field(default="")


@sandbox_config()
class _ToolWithWorkspace(FakeBaseTool):
    """Fake tool that holds a TaskWorkspace, mirroring real executor tools."""

    def __init__(self, workspace: TaskWorkspace) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "tool_with_workspace"

    def args_type(self) -> Type[BaseModel]:
        return _Args


@sandbox_config()
class _ToolNoWorkspace(FakeBaseTool):
    """Fake tool without any workspace attribute."""

    @property
    def name(self) -> str:
        return "tool_no_workspace"


def _make_sandbox_writing(file_to_create) -> MagicMock:
    """Mock sandbox whose exec writes a file (simulating the mounted volume)."""

    async def fake_exec(*_args: Any, **_kwargs: Any) -> FakeExecResult:
        file_to_create.parent.mkdir(parents=True, exist_ok=True)
        file_to_create.write_bytes(b"fake pptx bytes")
        return FakeExecResult(exit_code=0, stdout='{"success": true}')

    sb = MagicMock()
    sb.name = "sb-test"
    sb.write_file = AsyncMock()
    sb.exec = AsyncMock(side_effect=fake_exec)
    return sb


@pytest.fixture(autouse=True)
def _clear_class_state():
    """Reset shared class-level state between tests."""
    SandboxedToolWrapper._sandbox_deps_installed = {}
    SandboxedToolWrapper._sandbox_deps_locks = {}
    SandboxedToolWrapper._locks_lock = asyncio.Lock()
    yield
    SandboxedToolWrapper._sandbox_deps_installed = {}
    SandboxedToolWrapper._sandbox_deps_locks = {}


class TestSandboxedToolFileRegistration:
    """Verify workspace file registration happens on the host side."""

    @pytest.mark.asyncio
    async def test_registers_files_created_by_sandbox(self, tmp_path):
        """Files produced inside the sandbox must be registered on the host."""
        # Non-numeric workspace id keeps DB writes a no-op while still letting
        # the in-memory file_id cache populate.
        workspace = TaskWorkspace(id="test_pptx_ws", base_dir=str(tmp_path))
        generated = workspace.output_dir / "deck.pptx"

        sandbox = _make_sandbox_writing(generated)
        SandboxedToolWrapper._sandbox_deps_installed["sb-test"] = True

        wrapper = SandboxedToolWrapper(_ToolWithWorkspace(workspace), sandbox)

        await wrapper.run_json_async({"payload": "go"})

        assert generated.exists(), "sandbox.exec should have created the file"

        file_id = workspace.get_file_id_from_path(str(generated))
        assert file_id, (
            "auto_register_files should have assigned a file_id on the host "
            "after the sandbox exec returned"
        )

    @pytest.mark.asyncio
    async def test_no_workspace_runs_unchanged(self, tmp_path):
        """Tools without a workspace must still execute, with no extra wrapping."""
        sandbox = MagicMock()
        sandbox.name = "sb-no-ws"
        sandbox.write_file = AsyncMock()
        sandbox.exec = AsyncMock(
            return_value=FakeExecResult(exit_code=0, stdout='{"ok": true}')
        )
        SandboxedToolWrapper._sandbox_deps_installed["sb-no-ws"] = True

        wrapper = SandboxedToolWrapper(_ToolNoWorkspace(), sandbox)

        assert wrapper._get_target_workspace() is None
        result = await wrapper.run_json_async({})
        assert result == {"ok": True}

    def test_get_target_workspace_finds_tool_attribute(self, tmp_path):
        """_get_target_workspace finds a TaskWorkspace on the target tool."""
        workspace = TaskWorkspace(id="lookup_ws", base_dir=str(tmp_path))
        sandbox = MagicMock()
        sandbox.name = "sb-lookup"
        sandbox.write_file = AsyncMock()
        sandbox.exec = AsyncMock()

        wrapper = SandboxedToolWrapper(_ToolWithWorkspace(workspace), sandbox)

        assert wrapper._get_target_workspace() is workspace
