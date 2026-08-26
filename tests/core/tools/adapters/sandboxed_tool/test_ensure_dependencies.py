"""
Test _ensure_dependencies logic in SandboxedToolWrapper

All sandbox interactions are mocked.
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.core.tools.adapters.sandboxed_tool.conftest import (
    FakeBaseTool,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SANDBOX_BASE_DEPENDENCIES,
    SandboxDependencyManager,
    SandboxedToolWrapper,
)
from xagent.sandbox.base import Sandbox


@dataclass
class FakeExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


def _make_sandbox(name: str = "sandbox-1") -> MagicMock:
    """Create a mock Sandbox with async methods.

    spec=Sandbox so isinstance(sb, Sandbox) holds -- an unspecced
    MagicMock auto-vivifies any attribute touched, including
    `.primary_sandbox`, which would make resolve_primary_sandbox mistake
    this plain-sandbox double for a SandboxLeaseProvider.
    """
    sb = MagicMock(spec=Sandbox)
    sb.name = name
    sb.write_file = AsyncMock()
    sb.exec = AsyncMock(return_value=FakeExecResult(exit_code=0))
    return sb


@sandbox_config()
class _DefaultTool(FakeBaseTool):
    """Fake tool with no extra runtime packages."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "default_tool"


@pytest.fixture(autouse=True)
def _clear_class_state():
    """Reset class-level state between tests."""
    SandboxDependencyManager.reset()
    SandboxDependencyManager._locks_lock = asyncio.Lock()
    yield
    SandboxDependencyManager.reset()
    SandboxDependencyManager._locks_lock = asyncio.Lock()


class TestEnsureDependencies:
    """Test _ensure_dependencies with mocked sandbox."""

    @pytest.mark.asyncio
    async def test_first_call_installs(self):
        """First call should write requirements and run pip install."""
        sandbox = _make_sandbox("sb-install")
        wrapper = SandboxedToolWrapper(_DefaultTool(), sandbox)

        await wrapper._ensure_dependencies()

        sandbox.write_file.assert_called_once()
        sandbox.exec.assert_called_once()
        assert SandboxDependencyManager._sandbox_installed_requirements.get(
            "sb-install"
        ) == set(SANDBOX_BASE_DEPENDENCIES)

    @pytest.mark.asyncio
    async def test_second_call_skips(self):
        """Second call on same sandbox should skip installation."""
        sandbox = _make_sandbox("sb-skip")
        wrapper = SandboxedToolWrapper(_DefaultTool(), sandbox)

        await wrapper._ensure_dependencies()
        sandbox.write_file.reset_mock()
        sandbox.exec.reset_mock()

        await wrapper._ensure_dependencies()

        sandbox.write_file.assert_not_called()
        sandbox.exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_wrappers_same_sandbox_install_once(self):
        """Two wrappers sharing the same sandbox should only install once."""
        sandbox = _make_sandbox("sb-shared")
        w1 = SandboxedToolWrapper(_DefaultTool(), sandbox)
        w2 = SandboxedToolWrapper(_DefaultTool(), sandbox)

        await w1._ensure_dependencies()
        sandbox.exec.reset_mock()
        sandbox.write_file.reset_mock()

        await w2._ensure_dependencies()

        sandbox.exec.assert_not_called()
        sandbox.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_different_sandboxes_independent(self):
        """Different sandboxes should install independently."""
        sb1 = _make_sandbox("sb-a")
        sb2 = _make_sandbox("sb-b")
        w1 = SandboxedToolWrapper(_DefaultTool(), sb1)
        w2 = SandboxedToolWrapper(_DefaultTool(), sb2)

        await w1._ensure_dependencies()
        await w2._ensure_dependencies()

        sb1.exec.assert_called_once()
        sb2.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_pip_failure_does_not_mark_installed(self):
        """If pip install fails, the sandbox should NOT be marked as installed."""
        sandbox = _make_sandbox("sb-fail")
        sandbox.exec = AsyncMock(
            return_value=FakeExecResult(exit_code=1, stderr="pip error")
        )
        wrapper = SandboxedToolWrapper(_DefaultTool(), sandbox)

        with pytest.raises(RuntimeError, match="Dependency installation failed"):
            await wrapper._ensure_dependencies()

        assert "sb-fail" not in SandboxDependencyManager._sandbox_installed_requirements

    @pytest.mark.asyncio
    async def test_no_extra_packages_still_installs_base(self):
        """Even with no extra packages, base deps (pydantic) should be installed."""
        sandbox = _make_sandbox("sb-base")
        wrapper = SandboxedToolWrapper(_DefaultTool(), sandbox)

        await wrapper._ensure_dependencies()

        assert SandboxDependencyManager._sandbox_installed_requirements.get(
            "sb-base"
        ) == set(SANDBOX_BASE_DEPENDENCIES)
        sandbox.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_calls_same_sandbox(self):
        """Concurrent _ensure_dependencies on the same sandbox should only install once."""
        call_count = 0
        original_result = FakeExecResult(exit_code=0)

        async def slow_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return original_result

        sandbox = _make_sandbox("sb-concurrent")
        sandbox.exec = slow_exec
        w1 = SandboxedToolWrapper(_DefaultTool(), sandbox)
        w2 = SandboxedToolWrapper(_DefaultTool(), sandbox)

        await asyncio.gather(
            w1._ensure_dependencies(),
            w2._ensure_dependencies(),
        )

        # Only one pip install should have happened
        assert call_count == 1
        assert SandboxDependencyManager._sandbox_installed_requirements.get(
            "sb-concurrent"
        ) == set(SANDBOX_BASE_DEPENDENCIES)
