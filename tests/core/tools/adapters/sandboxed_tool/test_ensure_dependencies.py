"""
Test _ensure_dependencies logic in SandboxedToolWrapper

All sandbox interactions are mocked.
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)


@dataclass
class FakeExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


def _make_sandbox(name: str = "sandbox-1") -> MagicMock:
    """Create a mock Sandbox with async methods."""
    sb = MagicMock()
    sb.name = name
    sb.write_file = AsyncMock()
    sb.exec = AsyncMock(return_value=FakeExecResult(exit_code=0))
    return sb


def _make_tool(name: str = "python_executor") -> MagicMock:
    """Create a mock AbstractBaseTool."""
    tool = MagicMock()
    tool.name = name
    return tool


@pytest.fixture(autouse=True)
def _clear_class_state():
    """Reset class-level state between tests."""
    SandboxedToolWrapper._sandbox_deps_installed = {}
    SandboxedToolWrapper._sandbox_deps_locks = {}
    SandboxedToolWrapper._locks_lock = asyncio.Lock()
    yield
    SandboxedToolWrapper._sandbox_deps_installed = {}
    SandboxedToolWrapper._sandbox_deps_locks = {}


_FAKE_CONFIG_PATH = (
    "src.xagent.core.tools.adapters.vibe.sandboxed_tool"
    ".sandboxed_tool_wrapper.get_sandbox_tool_config"
)


def _fake_config(packages: list[str] | None = None):
    cfg = MagicMock()
    cfg.packages = packages or ["numpy"]
    cfg.env_vars = []
    return cfg


class TestEnsureDependencies:
    """Test _ensure_dependencies with mocked sandbox."""

    @pytest.mark.asyncio
    async def test_first_call_installs(self):
        """First call should write requirements and run pip install."""
        sandbox = _make_sandbox("sb-install")

        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config(["numpy"])):
            wrapper = SandboxedToolWrapper(_make_tool(), sandbox)

        await wrapper._ensure_dependencies()

        sandbox.write_file.assert_called_once()
        sandbox.exec.assert_called_once()
        assert SandboxedToolWrapper._sandbox_deps_installed.get("sb-install") is True

    @pytest.mark.asyncio
    async def test_second_call_skips(self):
        """Second call on same sandbox should skip installation."""
        sandbox = _make_sandbox("sb-skip")

        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
            wrapper = SandboxedToolWrapper(_make_tool(), sandbox)

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

        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
            w1 = SandboxedToolWrapper(_make_tool("tool_a"), sandbox)
            w2 = SandboxedToolWrapper(_make_tool("tool_b"), sandbox)

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

        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
            w1 = SandboxedToolWrapper(_make_tool(), sb1)
            w2 = SandboxedToolWrapper(_make_tool(), sb2)

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

        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
            wrapper = SandboxedToolWrapper(_make_tool(), sandbox)

        with pytest.raises(RuntimeError, match="Dependency installation failed"):
            await wrapper._ensure_dependencies()

        assert "sb-fail" not in SandboxedToolWrapper._sandbox_deps_installed

    @pytest.mark.asyncio
    async def test_no_extra_packages_still_installs_base(self):
        """Even with no extra packages, base deps (pydantic) should be installed."""
        sandbox = _make_sandbox("sb-base")

        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config(packages=[])):
            wrapper = SandboxedToolWrapper(_make_tool(), sandbox)

        await wrapper._ensure_dependencies()

        assert SandboxedToolWrapper._sandbox_deps_installed.get("sb-base") is True
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

        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
            w1 = SandboxedToolWrapper(_make_tool("t1"), sandbox)
            w2 = SandboxedToolWrapper(_make_tool("t2"), sandbox)

        await asyncio.gather(
            w1._ensure_dependencies(),
            w2._ensure_dependencies(),
        )

        # Only one pip install should have happened
        assert call_count == 1
        assert SandboxedToolWrapper._sandbox_deps_installed.get("sb-concurrent") is True
