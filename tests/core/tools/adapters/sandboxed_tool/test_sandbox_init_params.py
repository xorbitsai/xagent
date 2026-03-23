"""
Test init params extraction, serialization, and script generation for sandbox tool reconstruction
"""

import base64
import threading
from typing import Any, Mapping, Optional, Type
from unittest.mock import AsyncMock, MagicMock, patch

import cloudpickle
import pytest
from pydantic import BaseModel, Field

from xagent.core.tools.adapters.vibe.base import AbstractBaseTool
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
    _extract_init_params,
    _serialize_init_params,
)
from xagent.core.workspace import TaskWorkspace


class _FakeArgs(BaseModel):
    code: str = Field(default="")


class _FakeResult(BaseModel):
    output: str = Field(default="")


class _FakeToolWithWorkspace(AbstractBaseTool):
    """Fake tool with init params."""

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "fake_tool_ws"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def tags(self) -> list[str]:
        return []

    def args_type(self) -> Type[BaseModel]:
        return _FakeArgs

    def return_type(self) -> Type[BaseModel]:
        return _FakeResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return {}

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return {}


class _FakeToolNoParams(AbstractBaseTool):
    """Fake tool with no init params."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "fake_tool_nop"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def tags(self) -> list[str]:
        return []

    def args_type(self) -> Type[BaseModel]:
        return _FakeArgs

    def return_type(self) -> Type[BaseModel]:
        return _FakeResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return {}

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return {}


_FAKE_CONFIG_PATH = (
    "xagent.core.tools.adapters.vibe.sandboxed_tool"
    ".sandboxed_tool_config.get_sandbox_tool_config"
)


def _fake_config():
    """Return a MagicMock that mimics SandboxToolConfig."""
    cfg = MagicMock()
    cfg.packages = []
    cfg.env_vars = []
    cfg.tool_class = "some.module:FakeClass"
    return cfg


def _make_sandbox(name: str = "sandbox-test") -> MagicMock:
    """Return a MagicMock that mimics Sandbox instance."""
    sb = MagicMock()
    sb.name = name
    sb.write_file = AsyncMock()
    sb.exec = AsyncMock()
    return sb


def _create_test_wrapper(tool: AbstractBaseTool) -> SandboxedToolWrapper:
    """Create a SandboxedToolWrapper with mocked config."""
    with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
        return SandboxedToolWrapper(tool, _make_sandbox())


class TestExtractInitParams:
    """Tests for _extract_init_params()."""

    def test_with_params(self):
        """Tool with params should be extracted correctly."""
        mock_ws = MagicMock(spec=TaskWorkspace)
        tool = _FakeToolWithWorkspace(workspace=mock_ws)
        params = _extract_init_params(tool)
        assert params == {"workspace": mock_ws}

    def test_no_params(self):
        """Tool with no init params should return empty dict."""
        tool = _FakeToolNoParams()
        params = _extract_init_params(tool)
        assert params == {}


class TestSerializeInitParams:
    """Tests for _serialize_init_params()."""

    def test_with_params(self, tmp_path):
        """Params serialize and deserialize."""
        ws = TaskWorkspace(id="test-ws", base_dir=str(tmp_path))
        params = {"workspace": ws, "task_id": "test-123"}
        b64_str = _serialize_init_params(params)
        assert b64_str is not None
        restored = cloudpickle.loads(base64.b64decode(b64_str))
        assert restored["workspace"].id == "test-ws"
        assert restored["task_id"] == "test-123"

    def test_empty(self):
        """Empty params should return None."""
        assert _serialize_init_params({}) is None

    def test_non_serializable(self):
        """Non-serializable param should raise RuntimeError."""
        params = {"bad_param": threading.Lock()}
        with pytest.raises(RuntimeError, match="bad_param"):
            _serialize_init_params(params)


class TestGenerateExecutionScript:
    """Tests for _generate_execution_script()."""

    def test_with_init_params(self):
        """Script should contain pickle deserialization when init params exist."""
        wrapper = _create_test_wrapper(_FakeToolWithWorkspace(workspace=None))
        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
            script = wrapper._generate_execution_script(
                {"code": "print(1)"}, "/tmp/result.json"
            )
        assert "cloudpickle.loads" in script
        assert "**init_params" in script

    def test_without_init_params(self):
        """Script should use no-arg construction when no init params."""
        wrapper = _create_test_wrapper(_FakeToolNoParams())
        with patch(_FAKE_CONFIG_PATH, return_value=_fake_config()):
            script = wrapper._generate_execution_script(
                {"code": "print(1)"}, "/tmp/result.json"
            )
        assert "cloudpickle.loads" not in script
        assert "**init_params" not in script
