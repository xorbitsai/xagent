"""
Test init params extraction, serialization, and script generation for sandbox tool reconstruction
"""

import base64
import json
import threading
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import cloudpickle
import pytest

from tests.core.tools.adapters.sandboxed_tool.conftest import (
    FakeBaseTool,
)
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool
from xagent.core.tools.adapters.vibe.function import FunctionTool
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SANDBOX_SRC_ROOT,
    SandboxedToolWrapper,
    _extract_init_params,
    _serialize_init_params,
)
from xagent.core.workspace import TaskWorkspace


@sandbox_config()
class _FakeToolWithWorkspace(FakeBaseTool):
    """Fake tool with init params."""

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "fake_tool_ws"


@sandbox_config()
class _FakeToolNoParams(FakeBaseTool):
    """Fake tool with no init params."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "fake_tool_nop"


@sandbox_config()
class _AbstractToolMethodOwner(_FakeToolWithWorkspace):
    """Method owner that is also an AbstractBaseTool subclass."""

    def say(self, code: str = "") -> dict[str, Any]:
        return {"output": code}


@sandbox_config(packages=["sqlalchemy"], env_vars=["DB_URL"])
class _PlainMethodOwner:
    """Method owner that is a regular Python class, not a tool subclass."""

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self.workspace = workspace

    def say(self, code: str = "") -> dict[str, Any]:
        return {"output": code}


def _make_abstract_tool_bound_method_function_tool(
    workspace: Optional[TaskWorkspace] = None,
) -> FunctionTool:
    """Build a FunctionTool from a bound method on an AbstractBaseTool owner."""
    composite = _AbstractToolMethodOwner(workspace=workspace)
    return FunctionTool(composite.say, name="fake_bound_method")


def _make_plain_bound_method_function_tool(
    workspace: Optional[TaskWorkspace] = None,
) -> FunctionTool:
    """Build a FunctionTool from a bound method on a regular class owner."""
    composite = _PlainMethodOwner(workspace=workspace)
    return FunctionTool(composite.say, name="fake_plain_bound_method")


def _make_sandbox(name: str = "sandbox-test") -> MagicMock:
    """Return a MagicMock that mimics Sandbox instance."""
    sb = MagicMock()
    sb.name = name
    sb.write_file = AsyncMock()
    sb.exec = AsyncMock()
    return sb


def _create_test_wrapper(tool: AbstractBaseTool) -> SandboxedToolWrapper:
    """Create a SandboxedToolWrapper for a tool."""
    return SandboxedToolWrapper(tool, _make_sandbox())


class TestWrapperSandboxConfig:
    """Tests for wrapper-level sandbox config resolution."""

    def test_wrapper_uses_plain_owner_sandbox_config(self):
        """Wrapper should inherit sandbox config from a plain owner."""
        wrapper = SandboxedToolWrapper(
            _make_plain_bound_method_function_tool(),
            _make_sandbox(),
        )
        assert "sqlalchemy" in wrapper._requirements
        assert wrapper._env_vars == ["DB_URL"]


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


class TestBuildExecutionCommand:
    """Tests for _build_execution_command()."""

    def test_with_init_params(self):
        """Command should include init params when they exist."""
        wrapper = _create_test_wrapper(_FakeToolWithWorkspace(workspace=None))
        command = wrapper._build_execution_command(
            {"code": "print(1)"}, "/tmp/result.json"
        )
        assert command[:2] == [
            "python",
            f"{SANDBOX_SRC_ROOT}/xagent/core/tools/adapters/vibe/sandboxed_tool/tool_runner.py",
        ]
        assert "--execution-spec-b64" in command
        spec_idx = command.index("--execution-spec-b64")
        execution_spec = json.loads(
            base64.b64decode(command[spec_idx + 1]).decode("utf-8")
        )
        assert execution_spec == {
            "kind": "tool",
            "tool_class": (
                f"{_FakeToolWithWorkspace.__module__}:{_FakeToolWithWorkspace.__name__}"
            ),
        }
        assert "--init-params-b64" in command
        # Verify the b64 value can be deserialized
        idx = command.index("--init-params-b64")
        restored = cloudpickle.loads(base64.b64decode(command[idx + 1]))
        assert restored == {"workspace": None}

    def test_without_init_params(self):
        """Command should omit init params for no-arg tools."""
        wrapper = _create_test_wrapper(_FakeToolNoParams())
        command = wrapper._build_execution_command(
            {"code": "print(1)"}, "/tmp/result.json"
        )
        assert "--init-params-b64" not in command

    def test_functiontool_bound_method_auto_infers_method_execution(self):
        """Bound methods on tool subclasses should serialize as method execution."""
        wrapper = _create_test_wrapper(
            _make_abstract_tool_bound_method_function_tool(workspace=None)
        )
        command = wrapper._build_execution_command({"code": "x"}, "/tmp/result.json")
        spec_idx = command.index("--execution-spec-b64")
        execution_spec = json.loads(
            base64.b64decode(command[spec_idx + 1]).decode("utf-8")
        )
        assert execution_spec == {
            "kind": "method",
            "tool_class": (
                f"{_AbstractToolMethodOwner.__module__}:"
                f"{_AbstractToolMethodOwner.__name__}"
            ),
            "method_name": "say",
        }

    def test_plain_functiontool_bound_method_auto_infers_method_execution(self):
        """Bound methods on regular classes should serialize the same way."""
        wrapper = _create_test_wrapper(
            _make_plain_bound_method_function_tool(workspace=None)
        )
        command = wrapper._build_execution_command({"code": "x"}, "/tmp/result.json")
        spec_idx = command.index("--execution-spec-b64")
        execution_spec = json.loads(
            base64.b64decode(command[spec_idx + 1]).decode("utf-8")
        )
        assert execution_spec == {
            "kind": "method",
            "tool_class": (
                f"{_PlainMethodOwner.__module__}:{_PlainMethodOwner.__name__}"
            ),
            "method_name": "say",
        }


class TestBuildExecutionEnv:
    """Tests for _build_execution_env()."""

    def test_always_includes_pythonpath(self):
        wrapper = _create_test_wrapper(_FakeToolNoParams())
        env = wrapper._build_execution_env()
        assert env["PYTHONPATH"] == SANDBOX_SRC_ROOT

    def test_tool_env_vars_cannot_override_the_sandbox_seeds(self, monkeypatch):
        """A host PYTHONPATH would point the runner outside the sandbox."""
        from xagent.config import SANDBOX_TOOL_RUNNER

        monkeypatch.setenv("PYTHONPATH", "/host/src")
        monkeypatch.setenv(SANDBOX_TOOL_RUNNER, "0")
        wrapper = _create_test_wrapper(_FakeToolNoParams())
        wrapper._env_vars = ["PYTHONPATH", SANDBOX_TOOL_RUNNER]

        env = wrapper._build_execution_env()

        assert env["PYTHONPATH"] == SANDBOX_SRC_ROOT
        assert env[SANDBOX_TOOL_RUNNER] == "1"

    def test_picks_up_host_env(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret")
        wrapper = _create_test_wrapper(_FakeToolNoParams())
        wrapper._env_vars = ["MY_API_KEY"]
        env = wrapper._build_execution_env()
        assert env["MY_API_KEY"] == "secret"

    def test_missing_env_var_warns(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        wrapper = _create_test_wrapper(_FakeToolNoParams())
        wrapper._env_vars = ["NONEXISTENT_VAR"]

        with patch(
            "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper."
            "logger.warning"
        ) as mock_warning:
            env = wrapper._build_execution_env()

        assert "NONEXISTENT_VAR" not in env
        mock_warning.assert_called_once()
        assert "NONEXISTENT_VAR" in mock_warning.call_args.args[0]
