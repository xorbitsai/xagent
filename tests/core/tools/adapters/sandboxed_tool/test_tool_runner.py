"""Tests for tool_runner.py helper functions and main()."""

import argparse
import base64
import json
import subprocess
import sys
import textwrap
from typing import Any, Mapping
from unittest.mock import patch

import cloudpickle
import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.tool_runner import (
    _execute_from_spec,
    _load_args,
    _load_execution_spec,
    _load_init_params,
    _load_tool_class,
    _run_method,
    _run_tool,
    _validate_spec,
    main,
)


class _FakeTool:
    """Minimal fake tool for testing tool_runner."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def run_json_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"echo": args.get("msg", "")}

    async def run_json_async(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"echo": args.get("msg", "")}


class _FakeMethodOwner:
    """Simple method owner used to test method-based execution specs."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def say(self, msg: str) -> dict[str, Any]:
        return {"echo": f"{self.prefix}{msg}"}


class TestLoadArgs:
    """Tests for _load_args()."""

    def test_roundtrip(self):
        """Base64-encoded JSON should decode back to original dict."""
        original = {"msg": "hello", "count": 42}
        b64 = base64.b64encode(json.dumps(original).encode()).decode()
        assert _load_args(b64) == original


class TestLoadInitParams:
    """Tests for _load_init_params()."""

    def test_none_returns_empty(self):
        """None input should return empty dict."""
        assert _load_init_params(None) == {}

    def test_roundtrip(self):
        """Cloudpickle-serialized params should deserialize correctly."""
        params = {"key": "value"}
        b64 = base64.b64encode(cloudpickle.dumps(params)).decode()
        assert _load_init_params(b64) == params


class TestLoadToolClass:
    """Tests for _load_tool_class()."""

    def test_valid_import(self):
        """Valid import path should resolve to the correct class."""
        cls = _load_tool_class(
            "tests.core.tools.adapters.sandboxed_tool.test_tool_runner:_FakeTool"
        )
        assert cls.__name__ == "_FakeTool"

    def test_invalid_module(self):
        """Non-existent module should raise ModuleNotFoundError."""
        with pytest.raises(ModuleNotFoundError):
            _load_tool_class("no.such.module:Cls")

    def test_mcp_tool_adapter_loads_without_pypinyin(self):
        """Regression test for the reported sandbox failure (PR #1710).

        A reduced sandbox never installs pypinyin (only mcp/pydantic/
        cloudpickle -- see SANDBOX_BASE_DEPENDENCIES), yet every
        sandboxed MCP tool call must still import mcp_adapter:
        MCPToolAdapter through this exact function, because
        mcp_adapter.py transitively imports agent_tool_names.py for an
        unrelated constant that agent_tool_names.py used to require
        pypinyin just to define. Runs in a subprocess -- rather than
        monkeypatching sys.modules/builtins.__import__ in-process -- so
        blocking "pypinyin" at the import-system level can't be
        undone by pypinyin already sitting in sys.modules from an
        earlier test in this same process, and so this exercises the
        real, unmodified import machinery tool_runner.py itself runs
        under in a sandbox.
        """
        script = textwrap.dedent(
            """
            import sys

            class _BlockPypinyin:
                def find_spec(self, name, path=None, target=None):
                    if name == "pypinyin" or name.startswith("pypinyin."):
                        raise ModuleNotFoundError(f"No module named {name!r}")
                    return None

            sys.meta_path.insert(0, _BlockPypinyin())

            from xagent.core.tools.adapters.vibe.sandboxed_tool.tool_runner import (
                _load_tool_class,
            )

            cls = _load_tool_class(
                "xagent.core.tools.adapters.vibe.mcp_adapter:MCPToolAdapter"
            )
            assert cls.__name__ == "MCPToolAdapter"

            from xagent.core.tools.adapters.vibe import agent_tool_names

            assert agent_tool_names.lazy_pinyin is None

            ascii_name = agent_tool_names.gen_workforce_agent_tool_name(
                1, "ASCII Name"
            )
            assert ascii_name.isascii()
            assert agent_tool_names.parse_agent_tool_id(ascii_name) == 1

            print("REGRESSION_TEST_OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "REGRESSION_TEST_OK" in result.stdout


class TestLoadExecutionSpec:
    def test_from_tool_execution_spec_b64(self):
        """Tool execution spec load."""
        execution_spec_b64 = base64.b64encode(
            json.dumps({"kind": "tool", "tool_class": "a.b:Tool"}).encode()
        ).decode()
        parsed = argparse.Namespace(execution_spec_b64=execution_spec_b64)
        assert _load_execution_spec(parsed) == {
            "kind": "tool",
            "tool_class": "a.b:Tool",
        }

    def test_from_method_execution_spec_b64(self):
        """Method execution spec load."""
        execution_spec_b64 = base64.b64encode(
            json.dumps(
                {"kind": "method", "tool_class": "a.b:Tool", "method_name": "run"}
            ).encode()
        ).decode()
        parsed = argparse.Namespace(execution_spec_b64=execution_spec_b64)
        assert _load_execution_spec(parsed) == {
            "kind": "method",
            "tool_class": "a.b:Tool",
            "method_name": "run",
        }


class TestRunTool:
    """Tests for _run_tool()."""

    def test_sync(self):
        """Sync tool should return result directly."""
        tool = _FakeTool()
        assert _run_tool(tool, {"msg": "hi"}) == {"echo": "hi"}


class TestRunMethod:
    def test_sync(self):
        """Sync bound methods should receive decoded kwargs and return output."""
        owner = _FakeMethodOwner(prefix="hello ")
        assert _run_method(owner.say, {"msg": "world"}) == {"echo": "hello world"}


class TestExecuteFromSpec:
    def test_tool_spec(self):
        """Tool specs should reconstruct the class and call run_json_*."""
        spec = {
            "kind": "tool",
            "tool_class": "tests.core.tools.adapters.sandboxed_tool.test_tool_runner:_FakeTool",
        }
        result = _execute_from_spec(spec, {}, {"msg": "ok"})
        assert result == {"echo": "ok"}

    def test_method_spec(self):
        """Method specs should reconstruct the class and call the target method."""
        spec = {
            "kind": "method",
            "tool_class": (
                "tests.core.tools.adapters.sandboxed_tool.test_tool_runner:"
                "_FakeMethodOwner"
            ),
            "method_name": "say",
        }
        result = _execute_from_spec(spec, {"prefix": "hi "}, {"msg": "there"})
        assert result == {"echo": "hi there"}


class TestValidateSpec:
    """Tests for _validate_spec()."""

    def test_unsupported_kind(self):
        with pytest.raises(ValueError, match="Unsupported execution kind"):
            _validate_spec({"kind": "unknown", "tool_class": "a:B"})

    def test_missing_kind(self):
        with pytest.raises(ValueError, match="Unsupported execution kind"):
            _validate_spec({"tool_class": "a:B"})

    def test_missing_tool_class(self):
        with pytest.raises(ValueError, match="missing required key 'tool_class'"):
            _validate_spec({"kind": "tool"})

    def test_method_missing_method_name(self):
        with pytest.raises(ValueError, match="missing required key 'method_name'"):
            _validate_spec({"kind": "method", "tool_class": "a:B"})


class TestMain:
    """Tests for main() entrypoint."""

    def test_happy_path(self, tmp_path):
        """Successful execution should write result JSON to file."""
        result_file = str(tmp_path / "result.json")
        args_b64 = base64.b64encode(json.dumps({"msg": "ok"}).encode()).decode()
        execution_spec = {
            "kind": "tool",
            "tool_class": (
                "tests.core.tools.adapters.sandboxed_tool.test_tool_runner:_FakeTool"
            ),
        }
        execution_spec_b64 = base64.b64encode(
            json.dumps(execution_spec).encode()
        ).decode()
        argv = [
            "--execution-spec-b64",
            execution_spec_b64,
            "--args-b64",
            args_b64,
            "--result-file",
            result_file,
        ]
        with patch("sys.argv", ["tool_runner"] + argv):
            main()
        result = json.loads((tmp_path / "result.json").read_text())
        assert result == {"echo": "ok"}

    def test_bad_module_raises(self, tmp_path):
        """Invalid tool class should raise as Sandbox config error."""
        result_file = str(tmp_path / "result.json")
        args_b64 = base64.b64encode(b"{}").decode()
        execution_spec_b64 = base64.b64encode(
            json.dumps({"kind": "tool", "tool_class": "no.such.module:Cls"}).encode()
        ).decode()
        argv = [
            "--execution-spec-b64",
            execution_spec_b64,
            "--args-b64",
            args_b64,
            "--result-file",
            result_file,
        ]
        with patch("sys.argv", ["tool_runner"] + argv):
            with pytest.raises(ModuleNotFoundError):
                main()

    def test_method_happy_path(self, tmp_path):
        """Method execution should round-trip through the CLI entrypoint."""
        result_file = str(tmp_path / "result.json")
        args_b64 = base64.b64encode(json.dumps({"msg": "tool"}).encode()).decode()
        init_params_b64 = base64.b64encode(
            cloudpickle.dumps({"prefix": "from "})
        ).decode()
        execution_spec_b64 = base64.b64encode(
            json.dumps(
                {
                    "kind": "method",
                    "tool_class": (
                        "tests.core.tools.adapters.sandboxed_tool.test_tool_runner:"
                        "_FakeMethodOwner"
                    ),
                    "method_name": "say",
                }
            ).encode()
        ).decode()
        argv = [
            "--execution-spec-b64",
            execution_spec_b64,
            "--args-b64",
            args_b64,
            "--result-file",
            result_file,
            "--init-params-b64",
            init_params_b64,
        ]
        with patch("sys.argv", ["tool_runner"] + argv):
            main()
        result = json.loads((tmp_path / "result.json").read_text())
        assert result == {"echo": "from tool"}
