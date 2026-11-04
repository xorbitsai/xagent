"""Tests for the execute_python_code wall-clock limit on the host path."""

import os
import time
from unittest.mock import patch

import pytest

from xagent.core.tools.adapters.vibe import python_executor as python_executor_module
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorTool
from xagent.core.workspace import TaskWorkspace

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="process-group kill is POSIX-only"
)


def test_runaway_code_returns_timeout_error(monkeypatch):
    monkeypatch.setenv("XAGENT_PYTHON_EXECUTOR_TIMEOUT_SECONDS", "1")
    started = time.monotonic()

    result = PythonExecutorTool().run_json_sync({"code": "while True:\n    pass"})

    assert time.monotonic() - started < 10
    assert result["success"] is False
    assert "timed out after 1 seconds" in result["error"]


def test_timeout_returns_no_artifact_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_PYTHON_EXECUTOR_TIMEOUT_SECONDS", "1")
    workspace = TaskWorkspace("test_python_timeout", str(tmp_path))

    result = PythonExecutorTool(workspace=workspace).run_json_sync(
        {"code": "open('partial.txt', 'w').write('x')\nwhile True:\n    pass"}
    )

    assert result["success"] is False
    assert result["generated_files"] == []
    assert result["artifacts"] == []


def test_host_execution_runs_in_subprocess_with_configured_limit(monkeypatch):
    monkeypatch.setenv("XAGENT_PYTHON_EXECUTOR_TIMEOUT_SECONDS", "77")

    with patch.object(
        python_executor_module,
        "run_python_code_in_subprocess",
        return_value={"success": True, "output": "ok", "error": ""},
    ) as run_in_subprocess:
        result = PythonExecutorTool().run_json_sync({"code": "print('ok')"})

    assert result["output"] == "ok"
    assert run_in_subprocess.call_args.kwargs["timeout_seconds"] == 77


def test_sandbox_runner_executes_in_process(monkeypatch):
    monkeypatch.setattr(python_executor_module, "in_sandbox_tool_runner", lambda: True)

    with patch.object(
        python_executor_module, "run_python_code_in_subprocess"
    ) as run_in_subprocess:
        result = PythonExecutorTool().run_json_sync({"code": "print('inline')"})

    run_in_subprocess.assert_not_called()
    assert result["success"] is True
    assert "inline" in result["output"]
