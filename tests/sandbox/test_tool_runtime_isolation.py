"""Execute the guest payload with no backend package or source available."""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import Tool as MCPTool

from xagent.core.tools.adapters.vibe.command_executor import CommandExecutorToolForBasic
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPToolAdapter
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    SandboxConfig,
    set_instance_sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
    build_code_mount_volumes,
)
from xagent.sandbox.tool_runtime.bundle import (
    build_runtime_archive,
    build_runtime_command,
)

# Emulate the clean image's third-party packages without executing the local
# editable-install .pth files, which would expose this checkout to the child.
_GUEST_PACKAGES = f"import sys; sys.path.extend({[p for p in sys.path if Path(p).name == 'site-packages']!r})\n"
_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def _invoke(request, tmp_path, *, deny_source=False, extra_path=None):
    result_file = tmp_path / "result.json"
    command = build_runtime_command(request, str(result_file))
    command[0] = sys.executable
    command[1] = "-S"
    prefix = _GUEST_PACKAGES
    if extra_path is not None:
        prefix += f"sys.path.insert(0, {str(extra_path)!r})\n"
    command[3] = prefix + command[3]
    if deny_source:
        profile = (
            f'(version 1)(allow default)(deny file-read* (subpath "{_SOURCE_ROOT}"))'
        )
        command = ["sandbox-exec", "-p", profile, *command]
    process = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=45
    )
    return process, json.loads(
        result_file.read_text()
    ) if result_file.exists() else None


def test_runtime_archive_contains_execution_code_only():
    with zipfile.ZipFile(io.BytesIO(build_runtime_archive())) as archive:
        names = set(archive.namelist())
        assert not any(name.startswith("xagent/") for name in names)
        assert not any(part in names for part in ("config.py", "web/", "tests/"))
        assert "tool_runtime/runner.py" in names
        for name in names:
            source = archive.read(name)
            assert b"from xagent" not in source
            assert b"import xagent" not in source
            assert b"JWT_SECRET_KEY" not in source
    assert build_code_mount_volumes() == []


def test_python_runs_and_collects_artifacts_without_platform_imports(tmp_path):
    code = """import importlib.util, pathlib, sys
assert importlib.util.find_spec('xagent') is None
assert not any(name.startswith('xagent.') for name in sys.modules)
assert pathlib.Path(WORKSPACE_OUTPUT_DIR) == pathlib.Path.cwd()
pathlib.Path('report.csv').write_text('answer\\n42\\n')
print(sum([20, 22]))
"""
    process, result = _invoke(
        {
            "kind": "python",
            "arguments": {"code": code},
            "working_directory": str(tmp_path),
            "workspace_variables": {"WORKSPACE_OUTPUT_DIR": str(tmp_path)},
        },
        tmp_path,
    )
    assert process.returncode == 0, process.stderr
    assert result["success"], result
    assert result["output"].strip() == "42"
    assert result["generated_files"] == ["report.csv"]
    assert result["file_refs"][0]["file_path"] == str(tmp_path / "report.csv")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_javascript_runs_and_collects_artifacts(tmp_path):
    process, result = _invoke(
        {
            "kind": "javascript",
            "arguments": {
                "code": "require('fs').writeFileSync('report.csv', 'answer\\n42'); "
                "console.log(6 * 7)",
            },
            "working_directory": str(tmp_path),
        },
        tmp_path,
    )
    assert process.returncode == 0, process.stderr
    assert result["success"], result
    assert result["output"].strip() == "42"
    assert result["generated_files"] == ["report.csv"]


@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None, reason="Requires macOS sandbox-exec"
)
@pytest.mark.parametrize("kind", ["command", "python", "javascript"])
def test_execution_succeeds_when_backend_source_reads_are_denied(tmp_path, kind):
    if kind == "javascript" and shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    protected = str(_SOURCE_ROOT / "xagent/web/auth_config.py")
    arguments = {
        "command": {
            "command": f"if cat {shlex.quote(protected)}; then exit 99; else printf blocked; fi"
        },
        "python": {
            "code": f"""from pathlib import Path
try:
    Path({protected!r}).read_text()
except PermissionError:
    print('blocked')
else:
    raise RuntimeError('backend source was readable')
"""
        },
        "javascript": {
            "code": f"""try {{
require('fs').readFileSync({json.dumps(protected)}); throw new Error('source readable'); }}
catch (e) {{ if (!['EPERM', 'EACCES'].includes(e.code)) throw e; console.log('blocked'); }}"""
        },
    }[kind]
    process, result = _invoke(
        {
            "kind": kind,
            "arguments": arguments,
            "working_directory": str(tmp_path),
        },
        tmp_path,
        deny_source=True,
    )
    assert process.returncode == 0, process.stderr
    assert result["success"], result
    assert result["output"].strip() == "blocked"


def test_guest_with_installed_platform_is_rejected_before_running_code(tmp_path):
    package = tmp_path / "legacy" / "xagent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    process, result = _invoke(
        {
            "kind": "command",
            "arguments": {"command": "touch should-not-exist"},
            "working_directory": str(tmp_path),
        },
        tmp_path,
        extra_path=package.parent,
    )
    assert process.returncode != 0
    assert "platform application is installed" in process.stderr
    assert result is None
    assert not (tmp_path / "should-not-exist").exists()


def test_stdio_mcp_lists_and_calls_without_backend_code(tmp_path):
    server = tmp_path / "server.py"
    server.write_text("""from mcp.server.fastmcp import FastMCP
mcp = FastMCP('isolated-tools')
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b
mcp.run(transport='stdio')
""")
    connection = {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-S", "-c", _GUEST_PACKAGES + f"exec(open({str(server)!r}).read())"],
        "env": {"PATH": os.environ.get("PATH", "")},
    }
    process, listed = _invoke({"kind": "mcp_list", "connection": connection}, tmp_path)
    assert process.returncode == 0, process.stderr
    assert [tool["name"] for tool in listed] == ["add"]
    (tmp_path / "result.json").unlink()
    process, called = _invoke(
        {
            "kind": "mcp_call",
            "connection": connection,
            "tool_name": "add",
            "arguments": {"a": 20, "b": 22},
            "meta": {},
        },
        tmp_path,
    )
    assert process.returncode == 0, process.stderr
    assert not called.get("isError", False)
    assert called["content"][0]["text"] == "42"


@pytest.mark.asyncio
async def test_mcp_authorization_runs_before_sandbox_transport():
    adapter = MCPToolAdapter(
        MCPTool(name="echo", inputSchema={"type": "object", "properties": {}}),
        {"transport": "stdio", "command": "npx", "args": []},
        allow_users=["allowed"],
    )
    set_instance_sandbox_config(adapter, SandboxConfig())
    wrapper = SandboxedToolWrapper(adapter, object())
    with (
        patch.object(adapter, "_get_current_user_id", return_value="denied"),
        patch.object(wrapper, "_execute_runtime_request", new=AsyncMock()) as execute,
    ):
        result = await wrapper.run_json_async({})
    assert result["is_error"]
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorized_mcp_normalizes_arguments_and_uses_guest_transport():
    adapter = MCPToolAdapter(
        MCPTool(
            name="echo",
            inputSchema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            },
        ),
        {
            "transport": "stdio",
            "command": "npx",
            "args": [],
            "_connector_runtime_refresh": lambda: None,
        },
        allow_users=["allowed"],
    )
    set_instance_sandbox_config(adapter, SandboxConfig())
    wrapper = SandboxedToolWrapper(adapter, object())
    payload = {"content": [{"type": "text", "text": "ok"}], "isError": False}
    with (
        patch.object(adapter, "_get_current_user_id", return_value="allowed"),
        patch.object(
            wrapper, "_execute_runtime_request", new=AsyncMock(return_value=payload)
        ) as execute,
    ):
        result = await wrapper.run_json_async({"count": 7})
        assert not result["is_error"]
        assert result["content"][0]["text"] == "ok"
    request = execute.await_args.args[0]
    assert request["kind"] == "mcp_call"
    assert request["arguments"] == {"count": 7}
    assert "_connector_runtime_refresh" not in request["connection"]


@pytest.mark.asyncio
async def test_wrapper_failure_never_runs_original_tool_on_backend():
    tool = CommandExecutorToolForBasic()
    with (
        patch(
            "xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper.create_sandboxed_tool",
            new=AsyncMock(side_effect=RuntimeError("unavailable")),
        ),
        pytest.raises(RuntimeError, match="backend execution is disabled"),
    ):
        await ToolFactory._wrap_sandbox_tools([tool], object())


def test_custom_mounts_cannot_reintroduce_backend_or_tests(tmp_path):
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    for source in (
        _SOURCE_ROOT,
        _SOURCE_ROOT.parent,
        _SOURCE_ROOT / "xagent/web",
        _SOURCE_ROOT.parent / "tests",
    ):
        with pytest.raises(SandboxRuntimeConflictError, match="Platform source"):
            _reject_platform_source_mounts([(str(source), "/opt/data", "ro")])
    alias = tmp_path / "aliased-code"
    alias.symlink_to(_SOURCE_ROOT, target_is_directory=True)
    with pytest.raises(SandboxRuntimeConflictError):
        _reject_platform_source_mounts([(str(alias), "/opt/data", "ro")])
    _reject_platform_source_mounts([(str(tmp_path), "/workspace", "rw")])


def test_host_side_checkout_mounts_are_rejected(monkeypatch):
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    monkeypatch.setenv("XAGENT_SANDBOX_HOST_PROJECT_ROOT", "/docker-host/project")
    with pytest.raises(SandboxRuntimeConflictError):
        _reject_platform_source_mounts(
            [("/docker-host/project/src", "/opt/code", "ro")]
        )


def test_uploads_root_cannot_exempt_a_mount_of_backend_source(monkeypatch):
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    monkeypatch.setattr("xagent.web.sandbox_manager.get_uploads_dir", lambda: Path("/"))
    with pytest.raises(SandboxRuntimeConflictError):
        _reject_platform_source_mounts([(str(_SOURCE_ROOT), "/data", "ro")])


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [True, False])
async def test_old_boxlite_guest_is_not_resumed_with_legacy_or_unknown_mounts(stored):
    pytest.importorskip("boxlite")
    import asyncio
    from unittest.mock import MagicMock

    from xagent.sandbox.base import SandboxConfig as GuestConfig
    from xagent.sandbox.base import (
        SandboxInfo,
        SandboxTemplate,
    )
    from xagent.sandbox.boxlite_sandbox import BoxliteSandboxService, MemBoxliteStore

    service = object.__new__(BoxliteSandboxService)
    service._locks = {}
    service._locks_lock = asyncio.Lock()
    service._runtime = MagicMock()
    service._runtime.get = AsyncMock(return_value=object())
    service._runtime.remove = AsyncMock()
    service._store = MemBoxliteStore()
    if stored:
        service._store.add_info(
            "legacy",
            SandboxInfo(
                name="legacy",
                state="stopped",
                template=SandboxTemplate(),
                config=GuestConfig(volumes=[("/backend/src", "/app/src", "ro")]),
            ),
        )
    with (
        patch(
            "xagent.sandbox.boxlite_sandbox._create_or_reuse_box",
            new=AsyncMock(side_effect=RuntimeError("fresh creation")),
        ) as resume,
        pytest.raises(RuntimeError, match="fresh creation"),
    ):
        await service.get_or_create("legacy", config=GuestConfig())
    service._runtime.remove.assert_awaited_once_with("legacy", force=True)
    assert not resume.await_args.args[2].volumes


@pytest.mark.asyncio
async def test_configured_sandbox_missing_fails_before_tool_creation(monkeypatch):
    from xagent.core.tools.adapters.vibe.config import ToolConfig

    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    config = ToolConfig({"workspace": {"task_id": "live-task"}})
    with pytest.raises(RuntimeError, match="backend execution is disabled"):
        await ToolFactory.create_all_tools(config)


@pytest.mark.asyncio
async def test_shell_outputs_keep_host_registration_and_bulk_limits(tmp_path):
    from xagent.core.workspace import TaskWorkspace

    workspace = TaskWorkspace(id="command-isolation", base_dir=str(tmp_path))
    tool = CommandExecutorToolForBasic(workspace)
    wrapper = SandboxedToolWrapper(tool, object())

    async def execute(_request):
        (workspace.output_dir / "report.csv").write_text("answer\n42\n")
        return {"success": True, "output": "", "error": "", "return_code": 0}

    with patch.object(wrapper, "_execute_runtime_request", side_effect=execute):
        result = await wrapper.run_json_async({"command": "create report"})
    assert result["generated_files"] == ["report.csv"]
    assert result["file_refs"][0]["file_id"]
