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
)
from xagent.sandbox.tool_runtime.bundle import (
    build_runtime_archive,
    build_runtime_command,
)

# Emulate the clean image's third-party packages without executing the local
# editable-install .pth files, which would expose this checkout to the child.
_GUEST_PACKAGES = f"import sys; sys.path.extend({[p for p in sys.path if Path(p).name == 'site-packages']!r})\n"
_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def _invoke(request, tmp_path, *, deny_source=False, extra_path=None, user_site=None):
    result_file = tmp_path / "result.json"
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request))
    command = build_runtime_command(
        request_file=str(request_file), result_file=str(result_file)
    )
    command[0] = sys.executable
    command[1] = "-S"
    prefix = _GUEST_PACKAGES
    if extra_path is not None:
        prefix += f"sys.path.insert(0, {str(extra_path)!r})\n"
    command[3] = prefix + command[3]
    if user_site is not None:
        command[3] = (
            f"import site; site.getusersitepackages = lambda: {str(user_site)!r}\n"
            + command[3]
        )
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
        assert not any(
            set(Path(name).parts) & {"config.py", "web", "tests", "providers", "agent"}
            for name in names
        )
        assert "tool_runtime/runner.py" in names
        for name in names:
            source = archive.read(name)
            assert b"from xagent" not in source
            assert b"import xagent" not in source
            assert b"JWT_SECRET_KEY" not in source


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
        patch.object(
            wrapper._mcp_target, "_get_current_user_id", return_value="denied"
        ),
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
        patch.object(
            wrapper._mcp_target, "_get_current_user_id", return_value="allowed"
        ),
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
async def test_configured_sandbox_missing_fails_before_tool_creation(monkeypatch):
    from xagent.core.tools.adapters.vibe.config import ToolConfig

    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    config = ToolConfig({"workspace": {"task_id": "live-task"}})
    with (
        patch(
            "xagent.core.tools.adapters.vibe.factory.ToolRegistry.create_registered_tools",
            new=AsyncMock(return_value=[CommandExecutorToolForBasic()]),
        ),
        pytest.raises(RuntimeError, match="backend execution is disabled"),
    ):
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

    with (
        patch.object(wrapper, "_execute_runtime_request", side_effect=execute),
        patch.object(
            workspace, "register_file", wraps=workspace.register_file
        ) as register,
    ):
        result = await wrapper.run_json_async({"command": "create report"})
    assert result["generated_files"] == ["report.csv"]
    assert result["file_refs"][0]["file_id"]
    register.assert_called_once()


@pytest.mark.asyncio
async def test_shared_mcp_adapter_keeps_each_wrappers_sandbox_transport():
    adapter = MCPToolAdapter(
        MCPTool(name="echo", inputSchema={"type": "object", "properties": {}}),
        {"transport": "stdio", "command": "npx", "args": []},
    )
    set_instance_sandbox_config(adapter, SandboxConfig())
    first = SandboxedToolWrapper(adapter, object())
    second = SandboxedToolWrapper(adapter, object())
    response = {"content": [], "isError": False}
    with (
        patch.object(
            first, "_execute_runtime_request", new=AsyncMock(return_value=response)
        ) as a,
        patch.object(
            second, "_execute_runtime_request", new=AsyncMock(return_value=response)
        ) as b,
    ):
        await first.run_json_async({})
        a.assert_awaited_once()
        b.assert_not_awaited()
        await second.run_json_async({})
        b.assert_awaited_once()
    assert adapter._sandbox_transport is None


@pytest.mark.parametrize("directory", ["web", "core"])
def test_uploads_configuration_cannot_exempt_application_modules(
    monkeypatch, directory
):
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    source = _SOURCE_ROOT / "xagent" / directory
    monkeypatch.setattr("xagent.web.sandbox_manager.get_uploads_dir", lambda: source)
    with pytest.raises(SandboxRuntimeConflictError):
        _reject_platform_source_mounts([(str(source), "/data", "rw")])


def test_fixed_legacy_uploads_subtree_remains_mountable(monkeypatch):
    from xagent.config import get_legacy_source_uploads_dir
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    data = get_legacy_source_uploads_dir()
    monkeypatch.setattr("xagent.web.sandbox_manager.get_uploads_dir", lambda: data)
    _reject_platform_source_mounts([(str(data / "user_1"), "/workspace", "rw")])


def test_legacy_uploads_symlink_cannot_exempt_source(tmp_path, monkeypatch):
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    source = _SOURCE_ROOT / "xagent" / "web"
    alias = tmp_path / "uploads"
    alias.symlink_to(source, target_is_directory=True)
    monkeypatch.setattr("xagent.web.sandbox_manager.get_uploads_dir", lambda: alias)
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_legacy_source_uploads_dir", lambda: alias
    )
    with pytest.raises(SandboxRuntimeConflictError):
        _reject_platform_source_mounts([(str(alias), "/data", "rw")])


def test_every_bundled_module_imports_from_zip_without_backend(tmp_path):
    process, result = _invoke(
        {
            "kind": "python",
            "arguments": {
                "code": """import importlib, pkgutil, tool_runtime
modules = sorted(item.name for item in pkgutil.walk_packages(tool_runtime.__path__, 'tool_runtime.'))
for name in modules:
    module = importlib.import_module(name)
    assert '.runtime.pyz/' in module.__file__, (name, module.__file__)
assert 'tool_runtime.command_executor' in modules
assert 'tool_runtime.chrome_daemon_runner' in modules
assert 'tool_runtime.mcp.sessions' in modules
print('all imported')
"""
            },
            "working_directory": str(tmp_path),
        },
        tmp_path,
    )
    assert process.returncode == 0, process.stderr
    assert result["success"], result
    assert result["output"].strip() == "all imported"
    assert not list(tmp_path.glob("*.runtime.pyz"))


def test_native_command_runs_in_isolated_zip_on_all_platforms(tmp_path):
    process, result = _invoke(
        {
            "kind": "command",
            "arguments": {"command": "printf native-command"},
            "working_directory": str(tmp_path),
        },
        tmp_path,
    )
    assert process.returncode == 0, process.stderr
    assert result["success"], result
    assert result["output"] == "native-command"


def test_user_installed_xagent_does_not_brick_existing_guest(tmp_path):
    user_site = tmp_path / "user-site"
    package = user_site / "xagent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    process, result = _invoke(
        {
            "kind": "python",
            "arguments": {"code": "print('still usable')"},
            "working_directory": str(tmp_path),
        },
        tmp_path,
        user_site=user_site,
    )
    assert process.returncode == 0, process.stderr
    assert result["success"]
    assert result["output"].strip() == "still usable"


@pytest.mark.parametrize("origin", ["cwd", "installation"])
def test_noneditable_install_also_protects_retained_build_checkout(
    tmp_path, monkeypatch, origin
):
    from types import SimpleNamespace

    from xagent import config
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    installed = tmp_path / "venv" / "site-packages" / "xagent"
    installed.mkdir(parents=True)
    monkeypatch.setattr(config, "__file__", str(installed / "config.py"))
    checkout = tmp_path / "checkout"
    (checkout / "src" / "xagent").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text('[project]\nname = "xagent-ai"\n')
    monkeypatch.chdir(checkout if origin == "cwd" else tmp_path)
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda name: SimpleNamespace(
            read_text=lambda filename: json.dumps({"url": checkout.as_uri()})
            if origin == "installation"
            else None,
        ),
    )
    for source in (
        installed,
        checkout / "src",
        checkout / "tests",
        checkout / ".git",
        checkout / ".git" / "objects",
    ):
        with pytest.raises(SandboxRuntimeConflictError):
            _reject_platform_source_mounts([(str(source), "/data", "ro")])
    assert tmp_path / "venv" / "tests" not in config.get_platform_source_roots()


def test_case_alias_cannot_exempt_platform_source(tmp_path, monkeypatch):
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    root = tmp_path / "BackendSource"
    root.mkdir()
    alias = tmp_path / "backendsource"
    if not alias.exists():
        pytest.skip("Filesystem has case-sensitive paths")
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_platform_source_roots", lambda: [root]
    )
    with pytest.raises(SandboxRuntimeConflictError):
        _reject_platform_source_mounts([(str(alias / "web"), "/data", "ro")])


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["no_tools", "safe_only"])
async def test_configured_sandbox_allows_tasks_without_selected_code_tools(
    monkeypatch, selection
):
    from xagent.core.tools.adapters.vibe.config import ToolConfig
    from xagent.core.tools.adapters.vibe.function import FunctionTool

    monkeypatch.setenv("SANDBOX_ENABLED", "true")

    def add(a: int, b: int) -> int:
        return a + b

    safe = FunctionTool(add)
    code = CommandExecutorToolForBasic()
    config = ToolConfig(
        {
            "workspace": {"task_id": "preview_test"},
            "allowed_tools": [] if selection == "no_tools" else [safe.name],
        }
    )
    with (
        patch(
            "xagent.core.tools.adapters.vibe.factory.ToolRegistry.create_registered_tools",
            new=AsyncMock(return_value=[safe, code]),
        ),
        patch.object(
            ToolFactory,
            "_apply_output_filters",
            side_effect=lambda tools, _config: tools,
        ),
    ):
        tools = await ToolFactory.create_all_tools(config)
    assert tools == ([] if selection == "no_tools" else [safe])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["python", "javascript"])
async def test_executor_result_contract_is_normalized_on_host(kind):
    from xagent.core.tools.adapters.vibe.javascript_executor import (
        JavaScriptExecutorToolForBasic,
    )
    from xagent.core.tools.adapters.vibe.python_executor import (
        PythonExecutorToolForBasic,
    )

    tool = (
        PythonExecutorToolForBasic()
        if kind == "python"
        else JavaScriptExecutorToolForBasic()
    )
    wrapper = SandboxedToolWrapper(tool, object())
    with patch.object(
        wrapper,
        "_run_json_in_sandbox",
        new=AsyncMock(return_value={"success": False, "error": "Execution timed out"}),
    ):
        result = await wrapper.run_json_async({})
    assert result == {
        "success": False,
        "output": "",
        "error": "Execution timed out",
        "generated_files": [],
        "file_refs": [],
        "artifacts": [],
    }


@pytest.mark.asyncio
async def test_javascript_generated_file_suffix_survives_registration_failure():
    from xagent.core.tools.adapters.vibe.javascript_executor import (
        JavaScriptExecutorToolForBasic,
    )

    wrapper = SandboxedToolWrapper(JavaScriptExecutorToolForBasic(), object())
    payload = {
        "success": True,
        "output": "done",
        "generated_files": ["report.csv"],
        "file_refs": [{"filename": "report.csv", "file_path": "/guest/report.csv"}],
        "artifacts": [],
    }
    with (
        patch.object(
            wrapper, "_run_json_in_sandbox", new=AsyncMock(return_value=payload)
        ),
        patch.object(
            wrapper,
            "_register_sandbox_outputs",
            new=AsyncMock(side_effect=RuntimeError("registration unavailable")),
        ),
    ):
        result = await wrapper.run_json_async({})
    assert result["output"] == "done\n\nGenerated files: report.csv"
    assert len(result["artifacts"]) == len(result["file_refs"]) == 1


@pytest.mark.skipif(
    shutil.which("git") is None, reason="Requires Git for the object-store regression"
)
@pytest.mark.parametrize("layout", ["directory", "separate", "worktree", "alternates"])
@pytest.mark.parametrize("origin", ["cwd", "host"])
def test_git_object_store_mounts_cannot_recover_platform_source(
    tmp_path, monkeypatch, layout, origin
):
    from xagent.sandbox.base import SandboxRuntimeConflictError
    from xagent.web.sandbox_manager import _reject_platform_source_mounts

    def git(*args, cwd=tmp_path):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Sandbox Test",
                "-c",
                "user.email=sandbox-test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-C",
                str(cwd),
                *args,
            ],
            env={
                **os.environ,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            },
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    checkout = tmp_path / "checkout"
    source = checkout / "src" / "xagent" / "fixture.py"
    source.parent.mkdir(parents=True)
    source_text = "backend_fixture = 'source recovered from Git objects'\n"
    source.write_text(source_text)
    (checkout / "pyproject.toml").write_text('[project]\nname = "xagent-ai"\n')
    init_args = ["init", "--quiet"]
    if layout == "separate":
        init_args += ["--separate-git-dir", str(tmp_path / "metadata")]
    git(*init_args, str(checkout))
    git("add", "pyproject.toml", "src", cwd=checkout)
    git("commit", "--quiet", "-m", "Fixture", cwd=checkout)
    if layout == "worktree":
        linked = tmp_path / "linked"
        git("worktree", "add", "--quiet", "--detach", str(linked), "HEAD", cwd=checkout)
        checkout = linked

    git_dir = Path(git("rev-parse", "--absolute-git-dir", cwd=checkout))
    common_dir = (
        checkout / git("rev-parse", "--git-common-dir", cwd=checkout)
    ).resolve()
    stores = {checkout / ".git", git_dir, common_dir, common_dir / "objects"}
    if layout == "alternates":
        objects = common_dir / "objects"
        pool = tmp_path / "shared objects-λ"
        shutil.move(str(objects), str(pool))
        (objects / "info").mkdir(parents=True)
        quoted = '"' + "".join(f"\\{byte:03o}" for byte in os.fsencode(str(pool))) + '"'
        (objects / "info" / "alternates").write_text(quoted + "\n")
        stores.add(pool)

    source.unlink(missing_ok=True)
    (checkout / "src" / "xagent" / "fixture.py").unlink(missing_ok=True)

    # This reads only the fixture's object database, not a mounted worktree:
    # a read-only metadata mount can supply the same tracked backend bytes.
    assert (
        git("--git-dir", str(common_dir), "show", "HEAD:src/xagent/fixture.py")
        == source_text.strip()
    )
    if origin == "host":
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XAGENT_SANDBOX_HOST_PROJECT_ROOT", str(checkout))
    else:
        monkeypatch.chdir(checkout)
        monkeypatch.delenv("XAGENT_SANDBOX_HOST_PROJECT_ROOT", raising=False)
    for store in stores:
        with pytest.raises(SandboxRuntimeConflictError, match="Platform source"):
            _reject_platform_source_mounts([(str(store), "/repository-metadata", "ro")])
    alias = tmp_path / "metadata-alias"
    alias.symlink_to(common_dir, target_is_directory=True)
    with pytest.raises(SandboxRuntimeConflictError):
        _reject_platform_source_mounts([(str(alias), "/repository-metadata", "ro")])
