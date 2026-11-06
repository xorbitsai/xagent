"""Deliver a fixed, minimal tool runtime without mounting backend source.

Only the explicitly listed execution implementations cross into the guest.
Never discover dependencies recursively: doing that would silently expand the
guest's access when a tool starts importing a backend module.
"""

from __future__ import annotations

import base64
import io
import zipfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

_RUNTIME_FILES = {
    "runner.py": "sandbox/tool_runtime/runner.py",
    "command_executor.py": "core/tools/core/command_executor.py",
    "command_policy.py": "core/tools/core/command_policy.py",
    "python_executor.py": "core/tools/core/python_executor.py",
    "javascript_executor.py": "core/tools/core/javascript_executor.py",
    "mcp/sessions.py": "core/tools/core/mcp/sessions.py",
    "mcp/tools.py": "core/tools/core/mcp/tools.py",
    "chrome_daemon_runner.py": "core/tools/adapters/vibe/sandboxed_tool/chrome_daemon_runner.py",
}

# -I ignores PYTHONPATH and the caller's cwd. Restore the guest's user-installed
# dependencies explicitly: Boxlite executes as the sandbox user, whose declared
# packages may have been installed with pip --user. No backend path is added.
_BOOTSTRAP = """import base64, importlib.util, os, pathlib, runpy, site, sys
if any(pathlib.Path(path).exists() for path in ('/app/src', '/app/tests')):
    raise RuntimeError('Legacy source is visible; recreate this sandbox before execution')
if importlib.util.find_spec('xagent') is not None:
    raise RuntimeError('The platform application is installed; use a clean tool image')
site.addsitedir(site.getusersitepackages())
archive = pathlib.Path(sys.argv[-1] + '.runtime.pyz')
try:
    with os.fdopen(os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
        output.write(base64.b64decode(sys.argv[1]))
    sys.argv = [str(archive), *sys.argv[2:]]
    runpy.run_path(str(archive), run_name='__main__')
finally:
    archive.unlink(missing_ok=True)
"""


def sandbox_connection(connection: Mapping[str, Any]) -> dict[str, Any]:
    """Send stdio transport settings, excluding host-only refresh callbacks."""
    if connection.get("transport") != "stdio":
        raise ValueError("The isolated MCP runtime only accepts stdio connections")
    fields = {
        "transport",
        "command",
        "args",
        "env",
        "cwd",
        "encoding",
        "encoding_error_handler",
        "session_kwargs",
    }
    result = {key: value for key, value in connection.items() if key in fields}
    if isinstance(result.get("cwd"), Path):
        result["cwd"] = str(result["cwd"])
    return result


@lru_cache(maxsize=1)
def build_runtime_archive() -> bytes:
    """Build a zipapp from reviewed execution-only sources."""
    source_root = Path(__file__).resolve().parents[2]
    files = {
        "__main__.py": "from tool_runtime.runner import main\nmain()\n",
        "tool_runtime/__init__.py": "",
        "tool_runtime/mcp/__init__.py": "",
    }
    files.update(
        {
            f"tool_runtime/{target}": (source_root / source).read_text(encoding="utf-8")
            for target, source in _RUNTIME_FILES.items()
        }
    )
    from ...core.tools.artifacts import GENERATED_ARTIFACT_EXTENSIONS

    files["tool_runtime/artifact_types.py"] = (
        f"GENERATED_ARTIFACT_EXTENSIONS = frozenset({sorted(GENERATED_ARTIFACT_EXTENSIONS)!r})\n"
    )
    javascript = files["tool_runtime/javascript_executor.py"]
    dependency = "from ..artifacts import GENERATED_ARTIFACT_EXTENSIONS"
    if dependency not in javascript:
        raise RuntimeError("Review JavaScript runtime dependencies before packaging")
    files["tool_runtime/javascript_executor.py"] = javascript.replace(
        dependency,
        "from .artifact_types import GENERATED_ARTIFACT_EXTENSIONS",
        1,
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return output.getvalue()


def build_runtime_command(*, request_file: str, result_file: str) -> list[str]:
    """Run private JSON input; argv carries only public execution runtime code.

    The caller also removes ``result_file + '.runtime.pyz'`` in its cleanup
    scope so SIGKILL cannot leave an archive in a long-lived sandbox.
    """
    return [
        "python",
        "-I",
        "-c",
        _BOOTSTRAP,
        base64.b64encode(build_runtime_archive()).decode("ascii"),
        "--request-file",
        request_file,
        "--result-file",
        result_file,
    ]
