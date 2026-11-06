"""Execution-only guest entrypoint. Must not import the Xagent application.

The host validates tool schemas, permissions and connector bindings. Requests
contain only primitive execution data; application objects, pickle payloads and
dynamic Python import paths are never accepted by this entrypoint.
"""

from __future__ import annotations

# mypy: ignore-errors
# bundle.py packages this file with execution-only sibling modules that are
# intentionally absent from the backend package tree.
import argparse
import asyncio
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from .artifact_types import GENERATED_ARTIFACT_EXTENSIONS


def _snapshot(root: str | None) -> dict[Path, tuple[int, int]]:
    files: dict[Path, tuple[int, int]] = {}
    if root is None:
        return files
    directory = Path(root)
    for path in directory.rglob("*"):
        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or any(
                    part.startswith(".") for part in path.relative_to(directory).parts
                )
                or path.suffix.lower() not in GENERATED_ARTIFACT_EXTENSIONS
            ):
                continue
            stat = path.stat()
            files[path] = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            continue
    return files


async def _mcp(request: dict[str, Any]) -> Any:
    from .mcp.sessions import create_session
    from .mcp.tools import (
        SANDBOX_RAW_ANNOTATIONS_KEY,
        _list_all_tools,
        raw_annotations_for,
    )

    async with create_session(request["connection"]) as session:
        await session.initialize()
        if request["kind"] == "mcp_list":
            serialized = []
            for tool in await _list_all_tools(session):
                item = tool.model_dump(mode="json")
                raw = raw_annotations_for(tool)
                if raw is not None:
                    item[SANDBOX_RAW_ANNOTATIONS_KEY] = raw
                serialized.append(item)
            return serialized
        result = await session.call_tool(
            request["tool_name"],
            request["arguments"],
            meta=request.get("meta") or None,
        )
        return result.model_dump(mode="json", by_alias=True)


def execute(request: dict[str, Any]) -> Any:
    """Run one of the finite, host-selected execution operations."""
    kind = request["kind"]
    if kind in {"mcp_list", "mcp_call"}:
        return asyncio.run(_mcp(request))

    working_directory = request.get("working_directory")
    args = request["arguments"]
    before = _snapshot(working_directory) if kind in {"python", "javascript"} else {}
    if kind == "command":
        from .command_executor import CommandExecutorCore

        return CommandExecutorCore(working_directory).execute_command(
            args["command"],
            timeout=args.get("timeout"),
        )
    if kind == "python":
        from .python_executor import PythonExecutorCore

        workspace_variables = request.get("workspace_variables", {})
        assignments = "\n".join(
            f"{name} = {value!r}" for name, value in workspace_variables.items()
        )
        result = PythonExecutorCore(working_directory).execute_code(
            f"{assignments}\n{args['code']}",
            args.get("capture_output", True),
        )
    elif kind == "javascript":
        from .javascript_executor import JavaScriptExecutorCore

        packages = args.get("packages")
        result = JavaScriptExecutorCore(working_directory).execute_code(
            args["code"],
            packages=[part.strip() for part in packages.split(",")]
            if packages
            else None,
        )
    else:
        raise ValueError("Unsupported isolated tool operation")

    # Registration and authoritative file IDs remain on the host. Guest paths
    # are untrusted and are checked again against the host's output directory.
    paths = (
        sorted(
            path
            for path, stat in _snapshot(working_directory).items()
            if before.get(path) != stat
        )
        if result.get("success")
        else []
    )
    result["generated_files"] = [path.name for path in paths]
    result["file_refs"] = [
        {
            "file_path": str(path),
            "filename": path.name,
            "mime_type": mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
        }
        for path in paths
    ]
    result["artifacts"] = []
    return result


def main() -> None:
    # Match the existing runner's support for pip --user installed executables.
    import site

    os.environ["PATH"] = (
        str(Path(site.getuserbase()) / "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--result-file", required=True)
    parsed = parser.parse_args()
    request = json.loads(Path(parsed.request_file).read_text(encoding="utf-8"))
    if request.get("kind") == "chrome":
        import sys

        from .chrome_daemon_runner import main as chrome_main

        sys.argv = [
            "chrome-runtime",
            *request["arguments"],
            "--result-file",
            parsed.result_file,
        ]
        chrome_main()
        return
    result = execute(request)
    Path(parsed.result_file).write_text(
        json.dumps(result, ensure_ascii=True), encoding="utf-8"
    )
