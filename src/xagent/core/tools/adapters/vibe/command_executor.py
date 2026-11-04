"""
Command Line Execution Tool for xagent
Framework wrapper around the pure command executor tool
"""

import asyncio
import logging
import os
import stat
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ....workspace import SPILL_DIR_NAME, TaskWorkspace
from ...artifacts import (
    GENERATED_ARTIFACT_EXTENSIONS,
    GeneratedArtifactSnapshot,
    artifact_type_for_filename,
    build_generated_file_metadata,
    changed_generated_artifact_files,
)
from ...core.command_executor import CommandExecutorCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .function import FunctionTool
from .sandboxed_tool.sandbox_config import sandbox_config

logger = logging.getLogger(__name__)

# One shell command can unpack an archive or install packages into the cwd.
# Past this many changed artifacts the call is treated as bulk extraction, not
# deliverable output, and nothing is registered.
MAX_REGISTERED_FILES_PER_COMMAND = 20
# Dependency and cache trees never hold deliverables, so the snapshot does not
# walk them. Hidden directories are skipped as well.
_SKIPPED_SNAPSHOT_DIRS = frozenset(
    {"__pycache__", "node_modules", "site-packages", "venv"}
)


class CommandExecutorFunctionTool(FunctionTool):
    """Command executor tool with BASIC category."""

    category = ToolCategory.BASIC


class CommandExecutorArgs(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout: Optional[int] = Field(
        default=None, description="Execution timeout in seconds (default: 300)"
    )


class CommandExecutorResult(BaseModel):
    success: bool = Field(description="Whether the command executed successfully")
    output: str = Field(description="Standard output from the command")
    error: str = Field(default="", description="Standard error from the command")
    return_code: int = Field(description="Process exit code")
    generated_files: list[str] = Field(
        default_factory=list, description="Names of registered output files"
    )
    file_refs: list[dict[str, Any]] = Field(
        default_factory=list, description="FileRefs of registered output files"
    )
    artifacts: list[dict[str, Any]] = Field(
        default_factory=list, description="Inline artifacts, 1:1 with file_refs"
    )
    registration_note: str = Field(
        default="", description="Why changed files got no file_id, if any did not"
    )


class CommandExecutorTool(AbstractBaseTool):
    """Framework wrapper for the pure command executor tool"""

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "command_executor"

    @property
    def description(self) -> str:
        working_directory = self._get_working_directory()
        workspace_line = (
            f"Commands run with current working directory: {working_directory}."
            if working_directory
            else "Commands run in the current process working directory."
        )
        return f"""Execute shell commands and scripts.
Supports any shell command including system commands, scripts, pipes, and redirects.
{workspace_line}
Use concrete paths, URLs, or file identifiers already returned by previous tool results directly. If a tool returned an absolute path or a path relative to the command working directory, pass that path to the next command instead of rediscovering it.
Only search for files when no usable path was provided, and keep searches scoped to the command working directory or another explicitly relevant directory. Do not run broad recursive searches from `/` or the user's home directory unless the user explicitly asks for that scope.
Documents, spreadsheets, PDFs, images and videos the command writes under the working directory are registered and returned in file_refs with a file_id; registration_note explains files that failed to register. Hidden files and directories, node_modules, venv, site-packages, __pycache__ and symlinks are not registered.
Examples: ls -la output, grep -r 'pattern' ./output, ./deploy.sh, cat file.txt | grep error"""

    @property
    def tags(self) -> list[str]:
        return ["shell", "command", "bash", "script", "terminal"]

    def args_type(self) -> Type[BaseModel]:
        return CommandExecutorArgs

    def return_type(self) -> Type[BaseModel]:
        return CommandExecutorResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        exec_args = CommandExecutorArgs.model_validate(args)

        # Determine working directory
        working_directory = self._get_working_directory()

        # Create core executor instance
        executor = CommandExecutorCore(working_directory)

        if not (self._workspace and working_directory):
            result = executor.execute_command(
                exec_args.command, timeout=exec_args.timeout
            )
            return _dump_result(result)

        # Registered regardless of exit code: a shell's status describes only
        # its last command, and a partial file is reported with its validation
        # status rather than hidden.
        files_before = _snapshot_artifacts(working_directory, self._workspace)
        result = executor.execute_command(exec_args.command, timeout=exec_args.timeout)
        changed = changed_generated_artifact_files(
            files_before, _snapshot_artifacts(working_directory, self._workspace)
        )
        result.update(
            _register_changed_files(self._workspace, changed, working_directory)
        )
        return _dump_result(result)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self.run_json_sync, args)

    def _get_working_directory(self) -> Optional[str]:
        """Determine the working directory based on workspace settings"""
        if self._workspace:
            # Use workspace output directory as working directory
            return str(self._workspace.resolve_path(""))
        return None


_REGISTRATION_FIELDS = (
    "generated_files",
    "file_refs",
    "artifacts",
    "registration_note",
)


def _dump_result(result: dict[str, Any]) -> dict[str, Any]:
    """Dump the result, omitting registration fields when nothing was registered.

    An ``artifacts`` list routes the observation through the artifact
    formatter, so a plain ``ls`` must keep the stdout-only shape it had.
    """
    dumped = CommandExecutorResult(**result).model_dump()
    if not any(dumped[field] for field in _REGISTRATION_FIELDS):
        for field in _REGISTRATION_FIELDS:
            del dumped[field]
    return dumped


def _display_path(file_path: Path, root: str) -> str:
    """``file_path`` relative to the command working directory, never absolute."""
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.name


def _register_changed_files(
    workspace: TaskWorkspace, changed: set[Path], root: str
) -> dict[str, Any]:
    if not changed:
        return {}
    if len(changed) > MAX_REGISTERED_FILES_PER_COMMAND:
        # Images are what bulk extraction floods the directory with; listing the
        # rest shows which deliverables the model has to touch.
        others = sorted(
            _display_path(p, root)
            for p in changed
            if artifact_type_for_filename(p.name) != "image"
        )
        shown = ", ".join(others[:MAX_REGISTERED_FILES_PER_COMMAND])
        if len(others) > MAX_REGISTERED_FILES_PER_COMMAND:
            shown += f" and {len(others) - MAX_REGISTERED_FILES_PER_COMMAND} more"
        listing = f" Changed files that are not images: {shown}." if others else ""
        return {
            "registration_note": (
                f"{len(changed)} files changed, more than the "
                f"{MAX_REGISTERED_FILES_PER_COMMAND} registered per command, "
                "so none got a file_id."
                f"{listing} To register a deliverable, run `touch <path>` on at "
                f"most {MAX_REGISTERED_FILES_PER_COMMAND} paths per command, in a "
                "separate command from the one that wrote them; do not rewrite "
                "the files."
            )
        }
    # register_file is called explicitly: build_workspace_file_ref reuses an
    # existing id without re-staging, which would leave an overwritten
    # file's previous bytes as the served version.
    registered: list[Path] = []
    failed: list[str] = []
    for file_path in sorted(changed):
        try:
            workspace.register_file(str(file_path))
        except Exception:
            logger.warning(
                "Failed to register command output %s", file_path, exc_info=True
            )
            failed.append(_display_path(file_path, root))
            continue
        registered.append(file_path)
    result: dict[str, Any] = build_generated_file_metadata(
        workspace=workspace, file_paths=registered
    )
    described = {Path(ref["file_path"]).resolve() for ref in result["file_refs"]}
    no_ref = [
        _display_path(p, root) for p in registered if p.resolve() not in described
    ]
    # Without a note the file reads as never written, which invites a rewrite.
    notes: list[str] = []
    if failed:
        notes.append(
            f"Written but not registered: {', '.join(failed)}. "
            "No file_id is available for these in this call."
        )
    if no_ref:
        notes.append(
            f"Registered, but no file reference could be built: {', '.join(no_ref)}. "
            "get_workspace_output_files lists their file_id."
        )
    if notes:
        result["registration_note"] = " ".join(notes)
    return result


def _snapshot_artifacts(
    root: str, workspace: TaskWorkspace
) -> GeneratedArtifactSnapshot:
    """Snapshot regular artifact files under ``root``.

    Dependency trees and the engine-owned spill directory are pruned: neither
    holds files the command produced as deliverables (#2545).
    """
    snapshot: GeneratedArtifactSnapshot = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".")
            and name not in _SKIPPED_SNAPSHOT_DIRS
            and not (
                name.casefold() == SPILL_DIR_NAME.casefold()
                and workspace.is_engine_owned_path(Path(dirpath) / name)
            )
        ]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if (
                filename.startswith(".")
                or file_path.suffix.lower() not in GENERATED_ARTIFACT_EXTENSIONS
            ):
                continue
            try:
                file_stat = file_path.lstat()
            except OSError:
                # Gone or unreadable: a snapshot must never cost the command result.
                continue
            # A symlink is not output the command wrote; it can point anywhere.
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            snapshot[file_path] = (file_stat.st_mtime_ns, file_stat.st_size)
    return snapshot


@sandbox_config()
class CommandExecutorToolForBasic(CommandExecutorTool):
    """Command executor tool with BASIC category."""

    category = ToolCategory.BASIC

    @property
    def name(self) -> str:
        return "execute_command"


def get_command_executor_tool(info: Optional[dict[str, Any]] = None) -> FunctionTool:
    """
    Create a workspace-bound command executor tool.

    Args:
        info: Dictionary containing workspace information

    Returns:
        A command executor tool bound to the specified workspace
    """
    # Extract workspace from info if provided
    workspace = None
    if info and "workspace" in info:
        workspace = info["workspace"]

    # Create workspace-bound command executor
    executor = CommandExecutorTool(workspace=workspace)

    # Wrap as LangChain tool
    def execute_command(command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute shell command."""
        result: Dict[str, Any] = executor.run_json_sync(
            {"command": command, "timeout": timeout}
        )
        return result

    return CommandExecutorFunctionTool(execute_command)


def create_command_executor_tool(
    workspace: TaskWorkspace,
) -> AbstractBaseTool:
    """Create command executor tool bound to workspace"""
    return CommandExecutorTool(workspace)
