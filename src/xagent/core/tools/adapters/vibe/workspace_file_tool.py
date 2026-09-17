"""
Workspace-bound file tools for xagent

This module provides file tools that are bound to specific workspace instances.
Each tool instance operates within its designated workspace only.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List

from .....core.file_ref import WORKSPACE_OUTPUT_FILES_TOOL_NAME
from .....core.workspace import DEFAULT_USER_FILE_LIST_LIMIT, TaskWorkspace
from ...core.workspace_file_tool import FileInfo, WorkspaceFileOperations
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class FileTool(FunctionTool):
    """Base class for file tools with FILE category."""

    category = ToolCategory.FILE


class WorkspaceFileTools(WorkspaceFileOperations):
    """
    Workspace-bound file tools.

    Each instance is bound to a specific workspace and provides
    file operations restricted to that workspace.
    """

    def __init__(self, workspace: TaskWorkspace):
        """
        Initialize with workspace binding.

        Args:
            workspace: The workspace to bind to
        """
        self.inner = WorkspaceFileOperations(workspace)
        self.workspace = workspace

    def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read file content in workspace"""
        return self.inner.read_file(file_path, encoding, start_line, end_line)

    def write_file(
        self,
        file_path: str | None = None,
        content: str | None = None,
        encoding: str = "utf-8",
        create_dirs: bool = True,
        filename: str | None = None,
    ) -> Dict[str, Any]:
        """Write file content in workspace"""
        if file_path is None:
            file_path = filename
        if not file_path:
            raise ValueError("file_path is required")
        if content is None:
            raise ValueError("content is required")
        return self.inner.write_file(file_path, content, encoding, create_dirs)

    def prepare_html_asset(
        self,
        file_id: str,
        html_path: str,
        alias: str | None = None,
        assets_subdir: str = "assets",
    ) -> Dict[str, Any]:
        """Copy a file_id-referenced asset into the current output bundle."""
        return self.inner.prepare_html_asset(file_id, html_path, alias, assets_subdir)

    def append_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> bool:
        """Append content to file in workspace"""
        return self.inner.append_file(file_path, content, encoding, create_dirs)

    def delete_file(self, file_path: str) -> bool:
        """Delete file in workspace"""
        return self.inner.delete_file(file_path)

    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in workspace"""
        return self.inner.file_exists(file_path)

    def list_files(
        self,
        directory_path: str = ".",
        show_hidden: bool = False,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """List files in workspace directory (defaults to all directories)"""
        return self.inner.list_files(directory_path, show_hidden, recursive)

    def create_directory(self, directory_path: str, parents: bool = True) -> bool:
        """Create directory in workspace"""
        return self.inner.create_directory(directory_path, parents)

    def get_file_info(self, file_path_or_id: str) -> FileInfo:
        """Get detailed file information in workspace. Accepts either file paths or file_ids."""
        return self.inner.get_file_info(file_path_or_id)

    def read_json_file(self, file_path: str, encoding: str = "utf-8") -> Any:
        """Read JSON file in workspace"""
        return self.inner.read_json_file(file_path, encoding)

    def write_json_file(
        self,
        file_path: str,
        data: Dict[str, Any],
        encoding: str = "utf-8",
        indent: int = 2,
    ) -> Dict[str, Any]:
        """Write JSON file in workspace"""
        return self.inner.write_json_file(file_path, data, encoding, indent)

    def read_csv_file(
        self, file_path: str, encoding: str = "utf-8", delimiter: str = ","
    ) -> List[Dict[str, str]]:
        """Read CSV file in workspace"""
        return self.inner.read_csv_file(file_path, encoding, delimiter)

    def write_csv_file(
        self,
        file_path: str,
        data: List[Dict[str, str]],
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> Dict[str, Any]:
        """Write CSV file in workspace"""
        return self.inner.write_csv_file(file_path, data, encoding, delimiter)

    def get_workspace_output_files(self) -> Dict[str, Any]:
        """Get output file list from current workspace"""
        return self.inner.get_workspace_output_files()

    def list_all_user_files(  # type: ignore[override]
        self,
        include_workspace_files: bool = True,
        limit: int = DEFAULT_USER_FILE_LIST_LIMIT,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List the user's files within this task's reach.

        The core listing's ``openable_only`` switch is deliberately absent from
        this signature: FunctionTool turns every parameter into a field of the
        tool schema, so exposing it would let the model turn the narrowing off.

        Args:
            include_workspace_files: Whether to include current workspace files
            limit: Maximum number of registered uploads to return (default: 50)
            offset: Number of registered uploads to skip for pagination (default: 0)

        Returns:
            Dictionary of files with file_id, filename, size and mime_type, with
            absolute paths withheld; see :meth:`_without_foreign_paths`.
        """
        listing = self.inner.list_all_user_files(
            include_workspace_files, limit, offset, openable_only=True
        )
        return self._without_foreign_paths(listing)

    @staticmethod
    def _without_foreign_paths(listing: Dict[str, Any]) -> Dict[str, Any]:
        """Withhold absolute paths, keeping genuine workspace-relative ones.

        This is not a path boundary — ``list_files`` and ``get_file_info`` still
        return absolute paths, and the whitelisted upload root still resolves
        them. It removes the filenames a listing would otherwise hand the model
        to aim at. Uploads carry an absolute path under ``relative_path`` too,
        so only rows whose path really is relative to the workspace keep it.
        """
        sanitized = []
        for entry in listing["files"]:
            visible = {
                key: value
                for key, value in entry.items()
                if key not in ("storage_path", "relative_path")
            }
            relative_path = entry.get("relative_path")
            if (
                entry.get("in_current_workspace")
                and relative_path
                and not os.path.isabs(relative_path)
            ):
                visible["relative_path"] = relative_path
            sanitized.append(visible)
        return {**listing, "files": sanitized}

    def get_tools(self) -> List[FunctionTool]:
        """Get all tool instances"""
        return [
            FileTool(
                self.read_file,
                name="read_file",
                description="Read file content in workspace. Accepts either file paths (e.g., 'filename.txt') or file_ids (e.g., 'abc-123-def'). Automatically detects input type. For large files, results may be truncated in model context; use start_line/end_line to inspect a specific 1-based inclusive line range instead of repeating the same full-file read. SVG files are XML text: use read_file for exact markup, colors, viewBox, paths, fill, stroke, and gradient values.",
            ),
            FileTool(
                self.write_file,
                name="write_file",
                description="Write file content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths. Returns a FileRef with file_id, preview_url, download_url, and markdown_link.\n\nImportant: For HTML files, do not guess paths to uploaded files or files from other tasks. First call prepare_html_asset(file_id, html_path, alias) for every external image/CSS/JS asset, then use the returned html_src in the HTML.",
            ),
            FileTool(
                self.prepare_html_asset,
                name="prepare_html_asset",
                description="Prepare an uploaded or registered file for use inside an HTML artifact. Pass the source file_id, the target HTML output path such as 'index.html' or 'reports/index.html', and an optional alias such as 'logo.png'. The tool copies the asset next to that HTML file under assets_subdir and returns html_src relative to the HTML file. Use html_src in <img src>, <link href>, <script src>, or CSS url(). Do not compute ../ paths yourself.",
            ),
            FileTool(
                self.append_file,
                name="append_file",
                description="Append content to file in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.delete_file,
                name="delete_file",
                description="Delete file in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.list_files,
                name="list_files",
                description="List files in workspace directory (defaults to all directories including input, output, temp. Can also specify specific directory like list_files('input'))",
            ),
            FileTool(
                self.create_directory,
                name="create_directory",
                description="Create directory in workspace",
            ),
            FileTool(
                self.file_exists,
                name="file_exists",
                description="Check if file exists in workspace",
            ),
            FileTool(
                self.get_file_info,
                name="get_file_info",
                description="Get detailed file information in workspace. Accepts either file paths (e.g., 'filename.txt') or file_ids (e.g., 'abc-123-def'). Automatically detects input type.",
            ),
            FileTool(
                self.read_json_file,
                name="read_json_file",
                description="Read JSON file in workspace. Accepts either file paths (e.g., 'filename.txt') or file_ids (e.g., 'abc-123-def'). Automatically detects input type.",
            ),
            FileTool(
                self.write_json_file,
                name="write_json_file",
                description="Write JSON file in workspace. Use relative paths (e.g., 'data.json'), not absolute paths. Returns a FileRef with file_id, preview_url, download_url, and markdown_link.",
            ),
            FileTool(
                self.read_csv_file,
                name="read_csv_file",
                description="Read CSV file in workspace. Accepts either file paths (e.g., 'filename.txt') or file_ids (e.g., 'abc-123-def'). Automatically detects input type.",
            ),
            FileTool(
                self.write_csv_file,
                name="write_csv_file",
                description="Write CSV file in workspace. Use relative paths (e.g., 'data.csv'), not absolute paths. Returns a FileRef with file_id, preview_url, download_url, and markdown_link.",
            ),
            FileTool(
                self.get_workspace_output_files,
                name=WORKSPACE_OUTPUT_FILES_TOOL_NAME,
                description="Get output file list from current workspace",
            ),
            FileTool(
                self.list_all_user_files,
                name="list_all_user_files",
                description="List the user's files so you can find one they named that has no file_id in the current context, such as an attachment from an earlier turn; attachments are injected per turn. Scan the returned page yourself — there is no search parameter — and expect only what is within this task's reach: this task's own uploads and nothing else. Files belonging to the user's other tasks are left out, as are files left behind by a task that was deleted, and no filesystem path to any of them is returned here. Registered uploads come back newest-first in slices of limit (50 by default) starting at offset, and a slice can be short without being the last one, so raise the offset before concluding a named file does not exist — but stop once offset reaches total_count, which is the whole upload count and the point past which no upload remains to find. When include_workspace_files is on, current-workspace files are appended to every page outside that slicing, so they repeat as you page; the unregistered ones among them have no file_id, so read those by their relative path; when total_count is 0 they are all you get and paging further returns the same rows. Do not call this to discover the current turn's inputs, to hunt for reference material nobody gave you, or to take inventory before starting work. Returns file_id, filename, size, mime_type, and whether a file is in the current workspace; open a listed upload by passing its file_id to read_file.",
            ),
            FileTool(
                self.edit_file,
                name="edit_file",
                description="Precisely edit file content in workspace, supporting multiple edit operations based on line numbers and pattern matching. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.find_and_replace,
                name="find_and_replace",
                description="Convenience function to find and replace text content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
        ]


def create_workspace_file_tools(workspace: TaskWorkspace) -> List[FunctionTool]:
    """
    Create list of file tools bound to specified workspace

    Args:
        workspace: Workspace to bind to

    Returns:
        List of tool instances
    """
    tools_instance = WorkspaceFileTools(workspace)
    return tools_instance.get_tools()


# Register tool creator for auto-discovery
# Import at bottom to avoid circular import with factory
from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool(categories={"file"})
async def create_file_tools(config: "BaseToolConfig") -> List[Any]:
    """Create workspace-bound file tools."""
    if not config.get_file_tools_enabled():
        return []

    workspace = ToolFactory.create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        return create_workspace_file_tools(workspace)
    except Exception as e:
        logger.warning(f"Failed to create file tools: {e}")
        return []
