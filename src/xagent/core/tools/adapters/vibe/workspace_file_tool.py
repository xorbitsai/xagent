"""
Workspace-bound file tools for xagent

This module provides file tools that are bound to specific workspace instances.
Each tool instance operates within its designated workspace only.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .....core.workspace import TaskWorkspace
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

    def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """Read file content in workspace"""
        return self.inner.read_file(file_path, encoding)

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
    ) -> bool:
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
    ) -> bool:
        """Write CSV file in workspace"""
        return self.inner.write_csv_file(file_path, data, encoding, delimiter)

    def get_workspace_output_files(self) -> Dict[str, Any]:
        """Get output file list from current workspace"""
        return self.inner.get_workspace_output_files()

    def list_all_user_files(
        self,
        include_workspace_files: bool = True,
        limit: int = 1000,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all user files across all workspaces and uploaded files.

        Args:
            include_workspace_files: Whether to include current workspace files
            limit: Maximum number of files to return (default: 1000)
            offset: Number of files to skip for pagination (default: 0)

        Returns:
            Dictionary with list of all user files with metadata including file_id,
            filename, storage_path, size, mime_type, etc.
        """
        return self.workspace.list_all_user_files(
            include_workspace_files, limit, offset
        )

    def get_tools(self) -> List[FunctionTool]:
        """Get all tool instances"""
        return [
            FileTool(
                self.read_file,
                name="read_file",
                description="Read file content in workspace. Accepts either file paths (e.g., 'filename.txt') or file_ids (e.g., 'abc-123-def'). Automatically detects input type.",
            ),
            FileTool(
                self.write_file,
                name="write_file",
                description="Write file content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.\n\nImportant: For HTML files, when referencing resources in the same directory (CSS, JS, images), only use filenames (e.g., '1.png'), not absolute paths (e.g., '/uploads/xxx/1.png'). All files are in the workspace, and browsers will automatically resolve relative paths.",
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
                description="Write JSON file in workspace",
            ),
            FileTool(
                self.read_csv_file,
                name="read_csv_file",
                description="Read CSV file in workspace. Accepts either file paths (e.g., 'filename.txt') or file_ids (e.g., 'abc-123-def'). Automatically detects input type.",
            ),
            FileTool(
                self.write_csv_file,
                name="write_csv_file",
                description="Write CSV file in workspace",
            ),
            FileTool(
                self.get_workspace_output_files,
                name="get_workspace_output_files",
                description="Get output file list from current workspace",
            ),
            FileTool(
                self.list_all_user_files,
                name="list_all_user_files",
                description="List all user files across all workspaces and uploaded files. Supports pagination and filtering. Returns file_id, filename, storage_path, size, mime_type, and whether files are in current workspace.",
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


@register_tool
async def create_file_tools(config: "BaseToolConfig") -> List[Any]:
    """Create workspace-bound file tools."""
    if not config.get_file_tools_enabled():
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        return create_workspace_file_tools(workspace)
    except Exception as e:
        logger.warning(f"Failed to create file tools: {e}")
        return []
