"""
Workspace-bound file tools for xagent

This module provides file tools that are bound to specific workspace instances.
Each tool instance operates within its designated workspace only.
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .....core.workspace import TaskWorkspace
from .....skills.utils import create_skill_manager
from ...core.workspace_file_tool import FileInfo, WorkspaceFileOperations
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


def _get_all_skill_roots() -> List[Path]:
    """Get all skill directories (reusing skills/utils.py logic)."""
    skill_manager = create_skill_manager()
    return skill_manager.skills_roots


def _validate_skill_name(skill_name: str) -> None:
    """Validate skill name to prevent path traversal attacks.

    Only allows alphanumeric characters, underscores, hyphens.

    Args:
        skill_name: Name of the skill to validate

    Raises:
        ValueError: If skill_name contains invalid characters
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", skill_name):
        raise ValueError(
            f"Invalid skill name: '{skill_name}'. "
            "Skill names must contain only letters, numbers, underscores, and hyphens."
        )


def _validate_file_path(file_path: str) -> None:
    """Validate file path to prevent path traversal attacks.

    Args:
        file_path: File path to validate

    Raises:
        ValueError: If file_path contains path traversal attempts
    """
    # Check for path traversal patterns
    if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
        raise ValueError(
            f"Invalid file path: '{file_path}'. "
            "Relative paths within the skill directory are allowed (no '..' or absolute paths)."
        )


class FileTool(FunctionTool):
    """Base class for file tools with FILE category."""

    category = ToolCategory.FILE


class SkillTool(FunctionTool):
    """Base class for skill tools with SKILL category."""

    category = ToolCategory.SKILL


class WorkspaceFileTools(WorkspaceFileOperations):
    """
    Workspace-bound file tools.

    Each instance is bound to a specific workspace and provides
    file operations restricted to that workspace.
    """

    def __init__(
        self,
        workspace: TaskWorkspace,
        skills_roots: Optional[List[str]] = None,
    ):
        """
        Initialize with workspace binding.

        Args:
            workspace: The workspace to bind to
            skills_roots: Optional list of skills directory paths. If None, uses default:
                        - Built-in skills directory
                        - User skills directory (.xagent/skills)
                        - External directories from XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS
        """
        self.inner = WorkspaceFileOperations(workspace)
        self.workspace = workspace

        if skills_roots is None:
            self.skills_roots = _get_all_skill_roots()
        else:
            self.skills_roots = [Path(p) for p in skills_roots]

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

    def get_file_info(self, file_path: str) -> FileInfo:
        """Get detailed file information in workspace"""
        return self.inner.get_file_info(file_path)

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

    def read_skill_file(self, skill_name: str, file_path: str) -> str:
        """
        Read a file from a skill directory.

        Skills are searched in configured skill root directories in order,
        returning the first match.

        Args:
            skill_name: Name of the skill
            file_path: Relative path within the skill

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If the skill or file doesn't exist
            ValueError: If skill_name or file_path contains invalid characters
        """
        _validate_skill_name(skill_name)
        _validate_file_path(file_path)

        # Search for skill in all roots (first match wins)
        skill_dir = None
        for root in self.skills_roots:
            candidate = root / skill_name
            if candidate.exists() and candidate.is_dir():
                skill_dir = candidate
                break

        if skill_dir is None:
            raise FileNotFoundError(f"Skill not found: '{skill_name}'")

        full_path = skill_dir / file_path

        if not full_path.exists():
            raise FileNotFoundError(
                f"File not found: '{file_path}' in skill '{skill_name}'"
            )

        return full_path.read_text(encoding="utf-8")

    def list_skill_files(
        self,
        skill_name: str,
        directory_path: str = ".",
        show_hidden: bool = False,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        List files in a skill directory.

        Skills are searched in configured skill root directories in order,
        returning the first match.

        Args:
            skill_name: Name of the skill
            directory_path: Optional subdirectory path (default: '.' for all files)
            show_hidden: Whether to show hidden files (default: False)
            recursive: Whether to list recursively (default: True)

        Returns:
            Dict with files list, total_count, current_path, and directory name

        Raises:
            FileNotFoundError: If the skill directory doesn't exist
            ValueError: If skill_name or directory_path contains invalid characters
        """
        _validate_skill_name(skill_name)
        if directory_path != ".":
            _validate_file_path(directory_path)

        # Search for skill in all roots (first match wins)
        skill_dir = None
        for root in self.skills_roots:
            candidate = root / skill_name
            if candidate.exists() and candidate.is_dir():
                skill_dir = candidate
                break

        if skill_dir is None:
            raise FileNotFoundError(f"Skill not found: '{skill_name}'")

        # Determine search path
        if directory_path == ".":
            search_path = skill_dir
        else:
            search_path = skill_dir / directory_path
            if not search_path.exists():
                raise FileNotFoundError(
                    f"Directory not found: '{directory_path}' in skill '{skill_name}'"
                )

        files = []

        def scan_directory(current_path: Path) -> None:
            try:
                for item in current_path.iterdir():
                    if not show_hidden and item.name.startswith("."):
                        continue

                    stat = item.stat()
                    # Use relative path as the "path" field for compatibility
                    rel_path = item.relative_to(skill_dir)
                    file_info = FileInfo(
                        name=item.name,
                        path=str(rel_path),
                        size=stat.st_size,
                        is_file=item.is_file(),
                        is_dir=item.is_dir(),
                        modified_time=stat.st_mtime,
                    )
                    files.append(file_info)

                    if recursive and item.is_dir():
                        scan_directory(item)

            except PermissionError:
                pass

        scan_directory(search_path)

        # Return relative path as current_path to avoid exposing system paths
        relative_search_path = (
            search_path.relative_to(skill_dir) if search_path != skill_dir else "."
        )

        return {
            "files": [file.model_dump() for file in files],
            "total_count": len(files),
            "current_path": str(relative_search_path),
            "directory": skill_name,
        }

    def get_tools(self) -> List[FunctionTool]:
        """Get all tool instances"""
        return [
            FileTool(
                self.read_file,
                name="read_file",
                description="Read file content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
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
                description="Get detailed file information in workspace",
            ),
            FileTool(
                self.read_json_file,
                name="read_json_file",
                description="Read JSON file in workspace",
            ),
            FileTool(
                self.write_json_file,
                name="write_json_file",
                description="Write JSON file in workspace",
            ),
            FileTool(
                self.read_csv_file,
                name="read_csv_file",
                description="Read CSV file in workspace",
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
                self.edit_file,
                name="edit_file",
                description="Precisely edit file content in workspace, supporting multiple edit operations based on line numbers and pattern matching. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            FileTool(
                self.find_and_replace,
                name="find_and_replace",
                description="Convenience function to find and replace text content in workspace. Use relative paths (e.g., 'filename.txt'), not absolute paths.",
            ),
            SkillTool(
                self.read_skill_file,
                name="read_skill_file",
                description="Read a text format file from a skill directory by skill name and file path.",
            ),
            SkillTool(
                self.list_skill_files,
                name="list_skill_files",
                description="List files in a skill directory. Returns file names, sizes, and types.",
            ),
        ]


def create_workspace_file_tools(
    workspace: TaskWorkspace, skills_roots: Optional[List[str]] = None
) -> List[FunctionTool]:
    """
    Create list of file tools bound to specified workspace

    Args:
        workspace: Workspace to bind to
        skills_roots: Optional list of skills directory paths. If None, uses default:
                    - Built-in skills directory
                    - User skills directory (.xagent/skills)
                    - External directories from XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS

    Returns:
        List of tool instances
    """
    tools_instance = WorkspaceFileTools(workspace, skills_roots=skills_roots)
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
