"""Skill documentation access tools.

Provides tools for agents to access skill documentation (SKILL.md, examples,
reference materials) from skill directories. Uses "doc" terminology to avoid
confusion with MCP "resources" while remaining flexible for future storage
backends.
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .....core.workspace import TaskWorkspace
from .....skills.utils import create_skill_manager
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


def _get_all_skill_roots() -> List[Path]:
    """Get all skill directories."""
    skill_manager = create_skill_manager()
    return skill_manager.skills_roots


def _validate_skill_name(skill_name: str) -> None:
    """Validate skill name to prevent path traversal attacks."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", skill_name):
        raise ValueError(
            f"Invalid skill name: '{skill_name}'. "
            "Skill names must contain only letters, numbers, underscores, and hyphens."
        )


def _validate_doc_path(doc_path: str) -> None:
    """Validate doc path to prevent path traversal attacks."""
    if ".." in doc_path or doc_path.startswith("/") or doc_path.startswith("\\"):
        raise ValueError(
            f"Invalid doc path: '{doc_path}'. "
            "Relative paths within the skill are allowed."
        )


class SkillTool(FunctionTool):
    """Base class for skill tools with SKILL category."""

    category = ToolCategory.SKILL


class SkillTools:
    """Manager for skill documentation access.

    Uses "doc" terminology (short for documentation) which:
    - Avoids confusion with MCP "resources"
    - Is generic enough for future storage backends
    - Clearly indicates these are documentation/reference materials
    """

    def __init__(
        self,
        workspace: TaskWorkspace,
        skills_roots: Optional[List[str]] = None,
    ):
        """Initialize with workspace binding.

        Args:
            workspace: The workspace to bind to
            skills_roots: Optional list of skills directory paths. If None, uses default.
        """
        self.workspace = workspace

        if skills_roots is None:
            self.skills_roots = _get_all_skill_roots()
        else:
            self.skills_roots = [Path(p) for p in skills_roots]

    def _find_skill_dir(self, skill_name: str) -> Optional[Path]:
        """Find skill directory across all roots."""
        for root in self.skills_roots:
            candidate = root / skill_name
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def read_skill_doc(
        self,
        skill_name: str,
        doc_path: str,
        encoding: str = "utf-8",
    ) -> str:
        """Read documentation from a skill.

        Args:
            skill_name: Name of the skill
            doc_path: Location identifier for the documentation within the skill
            encoding: Text encoding (default: utf-8)

        Returns:
            Documentation content as string

        Raises:
            FileNotFoundError: If the skill or doc doesn't exist
            ValueError: If skill_name or doc_path contains invalid characters
        """
        _validate_skill_name(skill_name)
        _validate_doc_path(doc_path)

        skill_dir = self._find_skill_dir(skill_name)
        if skill_dir is None:
            raise FileNotFoundError(f"Skill not found: '{skill_name}'")

        full_path = skill_dir / doc_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Documentation not found: '{doc_path}' in skill '{skill_name}'"
            )

        return full_path.read_text(encoding=encoding)

    def list_skill_docs(
        self,
        skill_name: str,
        directory_path: str = ".",
        show_hidden: bool = False,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """List documentation within a skill.

        Args:
            skill_name: Name of the skill
            directory_path: Optional sub-location to scope the listing (default: '.' for all)
            show_hidden: Whether to include hidden items (default: False)
            recursive: Whether to list nested items (default: True)

        Returns:
            Simplified dict with documents list and count:
            {
                "documents": [
                    {"name": "SKILL.md", "size": 1234},
                    {"name": "examples/example.py", "size": 5678}
                ],
                "count": 2
            }

        Raises:
            FileNotFoundError: If the skill directory doesn't exist
            ValueError: If skill_name or directory_path contains invalid characters
        """
        _validate_skill_name(skill_name)
        if directory_path != ".":
            _validate_doc_path(directory_path)

        skill_dir = self._find_skill_dir(skill_name)
        if skill_dir is None:
            raise FileNotFoundError(f"Skill not found: '{skill_name}'")

        search_path = skill_dir / directory_path if directory_path != "." else skill_dir

        if not search_path.exists():
            raise FileNotFoundError(
                f"Directory not found: '{directory_path}' in skill '{skill_name}'"
            )

        documents = []

        def scan_directory(current_path: Path) -> None:
            try:
                for item in current_path.iterdir():
                    if not show_hidden and item.name.startswith("."):
                        continue

                    # Only include files, not directories
                    if item.is_file():
                        stat = item.stat()
                        rel_path = item.relative_to(skill_dir)
                        # Normalize path separators to forward slashes for consistency
                        documents.append(
                            {
                                "name": str(rel_path).replace("\\", "/"),
                                "size": stat.st_size,
                            }
                        )

                    if recursive and item.is_dir():
                        scan_directory(item)

            except PermissionError:
                pass

        scan_directory(search_path)

        return {"documents": documents, "count": len(documents)}

    def get_tools(self) -> List[FunctionTool]:
        """Get all tool instances."""
        return [
            SkillTool(
                self.read_skill_doc,
                name="read_skill_doc",
                description="Read documentation from a skill by providing the skill name and the document location.",
            ),
            SkillTool(
                self.list_skill_docs,
                name="list_skill_docs",
                description="List available documentation within a skill. Returns document names and sizes.",
            ),
        ]


def create_skill_tools(
    workspace: TaskWorkspace, skills_roots: Optional[List[str]] = None
) -> List[FunctionTool]:
    """Create skill documentation access tools bound to workspace."""
    tools_instance = SkillTools(workspace, skills_roots=skills_roots)
    return tools_instance.get_tools()


# Register tool creator for auto-discovery
from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool
async def create_skill_tools_from_config(config: "BaseToolConfig") -> List[Any]:
    """Create skill documentation access tools from configuration."""
    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        return create_skill_tools(workspace)
    except Exception as e:
        logger.warning(f"Failed to create skill tools: {e}")
        return []
