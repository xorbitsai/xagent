"""PPTX Tool Adapter for xagent

Adapts pptx_tool.py to be used as an xagent tool via vibe adapter system.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...core.pptx_tool import (
    clean_pptx,
    pack_pptx,
    read_pptx,
    unpack_pptx,
)
from .function import FunctionTool

if TYPE_CHECKING:
    from ....workspace import TaskWorkspace
else:
    TaskWorkspace = Any  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PPTXGenerationTool:
    """
    PPTX tool wrapper that handles workspace integration.

    Similar to ImageGenerationTool, this wraps the pptx functions
    and ensures workspace is properly integrated.
    """

    def __init__(self, workspace: "TaskWorkspace | None" = None):
        """
        Initialize PPTX tool with workspace support.

        Args:
            workspace: Workspace for file management (optional but recommended)
        """
        self._workspace = workspace
        logger.debug(f"PPTX tool initialized with workspace: {workspace is not None}")

    async def read_pptx(self, pptx_path: str, extract_text: bool = False) -> Dict:
        """Read PPTX file and extract information."""
        return read_pptx(pptx_path, extract_text, self._workspace)

    async def unpack_pptx(self, pptx_path: str, output_dir: str) -> Dict:
        """Unpack PPTX file to directory."""
        return unpack_pptx(pptx_path, output_dir, self._workspace)

    async def pack_pptx(
        self, input_dir: str, output_path: str, validate: bool = True
    ) -> Dict:
        """Pack directory into PPTX file."""
        return pack_pptx(input_dir, output_path, validate, self._workspace)

    async def clean_pptx(self, unpacked_dir: str) -> Dict:
        """Clean orphaned files from unpacked PPTX directory."""
        return clean_pptx(unpacked_dir, self._workspace)

    def get_tools(self) -> list:
        """Get all tool instances."""
        return [
            FunctionTool(
                self.read_pptx,
                name="read_pptx",
                description="""Read PPTX file and extract information.

Use this tool to:
- Get slide count and slide information from a .pptx file
- Extract all text content from a presentation
- Read slide titles and structure
- Analyze existing presentations as reference

Args:
    pptx_path: Path to .pptx file to read
    extract_text: If True, extracts all text content; if False, returns slide structure

Returns:
    Dictionary with slide information or extracted text
""",
                tags=["pptx", "presentation", "file"],
            ),
            FunctionTool(
                self.unpack_pptx,
                name="unpack_pptx",
                description="""Unpack PPTX file to directory for advanced editing.

Extracts PPTX ZIP archive and pretty-prints XML files for manual editing.
Use when you need to:
- Manually edit PPTX XML structure
- Inspect PPTX internal files and structure
- Modify slide layouts directly
- Learn from existing presentation templates
- Before using pack_pptx to repackage

Args:
    pptx_path: Path to .pptx file to unpack
    output_dir: Directory path to extract files to

Returns:
    Dictionary with success status, output directory, and file count
""",
                tags=["pptx", "presentation", "unpack", "editing"],
            ),
            FunctionTool(
                self.pack_pptx,
                name="pack_pptx",
                description="""Pack directory into PPTX file.

Packages an unpacked PPTX directory back into a valid .pptx file.
Use after unpacking and editing with unpack_pptx.

Args:
    input_dir: Directory containing unpacked PPTX files
    output_path: Output .pptx file path
    validate: Whether to validate structure (default: True)

Returns:
    Dictionary with success status and output file path
""",
                tags=["pptx", "presentation", "pack", "editing"],
            ),
            FunctionTool(
                self.clean_pptx,
                name="clean_pptx",
                description="""Clean orphaned files from unpacked PPTX directory.

Removes slides that are not in presentation.xml slide list,
unreferenced media files, and orphaned relationship files.

Use this when:
- PPTX has unused/orphaned slides
- Media directory has unreferenced images
- After manually deleting slides
- Before packaging with pack_pptx

Args:
    unpacked_dir: Path to unpacked PPTX directory

Returns:
    Dictionary with success status and count of removed files
""",
                tags=["pptx", "presentation", "clean", "editing"],
            ),
        ]


def create_pptx_tool(
    workspace: Optional[TaskWorkspace] = None,
) -> list:
    """
    Create PPTX tools with workspace support.

    Args:
        workspace: Workspace for file management

    Returns:
        List of tool instances
    """
    tool_instance = PPTXGenerationTool(workspace)
    return tool_instance.get_tools()
