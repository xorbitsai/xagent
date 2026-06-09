"""
Image Web Search Tool for xagent
Framework wrapper around the pure image search tool
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel, Field

from .....config import get_uploads_dir
from ....workspace import TaskWorkspace
from ...core.image_web_search import ImageWebSearchCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


class ImageWebSearchArgs(BaseModel):
    query: str = Field(description="The image search query string")
    num_results: int = Field(
        default=5, description="Number of images to return (max 10)"
    )
    image_size: str = Field(
        default="medium",
        description="Image size: small, medium, large, xlarge, xxlarge, huge",
    )
    image_type: str = Field(
        default="photo",
        description="Image type: photo, clipart, lineart, animated, transparent",
    )
    save_to_workspace: bool = Field(
        default=True, description="Whether to save images to workspace temp directory"
    )
    workspace_id: Optional[str] = Field(
        default=None,
        description="Workspace ID for saving images (uses current workspace if not provided)",
    )


class ImageWebSearchResult(BaseModel):
    results: List[Dict[str, Any]] = Field(
        description="Search results with image information and local paths"
    )


class ImageWebSearchTool(AbstractBaseTool):
    """Framework wrapper for the pure image web search tool"""

    category: ToolCategory = ToolCategory.WEB_SEARCH

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "image_web_search"

    @property
    def description(self) -> str:
        return """Search the web for images using Google Search.
        Returns image results with URLs, metadata, and local file paths when saved to workspace.
        Images are automatically downloaded and saved to the workspace temp directory.
        Useful for finding reference images, stock photos, and visual content."""

    @property
    def tags(self) -> list[str]:
        return ["search", "web", "image", "visual", "download"]

    def args_type(self) -> Type[BaseModel]:
        return ImageWebSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return ImageWebSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("ImageWebSearchTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        search_args = ImageWebSearchArgs.model_validate(args)

        # Determine save directory
        save_directory = self._get_save_directory(
            search_args.save_to_workspace, search_args.workspace_id
        )

        # Perform search within auto_register context
        if self._workspace and save_directory and search_args.save_to_workspace:
            # Create core searcher instance
            searcher = ImageWebSearchCore(save_directory)

            with self._workspace.auto_register_files():
                # Perform search
                results = await searcher.search_images(
                    query=search_args.query,
                    num_results=search_args.num_results,
                    image_size=search_args.image_size,
                    image_type=search_args.image_type,
                    save_images=search_args.save_to_workspace,
                )
        else:
            # Create core searcher instance
            searcher = ImageWebSearchCore(save_directory)

            # Perform search
            results = await searcher.search_images(
                query=search_args.query,
                num_results=search_args.num_results,
                image_size=search_args.image_size,
                image_type=search_args.image_type,
                save_images=search_args.save_to_workspace,
            )

        return ImageWebSearchResult(results=results).model_dump()

    def _get_save_directory(
        self, save_to_workspace: bool, workspace_id: Optional[str]
    ) -> Optional[str]:
        """Determine the save directory based on workspace settings"""
        if not save_to_workspace:
            return None

        if workspace_id:
            # Use specified workspace
            return str(Path("workspaces") / workspace_id / "temp")
        elif self._workspace:
            # Use current workspace
            return str(self._workspace.temp_dir)
        else:
            # Fallback to default uploads/temp directory
            return str(get_uploads_dir() / "temp")


def get_image_web_search_tool(
    _info: Optional[dict[str, str]] = None, workspace: Optional[TaskWorkspace] = None
) -> AbstractBaseTool:
    """Factory function to create image web search tool instance"""
    return ImageWebSearchTool(workspace)


def create_image_web_search_tool(workspace: TaskWorkspace) -> AbstractBaseTool:
    """Create image web search tool bound to workspace"""
    return ImageWebSearchTool(workspace)
