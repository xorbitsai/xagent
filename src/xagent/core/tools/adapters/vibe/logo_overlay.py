"""
Logo Overlay Tool for xagent
Provides logo overlay functionality on base images using PIL
Supports both local paths and remote URLs
"""

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from .....config import get_uploads_dir
from ....file_ref import build_workspace_file_ref
from ....workspace import TaskWorkspace
from ...core.logo_overlay import LogoOverlayCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


class LogoOverlayArgs(BaseModel):
    base_image_uri: str = Field(description="Base image URI (local path or remote URL)")
    logo_image_uri: str = Field(description="Logo image URI (local path or remote URL)")
    position: str = Field(
        default="bottom-right",
        description="Logo position: top-left, top-right, bottom-left, bottom-right, center",
    )
    size_ratio: float = Field(
        default=0.2,
        description="Logo size relative to base image (0.1 to 0.5)",
        ge=0.1,
        le=0.5,
    )
    opacity: float = Field(
        default=1.0, description="Logo opacity (0.0 to 1.0)", ge=0.0, le=1.0
    )
    padding: int = Field(
        default=20, description="Padding from edges in pixels", ge=0, le=100
    )
    output_filename: Optional[str] = Field(
        default=None, description="Custom output filename (without extension)"
    )
    workspace_id: Optional[str] = Field(
        default=None,
        description="Workspace ID for saving output (uses current workspace if not provided)",
    )


class LogoOverlayResult(BaseModel):
    success: bool = Field(description="Whether the operation was successful")
    output_path: str = Field(description="Path to the output image")
    message: str = Field(description="Operation message")
    error: Optional[str] = Field(description="Error message if any")
    file_id: Optional[str] = Field(default=None, description="Registered file ID")
    file_ref: Optional[dict[str, Any]] = Field(
        default=None, description="Registered FileRef for the output image"
    )


class LogoOverlayTool(AbstractBaseTool):
    """Tool for overlaying logos on base images"""

    # Logo overlay is an image processing/editing tool
    category: ToolCategory = ToolCategory.IMAGE

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "logo_overlay"

    @property
    def description(self) -> str:
        return """Overlay a logo on a base image with customizable position, size, and opacity.
        Supports both local file paths and remote URLs for both base image and logo.
        The logo will be automatically resized and positioned according to your specifications.
        Output is saved to the workspace uploads directory."""

    @property
    def tags(self) -> list[str]:
        return ["image", "logo", "overlay", "design", "branding"]

    def args_type(self) -> Type[BaseModel]:
        return LogoOverlayArgs

    def return_type(self) -> Type[BaseModel]:
        return LogoOverlayResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("LogoOverlayTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        overlay_args = LogoOverlayArgs.model_validate(args)

        # Determine output directory
        if overlay_args.workspace_id:
            output_dir = Path("workspaces") / overlay_args.workspace_id / "output"
        elif self._workspace:
            output_dir = self._workspace.output_dir
        else:
            output_dir = get_uploads_dir() / "output"

        # Resolve image paths if workspace is available
        base_image_uri = self._resolve_image_path(overlay_args.base_image_uri)
        logo_image_uri = self._resolve_image_path(overlay_args.logo_image_uri)

        # Create core instance
        core = LogoOverlayCore(output_directory=str(output_dir))

        # Execute overlay within auto_register context
        if self._workspace:
            with self._workspace.auto_register_files():
                result = await core.overlay_logo(
                    base_image_uri=base_image_uri,
                    logo_image_uri=logo_image_uri,
                    position=overlay_args.position,
                    size_ratio=overlay_args.size_ratio,
                    opacity=overlay_args.opacity,
                    padding=overlay_args.padding,
                    output_filename=overlay_args.output_filename,
                )
        else:
            result = await core.overlay_logo(
                base_image_uri=base_image_uri,
                logo_image_uri=logo_image_uri,
                position=overlay_args.position,
                size_ratio=overlay_args.size_ratio,
                opacity=overlay_args.opacity,
                padding=overlay_args.padding,
                output_filename=overlay_args.output_filename,
            )

        if self._workspace and result.get("success") and result.get("output_path"):
            try:
                file_ref = build_workspace_file_ref(
                    workspace=self._workspace,
                    file_path=result["output_path"],
                )
                result["file_id"] = file_ref["file_id"]
                result["file_ref"] = file_ref
            except Exception as e:
                logger.warning("Failed to build logo overlay FileRef: %s", e)
                result["file_ref_warning"] = str(e)

        return LogoOverlayResult(**result).model_dump()

    def _resolve_image_path(self, image_input: str) -> str:
        """
        Resolve image input to appropriate format for image processing.
        Args:
            image_input: Either a URL string or a local file path
        Returns:
            str: Resolved image path/URL suitable for processing
        """
        # Check if it's a URL (http/https)
        if image_input.startswith(("http://", "https://")):
            return image_input

        # Treat as local file path
        if self._workspace:
            try:
                # Use workspace's resolve_path_with_search method
                resolved_path = self._workspace.resolve_path_with_search(image_input)
                logger.info(
                    f"Resolved image path using workspace search: {image_input} -> {resolved_path}"
                )
                return str(resolved_path)
            except ValueError as e:
                logger.warning(f"Cannot resolve image path in workspace: {e}")
            except Exception as e:
                logger.warning(f"Error using workspace path resolution: {e}")

        # Fallback: return as-is
        return image_input


def get_logo_overlay_tool(
    _info: Optional[dict[str, str]] = None, workspace: Optional[TaskWorkspace] = None
) -> AbstractBaseTool:
    """Factory function to create logo overlay tool instance"""
    return LogoOverlayTool(workspace)


def create_logo_overlay_tool(workspace: TaskWorkspace) -> AbstractBaseTool:
    """Create logo overlay tool bound to workspace"""
    return LogoOverlayTool(workspace)
