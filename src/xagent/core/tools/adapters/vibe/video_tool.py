"""
Video generation tool for xagent.

This module provides video generation capabilities using pre-configured video
models passed from the web layer.
"""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ....model.video.base import BaseVideoModel
from ....workspace import TaskWorkspace
from ...core.video_tool import VideoGenerationToolCore
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class VideoGenerationFunctionTool(FunctionTool):
    """VideoGenerationFunctionTool with ToolCategory.VIDEO category."""

    category = ToolCategory.VIDEO

    def _normalize_args(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.name != "generate_video":
            return args
        return VideoGenerationToolCore.normalize_raw_tool_args(args)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await super().run_json_async(self._normalize_args(args))

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return super().run_json_sync(self._normalize_args(args))


class VideoGenerationTool(VideoGenerationToolCore):
    """Video generation tool that uses pre-configured video models."""

    def __init__(
        self,
        video_models: Dict[str, BaseVideoModel],
        model_descriptions: Optional[Dict[str, str]] = None,
        workspace: Optional[TaskWorkspace] = None,
        default_video_model: Optional[BaseVideoModel] = None,
    ):
        super().__init__(
            video_models,
            model_descriptions,
            workspace,
            default_video_model,
        )

        if workspace is None:
            raise ValueError("Workspace is required for video generation tools")

    def get_tools(self) -> list:
        generate_description = self.GENERATE_VIDEO_DESCRIPTION.format(
            self._model_info_text
        )

        return [
            VideoGenerationFunctionTool(
                self.generate_video,
                name="generate_video",
                description=generate_description,
            ),
            VideoGenerationFunctionTool(
                self.list_available_models,
                name="list_video_models",
                description="List all available video generation models, including model ID, availability status, and detailed description information.",
            ),
        ]


def create_video_tool(
    video_models: Dict[str, BaseVideoModel],
    model_descriptions: Optional[Dict[str, str]] = None,
    workspace: Optional[TaskWorkspace] = None,
    default_video_model: Optional[BaseVideoModel] = None,
) -> list:
    if workspace is None:
        raise ValueError("Workspace is required for video generation tools")

    tool_instance = VideoGenerationTool(
        video_models,
        model_descriptions,
        workspace,
        default_video_model,
    )
    return tool_instance.get_tools()


from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool(categories={"video"})
async def create_video_tools_from_config(config: "BaseToolConfig") -> List[Any]:
    """Create video generation tools from configuration."""
    spec = (
        config.get_tool_selection_spec()
        if hasattr(config, "get_tool_selection_spec")
        else None
    )
    if spec is not None and not spec.includes_category("video"):
        return []

    video_models = config.get_video_models()
    if not video_models:
        return []

    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        return create_video_tool(
            video_models,
            workspace=workspace,
            default_video_model=config.get_video_model(),
        )
    except Exception as e:
        logger.warning("Failed to create video tools: %s", e)
        return []
