"""
Image generation tool for xagent

This module provides image generation capabilities using pre-configured image models
passed from the web layer.
"""

import logging
from typing import Dict, Optional

from ....model.image.base import BaseImageModel
from ....workspace import TaskWorkspace
from ...core.image_tool import ImageGenerationToolCore
from .function import FunctionTool

logger = logging.getLogger(__name__)


class ImageGenerationTool(ImageGenerationToolCore):
    """
    Image generation tool that uses pre-configured image models.
    """

    def __init__(
        self,
        image_models: Dict[str, BaseImageModel],
        model_descriptions: Optional[Dict[str, str]] = None,
        workspace: Optional[TaskWorkspace] = None,
        default_generate_model: Optional[BaseImageModel] = None,
        default_edit_model: Optional[BaseImageModel] = None,
    ):
        """
        Initialize with pre-configured image models.

        Args:
            image_models: Dictionary mapping model_id to BaseImageModel instances
            model_descriptions: Dictionary mapping model_id to description strings
            workspace: Workspace for saving generated images (required)
            default_generate_model: Default model for image generation
            default_edit_model: Default model for image editing
        """
        # Call parent class initialization first
        super().__init__(
            image_models,
            model_descriptions,
            workspace,
            default_generate_model,
            default_edit_model,
        )

        # Vibe-specific initialization: workspace is required
        if workspace is None:
            raise ValueError("Workspace is required for image generation tools")

    async def _download_image(
        self, image_url: str, filename: Optional[str] = None, timeout: int = 3600
    ) -> str:
        # Vibe adapter uses 3600 second timeout
        return await super()._download_image(image_url, filename, timeout=timeout)

    def get_tools(self) -> list:
        """Get all tool instances."""
        # Format descriptions with model information
        generate_description = self.GENERATE_IMAGE_DESCRIPTION.format(
            self._model_info_text
        )
        edit_description = self.EDIT_IMAGE_DESCRIPTION.format(
            self._edit_model_info_text
        )

        tools = [
            FunctionTool(
                self.generate_image,
                name="generate_image",
                description=generate_description,
            ),
            FunctionTool(
                self.edit_image,
                name="edit_image",
                description=edit_description,
            ),
            FunctionTool(
                self.list_available_models,
                name="list_image_models",
                description="List all available image generation models, including model ID, availability status, and detailed description information (Note: model information is already provided in the generate_image tool description)",
            ),
        ]

        return tools


def create_image_tool(
    image_models: Dict[str, BaseImageModel],
    model_descriptions: Optional[Dict[str, str]] = None,
    workspace: Optional[TaskWorkspace] = None,
    default_generate_model: Optional[BaseImageModel] = None,
    default_edit_model: Optional[BaseImageModel] = None,
) -> list:
    """
    Create image generation tools with pre-configured models.

    Args:
        image_models: Dictionary mapping model_id to BaseImageModel instances
        model_descriptions: Dictionary mapping model_id to description strings
        workspace: Workspace for saving generated images (required)
        default_generate_model: Default model for image generation
        default_edit_model: Default model for image editing

    Returns:
        List of tool instances
    """
    if workspace is None:
        raise ValueError("Workspace is required for image generation tools")

    tool_instance = ImageGenerationTool(
        image_models,
        model_descriptions,
        workspace,
        default_generate_model,
        default_edit_model,
    )
    return tool_instance.get_tools()
