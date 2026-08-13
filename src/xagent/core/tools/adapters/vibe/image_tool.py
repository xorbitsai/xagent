"""
Image generation tool for xagent

This module provides image generation capabilities using pre-configured image models
passed from the web layer.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ....model.image.base import BaseImageModel
from ....workspace import TaskWorkspace
from ...core.image_tool import ImageGenerationToolCore
from .base import ToolCategory
from .function import FunctionTool

logger = logging.getLogger(__name__)


class ImageGenerationFunctionTool(FunctionTool):
    """ImageGenerationFunctionTool with ToolCategory.IMAGE category."""

    category = ToolCategory.IMAGE


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
        # A note inside the description is not enough: models blind-retry a
        # registered-but-doomed tool, so an unusable one has to leave the schema.
        # list_image_models stays either way — it is read-only, cannot fail, and
        # is the only way to answer "why is there nothing here".
        can_generate = self._get_model() is not None
        can_edit = self._get_edit_model() is not None

        tools = []

        if can_generate:
            generate_description = self.GENERATE_IMAGE_DESCRIPTION.format(
                self._model_info_text
            )
            if not can_edit:
                generate_description = (
                    "IMAGE EDITING IS UNAVAILABLE here: no configured image model "
                    "has the edit ability, so edit_image is not offered and passing "
                    "images fails. Render each deliverable from a text prompt in "
                    "one call.\n\n" + generate_description
                )
            tools.append(
                ImageGenerationFunctionTool(
                    self.generate_image,
                    name="generate_image",
                    description=generate_description,
                )
            )

        if can_edit:
            tools.append(
                ImageGenerationFunctionTool(
                    self.edit_image,
                    name="edit_image",
                    description=self.EDIT_IMAGE_DESCRIPTION.format(
                        self._edit_model_info_text
                    ),
                )
            )

        tools.append(
            ImageGenerationFunctionTool(
                self.list_available_models,
                name="list_image_models",
                description="List all available image generation models, including model ID, availability status, and detailed description information",
            )
        )

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


# Register tool creator for auto-discovery
# Import at bottom to avoid circular import with factory
from .factory import ToolFactory, register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool(categories={"image"})
async def create_image_tools_from_config(config: "BaseToolConfig") -> List[Any]:
    """Create image generation tools from configuration.

    Internal short-circuit on ``ToolSelectionSpec.includes_category("image")``
    skips the image-model DB lookup (``config.get_image_models()``)
    when the spec excludes the image category. Registry-level skip
    via ``categories={"image"}`` handles the most common path where
    ``spec.categories`` is set and doesn't include image; the internal
    check covers the legacy spec=None backward-compat path.
    """
    spec = (
        config.get_tool_selection_spec()
        if hasattr(config, "get_tool_selection_spec")
        else None
    )
    if spec is not None and not spec.includes_category("image"):
        return []
    image_models = config.get_image_models()
    if not image_models:
        return []

    workspace = ToolFactory.create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        default_generate_model = config.get_image_generate_model()
        default_edit_model = config.get_image_edit_model()

        return create_image_tool(
            image_models,
            workspace=workspace,
            default_generate_model=default_generate_model,
            default_edit_model=default_edit_model,
        )
    except Exception as e:
        logger.warning(f"Failed to create image tools: {e}")
        return []
