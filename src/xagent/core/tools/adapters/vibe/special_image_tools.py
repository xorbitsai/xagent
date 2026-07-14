"""Special image tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool(categories={"image"})
async def create_special_image_tools(config: "BaseToolConfig") -> List[Any]:
    """Create special image tools."""
    tools = []
    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        from .logo_overlay import create_logo_overlay_tool

        logo_overlay_tool = create_logo_overlay_tool(workspace)
        tools.append(logo_overlay_tool)
    except Exception as e:
        logger.warning(f"Failed to create logo overlay tool: {e}")

    return tools
