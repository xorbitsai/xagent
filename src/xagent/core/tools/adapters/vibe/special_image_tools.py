"""Special image tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool(categories={"web_search"})
async def create_image_web_search_tools(config: "BaseToolConfig") -> List[Any]:
    """Create web-search image tools."""
    tools = []
    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    if not workspace:
        return []

    try:
        from .image_web_search import create_image_web_search_tool

        image_search_tool = create_image_web_search_tool(workspace)
        tools.append(image_search_tool)
    except Exception as e:
        logger.warning(f"Failed to create image web search tool: {e}")

    return tools


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
