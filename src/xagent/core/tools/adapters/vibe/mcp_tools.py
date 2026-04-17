"""MCP tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool
async def create_mcp_tools(config: "BaseToolConfig") -> List[Any]:
    """Create MCP tools from configuration."""
    mcp_configs = await config.get_mcp_server_configs()
    if not mcp_configs:
        return []

    try:
        from .factory import ToolFactory

        return await ToolFactory._create_mcp_tools_from_configs(mcp_configs)
    except Exception as e:
        logger.warning(f"Failed to create MCP tools: {e}")
        return []
