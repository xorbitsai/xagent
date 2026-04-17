"""Custom API Factory.

Responsible for discovering Custom API tools configured in the database
and providing them to the agent system.
"""

import logging
from typing import Sequence

from .api_tool_adapter import create_custom_api_tools
from .base import Tool
from .config import BaseToolConfig
from .factory import register_tool

logger = logging.getLogger(__name__)


@register_tool
async def create_db_custom_api_tools(config: BaseToolConfig) -> Sequence[Tool]:
    """Create Custom API tools from database configurations.

    Args:
        config: The tool configuration containing user/workspace context.

    Returns:
        List of Tool instances for each configured Custom API.
    """
    try:
        user_id = config.get_user_id()
        if not user_id:
            logger.debug("No user_id found in config, skipping database custom APIs")
            return []

        # Use the config to get Custom API configurations instead of querying the DB directly
        custom_api_configs = config.get_custom_api_configs()
        if not custom_api_configs:
            return []

        return create_custom_api_tools(custom_api_configs)

    except Exception as e:
        logger.error(
            f"Failed to create Custom API tools from config: {e}", exc_info=True
        )
        return []
