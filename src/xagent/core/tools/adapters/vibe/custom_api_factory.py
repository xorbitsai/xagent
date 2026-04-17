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

        # Get database session to query CustomApi
        from .....web.models.custom_api import UserCustomApi
        from .....web.models.database import get_session_local

        SessionLocal = get_session_local()
        with SessionLocal() as session:
            # Query active custom APIs for the user
            user_apis = (
                session.query(UserCustomApi)
                .filter(
                    UserCustomApi.user_id == int(user_id),
                    UserCustomApi.is_active,
                )
                .all()
            )

            if not user_apis:
                return []

            custom_api_configs = []
            for user_api in user_apis:
                api = user_api.custom_api
                if api:
                    custom_api_configs.append(
                        {
                            "name": api.name,
                            "description": api.description or "",
                            "env": api.env or {},
                        }
                    )

            if not custom_api_configs:
                return []

            return create_custom_api_tools(custom_api_configs)

    except Exception as e:
        logger.error(
            f"Failed to create Custom API tools from database: {e}", exc_info=True
        )
        return []
