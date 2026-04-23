"""Knowledge base tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool
async def create_knowledge_tools(config: "BaseToolConfig") -> List[Any]:
    """Create knowledge base search tools."""
    tools: List[Any] = []

    try:
        from .document_search import (
            get_knowledge_search_tool,
            get_list_knowledge_bases_tool,
        )

        embedding_model = config.get_embedding_model()
        allowed_collections = config.get_allowed_collections()
        user_id = config.get_user_id()
        is_admin = config.is_admin()

        # If collections are already selected, agents should search them directly
        # instead of spending a tool call rediscovering the same KB list.
        if not allowed_collections:
            list_tool = get_list_knowledge_bases_tool(
                allowed_collections=allowed_collections,
                user_id=user_id,
                is_admin=is_admin,
            )
            tools.append(list_tool)

        # Add search tool
        knowledge_tool = get_knowledge_search_tool(
            embedding_model_id=embedding_model,
            allowed_collections=allowed_collections,
            user_id=user_id,
            is_admin=is_admin,
        )
        tools.append(knowledge_tool)
    except Exception as e:
        logger.warning(f"Failed to create knowledge tools: {e}")

    return tools
