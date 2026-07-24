"""Browser automation tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool(categories={"browser"})
async def create_browser_tools(config: "BaseToolConfig") -> List[Any]:
    """Create browser automation tools."""
    if not config.get_browser_tools_enabled():
        return []

    task_id = config.get_task_id()
    user_id = config.get_user_id()
    workspace = ToolFactory._create_workspace(config.get_workspace_config())

    try:
        from .browser_use import create_browser_tools

        return create_browser_tools(
            task_id=task_id,
            workspace=workspace,
            user_id=user_id,
            computer_runtime_kind=config.get_browser_runtime_kind(),
            browser_profile_id=config.get_browser_profile_id(),
            browser_profile_root=config.get_browser_profile_root(),
        )
    except Exception as e:
        logger.warning(f"Failed to create browser tools: {e}")
        return []
