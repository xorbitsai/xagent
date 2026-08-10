"""Browser automation tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


def _has_local_browser_runtime(contribution: Any) -> bool:
    from ....computer.native_browser import (
        LOCAL_BROWSER_TASK_EXTENSION,
    )
    from ....task_runtime import (
        TaskRuntimeContribution,
        full_task_runtime_contribution,
    )

    if not isinstance(contribution, TaskRuntimeContribution):
        return False
    full_contribution = full_task_runtime_contribution(contribution)
    return any(
        provider_name == LOCAL_BROWSER_TASK_EXTENSION
        and any(
            getattr(tool, "name", None) == "computer"
            for tool in provider_contribution.tools
        )
        for provider_name, provider_contribution in (
            full_contribution.provider_contributions
        )
    )


@register_tool(categories={"browser"})
async def create_browser_tools(config: "BaseToolConfig") -> List[Any]:
    """Create browser automation tools."""
    if not config.get_browser_tools_enabled():
        return []

    # A bound Local browser task contributes its own ``computer`` instance.
    # Suppress the Playwright browser family as one unit so the runtime tool
    # does not collide with the core tool or accidentally expose a second,
    # unrelated browser to the model.
    contribution = config.get_task_runtime_contribution()
    if _has_local_browser_runtime(contribution):
        return []

    task_id = config.get_task_id()
    workspace = ToolFactory.create_workspace(config.get_workspace_config())

    try:
        from .browser_use import create_browser_tools

        return create_browser_tools(task_id=task_id, workspace=workspace)
    except Exception as e:
        logger.warning(f"Failed to create browser tools: {e}")
        return []
