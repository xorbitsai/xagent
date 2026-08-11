"""Browser automation tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


def _has_computer_runtime(contribution: Any) -> bool:
    """Return whether the full pre-policy contribution declares ``computer``."""
    from ....task_runtime import (
        TaskRuntimeContribution,
        full_task_runtime_contribution,
    )

    if not isinstance(contribution, TaskRuntimeContribution):
        return False
    full_contribution = full_task_runtime_contribution(contribution)
    return any(
        isinstance((name := getattr(tool, "name", None)), str)
        and name.strip() == "computer"
        for tool in full_contribution.tools
    )


@register_tool(categories={"browser"})
async def create_browser_tools(config: "BaseToolConfig") -> list[Any]:
    """Create browser automation tools."""
    if not config.get_browser_tools_enabled():
        return []

    # A task runtime may contribute its own ``computer`` instance. Suppress the
    # Playwright browser family as one unit so an out-of-tree runtime does not
    # collide with the core tool or silently fall back to an unrelated browser.
    # This is deliberately capability-based rather than provider-name-based:
    # the extension hook is public and core cannot enumerate its providers.
    contribution = config.get_task_runtime_contribution()
    if _has_computer_runtime(contribution):
        return []

    task_id = config.get_task_id()
    workspace = ToolFactory.create_workspace(config.get_workspace_config())

    try:
        from .browser_use import create_browser_tools

        return create_browser_tools(task_id=task_id, workspace=workspace)
    except Exception as e:
        logger.warning(f"Failed to create browser tools: {e}")
        return []
