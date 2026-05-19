"""
Integration utilities for process-isolated tools.

Provides helper functions to automatically wrap tools with process isolation
when ProcessService is available.
"""

import logging
from typing import Any

from ..base import AbstractBaseTool
from .process_isolated_tool_wrapper import (
    create_process_isolated_tool,
)

logger = logging.getLogger(__name__)


def supports_process_isolation(tool: AbstractBaseTool) -> bool:
    """Return whether a tool explicitly supports process-isolated execution."""
    return bool(getattr(tool, "supports_process_isolation", False))


def should_use_process_isolation(
    tool_name: str,
    tool: AbstractBaseTool | None = None,
) -> bool:
    """Check if a tool should use process isolation.

    Args:
        tool_name: Name of the tool

    Returns:
        True if process isolation should be used
    """
    from .....execution.service.manager import get_process_service

    try:
        from xagent.web.sandbox_manager import (
            get_sandbox_manager,  # type: ignore[import]
        )
    except ImportError:
        # If web module is not available, define a stub
        def get_sandbox_manager() -> Any:  # type: ignore[no-redef,misc]
            return None

    # Check if ProcessService is available
    process_service = get_process_service()
    if not process_service:
        return False

    if tool is not None and not supports_process_isolation(tool):
        return False

    # Check if sandbox is enabled (sandbox has higher priority)
    sandbox_mgr = get_sandbox_manager()
    if sandbox_mgr:
        # Sandbox is enabled, don't use process isolation
        return False

    # Process isolation is available and sandbox is not enabled
    return True


def maybe_wrap_tool(
    tool: AbstractBaseTool,
    timeout: int = 300,
) -> AbstractBaseTool:
    """Wrap tool with process isolation if appropriate.

    Args:
        tool: Tool to potentially wrap
        timeout: Execution timeout in seconds

    Returns:
        Original tool or process-isolated wrapper
    """
    # Check if we should use process isolation
    if not should_use_process_isolation(tool.name, tool):
        return tool

    try:
        # Wrap tool with process isolation
        wrapped_tool = create_process_isolated_tool(
            tool=tool,
            timeout=timeout,
        )
        logger.info(f"Tool '{tool.name}' wrapped with process isolation")
        return wrapped_tool

    except Exception as e:
        logger.error(
            f"Failed to wrap tool '{tool.name}' with process isolation: {e}",
            exc_info=True,
        )
        # Fall back to original tool
        return tool


def wrap_tools(
    tools: list[AbstractBaseTool],
    timeout: int = 300,
) -> list[AbstractBaseTool]:
    """Wrap multiple tools with process isolation if appropriate.

    Args:
        tools: List of tools to potentially wrap
        timeout: Execution timeout in seconds

    Returns:
        List of tools (some may be wrapped)
    """
    wrapped_tools = []
    for tool in tools:
        wrapped_tool = maybe_wrap_tool(tool, timeout=timeout)
        wrapped_tools.append(wrapped_tool)

    return wrapped_tools
