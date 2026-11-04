"""MCP prompts.
This module provides MCP prompts.
"""

from typing import Any

from mcp import ClientSession
from mcp.types import GetPromptResult


async def load_mcp_prompt(
    session: ClientSession,
    name: str,
    *,
    arguments: dict[str, Any] | None = None,
) -> GetPromptResult:
    """Load MCP prompt.

    Args:
        session: The MCP client session.
        name: Name of the prompt to load.
        arguments: Optional arguments to pass to the prompt.

    Returns:
        MCP prompt.
    """
    response = await session.get_prompt(name, arguments)
    return response
