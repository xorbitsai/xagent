"""Basic tools registration using @register_tool decorator."""

import logging
import os
from typing import TYPE_CHECKING, Any, List

from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool
async def create_basic_tools(config: "BaseToolConfig") -> List[Any]:
    """Create basic tools (web search, code executors)."""
    if not config.get_basic_tools_enabled():
        return []

    tools: List[Any] = []
    workspace = ToolFactory._create_workspace(config.get_workspace_config())

    # Web search tool preference: Zhipu -> Tavily -> Google -> none
    zhipu_api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("BIGMODEL_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")

    if zhipu_api_key:
        from .zhipu_web_search import ZhipuWebSearchTool

        tools.append(ZhipuWebSearchTool())
    elif tavily_api_key:
        from .tavily_web_search import TavilyWebSearchTool

        tools.append(TavilyWebSearchTool())
    elif google_api_key and google_cse_id:
        from .web_search import WebSearchTool

        tools.append(WebSearchTool())

    # Python executor tool (if workspace available)
    if workspace:
        from .python_executor import get_python_executor_tool

        tools.append(get_python_executor_tool({"workspace": workspace}))

    # JavaScript executor tool (if workspace available)
    if workspace:
        from .javascript_executor import get_javascript_executor_tool

        tools.append(get_javascript_executor_tool({"workspace": workspace}))

    # API tool
    from .api_tool import APITool

    tools.append(APITool())

    # Command executor tool (if workspace available)
    if workspace:
        from .command_executor import get_command_executor_tool

        tools.append(get_command_executor_tool({"workspace": workspace}))

    return tools
