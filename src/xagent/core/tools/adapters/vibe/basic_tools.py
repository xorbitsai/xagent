"""Basic tools registration using @register_tool decorator."""

import logging
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
    zhipu_api_key = config.get_tool_credential("zhipu_web_search", "api_key")
    zhipu_base_url = config.get_tool_credential("zhipu_web_search", "base_url")
    tavily_api_key = config.get_tool_credential("tavily_web_search", "api_key")
    google_api_key = config.get_tool_credential("web_search", "api_key")
    google_cse_id = config.get_tool_credential("web_search", "cse_id")

    if zhipu_api_key:
        from .zhipu_web_search import ZhipuWebSearchTool

        tools.append(ZhipuWebSearchTool(api_key=zhipu_api_key, base_url=zhipu_base_url))
    elif tavily_api_key:
        from .tavily_web_search import TavilyWebSearchTool

        tools.append(TavilyWebSearchTool(api_key=tavily_api_key))
    elif google_api_key and google_cse_id:
        from .web_search import WebSearchTool

        tools.append(WebSearchTool(api_key=google_api_key, cse_id=google_cse_id))

    # Python executor tool (if workspace available)
    if workspace:
        from .python_executor import PythonExecutorToolForBasic

        tools.append(PythonExecutorToolForBasic(workspace=workspace))

    # JavaScript executor tool (if workspace available)
    if workspace:
        from .javascript_executor import JavaScriptExecutorToolForBasic

        tools.append(JavaScriptExecutorToolForBasic(workspace=workspace))

    # API tool
    from .api_tool import APITool

    tools.append(APITool())

    # Command executor tool (if workspace available)
    if workspace:
        from .command_executor import CommandExecutorToolForBasic

        tools.append(CommandExecutorToolForBasic(workspace=workspace))

    return tools
