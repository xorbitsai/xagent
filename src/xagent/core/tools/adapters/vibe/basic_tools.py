"""Network and code capability registration; basic helpers register separately."""

import logging
from typing import TYPE_CHECKING, Any, List

from .....config import get_web_search_provider
from ....utils.encryption import ToolCredentialUnreadableError
from .base import ToolCategory
from .factory import ToolFactory, register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool(categories={ToolCategory.WEB_TOOLS.value})
async def create_web_tools(config: "BaseToolConfig") -> List[Any]:
    """Create search, public webpage fetching and HTTP API tools."""
    if not config.get_basic_tools_enabled():
        return []

    tools: List[Any] = []
    workspace = await ToolFactory.create_workspace_async(config)

    def credentials(tool_name: str, *fields: str) -> list[str | None]:
        try:
            return [config.get_tool_credential(tool_name, field) for field in fields]
        except ToolCredentialUnreadableError:
            # Disable this provider, not API/code/fetch tools or other providers.
            # The resolver still refuses environment fallback for this stored key.
            logger.warning("Skipping %s: stored credential is unreadable", tool_name)
            return [None for _ in fields]

    # Web search tool preference in auto mode: Zhipu -> Tavily -> Exa -> Google -> Brave -> none
    zhipu_api_key, zhipu_base_url = credentials(
        "zhipu_web_search", "api_key", "base_url"
    )
    (tavily_api_key,) = credentials("tavily_web_search", "api_key")
    google_api_key, google_cse_id = credentials("web_search", "api_key", "cse_id")
    (exa_api_key,) = credentials("exa_web_search", "api_key")
    (brave_api_key,) = credentials("brave_web_search", "api_key")
    provider = get_web_search_provider()

    if provider in ("auto", "zhipu") and zhipu_api_key:
        from .zhipu_web_search import ZhipuWebSearchTool

        tools.append(ZhipuWebSearchTool(api_key=zhipu_api_key, base_url=zhipu_base_url))
    elif provider in ("auto", "tavily") and tavily_api_key:
        from .tavily_web_search import TavilyWebSearchTool

        tools.append(TavilyWebSearchTool(api_key=tavily_api_key))
    elif provider in ("auto", "exa") and exa_api_key:
        from .exa_web_search import ExaWebSearchTool

        tools.append(ExaWebSearchTool(api_key=exa_api_key))
    elif provider in ("auto", "google") and google_api_key and google_cse_id:
        from .web_search import WebSearchTool

        tools.append(WebSearchTool(api_key=google_api_key, cse_id=google_cse_id))
    elif provider in ("auto", "brave") and brave_api_key:
        from .brave_web_search import BraveWebSearchTool

        tools.append(BraveWebSearchTool(api_key=brave_api_key))

    from .fetch_web_content import FetchWebContentTool

    tools.append(FetchWebContentTool())

    if workspace:
        from .download_web_asset import DownloadWebAssetTool

        tools.append(DownloadWebAssetTool(workspace=workspace))

    # API tool
    from .api_tool import APITool

    tools.append(APITool())

    return tools


@register_tool(categories={ToolCategory.CODE_EXECUTION.value})
async def create_code_tools(config: "BaseToolConfig") -> List[Any]:
    """Create Python, JavaScript and shell tools under the configured sandbox policy."""
    if not config.get_basic_tools_enabled():
        return []
    workspace = await ToolFactory.create_workspace_async(config)
    tools: List[Any] = []
    # Python executor tool (if workspace available)
    if workspace:
        from .python_executor import PythonExecutorToolForBasic

        tools.append(PythonExecutorToolForBasic(workspace=workspace))

    # JavaScript executor tool (if workspace available)
    if workspace:
        from .javascript_executor import JavaScriptExecutorToolForBasic

        tools.append(JavaScriptExecutorToolForBasic(workspace=workspace))

    # Command executor tool (if workspace available)
    if workspace:
        from .command_executor import CommandExecutorToolForBasic

        tools.append(CommandExecutorToolForBasic(workspace=workspace))

    return tools
