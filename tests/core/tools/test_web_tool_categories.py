"""Tests for web research tool category metadata."""

from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.exa_web_search import ExaWebSearchTool
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.adapters.vibe.tavily_web_search import TavilyWebSearchTool
from xagent.core.tools.adapters.vibe.web_search import WebSearchTool
from xagent.core.tools.adapters.vibe.zhipu_web_search import ZhipuWebSearchTool


def test_web_research_tools_share_web_search_category() -> None:
    tools = [
        WebSearchTool(),
        TavilyWebSearchTool(),
        ExaWebSearchTool(),
        ZhipuWebSearchTool(),
        FetchWebContentTool(),
    ]

    assert {tool.metadata.category for tool in tools} == {ToolCategory.WEB_SEARCH}
