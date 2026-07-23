"""Tests for web research tool category metadata."""

from pathlib import Path

from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.download_web_asset import DownloadWebAssetTool
from xagent.core.tools.adapters.vibe.exa_web_search import ExaWebSearchTool
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.adapters.vibe.tavily_web_search import TavilyWebSearchTool
from xagent.core.tools.adapters.vibe.web_search import WebSearchTool
from xagent.core.tools.adapters.vibe.zhipu_web_search import ZhipuWebSearchTool
from xagent.core.workspace import TaskWorkspace


def test_web_research_tools_share_web_search_category(tmp_path: Path) -> None:
    workspace = TaskWorkspace("web-asset-category", base_dir=str(tmp_path))
    tools = [
        WebSearchTool(),
        TavilyWebSearchTool(),
        ExaWebSearchTool(),
        ZhipuWebSearchTool(),
        FetchWebContentTool(),
        DownloadWebAssetTool(workspace),
    ]

    assert {tool.metadata.category for tool in tools} == {ToolCategory.WEB_SEARCH}
