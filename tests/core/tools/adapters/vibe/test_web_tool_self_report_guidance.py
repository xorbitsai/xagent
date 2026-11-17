"""Tool descriptions tell the model which names to use when it reports its work."""

import pytest

from xagent.core.tools.adapters.vibe.brave_web_search import BraveWebSearchTool
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.adapters.vibe.web_search import WebSearchTool


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("tool", [BraveWebSearchTool(), WebSearchTool()])
def test_search_tool_names_both_tools_for_self_report(tool):
    text = _flat(tool.description)
    assert tool.name == "web_search"
    assert "name web_search for searching" in text
    assert "fetch_web_content for any pages that were read" in text


def test_fetch_tool_names_itself_for_self_report():
    text = _flat(FetchWebContentTool().description)
    assert "name fetch_web_content for pages that were read" in text
    assert "web_search for searching" in text
