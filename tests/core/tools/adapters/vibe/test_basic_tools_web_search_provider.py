import pytest

from xagent.core.tools.adapters.vibe.basic_tools import create_basic_tools
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec


def _tool_names(tools):
    return [tool.name for tool in tools if hasattr(tool, "name")]


@pytest.mark.asyncio
async def test_auto_web_search_provider_preserves_existing_priority(monkeypatch):
    monkeypatch.delenv("XAGENT_WEB_SEARCH_PROVIDER", raising=False)

    tools = await create_basic_tools(
        ToolConfig(
            {
                "workspace": None,
                "tool_credentials": {
                    "zhipu_web_search": {"api_key": "zhipu-key"},
                    "web_search": {
                        "api_key": "google-key",
                        "cse_id": "google-cse-id",
                    },
                },
            }
        )
    )

    assert _tool_names(tools) == [
        "zhipu_web_search",
        "fetch_web_content",
        "api_call",
    ]


@pytest.mark.asyncio
async def test_explicit_google_web_search_provider_uses_google(monkeypatch):
    monkeypatch.setenv("XAGENT_WEB_SEARCH_PROVIDER", "google")

    tools = await create_basic_tools(
        ToolConfig(
            {
                "workspace": None,
                "tool_credentials": {
                    "zhipu_web_search": {"api_key": "zhipu-key"},
                    "web_search": {
                        "api_key": "google-key",
                        "cse_id": "google-cse-id",
                    },
                },
            }
        )
    )

    assert _tool_names(tools) == ["web_search", "fetch_web_content", "api_call"]
    assert tools[0].__class__.__name__ == "WebSearchTool"


@pytest.mark.asyncio
async def test_explicit_exa_web_search_provider_uses_exa(monkeypatch):
    monkeypatch.setenv("XAGENT_WEB_SEARCH_PROVIDER", "exa")

    tools = await create_basic_tools(
        ToolConfig(
            {
                "workspace": None,
                "tool_credentials": {
                    "exa_web_search": {"api_key": "exa-key"},
                    "web_search": {
                        "api_key": "google-key",
                        "cse_id": "google-cse-id",
                    },
                },
            }
        )
    )

    assert _tool_names(tools) == ["exa_web_search", "fetch_web_content", "api_call"]


@pytest.mark.asyncio
async def test_explicit_provider_without_credentials_adds_no_search_tool(monkeypatch):
    monkeypatch.setenv("XAGENT_WEB_SEARCH_PROVIDER", "google")

    tools = await create_basic_tools(
        ToolConfig(
            {
                "workspace": None,
                "tool_credentials": {
                    "zhipu_web_search": {"api_key": "zhipu-key"},
                },
            }
        )
    )

    assert _tool_names(tools) == ["fetch_web_content", "api_call"]


@pytest.mark.asyncio
async def test_basic_category_selection_excludes_web_fetch_tool(monkeypatch):
    monkeypatch.setenv("XAGENT_WEB_SEARCH_PROVIDER", "google")
    config = ToolConfig(
        {
            "workspace": None,
            "tool_credentials": {},
        }
    )
    config._tool_selection_spec = ToolSelectionSpec.from_raw(tool_categories=["basic"])

    tools = await ToolFactory.create_all_tools(config)

    names = _tool_names(tools)
    assert "api_call" in names
    assert "fetch_web_content" not in names


@pytest.mark.asyncio
async def test_web_search_category_selection_keeps_web_fetch_tool(monkeypatch):
    monkeypatch.setenv("XAGENT_WEB_SEARCH_PROVIDER", "google")
    config = ToolConfig(
        {
            "workspace": None,
            "tool_credentials": {},
        }
    )
    config._tool_selection_spec = ToolSelectionSpec.from_raw(
        tool_categories=["web_search"]
    )

    tools = await ToolFactory.create_all_tools(config)

    assert _tool_names(tools) == ["fetch_web_content"]
