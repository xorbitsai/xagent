"""Capability selections must preserve old grants without widening new ones."""

import pytest

from xagent.core.tools.adapters.vibe.base import INTRINSIC_TOOL_NAMES, ToolCategory
from xagent.core.tools.adapters.vibe.categories import (
    LEGACY_CATEGORY_TOOL_NAMES,
    category_catalog,
    configurable_tool_category,
)
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection,expected",
    [
        (["basic_tools"], {"ask_user_question", "translate_json"}),
        (
            ["code_execution"],
            {"execute_command", "execute_python_code", "execute_javascript_code"},
        ),
        (
            ["web_tools"],
            {"web_search", "fetch_web_content", "download_web_asset", "api_call"},
        ),
        (["basic"], set(LEGACY_CATEGORY_TOOL_NAMES["basic"])),
        (["web_search"], {"web_search", "fetch_web_content", "download_web_asset"}),
        ([], set()),
    ],
)
async def test_actual_factory_capability_membership(
    monkeypatch, tmp_path, selection, expected
):
    monkeypatch.setenv("SANDBOX_ENABLED", "false")
    monkeypatch.setenv("XAGENT_WEB_SEARCH_PROVIDER", "google")
    config = ToolConfig(
        {
            "workspace": {"base_dir": str(tmp_path), "task_id": "capabilities"},
            "tool_credentials": {
                "web_search": {"api_key": "test-key", "cse_id": "test-cse"}
            },
        }
    )
    config._tool_selection_spec = ToolSelectionSpec.from_raw(tool_categories=selection)
    tools = await ToolFactory.create_all_tools(config)
    names = {tool.name for tool in tools}
    intrinsic = set(INTRINSIC_TOOL_NAMES) if selection else set()
    assert names == expected | intrinsic
    assert "calculator" not in names


def test_canonical_categories_are_explicit_and_legacy_selectors_are_not_advertised():
    catalog = {row["id"]: row["label"] for row in category_catalog()}
    assert catalog[ToolCategory.BASIC.value] == "Basic Tools"
    assert catalog[ToolCategory.CODE_EXECUTION.value] == "Code Execution"
    assert catalog[ToolCategory.WEB_SEARCH.value] == "Web Tools"
    assert "basic" not in catalog
    assert "web_search" not in catalog


@pytest.mark.parametrize(
    "provider",
    ["web_search", "tavily_web_search", "zhipu_web_search", "exa_web_search"],
)
def test_search_configuration_and_runtime_share_the_web_category(provider):
    assert configurable_tool_category(provider) == ToolCategory.WEB_SEARCH.value


@pytest.mark.parametrize(
    "raw,canonical",
    [("basic", ToolCategory.BASIC), ("web_search", ToolCategory.WEB_TOOLS)],
)
def test_saved_tool_metadata_decodes_legacy_category_values(raw, canonical):
    from xagent.core.tools.adapters.vibe.base import ToolMetadata

    assert ToolMetadata(name="saved", category=raw).category is canonical


@pytest.mark.parametrize(
    "selection,needs_credentials",
    [
        (["basic_tools"], False),
        (["code_execution"], False),
        (["web_tools"], True),
        (["basic"], True),
        (["web_search"], True),
        ([], False),
    ],
)
def test_detached_factory_prefetch_follows_network_selection(
    selection, needs_credentials
):
    from xagent.web.tools.config import WebToolConfig

    config = WebToolConfig(
        db=None,
        request=None,
        user_id=7,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=selection),
        include_mcp_tools=False,
    )
    assert config._build_factory_runtime_load_plan().load_basic is needs_credentials


@pytest.mark.parametrize(
    "old_tag,selection",
    [
        ("basic", "basic_tools"),
        ("basic", "code_execution"),
        ("web_search", "web_tools"),
    ],
)
def test_extension_registration_tags_keep_working_with_new_selections(
    old_tag, selection
):
    from xagent.core.tools.adapters.vibe.factory import ToolRegistry

    spec = ToolSelectionSpec.from_raw(tool_categories=[selection])
    assert ToolRegistry._should_run_creator(frozenset({old_tag}), spec, None)
