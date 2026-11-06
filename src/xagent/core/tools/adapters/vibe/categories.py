"""Capability categories shared by discovery, execution and the web catalog.

Old category selectors remain valid: ``basic`` retains code/API access, while
``web_search`` remains search-only. New selectors do not reuse those grants.
"""

from .base import ToolCategory

LEGACY_CATEGORY_TOOL_NAMES: dict[str, frozenset[str]] = {
    "basic": frozenset(
        {
            "ask_user_question",
            "translate_json",
            "execute_python_code",
            "execute_javascript_code",
            "execute_command",
            "api_call",
        }
    ),
    "web_search": frozenset(
        {
            "web_search",
            "zhipu_web_search",
            "exa_web_search",
            "fetch_web_content",
            "download_web_asset",
        }
    ),
}
LEGACY_CATEGORY_DISPATCH: dict[str, frozenset[str]] = {
    "basic": frozenset(
        {
            ToolCategory.BASIC.value,
            ToolCategory.CODE_EXECUTION.value,
            ToolCategory.WEB_TOOLS.value,
        }
    ),
    "web_search": frozenset({ToolCategory.WEB_TOOLS.value}),
}

CATEGORY_DISPLAY_NAMES = {
    "basic_tools": "Basic Tools",
    "code_execution": "Code Execution",
    "web_tools": "Web Tools",
    "vision": "Vision",
    "image": "Image",
    "video": "Video",
    "audio": "Audio",
    "knowledge": "Knowledge",
    "file": "File",
    "browser": "Browser",
    "ppt": "PPT",
    "agent": "Agent",
    "mcp": "MCP",
    "skill": "Skill",
    "ssh": "SSH",
    "database": "Database",
    "other": "Other",
}


def category_catalog() -> list[dict[str, str]]:
    return [
        {"id": category.value, "label": CATEGORY_DISPLAY_NAMES[category.value]}
        for category in ToolCategory
    ]


def legacy_category_catalog() -> list[dict[str, object]]:
    return [
        {"id": key, "tool_names": sorted(names)}
        for key, names in LEGACY_CATEGORY_TOOL_NAMES.items()
    ]


def legacy_category_admits(
    selected: frozenset[str], tool_name: str, category: str
) -> bool:
    if "basic" in selected and category == ToolCategory.BASIC.value:
        return True
    return any(tool_name in LEGACY_CATEGORY_TOOL_NAMES.get(key, ()) for key in selected)


def configurable_tool_category(tool_name: str) -> str:
    if tool_name in {
        "web_search",
        "tavily_web_search",
        "zhipu_web_search",
        "exa_web_search",
    }:
        return ToolCategory.WEB_TOOLS.value
    if tool_name == "sql_query":
        return ToolCategory.DATABASE.value
    return ToolCategory.OTHER.value
