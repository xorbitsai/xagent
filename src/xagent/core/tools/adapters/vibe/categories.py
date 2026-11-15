"""Capability categories shared by discovery, execution and the web catalog."""

from .base import ToolCategory

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


def configurable_tool_category(tool_name: str) -> str:
    if tool_name in {
        "web_search",
        "tavily_web_search",
        "zhipu_web_search",
        "exa_web_search",
        "brave_web_search",
    }:
        return ToolCategory.WEB_TOOLS.value
    if tool_name == "sql_query":
        return ToolCategory.DATABASE.value
    return ToolCategory.OTHER.value
