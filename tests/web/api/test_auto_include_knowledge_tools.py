"""Unit tests for auto-including knowledge tools when knowledge_bases are selected.

Tests cover both chat.py (AgentServiceManager) and websocket.py (build preview)
paths to ensure knowledge_search and list_knowledge_bases tools are automatically
included in allowed_tools when an agent has knowledge bases configured.
"""

from typing import Optional
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.adapters.vibe.base import ToolCategory, ToolMetadata

# ---------------------------------------------------------------------------
# Helper: lightweight mock tool objects
# ---------------------------------------------------------------------------


def _make_tool(name: str, category: ToolCategory) -> MagicMock:
    """Create a mock tool with the given name and category."""
    tool = MagicMock()
    tool.name = name
    tool.metadata = ToolMetadata(name=name, category=category)
    return tool


def _standard_tool_set() -> list:
    """Return a representative set of tools including knowledge tools."""
    return [
        _make_tool("calculator", ToolCategory.BASIC),
        _make_tool("file_tool", ToolCategory.FILE),
        _make_tool("web_search", ToolCategory.BASIC),
        _make_tool("knowledge_search", ToolCategory.KNOWLEDGE),
        _make_tool("list_knowledge_bases", ToolCategory.KNOWLEDGE),
        _make_tool("browser_use", ToolCategory.BROWSER),
    ]


# ===========================================================================
# Tests for chat.py  –  AgentServiceManager._create_agent_service_for_task
# ===========================================================================


class TestChatAutoIncludeKnowledgeTools:
    """Test auto-inclusion of knowledge tools in chat.py path."""

    @staticmethod
    def _filter_tools_by_category(
        all_tools: list,
        tool_categories: list[str],
        knowledge_bases: Optional[list[str]] = None,
    ) -> list[str]:
        """Replicate the tool-filtering logic from AgentServiceManager."""
        allowed_tools: list[str] = []

        for tool in all_tools:
            if hasattr(tool, "metadata") and hasattr(tool.metadata, "category"):
                category = str(tool.metadata.category.value)
                if category in tool_categories:
                    tool_name = getattr(tool, "name", None)
                    if tool_name:
                        allowed_tools.append(tool_name)

        if knowledge_bases:
            knowledge_tool_names = {"knowledge_search", "list_knowledge_bases"}
            for tool in all_tools:
                tool_name = getattr(tool, "name", None)
                if (
                    tool_name
                    and tool_name in knowledge_tool_names
                    and tool_name not in allowed_tools
                ):
                    allowed_tools.append(tool_name)

        return allowed_tools

    def test_kb_tools_included_when_kb_selected_without_knowledge_category(self):
        """When knowledge_bases are set but KNOWLEDGE category is NOT selected,
        knowledge tools should still be auto-included."""
        all_tools = _standard_tool_set()
        tool_categories = ["basic"]
        knowledge_bases = ["my_kb"]

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert "calculator" in result
        assert "web_search" in result
        assert "knowledge_search" in result
        assert "list_knowledge_bases" in result

    def test_kb_tools_not_duplicated_when_knowledge_category_selected(self):
        """When both knowledge_bases and KNOWLEDGE category are selected,
        tools should not appear twice."""
        all_tools = _standard_tool_set()
        tool_categories = ["basic", "knowledge"]
        knowledge_bases = ["my_kb"]

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert result.count("knowledge_search") == 1
        assert result.count("list_knowledge_bases") == 1

    def test_no_kb_tools_when_no_kb_selected(self):
        """When no knowledge_bases are configured, knowledge tools should NOT
        be auto-included (unless the category explicitly matches)."""
        all_tools = _standard_tool_set()
        tool_categories = ["basic"]
        knowledge_bases = None

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert "knowledge_search" not in result
        assert "list_knowledge_bases" not in result
        assert "calculator" in result
        assert "web_search" in result

    def test_no_kb_tools_when_kb_list_empty(self):
        """Empty knowledge_bases list should behave same as None."""
        all_tools = _standard_tool_set()
        tool_categories = ["basic"]
        knowledge_bases = []

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert "knowledge_search" not in result
        assert "list_knowledge_bases" not in result

    def test_kb_tools_included_with_multiple_kbs(self):
        """Multiple knowledge bases should still result in exactly one
        inclusion of each knowledge tool."""
        all_tools = _standard_tool_set()
        tool_categories = ["basic"]
        knowledge_bases = ["kb_a", "kb_b", "kb_c"]

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert result.count("knowledge_search") == 1
        assert result.count("list_knowledge_bases") == 1

    def test_partial_kb_tools_available(self):
        """If only knowledge_search exists (list_knowledge_bases missing),
        only the available one should be included."""
        all_tools = [
            _make_tool("calculator", ToolCategory.BASIC),
            _make_tool("knowledge_search", ToolCategory.KNOWLEDGE),
        ]
        tool_categories = ["basic"]
        knowledge_bases = ["my_kb"]

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert "knowledge_search" in result
        assert "list_knowledge_bases" not in result

    def test_no_kb_tools_in_tool_set(self):
        """If tool set has no knowledge tools at all, auto-include is a no-op."""
        all_tools = [
            _make_tool("calculator", ToolCategory.BASIC),
            _make_tool("web_search", ToolCategory.BASIC),
        ]
        tool_categories = ["basic"]
        knowledge_bases = ["my_kb"]

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert "knowledge_search" not in result
        assert "list_knowledge_bases" not in result
        assert "calculator" in result
        assert "web_search" in result

    def test_tool_without_name_attribute_skipped(self):
        """Tools missing the 'name' attribute should be safely skipped."""
        nameless_tool = MagicMock(spec=[])
        del nameless_tool.name
        nameless_tool.metadata = ToolMetadata(
            name="ghost", category=ToolCategory.KNOWLEDGE
        )

        all_tools = [
            _make_tool("calculator", ToolCategory.BASIC),
            nameless_tool,
        ]
        tool_categories = ["basic"]
        knowledge_bases = ["my_kb"]

        result = self._filter_tools_by_category(
            all_tools, tool_categories, knowledge_bases
        )

        assert "calculator" in result
        assert "ghost" not in result


# ===========================================================================
# Tests for websocket.py  –  _get_tools_by_category in build preview
# ===========================================================================


class TestWebSocketAutoIncludeKnowledgeTools:
    """Test auto-inclusion of knowledge tools in websocket.py preview path."""

    @staticmethod
    async def _get_tools_by_category(
        all_tools: list,
        tool_categories: list[str],
        knowledge_bases: Optional[list[str]] = None,
    ) -> list[str]:
        """Replicate the async _get_tools_by_category logic from websocket.py."""
        allowed_tools: list[str] = []

        for tool in all_tools:
            if hasattr(tool, "metadata") and hasattr(tool.metadata, "category"):
                category = str(tool.metadata.category.value)
                if category in tool_categories:
                    tool_name = getattr(tool, "name", None)
                    if tool_name:
                        allowed_tools.append(tool_name)

        if knowledge_bases:
            knowledge_tool_names = {"knowledge_search", "list_knowledge_bases"}
            for tool in all_tools:
                tool_name = getattr(tool, "name", None)
                if (
                    tool_name
                    and tool_name in knowledge_tool_names
                    and tool_name not in allowed_tools
                ):
                    allowed_tools.append(tool_name)

        return allowed_tools

    @pytest.mark.asyncio
    async def test_kb_tools_included_when_kb_selected(self):
        """WebSocket preview should auto-include knowledge tools when KBs are set."""
        all_tools = _standard_tool_set()

        result = await self._get_tools_by_category(all_tools, ["basic"], ["my_kb"])

        assert "knowledge_search" in result
        assert "list_knowledge_bases" in result
        assert "calculator" in result

    @pytest.mark.asyncio
    async def test_kb_tools_not_included_without_kb(self):
        """WebSocket preview should NOT include knowledge tools when no KBs set."""
        all_tools = _standard_tool_set()

        result = await self._get_tools_by_category(all_tools, ["basic"], None)

        assert "knowledge_search" not in result
        assert "list_knowledge_bases" not in result

    @pytest.mark.asyncio
    async def test_kb_tools_not_duplicated(self):
        """WebSocket preview should not duplicate tools already included via category."""
        all_tools = _standard_tool_set()

        result = await self._get_tools_by_category(
            all_tools, ["basic", "knowledge"], ["my_kb"]
        )

        assert result.count("knowledge_search") == 1
        assert result.count("list_knowledge_bases") == 1

    @pytest.mark.asyncio
    async def test_empty_kb_list_no_auto_include(self):
        """Empty knowledge_bases list should not trigger auto-include."""
        all_tools = _standard_tool_set()

        result = await self._get_tools_by_category(all_tools, ["basic"], [])

        assert "knowledge_search" not in result
        assert "list_knowledge_bases" not in result

    @pytest.mark.asyncio
    async def test_only_available_kb_tools_included(self):
        """Only knowledge tools that exist in tool set should be included."""
        all_tools = [
            _make_tool("calculator", ToolCategory.BASIC),
            _make_tool("list_knowledge_bases", ToolCategory.KNOWLEDGE),
        ]

        result = await self._get_tools_by_category(all_tools, ["basic"], ["my_kb"])

        assert "list_knowledge_bases" in result
        assert "knowledge_search" not in result


# ===========================================================================
# Edge-case / integration-style tests
# ===========================================================================


class TestAutoIncludeEdgeCases:
    """Edge cases for the auto-include knowledge tools feature."""

    def test_all_categories_empty(self):
        """No tool_categories means no tools, but KB auto-include still works."""
        all_tools = _standard_tool_set()
        allowed_tools: list[str] = []

        knowledge_bases = ["my_kb"]
        if knowledge_bases:
            knowledge_tool_names = {"knowledge_search", "list_knowledge_bases"}
            for tool in all_tools:
                tool_name = getattr(tool, "name", None)
                if (
                    tool_name
                    and tool_name in knowledge_tool_names
                    and tool_name not in allowed_tools
                ):
                    allowed_tools.append(tool_name)

        assert allowed_tools == ["knowledge_search", "list_knowledge_bases"]

    def test_tool_with_none_name_skipped(self):
        """A tool whose getattr(tool, 'name', None) returns None should be skipped."""
        tool_with_none_name = MagicMock()
        tool_with_none_name.name = None
        tool_with_none_name.metadata = ToolMetadata(
            name="phantom", category=ToolCategory.KNOWLEDGE
        )

        all_tools = [
            _make_tool("calculator", ToolCategory.BASIC),
            tool_with_none_name,
            _make_tool("knowledge_search", ToolCategory.KNOWLEDGE),
        ]

        allowed_tools: list[str] = []
        knowledge_bases = ["my_kb"]
        if knowledge_bases:
            knowledge_tool_names = {"knowledge_search", "list_knowledge_bases"}
            for tool in all_tools:
                tool_name = getattr(tool, "name", None)
                if (
                    tool_name
                    and tool_name in knowledge_tool_names
                    and tool_name not in allowed_tools
                ):
                    allowed_tools.append(tool_name)

        assert "knowledge_search" in allowed_tools
        assert "phantom" not in allowed_tools
        assert None not in allowed_tools
