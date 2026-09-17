"""The governing-team id, the agent's creator, and its stored knowledge-base
declaration must actually reach the knowledge-base tool instances that
``knowledge_tools.py`` builds -- not just exist as parameters somewhere in
the call chain. Each test here breaks one link of that chain and would go
red if that link silently dropped its value. The list tool takes only the
governing-team id; the creator and the declaration belong to the search
tool alone.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from xagent.core.tools.adapters.vibe import knowledge_tools
from xagent.core.tools.adapters.vibe.document_search import (
    KnowledgeSearchTool,
    ListKnowledgeBasesTool,
)
from xagent.web.tools.config import WebToolConfig


class _FakeConfig:
    """A minimal ``BaseToolConfig``-shaped stand-in.

    ``_connector_team_id`` is a bare attribute, not a method: this mirrors
    how ``WebToolConfig`` actually stores it and how ``knowledge_tools.py``
    reads it (``getattr(config, "_connector_team_id", None)``, the
    accessor-less option this seam settled on).
    """

    def __init__(
        self,
        *,
        allowed_collections: Optional[List[str]] = None,
        user_id: Optional[int] = 7,
        is_admin: bool = False,
        connector_team_id: Optional[int] = None,
        agent_creator_user_id: Optional[int] = None,
        declared_knowledge_bases: Optional[List[str]] = None,
    ) -> None:
        self.allowed_collections = allowed_collections
        self.user_id = user_id
        self.is_admin_value = is_admin
        self._connector_team_id = connector_team_id
        self._agent_creator_user_id = agent_creator_user_id
        self._declared_knowledge_bases = declared_knowledge_bases

    def get_allowed_collections(self) -> Optional[List[str]]:
        return self.allowed_collections

    def get_user_id(self) -> Optional[int]:
        return self.user_id

    def is_admin(self) -> bool:
        return self.is_admin_value

    def get_agent_creator_user_id(self) -> Optional[int]:
        return self._agent_creator_user_id

    def get_declared_knowledge_bases(self) -> Optional[List[str]]:
        return self._declared_knowledge_bases


async def _build_tools(
    config: Any,
) -> tuple[ListKnowledgeBasesTool, KnowledgeSearchTool]:
    tools = await knowledge_tools._create_knowledge_tools_impl(config)
    list_tool = next(t for t in tools if isinstance(t, ListKnowledgeBasesTool))
    search_tool = next(t for t in tools if isinstance(t, KnowledgeSearchTool))
    return list_tool, search_tool


@pytest.mark.asyncio
async def test_missing_governing_team_id_leaves_tools_untouched_by_default() -> None:
    """Negative control: a config with no team id set (the common,
    non-team-governed case) must not accidentally acquire one."""
    config = _FakeConfig()
    list_tool, search_tool = await _build_tools(config)

    assert list_tool.governing_team_id is None
    assert search_tool.governing_team_id is None
    assert search_tool.agent_creator_user_id is None
    # The list tool carries no creator at all: it reports what is available
    # and classifies no individual declared name.
    assert not hasattr(list_tool, "agent_creator_user_id")


@pytest.mark.asyncio
async def test_running_the_search_tool_forwards_all_three_values_through_the_real_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tests above build tools and inspect their instance attributes,
    which stops one hop short of proving anything: those attributes still
    have to survive ``run_json_async`` -> the *real* compatibility facade
    (not a monkeypatched stand-in) -> the core impl function. This test
    walks the whole chain by actually invoking the tool and intercepting
    only the innermost core function, so a facade method that quietly
    drops one of the three values on its way through is caught here even
    though it would not be caught by asserting on the tool object alone.
    """
    from xagent.core.tools.core import document_search as core_document_search

    captured: dict = {}

    async def _fake_search_impl(tool_args, **kwargs):
        captured.update(kwargs)
        return core_document_search.KnowledgeSearchResult(results=[], summary="ok")

    monkeypatch.setattr(
        core_document_search, "_search_knowledge_base_impl", _fake_search_impl
    )

    config = WebToolConfig(
        db=None,
        request=None,
        user_id=7,
        connector_team_id=42,
        agent_creator_user_id=99,
        declared_knowledge_bases=["kb1"],
    )
    _, search_tool = await _build_tools(config)

    await search_tool.run_json_async({"query": "hello"})

    assert captured.get("governing_team_id") == 42
    assert captured.get("agent_creator_user_id") == 99
    assert captured.get("declared_knowledge_bases") == ["kb1"]


@pytest.mark.asyncio
async def test_running_the_list_tool_forwards_only_the_governing_team_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The governing team id must survive the same whole chain for the list
    tool, and nothing else must ride along with it: the config below does
    carry a creator and a declaration (the search tool reads both), so this
    also pins that the list path leaves them behind.
    """
    from xagent.core.tools.core import document_search as core_document_search

    captured: dict = {}

    async def _fake_list_impl(tool_args, **kwargs):
        captured.update(kwargs)
        return core_document_search.ListKnowledgeBasesResult(knowledge_bases=[])

    monkeypatch.setattr(
        core_document_search, "_list_knowledge_bases_impl", _fake_list_impl
    )

    config = WebToolConfig(
        db=None,
        request=None,
        user_id=7,
        allowed_collections=None,  # required for the list tool to be built
        connector_team_id=42,
        agent_creator_user_id=99,
        declared_knowledge_bases=["kb1"],
    )
    list_tool, _ = await _build_tools(config)

    await list_tool.run_json_async({})

    assert captured.get("governing_team_id") == 42
    assert "agent_creator_user_id" not in captured
    assert "declared_knowledge_bases" not in captured
