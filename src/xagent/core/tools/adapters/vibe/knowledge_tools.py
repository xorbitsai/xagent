"""Knowledge base tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from ...core.knowledge_base_scope import KnowledgeBaseScopeError
from .factory import register_tool

if TYPE_CHECKING:
    from ...core.RAG_tools.kb import KBToolCompatibilityFacade
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned KB tool compatibility facade."""
    from ...core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


@register_tool(categories={"knowledge"})
async def create_knowledge_tools(config: "BaseToolConfig") -> List[Any]:
    """Create knowledge base search tools through the tool facade."""
    return await _get_tool_compatibility_facade().create_knowledge_tools(config)


async def _create_knowledge_tools_impl(config: "BaseToolConfig") -> List[Any]:
    """Create knowledge base search tools."""
    tools: List[Any] = []

    try:
        from .document_search import (
            get_knowledge_search_tool,
            get_list_knowledge_bases_tool,
        )

        allowed_collections = config.get_allowed_collections()
        user_id = config.get_user_id()
        is_admin = config.is_admin()
        # The governing agent's team, read the same way the six existing
        # WebToolConfig-internal call sites already read it: as the private
        # attribute, defaulting to None for every config that has no such
        # attribute at all (every non-web config).
        #
        # Deliberately not behind a dedicated accessor. The attribute already
        # has six direct readers inside WebToolConfig (config.py:1593, :1792,
        # :2078, :3530, :3546, :3592), none of which an accessor added here
        # would convert; a seventh reader asking the same question a seventh
        # way is a second convention nothing else adopts, and the next reader
        # then has to learn which of the two is authoritative. The two
        # genuinely new values below get accessors because they have no
        # existing readers to be inconsistent with.
        governing_team_id = getattr(config, "_connector_team_id", None)
        agent_creator_user_id = config.get_agent_creator_user_id()
        declared_knowledge_bases = config.get_declared_knowledge_bases()

        if allowed_collections is not None and len(allowed_collections) == 0:
            return []

        if allowed_collections is None:
            # No creator and no declaration here: the list tool only reports
            # which collections are available, classifies no individual
            # declared name, and never runs the creator-existence probe.
            list_tool = get_list_knowledge_bases_tool(
                allowed_collections=allowed_collections,
                user_id=user_id,
                is_admin=is_admin,
                governing_team_id=governing_team_id,
            )
            tools.append(list_tool)

        # NOTE: Do not inject the user's default rerank model here.
        # rerank is per-KB: the search pipeline only reranks when the
        # KB it is querying has rerank_model_id configured in its
        # collection metadata.
        knowledge_tool = get_knowledge_search_tool(
            allowed_collections=allowed_collections,
            user_id=user_id,
            is_admin=is_admin,
            governing_team_id=governing_team_id,
            agent_creator_user_id=agent_creator_user_id,
            declared_knowledge_bases=declared_knowledge_bases,
        )
        tools.append(knowledge_tool)
    except KnowledgeBaseScopeError:
        # Defence in depth only: the team hook is never invoked while tools
        # are being built (resolution happens per search call, not here), so
        # nothing today can raise this from inside this try block. Kept so a
        # future change that does resolve the team layer during tool build
        # cannot have its typed error silently swallowed by the blanket
        # handler below.
        raise
    except Exception as e:
        logger.warning(f"Failed to create knowledge tools: {e}")

    return tools
