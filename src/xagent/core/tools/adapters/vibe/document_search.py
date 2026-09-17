"""Knowledge base search tools for Vibe agents."""

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel

from ...core.document_search import (
    KnowledgeSearchArgs,
    KnowledgeSearchResult,
    ListKnowledgeBasesArgs,
    ListKnowledgeBasesResult,
)
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.RAG_tools.kb import KBToolCompatibilityFacade


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned KB tool compatibility facade."""
    from ...core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


class ListKnowledgeBasesTool(AbstractBaseTool):
    category = ToolCategory.KNOWLEDGE
    """Tool to list all available knowledge bases."""

    def __init__(
        self,
        allowed_collections: Optional[list[str]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        governing_team_id: Optional[int] = None,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self.allowed_collections = allowed_collections
        self.user_id = user_id
        self.is_admin = is_admin
        self.governing_team_id = governing_team_id

    @property
    def name(self) -> str:
        return "list_knowledge_bases"

    @property
    def description(self) -> str:
        return """List all available knowledge bases with their statistics.
        Use this tool to see what knowledge bases are available before searching.
        Returns collection names, document counts, chunk counts, and embedding counts."""

    @property
    def tags(self) -> list[str]:
        return ["knowledge", "list", "inventory"]

    def args_type(self) -> Type[BaseModel]:
        return ListKnowledgeBasesArgs

    def return_type(self) -> Type[BaseModel]:
        return ListKnowledgeBasesResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError(
            "ListKnowledgeBasesTool only supports async execution."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Merge allowed_collections from tool init with args
        if self.allowed_collections is not None:
            args = dict(args)
            args.setdefault("allowed_collections", self.allowed_collections)
        tool_args = ListKnowledgeBasesArgs.model_validate(args)
        return await _get_tool_compatibility_facade().list_knowledge_bases(
            tool_args,
            user_id=self.user_id,
            is_admin=self.is_admin,
            governing_team_id=self.governing_team_id,
        )


def get_list_knowledge_bases_tool(
    allowed_collections: Optional[list[str]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
) -> ListKnowledgeBasesTool:
    """Create a tool to list all knowledge bases through the tool facade."""
    return _get_tool_compatibility_facade().get_list_knowledge_bases_tool(
        allowed_collections=allowed_collections,
        user_id=user_id,
        is_admin=is_admin,
        governing_team_id=governing_team_id,
    )


def _get_list_knowledge_bases_tool_impl(
    allowed_collections: Optional[list[str]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
) -> ListKnowledgeBasesTool:
    """Create a tool to list all knowledge bases.

    Args:
        allowed_collections: Optional list of allowed collection names to filter.
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the user has admin privileges.
        governing_team_id: The governing agent's owning team, if any.

    Returns:
        ListKnowledgeBasesTool instance
    """
    return ListKnowledgeBasesTool(
        allowed_collections=allowed_collections,
        user_id=user_id,
        is_admin=is_admin,
        governing_team_id=governing_team_id,
    )


class KnowledgeSearchTool(AbstractBaseTool):
    """Knowledge base search tool for Vibe agents."""

    category = ToolCategory.KNOWLEDGE

    def __init__(
        self,
        embedding_model_id: str | None = None,
        rerank_model_id: str | None = None,
        allowed_collections: Optional[list[str]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        governing_team_id: Optional[int] = None,
        agent_creator_user_id: Optional[int] = None,
        declared_knowledge_bases: Optional[list[str]] = None,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self.embedding_model_id = embedding_model_id
        self.rerank_model_id = rerank_model_id
        self.allowed_collections = allowed_collections
        self.user_id = user_id
        self.is_admin = is_admin
        self.governing_team_id = governing_team_id
        self.agent_creator_user_id = agent_creator_user_id
        self.declared_knowledge_bases = declared_knowledge_bases

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return """Search across knowledge bases for relevant documents.
        Use this tool when you need to find information from uploaded documents,
        knowledge bases, or document collections. Supports semantic search,
        keyword search, and hybrid search. Returns relevant document chunks
        with relevance scores. Can search all collections or specific ones by name.
        Treat one call as a top-k evidence set: inspect all returned results before
        deciding whether another search is needed. Search again only when the
        returned results as a group do not contain enough information to answer
        the current question."""

    @property
    def tags(self) -> list[str]:
        return ["search", "knowledge", "documents", "rag"]

    def args_type(self) -> Type[BaseModel]:
        return KnowledgeSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return KnowledgeSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("KnowledgeSearchTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Merge tool-level settings with args
        args = dict(args)
        if self.embedding_model_id:
            args.setdefault("embedding_model_id", self.embedding_model_id)
        if self.rerank_model_id:
            args.setdefault("rerank_model_id", self.rerank_model_id)
        if self.allowed_collections is not None:
            args.setdefault("allowed_collections", self.allowed_collections)

        # Debug: Log tool-level allowed_collections
        if self.allowed_collections is not None:
            logger.info(
                f"🔍 KnowledgeSearchTool allowed_collections: {self.allowed_collections}"
            )

        tool_args = KnowledgeSearchArgs.model_validate(args)
        return await _get_tool_compatibility_facade().search_knowledge_base(
            tool_args,
            user_id=self.user_id,
            is_admin=self.is_admin,
            governing_team_id=self.governing_team_id,
            agent_creator_user_id=self.agent_creator_user_id,
            declared_knowledge_bases=self.declared_knowledge_bases,
        )


def get_knowledge_search_tool(
    embedding_model_id: Optional[str] = None,
    rerank_model_id: Optional[str] = None,
    allowed_collections: Optional[list[str]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
    agent_creator_user_id: Optional[int] = None,
    declared_knowledge_bases: Optional[list[str]] = None,
) -> KnowledgeSearchTool:
    """Create a knowledge base search tool through the tool facade."""
    return _get_tool_compatibility_facade().get_knowledge_search_tool(
        embedding_model_id=embedding_model_id,
        rerank_model_id=rerank_model_id,
        allowed_collections=allowed_collections,
        user_id=user_id,
        is_admin=is_admin,
        governing_team_id=governing_team_id,
        agent_creator_user_id=agent_creator_user_id,
        declared_knowledge_bases=declared_knowledge_bases,
    )


def _get_knowledge_search_tool_impl(
    embedding_model_id: Optional[str] = None,
    rerank_model_id: Optional[str] = None,
    allowed_collections: Optional[list[str]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
    agent_creator_user_id: Optional[int] = None,
    declared_knowledge_bases: Optional[list[str]] = None,
) -> KnowledgeSearchTool:
    """Create a knowledge base search tool for Vibe agents.

    Args:
        embedding_model_id: Optional embedding model ID to use for searches.
        rerank_model_id: Optional rerank model ID (from model hub) to rerank results.
        allowed_collections: Optional list of allowed collection names. Used as default when collections is not specified.
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the user has admin privileges.
        governing_team_id: The governing agent's owning team, if any.
        agent_creator_user_id: The governing agent's creator, if any.
        declared_knowledge_bases: The governing agent's stored knowledge-base
            declaration, if any.

    Returns:
        KnowledgeSearchTool instance
    """
    return KnowledgeSearchTool(
        embedding_model_id=embedding_model_id,
        rerank_model_id=rerank_model_id,
        allowed_collections=allowed_collections,
        user_id=user_id,
        is_admin=is_admin,
        governing_team_id=governing_team_id,
        agent_creator_user_id=agent_creator_user_id,
        declared_knowledge_bases=declared_knowledge_bases,
    )
