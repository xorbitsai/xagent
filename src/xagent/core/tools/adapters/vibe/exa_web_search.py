"""
Exa Web Search Tool for xagent
Framework wrapper around the Exa AI-powered search API.
"""

from typing import Any, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ...core.exa_web_search import ExaWebSearchCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class ExaWebSearchArgs(BaseModel):
    query: str = Field(description="The search query string")
    num_results: int = Field(
        default=10, description="Number of results to return (max 100)"
    )
    search_type: str = Field(
        default="auto",
        description="Search type: 'auto', 'neural', 'fast', or 'instant'",
    )
    content_mode: str = Field(
        default="highlights",
        description="Content retrieval mode: 'highlights', 'text', 'summary', or 'none'",
    )
    category: Optional[str] = Field(
        default=None,
        description="Focus category: 'company', 'research paper', 'news', 'personal site', 'financial report', 'people'",
    )
    include_domains: Optional[List[str]] = Field(
        default=None, description="Restrict results to these domains"
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None, description="Exclude results from these domains"
    )
    include_text: Optional[List[str]] = Field(
        default=None, description="Strings that must appear in webpage text"
    )
    exclude_text: Optional[List[str]] = Field(
        default=None, description="Strings to exclude from results"
    )
    start_published_date: Optional[str] = Field(
        default=None, description="Filter by published date start (ISO 8601)"
    )
    end_published_date: Optional[str] = Field(
        default=None, description="Filter by published date end (ISO 8601)"
    )


class ExaWebSearchResult(BaseModel):
    results: List[Dict[str, str]] = Field(
        description="Search results with title, link, snippet and content"
    )


class ExaWebSearchTool(AbstractBaseTool):
    """Framework wrapper for the Exa AI-powered web search tool."""

    category = ToolCategory.BASIC

    def __init__(self, api_key: str | None = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "exa_web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Exa AI-powered search. "
            "Returns results with titles, links, snippets, and content (highlights, full text, or summaries). "
            "Supports category filtering, domain filtering, text filtering, and date range filtering. "
            "Useful for finding current information, research, news, and company data."
        )

    @property
    def tags(self) -> list[str]:
        return ["search", "web", "information", "research", "exa"]

    def args_type(self) -> Type[BaseModel]:
        return ExaWebSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return ExaWebSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("ExaWebSearchTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        search_args = ExaWebSearchArgs.model_validate(args)
        searcher = ExaWebSearchCore(api_key=self._api_key)

        results = await searcher.search(
            query=search_args.query,
            num_results=search_args.num_results,
            search_type=search_args.search_type,
            content_mode=search_args.content_mode,
            category=search_args.category,
            include_domains=search_args.include_domains,
            exclude_domains=search_args.exclude_domains,
            include_text=search_args.include_text,
            exclude_text=search_args.exclude_text,
            start_published_date=search_args.start_published_date,
            end_published_date=search_args.end_published_date,
        )

        return ExaWebSearchResult(results=results).model_dump()
