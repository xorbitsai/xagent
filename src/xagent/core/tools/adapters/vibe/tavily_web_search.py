"""
Tavily Web Search Tool for xagent
Framework wrapper around the Tavily web search API.
"""

from typing import Any, Dict, List, Mapping, Type

from pydantic import BaseModel, Field

from ...core.tavily_web_search import TavilyWebSearchCore
from .base import AbstractBaseTool, ToolVisibility


class TavilyWebSearchArgs(BaseModel):
    query: str = Field(description="The search query string")
    num_results: int = Field(
        default=5, description="Number of results to return (max 20)"
    )
    include_content: bool = Field(
        default=True, description="Include full webpage content"
    )


class TavilyWebSearchResult(BaseModel):
    results: List[Dict[str, str]] = Field(
        description="Search results with title, link, snippet and content"
    )


class TavilyWebSearchTool(AbstractBaseTool):
    """Framework wrapper for the Tavily web search tool."""

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information using Tavily Search. "
            "Returns results with titles, links, snippets, and full webpage content. "
            "Useful for finding current information, research, and factual data."
        )

    @property
    def tags(self) -> list[str]:
        return ["search", "web", "information", "research", "tavily"]

    def args_type(self) -> Type[BaseModel]:
        return TavilyWebSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return TavilyWebSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("TavilyWebSearchTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        search_args = TavilyWebSearchArgs.model_validate(args)
        searcher = TavilyWebSearchCore()

        results = await searcher.search(
            query=search_args.query,
            max_results=search_args.num_results,
            include_content=search_args.include_content,
        )

        return TavilyWebSearchResult(results=results).model_dump()
