"""
Tavily Web Search Tool for xagent
Framework wrapper around the Tavily web search API.
"""

from typing import Any, Dict, List, Mapping, Type

from pydantic import BaseModel, Field

from ...core.tavily_web_search import TavilyWebSearchCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class TavilyWebSearchArgs(BaseModel):
    query: str = Field(description="The search query string")
    num_results: int = Field(
        default=5, description="Number of results to return (max 20)"
    )
    include_content: bool = Field(
        default=False,
        description=(
            "Also include raw webpage content for each result. Defaults to false; "
            "prefer fetch_web_content for reading one selected URL."
        ),
    )


class TavilyWebSearchResult(BaseModel):
    results: List[Dict[str, str]] = Field(
        description="Search results with title, link, snippet, and optional content"
    )


class TavilyWebSearchTool(AbstractBaseTool):
    """Framework wrapper for the Tavily web search tool."""

    category = ToolCategory.WEB_SEARCH
    read_only = True  # read-only HTTP search ⇒ concurrency-safe

    def __init__(self, api_key: str | None = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information using Tavily Search. "
            "Returns lightweight results with titles, links, and snippets by default. "
            "Set include_content=true only when every result needs raw page content; "
            "otherwise use fetch_web_content to read one selected URL."
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
        searcher = TavilyWebSearchCore(api_key=self._api_key)

        results = await searcher.search(
            query=search_args.query,
            max_results=search_args.num_results,
            include_content=search_args.include_content,
        )

        return TavilyWebSearchResult(results=results).model_dump()
