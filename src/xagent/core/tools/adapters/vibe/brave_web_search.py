"""
Brave Web Search Tool for xagent
Framework wrapper around the Brave Search API.
"""

from typing import Any, Dict, List, Mapping, Type

from pydantic import BaseModel, Field

from ...core.brave_web_search import BraveWebSearchCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class BraveWebSearchArgs(BaseModel):
    query: str = Field(
        description="The search query string (max 600 characters and 75 words)"
    )
    num_results: int = Field(
        default=5, description="Number of results to return (max 20)"
    )
    include_content: bool = Field(
        default=False,
        description=(
            "Also include page content for each result. Defaults to false; "
            "prefer fetch_web_content for reading one selected URL."
        ),
    )


class BraveWebSearchResult(BaseModel):
    results: List[Dict[str, str]] = Field(
        description="Search results with title, link, snippet, and optional content"
    )


class BraveWebSearchTool(AbstractBaseTool):
    """Framework wrapper for the Brave web search tool."""

    category = ToolCategory.WEB_TOOLS
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
            "Search the web for information using Brave Search. "
            "Returns lightweight results with titles, links, and snippets by default. "
            "Set include_content=true only when every result needs page content; "
            "otherwise use fetch_web_content to read one selected URL."
        )

    @property
    def tags(self) -> list[str]:
        return ["search", "web", "information", "research", "brave"]

    def args_type(self) -> Type[BaseModel]:
        return BraveWebSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return BraveWebSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("BraveWebSearchTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        search_args = BraveWebSearchArgs.model_validate(args)
        searcher = BraveWebSearchCore(api_key=self._api_key)

        results = await searcher.search(
            query=search_args.query,
            max_results=search_args.num_results,
            include_content=search_args.include_content,
        )

        return BraveWebSearchResult(results=results).model_dump()
