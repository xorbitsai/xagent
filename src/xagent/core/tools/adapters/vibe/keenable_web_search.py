"""
Keenable Web Search Tool for xagent
Framework wrapper around the Keenable Search API.
"""

from typing import Any, Dict, List, Mapping, Type

from pydantic import BaseModel, Field

from ...core.keenable_web_search import KeenableWebSearchCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class KeenableWebSearchArgs(BaseModel):
    query: str = Field(description="The search query string")
    num_results: int = Field(
        default=5, description="Number of results to return (max 50)"
    )
    include_content: bool = Field(
        default=False,
        description=(
            "Also include page content for each result. Defaults to false; "
            "prefer fetch_web_content for reading one selected URL."
        ),
    )


class KeenableWebSearchResult(BaseModel):
    results: List[Dict[str, str]] = Field(
        description="Search results with title, link, snippet, and optional content"
    )


class KeenableWebSearchTool(AbstractBaseTool):
    """Framework wrapper for the Keenable web search tool."""

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
            "Search the web for information using Keenable. "
            "Returns lightweight results with titles, links, and snippets by default. "
            "Set include_content=true only when every result needs page content; "
            "otherwise use fetch_web_content to read one selected URL."
        )

    @property
    def tags(self) -> list[str]:
        return ["search", "web", "information", "research", "keenable"]

    def args_type(self) -> Type[BaseModel]:
        return KeenableWebSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return KeenableWebSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError(
            "KeenableWebSearchTool only supports async execution."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        search_args = KeenableWebSearchArgs.model_validate(args)
        searcher = KeenableWebSearchCore(api_key=self._api_key)

        results = await searcher.search(
            query=search_args.query,
            max_results=search_args.num_results,
            include_content=search_args.include_content,
        )

        return KeenableWebSearchResult(results=results).model_dump()
