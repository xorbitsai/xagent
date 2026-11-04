"""
Web Search Tool for xagent
Framework wrapper around the pure web search tool
"""

import logging
from typing import Any, Dict, List, Mapping, Type

from pydantic import BaseModel, Field

from ...core.web_search import WebSearchCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query string")
    num_results: int = Field(
        default=3, description="Number of results to return (max 10)"
    )
    include_content: bool = Field(
        default=False,
        description=(
            "Also fetch webpage content for each result. Defaults to false; "
            "prefer fetch_web_content for reading one selected URL."
        ),
    )


class WebSearchResult(BaseModel):
    results: List[Dict[str, str]] = Field(
        description="Search results with title, link, snippet, and optional content"
    )


class WebSearchTool(AbstractBaseTool):
    category = ToolCategory.WEB_SEARCH
    read_only = True  # read-only HTTP search ⇒ concurrency-safe
    """Framework wrapper for the pure web search tool"""

    def __init__(self, api_key: str | None = None, cse_id: str | None = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._api_key = api_key
        self._cse_id = cse_id

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return """Search the web for information using Google Search.
        Returns lightweight results with titles, links, and snippets by default.
        Set include_content=true only when every result needs page text; otherwise
        use fetch_web_content to read one selected URL after search."""

    @property
    def tags(self) -> list[str]:
        return ["search", "web", "information", "research"]

    def args_type(self) -> Type[BaseModel]:
        return WebSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return WebSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("WebSearchTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        search_args = WebSearchArgs.model_validate(args)

        # Create core searcher instance
        searcher = WebSearchCore(api_key=self._api_key, cse_id=self._cse_id)

        # Perform search
        results = await searcher.search(
            query=search_args.query,
            num_results=search_args.num_results,
            include_content=search_args.include_content,
        )

        return WebSearchResult(results=results).model_dump()
