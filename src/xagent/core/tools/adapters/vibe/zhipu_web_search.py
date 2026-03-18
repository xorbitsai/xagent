"""
Zhipu Web Search Tool for xagent
Framework wrapper around the Zhipu web search API.
"""

from typing import Any, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ...core.zhipu_web_search import ZhipuWebSearchCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class ZhipuWebSearchArgs(BaseModel):
    query: str = Field(description="The search query string")
    search_engine: str = Field(
        default="search_pro_sogou",
        description="Search engine code (search_std/search_pro/search_pro_sogou/search_pro_quark)",
    )
    search_intent: bool = Field(
        default=False, description="Whether to include search intent analysis"
    )
    count: int = Field(default=10, description="Number of results to return (1-50)")
    search_domain_filter: Optional[str] = Field(
        default=None, description="Restrict results to a domain"
    )
    search_recency_filter: str = Field(default="noLimit", description="Recency filter")
    content_size: str = Field(
        default="medium", description="Summary length (low/medium/high)"
    )
    request_id: Optional[str] = Field(default=None, description="Optional request id")
    user_id: Optional[str] = Field(default=None, description="Optional user id")


class ZhipuWebSearchResult(BaseModel):
    results: List[Dict[str, Any]] = Field(
        description="Search results with title, link, snippet and content"
    )
    search_intent: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Search intent analysis"
    )
    request_id: Optional[str] = Field(default=None, description="Request id")
    id: Optional[str] = Field(default=None, description="Response id")
    created: Optional[int] = Field(default=None, description="Response timestamp")


class ZhipuWebSearchTool(AbstractBaseTool):
    category = ToolCategory.BASIC
    """Framework wrapper for the Zhipu web search tool."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._api_key = api_key
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "zhipu_web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Zhipu Web Search API. "
            "Returns results with titles, links, snippets, and summaries."
        )

    @property
    def tags(self) -> list[str]:
        return ["search", "web", "information", "zhipu", "bigmodel"]

    def args_type(self) -> Type[BaseModel]:
        return ZhipuWebSearchArgs

    def return_type(self) -> Type[BaseModel]:
        return ZhipuWebSearchResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("ZhipuWebSearchTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        search_args = ZhipuWebSearchArgs.model_validate(args)
        searcher = ZhipuWebSearchCore(api_key=self._api_key, base_url=self._base_url)

        response = await searcher.search(
            query=search_args.query,
            search_engine=search_args.search_engine,
            search_intent=search_args.search_intent,
            count=search_args.count,
            search_domain_filter=search_args.search_domain_filter,
            search_recency_filter=search_args.search_recency_filter,
            content_size=search_args.content_size,
            request_id=search_args.request_id,
            user_id=search_args.user_id,
        )

        results = ZhipuWebSearchCore.normalize_results(response)

        return ZhipuWebSearchResult(
            results=results,
            search_intent=response.get("search_intent"),
            request_id=response.get("request_id"),
            id=response.get("id"),
            created=response.get("created"),
        ).model_dump()
