"""Fetch webpage content tool for xagent."""

from __future__ import annotations

from typing import Any, Mapping, Type

from pydantic import BaseModel, Field

from ...core.web_content import fetch_web_content
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class FetchWebContentArgs(BaseModel):
    url: str = Field(description="HTTP or HTTPS URL to fetch and convert to text")


class FetchWebContentResult(BaseModel):
    success: bool = Field(description="Whether the page was fetched successfully")
    url: str = Field(description="Fetched URL")
    title: str = Field(default="", description="Page title, when available")
    content: str = Field(
        default="", description="Cleaned webpage content converted to markdown"
    )
    status_code: int | None = Field(
        default=None, description="HTTP status code, when available"
    )
    content_type: str = Field(default="", description="HTTP content type")
    error: str | None = Field(default=None, description="Error message if failed")


class FetchWebContentTool(AbstractBaseTool):
    """Tool for reading a specific webpage after search discovers a URL."""

    category = ToolCategory.WEB_SEARCH
    read_only = True  # read-only HTTP GET ⇒ concurrency-safe

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "fetch_web_content"

    @property
    def description(self) -> str:
        return (
            "Fetch a specific webpage URL and convert its readable HTML content "
            "to markdown. Use this after web_search finds a promising source, or "
            "when the user provides a URL that needs to be read. This is for "
            "retrieving page content; use web_search first when you need to "
            "discover sources."
        )

    @property
    def tags(self) -> list[str]:
        return ["web", "fetch", "content", "url"]

    def args_type(self) -> Type[BaseModel]:
        return FetchWebContentArgs

    def return_type(self) -> Type[BaseModel]:
        return FetchWebContentResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("FetchWebContentTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        fetch_args = FetchWebContentArgs.model_validate(args)
        result = await fetch_web_content(fetch_args.url)
        return FetchWebContentResult.model_validate(result.as_dict()).model_dump()
