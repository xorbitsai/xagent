"""Fetch webpage content tool for xagent."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Type

from pydantic import BaseModel, Field

from .....config import (
    get_fetch_web_content_budget_enabled,
    get_fetch_web_content_max_chars,
)
from ...core.web_content import TRUNCATION_NOTICE, fetch_web_content, truncate_page_text
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class FetchWebContentArgs(BaseModel):
    url: str = Field(description="HTTP or HTTPS URL to fetch and convert to text")
    include_assets: bool = Field(
        default=False,
        description=(
            "Also discover image, icon, manifest, script, and stylesheet URLs. "
            "Enable this when looking for logos, brand assets, or images."
        ),
    )
    asset_query: str | None = Field(
        default=None,
        description=(
            "Optional substring such as 'logo' used to filter assets discovered "
            "in the fetched HTML, matched against each asset's URL, name, and "
            "alt text. A <link rel=manifest> URL may be included among the "
            "discovered assets, but its contents are not fetched or searched."
        ),
    )

    max_chars: int | None = Field(
        default=None,
        ge=1000,
        le=50000,
        description=(
            "Maximum characters of page text to return. Defaults to a small "
            "budget (about 3k tokens) that is enough to answer most questions; "
            "raise it only when the part you need was cut off."
        ),
    )


class WebAssetReferenceResult(BaseModel):
    url: str
    kind: str
    name: str = ""
    alt: str = ""
    source: str = "html"


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
    blocked_by_site: bool = Field(
        default=False,
        description=(
            "True when the site refused automated access (for example HTTP "
            "401/403/429/451 or a bot challenge); do not retry the URL"
        ),
    )
    truncated: bool = Field(
        default=False,
        description="True when the page text was cut to the character budget",
    )
    total_chars: int | None = Field(
        default=None,
        description="Length of the full page text before it was cut, when known",
    )
    assets: list[WebAssetReferenceResult] = Field(
        default_factory=list,
        description=(
            "Static asset URLs discovered by parsing the fetched HTML (images, "
            "icons, stylesheets, scripts, and any linked manifest URL) — the "
            "manifest itself, if present, is not fetched or searched"
        ),
    )


class FetchWebContentTool(AbstractBaseTool):
    """Tool for reading a specific webpage after search discovers a URL."""

    category = ToolCategory.WEB_TOOLS
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
            "discover sources. Search snippets often already answer simple "
            "questions: fetch only when they do not, and read at most one or two "
            "pages. The returned text is capped (see max_chars); a cut page says "
            "so. When the task needs a logo or other exact web "
            "asset, set include_assets=true and usually asset_query='logo'. The "
            "returned assets preserve official page URLs; pass the chosen URL to "
            "download_web_asset instead of recreating it with api_call/write_file."
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
        budget = get_fetch_web_content_budget_enabled()
        result = await fetch_web_content(
            fetch_args.url,
            include_assets=fetch_args.include_assets,
            asset_query=fetch_args.asset_query,
            trim_boilerplate=budget,
        )
        if budget and result.success:
            limit = fetch_args.max_chars or get_fetch_web_content_max_chars()
            text, cut = truncate_page_text(result.content, limit)
            if cut:
                result = replace(
                    result,
                    content=text
                    + TRUNCATION_NOTICE.format(
                        shown=len(text), total=len(result.content)
                    ),
                    truncated=True,
                    total_chars=len(result.content),
                )
        return FetchWebContentResult.model_validate(result.as_dict()).model_dump()
