"""Fetch webpage content tool for xagent."""

from __future__ import annotations

from typing import Any, Mapping, Type

from pydantic import BaseModel, Field

from .....config import (
    get_fetch_web_content_budget_enabled,
    get_fetch_web_content_max_chars,
)
from ...core.web_content import (
    FINAL_SLICE_NOTICE,
    TRUNCATION_NOTICE,
    fetch_web_content,
    slice_page_text,
)
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
            "Maximum characters of page text to return in this call. Leave it "
            "unset for the default budget, which is enough for most questions."
        ),
    )
    start_char: int = Field(
        default=0,
        ge=0,
        description=(
            "Character offset to start reading from. When a page was cut, the "
            "notice at the end of its text gives the start_char that continues it."
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
        description="True when more page text remains after this result",
    )
    total_chars: int | None = Field(
        default=None,
        description="Length of the full page text, set when the budget applied",
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
            "pages. "
            + self._budget_description()
            + "When the task needs a logo or other exact web "
            "asset, set include_assets=true and usually asset_query='logo'. The "
            "returned assets preserve official page URLs; pass the chosen URL to "
            "download_web_asset instead of recreating it with api_call/write_file."
        )

    @staticmethod
    def _budget_description() -> str:
        if not get_fetch_web_content_budget_enabled():
            return "The full page text is returned. "
        return (
            f"The returned text is capped at {get_fetch_web_content_max_chars()} "
            "characters per call; a cut page says so and gives the start_char "
            "to continue from. "
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
        payload = result.as_dict()
        if budget and result.success:
            limit = fetch_args.max_chars or get_fetch_web_content_max_chars()
            text, end, total = slice_page_text(
                result.content, fetch_args.start_char, limit
            )
            payload["total_chars"] = total
            if fetch_args.start_char >= total > 0:
                text = f"[start_char is past the end of the page ({total} characters).]"
            elif end < total:
                text += TRUNCATION_NOTICE.format(
                    start=fetch_args.start_char, end=end, total=total
                )
            elif fetch_args.start_char > 0:
                text += FINAL_SLICE_NOTICE.format(
                    start=fetch_args.start_char, end=end, total=total
                )
            payload["content"] = text
            payload["truncated"] = end < total
        return FetchWebContentResult.model_validate(payload).model_dump()
