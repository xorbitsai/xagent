"""``show_result_card``: show a connector-sourced answer as a chat card.

Offered only when the host enabled it (see ``agent.result_card``). The card is
validated against the connectors the run really invoked, bounded in size, and
emitted as a persisted assistant message that does not pause the run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from ....agent.result_card import (
    MAX_FOOTNOTE_CHARS,
    MAX_ITEM_CHARS,
    MAX_ITEMS,
    MAX_TITLE_CHARS,
    RESULT_CARD_DISPLAY,
    RESULT_CARD_MESSAGE_TYPE,
    RESULT_CARD_TOOL_NAME,
    current_run_display_context,
    resolve_result_card_source,
    result_card_enabled,
)
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool
from .selection_spec import normalize_mcp_server_name


class ShowResultCardArgs(BaseModel):
    title: str = Field(
        description=f"Short card title, at most {MAX_TITLE_CHARS} characters."
    )
    source_connector: str = Field(
        description=(
            "Name of the connector (MCP server) the data came from. It must be "
            "a connector you already called successfully in this run."
        )
    )
    items: list[str] = Field(
        description=(
            f"1 to {MAX_ITEMS} result lines, each at most {MAX_ITEM_CHARS} characters."
        )
    )
    footnote: Optional[str] = Field(
        default=None,
        description=f"Optional one-line note, at most {MAX_FOOTNOTE_CHARS} characters.",
    )


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


def _validated(args: ShowResultCardArgs) -> tuple[dict[str, Any] | None, str | None]:
    title = args.title.strip()
    items = [item.strip() for item in args.items]
    footnote = (args.footnote or "").strip() or None
    if not title or len(title) > MAX_TITLE_CHARS:
        return None, f"title must be 1-{MAX_TITLE_CHARS} characters"
    if not items or len(items) > MAX_ITEMS:
        return None, f"items must hold 1-{MAX_ITEMS} entries"
    if any(not item or len(item) > MAX_ITEM_CHARS for item in items):
        return None, f"each item must be 1-{MAX_ITEM_CHARS} characters"
    if footnote is not None and len(footnote) > MAX_FOOTNOTE_CHARS:
        return None, f"footnote must be at most {MAX_FOOTNOTE_CHARS} characters"
    return {"title": title, "items": items, "footnote": footnote}, None


class ShowResultCardTool(AbstractBaseTool):
    category = ToolCategory.BASIC

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return RESULT_CARD_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Show the user a compact result card with data you just read from a "
            "connector (for example a database query). The card names its "
            "source connector, which must be one you already called "
            "successfully in this run. It does not end or pause the run; still "
            "answer the user in your final answer."
        )

    def args_type(self) -> Type[BaseModel]:
        return ShowResultCardArgs

    def return_type(self) -> Type[BaseModel]:
        return BaseModel

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("show_result_card is async only")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        parsed = ShowResultCardArgs.model_validate(args)
        context = current_run_display_context()
        if not result_card_enabled() or context is None:
            return _error("Result cards are not available here.")
        connector = normalize_mcp_server_name(parsed.source_connector)
        if not connector or connector not in context.invoked_connectors:
            return _error(
                f"source_connector {parsed.source_connector!r} was not invoked "
                "in this run. Call a tool of that connector first, or name the "
                "connector you actually used."
            )
        body, problem = _validated(parsed)
        if body is None:
            return _error(f"Invalid result card: {problem}.")
        source = resolve_result_card_source(connector)
        card = {
            "title": body["title"],
            "source": {
                "name": source.name,
                "app_id": source.app_id,
                "logo": source.logo,
            },
            "items": body["items"],
            "footnote": body["footnote"],
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        await context.send_message(
            message=body["title"],
            message_type=RESULT_CARD_MESSAGE_TYPE,
            expect_response=False,
            visible=True,
            metadata={"display": RESULT_CARD_DISPLAY, "result_card": card},
        )
        return {"success": True, "shown": True}


@register_tool(selection_gate="intrinsic")
async def create_result_card_tool(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Offer the tool only when the host enabled result cards."""
    return [ShowResultCardTool()] if result_card_enabled() else []
