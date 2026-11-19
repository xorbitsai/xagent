"""``show_result_card``: show a connector-sourced answer as a chat card.

Offered only when the host enabled it (see ``agent.result_card``). The card is
validated against the connectors the run really invoked, bounded in size, and
emitted as a persisted assistant message that does not pause the run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Mapping, Optional, Type

from pydantic import BaseModel, Field, StringConstraints, ValidationError

from .....web.tools.config import WebToolConfig
from ....agent.result_card import (
    MAX_FOOTNOTE_CHARS,
    MAX_ITEM_CHARS,
    MAX_ITEMS,
    MAX_TITLE_CHARS,
    RESULT_CARD_DISPLAY,
    RESULT_CARD_MESSAGE_TYPE,
    RESULT_CARD_TOOL_NAME,
    ResultCardConnector,
    current_run_display_context,
    resolve_result_card_source,
    result_card_enabled,
    result_card_metadata,
)
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool
from .selection_spec import normalize_mcp_server_name


def _text(max_chars: int) -> Any:
    return StringConstraints(strip_whitespace=True, min_length=1, max_length=max_chars)


class ShowResultCardArgs(BaseModel):
    title: Annotated[str, _text(MAX_TITLE_CHARS)] = Field(
        description=f"Short card title, at most {MAX_TITLE_CHARS} characters."
    )
    source_connector: str = Field(
        description=(
            "Name of the connector (MCP server) the data came from. It must be "
            "a connector you already called successfully in this run."
        )
    )
    items: list[Annotated[str, _text(MAX_ITEM_CHARS)]] = Field(
        min_length=1,
        max_length=MAX_ITEMS,
        description=(
            f"1 to {MAX_ITEMS} result lines, each at most {MAX_ITEM_CHARS} characters."
        ),
    )
    footnote: Optional[
        Annotated[
            str, StringConstraints(strip_whitespace=True, max_length=MAX_FOOTNOTE_CHARS)
        ]
    ] = Field(
        default=None,
        description=f"Optional one-line note, at most {MAX_FOOTNOTE_CHARS} characters.",
    )


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


def _identity(connector: str, invoked: frozenset[ResultCardConnector]) -> Any:
    """The invoked connector ``connector`` names, or ``None`` if none does.

    Two connectors of different kinds can share a name; the card is then
    attributed with an unknown kind so a host never brands it as either.
    """
    matches = [c for c in invoked if c.name == connector]
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else ResultCardConnector(connector, None)


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
        try:
            parsed = ShowResultCardArgs.model_validate(args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            return _error(f"Invalid result card: {problems}.")
        context = current_run_display_context()
        if not result_card_enabled() or context is None:
            return _error("Result cards are not available here.")
        name = normalize_mcp_server_name(parsed.source_connector)
        connector = _identity(name, context.invoked_connectors) if name else None
        if connector is None:
            return _error(
                f"source_connector {parsed.source_connector!r} was not invoked "
                "in this run. Call a tool of that connector first, or name the "
                "connector you actually used."
            )
        source = resolve_result_card_source(connector)
        card = {
            "title": parsed.title,
            "source": {
                "name": source.name,
                "app_id": source.app_id,
                "logo": source.logo,
            },
            "items": list(parsed.items),
            "footnote": parsed.footnote or None,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        sent = await context.send_message(
            message=parsed.title,
            message_type=RESULT_CARD_MESSAGE_TYPE,
            expect_response=False,
            visible=True,
            metadata={
                **context.message_source,
                "display": RESULT_CARD_DISPLAY,
                "result_card": card,
            },
        )
        if result_card_metadata(sent) is None:
            # An output guardrail replaces a blocked card with a plain notice.
            return {"success": True, "shown": False}
        return {"success": True, "shown": True}


@register_tool(selection_gate="intrinsic")
async def create_result_card_tool(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Offer the tool only when the host enabled result cards."""
    return [ShowResultCardTool()] if result_card_enabled() else []
