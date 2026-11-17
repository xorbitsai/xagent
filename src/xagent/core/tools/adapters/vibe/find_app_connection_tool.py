"""``find_app_connection``: let the agent connect an approved app mid-run.

The tool delegates resolution to the host's :class:`ConnectorResolver`. When the
user still has to connect the app, it returns a ``waiting_for_user`` result
carrying one tool-authored ``connect_connector`` interaction. That type is
deliberately not in ``INTERACTION_TYPES``, which feeds the model-facing
``ask_user_question`` schema; the front end renders it from the opaque
``connector`` payload and collects the credential out of band.

``live_widget`` marks the interaction as a control the user acts on directly,
so the engine does not append its default text box (see
``ReActPattern._send_waiting_message``).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Type
from uuid import uuid4

from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from ...connector_resolver import (
    ConnectorResolveRequest,
    ConnectorResolver,
    get_connector_resolver_hook,
)
from ...user_interaction import WAITING_FOR_USER_STATUS, ToolInteractionSettlement
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool
from .interaction_types import LIVE_WIDGET_KEY
from .mcp_approval_gate import current_tool_call_execution_context

logger = logging.getLogger(__name__)

CONNECT_CONNECTOR_INTERACTION_TYPE = "connect_connector"


class FindAppConnectionArgs(BaseModel):
    app_name: str = Field(
        description="Name of the app or data source the user wants to use."
    )
    purpose: str = Field(
        description="What you need the app for, in one sentence (shown to the user)."
    )


class FindAppConnectionResult(BaseModel):
    status: str = Field(description="connected, not_found or waiting_for_user")
    message: str = Field(description="What happened, for the agent")


class FindAppConnectionTool(AbstractBaseTool):
    category: ToolCategory = ToolCategory.BASIC

    def __init__(
        self,
        *,
        resolver: ConnectorResolver,
        user_id: int | None = None,
        task_id: str | None = None,
    ) -> None:
        self._resolver = resolver
        self._user_id = user_id
        self._task_id = task_id
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "find_app_connection"

    @property
    def description(self) -> str:
        return (
            "Use when the user needs an app or data source that none of your "
            "current tools covers. Looks for an approved app and, if the user "
            "has not connected it yet, shows them a card to connect it. Never "
            "ask the user to paste keys or secrets into the chat. After the "
            "card is completed the connected app's tools become available."
        )

    def args_type(self) -> Type[BaseModel]:
        return FindAppConnectionArgs

    def return_type(self) -> Type[BaseModel]:
        return FindAppConnectionResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        parsed = FindAppConnectionArgs.model_validate(args)
        context = current_tool_call_execution_context()
        interaction_id = str(uuid4())
        resolution = await self._resolver.resolve(
            ConnectorResolveRequest(
                app_name=parsed.app_name,
                purpose=parsed.purpose,
                interaction_id=interaction_id,
                user_id=self._user_id,
                task_id=self._task_id,
                task_source=context.task_source if context else None,
            )
        )
        if resolution.status == "connected":
            return {
                "success": True,
                "status": "connected",
                "message": resolution.message
                or f"{parsed.app_name} is already connected.",
                **dict(resolution.result or {}),
            }
        if resolution.status == "needs_connection" and resolution.connector:
            message = resolution.message or (f"Connect {parsed.app_name} to continue.")
            return {
                "success": False,
                "status": WAITING_FOR_USER_STATUS,
                "interaction_id": interaction_id,
                "message": message,
                "message_type": "question",
                "interactions": [
                    {
                        "type": CONNECT_CONNECTOR_INTERACTION_TYPE,
                        "field": CONNECT_CONNECTOR_INTERACTION_TYPE,
                        "label": message,
                        LIVE_WIDGET_KEY: True,
                        "connector": dict(resolution.connector),
                    }
                ],
            }
        return {
            "success": True,
            "status": "not_found",
            "message": resolution.message
            or (
                f"No approved app matches '{parsed.app_name}'. Tell the user "
                "it is not available; do not guess credentials or endpoints."
            ),
        }

    async def resume_user_interaction(
        self, *, interaction_id: str, response: str
    ) -> ToolInteractionSettlement | None:
        # ``response`` is an ordinary user message and always enters model
        # context, so nothing is derived from it. Only server-side state counts.
        del response
        state = await self._resolver.connection_state(
            interaction_id=interaction_id,
            user_id=self._user_id,
            task_id=self._task_id,
        )
        if state is None:
            return None
        if state.status == "connected":
            return ToolInteractionSettlement.succeeded(
                {
                    "success": True,
                    "status": "connected",
                    "message": "The app is connected.",
                    **dict(state.result or {}),
                }
            )
        if state.status == "skipped":
            return ToolInteractionSettlement.succeeded({"status": "skipped"})
        return None


@register_tool(selection_gate="intrinsic")
async def create_find_app_connection_tool(
    config: WebToolConfig,
) -> list[AbstractBaseTool]:
    """Offer the tool only when a host registered a connector resolver."""
    resolver = get_connector_resolver_hook()
    if resolver is None:
        return []
    return [
        FindAppConnectionTool(
            resolver=resolver,
            user_id=config.get_user_id(),
            task_id=config.get_task_id(),
        )
    ]
