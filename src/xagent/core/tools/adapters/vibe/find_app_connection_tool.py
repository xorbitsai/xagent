"""``find_app_connection``: let the agent connect an approved app mid-run.

The tool delegates resolution to the host's :class:`ConnectorResolver`. When the
user still has to connect the app, it returns a ``waiting_for_user`` result
carrying one tool-authored ``connect_connector`` interaction. That type is
deliberately not in ``INTERACTION_TYPES``, which feeds the model-facing
``ask_user_question`` schema.

The host must ship a renderer for ``connect_connector``: it draws a card from
the opaque ``connector`` payload and collects the credential out of band. The
interaction item carries ``interaction_id`` (also returned at the top level of
the waiting result): the renderer reads it from the item and passes it to the
host's save and skip endpoints, so the host does not need to put it into
``connector``. The
stock front end does not know the type. Without a renderer the user sees only
the message text and, because ``live_widget`` suppresses the default text box,
no field to answer in.

The tool is registered only while a resolver is set, is refused for the public
``widget`` and ``shared_link`` task sources, and is not offered to delegated
child agents, which cannot pause for the user.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Type
from uuid import uuid4

from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from ...connector_resolver import (
    ConnectorResolver,
    ConnectorResolveRequest,
    get_connector_resolver_hook,
)
from ...user_interaction import WAITING_FOR_USER_STATUS, ToolInteractionSettlement
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool
from .interaction_types import LIVE_WIDGET_KEY
from .mcp_approval_gate import current_tool_call_execution_context

logger = logging.getLogger(__name__)

CONNECT_CONNECTOR_INTERACTION_TYPE = "connect_connector"

# Task sources whose runs are driven by a guest while executing as the owner.
GUEST_TASK_SOURCES = frozenset({"widget", "shared_link"})

_GUEST_REFUSAL = {
    "success": False,
    "status": "unavailable",
    "message": (
        "Connecting an app is not available in this conversation. Tell the user "
        "it cannot be done here."
    ),
}


class FindAppConnectionArgs(BaseModel):
    app_name: str = Field(
        description="Name of the app or data source the user wants to use."
    )
    purpose: str = Field(
        description="What you need the app for, in one sentence (shown to the user)."
    )


class FindAppConnectionResult(BaseModel):
    status: str = Field(description="connected, not_found, skipped or waiting_for_user")
    message: str = Field(description="What happened, for the agent")


_CONTROL_KEYS = frozenset({"success", "status", "message", "is_error"})


def _host_fields(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Host-supplied result data minus every key the tool result is judged by.

    ``success``, ``status`` and ``is_error`` decide whether a result counts as
    succeeded (``tool_result_succeeded``), so host data must not carry them.
    """

    return {k: v for k, v in (data or {}).items() if k not in _CONTROL_KEYS}


def _current_task_source() -> str | None:
    context = current_tool_call_execution_context()
    return context.task_source if context else None


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
        task_source = _current_task_source()
        if task_source in GUEST_TASK_SOURCES:
            return dict(_GUEST_REFUSAL)
        interaction_id = str(uuid4())
        try:
            resolution = await self._resolver.resolve(
                ConnectorResolveRequest(
                    app_name=parsed.app_name,
                    purpose=parsed.purpose,
                    interaction_id=interaction_id,
                    user_id=self._user_id,
                    task_id=self._task_id,
                    task_source=task_source,
                )
            )
        except Exception as exc:
            logger.warning(
                "find_app_connection resolver failed (%s)", type(exc).__name__
            )
            return {
                "success": False,
                "status": "unavailable",
                "message": "Looking up the app failed. Tell the user to try again.",
            }
        if resolution.status == "connected":
            return {
                **_host_fields(resolution.result),
                "success": True,
                "status": "connected",
                "message": resolution.message
                or f"{parsed.app_name} is already connected.",
            }
        if resolution.status == "needs_connection":
            if resolution.connector:
                message = resolution.message or (
                    f"Connect {parsed.app_name} to continue."
                )
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
                            # The card calls the host's endpoints with this id;
                            # it is the same value as the top-level one.
                            "interaction_id": interaction_id,
                            "connector": dict(resolution.connector),
                        }
                    ],
                }
            logger.warning(
                "find_app_connection resolver returned needs_connection "
                "without a connector payload; reporting not_found"
            )
        elif resolution.status != "not_found":
            logger.warning(
                "find_app_connection resolver returned an unknown status %r; "
                "reporting not_found",
                resolution.status,
            )
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
        task_source = _current_task_source()
        if task_source in GUEST_TASK_SOURCES:
            return None
        try:
            state = await self._resolver.connection_state(
                interaction_id=interaction_id,
                user_id=self._user_id,
                task_id=self._task_id,
                task_source=task_source,
            )
        except Exception as exc:
            # A transient host error must not fail the resumed run; returning
            # None lets the model replan from the user's message.
            logger.warning(
                "find_app_connection state lookup failed (%s)", type(exc).__name__
            )
            return None
        if state is None:
            return None
        if state.status == "connected":
            return ToolInteractionSettlement.succeeded(
                {
                    **_host_fields(state.result),
                    "success": True,
                    "status": "connected",
                    "message": "The app is connected.",
                }
            )
        if state.status == "skipped":
            return ToolInteractionSettlement.succeeded(
                {
                    "status": "skipped",
                    "message": (
                        "The user chose to skip connecting the app. Do not call "
                        "find_app_connection for it again; continue without it "
                        "or tell the user what cannot be done."
                    ),
                }
            )
        return None


@register_tool(selection_gate="intrinsic")
async def create_find_app_connection_tool(
    config: WebToolConfig,
) -> list[AbstractBaseTool]:
    """Offer the tool only when a host set a resolver and the run can pause."""
    resolver = get_connector_resolver_hook()
    if resolver is None:
        return []
    if config.get_parent_task_id():
        # Delegated child agents run with user interaction disabled: a
        # needs_connection pause would fail the whole child run.
        return []
    return [
        FindAppConnectionTool(
            resolver=resolver,
            user_id=config.get_user_id(),
            task_id=config.get_task_id(),
        )
    ]
