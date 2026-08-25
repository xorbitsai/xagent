"""Shared authentication ownership for authenticated WebSocket transports."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from ..auth_dependencies import get_user_from_websocket_token
from ..models.database import get_session_local
from ..services.db_runtime import run_db_io_cancellation_safe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSocketPrincipal:
    """The complete authenticated identity needed by WebSocket transports."""

    id: int
    is_admin: bool
    guest_id: str | None = None
    widget_entity_key: str | None = None
    # Onboarding "Launch" step voice choice, verbatim from the raw
    # preferences JSON (not validated against VALID_VOICES here - an
    # unrecognized value is stored as-is and only becomes an inert no-op
    # later, inside apply_output_voice's own isinstance/lookup guard in
    # api/agents.py), or None if the key is unset.
    voice: str | None = None


class _WebSocketAuthenticationTerminated(Exception):
    """Authentication already sent a terminal transport response."""


async def send_websocket_authentication_infrastructure_failure(
    websocket: WebSocket,
    original_error: Exception,
) -> None:
    """Send the shared sanitized response for an authentication outage."""
    route_template = getattr(websocket.scope.get("route"), "path", "<unresolved>")
    logger.error(
        "WebSocket authentication infrastructure failure transport=websocket route=%s",
        route_template,
        exc_info=(
            type(original_error),
            original_error,
            original_error.__traceback__,
        ),
    )
    try:
        if websocket.application_state is WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Internal server error")
        elif "websocket.http.response" in (websocket.scope.get("extensions") or {}):
            await websocket.send_denial_response(
                JSONResponse(
                    status_code=503,
                    content={"detail": "Service temporarily unavailable"},
                )
            )
        else:
            await websocket.accept()
            await websocket.close(code=1011, reason="Internal server error")
    except Exception:
        logger.exception(
            "WebSocket authentication terminal response failure "
            "transport=websocket route=%s",
            route_template,
        )


def _load_websocket_principal_sync(token: str) -> WebSocketPrincipal | None:
    """Authenticate one token inside a worker-owned short Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        user = get_user_from_websocket_token(token, db)
        if user is None or user.id is None:
            return None
        preferences = getattr(user, "preferences", None)
        voice = preferences.get("voice") if isinstance(preferences, dict) else None
        return WebSocketPrincipal(
            id=int(user.id), is_admin=bool(user.is_admin), voice=voice
        )


async def get_authenticated_user(
    websocket: WebSocket, token: str | None = None
) -> WebSocketPrincipal | None:
    """Load a detached principal or return ``None`` for rejected credentials.

    Operational authentication failures are raised after this owner sends its
    terminal transport response. Cancellation and other process-control signals
    propagate unchanged.
    """

    if not token:
        return None
    try:
        return await run_db_io_cancellation_safe(
            lambda: _load_websocket_principal_sync(token)
        )
    except Exception as exc:
        await send_websocket_authentication_infrastructure_failure(
            websocket,
            exc,
        )
        raise _WebSocketAuthenticationTerminated() from exc
