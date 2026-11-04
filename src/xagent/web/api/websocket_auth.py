"""Shared authentication ownership for authenticated WebSocket transports."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar, cast

from fastapi import WebSocket
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from starlette.websockets import WebSocketState

from ...core.agent.voice_policy import voice_from_preferences
from ..auth_dependencies import get_user_from_websocket_token
from ..models.database import get_session_local
from ..services.db_runtime import (
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ..websocket_credentials import (
    AUTH_PROTOCOL,
    InvalidWebSocketCredentials,
    websocket_header_token,
    websocket_query_carries_credential,
)

logger = logging.getLogger(__name__)

# Bound idle/streaming revocation latency without a process-local event bus.
_SESSION_RECHECK_SECONDS = 5.0
_SESSION_MAX_POLL_FAILURES = 3
_Endpoint = TypeVar("_Endpoint", bound=Callable[..., Coroutine[Any, Any, None]])


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


async def accept_authenticated_websocket(websocket: WebSocket) -> None:
    """Negotiate only the fixed protocol, never the offered credential."""
    scope = getattr(websocket, "scope", {})
    if isinstance(scope, dict) and AUTH_PROTOCOL in scope.get("subprotocols", []):
        await websocket.accept(subprotocol=AUTH_PROTOCOL)
    else:
        await websocket.accept()


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
        if websocket.application_state is WebSocketState.DISCONNECTED:
            return
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


def _load_websocket_principal_sync(
    token: str, *, verify_exp: bool = True
) -> WebSocketPrincipal | None:
    """Authenticate one token inside a worker-owned short Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        user = get_user_from_websocket_token(token, db, verify_exp=verify_exp)
        if user is None or user.id is None:
            return None
        voice = voice_from_preferences(getattr(user, "preferences", None))
        return WebSocketPrincipal(
            id=int(user.id), is_admin=bool(user.is_admin), voice=voice
        )


async def get_authenticated_user(
    websocket: WebSocket, token: str | None = None, *, revalidate: bool = False
) -> WebSocketPrincipal | None:
    """Load a detached principal or return ``None`` for rejected credentials.

    Operational authentication failures are raised after this owner sends its
    terminal transport response. Cancellation and other process-control signals
    propagate unchanged. ``revalidate`` is only for accepted connections: it
    retains signature/identity/generation checks without expiring live streams.
    """

    if not token or (
        revalidate and websocket.application_state is not WebSocketState.CONNECTED
    ):
        return None
    try:
        principal = await run_db_io_cancellation_safe(
            lambda: _load_websocket_principal_sync(token, verify_exp=not revalidate)
        )
        if revalidate and websocket.application_state is not WebSocketState.CONNECTED:
            return None
        if revalidate and principal is None:
            await _close_revoked_session(websocket)
        return principal
    except Exception as exc:
        await send_websocket_authentication_infrastructure_failure(
            websocket,
            exc,
        )
        raise _WebSocketAuthenticationTerminated() from exc


async def _close_revoked_session(websocket: WebSocket) -> None:
    if websocket.application_state is WebSocketState.CONNECTED:
        await websocket.close(code=4001, reason="Session expired")


def revalidate_websocket_session(endpoint: _Endpoint) -> _Endpoint:
    """Own the endpoint and its durable-session monitor for the same lifetime.

    Polling also closes idle sockets and cancels in-flight streaming handlers,
    including when the password changed in a different API process. Endpoint
    finally blocks must finish unsubscribing/cancelling their own work before
    this wrapper returns. Inbound commands additionally recheck before dispatch.
    """
    signature = inspect.signature(endpoint, eval_str=True)

    @wraps(endpoint)
    async def guarded(*args: Any, **kwargs: Any) -> None:
        bound = signature.bind(*args, **kwargs)
        arguments = bound.arguments
        websocket = arguments["websocket"]
        token = arguments.get("token")
        scope = getattr(websocket, "scope", None)
        if isinstance(scope, dict):
            # Reject the legacy URI contract even when a valid header is also
            # present. Do not retain a credential-leaking fallback.
            if websocket_query_carries_credential(scope):
                await websocket.close(code=4001)
                return
            try:
                header_token = websocket_header_token(scope)
            except InvalidWebSocketCredentials:
                await websocket.close(code=4001, reason="Authentication required")
                return
            if header_token is not None:
                token = header_token
                bound.arguments["token"] = token
        args, kwargs = bound.args, bound.kwargs
        if not isinstance(token, str) or not token:
            await endpoint(*args, **kwargs)
            return

        async def monitor() -> None:
            failures = 0
            while True:
                # Stagger connection-time cohorts instead of polling in lockstep.
                await asyncio.sleep(_SESSION_RECHECK_SECONDS * random.uniform(0.8, 1.2))
                if websocket.application_state is WebSocketState.DISCONNECTED:
                    return
                if websocket.application_state is not WebSocketState.CONNECTED:
                    continue
                try:
                    principal = await run_db_io_cancellation_safe(
                        lambda: _load_websocket_principal_sync(token, verify_exp=False)
                    )
                except (OperationalError, SQLAlchemyTimeoutError) as exc:
                    failures += 1
                    if failures < _SESSION_MAX_POLL_FAILURES:
                        continue
                    await send_websocket_authentication_infrastructure_failure(
                        websocket, exc
                    )
                    return
                except Exception as exc:
                    await send_websocket_authentication_infrastructure_failure(
                        websocket, exc
                    )
                    return
                failures = 0
                if principal is None:
                    await _close_revoked_session(websocket)
                    return

        handler = asyncio.create_task(endpoint(*args, **kwargs))
        watcher = asyncio.create_task(monitor(), name="websocket-session-monitor")
        try:
            done, _ = await asyncio.wait(
                (handler, watcher), return_when=asyncio.FIRST_COMPLETED
            )
            if handler in done:
                await handler
            else:
                await watcher
        finally:

            async def cleanup() -> None:
                await cancel_and_drain_async_task(watcher)
                await cancel_and_drain_async_task(handler)

            await drain_async_task_cancellation_safe(asyncio.create_task(cleanup()))

    # Resolve annotations against the original module, not this wrapper's globals.
    setattr(guarded, "__signature__", signature)
    return cast(_Endpoint, guarded)
