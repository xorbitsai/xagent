from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from ...core.computer.desktop_relay import (
    DESKTOP_RELAY_PROTOCOL_VERSION,
    get_desktop_relay_registry,
)
from ...core.computer.relay import (
    BROWSER_RELAY_MAX_MESSAGE_BYTES,
    BrowserRelayAuthenticationError,
    BrowserRelayConnection,
    BrowserRelayHello,
    BrowserRelayPing,
    BrowserRelayProtocolError,
    BrowserRelayResponse,
    DesktopRelayStatusMessage,
)
from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User

logger = logging.getLogger(__name__)

desktop_relay_router = APIRouter(tags=["desktop-relay"])


class DesktopRelayPairingResponse(BaseModel):
    pairing_token: str
    expires_at: str
    websocket_url: str
    protocol_version: int


class DesktopRelayStatusResponse(BaseModel):
    connected: bool
    attached: bool
    client_id: str | None = None
    client_name: str | None = None
    window_id: int | None = None
    title: str | None = None
    application: str | None = None
    bounds: dict[str, float] | None = None
    permissions: dict[str, bool] = Field(default_factory=dict)
    paused: bool = False
    emergency_stopped: bool = False
    connected_at: str | None = None


@desktop_relay_router.post(
    "/api/desktop-relay/pairings",
    response_model=DesktopRelayPairingResponse,
)
async def create_desktop_relay_pairing(
    request: Request,
    response: Response,
    user: User = Depends(get_current_user),
) -> DesktopRelayPairingResponse:
    pairing = await get_desktop_relay_registry().create_pairing(int(user.id))
    response.headers["Cache-Control"] = "no-store"
    websocket_url = str(request.url_for("desktop_relay_websocket"))
    return DesktopRelayPairingResponse(
        pairing_token=pairing.pairing_token,
        expires_at=pairing.expires_at.isoformat(),
        websocket_url=websocket_url,
        protocol_version=DESKTOP_RELAY_PROTOCOL_VERSION,
    )


@desktop_relay_router.get(
    "/api/desktop-relay/status",
    response_model=DesktopRelayStatusResponse,
)
async def get_desktop_relay_status(
    user: User = Depends(get_current_user),
) -> DesktopRelayStatusResponse:
    status = await get_desktop_relay_registry().status(int(user.id))
    return DesktopRelayStatusResponse.model_validate(status)


@desktop_relay_router.delete("/api/desktop-relay/session")
async def revoke_desktop_relay(
    user: User = Depends(get_current_user),
) -> dict[str, bool]:
    await get_desktop_relay_registry().revoke_user(int(user.id))
    return {"success": True}


@desktop_relay_router.websocket(
    "/ws/desktop-relay",
    name="desktop_relay_websocket",
)
async def desktop_relay_websocket(websocket: WebSocket) -> None:
    registry = get_desktop_relay_registry()
    connection: BrowserRelayConnection | None = None
    await websocket.accept()
    try:
        raw_hello = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        hello = BrowserRelayHello.model_validate(_load_message(raw_hello))
        authentication = await registry.authenticate(hello)
        if not _user_exists(authentication.user_id):
            raise BrowserRelayAuthenticationError(
                "Desktop relay user no longer exists."
            )

        async def close_transport(code: int, reason: str) -> None:
            await websocket.close(code=code, reason=reason[:123])

        connection = BrowserRelayConnection(
            user_id=authentication.user_id,
            client_id=authentication.client_id,
            client_name=authentication.client_name,
            send=websocket.send_json,
            close_transport=close_transport,
            authorization_id=authentication.session_id,
            target_kind="desktop",
        )
        await registry.register(connection)
        ready: dict[str, Any] = {
            "type": "ready",
            "protocol_version": DESKTOP_RELAY_PROTOCOL_VERSION,
            "paired": authentication.paired,
        }
        if authentication.session_token is not None:
            ready["session_token"] = authentication.session_token
        await websocket.send_json(ready)

        while True:
            message = _load_message(await websocket.receive_text())
            message_type = message.get("type")
            if message_type == "response":
                await connection.resolve(BrowserRelayResponse.model_validate(message))
            elif message_type == "status":
                status = DesktopRelayStatusMessage.model_validate(message)
                _require_protocol_version(status.protocol_version)
                await registry.update_connection_status(connection, status)
            elif message_type == "ping":
                ping = BrowserRelayPing.model_validate(message)
                _require_protocol_version(ping.protocol_version)
                await registry.touch_connection(connection)
                await websocket.send_json(
                    {
                        "type": "pong",
                        "protocol_version": DESKTOP_RELAY_PROTOCOL_VERSION,
                    }
                )
            else:
                raise BrowserRelayProtocolError(
                    f"Unsupported desktop relay message type: {message_type!r}."
                )
    except WebSocketDisconnect:
        pass
    except (
        BrowserRelayAuthenticationError,
        BrowserRelayProtocolError,
        ValueError,
    ) as exc:
        logger.info("Desktop relay rejected: %s", exc)
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "protocol_version": DESKTOP_RELAY_PROTOCOL_VERSION,
                    "error": str(exc),
                }
            )
            await websocket.close(code=1008)
        except Exception:
            pass
    except Exception:
        logger.exception("Desktop relay WebSocket failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if connection is not None:
            await registry.unregister(connection)


def _load_message(raw_message: str) -> dict[str, Any]:
    if len(raw_message.encode("utf-8")) > BROWSER_RELAY_MAX_MESSAGE_BYTES:
        raise BrowserRelayProtocolError("Desktop relay message is too large.")
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise BrowserRelayProtocolError("Desktop relay message must be JSON.") from exc
    if not isinstance(message, dict):
        raise BrowserRelayProtocolError("Desktop relay message must be an object.")
    return message


def _require_protocol_version(version: int) -> None:
    if version != DESKTOP_RELAY_PROTOCOL_VERSION:
        raise BrowserRelayProtocolError("Desktop relay protocol version mismatch.")


def _user_exists(user_id: int) -> bool:
    db_gen = get_db()
    db = next(db_gen)
    try:
        return db.query(User.id).filter(User.id == user_id).first() is not None
    finally:
        db.close()
