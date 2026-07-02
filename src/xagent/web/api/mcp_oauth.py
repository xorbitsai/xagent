"""Interactive OAuth connect/callback endpoints for remote MCP connectors."""

import logging
from datetime import datetime, timezone
from html import escape
from typing import Any, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.tools.core.mcp.oauth.flow import (
    build_authorization_url,
    code_challenge_for_verifier,
    decode_state,
    discover_auth_server,
    encode_state,
    exchange_code_for_tokens,
    new_pkce,
    register_client_dcr,
)
from ...core.tools.core.mcp.oauth.provider import (
    get_oauth_redirect_uri,
    oauth_client_metadata_dict,
)
from ...core.tools.core.mcp.oauth.ssrf_guard import UnsafeOAuthEndpointError
from ...core.utils.encryption import decrypt_value, encrypt_value
from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.mcp import MCPServer, UserMCPServer
from ..models.mcp_oauth import MCPUserOAuthToken
from ..models.user import User
from ..services.mcp_oauth_token_storage import DBTokenStorage
from .mcp import _can_edit_server_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp-oauth"])
OAUTH_MCP_TRANSPORTS = {"sse", "streamable_http"}


def _redirect_uri() -> str:
    try:
        return get_oauth_redirect_uri()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _oauth_client_metadata_dict() -> dict[str, Any]:
    try:
        return oauth_client_metadata_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _is_oauth_mcp_server(server: MCPServer) -> bool:
    auth = getattr(server, "auth", None)
    return isinstance(auth, dict) and auth.get("type") == "oauth_mcp"


def _supports_oauth_mcp_transport(server: MCPServer) -> bool:
    return cast(str, getattr(server, "transport", "")) in OAUTH_MCP_TRANSPORTS


def _client_info_from_registration(reg: dict) -> OAuthClientInformationFull:
    data = _oauth_client_metadata_dict()
    data.update(
        client_id=reg["client_id"],
        client_secret=reg.get("client_secret"),
        client_id_issued_at=reg.get("client_id_issued_at"),
        client_secret_expires_at=reg.get("client_secret_expires_at"),
    )
    return OAuthClientInformationFull.model_validate(data)


def _authorization_url(
    as_meta: dict, oauth_client: dict, state: str, code_challenge: str
) -> str:
    client_id = oauth_client.get("client_id")
    if not client_id:
        raise HTTPException(status_code=422, detail="OAuth client is missing client_id")
    scopes_supported = as_meta.get("scopes_supported")
    scope = " ".join(scopes_supported) if scopes_supported else None
    try:
        return build_authorization_url(
            as_meta,
            client_id=str(client_id),
            redirect_uri=_redirect_uri(),
            code_challenge=code_challenge,
            state=state,
            scope=scope,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _callback_html(message: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        f"<html><body><p>{escape(message)}</p></body></html>",
        status_code=status_code,
    )


def _get_authorized_server(db: Session, user_id: int, server_id: int) -> MCPServer:
    """Look up an MCP server, enforcing that ``user_id`` has access to it.

    Mirrors the ownership check in ``mcp.py``'s ``get_mcp_server``: server
    access is scoped via the ``UserMCPServer`` join table, not by id alone.
    """
    return _get_authorized_server_with_link(db, user_id, server_id)[1]


def _get_authorized_server_with_link(
    db: Session, user_id: int, server_id: int
) -> tuple[UserMCPServer, MCPServer]:
    """Like ``_get_authorized_server``, but also returns the ``UserMCPServer``
    link row so callers can additionally check edit/ownership permissions.
    """
    result = (
        db.query(UserMCPServer, MCPServer)
        .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return cast(UserMCPServer, result[0]), cast(MCPServer, result[1])


@router.post("/{server_id}/connect")
async def connect(
    server_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    user_id = int(cast(Any, user.id))
    user_mcp, server = _get_authorized_server_with_link(db, user_id, server_id)
    if not _is_oauth_mcp_server(server):
        raise HTTPException(
            status_code=400, detail="MCP server is not configured for OAuth"
        )
    if not _supports_oauth_mcp_transport(server):
        raise HTTPException(
            status_code=400,
            detail="OAuth MCP is only supported for SSE and Streamable HTTP transports",
        )
    server_data = cast(Any, server)
    if not server_data.url:
        raise HTTPException(status_code=400, detail="OAuth MCP server has no URL")
    client_metadata = _oauth_client_metadata_dict()

    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        as_meta = server_data.auth_server_metadata
        if not as_meta:
            try:
                as_meta = await discover_auth_server(server_data.url, client=client)
            except UnsafeOAuthEndpointError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "This server's OAuth configuration points to a "
                        "disallowed address."
                    ),
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Authorization server discovery failed: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Authorization server discovery failed",
                ) from exc
            server_data.auth_server_metadata = as_meta

        oauth_client = server_data.oauth_client
        if not oauth_client:
            if not _can_edit_server_config(user, user_mcp):
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to configure this connector",
                )
            try:
                reg = await register_client_dcr(as_meta, client_metadata, client=client)
            except UnsafeOAuthEndpointError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "This server's OAuth configuration points to a "
                        "disallowed address."
                    ),
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Automatic registration unavailable: {exc}. "
                        "Ask an admin to configure a client for this server."
                    ),
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502, detail="Automatic client registration failed"
                ) from exc
            client_info = _client_info_from_registration(reg)
            oauth_client = client_info.model_dump(mode="json", exclude_none=True)
            if oauth_client.get("client_secret"):
                oauth_client["client_secret"] = encrypt_value(
                    oauth_client["client_secret"]
                )
            oauth_client["source"] = "dcr"
            server_data.oauth_client = oauth_client
        db.commit()

    row = (
        db.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one_or_none()
    )
    row_data = cast(Any, row)
    if (
        row
        and row_data.status == "pending"
        and row_data.state
        and row_data.pkce_verifier
    ):
        verifier = decrypt_value(row_data.pkce_verifier)
        return {
            "authorization_url": _authorization_url(
                as_meta,
                oauth_client,
                row_data.state,
                code_challenge_for_verifier(verifier),
            )
        }

    verifier, challenge = new_pkce()
    state = encode_state(user_id=user_id, mcpserver_id=server_id)

    if row is None:
        row = MCPUserOAuthToken(user_id=user_id, mcpserver_id=server_id)
        db.add(row)
        row_data = cast(Any, row)
    row_data.status = "pending"
    row_data.pkce_verifier = encrypt_value(verifier)
    row_data.state = state
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        row = (
            db.query(MCPUserOAuthToken)
            .filter_by(user_id=user_id, mcpserver_id=server_id)
            .one_or_none()
        )
        row_data = cast(Any, row)
        if (
            row
            and row_data.status == "pending"
            and row_data.state
            and row_data.pkce_verifier
        ):
            verifier = decrypt_value(row_data.pkce_verifier)
            return {
                "authorization_url": _authorization_url(
                    as_meta,
                    oauth_client,
                    row_data.state,
                    code_challenge_for_verifier(verifier),
                )
            }
        raise HTTPException(
            status_code=409, detail="Authorization is already in progress"
        ) from exc

    return {
        "authorization_url": _authorization_url(as_meta, oauth_client, state, challenge)
    }


@router.get("/oauth/callback")
async def callback(
    code: str | None = Query(None),
    state: str = Query(...),
    error: str | None = Query(None),
    error_description: str | None = Query(None),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    try:
        user_id, server_id, _ = decode_state(state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid state") from exc

    row = (
        db.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id, state=state)
        .one_or_none()
    )
    row_data = cast(Any, row)
    if row is None or not row_data.pkce_verifier:
        raise HTTPException(status_code=400, detail="No pending authorization")

    if error:
        row_data.status = "error"
        row_data.pkce_verifier = None
        row_data.state = None
        db.commit()
        detail = error_description or error
        return _callback_html(
            f"Authorization failed: {detail}. Please try connecting again.",
            status_code=400,
        )

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    server = db.query(MCPServer).filter_by(id=server_id).one()
    server_data = cast(Any, server)
    oc = server_data.oauth_client or {}
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            tokens = await exchange_code_for_tokens(
                server_data.auth_server_metadata,
                client_id=oc["client_id"],
                client_secret=(
                    decrypt_value(oc["client_secret"])
                    if oc.get("client_secret")
                    else None
                ),
                code=code,
                code_verifier=decrypt_value(row_data.pkce_verifier),
                redirect_uri=_redirect_uri(),
                client=client,
            )
        oauth_token = OAuthToken.model_validate(tokens)
    except UnsafeOAuthEndpointError as exc:
        logger.warning(
            "OAuth token exchange blocked by SSRF guard for server %s: %s",
            server_id,
            exc,
        )
        row_data.status = "error"
        row_data.pkce_verifier = None
        row_data.state = None
        db.commit()
        return _callback_html(
            "This server's OAuth configuration points to a disallowed address.",
            status_code=400,
        )
    except (httpx.HTTPError, ValueError) as exc:
        # ValueError also covers json.JSONDecodeError (raised inside
        # exchange_code_for_tokens's resp.json()) and pydantic's
        # ValidationError, both of which can occur on a 200 response with a
        # malformed/incomplete token body -- treat that the same as a
        # transport-level failure rather than leaking an unhandled 500.
        logger.warning("OAuth token exchange failed for server %s: %s", server_id, exc)
        row_data.status = "error"
        row_data.pkce_verifier = None
        row_data.state = None
        db.commit()
        return _callback_html(
            "Authorization failed. Please try connecting again.", status_code=400
        )

    if not await DBTokenStorage(user_id, server_id, db).set_tokens_if_row_exists(
        oauth_token
    ):
        return _callback_html(
            "Authorization was canceled or expired. Please connect again.",
            status_code=400,
        )
    db.commit()

    return HTMLResponse(
        "<html><body><p>Connected. You can close this window.</p>"
        "<script>window.close()</script></body></html>"
    )


@router.get("/{server_id}/connection")
async def connection_status(
    server_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _get_authorized_server(db, int(cast(Any, user.id)), server_id)
    row = (
        db.query(MCPUserOAuthToken)
        .filter_by(user_id=int(cast(Any, user.id)), mcpserver_id=server_id)
        .one_or_none()
    )
    if not row:
        return {"status": "not_connected"}

    row_data = cast(Any, row)
    if row_data.status == "connected" and not row_data.refresh_token:
        expires_at = row_data.expires_at
        if isinstance(expires_at, datetime):
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= datetime.now(timezone.utc):
                return {"status": "expired"}

    return {"status": row_data.status}


@router.delete("/{server_id}/connection")
async def disconnect(
    server_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    user_id = int(cast(Any, user.id))
    _get_authorized_server(db, user_id, server_id)
    row = (
        db.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one_or_none()
    )
    if row:
        db.delete(row)
        db.commit()
    return {"status": "not_connected"}
