"""Interactive OAuth connect/callback endpoints for remote MCP connectors."""

import logging
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from mcp.shared.auth import OAuthClientInformationFull
from sqlalchemy.orm import Session

from ...config import get_oauth_callback_base_url
from ...core.tools.core.mcp.oauth.flow import (
    build_authorization_url,
    decode_state,
    discover_auth_server,
    encode_state,
    exchange_code_for_tokens,
    new_pkce,
    register_client_dcr,
)
from ...core.utils.encryption import decrypt_value, encrypt_value
from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.mcp import MCPServer, UserMCPServer
from ..models.mcp_oauth import MCPUserOAuthToken
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp-oauth"])


def _redirect_uri() -> str:
    return f"{get_oauth_callback_base_url().rstrip('/')}/api/mcp/oauth/callback"


def _get_authorized_server(db: Session, user_id: int, server_id: int) -> MCPServer:
    """Look up an MCP server, enforcing that ``user_id`` has access to it.

    Mirrors the ownership check in ``mcp.py``'s ``get_mcp_server``: server
    access is scoped via the ``UserMCPServer`` join table, not by id alone.
    """
    result = (
        db.query(UserMCPServer, MCPServer)
        .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return result[1]


@router.post("/{server_id}/connect")
async def connect(
    server_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    server = _get_authorized_server(db, user.id, server_id)
    if not server.url:
        raise HTTPException(status_code=404, detail="MCP server not found")

    async with httpx.AsyncClient(timeout=30) as client:
        as_meta = server.auth_server_metadata
        if not as_meta:
            as_meta = await discover_auth_server(server.url, client=client)
            server.auth_server_metadata = as_meta

        oauth_client = server.oauth_client
        if not oauth_client:
            try:
                reg = await register_client_dcr(
                    as_meta, [_redirect_uri()], client=client
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Automatic registration unavailable: {exc}. "
                        "Ask an admin to configure a client for this server."
                    ),
                ) from exc
            client_info = OAuthClientInformationFull(
                client_id=reg["client_id"],
                client_secret=reg.get("client_secret"),
                redirect_uris=[_redirect_uri()],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="client_secret_post",
            )
            oauth_client = client_info.model_dump(mode="json", exclude_none=True)
            if oauth_client.get("client_secret"):
                oauth_client["client_secret"] = encrypt_value(
                    oauth_client["client_secret"]
                )
            oauth_client["source"] = "dcr"
            server.oauth_client = oauth_client
        db.commit()

    verifier, challenge = new_pkce()
    state = encode_state(user_id=user.id, mcpserver_id=server_id)

    row = (
        db.query(MCPUserOAuthToken)
        .filter_by(user_id=user.id, mcpserver_id=server_id)
        .one_or_none()
    )
    if row is None:
        row = MCPUserOAuthToken(user_id=user.id, mcpserver_id=server_id)
        db.add(row)
    row.status = "pending"
    row.pkce_verifier = encrypt_value(verifier)
    row.state = state
    db.commit()

    auth_url = build_authorization_url(
        as_meta,
        client_id=oauth_client["client_id"],
        redirect_uri=_redirect_uri(),
        code_challenge=challenge,
        state=state,
    )
    return {"authorization_url": auth_url}


@router.get("/oauth/callback")
async def callback(
    code: str = Query(...),
    state: str = Query(...),
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
    if row is None or not row.pkce_verifier:
        raise HTTPException(status_code=400, detail="No pending authorization")

    server = db.query(MCPServer).filter_by(id=server_id).one()
    oc = server.oauth_client or {}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            tokens = await exchange_code_for_tokens(
                server.auth_server_metadata,
                client_id=oc["client_id"],
                client_secret=(
                    decrypt_value(oc["client_secret"])
                    if oc.get("client_secret")
                    else None
                ),
                code=code,
                code_verifier=decrypt_value(row.pkce_verifier),
                redirect_uri=_redirect_uri(),
                client=client,
            )
    except httpx.HTTPError:
        logger.warning("OAuth token exchange failed for server %s", server_id)
        row.status = "error"
        db.commit()
        return HTMLResponse(
            "<html><body><p>Authorization failed. Please try connecting again.</p>"
            "</body></html>",
            status_code=400,
        )

    row.access_token = encrypt_value(tokens["access_token"])
    row.refresh_token = (
        encrypt_value(tokens["refresh_token"]) if tokens.get("refresh_token") else None
    )
    row.token_type = tokens.get("token_type", "Bearer")
    row.scope = tokens.get("scope")
    if tokens.get("expires_in"):
        row.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=int(tokens["expires_in"])
        )
    row.status = "connected"
    row.pkce_verifier = None
    row.state = None
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
    _get_authorized_server(db, user.id, server_id)
    row = (
        db.query(MCPUserOAuthToken)
        .filter_by(user_id=user.id, mcpserver_id=server_id)
        .one_or_none()
    )
    return {"status": row.status if row else "not_connected"}


@router.delete("/{server_id}/connection")
async def disconnect(
    server_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _get_authorized_server(db, user.id, server_id)
    row = (
        db.query(MCPUserOAuthToken)
        .filter_by(user_id=user.id, mcpserver_id=server_id)
        .one_or_none()
    )
    if row:
        db.delete(row)
        db.commit()
    return {"status": "not_connected"}
