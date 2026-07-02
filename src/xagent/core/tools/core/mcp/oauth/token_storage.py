"""DB-backed TokenStorage for the MCP SDK OAuthClientProvider.

Composes SDK storage from two rows:
  - tokens        -> mcp_user_oauth_tokens (per user+server)
  - client_info   -> mcp_servers.oauth_client (shared per server)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from sqlalchemy.orm import Session

from .....utils.encryption import decrypt_value, encrypt_value
from ......web.models.mcp import MCPServer
from ......web.models.mcp_oauth import MCPUserOAuthToken

logger = logging.getLogger(__name__)


class DBTokenStorage:
    """Implements the ``mcp.client.auth.TokenStorage`` protocol against the DB."""

    def __init__(self, user_id: int, mcpserver_id: int, db: Session) -> None:
        self._user_id = user_id
        self._server_id = mcpserver_id
        self._db = db

    def _row(self) -> Optional[MCPUserOAuthToken]:
        return (
            self._db.query(MCPUserOAuthToken)
            .filter_by(user_id=self._user_id, mcpserver_id=self._server_id)
            .one_or_none()
        )

    async def get_tokens(self) -> Optional[OAuthToken]:
        row = self._row()
        if not row or not row.access_token or row.status != "connected":
            return None
        expires_in = None
        if row.expires_at:
            expires_at = row.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            delta = expires_at - datetime.now(timezone.utc)
            expires_in = max(int(delta.total_seconds()), 0)
        return OAuthToken(
            access_token=decrypt_value(row.access_token),
            token_type=row.token_type or "Bearer",
            refresh_token=decrypt_value(row.refresh_token)
            if row.refresh_token
            else None,
            expires_in=expires_in,
            scope=row.scope,
        )

    async def set_tokens(self, tokens: OAuthToken) -> None:
        row = self._row()
        if row is None:
            row = MCPUserOAuthToken(user_id=self._user_id, mcpserver_id=self._server_id)
            self._db.add(row)
        row.access_token = encrypt_value(tokens.access_token)
        row.refresh_token = (
            encrypt_value(tokens.refresh_token) if tokens.refresh_token else None
        )
        row.token_type = tokens.token_type or "Bearer"
        row.scope = tokens.scope
        if tokens.expires_in:
            row.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=int(tokens.expires_in)
            )
        row.status = "connected"
        row.pkce_verifier = None
        row.state = None
        self._db.commit()

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        server = self._db.query(MCPServer).filter_by(id=self._server_id).one_or_none()
        if not server or not getattr(server, "oauth_client", None):
            return None
        data = dict(server.oauth_client)
        if data.get("client_secret"):
            data["client_secret"] = decrypt_value(data["client_secret"])
        try:
            return OAuthClientInformationFull.model_validate(data)
        except Exception as exc:
            logger.debug("stored oauth_client failed to validate: %s", exc)
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        server = self._db.query(MCPServer).filter_by(id=self._server_id).one_or_none()
        if not server:
            return
        data = client_info.model_dump(mode="json", exclude_none=True)
        if data.get("client_secret"):
            data["client_secret"] = encrypt_value(data["client_secret"])
        data.setdefault("source", "dcr")
        server.oauth_client = data
        self._db.commit()
