"""DB-backed TokenStorage for the MCP SDK OAuthClientProvider.

Composes SDK storage from two rows:
  - tokens        -> mcp_user_oauth_tokens (per user+server)
  - client_info   -> mcp_servers.oauth_client (shared per server)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional, cast

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from sqlalchemy.orm import Session, sessionmaker

from ...core.utils.encryption import decrypt_value, encrypt_value
from ..models.mcp import MCPServer
from ..models.mcp_oauth import MCPUserOAuthToken

logger = logging.getLogger(__name__)


class DBTokenStorage:
    """Implements the ``mcp.client.auth.TokenStorage`` protocol against the DB."""

    def __init__(
        self,
        user_id: int,
        mcpserver_id: int,
        db: Session,
        *,
        create_missing: bool = False,
    ) -> None:
        self._user_id = user_id
        self._server_id = mcpserver_id
        self._db = db
        self._create_missing = create_missing

    def _row(self, db: Session) -> Optional[MCPUserOAuthToken]:
        return (
            db.query(MCPUserOAuthToken)
            .filter_by(user_id=self._user_id, mcpserver_id=self._server_id)
            .one_or_none()
        )

    @contextmanager
    def _session(self, *, write: bool) -> Iterator[Session]:
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._db.get_bind(),
        )
        db = SessionLocal()
        try:
            yield db
            if write:
                db.commit()
        except Exception:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def _read_session(self) -> Iterator[Session]:
        with self._session(write=False) as db:
            yield db

    @contextmanager
    def _write_session(self) -> Iterator[Session]:
        with self._session(write=True) as db:
            yield db

    async def get_tokens(self) -> Optional[OAuthToken]:
        with self._read_session() as db:
            row = self._row(db)
            row_data = cast(Any, row)
            if not row or not row_data.access_token or row_data.status != "connected":
                return None
            expires_in = None
            expires_at = cast(Optional[datetime], row_data.expires_at)
            if expires_at:
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                delta = (expires_at - datetime.now(timezone.utc)).total_seconds()
                if delta <= 0:
                    if not row_data.refresh_token:
                        return None
                    expires_in = -1
                else:
                    expires_in = max(int(delta), 1)
            return OAuthToken(
                access_token=decrypt_value(cast(str, row_data.access_token)),
                token_type=row_data.token_type or "Bearer",
                refresh_token=decrypt_value(cast(str, row_data.refresh_token))
                if row_data.refresh_token
                else None,
                expires_in=expires_in,
                scope=row_data.scope,
            )

    async def _set_tokens(self, tokens: OAuthToken) -> bool:
        with self._write_session() as db:
            row = self._row(db)
            if row is None:
                if not self._create_missing:
                    logger.info(
                        "Skipping MCP OAuth token write for missing row: user=%s server=%s",
                        self._user_id,
                        self._server_id,
                    )
                    return False
                row = MCPUserOAuthToken(
                    user_id=self._user_id, mcpserver_id=self._server_id
                )
                db.add(row)
            row_data = cast(Any, row)
            row_data.access_token = encrypt_value(tokens.access_token)
            if tokens.refresh_token:
                row_data.refresh_token = encrypt_value(tokens.refresh_token)
            # else: authorization server didn't rotate it; keep the existing value
            # (RFC 6749 §6 permits omitting refresh_token when it isn't rotated).
            row_data.token_type = tokens.token_type or "Bearer"
            row_data.scope = tokens.scope
            if tokens.expires_in is not None:
                row_data.expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=int(tokens.expires_in)
                )
            else:
                row_data.expires_at = None
            row_data.status = "connected"
            row_data.pkce_verifier = None
            row_data.state = None
            return True

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await self._set_tokens(tokens)

    async def set_tokens_if_row_exists(self, tokens: OAuthToken) -> bool:
        return await self._set_tokens(tokens)

    async def mark_error(self) -> bool:
        """Mark this (user, server) connection as needing reauthorization.

        Called when the SDK's execution-time auth flow determines the stored
        token cannot be used or refreshed (see MCPReauthorizationRequired), so
        the existing Connect/Reconnect UI reflects it on the next status poll.
        Returns False (no-op) if there is no row for this (user, server) pair.
        """
        with self._write_session() as db:
            row = self._row(db)
            if row is None:
                return False
            row_data = cast(Any, row)
            row_data.status = "error"
            row_data.pkce_verifier = None
            row_data.state = None
            return True

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        with self._read_session() as db:
            server = db.query(MCPServer).filter_by(id=self._server_id).one_or_none()
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
        with self._write_session() as db:
            server = db.query(MCPServer).filter_by(id=self._server_id).one_or_none()
            if not server:
                return
            data = client_info.model_dump(mode="json", exclude_none=True)
            if data.get("client_secret"):
                data["client_secret"] = encrypt_value(data["client_secret"])
            data.setdefault("source", "dcr")
            cast(Any, server).oauth_client = data
