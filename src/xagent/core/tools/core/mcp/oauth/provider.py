"""Build an execution-time OAuthClientProvider bound to a user's DB token store."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional, Tuple

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl
from sqlalchemy.orm import Session

from ......config import get_oauth_callback_base_url
from .errors import MCPReauthorizationRequired
from .token_storage import DBTokenStorage


def make_reauth_handlers(
    server_name: str, mcpserver_id: Optional[int]
) -> Tuple[
    Callable[[str], Awaitable[None]],
    Callable[[], Awaitable[Tuple[str, Optional[str]]]],
]:
    """Execution-time handlers: there is no interactive auth mid-run, so both
    raise MCPReauthorizationRequired to surface a 'reconnect' signal."""

    async def _redirect(_authorization_url: str) -> None:
        raise MCPReauthorizationRequired(server_name, mcpserver_id)

    async def _callback() -> Tuple[str, Optional[str]]:
        raise MCPReauthorizationRequired(server_name, mcpserver_id)

    return _redirect, _callback


def _client_metadata() -> OAuthClientMetadata:
    base = get_oauth_callback_base_url().rstrip("/")
    return OAuthClientMetadata(
        client_name="xagent",
        redirect_uris=[AnyUrl(f"{base}/api/mcp/oauth/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )


def build_execution_oauth_provider(
    server_url: str,
    server_name: str,
    user_id: int,
    mcpserver_id: int,
    db: Session,
) -> OAuthClientProvider:
    redirect, callback = make_reauth_handlers(server_name, mcpserver_id)
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=_client_metadata(),
        storage=DBTokenStorage(user_id=user_id, mcpserver_id=mcpserver_id, db=db),
        redirect_handler=redirect,
        callback_handler=callback,
    )
