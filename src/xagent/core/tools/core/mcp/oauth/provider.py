"""Build an execution-time OAuthClientProvider bound to a user's DB token store."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Tuple
from urllib.parse import urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from ......config import get_oauth_callback_base_url
from .errors import MCPReauthorizationRequired


class DBBackedOAuthClientProvider(OAuthClientProvider):
    async def _initialize(self) -> None:
        await super()._initialize()
        if self.context.current_tokens:
            self.context.update_token_expiry(self.context.current_tokens)


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


def get_oauth_redirect_uri() -> str:
    base = get_oauth_callback_base_url().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "XAGENT_OAUTH_CALLBACK_BASE_URL must include an http(s) scheme and host"
        )
    return f"{base}/api/mcp/oauth/callback"


def build_oauth_client_metadata() -> OAuthClientMetadata:
    return OAuthClientMetadata(
        client_name="xagent",
        redirect_uris=[AnyUrl(get_oauth_redirect_uri())],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )


def oauth_client_metadata_dict() -> dict[str, Any]:
    return build_oauth_client_metadata().model_dump(mode="json", exclude_none=True)


def build_execution_oauth_provider(
    server_url: str,
    server_name: str,
    mcpserver_id: Optional[int],
    storage: TokenStorage,
) -> OAuthClientProvider:
    redirect, callback = make_reauth_handlers(server_name, mcpserver_id)
    return DBBackedOAuthClientProvider(
        server_url=server_url,
        client_metadata=build_oauth_client_metadata(),
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
    )
