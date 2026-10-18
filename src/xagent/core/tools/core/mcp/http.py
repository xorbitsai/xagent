"""Connection-bound HTTP policy for configured remote MCP servers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from ....utils.public_http_transport import PublicHttpTransport, UnsafePublicHttpTarget
from ..web_content import get_trusted_proxy_url

if TYPE_CHECKING:
    from .sessions import McpHttpClientFactory


def mcp_http_client_factory(
    endpoint_url: str, custom: McpHttpClientFactory | None = None
) -> McpHttpClientFactory:
    """Bind the default factory once; preserve caller-owned custom factories."""
    return custom or partial(create_mcp_http_client, endpoint_url=endpoint_url)


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafePublicHttpTarget("MCP endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafePublicHttpTarget("MCP endpoint must not contain userinfo")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    if port <= 0:
        raise UnsafePublicHttpTarget("MCP endpoint port must be positive")
    # HTTPX sends IDNA/canonical hostnames. Bind the configured endpoint in
    # the same representation so a legitimate Unicode hostname stays usable.
    host = httpx.URL(url).raw_host.decode("ascii")
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}:{port}"


class MCPHttpTransport(PublicHttpTransport):
    """Public-only, with every request bound to this MCP server's origin."""

    def connection_limits(self) -> httpx.Limits:
        return httpx.Limits(max_keepalive_connections=20)

    def __init__(self, endpoint_url: str) -> None:
        self._origin = _origin(endpoint_url)
        super().__init__(proxy_url=get_trusted_proxy_url())

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Bind at HTTPX's public transport entrypoint, not an implementation
        # detail of the shared DNS resolver. A resolver refactor cannot drop it.
        origin = _origin(str(request.url))
        if origin != self._origin:
            raise UnsafePublicHttpTarget(
                "MCP requests must remain on their configured origin"
            )
        return await super().handle_async_request(request)


def create_mcp_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    *,
    endpoint_url: str,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout if timeout is not None else httpx.Timeout(30.0),
        auth=auth,
        transport=MCPHttpTransport(endpoint_url),
        trust_env=False,
        max_redirects=5,
    )
