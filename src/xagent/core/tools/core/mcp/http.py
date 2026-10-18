"""Connection-bound HTTP policy for configured remote MCP servers."""

from urllib.parse import urlsplit

import httpx

from ....utils.public_http_transport import PublicHttpTransport, UnsafePublicHttpTarget
from ..web_content import get_trusted_proxy_url


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

    def __init__(self, endpoint_url: str) -> None:
        self._origin = _origin(endpoint_url)
        super().__init__(proxy_url=get_trusted_proxy_url())

    async def _resolve_addresses(self, url: str) -> list[str]:
        origin = _origin(url)
        if origin != self._origin:
            raise UnsafePublicHttpTarget(
                "MCP requests must remain on their configured origin"
            )
        return await super()._resolve_addresses(url)


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
        follow_redirects=True,
        max_redirects=5,
    )
