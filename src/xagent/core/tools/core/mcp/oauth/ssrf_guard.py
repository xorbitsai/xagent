"""SSRF guard for outbound OAuth discovery/DCR/token-exchange requests.

Discovery, Dynamic Client Registration, and token-exchange all make
server-side HTTP requests to URLs derived from server-controlled or
discovered metadata (the configured ``server_url``, the discovered
``issuer``, ``registration_endpoint``, and ``token_endpoint``). Without
validation, a malicious or compromised MCP server could point these at an
internal address (or redirect DCR/token-exchange to an attacker-controlled
host), letting it capture the authorization code, PKCE verifier, and/or a
registered client_secret.

``assert_public_endpoint`` performs a pre-request DNS check and rejects
hosts that resolve to private/loopback/link-local/reserved/multicast
addresses. This is NOT a fully rebinding-proof guarantee: a malicious DNS
server could in principle return a public IP at check-time and a private IP
at actual-connection-time (TOCTOU/DNS-rebinding). Closing that gap
completely would require a custom httpx transport that validates the IP
actually connected to on every request -- a larger change. This guard closes
the realistic, primary attack vector (a static malicious/internal endpoint
configured in server metadata).
"""

from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urlparse


class UnsafeOAuthEndpointError(ValueError):
    """Raised when an OAuth-related URL resolves to a disallowed address."""


async def assert_public_endpoint(url: str) -> None:
    """Raise ``UnsafeOAuthEndpointError`` if ``url`` is not a safe, public
    HTTP(S) endpoint to make a server-side request to.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeOAuthEndpointError(
            f"unsupported scheme in OAuth endpoint URL: {url}"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeOAuthEndpointError(f"OAuth endpoint URL has no host: {url}")
    loop = asyncio.get_event_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except OSError as exc:
        raise UnsafeOAuthEndpointError(
            f"could not resolve OAuth endpoint host: {host}"
        ) from exc
    for info in infos:
        raw_ip = info[4][0]
        ip = ipaddress.ip_address(raw_ip)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
            or not ip.is_global
        ):
            raise UnsafeOAuthEndpointError(
                f"OAuth endpoint host resolves to a disallowed address: {host} -> {ip}"
            )
