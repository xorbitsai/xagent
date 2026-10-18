"""Public-only aiohttp clients for user-configured model API endpoints."""

from collections.abc import Sequence
from typing import Any
from urllib.parse import urljoin

import aiohttp
from aiohttp.abc import ResolveResult

from .security import (
    build_ca_bundle_ssl_context,
    public_http_destination,
    reject_private_network_host,
)


class PublicTCPConnector(aiohttp.TCPConnector):
    async def _resolve_host(
        self, host: str, port: int, traces: Sequence[Any] | None = None
    ) -> list[ResolveResult]:
        # aiohttp's literal-IP fast path bypasses custom resolvers; enforce
        # policy around the connector itself, including every returned IP.
        reject_private_network_host(host)
        addresses = await super()._resolve_host(host, port, traces)
        if not addresses:
            raise ValueError("Public API host did not resolve to an address")
        for address in addresses:
            reject_private_network_host(address["host"])
        return addresses


class _PublicClientRequest(aiohttp.ClientRequest):
    def __init__(self, method: str, url: Any, **kwargs: Any) -> None:
        public_http_destination(str(url))
        if kwargs.get("proxy") is not None:
            raise ValueError("Public API clients cannot bypass policy via a proxy")
        super().__init__(method, url, **kwargs)
        self.headers["Host"] = url.raw_authority
        self.headers.pop("Proxy-Authorization", None)
        self.headers.pop("Proxy-Connection", None)


async def _check_request(session: Any, context: Any, params: Any) -> None:
    # Run before aiohttp strips URL userinfo into an Authorization header.
    public_http_destination(str(params.url))


async def _check_redirect(session: Any, context: Any, params: Any) -> None:
    location = params.response.headers.get("Location")
    if not location:
        return
    try:
        target = params.url.join(type(params.url)(urljoin(str(params.url), location)))
        public_http_destination(str(target))
        if target.origin() != params.url.origin():
            # Model APIs carry custom credentials and non-replayable bodies.
            raise ValueError("Cross-origin model API redirects are not allowed")
        context.public_redirects = getattr(context, "public_redirects", 0) + 1
        if context.public_redirects > 5:
            raise ValueError("Too many public API redirects")
    except BaseException:
        # aiohttp invokes this hook before its own redirect-response release.
        params.response.release()
        raise


def create_public_aiohttp_session(**kwargs: Any) -> aiohttp.ClientSession:
    trace = aiohttp.TraceConfig()
    trace.on_request_start.append(_check_request)
    trace.on_request_redirect.append(_check_redirect)
    return aiohttp.ClientSession(
        connector=PublicTCPConnector(
            use_dns_cache=False, ssl=build_ca_bundle_ssl_context()
        ),
        request_class=_PublicClientRequest,
        trace_configs=[trace],
        trust_env=False,
        **kwargs,
    )
