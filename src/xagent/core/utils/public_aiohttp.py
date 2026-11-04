"""Public-only aiohttp clients for user-configured model API endpoints."""

from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any
from urllib.parse import urljoin

import aiohttp
from aiohttp.abc import ResolveResult
from yarl import URL

from .public_http_transport import UnsafePublicHttpTarget
from .public_policy import HttpDestinationPolicy
from .security import (
    build_ca_bundle_ssl_context,
    public_http_destination,
    reject_private_network_host,
    validate_public_http_url,
)


class PublicTCPConnector(aiohttp.TCPConnector):
    def __init__(
        self,
        *,
        proxy_url: str | None = None,
        _policy: HttpDestinationPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        self._policy = _policy
        self._request_url: ContextVar[str | None] = ContextVar(
            "public_aiohttp_request_url", default=None
        )
        self._proxy_destination = (
            (URL(proxy_url).host, URL(proxy_url).port) if proxy_url else None
        )
        super().__init__(**kwargs)

    async def connect(self, req: Any, traces: Any, timeout: Any) -> Any:
        token = self._request_url.set(str(req.url))
        try:
            return await super().connect(req, traces, timeout)
        finally:
            self._request_url.reset(token)

    async def _resolve_host(
        self, host: str, port: int, traces: Sequence[Any] | None = None
    ) -> list[ResolveResult]:
        # aiohttp's literal-IP fast path bypasses custom resolvers; enforce
        # policy around the connector itself, including every returned IP.
        if (host, port) == self._proxy_destination:
            # Only the explicitly selected operator proxy may be private.
            # Origin DNS is preflighted separately and that trusted proxy
            # must enforce public-only destinations at connect time.
            return await super()._resolve_host(host, port, traces)
        if self._policy is None:
            reject_private_network_host(host)
        addresses = await super()._resolve_host(host, port, traces)
        if not addresses:
            raise ValueError("Public API host did not resolve to an address")
        if self._policy is not None:
            url = self._request_url.get()
            if url is None:
                raise ValueError("A validated request URL is required")
            self._policy.validate_resolved(
                url, [address["host"] for address in addresses]
            )
        else:
            for address in addresses:
                reject_private_network_host(address["host"])
        return addresses


class _PublicClientRequest(aiohttp.ClientRequest):
    def __init__(
        self,
        method: str,
        url: Any,
        *,
        trusted_proxy: str | None = None,
        _policy: HttpDestinationPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        (_policy.destination if _policy is not None else public_http_destination)(
            str(url)
        )
        proxy = kwargs.get("proxy")
        if proxy is not None and (
            trusted_proxy is None or URL(proxy) != URL(trusted_proxy)
        ):
            raise ValueError("Public API clients cannot bypass policy via a proxy")
        super().__init__(method, url, **kwargs)
        self.headers["Host"] = url.raw_authority
        self.headers.pop("Proxy-Authorization", None)
        self.headers.pop("Proxy-Connection", None)


async def _check_request(
    session: Any,
    context: Any,
    params: Any,
    *,
    _policy: HttpDestinationPolicy | None = None,
) -> None:
    # Run before aiohttp strips URL userinfo into an Authorization header.
    (_policy.destination if _policy is not None else public_http_destination)(
        str(params.url)
    )


async def _check_redirect(
    session: Any,
    context: Any,
    params: Any,
    *,
    _policy: HttpDestinationPolicy | None = None,
) -> None:
    location = params.response.headers.get("Location")
    if not location:
        return
    try:
        target = params.url.join(type(params.url)(urljoin(str(params.url), location)))
        (_policy.destination if _policy is not None else public_http_destination)(
            str(target)
        )
        if target.origin() != params.url.origin():
            # Model APIs carry custom credentials and non-replayable bodies.
            raise UnsafePublicHttpTarget(
                "Cross-origin model API redirects are not allowed"
            )
        context.public_redirects = getattr(context, "public_redirects", 0) + 1
        if context.public_redirects > 5:
            raise UnsafePublicHttpTarget("Too many public API redirects")
    except BaseException:
        # aiohttp invokes this hook before its own redirect-response release.
        params.response.release()
        raise


def create_public_aiohttp_session(
    *,
    proxy_url: str | None = None,
    _policy: HttpDestinationPolicy | None = None,
    **kwargs: Any,
) -> aiohttp.ClientSession:
    if proxy_url and URL(proxy_url).scheme not in {"http", "https"}:
        raise ValueError("Public aiohttp clients require an HTTP(S) trusted proxy")
    trace = aiohttp.TraceConfig()

    async def check_request(session: Any, context: Any, params: Any) -> None:
        await _check_request(session, context, params, _policy=_policy)
        if proxy_url:
            await (
                _policy.resolve(str(params.url))
                if _policy is not None
                else validate_public_http_url(str(params.url))
            )

    async def check_redirect(session: Any, context: Any, params: Any) -> None:
        await _check_redirect(session, context, params, _policy=_policy)
        location = params.response.headers.get("Location")
        if proxy_url and location:
            try:
                target = urljoin(str(params.url), location)
                await (
                    _policy.resolve(target)
                    if _policy is not None
                    else validate_public_http_url(target)
                )
            except BaseException:
                params.response.release()
                raise

    trace.on_request_start.append(check_request)
    trace.on_request_redirect.append(check_redirect)

    class Request(_PublicClientRequest):
        def __init__(self, method: str, url: Any, **options: Any) -> None:
            super().__init__(
                method, url, trusted_proxy=proxy_url, _policy=_policy, **options
            )

    return aiohttp.ClientSession(
        connector=PublicTCPConnector(
            use_dns_cache=False,
            ssl=build_ca_bundle_ssl_context(),
            proxy_url=proxy_url,
            _policy=_policy,
        ),
        request_class=Request,
        proxy=proxy_url,
        trace_configs=[trace],
        trust_env=False,
        **kwargs,
    )
