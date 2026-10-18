"""Connection-bound public-only HTTP policy for untrusted request URLs."""

import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from .security import (
    build_ca_bundle_ssl_context,
    validate_public_http_url,
    validate_public_http_url_sync,
)


class UnsafePublicHttpTarget(ValueError):
    """A policy rejection, not a transient failure that should be retried."""


def _pin_request(request: httpx.Request, addresses: list[str]) -> httpx.Request:
    if request.method.upper() == "CONNECT":
        raise UnsafePublicHttpTarget("HTTP tunnels are not allowed")
    parsed = urlsplit(str(request.url))
    hostname = parsed.hostname
    if hostname is None:
        raise UnsafePublicHttpTarget("HTTP URL must have a hostname")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    if port <= 0:
        raise UnsafePublicHttpTarget("HTTP URL contains an invalid port")
    literal = f"[{hostname}]" if ":" in hostname else hostname
    host = (
        literal
        if port == (443 if parsed.scheme == "https" else 80)
        else f"{literal}:{port}"
    )
    headers = httpx.Headers(request.headers)
    headers["Host"] = host
    for name in ("Proxy-Authorization", "Proxy-Connection"):
        headers.pop(name, None)
    extensions = dict(request.extensions)
    if parsed.scheme == "https":
        extensions["sni_hostname"] = hostname
    else:
        extensions.pop("sni_hostname", None)
    return httpx.Request(
        request.method,
        request.url.copy_with(host=addresses[0]),
        headers=headers,
        stream=request.stream,
        extensions=extensions,
    )


class PublicHttpTransport(httpx.AsyncBaseTransport):
    """Resolve once per request and connect to exactly the checked public IP.

    HTTPX retains the logical URL for redirect handling, cookie scoping and
    error metadata. Only the inner connection request is rewritten; Host and
    TLS SNI retain the validated logical hostname, never a caller override.
    """

    def __init__(self, proxy_url: str | None = None) -> None:
        self._via_proxy = proxy_url is not None
        ssl_context = build_ca_bundle_ssl_context()
        proxy = (
            httpx.Proxy(proxy_url, ssl_context=ssl_context)
            if proxy_url and httpx.URL(proxy_url).scheme == "https"
            else proxy_url
        )
        self._transport = httpx.AsyncHTTPTransport(
            trust_env=False,
            http2=False,
            verify=ssl_context,
            limits=httpx.Limits(max_keepalive_connections=0),
            proxy=proxy,
        )

    async def _resolve_addresses(self, url: str) -> list[str]:
        return await validate_public_http_url(url)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        if request.method.upper() == "CONNECT":
            raise UnsafePublicHttpTarget("HTTP tunnels are not allowed")
        try:
            addresses = await self._resolve_addresses(str(request.url))
        except ValueError as exc:
            raise UnsafePublicHttpTarget(str(exc)) from exc
        except OSError as exc:
            raise httpx.ConnectError(
                "Public API host could not be resolved", request=request
            ) from exc
        hostname = request.url.host
        assert hostname is not None  # Guaranteed by validate_public_http_url.
        # An explicitly trusted proxy owns connect-time DNS enforcement, as
        # in the existing web fetchers. Direct connections stay IP-pinned.
        targets = [hostname] if self._via_proxy else addresses
        connect_budget = request.extensions.get("timeout", {}).get("connect")
        for index, address in enumerate(targets):
            pinned = _pin_request(request, [address])
            if connect_budget is not None:
                # DNS already consumed part of the whole-attempt budget.
                # Reserve time for remaining IPs rather than allowing the
                # first black-holed address to exhaust the outer deadline.
                remaining = max(0.0, connect_budget - (time.monotonic() - started))
                pinned.extensions["timeout"] = {
                    **request.extensions["timeout"],
                    "connect": remaining / (len(targets) - index),
                }
            try:
                return await self._transport.handle_async_request(pinned)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Connect failures precede sending the API body. Never retry
                # Read/WriteError or failures after an HTTP response instead.
                if index == len(targets) - 1:
                    raise
        raise httpx.ConnectError("Public API host did not resolve", request=request)

    async def aclose(self) -> None:
        await self._transport.aclose()


class PublicSyncHttpTransport(httpx.BaseTransport):
    """Synchronous SDK counterpart with the same actual-connection policy."""

    def __init__(self) -> None:
        self._transport = httpx.HTTPTransport(
            trust_env=False,
            http2=False,
            verify=build_ca_bundle_ssl_context(),
            # As in the async transport, pool keys use the pinned IP rather
            # than the logical TLS host. Never reuse across different names.
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        if request.method.upper() == "CONNECT":
            raise UnsafePublicHttpTarget("HTTP tunnels are not allowed")
        try:
            addresses = validate_public_http_url_sync(str(request.url))
        except ValueError as exc:
            raise UnsafePublicHttpTarget(str(exc)) from exc
        except OSError as exc:
            raise httpx.ConnectError(
                "Public API host could not be resolved", request=request
            ) from exc
        connect_budget = request.extensions.get("timeout", {}).get("connect")
        for index, address in enumerate(addresses):
            pinned = _pin_request(request, [address])
            if connect_budget is not None:
                remaining = max(0.0, connect_budget - (time.monotonic() - started))
                pinned.extensions["timeout"] = {
                    **request.extensions["timeout"],
                    "connect": remaining / (len(addresses) - index),
                }
            try:
                return self._transport.handle_request(pinned)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if index == len(addresses) - 1:
                    raise
        raise httpx.ConnectError("Public API host did not resolve", request=request)

    def close(self) -> None:
        self._transport.close()


def _redirect_guard(request: httpx.Request) -> None:
    origin = _http_origin(request.url)
    initial = request.extensions.setdefault("xagent_public_origin", origin)
    if _same_origin_or_upgrade(initial, origin):
        return
    if (
        request.method not in {"GET", "HEAD"}
        or not isinstance(request.stream, httpx.ByteStream)
        or any(request.stream)
    ):
        raise UnsafePublicHttpTarget(
            "Cross-origin redirects cannot replay API request bodies"
        )
    for name in list(request.headers):
        if name.lower() != "host":
            del request.headers[name]
    request.headers["Accept"] = "application/json"
    request.headers["User-Agent"] = "Xagent-Public-Integration/1.0"


async def _async_redirect_guard(request: httpx.Request) -> None:
    _redirect_guard(request)


def public_http_client_options(
    *, asynchronous: bool, proxy_url: str | None = None
) -> dict[str, Any]:
    """Transport options for SDKs that own and close their HTTPX clients."""
    if proxy_url and not asynchronous:
        raise ValueError("Synchronous public HTTPX clients do not support proxies")
    return {
        "transport": PublicHttpTransport(proxy_url=proxy_url)
        if asynchronous
        else PublicSyncHttpTransport(),
        "trust_env": False,
        "event_hooks": {
            "request": [_async_redirect_guard if asynchronous else _redirect_guard]
        },
        "max_redirects": 5,
    }


def create_public_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """SDK-compatible client; redirects and concurrent calls share no mutable origin."""
    return httpx.AsyncClient(
        **_client_options(asynchronous=True, kwargs=kwargs),
    )


def create_public_sync_client(**kwargs: Any) -> httpx.Client:
    return httpx.Client(
        **_client_options(asynchronous=False, kwargs=kwargs),
    )


def _client_options(*, asynchronous: bool, kwargs: dict[str, Any]) -> dict[str, Any]:
    # HTTPX proxy/mount routing takes precedence over the default transport.
    # Reject overrides rather than silently disabling connection-bound policy
    # or ignoring TLS settings which belong on our owned inner transport.
    unsupported = {"transport", "mounts", "proxy", "verify", "cert", "limits", "http2"}
    if unsupported.intersection(kwargs) or kwargs.get("trust_env", False):
        raise ValueError(
            "Public HTTP client transport/TLS options cannot be overridden"
        )
    kwargs.pop("trust_env", None)
    max_redirects = kwargs.pop("max_redirects", 5)
    if not isinstance(max_redirects, int) or not 0 <= max_redirects <= 5:
        raise ValueError("Public HTTP clients allow at most five redirects")
    hooks = {
        name: list(items)
        for name, items in (kwargs.pop("event_hooks", None) or {}).items()
    }
    guard = _async_redirect_guard if asynchronous else _redirect_guard
    hooks.setdefault("request", []).append(guard)
    proxy_url = kwargs.pop("proxy_url", None)
    options = public_http_client_options(asynchronous=asynchronous, proxy_url=proxy_url)
    options.update(kwargs, event_hooks=hooks, max_redirects=max_redirects)
    return options


def _http_origin(url: httpx.URL) -> tuple[str, str, int]:
    return url.scheme, url.host, url.port or (443 if url.scheme == "https" else 80)


def _same_origin_or_upgrade(
    initial: tuple[str, str, int], current: tuple[str, str, int]
) -> bool:
    return current == initial or (
        initial == ("http", current[1], 80) and current == ("https", initial[1], 443)
    )


def public_api_redirect_guard(
    initial_url: str,
) -> Callable[[httpx.Request], Awaitable[None]]:
    """Strip caller credentials on cross-origin redirects; reject body replay."""
    initial = httpx.URL(initial_url)
    origin = _http_origin(initial)

    async def guard(request: httpx.Request) -> None:
        current = _http_origin(request.url)
        if _same_origin_or_upgrade(origin, current):
            return
        if (
            request.method not in {"GET", "HEAD"}
            or not isinstance(request.stream, httpx.ByteStream)
            or any(request.stream)
        ):
            raise UnsafePublicHttpTarget(
                "Cross-origin redirects cannot replay API request bodies"
            )
        # Custom credentials may use arbitrary header names. An allowlist is
        # safer than recognizing Authorization/X-API-Key and missing another.
        for name in list(request.headers):
            if name.lower() != "host":
                del request.headers[name]
        # Do not retain caller-supplied values even in ordinary-looking
        # headers: credentials can be placed in any header name.
        request.headers["Accept"] = "application/json"
        request.headers["User-Agent"] = "Xagent-API-Tool/1.0"

    return guard
