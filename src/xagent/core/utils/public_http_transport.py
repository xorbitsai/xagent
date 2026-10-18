"""Connection-bound public-only HTTP policy for untrusted request URLs."""

import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from .security import (
    build_ca_bundle_ssl_context,
    validate_public_http_url,
)


class UnsafePublicHttpTarget(ValueError):
    """A policy rejection, not a transient failure that should be retried."""


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

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        if request.method.upper() == "CONNECT":
            raise UnsafePublicHttpTarget("HTTP tunnels are not allowed")
        try:
            addresses = await validate_public_http_url(str(request.url))
        except ValueError as exc:
            raise UnsafePublicHttpTarget(str(exc)) from exc
        except OSError as exc:
            raise httpx.ConnectError(
                "Public API host could not be resolved", request=request
            ) from exc
        parsed = urlsplit(str(request.url))
        hostname = parsed.hostname
        assert hostname is not None  # Guaranteed by validate_public_http_url.
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
        extensions["sni_hostname"] = hostname
        # An explicitly trusted proxy owns connect-time DNS enforcement, as
        # in the existing web fetchers. Direct connections stay IP-pinned.
        targets = [hostname] if self._via_proxy else addresses
        connect_budget = extensions.get("timeout", {}).get("connect")
        for index, address in enumerate(targets):
            attempt_extensions = dict(extensions)
            if connect_budget is not None:
                # DNS already consumed part of the whole-attempt budget.
                # Reserve time for remaining IPs rather than allowing the
                # first black-holed address to exhaust the outer deadline.
                remaining = max(0.0, connect_budget - (time.monotonic() - started))
                attempt_extensions["timeout"] = {
                    **extensions["timeout"],
                    "connect": remaining / (len(targets) - index),
                }
            pinned = httpx.Request(
                request.method,
                request.url.copy_with(host=address),
                headers=headers,
                stream=request.stream,
                extensions=attempt_extensions,
            )
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


def public_api_redirect_guard(
    initial_url: str,
) -> Callable[[httpx.Request], Awaitable[None]]:
    """Strip caller credentials on cross-origin redirects; reject body replay."""
    initial = httpx.URL(initial_url)
    origin = (
        initial.scheme,
        initial.host,
        initial.port or (443 if initial.scheme == "https" else 80),
    )

    async def guard(request: httpx.Request) -> None:
        current = (
            request.url.scheme,
            request.url.host,
            request.url.port or (443 if request.url.scheme == "https" else 80),
        )
        https_upgrade = origin == ("http", request.url.host, 80) and current == (
            "https",
            initial.host,
            443,
        )
        if current == origin or https_upgrade:
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
