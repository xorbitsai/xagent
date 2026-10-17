"""Connection-bound public-only HTTP policy for untrusted request URLs."""

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

    def __init__(self) -> None:
        self._transport = httpx.AsyncHTTPTransport(
            trust_env=False, http2=False, verify=build_ca_bundle_ssl_context()
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
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
        pinned = httpx.Request(
            request.method,
            request.url.copy_with(host=addresses[0]),
            headers=headers,
            stream=request.stream,
            extensions=extensions,
        )
        # Do not replay a streamed/non-idempotent body to another address.
        return await self._transport.handle_async_request(pinned)

    async def aclose(self) -> None:
        await self._transport.aclose()


def public_api_redirect_guard(
    initial_url: str,
) -> Callable[[httpx.Request], Awaitable[None]]:
    """Strip caller credentials on cross-origin redirects; reject body replay."""
    initial = httpx.URL(initial_url)
    origin = (initial.scheme, initial.host, initial.port)

    async def guard(request: httpx.Request) -> None:
        current = (request.url.scheme, request.url.host, request.url.port)
        if current == origin:
            return
        if request.method not in {"GET", "HEAD"}:
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
