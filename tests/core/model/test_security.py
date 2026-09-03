"""Tests for shared security helpers."""

import socket
from unittest.mock import patch

import httpx
import pytest

from xagent.core.utils.security import (
    PrivateNetworkHostError,
    fetch_public_http_bytes,
    redact_sensitive_text,
    redact_url_credentials_for_logging,
    reject_private_network_host,
    validate_public_http_url,
)


def test_reject_private_network_host_rejects_cgnat_range() -> None:
    with pytest.raises(PrivateNetworkHostError):
        reject_private_network_host("100.64.0.1")


@pytest.mark.asyncio
async def test_validate_public_http_url_rejects_private_dns_result() -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    with patch("socket.getaddrinfo", return_value=resolved):
        with pytest.raises(PrivateNetworkHostError):
            await validate_public_http_url("https://public.example/logo.png")


@pytest.mark.asyncio
async def test_validate_public_http_url_accepts_only_public_dns_results() -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    with patch("socket.getaddrinfo", return_value=resolved):
        assert await validate_public_http_url("https://public.example/logo.png") == [
            "93.184.216.34"
        ]


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_pins_validated_ip() -> None:
    """The connection must use the IP resolved at validation time, not a
    second, independently-resolved IP — this is what closes the DNS
    rebinding / TOCTOU window."""

    getaddrinfo_calls: list[tuple] = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        getaddrinfo_calls.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["host_header"] = request.headers.get("host")
        captured["sni_hostname"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        async with httpx.AsyncClient(transport=transport) as client:
            response = await fetch_public_http_bytes(
                client,
                "https://rebind.example/x",
                max_content_bytes=1024,
                timeout=5,
            )

    assert response.content == b"ok"
    assert captured["host"] == "93.184.216.34"
    assert captured["host_header"] == "rebind.example"
    assert captured["sni_hostname"] == "rebind.example"
    assert len(getaddrinfo_calls) == 1


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_via_proxy_does_not_rewrite_sni_to_ip() -> None:
    """When routed through an HTTP CONNECT proxy, the request must keep the
    original hostname as the connect target and TLS SNI. httpcore's CONNECT
    tunnel path derives SNI from the request's remote origin and ignores the
    ``sni_hostname`` extension entirely, so pinning the URL to a bare IP (as
    the direct-connection path does) sends the IP as SNI and breaks
    SNI-strict HTTPS servers behind the proxy."""

    getaddrinfo_calls: list[tuple] = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        getaddrinfo_calls.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["host_header"] = request.headers.get("host")
        captured["sni_hostname"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        async with httpx.AsyncClient(transport=transport) as client:
            response = await fetch_public_http_bytes(
                client,
                "https://rebind.example/x",
                max_content_bytes=1024,
                timeout=5,
                via_proxy=True,
            )

    assert response.content == b"ok"
    # The request must still target the original hostname (not the pinned
    # IP) so a CONNECT proxy performs its own DNS resolution and TLS SNI
    # matches the real origin.
    assert captured["host"] == "rebind.example"
    assert captured["sni_hostname"] is None
    # DNS validation still runs up front to reject private-network targets
    # before the request is ever dispatched to the proxy.
    assert len(getaddrinfo_calls) == 1


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_via_proxy_still_rejects_private_dns_result() -> (
    None
):
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            "private-network target must be rejected before connecting"
        )

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", return_value=resolved):
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(PrivateNetworkHostError):
                await fetch_public_http_bytes(
                    client,
                    "https://rebind.example/x",
                    max_content_bytes=1024,
                    timeout=5,
                    via_proxy=True,
                )


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_revalidates_redirect_target() -> None:
    """Each redirect hop must be re-validated — a redirect to an internal
    host must be rejected even though the initial hop was public."""

    resolutions = iter(
        [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
        ]
    )

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return next(resolutions)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(
                302, headers={"location": "https://internal.example/"}
            )
        raise AssertionError("second hop must be rejected before connecting")

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(PrivateNetworkHostError):
                await fetch_public_http_bytes(
                    client,
                    "https://rebind.example/x",
                    max_content_bytes=1024,
                    timeout=5,
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [300, 305])
async def test_fetch_rejects_unhandled_3xx(status_code: int) -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", return_value=resolved):
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await fetch_public_http_bytes(
                    client,
                    "https://public.example/x",
                    max_content_bytes=1024,
                    timeout=5,
                )


def test_redact_url_credentials_for_logging_masks_sensitive_query_values() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySecret&v=1"
    redacted = redact_url_credentials_for_logging(url)

    assert "AIzaSySecret" not in redacted
    assert "key=%2A%2A%2A" in redacted
    assert "v=1" in redacted


def test_redact_url_credentials_for_logging_masks_embedded_userinfo() -> None:
    # The query string isn't the only -- or even the most common -- place a
    # URL carries a credential; a proxy URL's "user:pass@host" needs the
    # same treatment, or it passes through this function unchanged.
    url = "https://alice:s3cret-pass@proxy.internal:8080/"
    redacted = redact_url_credentials_for_logging(url)

    assert "alice" not in redacted
    assert "s3cret-pass" not in redacted
    assert redacted == "https://***@proxy.internal:8080/"


def test_redact_url_credentials_for_logging_preserves_host_casing() -> None:
    # A URL with no userinfo keeps its original host casing (urlunsplit
    # just passes netloc through); the userinfo-redaction branch must not
    # be the only place that silently lowercases it.
    url = "https://Alice:s3cret-pass@Proxy-Host.Internal:8080/"
    redacted = redact_url_credentials_for_logging(url)

    assert redacted == "https://***@Proxy-Host.Internal:8080/"


def test_redact_url_credentials_for_logging_preserves_ipv6_brackets() -> None:
    url = "https://alice:s3cret-pass@[2001:db8::1]:443/v1"
    redacted = redact_url_credentials_for_logging(url)

    assert "s3cret-pass" not in redacted
    assert redacted == "https://***@[2001:db8::1]:443/v1"


def test_redact_url_credentials_for_logging_does_not_leak_on_parse_failure() -> None:
    # A malformed URL (unclosed IPv6 bracket) that urlsplit can't parse
    # must not fall back to returning the credential-bearing input
    # unchanged -- that would be worse than not attempting redaction at
    # all, since it looks sanitized but isn't.
    url = "https://alice:s3cret-pass@[::1/"
    redacted = redact_url_credentials_for_logging(url)

    assert "s3cret-pass" not in redacted


def test_redact_sensitive_text_does_not_leak_malformed_proxy_url() -> None:
    text = "Unable to connect to proxy https://alice:s3cret-pass@[::1/"
    redacted = redact_sensitive_text(text)

    assert "s3cret-pass" not in redacted


def test_redact_sensitive_text_masks_embedded_proxy_userinfo() -> None:
    text = "Unable to connect to proxy https://alice:s3cret-pass@proxy.internal:8080/"
    redacted = redact_sensitive_text(text)

    assert "s3cret-pass" not in redacted


def test_redact_sensitive_text_masks_bearer_and_header_keys() -> None:
    text = (
        "Authorization: Bearer sk-secret-value "
        "x-goog-api-key: AIzaSyHeaderSecret "
        "url=https://example.com/path?api_key=my_api_key"
    )
    redacted = redact_sensitive_text(text)

    assert "sk-secret-value" not in redacted
    assert "AIzaSyHeaderSecret" not in redacted
    assert "my_api_key" not in redacted


def test_redact_sensitive_text_masks_basic_auth() -> None:
    text = "Authorization: Basic am9objpzZWNyZXQtcGFzcw=="
    redacted = redact_sensitive_text(text)

    assert "am9objpzZWNyZXQtcGFzcw==" not in redacted


def test_redact_sensitive_text_masks_assignment_style_secrets() -> None:
    text = "api_key=sk-super-secret timeout=30"
    redacted = redact_sensitive_text(text)

    assert "sk-super-secret" not in redacted
    assert "api_key=***" in redacted
    assert "timeout=30" in redacted
