import socket

import httpx
import pytest

from xagent.core.utils.public_http_transport import (
    UnsafePublicHttpTarget,
    create_public_async_client,
    create_public_sync_client,
)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_real_client_follows_safe_redirect_without_re_resolving_at_connect(
    monkeypatch, asynchronous
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    seen = []
    options = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "https://other.example/"})
        return httpx.Response(200, json={})

    def transport(**kwargs):
        options.append(kwargs)
        return httpx.MockTransport(handler)

    monkeypatch.setattr(httpx, "HTTPTransport", transport)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    if asynchronous:
        async with create_public_async_client(follow_redirects=True) as client:
            response = await client.get(
                "https://example.com/", headers={"X-Secret": "key"}
            )
    else:
        with create_public_sync_client(follow_redirects=True) as client:
            response = client.get("https://example.com/", headers={"X-Secret": "key"})
    assert response.status_code == 200
    assert len(seen) == 2
    assert all(r.url.host == "93.184.216.34" for r in seen)
    assert seen[0].headers["Host"] == "example.com"
    assert seen[1].headers["Host"] == "other.example"
    assert "X-Secret" not in seen[1].headers
    assert seen[1].extensions["sni_hostname"] == "other.example"
    assert options[0]["trust_env"] is False
    assert options[0]["http2"] is False
    # Distinct logical hosts can resolve to the same IP. Do not reuse a TLS
    # connection authenticated with the preceding host's SNI/certificate name.
    assert options[0]["limits"].max_keepalive_connections == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_client_rejects_private_target_before_inner_transport(
    monkeypatch, asynchronous
):
    def unexpected(request):
        pytest.fail("private request dispatched")

    monkeypatch.setattr(
        httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(unexpected)
    )
    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(unexpected)
    )
    with pytest.raises(UnsafePublicHttpTarget):
        if asynchronous:
            async with create_public_async_client() as client:
                await client.get("http://192.168.1.10:8080/")
        else:
            with create_public_sync_client() as client:
                client.get("http://192.168.1.10:8080/")
