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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "option",
    ["proxy", "mounts", "transport", "verify", "cert", "limits", "http2", "trust_env"],
)
def test_factory_rejects_options_that_bypass_or_override_owned_transport(
    asynchronous, option
):
    factory = create_public_async_client if asynchronous else create_public_sync_client
    with pytest.raises(ValueError, match="cannot be overridden"):
        factory(**{option: True})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_public_clients_try_all_addresses_with_one_connect_budget(
    monkeypatch, asynchronous
):
    dns_calls = []

    def dns(*args):
        dns_calls.append(args)
        return [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("2606:4700:4700::1111", 443, 0, 0),
            ),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ConnectTimeout("unreachable", request=request)
        return httpx.Response(200)

    monkeypatch.setattr(
        httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    if asynchronous:
        async with create_public_async_client(timeout=2) as client:
            await client.post("https://example.com/", content=b"secret")
    else:
        with create_public_sync_client(timeout=2) as client:
            client.post("https://example.com/", content=b"secret")
    assert [request.url.host for request in seen] == ["2606:4700:4700::1111", "1.1.1.1"]
    assert 0 < seen[0].extensions["timeout"]["connect"] <= 1
    assert len(dns_calls) == 1
    assert all(request.headers["Host"] == "example.com" for request in seen)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_client_does_not_replay_body_after_write_error(monkeypatch, asynchronous):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        ],
    )
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.WriteError("body may have been sent", request=request)

    monkeypatch.setattr(
        httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    with pytest.raises(httpx.WriteError):
        if asynchronous:
            async with create_public_async_client() as client:
                await client.post("https://example.com/", content=b"secret")
        else:
            with create_public_sync_client() as client:
                client.post("https://example.com/", content=b"secret")
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_custom_hooks_are_merged_before_mandatory_redirect_guard(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))],
    )
    seen = []

    def handler(request):
        seen.append(request)
        return (
            httpx.Response(302, headers={"Location": "https://other.example/"})
            if len(seen) == 1
            else httpx.Response(200)
        )

    async def hook(request):
        request.headers["X-Secret"] = "custom-key"

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    async with create_public_async_client(
        event_hooks={"request": [hook]}, follow_redirects=True
    ) as client:
        await client.get("https://example.com/")
    assert seen[0].headers["X-Secret"] == "custom-key"
    assert "X-Secret" not in seen[1].headers


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_same_host_default_port_https_upgrade_retains_credentials(
    monkeypatch, asynchronous
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))],
    )
    seen = []

    def handler(request):
        seen.append(request)
        return (
            httpx.Response(307, headers={"Location": "https://example.com/"})
            if len(seen) == 1
            else httpx.Response(200)
        )

    monkeypatch.setattr(
        httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler)
    )
    if asynchronous:
        async with create_public_async_client(follow_redirects=True) as client:
            await client.post(
                "http://example.com/", content=b"secret", headers={"X-Key": "key"}
            )
    else:
        with create_public_sync_client(follow_redirects=True) as client:
            client.post(
                "http://example.com/", content=b"secret", headers={"X-Key": "key"}
            )
    assert seen[1].headers["X-Key"] == "key"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_server_selected_proxy_stays_on_guarded_transport(
    monkeypatch, asynchronous
):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))],
    )
    options, seen = [], []

    def transport(**kwargs):
        options.append(kwargs)
        return httpx.MockTransport(
            lambda request: seen.append(request) or httpx.Response(200)
        )

    monkeypatch.setattr(httpx, "HTTPTransport", transport)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    factory = create_public_async_client if asynchronous else create_public_sync_client
    client = factory(proxy_url="http://operator-proxy.example:3128")
    try:
        if asynchronous:
            await client.get("https://example.com/")
            with pytest.raises(UnsafePublicHttpTarget):
                await client.get("https://127.0.0.1/")
        else:
            client.get("https://example.com/")
            with pytest.raises(UnsafePublicHttpTarget):
                client.get("https://127.0.0.1/")
        assert options[0]["proxy"] == "http://operator-proxy.example:3128"
        assert seen[0].url.host == "example.com"
        assert seen[0].headers["Host"] == "example.com"
        assert len(seen) == 1
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()
