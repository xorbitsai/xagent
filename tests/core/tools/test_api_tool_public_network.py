"""Exercise the real generic API client and policy transport with fake sockets."""

import asyncio
import socket
import time
from unittest.mock import Mock

import anyio
import httpx
import pytest

from xagent.core.tools.core.api_tool import APIClientCore
from xagent.core.utils.public_http_transport import PublicHttpTransport

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def network(monkeypatch):
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]
    dns = Mock(return_value=records)
    monkeypatch.setattr(socket, "getaddrinfo", dns)
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    seen = []
    replies = []

    async def handler(request):
        seen.append(request)
        response = replies.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    constructor_options = []

    def create(**kwargs):
        constructor_options.append(kwargs)
        return httpx.MockTransport(handler)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", create)
    return dns, seen, replies, constructor_options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://10.0.0.1/",
        "http://localhost/",
        "file:///etc/passwd",
        "https://user:secret@example.com/",
        "https://example.com:0/",
        "http://127.0.0.1.:80/",
    ],
)
async def test_unsafe_destinations_are_not_dispatched(url, network):
    dns, seen, _, _ = network
    if "127.0.0.1." in url:
        dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ]
    result = await APIClientCore().call_api(url)
    assert result["success"] is False
    assert result["status_code"] == 0
    assert seen == []
    assert ("Invalid URL" if url.startswith("file:") else "network policy") in result[
        "error"
    ]


@pytest.mark.asyncio
async def test_dns_mixed_answers_are_rejected_before_any_connection(network):
    dns, seen, _, _ = network
    dns.return_value += [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 443))
    ]
    result = await APIClientCore().call_api("https://example.com/")
    assert result["status_code"] == 0
    assert "network policy" in result["error"]
    assert dns.call_count == 1
    assert seen == []


@pytest.mark.asyncio
async def test_actual_connection_is_pinned_host_and_tls_sni_are_logical(
    network, monkeypatch
):
    dns, seen, replies, options = network
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    replies.append(httpx.Response(200, json={"ok": True}))
    result = await APIClientCore().call_api(
        "https://example.com:8443/data?q=one",
        headers={
            "Host": "localhost",
            "Proxy-Authorization": "Basic private",
            "X-API-Key": "caller-secret",
        },
    )
    assert result["success"] is True
    assert result["body"] == {"ok": True}
    assert dns.call_count == 1
    request = seen[0]
    assert request.url.host == PUBLIC_IP
    assert request.url.port == 8443
    assert request.headers["host"] == "example.com:8443"
    assert request.extensions["sni_hostname"] == "example.com"
    assert "proxy-authorization" not in request.headers
    assert request.headers["x-api-key"] == "caller-secret"
    assert options[0]["trust_env"] is False
    assert options[0]["http2"] is False
    assert options[0]["limits"].max_keepalive_connections == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location", ["http://127.0.0.1/", "http://169.254.169.254/", "file:///etc/passwd"]
)
async def test_redirect_cannot_reach_private_or_non_http_target(location, network):
    _, seen, replies, _ = network
    replies.append(httpx.Response(302, headers={"Location": location}))
    result = await APIClientCore().call_api("https://example.com/")
    assert result["success"] is False
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_redirect_re_resolves_and_blocks_rebinding(network):
    dns, seen, replies, _ = network
    public = dns.return_value
    dns.side_effect = [
        public,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    replies.append(httpx.Response(302, headers={"Location": "/second"}))
    result = await APIClientCore().call_api("https://example.com/first")
    assert result["success"] is False
    assert len(seen) == 1
    assert dns.call_count == 2


@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_all_custom_credentials(network):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(302, headers={"Location": "https://other.example/finish"}),
            httpx.Response(401),
        ]
    )
    result = await APIClientCore().call_api(
        "https://example.com/",
        headers={
            "Authorization": "Bearer caller",
            "X-Custom-Credential": "secret",
            "Cookie": "session=private",
            "X-API-Key": "private",
            "Accept": "caller-private-media-type",
            "User-Agent": "caller-private-client-id",
        },
    )
    assert result["status_code"] == 401
    assert len(seen) == 2
    for name in ("authorization", "x-custom-credential", "cookie", "x-api-key"):
        assert name not in seen[1].headers
    assert seen[1].headers["accept"] == "application/json"
    assert seen[1].headers["user-agent"] == "Xagent-API-Tool/1.0"
    assert result["final_url"] == "https://other.example/finish"
    assert result["final_request_has_credential"] is False


@pytest.mark.asyncio
async def test_same_origin_redirect_retains_credentials_and_body(network):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(307, headers={"Location": "/second"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    result = await APIClientCore().call_api(
        "https://example.com/first",
        method="POST",
        body={"x": 1},
        auth_type="bearer",
        auth_token="private",
    )
    assert result["success"] is True
    assert len(seen) == 2
    assert seen[1].headers["authorization"] == "Bearer private"
    assert seen[1].method == "POST"
    assert seen[1].content == seen[0].content


@pytest.mark.asyncio
async def test_cross_origin_307_cannot_replay_post_body(network):
    _, seen, replies, _ = network
    replies.append(httpx.Response(307, headers={"Location": "https://other.example/"}))
    result = await APIClientCore().call_api(
        "https://example.com/", method="POST", body={"secret": "private"}
    )
    assert result["success"] is False
    assert "network policy" in result["error"]
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_cross_origin_get_or_head_cannot_replay_a_body(method, network):
    _, seen, replies, _ = network
    replies.append(httpx.Response(307, headers={"Location": "https://other.example/"}))
    result = await APIClientCore().call_api(
        "https://example.com/",
        method=method,
        body="private-body",
        headers={"Content-Length": "0"},
    )
    assert result["success"] is False
    assert "network policy" in result["error"]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_cross_origin_303_can_follow_an_actually_bodyless_get(network):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(303, headers={"Location": "https://other.example/"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    result = await APIClientCore().call_api(
        "https://example.com/", method="POST", body="private-body"
    )
    assert result["success"] is True
    assert len(seen) == 2
    assert seen[1].method == "GET"
    assert seen[1].content == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
async def test_write_response_failure_is_not_retried(method, network):
    _, seen, replies, _ = network
    replies.append(httpx.ReadTimeout("reply lost"))
    result = await APIClientCore().call_api(
        "https://example.com/", method=method, body={"change": "once"}
    )
    assert result["success"] is False
    assert len(seen) == 1
    assert "1 attempts" in result["error"]


@pytest.mark.asyncio
async def test_redirect_connect_failure_does_not_replay_original_post(network):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(307, headers={"Location": "/accepted-write-result"}),
            httpx.ConnectError("second hop failed"),
        ]
    )
    result = await APIClientCore().call_api(
        "https://example.com/", method="POST", body={"change": "once"}
    )
    assert result["success"] is False
    assert len(seen) == 2
    assert "1 attempts" in result["error"]


@pytest.mark.asyncio
async def test_opt_out_of_redirects_never_dispatches_location(network):
    dns, seen, replies, _ = network
    replies.append(httpx.Response(302, headers={"Location": "http://127.0.0.1/"}))
    result = await APIClientCore().call_api(
        "https://example.com/", allow_redirects=False
    )
    assert result["status_code"] == 302
    assert len(seen) == 1
    assert dns.call_count == 1


@pytest.mark.asyncio
async def test_bounded_redirects_do_not_retry_a_post(network):
    _, seen, replies, _ = network
    replies.extend(
        httpx.Response(307, headers={"Location": "/again"}) for _ in range(6)
    )
    result = await APIClientCore().call_api("https://example.com/", method="POST")
    assert result["success"] is False
    assert len(seen) == 6
    assert "1 attempts" in result["error"]


@pytest.mark.asyncio
async def test_streamed_response_still_obeys_size_limit(network):
    _, seen, replies, _ = network
    replies.append(httpx.Response(200, content=b"123456"))
    result = await APIClientCore(max_response_size=5).call_api("https://example.com/")
    assert result["success"] is False
    assert result["status_code"] == 200
    assert "too large" in result["error"]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_redirect_url_cannot_introduce_basic_credentials(network):
    _, seen, replies, _ = network
    replies.append(
        httpx.Response(307, headers={"Location": "https://user:secret@example.com/"})
    )
    result = await APIClientCore().call_api("https://example.com/")
    assert result["success"] is False
    assert "secret" not in result["error"]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_request_policy_has_a_whole_attempt_time_budget(network, monkeypatch):
    dns, seen, _, _ = network

    def blocked_dns(*args, **kwargs):
        time.sleep(0.3)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]

    dns.side_effect = blocked_dns
    started = time.monotonic()
    result = await APIClientCore(default_retry_count=0).call_api(
        "https://example.com/", timeout=0.02
    )
    assert result["success"] is False
    assert seen == []
    assert "timed out" in result["error"]
    assert time.monotonic() - started < 0.25


@pytest.mark.asyncio
async def test_connect_tunnel_is_rejected_before_dns(network):
    dns, seen, _, _ = network
    async with httpx.AsyncClient(
        transport=PublicHttpTransport(), trust_env=False
    ) as client:
        with pytest.raises(ValueError, match="tunnels"):
            await client.request("CONNECT", "https://example.com/")
    assert seen == []
    dns.assert_not_called()


@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_default_port_https_upgrade_keeps_credentials_and_body(method, network):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(307, headers={"Location": "https://example.com/secure"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    result = await APIClientCore().call_api(
        "http://example.com/start",
        method=method,
        body="payload" if method == "POST" else None,
        auth_type="bearer",
        auth_token="secret",
    )
    assert result["success"] is True
    assert len(seen) == 2
    assert seen[1].headers["Authorization"] == "Bearer secret"
    assert seen[1].content == (b"payload" if method == "POST" else b"")


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("unreachable"), httpx.ConnectTimeout("timeout")]
)
async def test_tries_remaining_validated_addresses_before_sending_body(error, network):
    dns, _, _, _ = network
    dns.return_value += [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]
    connected, bodies = [], []

    class ConnectionProbe(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            connected.append(request.url.host)
            if len(connected) == 1:
                raise error
            bodies.append(await request.aread())
            return httpx.Response(200, json={"ok": True})

    transport = PublicHttpTransport()
    transport._transport = ConnectionProbe()
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post("https://example.com/", content=b"one-write")
    assert response.status_code == 200
    assert connected == [PUBLIC_IP, "1.1.1.1"]
    assert bodies == [b"one-write"]


async def test_read_failure_does_not_try_a_second_address(network):
    dns, seen, replies, _ = network
    dns.return_value += [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]
    replies.append(httpx.ReadError("response lost"))
    result = await APIClientCore().call_api(
        "https://example.com/", method="POST", body="write"
    )
    assert not result["success"]
    assert len(seen) == 1


async def test_explicitly_trusted_proxy_retains_logical_target(network, monkeypatch):
    _, seen, replies, options = network
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    replies.append(httpx.Response(200, json={"ok": True}))
    result = await APIClientCore().call_api("https://example.com/")
    assert result["success"]
    assert seen[0].url.host == "example.com"
    assert options[0]["proxy"] == "http://127.0.0.1:3128"


@pytest.mark.parametrize("scheme", ["http", "https"])
async def test_proxy_tls_uses_the_configured_ca_only_for_https(
    scheme, network, monkeypatch
):
    _, _, _, options = network
    context = Mock()
    monkeypatch.setattr(
        "xagent.core.utils.public_http_transport.build_ca_bundle_ssl_context",
        lambda: context,
    )
    transport = PublicHttpTransport(proxy_url=f"{scheme}://proxy.test:3128")
    proxy = options[-1]["proxy"]
    assert options[-1]["verify"] is context
    if scheme == "https":
        assert isinstance(proxy, httpx.Proxy)
        assert proxy.ssl_context is context
    else:
        assert proxy == "http://proxy.test:3128"
    await transport.aclose()


@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_same_host_https_downgrade_never_leaks_credentials_or_body(
    method, network
):
    _, seen, replies, _ = network
    replies.extend(
        [
            httpx.Response(307, headers={"Location": "http://example.com/insecure"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    result = await APIClientCore().call_api(
        "https://example.com/",
        method=method,
        body="private" if method == "POST" else None,
        headers={"Authorization": "Bearer secret", "X-API-Key": "secret"},
    )
    if method == "POST":
        assert not result["success"]
        assert "network policy" in result["error"]
        assert len(seen) == 1
    else:
        assert result["success"]
        assert len(seen) == 2
        assert "authorization" not in seen[1].headers
        assert "x-api-key" not in seen[1].headers


async def test_blackholed_first_address_falls_back_with_real_httpcore(monkeypatch):
    """Use httpcore's actual AnyIO timeout, not an immediate injected exception."""
    connected, bodies = [], []
    original_connect = anyio.connect_tcp

    async def resolve(_):
        return [PUBLIC_IP, "1.1.1.1"]

    async def serve(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = int(
                next(
                    line.split(b":", 1)[1]
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
            )
            bodies.append(await reader.readexactly(length))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async def connect(*args, **kwargs):
        connected.append(kwargs["remote_host"])
        if len(connected) == 1:
            await asyncio.sleep(10)  # Cancelled by httpcore's connect deadline.
        return await original_connect(remote_host="127.0.0.1", remote_port=port)

    monkeypatch.setattr(
        "xagent.core.utils.public_http_transport.validate_public_http_url", resolve
    )
    monkeypatch.setattr(anyio, "connect_tcp", connect)
    async with (
        server,
        httpx.AsyncClient(transport=PublicHttpTransport(), timeout=0.8) as client,
    ):
        async with asyncio.timeout(0.8):
            response = await client.post("http://public.test/", content=b"one-write")
    assert response.text == "OK"
    assert connected == [PUBLIC_IP, "1.1.1.1"]
    assert bodies == [b"one-write"]


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.5/api",
        "http://compose-service/api",
        "https://user:secret@example.com/api",
    ],
)
async def test_configured_custom_api_uses_the_same_public_network_policy(url, network):
    from xagent.core.tools.adapters.vibe.api_tool_adapter import CustomApiTool

    dns, seen, _, _ = network
    if "compose-service" in url:
        dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("172.18.0.5", 80))
        ]
    tool = CustomApiTool(name="fixture", description="fixture", env={}, url=url)
    result = await tool.run_json_async({})
    assert result["success"] is False
    assert "network policy" in result["error"]
    assert "secret" not in result["error"]
    assert seen == []
