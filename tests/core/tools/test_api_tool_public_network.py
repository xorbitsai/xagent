"""Exercise the real generic API client and policy transport with fake sockets."""

import asyncio
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from xagent.core.tools.core.api_tool import APIClientCore
from xagent.core.utils.public_http_transport import PublicHttpTransport

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def network(monkeypatch):
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]
    dns = AsyncMock(return_value=records)

    async def to_thread(function, *args, **kwargs):
        assert function is socket.getaddrinfo
        return await dns(*args, **kwargs)

    monkeypatch.setattr("xagent.core.utils.security.asyncio.to_thread", to_thread)
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


@pytest.mark.asyncio
async def test_dns_mixed_answers_are_rejected_before_any_connection(network):
    dns, seen, _, _ = network
    dns.return_value += [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 443))
    ]
    result = await APIClientCore().call_api("https://example.com/")
    assert result["status_code"] == 0
    assert "network policy" in result["error"]
    assert dns.await_count == 1
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
    assert dns.await_count == 1
    request = seen[0]
    assert request.url.host == PUBLIC_IP
    assert request.url.port == 8443
    assert request.headers["host"] == "example.com:8443"
    assert request.extensions["sni_hostname"] == "example.com"
    assert "proxy-authorization" not in request.headers
    assert request.headers["x-api-key"] == "caller-secret"
    assert options[0]["trust_env"] is False
    assert options[0]["http2"] is False


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
    assert dns.await_count == 2


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
async def test_post_response_failure_is_not_retried(network):
    _, seen, replies, _ = network
    replies.append(httpx.ReadTimeout("reply lost"))
    result = await APIClientCore().call_api(
        "https://example.com/", method="POST", body={"change": "once"}
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
    assert dns.await_count == 1


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

    async def blocked_dns(*args, **kwargs):
        await asyncio.sleep(1)
        return []

    dns.side_effect = blocked_dns
    result = await APIClientCore(default_retry_count=0).call_api(
        "https://example.com/", timeout=0.02
    )
    assert result["success"] is False
    assert seen == []


@pytest.mark.asyncio
async def test_connect_tunnel_is_rejected_before_dns(network):
    dns, seen, _, _ = network
    async with httpx.AsyncClient(
        transport=PublicHttpTransport(), trust_env=False
    ) as client:
        with pytest.raises(ValueError, match="tunnels"):
            await client.request("CONNECT", "https://example.com/")
    assert seen == []
    dns.assert_not_awaited()
