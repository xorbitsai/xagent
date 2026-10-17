"""Exercise the real Requests redirect/pool path without opening sockets."""

import io
import socket
from unittest.mock import Mock

import pytest
import requests as original_requests
from urllib3.response import HTTPResponse

from xagent.core.utils import public_requests as requests
from xagent.core.utils.security import PrivateNetworkHostError

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def network(monkeypatch):
    dns = Mock(
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]
    )
    monkeypatch.setattr(socket, "getaddrinfo", dns)
    seen = []
    replies = []

    def urlopen(pool, method, url, **kwargs):
        seen.append((pool, method, url, dict(kwargs["headers"])))
        status, headers = replies.pop(0) if replies else (200, {})
        return HTTPResponse(
            body=io.BytesIO(b"{}"),
            status=status,
            headers=headers,
            preload_content=False,
        )

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    return dns, seen, replies


def test_real_pool_uses_checked_ip_with_original_tls_name_and_host(
    network, monkeypatch
):
    dns, seen, _ = network
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    response = requests.get(
        "https://example.com:8443/a?q=one",
        headers={"Host": "localhost", "Proxy-Authorization": "secret", "X-Key": "key"},
    )
    assert response.json() == {}
    assert response.url == "https://example.com:8443/a?q=one"
    pool, method, path, headers = seen[0]
    assert pool.host == PUBLIC_IP
    assert pool.port == 8443
    assert pool.assert_hostname == "example.com"
    assert pool.conn_kw["server_hostname"] == "example.com"
    assert headers["Host"] == "example.com:8443"
    assert "Proxy-Authorization" not in headers
    assert headers["X-Key"] == "key"
    assert method == "GET" and path == "/a?q=one"
    assert dns.call_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://192.168.1.10:8080/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://localhost/",
        "https://user:password@example.com/",
        "https://example.com:0/",
    ],
)
def test_private_and_invalid_destinations_never_reach_pool(url, network):
    _, seen, _ = network
    with pytest.raises(ValueError):
        requests.get(url)
    assert not seen


def test_mixed_public_private_dns_is_rejected(network):
    dns, seen, _ = network
    dns.return_value.append(
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
    )
    with pytest.raises(requests.RequestException):
        requests.get("https://example.com/")
    assert not seen


def test_redirect_revalidates_dns_without_re_resolution_at_connect(network):
    dns, seen, replies = network
    dns.side_effect = [
        dns.return_value,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    replies.append((302, {"Location": "/next"}))
    with pytest.raises(requests.RequestException):
        requests.get("https://example.com/")
    assert len(seen) == 1
    assert dns.call_count == 2


def test_cross_origin_redirect_strips_custom_headers_and_cookies(network):
    _, seen, replies = network
    replies.extend([(302, {"Location": "https://elsewhere.example/"}), (200, {})])
    requests.get(
        "https://example.com/",
        headers={"X-Secret": "secret", "Cookie": "session=secret"},
    )
    assert len(seen) == 2
    assert set(seen[1][3]) == {"Host", "Accept", "User-Agent"}


def test_cross_origin_307_does_not_replay_body(network):
    _, seen, replies = network
    replies.append((307, {"Location": "https://elsewhere.example/"}))
    with pytest.raises(ValueError, match="bodies"):
        requests.post("https://example.com/", json={"secret": "value"})
    assert len(seen) == 1


@pytest.mark.parametrize(
    "method", ["get", "post", "put", "patch", "delete", "head", "options"]
)
def test_all_convenience_functions_use_policy(method, network):
    _, seen, _ = network
    with pytest.raises(PrivateNetworkHostError):
        getattr(requests, method)("http://127.0.0.1/")
    assert not seen


def test_explicit_proxy_and_connect_rejected(network):
    _, seen, _ = network
    with pytest.raises(ValueError, match="proxy"):
        requests.get("https://example.com/", proxies={"https": "http://proxy.example/"})
    with pytest.raises(ValueError, match="tunnels"):
        requests.request("CONNECT", "https://example.com/")
    assert not seen


def test_stream_close_closes_session_without_global_monkeypatch(network, monkeypatch):
    close = Mock()
    monkeypatch.setattr(requests.Session, "close", close)
    response = requests.get("https://example.com/", stream=True)
    close.assert_not_called()
    response.close()
    close.assert_called_once()
    assert original_requests.Session is not requests.Session
