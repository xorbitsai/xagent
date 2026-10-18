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


def test_dns_oserror_is_a_requests_connection_error(network):
    dns, seen, _ = network
    dns.side_effect = socket.gaierror("unavailable")
    with pytest.raises(requests.ConnectionError) as error:
        requests.get("https://example.com/")
    assert error.value.request.url == "https://example.com/"
    assert not seen


def test_requests_default_verification_uses_ca_context_without_overriding_explicit_verify(
    network, monkeypatch
):
    import certifi

    from xagent.core.utils.security import build_ca_bundle_ssl_context

    _, seen, _ = network
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    context = build_ca_bundle_ssl_context()
    build = Mock(return_value=context)
    monkeypatch.setattr(requests, "build_ca_bundle_ssl_context", build)
    requests.get("https://example.com/")
    assert seen[0][0].conn_kw["ssl_context"] is context
    build.assert_called_once()
    build.reset_mock()
    requests.get("https://example.com/", verify=False)
    requests.get("https://example.com/", verify=certifi.where())
    build.assert_not_called()


@pytest.mark.parametrize("variable", ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_ca_environment_without_environment_proxy_or_netrc(monkeypatch, variable):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.setenv(variable, "/operator/ca.pem")
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.example:8080")
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    with requests.Session() as session:
        settings = session.merge_environment_settings(
            "https://example.com/", {}, None, None, None
        )
        assert settings["verify"] == "/operator/ca.pem"
        assert not settings["proxies"]
        assert session.trust_env is False
        assert (
            session.merge_environment_settings(
                "https://example.com/", {}, None, False, None
            )["verify"]
            is False
        )
        assert (
            session.merge_environment_settings(
                "https://example.com/", {}, None, "/explicit.pem", None
            )["verify"]
            == "/explicit.pem"
        )


def test_address_fallback_does_not_resolve_twice_or_replay_writes(network, monkeypatch):
    from urllib3.exceptions import MaxRetryError, NewConnectionError, ProtocolError

    dns, seen, _ = network
    dns.return_value.append(
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))
    )
    original = __import__("urllib3").connectionpool.HTTPConnectionPool.urlopen
    attempts = []

    def urlopen(pool, method, url, **kwargs):
        attempts.append((pool.host, kwargs["timeout"].connect_timeout))
        if len(attempts) == 1:
            raise MaxRetryError(pool, url, NewConnectionError(None, "unreachable"))
        return original(pool, method, url, **kwargs)

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    requests.post("https://example.com/", data=b"secret", timeout=(2, 10))
    assert [entry[0] for entry in attempts] == [PUBLIC_IP, "1.1.1.1"]
    assert 0 < attempts[0][1] <= 1 and attempts[1][1] <= 2
    assert dns.call_count == 1
    assert len(seen) == 1
    attempts.clear()

    def failed_write(pool, *args, **kwargs):
        attempts.append(pool.host)
        raise ProtocolError("body may have been sent")

    monkeypatch.setattr(
        "urllib3.connectionpool.HTTPConnectionPool.urlopen", failed_write
    )
    with pytest.raises(requests.ConnectionError):
        requests.post("https://example.com/", data=b"secret", timeout=2)
    assert len(attempts) == 1


def test_only_explicitly_trusted_proxy_is_usable(network, monkeypatch):
    _, seen, _ = network
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    requests.get("https://example.com/")
    assert seen[0][0].proxy.host == "proxy.example"
    assert seen[0][0].host == "example.com"
    with pytest.raises(ValueError, match="untrusted proxy"):
        requests.get(
            "https://example.com/", proxies={"https": "http://other.example:3128"}
        )


def test_upload_chunks_reuse_ca_context_and_connection_pool(network, monkeypatch):
    import ssl

    _, seen, _ = network
    build = Mock(side_effect=lambda: ssl.create_default_context())
    monkeypatch.setattr(requests, "build_ca_bundle_ssl_context", build)
    with requests.Session() as session:
        for chunk in (b"one", b"two", b"three"):
            session.put("https://upload.example/chunks", data=chunk)
    assert build.call_count == 1
    assert len(seen) == 3
    assert seen[0][0] is seen[1][0] is seen[2][0]


@pytest.mark.parametrize("variable", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_ca_context_cache_respects_env_and_client_certificate_isolation(
    network, monkeypatch, variable
):
    import ssl

    adapter = requests._PublicAdapter()
    first, second = ssl.create_default_context(), ssl.create_default_context()
    certificates = [ssl.create_default_context(), ssl.create_default_context()]
    build = Mock(side_effect=[first, second])
    monkeypatch.setattr(requests, "build_ca_bundle_ssl_context", build)
    monkeypatch.setenv(variable, "/operator/first")
    assert adapter._ca_context(None) is first
    assert adapter._ca_context(None) is first
    monkeypatch.setenv(variable, "/operator/second")
    assert adapter._ca_context(None) is second
    assert build.call_count == 2
    build.side_effect = certificates
    assert adapter._ca_context(("client.pem", "key.pem")) is certificates[0]
    assert adapter._ca_context(("client.pem", "key.pem")) is certificates[1]
    # Explicit client-cert contexts never pollute the cached CA-only context.
    assert adapter._ca_context(None) is second


def test_real_adapter_never_shares_client_cert_context_with_ca_cache(network, tmp_path):
    _, seen, _ = network
    cert, key = tmp_path / "client.pem", tmp_path / "key.pem"
    # The fake wire stops before TLS; real pool construction and Requests'
    # certificate-file existence checks are retained.
    cert.write_text("test client certificate")
    key.write_text("test private key placeholder")
    with requests.Session() as session:
        session.get("https://upload.example/chunk")
        session.get("https://upload.example/chunk", cert=(str(cert), str(key)))
        session.get("https://upload.example/chunk", cert=(str(cert), str(key)))
        session.get("https://upload.example/chunk")
    contexts = [pool.conn_kw["ssl_context"] for pool, *_ in seen]
    assert seen[0][0] is seen[3][0]
    assert contexts[0] is contexts[3]
    assert contexts[1] is not contexts[0]
    assert contexts[2] is not contexts[0] and contexts[2] is not contexts[1]
