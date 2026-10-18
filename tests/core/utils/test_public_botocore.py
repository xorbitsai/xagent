import io
import socket

import pytest
from botocore.awsrequest import AWSPreparedRequest
from urllib3.response import HTTPResponse

from xagent.core.utils.public_botocore import PublicBotocoreSession
from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget


@pytest.fixture
def network(monkeypatch):
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: records)
    seen = []

    def urlopen(pool, **kwargs):
        seen.append((pool, kwargs))
        return HTTPResponse(body=io.BytesIO(b"{}"), status=200, preload_content=False)

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    return records, seen


def test_real_aws_transport_pins_ip_and_preserves_signed_host_tls_and_streaming(
    network, monkeypatch
):
    _, seen = network
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    session = PublicBotocoreSession(timeout=2)
    request = AWSPreparedRequest(
        method="POST",
        url="https://bedrock.example:8443/model/converse",
        headers={"Authorization": "Bearer test-key", "Host": "localhost"},
        body=b"{}",
        stream_output=True,
    )
    try:
        response = session.send(request)
        assert response.status_code == 200
        pool, kwargs = seen[0]
        assert pool.host == "93.184.216.34" and pool.port == 8443
        assert pool.assert_hostname == "bedrock.example"
        assert pool.conn_kw["server_hostname"] == "bedrock.example"
        assert kwargs["headers"]["Host"] == "bedrock.example:8443"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        response.raw.close()
    finally:
        session.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://192.168.1.1/",
        "https://example.com:0/",
        "https://user:secret@example.com/",
    ],
)
def test_aws_transport_rejects_private_or_invalid_targets(url, network):
    _, seen = network
    session = PublicBotocoreSession(timeout=2)
    try:
        with pytest.raises(UnsafePublicHttpTarget):
            session.send(
                AWSPreparedRequest(
                    method="POST", url=url, headers={}, body=b"{}", stream_output=False
                )
            )
    finally:
        session.close()
    assert not seen


def test_aws_transport_rejects_mixed_dns(network):
    records, seen = network
    records.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443)))
    session = PublicBotocoreSession(timeout=2)
    try:
        with pytest.raises(UnsafePublicHttpTarget):
            session.send(
                AWSPreparedRequest(
                    method="POST",
                    url="https://example.com/",
                    headers={},
                    body=b"{}",
                    stream_output=False,
                )
            )
    finally:
        session.close()
    assert not seen


def test_public_ipv6_tls_names_and_constructor_options(network):
    records, seen = network
    records[:] = [
        (
            socket.AF_INET6,
            socket.SOCK_STREAM,
            6,
            "",
            ("2606:4700:4700::1111", 443, 0, 0),
        )
    ]
    with_session = PublicBotocoreSession(
        timeout=(2, 5), max_pool_connections=3, socket_options=[], proxies_config={}
    )
    try:
        response = with_session.send(
            AWSPreparedRequest(
                method="GET",
                url="https://[2606:4700:4700::1111]:8443/",
                headers={},
                body=None,
                stream_output=True,
            )
        )
        pool, kwargs = seen[0]
        assert pool.host == "2606:4700:4700::1111"
        assert pool.assert_hostname == "2606:4700:4700::1111"
        assert pool.conn_kw["server_hostname"] == "2606:4700:4700::1111"
        assert kwargs["headers"]["Host"] == "[2606:4700:4700::1111]:8443"
        response.raw.close()
    finally:
        with_session.close()


def test_aws_fallback_only_before_body_send(network, monkeypatch):
    from urllib3.exceptions import NewConnectionError, ProtocolError

    records, seen = network
    records.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)))
    original = __import__("urllib3").connectionpool.HTTPConnectionPool.urlopen
    attempts = []

    def urlopen(pool, **kwargs):
        attempts.append(pool.host)
        if len(attempts) == 1:
            raise NewConnectionError(None, "unreachable")
        return original(pool, **kwargs)

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    session = PublicBotocoreSession(timeout=2)
    request = AWSPreparedRequest(
        method="POST",
        url="https://example.com/",
        headers={},
        body=b"sensitive",
        stream_output=True,
    )
    try:
        session.send(request).raw.close()
        assert attempts == ["93.184.216.34", "1.1.1.1"]
        assert len(seen) == 1
        attempts.clear()

        def failed_write(pool, **kwargs):
            attempts.append(pool.host)
            raise ProtocolError("body may have been sent")

        monkeypatch.setattr(
            "urllib3.connectionpool.HTTPConnectionPool.urlopen", failed_write
        )
        from botocore.exceptions import ConnectionClosedError

        with pytest.raises(ConnectionClosedError):
            session.send(request)
        assert len(attempts) == 1
    finally:
        session.close()


def test_aws_operator_proxy_keeps_public_target_policy(network):
    _, seen = network
    session = PublicBotocoreSession(
        timeout=2, proxy_url="http://operator-proxy.example:3128"
    )
    try:
        response = session.send(
            AWSPreparedRequest(
                method="POST",
                url="https://bedrock.example/",
                headers={},
                body=b"{}",
                stream_output=True,
            )
        )
        response.raw.close()
        assert seen[0][0].proxy.host == "operator-proxy.example"
        assert seen[0][0].host == "bedrock.example"
        with pytest.raises(UnsafePublicHttpTarget):
            session.send(
                AWSPreparedRequest(
                    method="POST",
                    url="http://169.254.169.254/",
                    headers={},
                    body=b"{}",
                    stream_output=True,
                )
            )
        assert len(seen) == 1
    finally:
        session.close()
