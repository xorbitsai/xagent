import io
import socket

import pytest
from botocore.awsrequest import AWSPreparedRequest
from urllib3.response import HTTPResponse

from xagent.core.utils.public_botocore import PublicBotocoreSession


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
    ],
)
def test_aws_transport_rejects_private_or_invalid_targets(url, network):
    _, seen = network
    session = PublicBotocoreSession(timeout=2)
    try:
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
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
