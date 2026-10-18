"""Real-client regressions for public fetches behind a corporate proxy."""

import asyncio
import socket
import ssl
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from xagent.core.tools.core.web_content import WebContentFetcher, public_fetch_client
from xagent.core.utils.public_download import PublicDownloadError, download_public_input
from xagent.core.utils.public_network_proxy import PublicNetworkEgressProxy
from xagent.core.utils.security import fetch_public_http_bytes

PUBLIC_IP = "93.184.216.34"


@asynccontextmanager
async def upstream(*, tls_context=None):
    seen = []
    writers = set()

    async def serve(reader, writer):
        writers.add(writer)
        try:
            connect = await reader.readuntil(b"\r\n\r\n")
            seen.append(connect)
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            if tls_context is not None:
                await writer.start_tls(tls_context)
            request = await reader.readuntil(b"\r\n\r\n")
            seen.append(request)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nok"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            writers.discard(writer)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", seen
    finally:
        server.close()
        for writer in list(writers):
            writer.close()
        await server.wait_closed()


def fake_dns(monkeypatch, addresses):
    original = socket.getaddrinfo
    calls = []

    def resolve(host, port, *args, **kwargs):
        if host != "public.example":
            return original(host, port, *args, **kwargs)
        calls.append(host)
        address = addresses[min(len(calls) - 1, len(addresses) - 1)]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    return calls


async def test_upstream_gets_checked_literal_ip_and_logical_host(monkeypatch):
    calls = fake_dns(monkeypatch, [PUBLIC_IP])
    async with upstream() as (proxy_url, seen):
        async with public_fetch_client(proxy_url) as client:
            response = await fetch_public_http_bytes(
                client,
                "http://public.example/data",
                max_content_bytes=10,
                timeout=2,
                via_proxy=True,
            )
    assert response.content == b"ok"
    assert len(calls) == 2
    assert seen[0].startswith(f"CONNECT {PUBLIC_IP}:80 HTTP/1.1\r\n".encode())
    assert b"CONNECT public.example" not in seen[0]
    assert b"host: public.example" in seen[1].lower()
    assert b"proxy-authorization" not in seen[1].lower()


async def test_rebinding_after_preflight_never_reaches_upstream(monkeypatch):
    calls = fake_dns(monkeypatch, [PUBLIC_IP, "169.254.169.254"])
    async with upstream() as (proxy_url, seen):
        async with public_fetch_client(proxy_url) as client:
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await fetch_public_http_bytes(
                    client,
                    "http://public.example/data",
                    max_content_bytes=10,
                    timeout=2,
                    via_proxy=True,
                )
        assert seen == []
    assert len(calls) == 2
    assert "403" in str(caught.value)


async def test_gate_closes_on_cancellation(monkeypatch):
    closed = []
    original = PublicNetworkEgressProxy.close

    async def close(gate):
        await original(gate)
        closed.append(gate)

    monkeypatch.setattr(PublicNetworkEgressProxy, "close", close)
    with pytest.raises(asyncio.CancelledError):
        async with public_fetch_client("http://127.0.0.1:1"):
            raise asyncio.CancelledError()
    assert len(closed) == 1
    assert not closed[0]._client_tasks
    assert closed[0]._server is None


@pytest.mark.parametrize(
    "proxy_url", ["socks5://proxy.example", "http://proxy.example:0"]
)
async def test_invalid_proxy_has_no_direct_fallback(proxy_url):
    with pytest.raises(ValueError):
        async with public_fetch_client(proxy_url):
            pytest.fail("Invalid proxy must not yield a directly connected client")


async def test_fetcher_uses_gate_instead_of_plain_proxy_client(monkeypatch):
    fake_dns(monkeypatch, [PUBLIC_IP])
    async with upstream() as (proxy_url, seen):
        result = await WebContentFetcher(proxy_url=proxy_url).fetch(
            "http://public.example/data"
        )
    assert result.success and result.content == "ok"
    assert seen[0].startswith(f"CONNECT {PUBLIC_IP}:80 HTTP/1.1\r\n".encode())


@pytest.mark.parametrize("rebound", [False, True])
async def test_bounded_input_proxy_path_pins_connect_time_dns(monkeypatch, rebound):
    fake_dns(monkeypatch, [PUBLIC_IP, "169.254.169.254"] if rebound else [PUBLIC_IP])
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    async with upstream() as (proxy_url, seen):
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        if rebound:
            with pytest.raises(PublicDownloadError):
                await download_public_input(
                    "http://public.example/data", max_bytes=10, timeout=2
                )
            assert seen == []
        else:
            response = await download_public_input(
                "http://public.example/data", max_bytes=10, timeout=2
            )
            assert response.content == b"ok"
            assert seen[0].startswith(f"CONNECT {PUBLIC_IP}:80 HTTP/1.1\r\n".encode())


async def test_fetch_timeout_is_forwarded_to_gate(monkeypatch):
    from xagent.core.tools.core import web_content

    gates = []

    class Gate:
        def __init__(self, **kwargs):
            gates.append(kwargs)

        async def start(self):
            return "http://127.0.0.1:1"

        async def close(self):
            pass

    monkeypatch.setattr(web_content, "PublicNetworkEgressProxy", Gate)
    async with public_fetch_client("http://proxy.example", timeout=30) as client:
        assert client.timeout.connect == 30
    assert gates == [
        {
            "upstream_proxy_url": "http://proxy.example",
            "connect_timeout": 30,
            "response_timeout": 30,
        }
    ]


async def test_https_gate_preserves_verified_tls_sni_and_logical_host(
    monkeypatch, tmp_path
):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "public.example")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("public.example")]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    names = []
    context.set_servername_callback(lambda sock, name, context: names.append(name))
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    fake_dns(monkeypatch, [PUBLIC_IP])
    async with upstream(tls_context=context) as (proxy_url, seen):
        async with public_fetch_client(proxy_url, timeout=2) as client:
            response = await fetch_public_http_bytes(
                client,
                "https://public.example/data",
                max_content_bytes=10,
                timeout=2,
                via_proxy=True,
            )
    assert response.content == b"ok"
    assert seen[0].startswith(f"CONNECT {PUBLIC_IP}:443 HTTP/1.1\r\n".encode())
    assert names == ["public.example"]
    assert b"host: public.example" in seen[1].lower()
