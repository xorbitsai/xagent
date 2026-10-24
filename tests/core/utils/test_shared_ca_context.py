from __future__ import annotations

import asyncio
import datetime
import os
import socket
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import anyio
import anyio.to_thread
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from xagent.core.utils import public_http_transport, security
from xagent.core.utils.public_http_transport import (
    PublicHttpTransport,
    PublicSyncHttpTransport,
)

HOST = "shared-ca.test"


class _LoopbackPolicy:
    """Resolves the test hostname to the local TLS server."""

    async def resolve(self, url: str) -> list[str]:
        return ["127.0.0.1"]

    def resolve_sync(self, url: str) -> list[str]:
        return ["127.0.0.1"]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: Any) -> None:
        pass


class _QueuedServer(ThreadingHTTPServer):
    # Read by server_activate() while the server is constructed.
    request_queue_size = 64


@pytest.fixture
def tls_server(tmp_path, monkeypatch):
    """A local HTTPS server whose self-signed cert is the only trusted CA."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HOST)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(HOST)]), False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
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
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    monkeypatch.setattr(security, "_shared_ca_slot", None)
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    server = _QueuedServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.socket = server_context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://{HOST}:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def _async_client(**timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=PublicHttpTransport(_policy=_LoopbackPolicy()),
        trust_env=False,
        timeout=httpx.Timeout(10.0, **timeout),
    )


def _sync_client() -> httpx.Client:
    return httpx.Client(
        transport=PublicSyncHttpTransport(_policy=_LoopbackPolicy()),
        trust_env=False,
        timeout=10.0,
    )


class _SubclassedContext(ssl.SSLContext):
    pass


@pytest.mark.parametrize(
    ("kind", "expect_thread_hops"), [("shared", False), ("subclass_control", True)]
)
async def test_async_handshake_does_not_use_anyio_thread_pool(
    kind, expect_thread_hops, tls_server, monkeypatch
):
    if kind == "subclass_control":
        # A context of any other class proves the counter sees thread hops.
        control = _SubclassedContext(ssl.PROTOCOL_TLS_CLIENT)
        control.load_verify_locations(cafile=os.environ["SSL_CERT_FILE"])
        monkeypatch.setattr(
            public_http_transport, "shared_ca_bundle_ssl_context", lambda: control
        )
    hops: list[Any] = []
    real_run_sync = anyio.to_thread.run_sync

    async def counting(func, *args, **kwargs):
        hops.append(func)
        return await real_run_sync(func, *args, **kwargs)

    monkeypatch.setattr(anyio.to_thread, "run_sync", counting)

    async with _async_client() as client:
        response = await client.get(tls_server)

    assert response.content == b"ok"
    assert bool(hops) is expect_thread_hops
    if kind == "shared":
        assert type(security.shared_ca_bundle_ssl_context()) is ssl.SSLContext


async def test_async_connect_survives_saturated_anyio_thread_pool(tls_server):
    release = threading.Event()
    started = 0
    lock = threading.Lock()

    def block() -> None:
        nonlocal started
        with lock:
            started += 1
        release.wait(30)

    async with anyio.create_task_group() as group:
        for _ in range(40):
            group.start_soon(anyio.to_thread.run_sync, block)
        try:
            with anyio.fail_after(5):
                while started < 40:
                    await anyio.sleep(0.01)
            async with _async_client(connect=0.5) as client:
                response = await client.get(tls_server)
        finally:
            release.set()

    assert response.content == b"ok"


def test_shared_context_handshakes_from_threads_and_loop(tls_server):
    context = security.shared_ca_bundle_ssl_context()
    published = (
        context.verify_mode,
        context.check_hostname,
        context.options,
        context.minimum_version,
    )
    assert published[:2] == (ssl.CERT_REQUIRED, True)

    def sync_worker() -> int:
        with _sync_client() as client:
            return sum(client.get(tls_server).content == b"ok" for _ in range(20))

    async def async_worker() -> int:
        async with _async_client() as client:
            return sum(
                [(await client.get(tls_server)).content == b"ok" for _ in range(20)]
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(sync_worker) for _ in range(8)]
        async_ok = asyncio.run(async_worker())
        sync_ok = [future.result(timeout=60) for future in futures]

    assert sync_ok == [20] * 8
    assert async_ok == 20
    assert security.shared_ca_bundle_ssl_context() is context
    assert (
        context.verify_mode,
        context.check_hostname,
        context.options,
        context.minimum_version,
    ) == published


def test_two_clients_do_not_resume_each_others_session(tls_server):
    url = tls_server

    def handshake(client: httpx.Client) -> bool:
        with client.stream("GET", url) as response:
            ssl_object = response.extensions["network_stream"].get_extra_info(
                "ssl_object"
            )
            reused = bool(ssl_object.session_reused)
            response.read()
            return reused

    with _sync_client() as first, _sync_client() as second:
        first_reused = handshake(first)
        second_reused = handshake(second)

    assert (first_reused, second_reused) == (False, False)

    # Positive control: the server does resume when a session is handed over.
    port = httpx.URL(url).port
    context = security.shared_ca_bundle_ssl_context()
    with socket.create_connection(("127.0.0.1", port)) as raw:
        with context.wrap_socket(raw, server_hostname=HOST) as tls:
            tls.sendall(f"GET / HTTP/1.1\r\nHost: {HOST}\r\n\r\n".encode())
            tls.recv(4096)
            session = tls.session
    with socket.create_connection(("127.0.0.1", port)) as raw:
        with context.wrap_socket(raw, server_hostname=HOST, session=session) as tls:
            assert tls.session_reused is True
