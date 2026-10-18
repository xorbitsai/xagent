"""Regression contracts for origin pooling and deployment-policy compatibility."""

import ast
import asyncio
import ipaddress
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from yarl import URL

from xagent.core.model import public_http
from xagent.core.model.network_policy import ModelNetworkPolicy
from xagent.core.utils.public_aiohttp import _check_redirect
from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget
from xagent.core.utils.security import PrivateNetworkHostError


@pytest.mark.parametrize("value", ["93.184.216.34", "8.8.8.8"])
def test_dns64_public_addresses_keep_ipv6_for_the_connection(monkeypatch, value):
    nat64 = str(
        ipaddress.IPv6Address(
            int(ipaddress.IPv6Address("64:ff9b::")) + int(ipaddress.IPv4Address(value))
        )
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (nat64, 443, 0, 0))
        ],
    )
    policy = ModelNetworkPolicy()
    assert policy.resolve_sync("https://model.example/v1") == [nat64]
    assert policy.destination(f"https://[{nat64}]/") == (nat64, 443)


@pytest.mark.parametrize(
    "value",
    ["169.254.169.254", "100.100.100.200", "168.63.129.16", "0.0.0.0", "224.0.0.1"],
)
def test_nat64_cannot_hide_metadata_or_unsafe_destinations(value):
    nat64 = str(
        ipaddress.IPv6Address(
            int(ipaddress.IPv6Address("64:ff9b::")) + int(ipaddress.IPv4Address(value))
        )
    )
    with pytest.raises(PrivateNetworkHostError):
        ModelNetworkPolicy().validate_resolved("https://model.example", [nat64])


@pytest.mark.parametrize(
    "value",
    [
        "169.254.169.254",
        "169.254.1.1",
        "fd00:ec2::254",
        "168.63.129.16",
        "100.100.100.200",
        "0.0.0.0",
        "0.1.2.3",
        "224.0.0.1",
        "::",
        "ff02::1",
        "::ffff:169.254.169.254",
    ],
)
def test_hard_denial_wins_even_over_a_covering_approved_network(value):
    policy = ModelNetworkPolicy()
    # Deliberately bypass config validation to isolate hard-deny from CIDR
    # matching: deleting _hard_deny must make this test fail.
    policy.networks = (ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0"))
    with pytest.raises(PrivateNetworkHostError, match="unsafe"):
        policy._validate_addresses(True, [value])


def test_dual_stack_localhost_requires_both_address_families(monkeypatch):
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", "http://localhost:9997")
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "127.0.0.1/32")
    with pytest.raises(PrivateNetworkHostError):
        ModelNetworkPolicy().validate_resolved(
            "http://localhost:9997/v1", ["127.0.0.1", "::1"]
        )
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "127.0.0.1/32,::1/128")
    ModelNetworkPolicy().validate_resolved(
        "http://localhost:9997/v1", ["127.0.0.1", "::1"]
    )


@pytest.mark.parametrize(
    "variable,value",
    [
        ("XAGENT_MODEL_PRIVATE_ENDPOINTS", "http://user:secret@localhost:9997/api"),
        ("XAGENT_MODEL_PRIVATE_CIDRS", "10.0.0.1/8"),
    ],
)
def test_bad_config_names_variable_without_disclosing_value(
    monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    with pytest.raises(PrivateNetworkHostError) as error:
        ModelNetworkPolicy()
    assert variable in str(error.value)
    assert value not in str(error.value)
    assert error.value.__suppress_context__ is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location,count", [("https://other.example/", 0), ("/next", 5)]
)
async def test_aiohttp_redirect_refusals_are_terminal_and_release_response(
    location, count
):
    response = SimpleNamespace(headers={"Location": location}, release=Mock())
    with pytest.raises(UnsafePublicHttpTarget) as error:
        await _check_redirect(
            None,
            SimpleNamespace(public_redirects=count),
            SimpleNamespace(url=URL("https://model.example/v1"), response=response),
            _policy=ModelNetworkPolicy(),
        )
    assert public_http.is_model_network_policy_error(error.value)
    from xagent.core.model.chat.error import retry_on

    assert retry_on(error.value) is False
    response.release.assert_called_once()


@pytest.mark.asyncio
async def test_untrusted_proxy_rejects_search_and_reference_download_without_dispatch(
    monkeypatch,
):
    from xagent.core.tools.core.zhipu_web_search import ZhipuWebSearchCore
    from xagent.core.utils.public_download import (
        PublicDownloadError,
        download_public_input,
    )

    monkeypatch.setenv("HTTPS_PROXY", "http://operator-proxy.example:3128")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "false")
    wire = AsyncMock(side_effect=AssertionError("proxy denial reached network"))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", wire)
    with pytest.raises(ValueError, match="not marked as trusted"):
        await ZhipuWebSearchCore(api_key="key").search("query")
    with pytest.raises(PublicDownloadError, match="policy"):
        await download_public_input(
            "https://image.example/input.png", max_bytes=1024, timeout=1
        )
    wire.assert_not_awaited()


@pytest.mark.parametrize("with_pythonpath", [False, True])
def test_model_policy_import_does_not_load_web_scraping_dependencies(
    tmp_path, with_pythonpath
):
    # -S intentionally bypasses site/.pth/editable setup. Explicitly pass the
    # tested checkout and the parent's dependency paths rather than relying on
    # a developer PYTHONPATH (which is absent in CI). Use an unrelated cwd too.
    source = str(Path(__file__).resolve().parents[3] / "src")
    environment = os.environ.copy()
    if with_pythonpath:
        environment["PYTHONPATH"] = str(tmp_path / "unrelated")
    else:
        environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
import sys, json
sys.path[:] = json.loads(sys.argv[1])
import importlib.abc
class BlockTools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"html2text", "bs4", "xagent.core.tools.core.web_content"}:
            raise ImportError("web scraping dependency must not load", name=fullname)
sys.meta_path.insert(0, BlockTools())
from xagent.core.model.public_http import create_public_model_sync_client
with create_public_model_sync_client():
    pass
""",
            json.dumps([source, *[path for path in sys.path if path]]),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _bare_network_calls(source):
    tree = ast.parse(source)
    libraries = {"httpx", "aiohttp", "requests"}
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in libraries:
                    names[alias.asname or alias.name] = alias.name
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module in libraries
        ):
            for alias in node.names:
                names[alias.asname or alias.name] = node.module + "." + alias.name
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        parts = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name) and function.id in names:
            target = ".".join([names[function.id], *reversed(parts)])
            if target in {
                "httpx.Client",
                "httpx.AsyncClient",
                "aiohttp.ClientSession",
                "requests.Session",
                "requests.request",
                "requests.get",
                "requests.post",
                "requests.put",
                "requests.patch",
                "requests.delete",
                "requests.head",
                "requests.options",
                "requests.session",
            }:
                calls.append((node.lineno, target))
    return calls


def test_inventory_catches_aliased_and_direct_bare_network_clients():
    assert (
        len(
            _bare_network_calls(
                "import httpx as h\nh.AsyncClient()\nfrom requests import get as fetch\nfetch('url')"
            )
        )
        == 2
    )
    root = Path(__file__).resolve().parents[3] / "src/xagent/core/model"
    found = {
        str(path.relative_to(root)): calls
        for path in root.rglob("*.py")
        if (calls := _bare_network_calls(path.read_text()))
    }
    assert found == {}, f"Model code must use guarded factories/facades: {found}"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_tls_pooling_is_origin_isolated_and_rechecks_dns(
    monkeypatch, tmp_path, asynchronous
):
    import datetime
    import ssl
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from httpcore._backends.anyio import AnyIOBackend
    from httpcore._backends.sync import SyncBackend

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "model.example")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("model.example"), x509.DNSName("other.example")]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    ca, pem_key = tmp_path / "ca.pem", tmp_path / "key.pem"
    ca.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    pem_key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca, pem_key)
    names, received, sockets, resolutions = [], [], [], []
    context.set_servername_callback(lambda sock, name, ctx: names.append(name))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            received.append((self.headers["Host"], self.client_address))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    port = server.server_address[1]
    answer = ["93.184.216.34"]
    real_dns = socket.getaddrinfo

    def resolve(host, requested_port, *args, **kwargs):
        if host in {"model.example", "other.example"}:
            resolutions.append(host)
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (answer[0], requested_port))
            ]
        return real_dns(host, requested_port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    sync_connect, async_connect = SyncBackend.connect_tcp, AnyIOBackend.connect_tcp

    def connect(backend, host, port, **kwargs):
        assert host in {"93.184.216.34", "1.1.1.1"} and port == server.server_address[1]
        sockets.append(host)
        return sync_connect(backend, "127.0.0.1", port, **kwargs)

    async def aconnect(backend, host, port, **kwargs):
        assert host in {"93.184.216.34", "1.1.1.1"} and port == server.server_address[1]
        sockets.append(host)
        return await async_connect(backend, "127.0.0.1", port, **kwargs)

    monkeypatch.setattr(SyncBackend, "connect_tcp", connect)
    monkeypatch.setattr(AnyIOBackend, "connect_tcp", aconnect)
    base = f"https://model.example:{port}"

    def sync_requests():
        with public_http.create_public_model_sync_client() as client:
            assert client.get(base + "/one").json() == {}
            assert client.get(base + "/two").json() == {}
            assert client.get(f"https://other.example:{port}/").json() == {}
            answer[0] = "1.1.1.1"
            assert client.get(base + "/changed-public-ip").json() == {}
            answer[0] = "10.0.0.1"
            with pytest.raises(UnsafePublicHttpTarget):
                client.get(base + "/rebound")

    try:
        if asynchronous:
            async with public_http.create_public_model_async_client() as client:
                assert (await client.get(base + "/one")).json() == {}
                assert (await client.get(base + "/two")).json() == {}
                assert (await client.get(f"https://other.example:{port}/")).json() == {}
                answer[0] = "1.1.1.1"
                assert (await client.get(base + "/changed-public-ip")).json() == {}
                answer[0] = "10.0.0.1"
                with pytest.raises(UnsafePublicHttpTarget):
                    await client.get(base + "/rebound")
        else:
            await asyncio.to_thread(sync_requests)
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)
    assert len(received) == 4
    assert received[0][1] == received[1][1]
    assert received[2][1] != received[0][1] != received[3][1]
    assert names == ["model.example", "other.example", "model.example"]
    assert sockets == ["93.184.216.34", "93.184.216.34", "1.1.1.1"]
    assert len(resolutions) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_pool_cap_uses_unpooled_fallback_and_closes_owned_transports(
    monkeypatch, asynchronous
):
    options, closed = [], []

    class Wire(httpx.MockTransport):
        def close(self):
            closed.append(self)

        async def aclose(self):
            closed.append(self)

    def wire(**kwargs):
        options.append(kwargs)
        return Wire(lambda req: httpx.Response(200))

    monkeypatch.setattr(httpx, "HTTPTransport", wire)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", wire)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    if asynchronous:
        async with public_http.create_public_model_async_client() as client:
            for index in range(20):
                await client.get(f"https://model{index}.example/")
    else:
        with public_http.create_public_model_sync_client() as client:
            for index in range(20):
                client.get(f"https://model{index}.example/")
    assert len(options) == len(closed) == 17
    assert all(
        option["limits"].max_keepalive_connections == 20 for option in options[:16]
    )
    assert options[-1]["limits"].max_keepalive_connections == 0


def test_sync_concurrent_first_requests_create_only_one_origin_pool(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    options = []

    def wire(**kwargs):
        options.append(kwargs)
        return httpx.MockTransport(lambda req: httpx.Response(200))

    monkeypatch.setattr(httpx, "HTTPTransport", wire)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    with public_http.create_public_model_sync_client() as client:
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(
                executor.map(client.get, ["https://model.example/v1"] * 16)
            )
    assert all(response.status_code == 200 for response in responses)
    assert len(options) == 1
