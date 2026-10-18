from __future__ import annotations

import datetime
import os
import socket
import ssl
from typing import Any
from unittest.mock import Mock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from xagent.core.utils import media_download, security


@pytest.mark.parametrize(
    "host", ["127。0。0。1", "１２７．０．０．１", "127.0.0.1．", "169。254。169。254"]
)
def test_shared_url_shape_checks_the_normalized_ip(host):
    with pytest.raises(security.PrivateNetworkHostError):
        security.validate_public_http_url_shape(f"https://{host}/")


def test_shared_url_shape_preserves_public_internationalized_hosts():
    parsed = security.validate_public_http_url_shape("https://bücher.example/path")
    assert parsed.hostname == "xn--bcher-kva.example"
    assert parsed.path == "/path"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "https://127。0。0。1/",
        "https://１２７．０．０．１/",
        "https://169。254。169。254/",
        "https://example.com:0/",
        "https://user:secret@example.com/",
    ],
)
async def test_sync_and_async_dns_keep_shape_rejections(url, asynchronous, monkeypatch):
    def unexpected_dns(*args):
        pytest.fail("invalid URL reached DNS")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    with pytest.raises(ValueError):
        if asynchronous:
            await security.validate_public_http_url(url)
        else:
            security.validate_public_http_url_sync(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_sync_and_async_dns_use_normalized_hostname(asynchronous, monkeypatch):
    dns = Mock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 8443))
        ]
    )
    monkeypatch.setattr(socket, "getaddrinfo", dns)
    url = "https://bücher.example:8443/path"
    addresses = (
        await security.validate_public_http_url(url)
        if asynchronous
        else security.validate_public_http_url_sync(url)
    )
    assert addresses == ["93.184.216.34"]
    dns.assert_called_once_with(
        "xn--bcher-kva.example", 8443, socket.AF_UNSPEC, socket.SOCK_STREAM
    )


def _generate_test_ca_pem() -> bytes:
    """Build a throwaway self-signed CA cert, purely to prove
    build_ca_bundle_ssl_context() actually loads whatever SSL_CERT_FILE
    points at into the resulting context's trust store -- generated at test
    time (matching this repo's convention, e.g.
    tests/web/api/test_gmail_oidc_real_signature.py) rather than checked in
    as an opaque static blob.
    """

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-root")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


@pytest.fixture
def media_ssl_cache():
    media_download._media_ssl_context.cache_clear()
    yield
    media_download._media_ssl_context.cache_clear()


def test_media_tls_keeps_the_stdlib_default_trust_store(monkeypatch, media_ssl_cache):
    context = Mock(spec=ssl.SSLContext)
    create = Mock(return_value=context)
    monkeypatch.setattr(media_download.ssl, "create_default_context", create)
    assert media_download._media_ssl_context(None, None) is context
    assert media_download._media_ssl_context(None, None) is context
    create.assert_called_once_with()


def test_media_tls_actually_loads_the_configured_ca_file(
    tmp_path, monkeypatch, media_ssl_cache
):
    pem = _generate_test_ca_pem()
    cert_file = tmp_path / "media-ca.pem"
    cert_file.write_bytes(pem)
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_file))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    context = media_download._media_ssl_context(str(cert_file), None)
    assert ssl.PEM_cert_to_DER_cert(pem.decode()) in context.get_ca_certs(
        binary_form=True
    )
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


@pytest.mark.parametrize("variable", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_media_tls_cache_refreshes_when_ca_paths_change(
    variable, monkeypatch, media_ssl_cache
):
    create = Mock(side_effect=[Mock(spec=ssl.SSLContext), Mock(spec=ssl.SSLContext)])
    monkeypatch.setattr(media_download.ssl, "create_default_context", create)
    for value in ("first-path", "second-path"):
        monkeypatch.setenv(variable, value)
        media_download._media_ssl_context(
            os.getenv("SSL_CERT_FILE"), os.getenv("SSL_CERT_DIR")
        )
    assert create.call_count == 2


class TestBuildCaBundleSslContext:
    """build_ca_bundle_ssl_context() backs every transport that pins
    trust_env=False. Transport-level trust_env has no effect on proxy
    handling (see _PinnedA2ATransport/SafeOAuthAsyncHTTPTransport, which get
    their proxy-bypass protection from passing an explicit transport= to
    AsyncClient, not from this flag) -- but it also feeds
    httpx.create_ssl_context() internally, which only honors SSL_CERT_FILE/
    SSL_CERT_DIR when trust_env is true. This helper builds that context
    with trust_env=True explicitly and passes it as verify=, which httpx
    then honors regardless of the transport's own trust_env setting.
    """

    def test_returns_a_concrete_ssl_context(self, monkeypatch) -> None:
        # Clear both env vars so this doesn't nondeterministically fail on
        # an ambient SSL_CERT_FILE/SSL_CERT_DIR left over in the process
        # environment (this test only cares about the return type, not
        # about a specific CA bundle).
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("SSL_CERT_DIR", raising=False)

        context = security.build_ca_bundle_ssl_context()

        assert isinstance(context, ssl.SSLContext)

    def test_builds_context_with_trust_env_true(self, monkeypatch) -> None:
        sentinel_ctx = object()
        captured: dict[str, object] = {}

        def _fake_create_ssl_context(**kwargs: Any) -> object:
            captured.update(kwargs)
            return sentinel_ctx

        monkeypatch.setattr(
            security.httpx, "create_ssl_context", _fake_create_ssl_context
        )

        result = security.build_ca_bundle_ssl_context()

        assert captured == {"trust_env": True}
        assert result is sentinel_ctx

    def test_actually_trusts_the_ca_named_by_ssl_cert_file(
        self, tmp_path, monkeypatch
    ) -> None:
        # End-to-end: prove the env var is actually loaded into the returned
        # context's trust store, not just that create_ssl_context(trust_env=True)
        # was called (a mocked-kwargs test would still pass if the underlying
        # cert never made it into the context).
        cert_file = tmp_path / "test-ca.pem"
        cert_file.write_bytes(_generate_test_ca_pem())
        monkeypatch.setenv("SSL_CERT_FILE", str(cert_file))
        monkeypatch.delenv("SSL_CERT_DIR", raising=False)

        context = security.build_ca_bundle_ssl_context()

        assert any(
            ("commonName", "test-root") in rdn
            for cert in context.get_ca_certs()
            for rdn in cert["subject"]
        )
