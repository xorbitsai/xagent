from __future__ import annotations

import datetime
import os
import socket
import ssl
import time
from typing import Any
from unittest.mock import Mock

import httpx
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


@pytest.fixture
def shared_ca_slot(monkeypatch):
    monkeypatch.setattr(security, "_shared_ca_slot", None)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


@pytest.fixture
def create_ssl_context_calls(monkeypatch):
    calls: list[dict[str, Any]] = []
    real = httpx.create_ssl_context

    def spy(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "create_ssl_context", spy)
    return calls


def _der(pem: bytes) -> bytes:
    return ssl.PEM_cert_to_DER_cert(pem.decode())


def test_shared_context_is_reused_for_same_key(
    shared_ca_slot, create_ssl_context_calls
):
    from xagent.core.utils.public_http_transport import (
        PublicHttpTransport,
        PublicSyncHttpTransport,
    )

    asynchronous = [PublicHttpTransport() for _ in range(100)]
    synchronous = [PublicSyncHttpTransport() for _ in range(100)]

    assert create_ssl_context_calls == [{"trust_env": True}]
    context = security.shared_ca_bundle_ssl_context()
    assert type(context) is ssl.SSLContext
    pools = [item._transport._pool for item in (*asynchronous, *synchronous)]
    assert all(pool._ssl_context is context for pool in pools)
    assert len(create_ssl_context_calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        "file_path",
        "file_content_mtime",
        "file_replaced_inode",
        "dir_path_when_no_file",
        "unset_both",
        "file_rewritten_in_place_preserving_mtime",
        "dir_entry_added",
        "dir_entry_removed",
    ],
)
def test_shared_context_rebuilds_when_ca_source_changes(
    change, tmp_path, monkeypatch, shared_ca_slot, create_ssl_context_calls
):
    old_pem, new_pem = _generate_test_ca_pem(), _generate_test_ca_pem()
    old_der, new_der = _der(old_pem), _der(new_pem)
    # Equal sizes and mtimes leave the path or inode as the only difference.
    size = max(len(old_pem), len(new_pem)) + 8
    old_pem, new_pem = (pem.ljust(size, b"#") for pem in (old_pem, new_pem))
    first = tmp_path / "first.pem"
    first.write_bytes(old_pem)
    os.utime(first, ns=(1_700_000_000_000_000_000,) * 2)
    old_dir, new_dir = tmp_path / "old-dir", tmp_path / "new-dir"
    old_dir.mkdir()
    new_dir.mkdir()

    if change == "dir_entry_removed":
        (old_dir / "entry.0").write_bytes(b"entry")
    uses_dir = change in {
        "dir_path_when_no_file",
        "dir_entry_added",
        "dir_entry_removed",
    }
    if uses_dir:
        monkeypatch.setenv("SSL_CERT_DIR", str(old_dir))
    else:
        monkeypatch.setenv("SSL_CERT_FILE", str(first))
    before = security.shared_ca_bundle_ssl_context()
    assert before is security.shared_ca_bundle_ssl_context()
    assert len(create_ssl_context_calls) == 1

    # Coarse filesystem clocks must not hide the change from the key.
    time.sleep(0.05)
    if change == "file_path":
        second = tmp_path / "second.pem"
        second.write_bytes(new_pem)
        os.utime(second, ns=(1_700_000_000_000_000_000,) * 2)
        monkeypatch.setenv("SSL_CERT_FILE", str(second))
    elif change == "file_content_mtime":
        first.write_bytes(new_pem)
        os.utime(first, ns=(1_700_000_010_000_000_000,) * 2)
    elif change == "file_replaced_inode":
        replacement = tmp_path / "replacement.pem"
        replacement.write_bytes(new_pem)
        os.utime(replacement, ns=(1_700_000_000_000_000_000,) * 2)
        old_stat = first.stat()
        os.replace(replacement, first)
        new_stat = first.stat()
        assert (new_stat.st_size, new_stat.st_mtime_ns) == (
            old_stat.st_size,
            old_stat.st_mtime_ns,
        )
        assert new_stat.st_ino != old_stat.st_ino
    elif change == "file_rewritten_in_place_preserving_mtime":
        old_stat = first.stat()
        first.write_bytes(new_pem)
        os.utime(first, ns=(old_stat.st_mtime_ns,) * 2)
        new_stat = first.stat()
        assert (new_stat.st_ino, new_stat.st_size, new_stat.st_mtime_ns) == (
            old_stat.st_ino,
            old_stat.st_size,
            old_stat.st_mtime_ns,
        )
    elif change == "dir_path_when_no_file":
        monkeypatch.setenv("SSL_CERT_DIR", str(new_dir))
    elif change == "dir_entry_added":
        (old_dir / "entry.0").write_bytes(b"entry")
    elif change == "dir_entry_removed":
        (old_dir / "entry.0").unlink()
    else:
        monkeypatch.delenv("SSL_CERT_FILE")

    after = security.shared_ca_bundle_ssl_context()

    assert after is not before
    assert len(create_ssl_context_calls) == 2
    trusted_before = before.get_ca_certs(binary_form=True)
    trusted_after = after.get_ca_certs(binary_form=True)
    if uses_dir:
        assert trusted_before == trusted_after == []
    elif change == "unset_both":
        assert old_der in trusted_before
        assert old_der not in trusted_after
        assert trusted_after
    else:
        assert old_der in trusted_before
        assert trusted_after == [new_der]
        assert old_der not in trusted_after


def test_shared_context_failure_is_not_cached(
    tmp_path, monkeypatch, shared_ca_slot, create_ssl_context_calls
):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    with pytest.raises(OSError):
        security.shared_ca_bundle_ssl_context()
    assert security._shared_ca_slot is None

    ca = tmp_path / "ca.pem"
    ca.write_bytes(_generate_test_ca_pem())
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    context = security.shared_ca_bundle_ssl_context()

    assert len(create_ssl_context_calls) == 2
    assert context is security.shared_ca_bundle_ssl_context()
    assert len(create_ssl_context_calls) == 2


@pytest.mark.parametrize(
    "operation",
    [
        "alpn_change",
        "load_cert_chain",
        "load_verify_locations",
        "load_default_certs",
        "set_default_verify_paths",
        "set_ciphers",
    ],
)
def test_shared_context_rejects_mutation(operation, shared_ca_slot):
    context = security.shared_ca_bundle_ssl_context()
    operations = {
        "alpn_change": lambda: context.set_alpn_protocols(["h2", "http/1.1"]),
        "load_cert_chain": lambda: context.load_cert_chain("cert.pem", "key.pem"),
        "load_verify_locations": lambda: context.load_verify_locations("ca.pem"),
        "load_default_certs": lambda: context.load_default_certs(),
        "set_default_verify_paths": lambda: context.set_default_verify_paths(),
        "set_ciphers": lambda: context.set_ciphers("DEFAULT"),
    }

    with pytest.raises(
        RuntimeError, match="shared CA context is guarded against mutation"
    ):
        operations[operation]()

    context.set_alpn_protocols(["http/1.1"])
    context.set_alpn_protocols(("http/1.1",))
    assert type(context) is ssl.SSLContext


@pytest.mark.parametrize(
    "case",
    [
        "both_set_dir_change_does_not_rebuild",
        "empty_file_falls_back_to_dir",
        "empty_file_and_unset_dir_falls_back_to_certifi",
    ],
)
def test_shared_context_key_follows_httpx_precedence(
    case, tmp_path, monkeypatch, shared_ca_slot, create_ssl_context_calls
):
    first_dir, second_dir = tmp_path / "first-dir", tmp_path / "second-dir"
    first_dir.mkdir()
    second_dir.mkdir()
    capaths: list[Any] = []
    real_create = ssl.create_default_context

    def spy(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        capaths.append(kwargs.get("capath"))
        return real_create(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", spy)

    if case == "both_set_dir_change_does_not_rebuild":
        ca = tmp_path / "ca.pem"
        ca.write_bytes(_generate_test_ca_pem())
        monkeypatch.setenv("SSL_CERT_FILE", str(ca))
        monkeypatch.setenv("SSL_CERT_DIR", str(first_dir))
        before = security.shared_ca_bundle_ssl_context()
        monkeypatch.setenv("SSL_CERT_DIR", str(second_dir))
        after = security.shared_ca_bundle_ssl_context()
        assert after is before
        assert len(create_ssl_context_calls) == 1
    elif case == "empty_file_falls_back_to_dir":
        monkeypatch.setenv("SSL_CERT_FILE", "")
        monkeypatch.setenv("SSL_CERT_DIR", str(first_dir))
        before = security.shared_ca_bundle_ssl_context()
        monkeypatch.setenv("SSL_CERT_DIR", str(second_dir))
        after = security.shared_ca_bundle_ssl_context()
        assert after is not before
        assert capaths == [str(first_dir), str(second_dir)]
    else:
        before = security.shared_ca_bundle_ssl_context()
        monkeypatch.setenv("SSL_CERT_FILE", "")
        after = security.shared_ca_bundle_ssl_context()
        assert after is before
        assert len(create_ssl_context_calls) == 1
