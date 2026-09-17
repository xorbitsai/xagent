from __future__ import annotations

import datetime
import ssl
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from xagent.core.utils import security


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
