from __future__ import annotations

import ssl
import textwrap
from typing import Any

from xagent.core.utils import security

# A short-lived, self-signed CA cert generated purely for this test -- it is
# never used to verify a real connection, only to prove build_ca_bundle_ssl_context()
# actually loads whatever SSL_CERT_FILE points at into the resulting context's
# trust store.
_TEST_CA_PEM = textwrap.dedent(
    """\
    -----BEGIN CERTIFICATE-----
    MIIBfjCCASOgAwIBAgIUJy8vvflj4iZFGcIOHxvXEKK4WHwwCgYIKoZIzj0EAwIw
    FDESMBAGA1UEAwwJdGVzdC1yb290MB4XDTI2MDkwNDA1MzgxM1oXDTM2MDkwMTA1
    MzgxM1owFDESMBAGA1UEAwwJdGVzdC1yb290MFkwEwYHKoZIzj0CAQYIKoZIzj0D
    AQcDQgAEJBvNMaVhLnQvDmVv+q13d1usFhEVDxM2GATOxbR7L3iQVGn6SpbLlQq+
    7S9t7XjlKhC/vDTaeYCFOW7DF44vLKNTMFEwHQYDVR0OBBYEFFJ8SEgt8Im88lSw
    TBsWnRdX6aFVMB8GA1UdIwQYMBaAFFJ8SEgt8Im88lSwTBsWnRdX6aFVMA8GA1Ud
    EwEB/wQFMAMBAf8wCgYIKoZIzj0EAwIDSQAwRgIhAI9qxcfjyWlf0/umk3rcz684
    QXTQwsywHV9QosmFrxXzAiEA4z1QKZ7W2DUnDWCGlK20Oam6O6USv4nI4S1kQES6
    3mI=
    -----END CERTIFICATE-----
    """
)


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

    def test_returns_a_concrete_ssl_context(self) -> None:
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
        cert_file.write_text(_TEST_CA_PEM)
        monkeypatch.setenv("SSL_CERT_FILE", str(cert_file))
        monkeypatch.delenv("SSL_CERT_DIR", raising=False)

        context = security.build_ca_bundle_ssl_context()

        assert any(
            ("commonName", "test-root") in rdn
            for cert in context.get_ca_certs()
            for rdn in cert["subject"]
        )
