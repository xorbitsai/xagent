from __future__ import annotations

import ssl
from typing import Any

from xagent.core.utils import security


class TestBuildIsolatedSslContext:
    """build_isolated_ssl_context() backs every transport that pins
    trust_env=False (to stop httpx from also falling back to the OS's own
    proxy configuration -- see _PinnedA2ATransport and
    SafeOAuthAsyncHTTPTransport). httpx.create_ssl_context() only honors
    SSL_CERT_FILE/SSL_CERT_DIR when trust_env is true, so the context must
    be built with trust_env=True explicitly and passed as verify=, which
    httpx then honors regardless of the transport's own trust_env setting.
    """

    def test_returns_a_concrete_ssl_context(self) -> None:
        context = security.build_isolated_ssl_context()

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

        result = security.build_isolated_ssl_context()

        assert captured == {"trust_env": True}
        assert result is sentinel_ctx
