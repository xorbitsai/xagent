"""Lightweight deployment-owned proxy selection, without tool dependencies."""

import os

from ...config import get_trusted_egress_proxy_enabled
from .security import PrivateNetworkHostError


def get_proxy_url() -> str | None:
    return (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )


def get_trusted_proxy_url() -> str | None:
    """Require explicit operator trust, not a socket-destination authorization.

    Gated fetchers use public_fetch_client to keep upstream CONNECT targets
    IP-pinned. Callers forwarding logical hosts still require the trusted
    proxy to enforce connect-time DNS/egress policy.
    """
    proxy_url = get_proxy_url()
    if proxy_url and not get_trusted_egress_proxy_enabled():
        raise PrivateNetworkHostError(
            "An HTTP(S) proxy is configured but not marked as trusted for "
            "public-network egress; set XAGENT_TRUSTED_EGRESS_PROXY=1 only "
            "if the proxy itself enforces private-range egress policy."
        )
    return proxy_url
