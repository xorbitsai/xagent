"""Packaged SPA policy; kept in parity with frontend/browser-security.mjs."""

import ipaddress
import posixpath
import re
from urllib.parse import urlsplit

from ..config import get_html_preview_origin


def _preview_origins() -> list[str]:
    """Same narrow frame allowlist as frontend/html-preview-config.mjs."""
    configured = get_html_preview_origin()
    if configured == "auto":
        return [
            "https://preview.xagent.co",
            "https://preview-stage.xagent.co",
            "https://preview-dev.xagent.co",
        ]
    try:
        url = urlsplit(configured)
        if url.username or url.password or not url.hostname:
            return []
        raw_host = url.hostname.lower()
        if url.netloc.startswith("[") and ":" not in raw_host:
            # Reject bracketed IPvFuture/hostname spellings; CSP would treat
            # the de-bracketed value as an ordinary host source.
            return []
        if url.scheme != "https" and not (
            url.scheme == "http" and raw_host in {"localhost", "127.0.0.1", "::1"}
        ):
            return []
        if ":" in raw_host:
            # Python 3.11 accepts scoped IPv6 addresses. A scope identifier is
            # not a valid CSP host source and may contain CSP delimiters.
            if not re.fullmatch(r"[0-9a-f:]+", raw_host):
                return []
            host = f"[{ipaddress.IPv6Address(raw_host).compressed}]"
        else:
            host = raw_host.encode("idna").decode("ascii")
            if not re.fullmatch(r"[a-z0-9.-]+", host):
                return []
        port = url.port
        suffix = (
            f":{port}"
            if port is not None and port != (443 if url.scheme == "https" else 80)
            else ""
        )
        return [f"{url.scheme}://{host}{suffix}"]
    except (ValueError, UnicodeError):
        return []


def is_widget_page(path: str) -> bool:
    """Classify the normalized URL path, never a traversal-shaped prefix."""
    path = posixpath.normpath("/" + path.lstrip("/"))
    return path == "/widget" or path.startswith("/widget/")


def page_security_headers(*, embeddable: bool = False) -> dict[str, str]:
    """A compatible baseline, not a strict nonce/hash script policy.

    Generated HTML uses sandboxed srcDoc frames which inherit their parent's
    CSP. Inline/HTTPS scripts remain allowed for those interactive previews;
    removing them requires changing the preview delivery boundary first.
    """
    directives = [
        "default-src 'self'",
        "base-uri 'none'",
        "object-src 'none'",
        "form-action 'self'",
        "script-src 'self' 'unsafe-inline' https:",
        "style-src 'self' 'unsafe-inline' https:",
        "img-src 'self' data: blob: https:",
        "media-src 'self' data: blob: https:",
        "font-src 'self' data: https:",
        "connect-src 'self' https: wss:",
        " ".join(
            ["frame-src 'self' blob: https://docs.google.com", *_preview_origins()]
        ),
        "worker-src 'self' blob:",
    ]
    headers = {
        "Content-Security-Policy": "; ".join(directives),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }
    if not embeddable:
        headers["Content-Security-Policy"] += "; frame-ancestors 'none'"
        headers["X-Frame-Options"] = "DENY"
    return headers
