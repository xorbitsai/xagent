"""Browser policy shared by API middleware and the packaged SPA server."""


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
        "frame-src 'self' blob:",
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
