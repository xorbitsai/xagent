"""Security helpers for redacting sensitive data from logs and errors."""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SENSITIVE_QUERY_KEYS = {
    "api_key",
    "apikey",
    "key",
    "access_token",
    "token",
    "password",
    "secret",
}

URL_PATTERN = re.compile(r"https?://[^\s\"'>]+")
ASSIGNMENT_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|password|secret|key)=([^&\s]+)"
)
BEARER_PATTERN = re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s,;]+)")
HEADER_KEY_PATTERNS = [
    re.compile(r"(?i)(x-goog-api-key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)([^\s,;]+)"),
]


def _mask_secret(value: str) -> str:
    """Mask a secret while preserving a short suffix for troubleshooting."""
    if not value:
        return "***"
    tail_len = 4 if len(value) > 8 else 2
    return "***" + value[-tail_len:]


def redact_url_credentials_for_logging(url: str) -> str:
    """Redact sensitive query credentials from a URL."""
    if not url:
        return url

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if not query_items:
        return url

    redacted_items: list[tuple[str, str]] = []
    for key, value in query_items:
        if key.lower() in SENSITIVE_QUERY_KEYS and value:
            redacted_items.append((key, _mask_secret(value)))
        else:
            redacted_items.append((key, value))

    redacted_query = urlencode(redacted_items, doseq=True)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, redacted_query, parsed.fragment)
    )


def redact_sensitive_text(text: str) -> str:
    """Redact common key/token patterns from arbitrary text."""
    if not text:
        return text

    redacted = URL_PATTERN.sub(
        lambda match: redact_url_credentials_for_logging(match.group(0)),
        text,
    )
    redacted = ASSIGNMENT_SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}={_mask_secret(match.group(2))}",
        redacted,
    )
    redacted = BEARER_PATTERN.sub(
        lambda match: f"{match.group(1)}{_mask_secret(match.group(2))}",
        redacted,
    )
    for pattern in HEADER_KEY_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}{_mask_secret(match.group(2))}",
            redacted,
        )
    return redacted
