"""Security helpers for outbound hosts and sensitive log data."""

import asyncio
import ipaddress
import re
import socket
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


class PrivateNetworkHostError(ValueError):
    """Raised when an outbound host targets a non-public network range."""


def reject_private_network_host(hostname: str) -> None:
    """Reject localhost and non-public literal IP address ranges."""

    normalized = hostname.rstrip(".").lower()
    if normalized in {"localhost", "ip6-localhost"}:
        raise PrivateNetworkHostError("Host must not resolve to a private network.")
    if "%" in normalized:
        normalized = normalized.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise PrivateNetworkHostError("Host must not resolve to a private network.")


async def validate_public_http_url(url: str) -> None:
    """Resolve an HTTP(S) URL and reject every non-public target address."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url must not contain embedded credentials")

    hostname = parsed.hostname
    reject_private_network_host(hostname)
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("url contains an invalid port") from exc

    addresses = await asyncio.to_thread(
        socket.getaddrinfo,
        hostname,
        port,
        socket.AF_UNSPEC,
        socket.SOCK_STREAM,
    )
    if not addresses:
        raise ValueError(f"Host {hostname!r} did not resolve to an address")
    for *_, socket_address in addresses:
        reject_private_network_host(str(socket_address[0]))


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
