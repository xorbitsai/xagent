"""Security helpers for outbound hosts and sensitive log data."""

import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import (
    SplitResult,
    parse_qsl,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

import httpx

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


@dataclass(frozen=True)
class PublicHttpResponse:
    """Bounded response returned by a public-network-only HTTP fetch."""

    content: bytes
    url: str
    status_code: int
    content_type: str
    encoding: str | None


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
        or not address.is_global
    ):
        raise PrivateNetworkHostError("Host must not resolve to a private network.")


async def validate_public_http_url(url: str) -> list[str]:
    """Resolve an HTTP(S) URL, reject non-public targets, return validated IPs."""

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
    resolved: list[str] = []
    for *_, socket_address in addresses:
        ip = str(socket_address[0])
        reject_private_network_host(ip)
        resolved.append(ip)
    return resolved


def _pin_url_to_ip(parsed: SplitResult, ip: str) -> tuple[str, str, str]:
    """Return (connect_url_on_ip, host_header, sni_hostname) for a validated IP."""

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("url must have a hostname")
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    host_literal = f"[{hostname}]" if ":" in hostname else hostname
    host_header = host_literal if port == default_port else f"{host_literal}:{port}"
    ip_literal = f"[{ip}]" if ":" in ip else ip
    connect_url = urlunsplit(
        (
            parsed.scheme,
            f"{ip_literal}:{port}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )
    return connect_url, host_header, hostname


_MAX_REDIRECTS = 5


async def fetch_public_http_bytes(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_content_bytes: int,
    timeout: float,
    headers: Mapping[str, str] | None = None,
    resource_name: str = "response body",
    require_non_empty: bool = False,
    via_proxy: bool = False,
) -> PublicHttpResponse:
    """Fetch a bounded HTTP body, validating and pinning every redirect hop.

    The IP validated by ``validate_public_http_url`` is the same IP the
    connection is made to (via a URL rewrite plus an explicit ``Host``
    header and TLS SNI override) — this closes the DNS-rebinding /
    TOCTOU window where a second, independent DNS lookup at connect
    time could return a different (private) address.

    When ``via_proxy`` is set, the request goes through an HTTP CONNECT
    proxy rather than connecting directly. httpcore's CONNECT tunnel path
    ignores the ``sni_hostname`` extension and derives TLS SNI from the
    request's remote origin, so rewriting the URL to the pinned IP would
    send that IP as SNI and break SNI-strict servers. The proxy — not this
    process — performs the actual outbound connection and its own DNS
    resolution, so pinning to a client-resolved IP offers no real
    protection there anyway. In this mode we keep the original hostname as
    the connect target and rely on the upfront ``validate_public_http_url``
    check alone to reject private-network targets before dispatch.
    """

    logical_url = url
    display_name = resource_name[:1].upper() + resource_name[1:]
    for redirect_count in range(_MAX_REDIRECTS + 1):
        resolved_ips = await validate_public_http_url(logical_url)
        parsed = urlsplit(logical_url)
        request_headers = dict(headers or {})
        stream_extensions: dict[str, str] = {}
        if via_proxy:
            connect_url = logical_url
        else:
            connect_url, host_header, sni = _pin_url_to_ip(parsed, resolved_ips[0])
            request_headers["Host"] = host_header
            stream_extensions["sni_hostname"] = sni
        async with client.stream(
            "GET",
            connect_url,
            headers=request_headers,
            timeout=timeout,
            follow_redirects=False,
            extensions=stream_extensions,
        ) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ValueError(f"{display_name} redirect has no Location")
                if redirect_count >= _MAX_REDIRECTS:
                    raise ValueError(f"{display_name} exceeded redirect limit")
                logical_url = urljoin(logical_url, location)
                continue
            if 300 <= response.status_code < 400:
                raise ValueError(
                    f"{display_name} returned unsupported redirect status "
                    f"{response.status_code}"
                )
            response.raise_for_status()
            declared_length = response.headers.get("content-length")
            if declared_length is not None:
                try:
                    length = int(declared_length)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {resource_name} content length: {declared_length}"
                    ) from exc
                if length < 0:
                    raise ValueError(
                        f"Invalid {resource_name} content length: {declared_length}"
                    )
                if length > max_content_bytes:
                    raise ValueError(
                        f"{display_name} exceeds maximum size of "
                        f"{max_content_bytes} bytes"
                    )

            chunks: list[bytes] = []
            downloaded = 0
            async for chunk in response.aiter_bytes():
                downloaded += len(chunk)
                if downloaded > max_content_bytes:
                    raise ValueError(
                        f"{display_name} exceeds maximum size of "
                        f"{max_content_bytes} bytes"
                    )
                chunks.append(chunk)
            content = b"".join(chunks)
            if require_non_empty and not content:
                raise ValueError(f"{display_name} response was empty")
            return PublicHttpResponse(
                content=content,
                url=logical_url,
                status_code=response.status_code,
                content_type=response.headers.get("content-type", ""),
                encoding=getattr(response, "encoding", None),
            )

    raise ValueError(f"{display_name} exceeded redirect limit")


def _mask_secret(value: str) -> str:
    """Mask a secret while preserving a short suffix for troubleshooting."""
    if not value:
        return "***"
    tail_len = 4 if len(value) > 8 else 2
    return "***" + value[-tail_len:]


_USERINFO_PREFIX_PATTERN = re.compile(r"://[^/\s@]*@")


def redact_url_credentials_for_logging(url: str) -> str:
    """Redact sensitive query credentials and any embedded userinfo from a
    URL (e.g. a proxy URL's "user:pass@host", the single most common place
    a URL carries a credential -- a query-string-only check would silently
    pass it through unchanged)."""
    if not url:
        return url

    try:
        parsed = urlsplit(url)
    except ValueError:
        # Malformed enough that urlsplit itself can't parse it (e.g. an
        # unclosed IPv6 bracket) -- fall back to a structure-agnostic scan
        # for a "user:pass@" prefix. Returning the input unchanged here
        # would be worse than not redacting at all: it would silently
        # leak a credential through the one code path meant to catch it.
        return _USERINFO_PREFIX_PATTERN.sub("://***@", url)

    has_userinfo = parsed.username is not None or parsed.password is not None
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if not query_items and not has_userinfo:
        return url

    redacted_items: list[tuple[str, str]] = []
    for key, value in query_items:
        if key.lower() in SENSITIVE_QUERY_KEYS and value:
            redacted_items.append((key, _mask_secret(value)))
        else:
            redacted_items.append((key, value))
    redacted_query = urlencode(redacted_items, doseq=True)

    netloc = parsed.netloc
    if has_userinfo:
        # Split the raw netloc on its last "@" rather than rebuilding
        # "host[:port]" from parsed.hostname/.port: those properties
        # lowercase the host and strip IPv6 brackets, which would make
        # this the only place in the URL that silently changes casing or
        # produces an ambiguous bracket-less "host:port"-looking string
        # for an IPv6 literal -- and .port can raise for a malformed tail
        # after "@" anyway, which this sidesteps entirely.
        netloc = "***@" + parsed.netloc.rsplit("@", 1)[-1]

    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, redacted_query, parsed.fragment)
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
