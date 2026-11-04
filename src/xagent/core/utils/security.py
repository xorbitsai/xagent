"""Security helpers for outbound hosts and sensitive log data."""

import asyncio
import ipaddress
import os
import re
import socket
import ssl
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, NoReturn
from urllib.parse import (
    SplitResult,
    parse_qsl,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

import httpx
from yarl import URL

SENSITIVE_QUERY_KEYS = {
    "api_key",
    "apikey",
    "key",
    "access_token",
    "token",
    "ticket",
    "password",
    "secret",
}

URL_PATTERN = re.compile(r"(?:https?|wss?)://[^\s\"'>]+")
WEBSOCKET_CREDENTIAL_PATTERN = re.compile(r"xagent-auth-token\.[A-Za-z0-9._~-]+")
# Every ``identifier=`` in the text; ``_is_credential_key`` decides which
# identifiers name a secret. Only a credential's value is consumed
# (``_ASSIGNMENT_VALUE_PATTERN``: up to ``&`` or whitespace). After any other
# identifier the scan resumes right behind its ``=``, so a credential that
# follows it without a space (``host=db;password=...``) is still judged on
# its own. The regex itself stays linear: a single identifier run followed
# by ``=``, anchored on a non-identifier boundary, so a long run of
# ``a_a_a_...`` without ``=`` fails once per run instead of re-trying every
# segment split (a nested ``(prefix_)*word`` quantifier did that and took
# seconds on a 20 kB hostile message).
ASSIGNMENT_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]+)=")
_ASSIGNMENT_VALUE_PATTERN = re.compile(r"[^&\s]+")
# A credential word may sit behind an identifier prefix (``MCP_API_KEY=``,
# ``SERVICE_ACCESS_TOKEN=``, ``DB_PASSWORD=``), so a key is a credential when it
# ends in one of these words and the word is the whole key or is joined to
# the prefix by ``_``/``-``. ``key`` itself is included: ``SECRET_KEY=``,
# ``AWS_SECRET_ACCESS_KEY=``, ``STRIPE_KEY=`` and ``PRIVATE_KEY=`` are all
# credentials, and an unknown ``*_key`` or ``*_token`` is treated as one (fail
# closed). A per-suffix qualifier below keeps a key readable only when the
# rest of the prefix carries no credential word, because this text also
# reaches user- and model-facing error messages: ``next_token=`` stays
# readable, ``secret_next_token=`` does not.
_CREDENTIAL_KEY_SUFFIXES = (
    "api_key",
    "api-key",
    "apikey",
    "access_token",
    "access-token",
    "accesstoken",
    "token",
    "password",
    "secret",
    "key",
)
_NON_CREDENTIAL_KEY_QUALIFIERS = frozenset(
    {
        "primary",
        "foreign",
        "unique",
        "composite",
        "index",
        "sort",
        "partition",
        "range",
        "hash",
        "lookup",
        "cache",
        "routing",
        "idempotency",
        "public",
        "object",
        "s3",
    }
)
# ``*_token`` names that mark a position or a request, not access: paging
# cursors (``next_token``, ``page_token``, ``continuation_token``, ...) and a
# client-chosen de-duplication id (``idempotency_token``). Kept apart from
# the ``key`` list: ``public_token`` is exchanged for an access token, and
# ``s3_token`` / ``cache_token`` may be credentials.
# ``page_token`` is ambiguous: some code also names a Facebook Page access
# token that way. The name alone cannot tell them apart (#2356).
_NON_CREDENTIAL_TOKEN_QUALIFIERS = frozenset(
    {"idempotency", "next", "page", "continuation", "cursor", "sync", "pagination"}
)
# Credential words that make a key a credential wherever they sit in its
# prefix, not only in the segment next to the suffix: ``secret_next_token``
# is a secret however structural its last qualifier reads. The
# single-segment credential suffixes are reused verbatim (a prefix segment
# can only ever equal one of those), plus the words that appear only as a
# qualifier and never as a suffix of their own.
_CREDENTIAL_PREFIX_WORDS = frozenset(
    {
        suffix
        for suffix in _CREDENTIAL_KEY_SUFFIXES
        if "_" not in suffix and "-" not in suffix
    }
    | {
        "access",
        "auth",
        "bearer",
        "client",
        "oauth",
        "private",
        "refresh",
        "session",
        "signing",
    }
)
# The qualifier is the ``_``/``-`` segment just before the suffix
# (``next_page_token`` -> ``page``). Every other suffix (``api_key``,
# ``access_token``, ``password``, ``secret`` and their spellings) has no
# exemption, so ``page_access_token=`` stays masked. The exemption below
# only applies when that segment is joined to the suffix with ``_``; a
# hyphen join (``next-token=``, ``cache-key=``) is treated as a credential
# and masked. It also requires the *whole* prefix to be free of credential
# words (``_CREDENTIAL_PREFIX_WORDS``), so ``secret_next_token`` is masked
# even though its qualifier is ``next``.
_NON_CREDENTIAL_QUALIFIERS_BY_SUFFIX = {
    "key": _NON_CREDENTIAL_KEY_QUALIFIERS,
    "token": _NON_CREDENTIAL_TOKEN_QUALIFIERS,
}


def _is_credential_key(key: str) -> bool:
    lowered = key.lower()
    for suffix in _CREDENTIAL_KEY_SUFFIXES:
        if not lowered.endswith(suffix):
            continue
        prefix = lowered[: -len(suffix)]
        if prefix.strip("-") == "":
            # Bare word, or a CLI-style flag such as ``--api-key=``.
            return True
        if prefix[-1] not in "_-":
            # ``monkey=`` / ``hotkey=`` / ``mytoken=``: the credential word is
            # not a separate segment of the identifier.
            continue
        # The exemption only reads as a deliberate structural-field name
        # when the qualifier is joined to the suffix by ``_``
        # (``primary_key=``, ``next_token=``). A hyphen join is treated as
        # a credential name: ``cache-key=`` / ``next-token=`` are masked;
        # only the underscore-joined spelling is exempt, and only when the
        # whole prefix is also free of credential words
        # (``_CREDENTIAL_PREFIX_WORDS``): ``secret_next_token=`` stays
        # masked even though its last qualifier is ``next``.
        exempt = _NON_CREDENTIAL_QUALIFIERS_BY_SUFFIX.get(suffix)
        if exempt is not None and prefix[-1] == "_":
            segments = re.split(r"[_-]", prefix.rstrip("_-"))
            if not _CREDENTIAL_PREFIX_WORDS.intersection(segments) and (
                segments[-1] in exempt
            ):
                return False
        return True
    return False


def _redact_assignments(text: str) -> str:
    # Values are read only for credentials and never overlap, and finditer
    # tries each position once, so every character is examined a constant
    # number of times however many ``identifier=`` runs the text holds.
    parts: list[str] = []
    copied = 0
    for match in ASSIGNMENT_PATTERN.finditer(text):
        if match.start() < copied or not _is_credential_key(match.group(1)):
            # Inside a value already masked, or not a credential: leave the
            # text as it is and keep scanning right after this ``=``.
            continue
        value = _ASSIGNMENT_VALUE_PATTERN.match(text, match.end())
        if value is None:
            continue
        parts.append(text[copied : match.end()])
        parts.append(_mask_secret(value.group(0)))
        copied = value.end()
    parts.append(text[copied:])
    return "".join(parts)


AUTH_HEADER_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*(?:bearer|basic)\s+)([^\s,;]+)"
)
HEADER_KEY_PATTERNS = [
    re.compile(r"(?i)(x-goog-api-key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(x-shopify-access-token\s*[:=]\s*)([^\s,;]+)"),
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


def build_ca_bundle_ssl_context() -> ssl.SSLContext:
    """Return an SSL context that still honors ``SSL_CERT_FILE``/``SSL_CERT_DIR``
    on a transport pinned to ``trust_env=False``.

    ``httpx.create_ssl_context()`` only reads ``SSL_CERT_FILE``/
    ``SSL_CERT_DIR`` when ``trust_env`` is true, since both env var lookups
    are gated behind that same flag. ``_PinnedA2ATransport`` and
    ``SafeOAuthAsyncHTTPTransport`` set ``trust_env=False`` on their inner
    ``httpx.AsyncHTTPTransport`` -- not to affect proxy handling (transport-
    level ``trust_env`` has no effect there; proxy-env fallback is gated at
    the ``httpx.AsyncClient`` level via ``allow_env_proxies = trust_env and
    transport is None``, which is already ``False`` once an explicit
    ``transport=`` is passed) but because it also feeds
    ``create_ssl_context()`` internally, and ``verify=True`` there would
    otherwise silently skip a private/internal CA bundle configured via
    those env vars. Build that context explicitly here, with
    ``trust_env=True``, and pass it as ``verify=`` instead.
    """

    return httpx.create_ssl_context(trust_env=True)


_SHARED_CA_GUARDED_MESSAGE = "shared CA context is guarded against mutation"
_SHARED_CA_ALPN = ["http/1.1"]

_shared_ca_lock = threading.Lock()
_shared_ca_slot: tuple[tuple[object, ...], ssl.SSLContext] | None = None


def _reject_shared_ca_mutation(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError(_SHARED_CA_GUARDED_MESSAGE)


def _set_shared_ca_alpn_protocols(protocols: Iterable[str]) -> None:
    # httpcore re-applies its ALPN list on every connection. The same value is
    # a no-op here so the ALPN setting of the published context is never
    # written again.
    if list(protocols) != _SHARED_CA_ALPN:
        raise RuntimeError(_SHARED_CA_GUARDED_MESSAGE)


def _shared_ca_key() -> tuple[object, ...]:
    """Mirror which CA source ``httpx.create_ssl_context(trust_env=True)`` uses."""
    cafile = os.environ.get("SSL_CERT_FILE")
    if cafile:
        try:
            stat = os.stat(cafile)
            file_identity: tuple[int, ...] | None = (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        except OSError:
            # Building the context reports the unreadable file.
            file_identity = None
        return ("file", cafile, file_identity)
    capath = os.environ.get("SSL_CERT_DIR")
    if capath:
        try:
            stat = os.stat(capath)
            dir_identity: tuple[int, ...] | None = (
                stat.st_dev,
                stat.st_ino,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        except OSError:
            dir_identity = None
        return ("dir", capath, dir_identity)
    return ("certifi",)


def shared_ca_bundle_ssl_context() -> ssl.SSLContext:
    """Return one process-wide CA-only context for HTTP/1.1 client transports.

    The context equals ``build_ca_bundle_ssl_context()``. It is rebuilt when
    the CA source changes: the key is the ``SSL_CERT_FILE`` path with that
    file's device, inode, size, mtime and ctime, or, when no file is set, the
    ``SSL_CERT_DIR`` path with that directory's device, inode, mtime and
    ctime. Any rewrite of the file or entry change in the directory therefore
    rebuilds the context. Only the latest context is kept; transports that
    already hold an older one keep using it. A failed build is not cached and
    raises as it does in ``build_ca_bundle_ssl_context()``.

    After publication the context is guarded against mutation:
    ``set_alpn_protocols`` with ``["http/1.1"]`` is a no-op, and any other
    ALPN list, ``load_cert_chain``, ``load_verify_locations``,
    ``load_default_certs``, ``set_default_verify_paths`` and ``set_ciphers``
    raise ``RuntimeError``. Attribute assignment (``verify_mode``,
    ``check_hostname``, ``options``) cannot be intercepted, so callers must
    not assign them. Use it only with ``http2=False`` transports and never
    where a client certificate is loaded; use ``build_ca_bundle_ssl_context()``
    there.

    The mutating methods are overridden on the instance, never through a
    subclass or ``__class__``. The type must stay exactly ``ssl.SSLContext``:
    for any other context class anyio's TLS stream creates the TLS object
    (``wrap_bio``) in its shared worker-thread pool, so a saturated pool
    delays every new handshake past its connect timeout.
    """
    global _shared_ca_slot
    with _shared_ca_lock:
        key = _shared_ca_key()
        slot = _shared_ca_slot
        if slot is not None and slot[0] == key:
            return slot[1]
        context = build_ca_bundle_ssl_context()
        context.set_alpn_protocols(_SHARED_CA_ALPN)
        context.set_alpn_protocols = _set_shared_ca_alpn_protocols  # type: ignore[method-assign,assignment]
        context.load_cert_chain = _reject_shared_ca_mutation  # type: ignore[method-assign]
        context.load_verify_locations = _reject_shared_ca_mutation  # type: ignore[method-assign]
        context.load_default_certs = _reject_shared_ca_mutation  # type: ignore[method-assign]
        context.set_default_verify_paths = _reject_shared_ca_mutation  # type: ignore[method-assign]
        context.set_ciphers = _reject_shared_ca_mutation  # type: ignore[method-assign]
        _shared_ca_slot = (key, context)
        return context


def validate_public_http_url_shape(
    url: str, *, host_validator: Callable[[str], None] | None = None
) -> SplitResult:
    """Validate the normalized host used by HTTP clients, without resolving DNS."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url must not contain embedded credentials")

    # aiohttp uses yarl's IDNA/UTS46 normalization before deciding whether to
    # bypass its resolver for an IP literal. Validate that same spelling.
    parsed = urlsplit(str(URL(url)))
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("url must have a hostname")
    host = hostname.rstrip(".")
    if ":" in host or host.replace(".", "").isdigit():
        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("url contains a non-canonical IP address") from exc
    (host_validator or reject_private_network_host)(host)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("url contains an invalid port") from exc
    if port == 0:
        raise ValueError("url contains an invalid port")
    return parsed


def public_http_destination(url: str) -> tuple[str, int]:
    """Share normalized URL validation across synchronous and async transports."""
    parsed = validate_public_http_url_shape(url)
    hostname = parsed.hostname
    assert hostname is not None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return hostname, port


def _public_addresses(addresses: list[tuple], hostname: str) -> list[str]:
    if not addresses:
        raise ValueError(f"Host {hostname!r} did not resolve to an address")
    resolved: list[str] = []
    for *_, socket_address in addresses:
        ip = str(socket_address[0])
        reject_private_network_host(ip)
        resolved.append(ip)
    return resolved


def validate_public_http_url_sync(url: str) -> list[str]:
    """Resolve and validate every address used by synchronous SDK clients."""
    hostname, port = public_http_destination(url)
    return _public_addresses(
        socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM),
        hostname,
    )


async def validate_public_http_url(url: str) -> list[str]:
    """Resolve an HTTP(S) URL, reject non-public targets, return validated IPs."""
    hostname, port = public_http_destination(url)

    addresses = await asyncio.to_thread(
        socket.getaddrinfo,
        hostname,
        port,
        socket.AF_UNSPEC,
        socket.SOCK_STREAM,
    )
    return _public_addresses(addresses, hostname)


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


PUBLIC_HTTP_MAX_REDIRECTS = 5


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
    allow_https_downgrade: bool = True,
    require_identity_encoding: bool = False,
    required_status_code: int | None = None,
    host_validator: Callable[[str], None] | None = None,
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
    send that IP as SNI and break SNI-strict servers. Keep the logical URL
    for TLS, and pair this mode with ``public_fetch_client``: its owned gate
    independently validates connect-time DNS and sends only a literal-IP
    CONNECT to the trusted upstream, closing the rebinding window. This body
    helper alone only performs preflight validation. A caller using a proxy
    without that gate must require public-only connect-time DNS enforcement
    by the proxy itself.

    The optional host_validator receives each normalized logical hostname
    before DNS and may raise to reject it. Public-network checks still apply.

    Input-download callers can forbid HTTPS downgrades, require a complete
    HTTP 200 response and reject compressed bodies before HTTPX decompression.
    """

    # Optional caller restrictions supplement the public-network checks.
    # Validate each logical host before DNS or any request, including redirects.
    logical_url = url
    display_name = resource_name[:1].upper() + resource_name[1:]
    for redirect_count in range(PUBLIC_HTTP_MAX_REDIRECTS + 1):
        if host_validator is not None:
            parsed_host = validate_public_http_url_shape(logical_url).hostname
            assert parsed_host is not None
            host_validator(parsed_host.rstrip("."))
        resolved_ips = await validate_public_http_url(logical_url)
        parsed = urlsplit(logical_url)
        request_headers = httpx.Headers(headers)
        if require_identity_encoding:
            request_headers["Accept-Encoding"] = "identity"
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
                if redirect_count >= PUBLIC_HTTP_MAX_REDIRECTS:
                    raise ValueError(f"{display_name} exceeded redirect limit")
                next_url = urljoin(logical_url, location)
                if (
                    not allow_https_downgrade
                    and parsed.scheme == "https"
                    and urlsplit(next_url).scheme == "http"
                ):
                    raise ValueError(f"{display_name} redirect downgrades HTTPS")
                logical_url = next_url
                continue
            if 300 <= response.status_code < 400:
                raise ValueError(
                    f"{display_name} returned unsupported redirect status "
                    f"{response.status_code}"
                )
            response.raise_for_status()
            if (
                required_status_code is not None
                and response.status_code != required_status_code
            ):
                raise httpx.HTTPStatusError(
                    f"{display_name} requires HTTP {required_status_code}",
                    request=response.request,
                    response=response,
                )
            if (
                require_identity_encoding
                and response.headers.get("content-encoding", "identity").strip().lower()
                != "identity"
            ):
                raise ValueError(f"{display_name} uses unsupported content encoding")
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


def host_matches_suffix(hostname: str, suffix: str) -> bool:
    """Return whether ``hostname`` is ``suffix`` or one of its subdomains."""
    hostname = hostname.lower()
    suffix = suffix.lower()
    return hostname == suffix or hostname.endswith(f".{suffix}")


def redact_url_credentials_for_logging(
    url: str, *, redact_all_query_values: bool = False
) -> str:
    """Redact sensitive query credentials and any embedded userinfo from a
    URL (e.g. a proxy URL's "user:pass@host", the single most common place
    a URL carries a credential -- a query-string-only check would silently
    pass it through unchanged).

    ``redact_all_query_values`` is for callers that accept arbitrary query
    parameters and therefore cannot know which values are credentials. The
    default remains selective because many existing diagnostic call sites
    intentionally retain ordinary query values. See issue #2356 for the
    broader, repository-wide limits of keyword-based secret redaction.
    """
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
        if value and (redact_all_query_values or key.lower() in SENSITIVE_QUERY_KEYS):
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
    """Redact common key/token patterns from arbitrary text.

    Recognition is by keyword and shape, so it is not exhaustive: it covers
    HTTP/WebSocket URLs (``redact_url_credentials_for_logging``),
    ``identifier=value`` assignments whose identifier ``_is_credential_key``
    accepts, and the header shapes in ``AUTH_HEADER_PATTERN`` and
    ``HEADER_KEY_PATTERNS``. Shapes it is known to miss are listed in
    #2356, and secrets in URL path segments in #2272; add a pattern here
    when a new one turns up.
    """
    if not text:
        return text

    redacted = WEBSOCKET_CREDENTIAL_PATTERN.sub("xagent-auth-token.[REDACTED]", text)
    redacted = URL_PATTERN.sub(
        lambda match: redact_url_credentials_for_logging(match.group(0)),
        redacted,
    )
    redacted = _redact_assignments(redacted)
    redacted = AUTH_HEADER_PATTERN.sub(
        lambda match: f"{match.group(1)}{_mask_secret(match.group(2))}",
        redacted,
    )
    for pattern in HEADER_KEY_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}{_mask_secret(match.group(2))}",
            redacted,
        )
    return redacted
