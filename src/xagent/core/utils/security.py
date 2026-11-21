"""Security helpers for outbound hosts and sensitive log data."""

import asyncio
import ipaddress
import os
import re
import socket
import ssl
import threading
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterable, Mapping, NoReturn
from urllib.parse import (
    SplitResult,
    parse_qsl,
    quote_plus,
    urljoin,
    urlsplit,
    urlunsplit,
)

import httpx
from yarl import URL

# Parameter names that name a credential or a one-time proof in a URL query
# string. This is the union of the names the URL redactor and the server
# request-target filter used to keep apart; the credential-name rule below
# (``_is_credential_key``) remains the common part every redactor shares, and
# a URL query additionally reads this list. A name the credential rule
# already matches (``access_token``, ``api_key``, ``token``, ``secret`` and
# the other spellings) is not repeated here.
SENSITIVE_QUERY_KEYS = frozenset(
    {
        "auth",
        "code",
        "connected_account_id",
        "credential",
        "jwt",
        "session",
        "session_uri",
        "sig",
        "state",
        "ticket",
    }
)
# Query-string suffixes that mark a credential or a one-time proof without
# the name carrying a credential word of its own. This is the union of the
# suffix list the server request-target filter applied before the name rule
# moved here and the ticket spellings the URL redactor added: a name such as
# ``page_token`` or ``primary_key`` stays masked there, because a log line may
# keep a paging cursor but must not keep an access token. Non-secret names
# such as ``page_size`` do not end in any of these suffixes and stay readable.
_REQUEST_TARGET_QUERY_SUFFIXES = (
    "_key",
    "_token",
    "_ticket",
    "-ticket",
)
# Names that are credentials wherever they appear, even outside a URL query:
# signed-URL and raw-header spellings that carry no credential word of their
# own.
_ALWAYS_SENSITIVE_PARAM_NAMES = frozenset(
    {
        "tempauth",
        "sig",
        "x-amz-credential",
    }
)

URL_PATTERN = re.compile(r"(?:https?|wss?)://[^\s\"'>]+")
# The value may already end in an outer layer's marker (or several); consume
# it so the replacement does not add a second placeholder
# (``xagent-auth-token.v1.[REDACTED]`` must become
# ``xagent-auth-token.[REDACTED]``, not ``...[REDACTED][REDACTED]``).
WEBSOCKET_CREDENTIAL_PATTERN = re.compile(
    r"xagent-auth-token\."
    r"(?:[A-Za-z0-9._~-]*+(?:\[REDACTED\])++)"
    r"|xagent-auth-token\.[A-Za-z0-9._~-]++"
)
# A JWT is ``header.payload.signature``, all base64url; the header of a JSON
# JWT always starts ``eyJ`` (``{"``). It can reach a log line without a URL or
# an ``identifier=`` around it (a bare token in a tool result, for instance).
# The scan below is token-based instead of one regex: a lookbehind that lets
# ``-`` open a header also lets every ``eyJ`` inside a dense run open one, and
# each attempt then scans the rest of the run for a dot (quadratic on
# ``"eyJ-" * n``). A token run is examined once.
_JWT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_JWT_HEADER_PREFIX = "eyJ"
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
    "signature",
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
    if lowered in _ALWAYS_SENSITIVE_PARAM_NAMES:
        return True
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


def is_credential_key(key: str) -> bool:
    """Whether ``key`` names a credential, by the same rule the redactor uses.

    Exposed for callers that find credentials in a shape
    :func:`redact_sensitive_text` does not parse -- a quoted field name, for
    instance -- so they can reuse this rule instead of writing a second one.
    """
    return _is_credential_key(key)


def is_sensitive_param_name(name: str, *, url_query: bool = False) -> bool:
    """Whether ``name`` names a credential or one-time proof worth masking.

    ``url_query`` selects the rules for a URL query string: the union of the
    names the URL redactor and the server request-target filter used to keep
    apart, plus credential suffixes such as ``_key`` and ``_signature``.
    Without it the rule is narrower on purpose: ``code=400``, ``ticket=42``
    and ``support_ticket=42`` in free text are diagnostics, not credentials,
    and this text also reaches user- and model-facing error messages.
    """
    if _is_credential_key(name):
        return True
    if not url_query:
        return False
    lowered = name.lower()
    return lowered in SENSITIVE_QUERY_KEYS or lowered.endswith(
        _REQUEST_TARGET_QUERY_SUFFIXES
    )


def _redact_assignments(text: str, mask: Callable[[str], str]) -> str:
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
        if value is None or _is_redaction_placeholder(value.group(0)):
            # Already masked by an outer layer or an earlier pass. Masking a
            # short ``***suffix`` marker again would eat part of the suffix
            # and break idempotency; leave the marker exactly as it is.
            continue
        parts.append(text[copied : match.end()])
        parts.append(mask(value.group(0)))
        copied = value.end()
    parts.append(text[copied:])
    return "".join(parts)


# One ``\s*`` on each side of the optional quote before the scheme: a second
# ``\s*`` after the quote made the whitespace between ``:`` and the scheme
# split at every position (quadratic on ``Authorization:`` followed by a long
# run of spaces). The value is read as everything up to whitespace, comma or
# semicolon -- the same span the pre-change pattern consumed -- so a quote or
# a bracket at the end of the value is masked instead of ending the match
# early. The wrappers are put back by ``_mask_value_keeping_wrappers``.
#
# The value also stops before ``&<another header name>[:=]``. A value used to
# consume the following ``&`` and the next header's name as part of itself,
# after which the second header's real value was no longer adjacent to a
# header name and stayed in the clear. Stopping only at a known header name
# keeps an ordinary value that happens to contain ``&`` masked as a whole.
# Each character is tested once by the negative lookahead, so this stays
# linear.
_HEADER_STOP_NAMES = (
    r"[\"']?(?:proxy-)?authorization[\"']?"
    r"|[\"']?x-goog-api-key[\"']?"
    r"|[\"']?x-api-key[\"']?"
    r"|[\"']?x-shopify-access-token[\"']?"
    # Any quoted field name: a JSON object or dict repr can put a
    # non-header credential name (``api_key``) right behind an unquoted
    # header value, and the value must not swallow that name.
    r"|[\"'][A-Za-z0-9_-]++[\"']"
)
_HEADER_VALUE = rf"(?:(?!&(?:{_HEADER_STOP_NAMES})\s*[:=])[^\s,;])+"
AUTH_HEADER_PATTERN = re.compile(
    r"(?i)([\"']?(?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?(?:bearer|basic)\s+)"
    rf"({_HEADER_VALUE})"
)
# The separator may repeat: rebuilding a query that names a credential without
# a value appends ``=`` in front of whatever separated the name from the value
# (``X-Api-Key = value`` becomes ``X-Api-Key= = value``), and the rule must
# still see the value instead of masking the second separator. A name followed
# only by whitespace (``x-api-key header``) is ordinary text and still does
# not match.
HEADER_KEY_PATTERNS = [
    re.compile(
        rf"(?i)([\"']?x-goog-api-key[\"']?\s*[:=]\s*(?:[:=]\s*)*)({_HEADER_VALUE})"
    ),
    re.compile(rf"(?i)([\"']?x-api-key[\"']?\s*[:=]\s*(?:[:=]\s*)*)({_HEADER_VALUE})"),
    re.compile(
        rf"(?i)([\"']?x-shopify-access-token[\"']?\s*[:=]\s*(?:[:=]\s*)*)"
        rf"({_HEADER_VALUE})"
    ),
]
# A quoted identifier and ``:`` or ``=`` separator, as a JSON object or a
# Python dict repr writes it (``{"x-api-key": "sk-..."}``, ``{'api_key':
# 'sk-...'}``, ``"access_token": "..."``). The header rules above only cover
# the fixed header spellings; this rule hands the name to
# ``_is_credential_key`` so every credential name gets the same treatment.
# The name run is possessive and the value cannot open a nested object or
# list, so a long run of quotes or a nested JSON document is still scanned
# once (see the linear-time test for quoted-name runs). The value also stops
# before a following ``&"name":`` field, as the header rule stops before the
# next known header name: otherwise the first field's value swallows the next
# field's name and its value is never examined. Each character is tested once
# by the lookahead, so this stays linear.
_QUOTED_FIELD_STOP = r"&[\"']?[A-Za-z0-9_-]++[\"']?\s*[:=]"
_QUOTED_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])(([\"'])([A-Za-z0-9_-]++)\2\s*[:=]\s*)"
    rf"((?:(?!{_QUOTED_FIELD_STOP})[^\s,;{{}}\[\]])+)"
)
# A coarse ``?name=value`` / ``&name=value`` shape for the pattern-only
# fallback, which deliberately does not parse a URL. The name still goes
# through ``is_sensitive_param_name(..., url_query=True)``, so URL-only names
# such as ``code``, ``state`` and ``ticket`` are covered.
_URL_QUERY_VALUE_PATTERN = re.compile(r"([?&])([A-Za-z0-9_.~-]+)=([^&\s]+)")


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


_MASK_SUFFIX_MAX_LENGTH = 4


def _mask_secret(value: str) -> str:
    """Mask a secret while preserving a short suffix for troubleshooting."""
    if not value:
        return "***"
    tail_len = _MASK_SUFFIX_MAX_LENGTH if len(value) > 8 else 2
    return "***" + value[-tail_len:]


REDACTED = "[REDACTED]"


def _redact_completely(value: str) -> str:
    """Mask a secret without keeping any part of it (log-only surfaces)."""
    return REDACTED


# Characters that can wrap a value when it is embedded in a JSON object, a
# dict repr, a header line or a parenthesised sentence. They are put back
# around the mask so the surrounding syntax survives.
_VALUE_WRAPPERS = "\"'`([{<)]}>"


def _is_redaction_placeholder(value: str) -> bool:
    """Whether ``value`` is already one of the markers a redactor emits.

    A second redaction pass must leave these alone: ``_mask_secret`` keeps a
    four-character suffix (``***c123``), and replacing the marker with itself
    is harmless only for the full ``[REDACTED]`` marker. Recognizing the
    markers also keeps a value an outer layer already masked from being
    masked a second time.
    """
    candidate = value.strip().strip(_VALUE_WRAPPERS).strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    if lowered.startswith("%5b") and lowered.endswith("%5d"):
        candidate = candidate[3:-3]
        lowered = candidate.lower()
    if lowered == "redacted":
        return True
    # ``***suffix`` / ``******``: the short mask ``_mask_secret`` emits. It is
    # ``***`` plus at most the four trailing characters kept for
    # troubleshooting, so a longer value that merely starts with ``***`` is a
    # real secret and must still be masked (a value such as
    # ``***QWERTYUIOPASDFGZ`` is not a marker this redactor produced).
    return (
        candidate.startswith("***")
        and len(candidate) <= len("***") + _MASK_SUFFIX_MAX_LENGTH
    )


def _mask_value_keeping_wrappers(value: str, mask: Callable[[str], str]) -> str:
    """Mask ``value`` but keep any quotes/brackets around it."""
    if _is_redaction_placeholder(value):
        return value
    start = 0
    end = len(value)
    while start < end and value[start] in _VALUE_WRAPPERS:
        start += 1
    while end > start and value[end - 1] in _VALUE_WRAPPERS:
        end -= 1
    if start >= end:
        return value
    return f"{value[:start]}{mask(value[start:end])}{value[end:]}"


_USERINFO_PREFIX_PATTERN = re.compile(r"://[^/\s@]*@")


def _query_item_needs_redaction(
    key: str, value: str, *, redact_all_query_values: bool
) -> bool:
    """Whether one query item gives the URL redactor a reason to rebuild."""
    if _is_redaction_placeholder(value):
        # Already masked by an outer layer: nothing left to do for this item.
        return False
    if redact_all_query_values and value:
        return True
    return is_sensitive_param_name(key, url_query=True)


def host_matches_suffix(hostname: str, suffix: str) -> bool:
    """Return whether ``hostname`` is ``suffix`` or one of its subdomains."""
    hostname = hostname.lower()
    suffix = suffix.lower()
    return hostname == suffix or hostname.endswith(f".{suffix}")


def _redact_url_credentials_for_logging(
    url: str, *, mask: Callable[[str], str], redact_all_query_values: bool = False
) -> str:
    """Redact query credentials and embedded userinfo from ``url``."""
    if not url:
        return url

    try:
        parsed = urlsplit(url)
    except ValueError:
        # Malformed enough that urlsplit itself can't parse it (e.g. an
        # unclosed IPv6 bracket) -- fall back to structure-agnostic scans:
        # mask a "user:pass@" prefix and every coarse ``?name=value`` the
        # shared query rule recognizes. Returning the input unchanged here
        # would be worse than not redacting at all: it would silently leak a
        # credential through the one code path meant to catch it. Callers
        # that share this function (the record factory and the server
        # request-target filter) get the same failure policy.
        redacted = _USERINFO_PREFIX_PATTERN.sub("://***@", url)
        return _redact_url_query_values(redacted, mask) if "?" in redacted else redacted

    has_userinfo = parsed.username is not None or parsed.password is not None
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if not query_items and not has_userinfo:
        return url

    # Nothing to mask: return the URL byte for byte. This check comes before
    # any re-encoding, so an untouched query keeps ``?flag`` (not ``?flag=``),
    # ``%20`` and a trailing ``)``, and a value that cannot be percent-encoded
    # (an unpaired surrogate, for instance) is never re-encoded. When a
    # rebuild is needed, ``quote_plus`` below runs with ``surrogatepass``
    # rather than raising and losing the whole record's redaction.
    # A credential *name* with no value still needs the rebuild: the previous
    # implementation rewrote ``?X-Api-Key`` as ``?X-Api-Key=``, and the header
    # and assignment rules that run after this one then masked a value the
    # query parser could not see -- the ``'token`` tail of a malformed
    # ``?MCP_API_KEY'token`` URL, for instance.
    if not has_userinfo and not any(
        _query_item_needs_redaction(
            key, value, redact_all_query_values=redact_all_query_values
        )
        for key, value in query_items
    ):
        return url

    encoded_items: list[str] = []
    for key, value in query_items:
        is_sensitive = bool(value) and (
            redact_all_query_values or is_sensitive_param_name(key, url_query=True)
        )
        if _is_redaction_placeholder(value):
            # A marker is not query data: keep it readable instead of
            # percent-encoding it as ``%5BREDACTED%5D``.
            encoded_value = value
        elif is_sensitive:
            masked = mask(value)
            encoded_value = (
                masked
                if masked == REDACTED
                else quote_plus(masked, errors="surrogatepass")
            )
        else:
            # ``surrogatepass`` keeps a request target that arrived through
            # ``surrogateescape`` from raising here. A raise would reach the
            # record factory, which would have to return the whole record
            # unredacted; encoding the lone surrogate is harmless in a log.
            encoded_value = quote_plus(value, errors="surrogatepass")
        encoded_items.append(
            f"{quote_plus(key, errors='surrogatepass')}={encoded_value}"
        )

    redacted_query = "&".join(encoded_items)

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
    return _redact_url_credentials_for_logging(
        url, mask=_mask_secret, redact_all_query_values=redact_all_query_values
    )


def redact_url_query_credentials(url: str) -> str:
    """Redact credential-like query values in a request target or URL.

    Shares the parser, the percent-encoder and the failure policy of
    :func:`redact_url_credentials_for_logging`, but replaces every masked
    value with the full ``[REDACTED]`` marker instead of the short
    ``***suffix`` one: a server access line is a log-only surface and should
    not keep part of a credential. A target with nothing to redact is
    returned byte for byte, so the line still matches the request on the
    wire.
    """
    return _redact_url_credentials_for_logging(url, mask=_redact_completely)


# A whitespace-delimited token. Relative request targets (``/cb?code=...``)
# and relative links are not matched by ``URL_PATTERN``, which requires a
# scheme. Tokenizing once and examining each token keeps this pass linear,
# unlike a ``/``-anchored pattern that rescans the rest of a long token from
# every slash when the token has no query string.
_RELATIVE_TARGET_TOKEN_PATTERN = re.compile(r"[^\s\"'<>]+")
_RELATIVE_TARGET_LEADING = "([{<"
_RELATIVE_TARGET_TRAILING = ")]}>.,;:!"


def _target_has_unmasked_sensitive_value(target: str) -> bool:
    """Whether a relative target still holds a sensitive value to mask."""
    try:
        parsed = urlsplit(target)
    except ValueError:
        return True
    return any(
        value
        and not _is_redaction_placeholder(value)
        and is_sensitive_param_name(key, url_query=True)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    )


def _redact_relative_target_token(token: str, *, mask: Callable[[str], str]) -> str:
    start = 0
    end = len(token)
    while start < end and token[start] in _RELATIVE_TARGET_LEADING:
        start += 1
    while end > start and token[end - 1] in _RELATIVE_TARGET_TRAILING:
        # A ``]`` that closes ``[REDACTED]`` is part of the marker, not
        # trailing punctuation. Stripping it made the URL rebuild insert an
        # ``=`` in the middle of the marker (``[REDACTED=]``).
        if (
            token[end - 1] == "]"
            and token[max(start, end - len(REDACTED)) : end].upper() == REDACTED
        ):
            break
        end -= 1
    target = token[start:end]
    if not target.startswith("/") or "?" not in target:
        return token
    if REDACTED in target and not _target_has_unmasked_sensitive_value(target):
        # Everything the URL rule would mask here is already a marker: the
        # header pre-pass ran before this one. Returning the token unchanged
        # keeps the marker readable instead of percent-encoding it as a query
        # key and appending ``=`` to it.
        return token
    redacted = _redact_url_credentials_for_logging(target, mask=mask)
    if redacted == target:
        return token
    if redacted == f"{target}=" and token[end:] == ":":
        # A trailing colon is the header separator in
        # ``/cb?X-Api-Key: value``, not part of the target. The rebuild only
        # added the ``=`` for a credential name without a value; inserting it
        # in front of the colon would hide the real separator and the
        # assignment rule would mask the colon instead. Leave the shape for
        # the header-key rule that runs next.
        return token
    return f"{token[:start]}{redacted}{token[end:]}"


def _redact_relative_targets(text: str, mask: Callable[[str], str]) -> str:
    if "?" not in text or "/" not in text:
        return text
    return _RELATIVE_TARGET_TOKEN_PATTERN.sub(
        lambda match: _redact_relative_target_token(match.group(0), mask=mask),
        text,
    )


def _jwt_header_offset(segment: str) -> int | None:
    """Return where a JWT header starts inside one token run, if anywhere.

    A header either starts the run or follows a hyphen: ``\\b`` treated a
    hyphen as a boundary, so ``prefix-eyJ...`` was masked, and a header inside
    a longer base64url run was not. At least one character must follow
    ``eyJ`` before the first dot, because the header segment of a JWT is never
    just those three characters.
    """
    offset = segment.find(_JWT_HEADER_PREFIX)
    while offset != -1:
        if offset == 0 or segment[offset - 1] == "-":
            if offset + len(_JWT_HEADER_PREFIX) < len(segment):
                return offset
            return None
        offset = segment.find(_JWT_HEADER_PREFIX, offset + 1)
    return None


def _redact_jwts(text: str) -> str:
    """Replace JWT-shaped ``header.payload.signature`` runs in ``text``.

    The text is split once into base64url token runs; a match is three runs
    joined by single dots whose first run opens with a JWT header. Scanning
    runs instead of every ``eyJ`` occurrence is what keeps this linear: a
    dense ``"eyJ-" * n`` run has one candidate header (the run start) or none,
    never one candidate per ``eyJ``.
    """
    if _JWT_HEADER_PREFIX not in text or "." not in text:
        return text
    segments = iter(_JWT_TOKEN_PATTERN.finditer(text))
    window = list(islice(segments, 3))
    parts: list[str] = []
    copied = 0
    changed = False
    while len(window) == 3:
        first, second, third = window
        if (
            second.start() - first.end() == 1
            and text[first.end()] == "."
            and third.start() - second.end() == 1
            and text[second.end()] == "."
        ):
            offset = _jwt_header_offset(first.group(0))
            if offset is not None:
                parts.append(text[copied : first.start() + offset])
                parts.append(REDACTED)
                copied = third.end()
                changed = True
                window = list(islice(segments, 3))
                continue
        window.pop(0)
        following = next(segments, None)
        if following is not None:
            window.append(following)
    if not changed:
        return text
    parts.append(text[copied:])
    return "".join(parts)


def _redact_url_query_values(text: str, mask: Callable[[str], str]) -> str:
    """Mask coarse ``?name=value`` shapes without parsing a URL.

    Used by the pattern-only fallback. A value already carrying a marker is
    left alone so a second pass stays idempotent.
    """

    def replace(match: re.Match[str]) -> str:
        if not is_sensitive_param_name(match.group(2), url_query=True):
            return match.group(0)
        value = match.group(3)
        if _is_redaction_placeholder(value):
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}={mask(value)}"

    return _URL_QUERY_VALUE_PATTERN.sub(replace, text)


def _mask_headers(text: str, mask: Callable[[str], str]) -> str:
    """Mask the header-shaped and quoted credential values in ``text``."""
    text = AUTH_HEADER_PATTERN.sub(
        lambda match: f"{match.group(1)}"
        f"{_mask_value_keeping_wrappers(match.group(2), mask)}",
        text,
    )
    for pattern in HEADER_KEY_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1)}"
            f"{_mask_value_keeping_wrappers(match.group(2), mask)}",
            text,
        )
    text = _QUOTED_CREDENTIAL_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}{_mask_value_keeping_wrappers(match.group(4), mask)}"
            if _is_credential_key(match.group(3))
            else match.group(0)
        ),
        text,
    )
    return text


def _redact_text(text: str, mask: Callable[[str], str]) -> str:
    redacted = WEBSOCKET_CREDENTIAL_PATTERN.sub("xagent-auth-token.[REDACTED]", text)
    # Mask header shapes before any URL or relative-target rebuild. A rebuild
    # percent-encodes the query key (``X-Api-Key%3A...``), after which the
    # header rules no longer recognize the value; it also appends ``=`` to a
    # credential name that has none, which can move the value away from the
    # separator the header rule reads. Masking here keeps both shapes covered.
    redacted = _mask_headers(redacted, mask)
    redacted = URL_PATTERN.sub(
        lambda match: _redact_url_credentials_for_logging(match.group(0), mask=mask),
        redacted,
    )
    redacted = _redact_relative_targets(redacted, mask)
    redacted = _redact_assignments(redacted, mask)
    # The rebuild can also create a header shape that was not there before
    # (``?X-Api-Key sk-...`` becomes ``?X-Api-Key= sk-...``), so the same rule
    # runs once more over the rebuilt text.
    return _mask_headers(redacted, mask)


def redact_sensitive_text(text: str) -> str:
    """Redact common key/token patterns from arbitrary text.

    Recognition is by keyword and shape, so it is not exhaustive: it covers
    HTTP/WebSocket URLs and relative request targets
    (``redact_url_credentials_for_logging``), ``identifier=value``
    assignments whose identifier ``_is_credential_key`` accepts, quoted
    ``"name": value`` fields (JSON objects and dict reprs), and the header
    shapes in ``AUTH_HEADER_PATTERN`` and ``HEADER_KEY_PATTERNS``.
    Shapes it is known to miss are listed in
    #2356, and secrets in URL path segments in #2272; add a pattern here
    when a new one turns up.

    This text also reaches users and models, so the URL rule is part of the
    contract: a query value whose name is ``code``, ``state``, ``session``,
    ``auth``, ``page_token`` or ``connected_account_id`` -- or ends in
    ``_key``, ``_token``, ``_ticket``, ``_secret``, ``_password`` or
    ``_signature`` -- is masked here too, not only in a log record. The mask
    keeps a short suffix (``***c123``) so a diagnostic stays readable.
    """
    if not text:
        return text
    return _redact_text(text, _mask_secret)


def redact_log_text(text: str) -> str:
    """Redact credentials from one log message at record creation.

    Extends :func:`redact_sensitive_text` to bare JWT-shaped strings, which
    can reach a log line without a URL or an ``identifier=`` around them (a
    token inside a tool result, for instance), and replaces every masked
    value outright instead of keeping a short suffix: a log line should not
    carry part of a credential either.

    This is the text half of the record-factory contract. It does not reach
    ``extra=`` fields (attached after the factory runs), the traceback when a
    handler renders ``exc_info`` itself instead of using ``exc_text``, or
    log lines a subprocess configures on its own.
    """
    if not text:
        return text
    redacted = _redact_text(text, _redact_completely)
    return _redact_jwts(redacted)


def redact_log_text_pattern_only(text: str) -> str:
    """Redact the header, assignment and JWT shapes without parsing URLs.

    A last-resort pass for the log-record factory: if
    :func:`redact_log_text` raises for any reason, this function still masks
    the shapes that need no structural parsing (``Authorization`` and the
    ``x-api-key``-style headers, ``identifier=value`` assignments, coarse
    ``?name=value`` query values, the websocket credential marker, and bare
    JWTs) so the record is not written out unredacted. It never parses or
    rebuilds a URL, so it cannot hit the encoding or URL-parser failures that
    the full pass can. Because it does not parse a URL, an encoded query name
    or a query name it cannot see is outside its scope; the full pass is what
    covers those.
    """
    if not text:
        return text
    redacted = WEBSOCKET_CREDENTIAL_PATTERN.sub("xagent-auth-token.[REDACTED]", text)
    redacted = _mask_headers(redacted, _redact_completely)
    redacted = _redact_url_query_values(redacted, _redact_completely)
    redacted = _redact_assignments(redacted, _redact_completely)
    return _redact_jwts(redacted)
