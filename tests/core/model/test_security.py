"""Tests for shared security helpers."""

import socket
import time
from unittest.mock import patch

import httpx
import pytest

from xagent.core.utils.security import (
    _NON_CREDENTIAL_KEY_QUALIFIERS,
    _NON_CREDENTIAL_QUALIFIERS_BY_SUFFIX,
    _NON_CREDENTIAL_TOKEN_QUALIFIERS,
    PrivateNetworkHostError,
    fetch_public_http_bytes,
    redact_sensitive_text,
    redact_url_credentials_for_logging,
    reject_private_network_host,
    validate_public_http_url,
)


def test_reject_private_network_host_rejects_cgnat_range() -> None:
    with pytest.raises(PrivateNetworkHostError):
        reject_private_network_host("100.64.0.1")


@pytest.mark.asyncio
async def test_validate_public_http_url_rejects_private_dns_result() -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    with patch("socket.getaddrinfo", return_value=resolved):
        with pytest.raises(PrivateNetworkHostError):
            await validate_public_http_url("https://public.example/logo.png")


@pytest.mark.asyncio
async def test_validate_public_http_url_accepts_only_public_dns_results() -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    with patch("socket.getaddrinfo", return_value=resolved):
        assert await validate_public_http_url("https://public.example/logo.png") == [
            "93.184.216.34"
        ]


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_pins_validated_ip() -> None:
    """The connection must use the IP resolved at validation time, not a
    second, independently-resolved IP — this is what closes the DNS
    rebinding / TOCTOU window."""

    getaddrinfo_calls: list[tuple] = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        getaddrinfo_calls.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["host_header"] = request.headers.get("host")
        captured["sni_hostname"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        async with httpx.AsyncClient(transport=transport) as client:
            response = await fetch_public_http_bytes(
                client,
                "https://rebind.example/x",
                max_content_bytes=1024,
                timeout=5,
            )

    assert response.content == b"ok"
    assert captured["host"] == "93.184.216.34"
    assert captured["host_header"] == "rebind.example"
    assert captured["sni_hostname"] == "rebind.example"
    assert len(getaddrinfo_calls) == 1


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_via_proxy_does_not_rewrite_sni_to_ip() -> None:
    """When routed through an HTTP CONNECT proxy, the request must keep the
    original hostname as the connect target and TLS SNI. httpcore's CONNECT
    tunnel path derives SNI from the request's remote origin and ignores the
    ``sni_hostname`` extension entirely, so pinning the URL to a bare IP (as
    the direct-connection path does) sends the IP as SNI and breaks
    SNI-strict HTTPS servers behind the proxy."""

    getaddrinfo_calls: list[tuple] = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        getaddrinfo_calls.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["host_header"] = request.headers.get("host")
        captured["sni_hostname"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        async with httpx.AsyncClient(transport=transport) as client:
            response = await fetch_public_http_bytes(
                client,
                "https://rebind.example/x",
                max_content_bytes=1024,
                timeout=5,
                via_proxy=True,
            )

    assert response.content == b"ok"
    # The request must still target the original hostname (not the pinned
    # IP) so a CONNECT proxy performs its own DNS resolution and TLS SNI
    # matches the real origin.
    assert captured["host"] == "rebind.example"
    assert captured["sni_hostname"] is None
    # DNS validation still runs up front to reject private-network targets
    # before the request is ever dispatched to the proxy.
    assert len(getaddrinfo_calls) == 1


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_via_proxy_still_rejects_private_dns_result() -> (
    None
):
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            "private-network target must be rejected before connecting"
        )

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", return_value=resolved):
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(PrivateNetworkHostError):
                await fetch_public_http_bytes(
                    client,
                    "https://rebind.example/x",
                    max_content_bytes=1024,
                    timeout=5,
                    via_proxy=True,
                )


@pytest.mark.asyncio
async def test_fetch_public_http_bytes_revalidates_redirect_target() -> None:
    """Each redirect hop must be re-validated — a redirect to an internal
    host must be rejected even though the initial hop was public."""

    resolutions = iter(
        [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
        ]
    )

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return next(resolutions)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(
                302, headers={"location": "https://internal.example/"}
            )
        raise AssertionError("second hop must be rejected before connecting")

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(PrivateNetworkHostError):
                await fetch_public_http_bytes(
                    client,
                    "https://rebind.example/x",
                    max_content_bytes=1024,
                    timeout=5,
                )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [300, 305])
async def test_fetch_rejects_unhandled_3xx(status_code: int) -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    transport = httpx.MockTransport(handler)

    with patch("socket.getaddrinfo", return_value=resolved):
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await fetch_public_http_bytes(
                    client,
                    "https://public.example/x",
                    max_content_bytes=1024,
                    timeout=5,
                )


def test_redact_url_credentials_for_logging_masks_sensitive_query_values() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySecret&v=1"
    redacted = redact_url_credentials_for_logging(url)

    assert "AIzaSySecret" not in redacted
    assert "key=%2A%2A%2A" in redacted
    assert "v=1" in redacted


def test_redact_url_credentials_for_logging_masks_embedded_userinfo() -> None:
    # The query string isn't the only -- or even the most common -- place a
    # URL carries a credential; a proxy URL's "user:pass@host" needs the
    # same treatment, or it passes through this function unchanged.
    url = "https://alice:s3cret-pass@proxy.internal:8080/"
    redacted = redact_url_credentials_for_logging(url)

    assert "alice" not in redacted
    assert "s3cret-pass" not in redacted
    assert redacted == "https://***@proxy.internal:8080/"


def test_redact_url_credentials_for_logging_preserves_host_casing() -> None:
    # A URL with no userinfo keeps its original host casing (urlunsplit
    # just passes netloc through); the userinfo-redaction branch must not
    # be the only place that silently lowercases it.
    url = "https://Alice:s3cret-pass@Proxy-Host.Internal:8080/"
    redacted = redact_url_credentials_for_logging(url)

    assert redacted == "https://***@Proxy-Host.Internal:8080/"


def test_redact_url_credentials_for_logging_preserves_ipv6_brackets() -> None:
    url = "https://alice:s3cret-pass@[2001:db8::1]:443/v1"
    redacted = redact_url_credentials_for_logging(url)

    assert "s3cret-pass" not in redacted
    assert redacted == "https://***@[2001:db8::1]:443/v1"


def test_redact_url_credentials_for_logging_does_not_leak_on_parse_failure() -> None:
    # A malformed URL (unclosed IPv6 bracket) that urlsplit can't parse
    # must not fall back to returning the credential-bearing input
    # unchanged -- that would be worse than not attempting redaction at
    # all, since it looks sanitized but isn't.
    url = "https://alice:s3cret-pass@[::1/"
    redacted = redact_url_credentials_for_logging(url)

    assert "s3cret-pass" not in redacted


def test_redact_sensitive_text_does_not_leak_malformed_proxy_url() -> None:
    text = "Unable to connect to proxy https://alice:s3cret-pass@[::1/"
    redacted = redact_sensitive_text(text)

    assert "s3cret-pass" not in redacted


def test_redact_sensitive_text_masks_embedded_proxy_userinfo() -> None:
    text = "Unable to connect to proxy https://alice:s3cret-pass@proxy.internal:8080/"
    redacted = redact_sensitive_text(text)

    assert "s3cret-pass" not in redacted


def test_redact_sensitive_text_masks_bearer_and_header_keys() -> None:
    text = (
        "Authorization: Bearer sk-secret-value "
        "x-goog-api-key: AIzaSyHeaderSecret "
        "url=https://example.com/path?api_key=my_api_key"
    )
    redacted = redact_sensitive_text(text)

    assert "sk-secret-value" not in redacted
    assert "AIzaSyHeaderSecret" not in redacted
    assert "my_api_key" not in redacted


def test_redact_sensitive_text_masks_shopify_header_case_insensitively() -> None:
    text = "x-ShOpIfY-aCcEsS-tOkEn: shpat-secret Other-Header: preserved"

    redacted = redact_sensitive_text(text)

    assert "shpat-secret" not in redacted
    assert "x-ShOpIfY-aCcEsS-tOkEn: ***" in redacted
    assert "Other-Header: preserved" in redacted


def test_redact_sensitive_text_masks_basic_auth_credential() -> None:
    text = "Authorization: Basic dXNlcjpzdXBlci1zZWNyZXQtdG9rZW4="
    redacted = redact_sensitive_text(text)

    assert "dXNlcjpzdXBlci1zZWNyZXQtdG9rZW4=" not in redacted
    # Not just "the secret is gone" -- confirm it was actually masked in
    # place rather than the whole header being silently dropped.
    assert "Authorization: Basic ***" in redacted


def test_redact_sensitive_text_masks_assignment_style_secrets() -> None:
    text = "api_key=sk-super-secret timeout=30"
    redacted = redact_sensitive_text(text)

    assert "sk-super-secret" not in redacted
    assert "api_key=***" in redacted
    assert "timeout=30" in redacted


def test_redact_sensitive_text_masks_prefixed_assignment_keys() -> None:
    text = (
        "MCP_API_KEY=SECRET-abc123 rejected; "
        "SERVICE_ACCESS_TOKEN=tok-987654 expired; "
        "DB_PASSWORD=pw-secret-1 refused; "
        "x-client-secret=cs-000111 invalid"
    )
    redacted = redact_sensitive_text(text)

    assert "SECRET-abc123" not in redacted
    assert "tok-987654" not in redacted
    assert "pw-secret-1" not in redacted
    assert "cs-000111" not in redacted
    assert redacted == (
        "MCP_API_KEY=***c123 rejected; "
        "SERVICE_ACCESS_TOKEN=***7654 expired; "
        "DB_PASSWORD=***et-1 refused; "
        "x-client-secret=***0111 invalid"
    )


def test_redact_sensitive_text_masks_prefixed_secret_key_names() -> None:
    # An unknown ``*_key`` is treated as a credential (fail closed): the
    # common secret names below end in ``key`` without any other credential
    # word in front of it.
    text = (
        "AWS_SECRET_ACCESS_KEY=AKIA-secret-1 "
        "SECRET_KEY=django-secret-1 "
        "STRIPE_KEY=sk_live_1234567 "
        "PRIVATE_KEY=pem-private-1 "
        "x-client-key=ck-000111"
    )
    redacted = redact_sensitive_text(text)

    for raw in (
        "AKIA-secret-1",
        "django-secret-1",
        "sk_live_1234567",
        "pem-private-1",
        "ck-000111",
    ):
        assert raw not in redacted
    assert redacted == (
        "AWS_SECRET_ACCESS_KEY=***et-1 "
        "SECRET_KEY=***et-1 "
        "STRIPE_KEY=***4567 "
        "PRIVATE_KEY=***te-1 "
        "x-client-key=***0111"
    )


def test_redact_sensitive_text_leaves_non_credential_key_suffixes() -> None:
    # Only a ``*_key`` whose qualifier names a structural field is left
    # alone; this text reaches user- and model-facing error messages, so
    # these must stay readable. ``hotkey=`` / ``monkey=`` have no separate
    # ``key`` segment and are not assignments of a credential either.
    text = (
        "primary_key=42 sort_key=created_at partition_key=tenant "
        "PUBLIC_KEY=pem-body cache_key=user:42 idempotency_key=uuid-1 "
        "s3_key=path/to/obj routing_key=orders.new hotkey=ctrl-k monkey=1"
    )

    assert redact_sensitive_text(text) == text


# Each hyphen-joined structural qualifier paired with its underscore-joined
# counterpart. The exemption requires an underscore join, so these
# hyphen-joined names are masked; their underscore counterparts stay
# exempt.
_HYPHEN_QUALIFIER_REGRESSION_PAIRS = [
    ("primary-key", "primary_key"),
    ("foreign-key", "foreign_key"),
    ("unique-key", "unique_key"),
    ("composite-key", "composite_key"),
    ("index-key", "index_key"),
    ("sort-key", "sort_key"),
    ("partition-key", "partition_key"),
    ("range-key", "range_key"),
    ("hash-key", "hash_key"),
    ("lookup-key", "lookup_key"),
    ("cache-key", "cache_key"),
    ("routing-key", "routing_key"),
    ("idempotency-key", "idempotency_key"),
    ("public-key", "public_key"),
    ("object-key", "object_key"),
    ("s3-key", "s3_key"),
    ("idempotency-token", "idempotency_token"),
    ("next-token", "next_token"),
    ("page-token", "page_token"),
    ("continuation-token", "continuation_token"),
    ("cursor-token", "cursor_token"),
    ("sync-token", "sync_token"),
    ("pagination-token", "pagination_token"),
]


@pytest.mark.parametrize(
    ("hyphen_key", "underscore_key"), _HYPHEN_QUALIFIER_REGRESSION_PAIRS
)
def test_redact_sensitive_text_masks_hyphenated_structural_qualifiers(
    hyphen_key: str, underscore_key: str
) -> None:
    hyphen_text = f"{hyphen_key}=SECRET-abc123"
    underscore_text = f"{underscore_key}=SECRET-abc123"

    assert redact_sensitive_text(hyphen_text) == f"{hyphen_key}=***c123"
    assert redact_sensitive_text(underscore_text) == underscore_text


def test_redact_sensitive_text_still_masks_prefixed_key_despite_hyphen_gate() -> None:
    # The hyphen-vs-underscore gate above only governs the non-credential
    # qualifier exemption; it must not weaken the fail-closed prefixed-key
    # masking (#2304), which has no exemption at all.
    assert redact_sensitive_text("MCP_API_KEY=SECRET-abc123") == "MCP_API_KEY=***c123"


# ``*_token`` names that locate a page or de-duplicate a request; they are not
# credentials, and this text reaches user- and model-facing error messages.
# The exemption only fires when the qualifier is joined to ``token`` by
# ``_`` (see ``_is_credential_key``); a hyphen join is masked, so
# hyphenated names like ``x-idempotency-token`` and ``--page-token`` are
# not listed here even though their qualifier is the same word.
_POSITION_MARKER_TOKEN_KEYS = (
    "idempotency_token next_token page_token continuation_token cursor_token "
    "sync_token pagination_token next_page_token X_IDEMPOTENCY_TOKEN"
).split()
# Credential ``*_token`` names (``page_access_token`` puts a position word in
# front of ``access_token``), the ``*_key`` family that must keep masking,
# every ``*_key`` qualifier that is not also a ``*_token`` one (``public_token``),
# and hyphen-joined position-marker names -- the exemption above only exempts
# an underscore join, so these are masked.
_CREDENTIAL_TOKEN_KEYS = (
    "token access_token refresh_token id_token session_token auth_token "
    "api_token bearer_token csrf_token oauth_token github_token "
    "MCP_ACCESS_TOKEN page_access_token client_token "
    "AWS_SECRET_ACCESS_KEY MCP_API_KEY api_key "
    "x-idempotency-token --page-token"
).split() + [
    f"{qualifier}_token"
    for qualifier in sorted(
        _NON_CREDENTIAL_KEY_QUALIFIERS - _NON_CREDENTIAL_TOKEN_QUALIFIERS
    )
]


@pytest.mark.parametrize("key", _POSITION_MARKER_TOKEN_KEYS)
def test_redact_sensitive_text_leaves_position_marker_tokens(key: str) -> None:
    text = f"{key}=opaque-cursor-123 rejected"

    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize("key", _CREDENTIAL_TOKEN_KEYS)
def test_redact_sensitive_text_masks_credential_tokens(key: str) -> None:
    text = f"{key}=SECRET-abc123 rejected"

    assert redact_sensitive_text(text) == f"{key}=***c123 rejected"


def test_non_credential_qualifiers_are_listed_per_suffix() -> None:
    # Each word here leaves ``<word>_token=`` unmasked. Before adding one, add
    # ``<word>_token`` to _POSITION_MARKER_TOKEN_KEYS and check that no
    # service uses that name for a credential.
    assert set(_NON_CREDENTIAL_QUALIFIERS_BY_SUFFIX) == {"key", "token"}
    assert _NON_CREDENTIAL_TOKEN_QUALIFIERS == {
        "idempotency",
        "next",
        "page",
        "continuation",
        "cursor",
        "sync",
        "pagination",
    }
    assert {f"{q}_token" for q in _NON_CREDENTIAL_TOKEN_QUALIFIERS} <= set(
        _POSITION_MARKER_TOKEN_KEYS
    )


def test_redact_sensitive_text_still_masks_bare_and_cli_style_keys() -> None:
    redacted = redact_sensitive_text("key=SECRET-bare --api-key=sk-cli-secret")

    assert redacted == "key=***bare --api-key=***cret"


def test_redact_sensitive_text_is_linear_on_hostile_identifier_runs() -> None:
    # A remote service controls this text. A long run of ``_``-joined
    # identifier segments with no ``=`` must fail once per run, not once per
    # segment split: a nested ``(prefix_)*word`` quantifier took seconds on
    # 20 kB of this input; the current pattern takes well under a
    # millisecond. The budget is loose on purpose so a slow CI worker cannot
    # trip it, while a quadratic regression (tens of seconds here) still does.
    hostile = "a_" * 50_000

    started = time.perf_counter()
    assert redact_sensitive_text(hostile) == hostile
    assert time.perf_counter() - started < 1.0


# Assignments that are not credentials, and characters that can sit between
# one of them and a credential right behind it: first the ones that do not
# end a value, then the three that do. The non-credential assignment must
# not take its value with it, or the credential is never looked at.
_NON_CREDENTIAL_ASSIGNMENTS = (
    "host=example.com",
    "Username=app",
    "error=invalid_request",
    "code=400",
    "primary_key=42",
    "next_token=CUR",
    "page_token=p1",
)
_ASSIGNMENT_SEPARATORS = tuple(";,|\"':/)]}=") + ("\t", "&", " ")
_TRAILING_CREDENTIAL_KEYS = (
    "api_key",
    "access_token",
    "Password",
    "token",
    "MCP_API_KEY",
)


@pytest.mark.parametrize("separator", _ASSIGNMENT_SEPARATORS, ids=repr)
def test_redact_sensitive_text_masks_a_credential_after_a_non_credential_one(
    separator: str,
) -> None:
    for prefix in _NON_CREDENTIAL_ASSIGNMENTS:
        for key in _TRAILING_CREDENTIAL_KEYS:
            text = f"{prefix}{separator}{key}=SECRET-abc123"

            assert redact_sensitive_text(text) == (
                f"{prefix}{separator}{key}=***c123"
            ), text


def test_redact_sensitive_text_masks_a_credential_value_as_one_piece() -> None:
    # A credential's value runs to the next ``&`` or whitespace, so it can
    # take a following field with it (#2356); a credential inside that value
    # is covered by the same mask and is not masked a second time.
    assert (
        redact_sensitive_text("Host=db;Username=app;Password=SECRET-pw;Database=x")
        == "Host=db;Username=app;Password=***se=x"
    )
    assert (
        redact_sensitive_text("api_key=SECRET-abc123;token=SECRET-def456 rest")
        == "api_key=***f456 rest"
    )


@pytest.mark.parametrize(
    "hostile",
    ["a=" * 50_000, "a=b;" * 25_000, "x_" * 50_000 + "=v", "api_key=" * 12_500],
    ids=["bare-assignments", "semicolon-assignments", "one-long-key", "credentials"],
)
def test_redact_sensitive_text_is_linear_on_hostile_assignment_runs(
    hostile: str,
) -> None:
    # 100 kB of back-to-back ``identifier=`` from a remote service. The scan
    # resumes right after a non-credential's ``=`` and reads a value only for
    # a credential, so this takes milliseconds; re-reading each value to the
    # end of the text after every identifier takes seconds here. The budget
    # is loose so a slow CI worker cannot trip it.
    started = time.perf_counter()
    redact_sensitive_text(hostile)
    assert time.perf_counter() - started < 1.0


def test_redact_sensitive_text_leaves_a_credential_without_a_value() -> None:
    # A credential name followed directly by ``&``, whitespace or the end of
    # the text has no value to mask; the text after it is still scanned.
    assert (
        redact_sensitive_text("api_key=&token=SECRET-abc123")
        == "api_key=&token=***c123"
    )
    assert (
        redact_sensitive_text("missing api_key= in request")
        == "missing api_key= in request"
    )
    assert redact_sensitive_text("host=db;api_key=") == "host=db;api_key="
