import gzip
import ssl
from pathlib import Path

import httpx
import pytest

from xagent.web.services import mcp_oauth as mcp_oauth_service
from xagent.web.services.mcp_oauth import (
    MCP_OAUTH_HTTP_TIMEOUT_SECONDS,
    MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
    MCPOAuthDiscoveryError,
    OAuthAuthorizationServerMetadata,
    SafeOAuthAsyncHTTPTransport,
    _same_url,
    authorization_server_metadata_urls,
    discover_mcp_oauth_metadata,
    oauth_get,
    oauth_post,
    parse_www_authenticate_bearer,
    protected_resource_metadata_urls,
    register_mcp_oauth_public_client,
    validate_oauth_http_url,
)


def test_parse_www_authenticate_bearer_challenge():
    challenge = parse_www_authenticate_bearer(
        'Basic realm="ignored", Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="records.read records.write"'
    )

    assert challenge is not None
    assert (
        challenge.resource_metadata_url
        == "https://mcp.example.com/.well-known/oauth-protected-resource"
    )
    assert challenge.scope == "records.read records.write"


def test_parse_www_authenticate_bearer_ignores_malformed_parameters(monkeypatch):
    def raise_malformed(_value):
        raise ValueError("malformed")

    monkeypatch.setattr(mcp_oauth_service, "parse_http_list", raise_malformed)

    assert parse_www_authenticate_bearer("Bearer malformed") is None


def test_protected_resource_metadata_urls_use_endpoint_path_before_root():
    assert protected_resource_metadata_urls("https://mcp.example.com/public/mcp") == (
        "https://mcp.example.com/.well-known/oauth-protected-resource/public/mcp",
        "https://mcp.example.com/.well-known/oauth-protected-resource",
    )


def test_authorization_server_metadata_urls_for_path_issuer():
    assert authorization_server_metadata_urls("https://auth.example.com/org1") == (
        "https://auth.example.com/.well-known/oauth-authorization-server/org1",
        "https://auth.example.com/.well-known/openid-configuration/org1",
        "https://auth.example.com/org1/.well-known/openid-configuration",
    )


def test_url_comparison_normalizes_default_ports():
    assert _same_url("https://auth.example.com:443", "https://AUTH.example.com")
    assert _same_url("http://auth.example.com:80/path/", "http://auth.example.com/path")
    assert not _same_url("https://auth.example.com:8443", "https://auth.example.com")


@pytest.mark.asyncio
async def test_oauth_url_dns_resolution_runs_in_executor(monkeypatch):
    calls: list[tuple[object, object, tuple[object, ...]]] = []

    class FakeLoop:
        async def run_in_executor(self, executor, func, *args):
            calls.append((executor, func, args))
            return [
                (
                    0,
                    0,
                    0,
                    "",
                    ("93.184.216.34", 443),
                )
            ]

    monkeypatch.setattr(
        mcp_oauth_service.asyncio,
        "get_running_loop",
        lambda: FakeLoop(),
    )

    await validate_oauth_http_url("https://auth.example.com/token", resolve_dns=True)

    assert len(calls) == 1
    assert calls[0][1] is mcp_oauth_service.socket.getaddrinfo


@pytest.mark.asyncio
async def test_oauth_url_policy_rejects_invalid_port_without_500():
    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        await validate_oauth_http_url(
            "https://auth.example.com:not-a-port/token",
            resolve_dns=False,
        )

    assert exc.value.code == "invalid_resource"
    assert "invalid port" in exc.value.message


@pytest.mark.asyncio
async def test_oauth_url_policy_rejects_localhost_by_default(monkeypatch):
    monkeypatch.delenv("XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS", raising=False)

    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        await validate_oauth_http_url("http://localhost:8080/token", resolve_dns=False)

    assert exc.value.code == "invalid_resource"


@pytest.mark.asyncio
async def test_oauth_url_policy_allows_localhost_when_explicitly_configured(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS", "true")

    await validate_oauth_http_url("http://localhost:8080/token", resolve_dns=False)


@pytest.mark.asyncio
async def test_oauth_url_policy_allows_private_resolved_ip_when_explicitly_configured(
    monkeypatch,
):
    calls: list[tuple[object, object, tuple[object, ...]]] = []

    class FakeLoop:
        async def run_in_executor(self, executor, func, *args):
            calls.append((executor, func, args))
            return [
                (
                    0,
                    0,
                    0,
                    "",
                    ("127.0.0.1", 8080),
                )
            ]

    monkeypatch.setenv("XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS", "true")
    monkeypatch.setattr(
        mcp_oauth_service.asyncio,
        "get_running_loop",
        lambda: FakeLoop(),
    )

    await validate_oauth_http_url("http://auth.example.com/token", resolve_dns=True)

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_oauth_url_policy_still_rejects_userinfo_when_private_hosts_allowed(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS", "true")

    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        await validate_oauth_http_url(
            "http://user:password@localhost:8080/token",
            resolve_dns=False,
        )

    assert exc.value.code == "invalid_resource"
    assert "userinfo" in exc.value.message


@pytest.mark.asyncio
async def test_safe_oauth_transport_pins_resolved_ip_and_preserves_host(monkeypatch):
    captured: list[tuple[str, str, str | None]] = []

    class CaptureTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured.append(
                (
                    str(request.url),
                    request.headers["Host"],
                    request.extensions.get("sni_hostname"),
                )
            )
            return httpx.Response(200, json={"ok": True}, request=request)

        async def aclose(self) -> None:
            return None

    async def fake_resolve(value: str) -> list[str]:
        assert value == "https://auth.example.com/token"
        return ["93.184.216.34"]

    monkeypatch.setattr(
        mcp_oauth_service,
        "_resolve_allowed_addresses",
        fake_resolve,
    )
    transport = SafeOAuthAsyncHTTPTransport()
    transport._transport = CaptureTransport()

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://auth.example.com/token")

    assert response.status_code == 200
    assert captured[0] == (
        "https://93.184.216.34/token",
        "auth.example.com",
        "auth.example.com",
    )
    assert str(response.request.url) == "https://auth.example.com/token"


@pytest.mark.asyncio
async def test_safe_oauth_transport_formats_resolved_ipv6_url(monkeypatch):
    captured: list[str] = []

    class CaptureTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured.append(str(request.url))
            return httpx.Response(200, json={"ok": True}, request=request)

        async def aclose(self) -> None:
            return None

    async def fake_resolve(value: str) -> list[str]:
        assert value == "https://auth.example.com/token"
        return ["2001:db8::1"]

    monkeypatch.setattr(
        mcp_oauth_service,
        "_resolve_allowed_addresses",
        fake_resolve,
    )
    transport = SafeOAuthAsyncHTTPTransport()
    transport._transport = CaptureTransport()

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://auth.example.com/token")

    assert response.status_code == 200
    assert captured == ["https://[2001:db8::1]/token"]


@pytest.mark.asyncio
async def test_safe_oauth_transport_falls_back_within_validated_addresses(monkeypatch):
    requested_urls: list[str] = []

    class CaptureTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            if len(requested_urls) == 1:
                raise httpx.ConnectError("first address unavailable", request=request)
            return httpx.Response(200, json={"ok": True}, request=request)

        async def aclose(self) -> None:
            return None

    async def fake_resolve(value: str) -> list[str]:
        assert value == "https://auth.example.com/token"
        return ["2001:db8::1", "93.184.216.34"]

    monkeypatch.setattr(
        mcp_oauth_service,
        "_resolve_allowed_addresses",
        fake_resolve,
    )
    transport = SafeOAuthAsyncHTTPTransport()
    transport._transport = CaptureTransport()

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://auth.example.com/token")

    assert response.status_code == 200
    assert requested_urls == [
        "https://[2001:db8::1]/token",
        "https://93.184.216.34/token",
    ]
    assert str(response.request.url) == "https://auth.example.com/token"


def test_safe_oauth_transport_disables_proxy_http2_and_keepalive(monkeypatch):
    captured: dict[str, object] = {}

    class CaptureTransport(httpx.AsyncBaseTransport):
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncHTTPTransport", CaptureTransport)
    monkeypatch.delenv("XAGENT_MCP_OAUTH_PROXY_URL", raising=False)
    # This constructs the real build_ca_bundle_ssl_context() (only
    # AsyncHTTPTransport is mocked); an ambient SSL_CERT_FILE/SSL_CERT_DIR
    # left over from the environment could otherwise make this fail on a
    # value unrelated to what's under test here.
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    SafeOAuthAsyncHTTPTransport()

    assert captured["trust_env"] is False
    assert captured["proxy"] is None
    assert captured["http2"] is False
    assert captured["limits"].max_keepalive_connections == 0
    assert isinstance(captured["verify"], ssl.SSLContext)


def test_safe_oauth_transport_uses_explicit_proxy_configuration(monkeypatch):
    # A plain http:// proxy has no TLS leg of its own -- httpcore rejects a
    # non-None proxy_ssl_context outright for this scheme (RuntimeError: "The
    # proxy_ssl_context argument is not allowed for the http scheme"), so the
    # proxy must stay a bare string/URL, not an httpx.Proxy carrying one.
    captured: dict[str, object] = {}

    class CaptureTransport(httpx.AsyncBaseTransport):
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncHTTPTransport", CaptureTransport)
    monkeypatch.setenv("XAGENT_MCP_OAUTH_PROXY_URL", "http://proxy.example.com:8080")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    SafeOAuthAsyncHTTPTransport()

    assert captured["trust_env"] is False
    assert captured["proxy"] == "http://proxy.example.com:8080"
    assert isinstance(captured["verify"], ssl.SSLContext)


def test_safe_oauth_transport_attaches_ca_bundle_to_https_proxy_leg(monkeypatch):
    captured: dict[str, object] = {}

    class CaptureTransport(httpx.AsyncBaseTransport):
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(mcp_oauth_service.httpx, "AsyncHTTPTransport", CaptureTransport)
    monkeypatch.setenv("XAGENT_MCP_OAUTH_PROXY_URL", "https://proxy.example.com:8443")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    SafeOAuthAsyncHTTPTransport()

    assert captured["trust_env"] is False
    assert isinstance(captured["proxy"], httpx.Proxy)
    assert captured["proxy"].url == httpx.URL("https://proxy.example.com:8443")
    # The proxy's own CONNECT/TLS leg must trust the same isolated CA bundle
    # as the tunneled target origin, not fall back to httpcore's
    # certifi-only default (see build_ca_bundle_ssl_context()'s docstring).
    assert captured["proxy"].ssl_context is captured["verify"]
    assert isinstance(captured["verify"], ssl.SSLContext)


@pytest.mark.parametrize(
    "proxy_url",
    ["http://proxy.example.com:8080", "https://proxy.example.com:8443", None],
)
def test_safe_oauth_transport_constructs_against_real_httpx_transport(
    monkeypatch, proxy_url
):
    # Regression test for a real bug: mocking httpx.AsyncHTTPTransport (as
    # the tests above do, to inspect kwargs) hid that httpcore raises
    # RuntimeError when a plain http:// proxy is given a non-None
    # proxy_ssl_context. Construct against the *real* AsyncHTTPTransport so
    # httpcore's own scheme validation actually runs.
    if proxy_url is None:
        monkeypatch.delenv("XAGENT_MCP_OAUTH_PROXY_URL", raising=False)
    else:
        monkeypatch.setenv("XAGENT_MCP_OAUTH_PROXY_URL", proxy_url)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    transport = SafeOAuthAsyncHTTPTransport()

    # Prove a real, non-mocked connection pool was actually built (not just
    # that construction didn't raise, which pytest would already catch).
    assert isinstance(transport._transport, httpx.AsyncHTTPTransport)


def test_safe_oauth_transport_reports_bad_ca_bundle_as_discovery_error(monkeypatch):
    # SSL_CERT_FILE/SSL_CERT_DIR were never read on this path before this
    # transport started verifying against an explicit context, so a stale
    # misconfigured value used to be harmless. A FileNotFoundError (an
    # OSError subclass, same as ssl.SSLError) from building that context
    # must surface as a structured MCPOAuthDiscoveryError, not propagate as
    # a raw OSError past every caller's `except MCPOAuthDiscoveryError`.
    def _broken_ssl_context():
        raise FileNotFoundError("no such CA bundle")

    monkeypatch.setattr(
        mcp_oauth_service, "build_ca_bundle_ssl_context", _broken_ssl_context
    )
    monkeypatch.setenv("SSL_CERT_FILE", "/no/such/file.pem")
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        SafeOAuthAsyncHTTPTransport()

    assert exc.value.code == "invalid_configuration"
    assert "SSL_CERT_FILE" in exc.value.message
    assert "SSL_CERT_DIR" not in exc.value.message


def test_safe_oauth_transport_does_not_blame_env_vars_that_are_unset(monkeypatch):
    # A build_ca_bundle_ssl_context() failure can also come from the
    # certifi-backed default path (neither env var set) -- e.g. a
    # broken/partial certifi install. Don't misattribute that to
    # SSL_CERT_FILE/SSL_CERT_DIR when neither is actually configured.
    def _broken_ssl_context():
        raise FileNotFoundError("certifi bundle missing")

    monkeypatch.setattr(
        mcp_oauth_service, "build_ca_bundle_ssl_context", _broken_ssl_context
    )
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        SafeOAuthAsyncHTTPTransport()

    assert exc.value.code == "invalid_configuration"
    assert "SSL_CERT_FILE" not in exc.value.message
    assert "SSL_CERT_DIR" not in exc.value.message


def test_safe_oauth_transport_rejects_ssl_cert_dir_that_loads_no_certs(
    monkeypatch, tmp_path
):
    # ssl.create_default_context(capath=...) does NOT raise for a directory
    # that doesn't exist or contains no certs (unlike SSL_CERT_FILE, which
    # does raise for a missing/empty file) -- it silently returns a context
    # with zero trusted CAs. Left unguarded, this transport would build
    # "successfully" with an effectively empty trust store, and every TLS
    # request through it would fail later with a generic httpx.ConnectError
    # instead of this structured, actionable error.
    empty_dir = tmp_path / "empty-ca-dir"
    empty_dir.mkdir()
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("SSL_CERT_DIR", str(empty_dir))

    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        SafeOAuthAsyncHTTPTransport()

    assert exc.value.code == "invalid_configuration"
    assert "SSL_CERT_DIR" in exc.value.message


@pytest.mark.asyncio
async def test_oauth_get_revalidates_redirects_and_blocks_private_target():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"Location": "http://169.254.169.254/latest/meta-data"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_get(
                "https://auth.example.com/.well-known/oauth-authorization-server",
                client=client,
            )

    assert exc.value.code == "invalid_resource"
    assert requested_urls == [
        "https://auth.example.com/.well-known/oauth-authorization-server"
    ]


@pytest.mark.asyncio
async def test_oauth_get_strips_sensitive_headers_on_cross_origin_redirect():
    captured_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append(dict(request.headers))
        if str(request.url) == "https://auth.example.com/start":
            return httpx.Response(
                302,
                headers={"Location": "https://login.example.com/metadata"},
            )
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await oauth_get(
            "https://auth.example.com/start",
            client=client,
            headers={
                "Authorization": "Bearer secret",
                "Cookie": "session=secret",
                "X-Trace": "trace-id",
            },
        )

    assert response.status_code == 200
    assert captured_headers[0]["authorization"] == "Bearer secret"
    assert captured_headers[0]["cookie"] == "session=secret"
    assert "authorization" not in captured_headers[1]
    assert "cookie" not in captured_headers[1]
    assert captured_headers[1]["x-trace"] == "trace-id"


@pytest.mark.asyncio
async def test_oauth_get_rejects_missing_redirect_location():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_get("https://auth.example.com/start", client=client)

    assert exc.value.code == "metadata_not_found"


@pytest.mark.asyncio
async def test_oauth_get_rejects_malformed_redirect_location():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://[::1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_get("https://auth.example.com/start", client=client)

    assert exc.value.code == "metadata_not_found"


@pytest.mark.asyncio
async def test_oauth_get_sanitizes_response_protocol_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("secret response detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_get("https://auth.example.com/metadata", client=client)

    assert exc.value.code == "metadata_not_found"
    assert exc.value.message == "OAuth metadata response could not be read"
    assert "secret" not in exc.value.message


@pytest.mark.asyncio
async def test_oauth_post_rejects_redirect_without_following():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://auth.example.com/new"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_post("https://auth.example.com/token", client=client, data={})

    assert exc.value.code == "invalid_resource"
    assert requested_urls == ["https://auth.example.com/token"]


@pytest.mark.asyncio
async def test_oauth_post_sanitizes_response_protocol_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("secret response detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_post(
                "https://auth.example.com/register",
                client=client,
                max_response_bytes=1024,
                json={"client_name": "Xagent"},
            )

    assert exc.value.code == "invalid_resource"
    assert exc.value.message == "OAuth endpoint response could not be read"
    assert "secret" not in exc.value.message


@pytest.mark.asyncio
async def test_oauth_post_rejects_response_larger_than_streaming_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"response-too-large")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_post(
                "https://auth.example.com/register",
                client=client,
                max_response_bytes=8,
                json={"client_name": "Xagent"},
            )

    assert exc.value.code == "response_too_large"


@pytest.mark.asyncio
async def test_oauth_post_rejects_compressed_response_without_decoding():
    # The wire payload is tiny and its *decoded* size stays comfortably
    # under max_response_bytes, so a decoded-size check alone would let this
    # through. The rejection must come from the Content-Encoding check
    # firing before any decompression happens -- if that check is removed,
    # this response decodes cleanly under the cap and the call would
    # succeed instead of raising, which is what proves the fix is load
    # bearing (a decoded-size-only guard cannot catch this case).
    payload = b"0" * (1024 * 1024)
    compressed = gzip.compress(payload)
    assert len(compressed) < len(payload) / 100

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=compressed,
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_post(
                "https://auth.example.com/register",
                client=client,
                max_response_bytes=2 * 1024 * 1024,
                json={"client_name": "Xagent"},
            )

    assert exc.value.code == "unsupported_response_encoding"


@pytest.mark.asyncio
async def test_oauth_post_refuses_encoded_body_before_reading_any_of_it():
    # The ordering is the actual bomb defense: the refusal must fire from the
    # headers alone, before a single body byte is pulled (and decompressed).
    # A counting stream makes the guarantee mechanical -- if the encoding
    # check moves after the read loop, reads becomes nonzero and this fails
    # even though the decoded size would still be under the cap.
    reads = {"count": 0}

    class _CountingStream(httpx.AsyncByteStream):
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        async def __aiter__(self):
            for chunk in self._chunks:
                reads["count"] += 1
                yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_CountingStream([gzip.compress(b"0" * 4096)]),
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await oauth_post(
                "https://auth.example.com/register",
                client=client,
                max_response_bytes=1024 * 1024,
                json={"client_name": "Xagent"},
            )

    assert exc.value.code == "unsupported_response_encoding"
    assert reads["count"] == 0


@pytest.mark.asyncio
async def test_oauth_post_accepts_identity_encoded_response_under_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"client_id":"dynamic-client"}',
            headers={
                "Content-Encoding": "identity",
                "Content-Type": "application/json",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await oauth_post(
            "https://auth.example.com/register",
            client=client,
            max_response_bytes=1024,
            json={"client_name": "Xagent"},
        )

    assert response.json() == {"client_id": "dynamic-client"}


@pytest.mark.asyncio
async def test_oauth_post_bounded_requests_ask_for_identity_encoding():
    # httpx advertises gzip/br/zstd by default; a compliant server honoring
    # that would then be refused by the response-side encoding guard. A
    # bounded read must therefore ask for identity itself -- the caller
    # should not have to know to do it.
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            content=b'{"client_id":"dynamic-client"}',
            headers={"Content-Type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await oauth_post(
            "https://auth.example.com/register",
            client=client,
            max_response_bytes=1024,
            json={"client_name": "Xagent"},
        )

    assert seen[0].headers["accept-encoding"] == "identity"


@pytest.mark.asyncio
async def test_oauth_post_strips_framing_headers_from_buffered_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            content=b'{"client_id":"dynamic-client"}',
            headers={
                "Content-Type": "application/json",
                "Content-Length": "999",
                "Transfer-Encoding": "chunked",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await oauth_post(
            "https://auth.example.com/register",
            client=client,
            max_response_bytes=1024,
            json={"client_name": "Xagent"},
        )

    assert response.json() == {"client_id": "dynamic-client"}
    assert response.headers["content-length"] == str(len(response.content))
    assert "transfer-encoding" not in response.headers


@pytest.mark.asyncio
async def test_dynamic_registration_rejects_confidential_client_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "client_id": "confidential-client",
                "client_secret": "must-not-be-used",
                "token_endpoint_auth_method": "client_secret_post",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        mcp_oauth_service,
        "create_mcp_oauth_http_client",
        lambda **kwargs: client,
    )
    metadata = OAuthAuthorizationServerMetadata(
        url="https://auth.example.com/.well-known/oauth-authorization-server",
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        registration_endpoint="https://auth.example.com/register",
        client_id_metadata_document_supported=True,
        raw={},
    )

    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        await register_mcp_oauth_public_client(
            metadata,
            redirect_uri="https://api.xagent.test/api/mcp/oauth/callback",
        )

    assert exc.value.code == "client_registration_failed"
    assert "public client" in exc.value.message


def test_owned_oauth_paths_use_shared_helpers_without_redirect_following():
    service_source = Path(mcp_oauth_service.__file__).read_text()
    api_source = (
        Path(mcp_oauth_service.__file__)
        .parents[1]
        .joinpath("api", "mcp.py")
        .read_text()
    )

    assert "follow_redirects=True" not in service_source
    assert "follow_redirects=True" not in api_source
    assert "timeout=10.0" not in service_source
    assert "timeout=10.0" not in api_source
    assert "response = await client.get(endpoint_url" not in service_source
    assert "response = await client.get(metadata_url" not in service_source
    assert (
        "response = await client.post(\n                str(oauth_client.token_endpoint)"
        not in service_source
    )
    assert MCP_OAUTH_HTTP_TIMEOUT_SECONDS == 10.0


@pytest.mark.parametrize(
    "url",
    [
        "http://[fe80::1%en0]/metadata",
        "http://[fe80::1%25en0]/metadata",
    ],
)
@pytest.mark.asyncio
async def test_oauth_url_rejects_ipv6_zone_identifiers(url):
    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        await validate_oauth_http_url(url, resolve_dns=False)

    assert exc.value.code == "invalid_resource"


@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:127.0.0.1]/metadata",
        "http://[::ffff:169.254.169.254]/metadata",
        "http://[::ffff:10.0.0.1]/metadata",
    ],
)
@pytest.mark.asyncio
async def test_oauth_url_rejects_ipv4_mapped_ipv6_private_targets(url):
    with pytest.raises(MCPOAuthDiscoveryError) as exc:
        await validate_oauth_http_url(url, resolve_dns=False)

    assert exc.value.code == "invalid_resource"


@pytest.mark.asyncio
async def test_discover_rejects_loopback_endpoint_before_request():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "http://127.0.0.1:8500/mcp", client=client
            )

    assert exc.value.code == "invalid_resource"
    assert requested_urls == []


@pytest.mark.asyncio
async def test_discover_rejects_link_local_resource_metadata_before_request():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": 'Bearer resource_metadata="http://169.254.169.254/latest/meta-data"'
                },
            )
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp", client=client
            )

    assert exc.value.code == "invalid_resource"
    assert requested_urls == ["https://mcp.example.com/mcp"]


@pytest.mark.asyncio
async def test_discover_rejects_invalid_authorization_endpoint_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "javascript:alert(1)",
                    "token_endpoint": "https://auth.example.com/token",
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp", client=client
            )

    assert exc.value.code == "invalid_resource"


@pytest.mark.asyncio
async def test_discover_rejects_metadata_values_that_cannot_fit_persistence():
    oversized_issuer = "https://auth.example.com/" + (
        "x" * MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": oversized_issuer,
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp", client=client
            )

    assert exc.value.code == "invalid_resource"
    assert "issuer" in exc.value.message


@pytest.mark.asyncio
async def test_discover_uses_challenge_resource_metadata_and_scope():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="records.read"'
                },
            )
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ["records.read", "records.write"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                    "client_id_metadata_document_supported": True,
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_mcp_oauth_metadata(
            "https://mcp.example.com/mcp", client=client
        )

    assert result.resource == "https://mcp.example.com/mcp"
    assert result.scopes == ("records.read",)
    assert result.authorization_server.issuer == "https://auth.example.com"
    assert result.authorization_server.client_id_metadata_document_supported is True
    assert requested_urls[:3] == [
        "https://mcp.example.com/mcp",
        "https://mcp.example.com/.well-known/oauth-protected-resource",
        "https://auth.example.com/.well-known/oauth-authorization-server",
    ]


@pytest.mark.asyncio
async def test_discover_falls_back_to_well_known_resource_metadata():
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == "https://mcp.example.com/public/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/public/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/public/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ["records.read"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://AUTH.EXAMPLE.com:443/",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_mcp_oauth_metadata(
            "https://mcp.example.com/public/mcp", client=client
        )

    assert result.scopes == ("records.read",)
    assert (
        "https://mcp.example.com/.well-known/oauth-protected-resource/public/mcp"
        in requested_urls
    )


@pytest.mark.asyncio
async def test_discover_rejects_configured_resource_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp",
                configured_resource="https://other.example.com/mcp",
                client=client,
            )

    assert exc.value.code == "resource_mismatch"


@pytest.mark.asyncio
async def test_discover_rejects_configured_issuer_not_advertised():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp",
                configured_issuer="https://login.example.com",
                client=client,
            )

    assert exc.value.code == "issuer_mismatch"


@pytest.mark.asyncio
async def test_discover_accepts_case_variant_configured_resource_and_issuer():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={
                    "resource": "https://MCP.EXAMPLE.com:443/mcp/",
                    "authorization_servers": ["https://AUTH.EXAMPLE.com/"],
                },
            )
        if (
            str(request.url)
            == "https://auth.example.com/.well-known/oauth-authorization-server"
        ):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_mcp_oauth_metadata(
            "https://mcp.example.com/mcp",
            configured_resource="https://mcp.example.com/mcp/",
            configured_issuer="https://auth.example.com",
            client=client,
        )

    assert result.resource == "https://mcp.example.com/mcp"
    assert result.authorization_server.issuer == "https://auth.example.com"


@pytest.mark.asyncio
async def test_discover_rejects_missing_authorization_servers():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://mcp.example.com/mcp":
            return httpx.Response(401)
        if (
            str(request.url)
            == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx.Response(
                200,
                json={"resource": "https://mcp.example.com/mcp"},
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp", client=client
            )

    assert exc.value.code == "authorization_server_not_found"


@pytest.mark.asyncio
async def test_discover_surfaces_probe_network_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MCPOAuthDiscoveryError) as exc:
            await discover_mcp_oauth_metadata(
                "https://mcp.example.com/mcp", client=client
            )

    assert exc.value.code == "metadata_not_found"
