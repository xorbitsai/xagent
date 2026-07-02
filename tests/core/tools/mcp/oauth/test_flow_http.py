import json

import httpx
import pytest

from xagent.core.tools.core.mcp.oauth.flow import (
    build_authorization_url,
    discover_auth_server,
    exchange_code_for_tokens,
    register_client_dcr,
)

AS_META = {
    "issuer": "https://auth.example",
    "authorization_endpoint": "https://auth.example/authorize",
    "token_endpoint": "https://auth.example/token",
    "registration_endpoint": "https://auth.example/register",
}


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _bypass_ssrf_guard(monkeypatch):
    """These are pure HTTP-layer unit tests against ``httpx.MockTransport``
    (no real network I/O) using ``.example`` placeholder hosts. The SSRF
    guard added in ``flow.py`` does a real DNS lookup before each request,
    which these tests should not depend on -- bypass it here and exercise it
    separately (see ``test_ssrf_guard.py`` and the
    ``test_*_invokes_ssrf_guard`` tests below).
    """
    monkeypatch.setattr(
        "xagent.core.tools.core.mcp.oauth.flow.assert_public_endpoint",
        lambda url: _noop(),
    )


async def _noop():
    return None


@pytest.mark.asyncio
async def test_discover_auth_server():
    def handler(request):
        if request.url.path == "/.well-known/oauth-protected-resource/notion":
            return httpx.Response(
                200, json={"authorization_servers": ["https://auth.example"]}
            )
        if request.url.path == "/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=AS_META)
        return httpx.Response(404)

    async with _client(handler) as c:
        meta = await discover_auth_server("https://mcp.example/notion", client=c)
    assert meta["token_endpoint"] == "https://auth.example/token"


@pytest.mark.asyncio
async def test_register_client_dcr():
    metadata = {
        "client_name": "xagent",
        "redirect_uris": ["https://app/cb"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
    }

    def handler(request):
        assert request.url.path.endswith("/register")
        assert json.loads(request.content) == metadata
        return httpx.Response(201, json={"client_id": "cid", "client_secret": "csec"})

    async with _client(handler) as c:
        info = await register_client_dcr(AS_META, metadata, client=c)
    assert info["client_id"] == "cid"


@pytest.mark.asyncio
async def test_exchange_code():
    def handler(request):
        assert request.url.path.endswith("/token")
        return httpx.Response(
            200, json={"access_token": "AT", "token_type": "Bearer", "expires_in": 3600}
        )

    async with _client(handler) as c:
        tok = await exchange_code_for_tokens(
            AS_META,
            client_id="cid",
            client_secret="csec",
            code="abc",
            code_verifier="v",
            redirect_uri="https://app/cb",
            client=c,
        )
    assert tok["access_token"] == "AT"


@pytest.mark.asyncio
async def test_discover_auth_server_invokes_ssrf_guard(monkeypatch):
    """Confirm discover_auth_server actually calls the SSRF guard (rather
    than proceeding straight to the network) by making the guard raise.
    """
    from xagent.core.tools.core.mcp.oauth.ssrf_guard import UnsafeOAuthEndpointError

    async def blow_up(url):
        raise UnsafeOAuthEndpointError(f"blocked: {url}")

    monkeypatch.setattr(
        "xagent.core.tools.core.mcp.oauth.flow.assert_public_endpoint", blow_up
    )

    def handler(request):
        raise AssertionError("must not reach the network when the guard rejects")

    async with _client(handler) as c:
        with pytest.raises(UnsafeOAuthEndpointError):
            await discover_auth_server("https://mcp.example/notion", client=c)


@pytest.mark.asyncio
async def test_register_client_dcr_invokes_ssrf_guard(monkeypatch):
    from xagent.core.tools.core.mcp.oauth.ssrf_guard import UnsafeOAuthEndpointError

    async def blow_up(url):
        raise UnsafeOAuthEndpointError(f"blocked: {url}")

    monkeypatch.setattr(
        "xagent.core.tools.core.mcp.oauth.flow.assert_public_endpoint", blow_up
    )

    def handler(request):
        raise AssertionError("must not reach the network when the guard rejects")

    async with _client(handler) as c:
        with pytest.raises(UnsafeOAuthEndpointError):
            await register_client_dcr(AS_META, {"client_name": "xagent"}, client=c)


@pytest.mark.asyncio
async def test_exchange_code_invokes_ssrf_guard(monkeypatch):
    from xagent.core.tools.core.mcp.oauth.ssrf_guard import UnsafeOAuthEndpointError

    async def blow_up(url):
        raise UnsafeOAuthEndpointError(f"blocked: {url}")

    monkeypatch.setattr(
        "xagent.core.tools.core.mcp.oauth.flow.assert_public_endpoint", blow_up
    )

    def handler(request):
        raise AssertionError("must not reach the network when the guard rejects")

    async with _client(handler) as c:
        with pytest.raises(UnsafeOAuthEndpointError):
            await exchange_code_for_tokens(
                AS_META,
                client_id="cid",
                client_secret="csec",
                code="abc",
                code_verifier="v",
                redirect_uri="https://app/cb",
                client=c,
            )


def test_build_authorization_url_missing_endpoint_raises():
    with pytest.raises(ValueError):
        build_authorization_url(
            {
                "token_endpoint": "https://auth.example/token"
            },  # no authorization_endpoint
            client_id="cid",
            redirect_uri="https://app/cb",
            code_challenge="chal",
            state="st",
        )


@pytest.mark.asyncio
async def test_discover_auth_server_passes_through_scopes_supported():
    """When the discovered authorization-server metadata advertises
    ``scopes_supported``, discover_auth_server must return it unchanged so
    callers (connect()) can build an authorization URL that requests those
    scopes, instead of always falling back to no/default scope.
    """
    meta_with_scopes = dict(AS_META, scopes_supported=["read", "write"])

    def handler(request):
        if request.url.path == "/.well-known/oauth-protected-resource/notion":
            return httpx.Response(
                200, json={"authorization_servers": ["https://auth.example"]}
            )
        if request.url.path == "/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=meta_with_scopes)
        return httpx.Response(404)

    async with _client(handler) as c:
        meta = await discover_auth_server("https://mcp.example/notion", client=c)

    assert meta["scopes_supported"] == ["read", "write"]


@pytest.mark.asyncio
async def test_discover_auth_server_fallback_has_no_invented_scopes():
    """The synthesized-fallback metadata (used when no well-known discovery
    document responds) must not invent a scopes_supported the server never
    published.
    """

    def handler(request):
        return httpx.Response(404)

    async with _client(handler) as c:
        meta = await discover_auth_server("https://mcp.example/notion", client=c)

    assert "scopes_supported" not in meta


def test_build_authorization_url_includes_scope_when_provided():
    url = build_authorization_url(
        AS_META,
        client_id="cid",
        redirect_uri="https://app/cb",
        code_challenge="chal",
        state="st",
        scope="read write",
    )
    assert "scope=read+write" in url or "scope=read%20write" in url


@pytest.mark.asyncio
async def test_discover_auth_server_path_scoped_server():
    """Both the protected resource and its authorization server live at a
    non-root path (e.g. a multi-tenant gateway distinguishing tenants by
    path). Per RFC 9728 / RFC 8414, the well-known segment must be inserted
    right after the origin, with the resource/issuer's own path preserved
    (protected-resource) or appended (authorization-server) after it --
    NOT root-anchored in a way that drops the path.
    """
    seen_paths = []

    def handler(request):
        seen_paths.append(request.url.path)
        if request.url.path == "/.well-known/oauth-protected-resource/tenant-a":
            return httpx.Response(
                200,
                json={"authorization_servers": ["https://mcp.example/tenant-a"]},
            )
        if request.url.path == "/.well-known/oauth-authorization-server/tenant-a":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://mcp.example/tenant-a",
                    "authorization_endpoint": "https://mcp.example/tenant-a/authorize",
                    "token_endpoint": "https://mcp.example/tenant-a/token",
                    "registration_endpoint": "https://mcp.example/tenant-a/register",
                },
            )
        return httpx.Response(404)

    async with _client(handler) as c:
        meta = await discover_auth_server("https://mcp.example/tenant-a", client=c)

    assert meta["token_endpoint"] == "https://mcp.example/tenant-a/token"
    # Confirm the exact well-known paths were hit (not the old, root-anchored
    # /.well-known/oauth-protected-resource with the "/tenant-a" path dropped).
    assert "/.well-known/oauth-protected-resource/tenant-a" in seen_paths
    assert "/.well-known/oauth-authorization-server/tenant-a" in seen_paths


def test_build_authorization_url_omits_scope_when_absent():
    url = build_authorization_url(
        AS_META,
        client_id="cid",
        redirect_uri="https://app/cb",
        code_challenge="chal",
        state="st",
    )
    assert "scope=" not in url
