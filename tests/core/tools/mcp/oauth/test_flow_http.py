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


@pytest.mark.asyncio
async def test_discover_auth_server():
    def handler(request):
        if request.url.path.endswith(".well-known/oauth-protected-resource"):
            return httpx.Response(200, json={"authorization_servers": ["https://auth.example"]})
        if request.url.path.endswith(".well-known/oauth-authorization-server"):
            return httpx.Response(200, json=AS_META)
        return httpx.Response(404)

    async with _client(handler) as c:
        meta = await discover_auth_server("https://mcp.example/notion", client=c)
    assert meta["token_endpoint"] == "https://auth.example/token"


@pytest.mark.asyncio
async def test_register_client_dcr():
    def handler(request):
        assert request.url.path.endswith("/register")
        return httpx.Response(201, json={"client_id": "cid", "client_secret": "csec"})

    async with _client(handler) as c:
        info = await register_client_dcr(AS_META, ["https://app/cb"], client=c)
    assert info["client_id"] == "cid"


@pytest.mark.asyncio
async def test_exchange_code():
    def handler(request):
        assert request.url.path.endswith("/token")
        return httpx.Response(200, json={"access_token": "AT", "token_type": "Bearer", "expires_in": 3600})

    async with _client(handler) as c:
        tok = await exchange_code_for_tokens(
            AS_META, client_id="cid", client_secret="csec",
            code="abc", code_verifier="v", redirect_uri="https://app/cb", client=c,
        )
    assert tok["access_token"] == "AT"


def test_build_authorization_url_missing_endpoint_raises():
    with pytest.raises(ValueError):
        build_authorization_url(
            {"token_endpoint": "https://auth.example/token"},  # no authorization_endpoint
            client_id="cid",
            redirect_uri="https://app/cb",
            code_challenge="chal",
            state="st",
        )
