import pytest
from mcp.client.auth import OAuthClientProvider

from xagent.core.tools.core.mcp.oauth.errors import MCPReauthorizationRequired
from xagent.core.tools.core.mcp.oauth.provider import (
    build_execution_oauth_provider,
    build_oauth_client_metadata,
    make_reauth_handlers,
)
from xagent.web.services.mcp_oauth_token_storage import DBTokenStorage


def test_build_returns_httpx_auth(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    storage = DBTokenStorage(user_id=user_id, mcpserver_id=server_id, db=db_session)
    provider = build_execution_oauth_provider(
        server_url="https://mcp.example/notion",
        server_name="notion",
        mcpserver_id=server_id,
        storage=storage,
    )
    assert isinstance(provider, OAuthClientProvider)


def test_oauth_callback_base_url_requires_scheme(monkeypatch):
    monkeypatch.setenv("XAGENT_OAUTH_CALLBACK_BASE_URL", "localhost:8000")

    with pytest.raises(ValueError, match="http\\(s\\) scheme and host"):
        build_oauth_client_metadata()


@pytest.mark.asyncio
async def test_reauth_handlers_raise():
    redirect, callback = make_reauth_handlers("notion", 7)
    with pytest.raises(MCPReauthorizationRequired):
        await redirect("https://auth.example/authorize")
    with pytest.raises(MCPReauthorizationRequired):
        await callback()
