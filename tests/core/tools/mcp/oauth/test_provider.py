import pytest
from mcp.client.auth import OAuthClientProvider

from xagent.core.tools.core.mcp.oauth.errors import MCPReauthorizationRequired
from xagent.core.tools.core.mcp.oauth.provider import (
    build_execution_oauth_provider,
    make_reauth_handlers,
)


def test_build_returns_httpx_auth(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    provider = build_execution_oauth_provider(
        server_url="https://mcp.example/notion",
        server_name="notion",
        user_id=user_id,
        mcpserver_id=server_id,
        db=db_session,
    )
    assert isinstance(provider, OAuthClientProvider)


@pytest.mark.asyncio
async def test_reauth_handlers_raise():
    redirect, callback = make_reauth_handlers("notion", 7)
    with pytest.raises(MCPReauthorizationRequired):
        await redirect("https://auth.example/authorize")
    with pytest.raises(MCPReauthorizationRequired):
        await callback()
