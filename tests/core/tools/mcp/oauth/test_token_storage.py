import pytest
from mcp.shared.auth import OAuthToken

from xagent.core.tools.core.mcp.oauth.token_storage import DBTokenStorage


@pytest.mark.asyncio
async def test_roundtrip_tokens(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(user_id=user_id, mcpserver_id=server_id, db=db_session)
    assert await store.get_tokens() is None
    await store.set_tokens(
        OAuthToken(access_token="AT", token_type="Bearer", refresh_token="RT", expires_in=3600)
    )
    got = await store.get_tokens()
    assert got is not None
    assert got.access_token == "AT"
    assert got.refresh_token == "RT"


@pytest.mark.asyncio
async def test_tokens_encrypted_at_rest(db_session, seed_user_and_server):
    from xagent.web.models.mcp_oauth import MCPUserOAuthToken
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(user_id=user_id, mcpserver_id=server_id, db=db_session)
    await store.set_tokens(OAuthToken(access_token="secret-at", token_type="Bearer"))
    row = db_session.query(MCPUserOAuthToken).filter_by(user_id=user_id, mcpserver_id=server_id).one()
    assert row.access_token != "secret-at"  # stored encrypted
