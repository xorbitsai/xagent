import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from xagent.web.models.mcp import MCPServer
from xagent.web.models.mcp_oauth import MCPUserOAuthToken
from xagent.web.services.mcp_oauth_token_storage import DBTokenStorage


@pytest.mark.asyncio
async def test_roundtrip_tokens(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    assert await store.get_tokens() is None
    await store.set_tokens(
        OAuthToken(
            access_token="AT",
            token_type="Bearer",
            refresh_token="RT",
            expires_in=3600,
        )
    )
    got = await store.get_tokens()
    assert got is not None
    assert got.access_token == "AT"
    assert got.refresh_token == "RT"


@pytest.mark.asyncio
async def test_tokens_encrypted_at_rest(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(OAuthToken(access_token="secret-at", token_type="Bearer"))
    row = (
        db_session.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one()
    )
    assert row.access_token != "secret-at"  # stored encrypted


@pytest.mark.asyncio
async def test_missing_token_row_is_not_recreated(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(user_id=user_id, mcpserver_id=server_id, db=db_session)

    wrote = await store.set_tokens_if_row_exists(
        OAuthToken(access_token="stale-at", token_type="Bearer")
    )

    assert wrote is False
    assert (
        db_session.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one_or_none()
        is None
    )


@pytest.mark.asyncio
async def test_expired_tokens_without_refresh_are_not_returned(
    db_session, seed_user_and_server
):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(
        OAuthToken(access_token="AT", token_type="Bearer", expires_in=-60)
    )
    assert await store.get_tokens() is None


@pytest.mark.asyncio
async def test_expired_tokens_with_refresh_trigger_refresh_semantics(
    db_session, seed_user_and_server
):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(
        OAuthToken(
            access_token="AT",
            token_type="Bearer",
            refresh_token="RT",
            expires_in=-60,
        )
    )
    got = await store.get_tokens()
    assert got is not None
    assert got.refresh_token == "RT"
    assert got.expires_in is not None
    assert got.expires_in < 0


@pytest.mark.asyncio
async def test_missing_expires_in_clears_old_expiry(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(
        OAuthToken(access_token="AT", token_type="Bearer", expires_in=3600)
    )
    row = (
        db_session.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one()
    )
    assert row.expires_at is not None

    await store.set_tokens(OAuthToken(access_token="AT2", token_type="Bearer"))

    db_session.refresh(row)
    assert row.expires_at is None


@pytest.mark.asyncio
async def test_refresh_without_new_refresh_token_preserves_existing(
    db_session, seed_user_and_server
):
    """RFC 6749 §6: an AS refreshing a token may omit ``refresh_token`` in the
    response when it isn't rotating it -- the original refresh_token remains
    valid. ``_set_tokens`` must not null out a still-valid stored value just
    because a later response didn't repeat it.
    """
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(
        OAuthToken(
            access_token="AT",
            token_type="Bearer",
            refresh_token="ORIGINAL-RT",
            expires_in=3600,
        )
    )

    # Simulate a refresh response that rotates the access token but omits
    # refresh_token (server keeps the original refresh_token valid).
    await store.set_tokens(
        OAuthToken(access_token="AT2", token_type="Bearer", expires_in=3600)
    )

    got = await store.get_tokens()
    assert got is not None
    assert got.access_token == "AT2"
    assert got.refresh_token == "ORIGINAL-RT"


@pytest.mark.asyncio
async def test_refresh_with_new_refresh_token_rotates_it(
    db_session, seed_user_and_server
):
    """When the AS DOES rotate the refresh_token, the new value must
    overwrite the old one (rotation must keep working)."""
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(
        OAuthToken(
            access_token="AT",
            token_type="Bearer",
            refresh_token="ORIGINAL-RT",
            expires_in=3600,
        )
    )

    await store.set_tokens(
        OAuthToken(
            access_token="AT2",
            token_type="Bearer",
            refresh_token="ROTATED-RT",
            expires_in=3600,
        )
    )

    got = await store.get_tokens()
    assert got is not None
    assert got.refresh_token == "ROTATED-RT"


@pytest.mark.asyncio
async def test_storage_does_not_commit_shared_session(
    db_session, seed_user_and_server, monkeypatch
):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )

    def fail_commit():
        raise AssertionError("DBTokenStorage must not commit a shared session")

    monkeypatch.setattr(db_session, "commit", fail_commit)

    await store.set_tokens(OAuthToken(access_token="AT", token_type="Bearer"))
    await store.set_client_info(
        OAuthClientInformationFull(
            client_id="client-id",
            redirect_uris=["https://xagent.test/oauth/callback"],
        )
    )

    db_session.expire_all()
    row = (
        db_session.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one()
    )
    assert row.status == "connected"
    server = db_session.query(MCPServer).filter_by(id=server_id).one()
    assert server.oauth_client["client_id"] == "client-id"


@pytest.mark.asyncio
async def test_mark_error_sets_status_and_clears_pending_fields(
    db_session, seed_user_and_server
):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(OAuthToken(access_token="AT", token_type="Bearer"))

    assert await store.mark_error() is True

    row = (
        db_session.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one()
    )
    assert row.status == "error"
    assert row.pkce_verifier is None
    assert row.state is None


@pytest.mark.asyncio
async def test_mark_error_noop_when_no_row(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(user_id=user_id, mcpserver_id=server_id, db=db_session)

    assert await store.mark_error() is False
    assert (
        db_session.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one_or_none()
        is None
    )


@pytest.mark.asyncio
async def test_storage_reads_do_not_use_shared_session_query(
    db_session, seed_user_and_server, monkeypatch
):
    user_id, server_id = seed_user_and_server
    store = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await store.set_tokens(OAuthToken(access_token="AT", token_type="Bearer"))
    await store.set_client_info(
        OAuthClientInformationFull(
            client_id="client-id",
            redirect_uris=["https://xagent.test/oauth/callback"],
        )
    )

    def fail_query(*args, **kwargs):
        raise AssertionError("DBTokenStorage reads must not use a shared session")

    monkeypatch.setattr(db_session, "query", fail_query)

    got_tokens = await store.get_tokens()
    got_client = await store.get_client_info()

    assert got_tokens is not None
    assert got_tokens.access_token == "AT"
    assert got_client is not None
    assert got_client.client_id == "client-id"
