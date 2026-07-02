"""Integration test for ``WebToolConfig.on_mcp_reauthorization_required``.

When the factory's registered-creator loop catches ``MCPReauthorizationRequired``
during a real agent run, it calls this hook (if present on the config) so the
stored OAuth connection is flagged as needing reconnection -- surfaced by the
existing Connect/Reconnect UI on its next status poll.
"""

import pytest

from xagent.web.models.mcp_oauth import MCPUserOAuthToken
from xagent.web.services.mcp_oauth_token_storage import DBTokenStorage
from xagent.web.tools.config import WebToolConfig


@pytest.mark.asyncio
async def test_hook_marks_connected_row_as_error(db_session, seed_user_and_server):
    user_id, server_id = seed_user_and_server

    # Seed a connected token row directly via DBTokenStorage.
    from mcp.shared.auth import OAuthToken

    storage = DBTokenStorage(
        user_id=user_id, mcpserver_id=server_id, db=db_session, create_missing=True
    )
    await storage.set_tokens(OAuthToken(access_token="AT", token_type="Bearer"))

    row = (
        db_session.query(MCPUserOAuthToken)
        .filter_by(user_id=user_id, mcpserver_id=server_id)
        .one()
    )
    assert row.status == "connected"

    cfg = WebToolConfig(db=db_session, request=None, user_id=user_id)
    await cfg.on_mcp_reauthorization_required(server_id)

    db_session.refresh(row)
    assert row.status == "error"


@pytest.mark.asyncio
async def test_hook_is_noop_when_mcpserver_id_is_none(db_session, seed_user_and_server):
    user_id, _server_id = seed_user_and_server
    cfg = WebToolConfig(db=db_session, request=None, user_id=user_id)

    # Must not raise even though there is nothing to mark.
    await cfg.on_mcp_reauthorization_required(None)


@pytest.mark.asyncio
async def test_hook_is_noop_when_user_id_is_none(db_session, seed_user_and_server):
    _user_id, server_id = seed_user_and_server
    cfg = WebToolConfig(db=db_session, request=None, user_id=1)
    # Force the "no resolvable user" state directly -- the constructor
    # always falls back to a default int user id when request=None, so
    # this is the only way to exercise the hook's own None-guard.
    cfg._user_id = None

    # Must not raise; there's no per-user row to mark without a user id.
    await cfg.on_mcp_reauthorization_required(server_id)
