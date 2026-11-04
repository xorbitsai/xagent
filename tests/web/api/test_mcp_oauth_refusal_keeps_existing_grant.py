"""A refused callback leaves the owner's earlier grant as it was.

When the provider grants less than ``required_scope``, the callback stores
nothing and revokes only the tokens it just received. A grant the same
resource owner already holds for the server keeps its tokens, scope and
status, and stays usable.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.api.test_mcp_oauth_required_scope_callback import (
    ACTOR_OWNER,
    BITBUCKET_SCOPES,
    GRANTED_WITHOUT_BITBUCKET,
    ISSUED_TOKENS,
    ISSUER,
    MCP_URL,
    _add_flow,
    _atlassian_auth,
    _callback_request,
    _redirect_query,
    _scope_without,
    _token_response,
)
from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import mcp_oauth_callback
from xagent.web.builtin_mcp_registry import ATLASSIAN_MCP_OAUTH_SCOPE
from xagent.web.models import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.mcp_oauth import select_mcp_oauth_grants

# What the refused authorization is granted: no read:me, which Atlassian's
# required_scope names.
REFUSED_SCOPE = _scope_without(*BITBUCKET_SCOPES, "read:me")


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'refusal-keeps-grant.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


def _add_existing_grant(db, server, user, *, scope: str) -> MCPOAuthGrant:
    """A grant the owner got from an earlier authorization with the same
    client, issuer and resource as the flow under test."""
    client = (
        db.query(MCPOAuthClient).filter(MCPOAuthClient.mcp_server_id == server.id).one()
    )
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=client.id,
        resource_owner_key=ACTOR_OWNER,
        issuer=ISSUER,
        resource=MCP_URL,
        scope=scope,
        access_token=encrypt_value("existing-access-token"),
        refresh_token=encrypt_value("existing-refresh-token"),
        expires_at=mcp_api._utc_now() + timedelta(hours=1),
        status="active",
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    return grant


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing_scope", "usable"),
    [
        pytest.param(GRANTED_WITHOUT_BITBUCKET, True, id="usable-grant"),
        # Stored before required_scope was configured, with the very scope the
        # refused authorization returns, so both have the same lookup hash.
        pytest.param(REFUSED_SCOPE, False, id="same-scope-as-the-refusal"),
    ],
)
async def test_an_insufficient_scope_refusal_keeps_the_owners_earlier_grant(
    db_session, monkeypatch, existing_scope, usable
):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_flow(
        db,
        user,
        auth,
        state="state-refused-again",
        requested_scope=ATLASSIAN_MCP_OAUTH_SCOPE,
    )
    existing = _add_existing_grant(db, server, user, scope=existing_scope)
    existing_id = existing.id
    existing_lookup_hash = existing.lookup_hash
    revoked = _token_response(monkeypatch, {**ISSUED_TOKENS, "scope": REFUSED_SCOPE})

    response = await mcp_oauth_callback(_callback_request("state-refused-again"), db)

    assert _redirect_query(response)["mcp_oauth_error"] == ["insufficient_scope"]
    # Only the tokens of the refused authorization are revoked.
    assert [snapshot.access_token for snapshot in revoked] == ["issued-access-token"]
    db.expire_all()
    kept = db.query(MCPOAuthGrant).one()
    assert kept.id == existing_id
    assert kept.lookup_hash == existing_lookup_hash
    assert kept.scope == existing_scope
    assert kept.status == "active"
    assert kept.revoked_at is None
    assert decrypt_value(str(kept.access_token)) == "existing-access-token"
    assert decrypt_value(str(kept.refresh_token)) == "existing-refresh-token"
    selected = select_mcp_oauth_grants(
        db,
        server_id=server.id,
        user_id=user.id,
        auth_config=auth,
        resource_owner_key=ACTOR_OWNER,
    )
    assert [grant.id for grant in selected] == ([existing_id] if usable else [])
