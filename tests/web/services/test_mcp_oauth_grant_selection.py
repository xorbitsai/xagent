"""Which OAuth grants the selector returns depends on whether the caller holds
an association row for the server.

A caller without an association is a team member who authorized a shared
connector themselves. The selector returns nothing for that caller unless the
call site passes ``allow_team_member_grants=True``. A caller with an
association is unaffected by the parameter.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.user import User
from xagent.web.services.mcp_oauth import select_mcp_oauth_grants

MCP_URL = "https://mcp.selection.test/mcp"
ISSUER = "https://auth.selection.test"
AUTH = {"type": "mcp_oauth"}


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'grant-selection.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _add_user(db, name: str) -> User:
    user = User(username=name, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _add_server(db) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "selection-server",
            "managed": "external",
            "transport": "streamable_http",
            "url": MCP_URL,
            "auth": AUTH,
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _add_grant(db, *, server: MCPServer, user: User) -> MCPOAuthGrant:
    oauth_client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer=ISSUER,
        authorization_endpoint=f"{ISSUER}/authorize",
        token_endpoint=f"{ISSUER}/token",
        client_id="selection-client",
        token_endpoint_auth_method="none",
        redirect_uri="https://api.selection.test/api/mcp/oauth/callback",
    )
    db.add(oauth_client)
    db.flush()
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=oauth_client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer=ISSUER,
        resource=MCP_URL,
        scope="",
        access_token=encrypt_value("selection-access-token"),
        status="active",
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    return grant


@pytest.mark.parametrize(
    ("holds_association", "expect_returned"),
    [
        pytest.param(False, False, id="no_association_not_allowed"),
        pytest.param(True, True, id="association_holder_not_allowed"),
    ],
)
def test_grant_selection_resource_rule_for_callers_without_association(
    db_session, holds_association: bool, expect_returned: bool
) -> None:
    user = _add_user(db_session, "selection-member")
    server = _add_server(db_session)
    grant = _add_grant(db_session, server=server, user=user)
    association_states = [True, False] if holds_association else [None]
    for is_active in association_states:
        if is_active is not None:
            db_session.add(
                UserMCPServer(
                    user_id=user.id,
                    mcpserver_id=server.id,
                    is_owner=False,
                    is_active=is_active,
                )
            )
            db_session.commit()

        selected = select_mcp_oauth_grants(
            db_session,
            server_id=server.id,
            user_id=user.id,
            auth_config=AUTH,
            resource_owner_key=f"xagent:user:{user.id}",
        )

        assert [g.id for g in selected] == ([grant.id] if expect_returned else [])
        if not holds_association:
            # Under another owner key (an actor policy) the grant is returned
            # even without an association.
            db_session.query(MCPOAuthGrant).filter(MCPOAuthGrant.id == grant.id).update(
                {MCPOAuthGrant.resource_owner_key: "external:account:1"}
            )
            db_session.commit()
            other = select_mcp_oauth_grants(
                db_session,
                server_id=server.id,
                user_id=user.id,
                auth_config=AUTH,
                resource_owner_key="external:account:1",
            )
            assert [g.id for g in other] == [grant.id]
            db_session.query(MCPOAuthGrant).filter(MCPOAuthGrant.id == grant.id).update(
                {MCPOAuthGrant.resource_owner_key: f"xagent:user:{user.id}"}
            )
            db_session.commit()
            no_key = select_mcp_oauth_grants(
                db_session,
                server_id=server.id,
                user_id=user.id,
                auth_config=AUTH,
                resource_owner_key=None,
            )
            assert no_key == []
        if is_active is not None:
            db_session.query(UserMCPServer).filter(
                UserMCPServer.user_id == user.id,
                UserMCPServer.mcpserver_id == server.id,
            ).delete()
            db_session.commit()
