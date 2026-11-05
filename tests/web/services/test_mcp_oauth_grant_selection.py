"""Which OAuth grants the selector returns depends on whether the caller holds
an association row for the server.

A caller without an association is a team member who authorized a shared
connector themselves. The selector returns nothing for that caller unless the
call site passes ``allow_team_member_grants=True``, and even then only grants
whose resource equals the server's current URL. A caller with an association
is unaffected by the parameter and by the resource rule.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
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


def _add_server(db, *, url: str | None = MCP_URL) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "selection-server",
            "managed": "external",
            "transport": "streamable_http",
            "url": MCP_URL,
            "auth": AUTH,
        }
    )
    server.url = url
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _add_grant(
    db, *, server: MCPServer, user: User, resource: str = MCP_URL
) -> MCPOAuthGrant:
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
        resource=resource,
        scope="",
        access_token=encrypt_value("selection-access-token"),
        status="active",
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    return grant


def _add_other_server(db) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "selection-other-server",
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.other-selection.test/mcp",
            "auth": AUTH,
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


@pytest.mark.parametrize(
    ("unrelated_association", "allow", "expect_returned"),
    [
        # An association row only counts when it is the caller's own row for
        # this server; each case below holds a row that is not.
        pytest.param(
            "other_user_same_server",
            False,
            False,
            id="other_users_association_on_server",
        ),
        pytest.param(
            "caller_other_server",
            False,
            False,
            id="callers_association_on_other_server",
        ),
        pytest.param(None, True, True, id="no_association_allowed"),
    ],
)
def test_grant_selection_counts_only_the_callers_association_for_the_server(
    db_session, unrelated_association: str | None, allow: bool, expect_returned: bool
) -> None:
    user = _add_user(db_session, "selection-member")
    server = _add_server(db_session)
    grant = _add_grant(db_session, server=server, user=user)
    if unrelated_association == "other_user_same_server":
        other_user = _add_user(db_session, "selection-other-user")
        db_session.add(UserMCPServer(user_id=other_user.id, mcpserver_id=server.id))
    elif unrelated_association == "caller_other_server":
        other_server = _add_other_server(db_session)
        db_session.add(UserMCPServer(user_id=user.id, mcpserver_id=other_server.id))
    db_session.commit()

    selected = select_mcp_oauth_grants(
        db_session,
        server_id=server.id,
        user_id=user.id,
        auth_config=AUTH,
        resource_owner_key=f"xagent:user:{user.id}",
        allow_team_member_grants=allow,
    )

    assert [g.id for g in selected] == ([grant.id] if expect_returned else [])


@pytest.mark.parametrize(
    ("seed_grant", "owner_key", "expect_lookups", "expect_returned"),
    [
        pytest.param(False, "default", 0, False, id="no_matching_grant"),
        pytest.param(True, "external:account:1", 0, True, id="other_owner_key"),
        pytest.param(True, "default", 1, False, id="default_owner_key"),
    ],
)
def test_grant_selection_looks_up_association_only_when_the_rule_applies(
    db_session,
    seed_grant: bool,
    owner_key: str,
    expect_lookups: int,
    expect_returned: bool,
) -> None:
    user = _add_user(db_session, "selection-member")
    server = _add_server(db_session)
    resource_owner_key = (
        f"xagent:user:{user.id}" if owner_key == "default" else owner_key
    )
    if seed_grant:
        grant = _add_grant(db_session, server=server, user=user)
        db_session.query(MCPOAuthGrant).filter(MCPOAuthGrant.id == grant.id).update(
            {MCPOAuthGrant.resource_owner_key: resource_owner_key}
        )
        db_session.commit()

    statements: list[str] = []

    def _record(_conn, _cursor, statement, *_args) -> None:
        statements.append(statement)

    engine = db_session.get_bind()
    event.listen(engine, "before_cursor_execute", _record)
    try:
        selected = select_mcp_oauth_grants(
            db_session,
            server_id=server.id,
            user_id=user.id,
            auth_config=AUTH,
            resource_owner_key=resource_owner_key,
        )
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    assert sum("user_mcpservers" in s for s in statements) == expect_lookups
    assert len(selected) == (1 if expect_returned else 0)


OTHER_URL = "https://other.selection.test/mcp"


@pytest.mark.parametrize(
    (
        "holds_association",
        "allow",
        "server_url",
        "grant_resource",
        "expect_returned",
    ),
    [
        pytest.param(
            False, False, MCP_URL, MCP_URL, False, id="no_association_not_allowed"
        ),
        pytest.param(
            True, False, MCP_URL, MCP_URL, True, id="association_holder_gets_grant"
        ),
        pytest.param(
            True,
            True,
            MCP_URL,
            OTHER_URL,
            True,
            id="association_holder_resource_mismatch",
        ),
        # The stored URL is not in canonical form; the grant stores the
        # canonical resource, so the comparison must canonicalize the URL.
        pytest.param(
            False,
            True,
            "https://MCP.selection.test:443/mcp/?tenant=1",
            MCP_URL,
            True,
            id="no_association_resource_match",
        ),
        pytest.param(
            False,
            True,
            MCP_URL,
            OTHER_URL,
            False,
            id="no_association_resource_mismatch",
        ),
        pytest.param(
            False, True, None, MCP_URL, False, id="no_association_server_url_none"
        ),
        # A URL that cannot be parsed matches no stored resource.
        pytest.param(
            False,
            True,
            "http://[bad/mcp",
            MCP_URL,
            False,
            id="no_association_server_url_unparsable",
        ),
    ],
)
def test_grant_selection_resource_rule_for_callers_without_association(
    db_session,
    holds_association: bool,
    allow: bool,
    server_url: str | None,
    grant_resource: str,
    expect_returned: bool,
) -> None:
    user = _add_user(db_session, "selection-member")
    # A second connector with a different URL exists in every case, so the
    # server's current URL has to be read for this connector only.
    _add_other_server(db_session)
    server = _add_server(db_session, url=server_url)
    grant = _add_grant(db_session, server=server, user=user, resource=grant_resource)
    # An association counts whatever its is_active flag says, so a holder is
    # exercised with both values; a caller without one runs once.
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
            allow_team_member_grants=allow,
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
