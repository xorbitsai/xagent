"""Which OAuth grants the selector returns depends on whether the caller holds
an association row for the server.

A caller without an association is a team member who authorized a shared
connector themselves. The selector returns nothing for that caller unless the
call site passes ``allow_team_member_grants=True``, and even then only grants
whose resource equals the server's current URL, or the URL the call site says
it sends the token to. A caller with an association is unaffected by the
parameter and by the resource rule.
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
from xagent.web.services.mcp_runtime import build_mcp_runtime_connection

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
    db,
    *,
    server: MCPServer,
    user: User,
    resource: str = MCP_URL,
    oauth_client_id: int | None = None,
    resource_owner_key: str | None = None,
) -> MCPOAuthGrant:
    if oauth_client_id is None:
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
        oauth_client_id = oauth_client.id
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=oauth_client_id,
        resource_owner_key=resource_owner_key or f"xagent:user:{user.id}",
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
        # A protected resource that is the URL's origin is not the URL; such a
        # grant is not one a member authorization may create.
        pytest.param(
            False,
            True,
            MCP_URL,
            "https://mcp.selection.test",
            False,
            id="no_association_resource_is_url_origin",
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


def test_grant_selection_without_owner_key_applies_the_url_rule_to_every_owner_key(
    db_session,
) -> None:
    # No production caller omits the owner key; this pins what such a call
    # returns for a caller without an association when the flag is set.
    user = _add_user(db_session, "selection-member")
    server = _add_server(db_session)
    default_key = _add_grant(db_session, server=server, user=user)
    client_id = default_key.mcp_oauth_client_id
    other_key = _add_grant(
        db_session,
        server=server,
        user=user,
        oauth_client_id=client_id,
        resource_owner_key="external:account:1",
    )
    other_resource = _add_grant(
        db_session,
        server=server,
        user=user,
        resource=OTHER_URL,
        oauth_client_id=client_id,
    )

    selected = select_mcp_oauth_grants(
        db_session,
        server_id=server.id,
        user_id=user.id,
        auth_config=AUTH,
        resource_owner_key=None,
        allow_team_member_grants=True,
    )

    assert {g.id for g in selected} == {default_key.id, other_key.id}
    assert other_resource.id not in {g.id for g in selected}


@pytest.mark.parametrize(
    ("stored_url", "endpoint_url", "expect_returned"),
    [
        pytest.param(MCP_URL, OTHER_URL, False, id="stored_matches_endpoint_differs"),
        pytest.param(OTHER_URL, MCP_URL, True, id="stored_differs_endpoint_matches"),
        pytest.param(MCP_URL, "", False, id="endpoint_empty"),
    ],
)
def test_grant_selection_compares_the_endpoint_url_when_given(
    db_session, stored_url: str, endpoint_url: str, expect_returned: bool
) -> None:
    user = _add_user(db_session, "selection-member")
    server = _add_server(db_session, url=stored_url)
    grant = _add_grant(db_session, server=server, user=user)
    server_id, user_id, grant_id = server.id, user.id, grant.id

    statements: list[str] = []

    def _record(_conn, _cursor, statement, *_args) -> None:
        statements.append(statement)

    engine = db_session.get_bind()
    event.listen(engine, "before_cursor_execute", _record)
    try:
        selected = select_mcp_oauth_grants(
            db_session,
            server_id=server_id,
            user_id=user_id,
            auth_config=AUTH,
            resource_owner_key=f"xagent:user:{user_id}",
            allow_team_member_grants=True,
            endpoint_url=endpoint_url,
        )
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    assert [g.id for g in selected] == ([grant_id] if expect_returned else [])
    # The stored URL is not read when the call site passes the URL.
    assert not any("FROM mcp_servers" in s for s in statements)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection_url", "stored_url", "expect_token"),
    [
        pytest.param(MCP_URL, OTHER_URL, True, id="connection_url_matches"),
        pytest.param(OTHER_URL, MCP_URL, False, id="connection_url_differs"),
        pytest.param(None, MCP_URL, False, id="connection_url_missing"),
    ],
)
async def test_runtime_connection_selects_against_the_url_it_sends_to(
    db_session, connection_url: str | None, stored_url: str, expect_token: bool
) -> None:
    user = _add_user(db_session, "selection-member")
    server = _add_server(db_session, url=stored_url)
    _add_grant(db_session, server=server, user=user)
    # The loaded row and the stored row disagree, as when the URL is edited
    # after the runtime loaded the server. The session does not autoflush, so
    # a read of the row still sees the stored URL.
    server.url = connection_url

    build = await build_mcp_runtime_connection(
        db_session, server, user_id=user.id, allow_team_member_grants=True
    )

    if expect_token:
        assert build.connection is not None
        assert build.connection["url"] == connection_url
        assert build.connection["headers"]["Authorization"] == (
            "Bearer selection-access-token"
        )
    else:
        assert build.connection is None
        assert build.diagnostic is not None
