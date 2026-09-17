"""`/api/mcp/apps` must emit its attachability and configurability decisions,
not their inputs (#1347).

The connector picker used to reconstruct these decisions from
`is_connected` + `is_custom` + `auth_type`, which made three unstated backend
emission rules load-bearing on the client: that `is_connected` for an
mcp_oauth entry requires an active grant, that `auth_type` appears on a local
entry only alongside an active personal association, and that `is_custom` is
never set by the catalog branch. These tests pin `can_attach`, `can_authorize`
and `can_configure` directly, so changing any of those rules fails here
instead of silently mis-gating the picker.

The three fields answer different questions and diverge on real populations:
a team-owned mcp_oauth connector is attachable but has no consent flow the
member could start; a hook-resolved connector is attachable with no consent
flow existing at all; and `can_configure` answers whether the entry's edit
route would resolve for the caller -- a personal association row for a local
entry, or an existing connection for a catalog entry -- which is independent
of both.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.mcp import list_mcp_apps
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import connector_team_scope
from xagent.web.tools.config import TokenRequest, set_oauth_token_resolver_hook


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "mcp-apps-attachability.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    owner = User(username="alice", password_hash="x", is_admin=False)
    member = User(username="bob", password_hash="x", is_admin=False)
    db.add_all([owner, member])
    db.commit()
    db.refresh(owner)
    db.refresh(member)

    yield db, owner, member
    db.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _reset_process_global_hooks():
    """Both hooks are process-global; never leak one into a sibling test."""
    yield
    connector_team_scope.set_connector_team_hooks()
    set_oauth_token_resolver_hook(None)


def _install_token_resolver() -> None:
    """Install a resolver hook shaped like an embedder's.

    Its answers are irrelevant: `can_attach`/`can_authorize` are decided on
    hook *presence*, because a truthful per-connector answer would mean one
    embedder round-trip per listed connector on every list request.
    """

    def resolver(_request: TokenRequest) -> None:
        raise AssertionError("list_mcp_apps must never invoke the resolver")

    set_oauth_token_resolver_hook(resolver)


def _install_visibility(mapping: dict[int, dict[str, set[int]]]) -> None:
    def visibility(_db, user_id: int) -> dict[str, set[int]]:
        answer = mapping.get(int(user_id))
        if answer is None:
            return {"mcp": set(), "custom_api": set()}
        return {"mcp": set(answer["mcp"]), "custom_api": set(answer["custom_api"])}

    connector_team_scope.set_connector_team_hooks(visibility=visibility)


def _add_oauth_server(
    db,
    owner: User | None,
    name: str = "records",
    *,
    is_active: bool = True,
) -> MCPServer:
    """An mcp_oauth-shaped custom server. ``owner=None`` writes no association
    at all, which is how a team-owned connector reaches a member."""
    server = MCPServer.from_config(
        {
            "name": name,
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.example.com/mcp",
            "auth": {
                "type": "mcp_oauth",
                "resource": "https://mcp.example.com/mcp",
                "issuer": "https://auth.example.com",
                "scope": "records.read",
            },
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    if owner is not None:
        db.add(
            UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                is_active=is_active,
            )
        )
        db.commit()
    return server


def _add_stdio_server(
    db, owner: User, name: str = "files", *, is_active: bool = True
) -> MCPServer:
    """A non-mcp_oauth custom server: credentials live on the row and its env
    layers, so nothing but the association can gate it."""
    server = MCPServer.from_config(
        {
            "name": name,
            "managed": "external",
            "transport": "stdio",
            "command": f"{name}-mcp",
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=owner.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=is_active,
        )
    )
    db.commit()
    return server


def _grant(db, server: MCPServer, user: User, *, status: str = "active") -> None:
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        client_id="client-123",
        token_endpoint_auth_method="none",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
    )
    db.add(client)
    db.flush()
    db.add(
        MCPOAuthGrant(
            mcp_server_id=server.id,
            user_id=user.id,
            mcp_oauth_client_id=client.id,
            resource_owner_key=f"xagent:user:{user.id}",
            issuer="https://auth.example.com",
            resource="https://mcp.example.com/mcp",
            scope="records.read",
            access_token=encrypt_value("runtime-token"),
            status=status,
        )
    )
    db.commit()


def _add_custom_api(
    db, owner: User | None, name: str = "billing", *, is_active: bool = True
) -> CustomApi:
    """A custom API. ``owner=None`` writes no association at all, which is how
    a team-owned custom API reaches a member."""
    api = CustomApi(
        name=name,
        description=f"{name} API",
        url="https://api.example.com/v1",
        method="GET",
    )
    db.add(api)
    db.commit()
    db.refresh(api)
    if owner is not None:
        db.add(
            UserCustomApi(
                user_id=owner.id,
                custom_api_id=api.id,
                is_owner=True,
                # Matches the one production write point (custom_api.py:248);
                # the column defaults to False, and PUT's second gate reads it,
                # so a fixture without it is a shape production never creates.
                can_edit=True,
                is_active=is_active,
            )
        )
        db.commit()
    return api


def _add_catalog_oauth_app(db, app_id: str = "granola") -> PublicMCPApp:
    """A catalog app carrying the same mcp_oauth auth_type as the custom
    population — the entry that used to be gated only by `is_custom` being
    absent from this branch."""
    app = PublicMCPApp(
        app_id=app_id,
        name="Granola",
        description="A catalog app",
        icon="",
        category="Productivity",
        # classify_app_auth derives auth_type from the entry's own fields: a
        # remote transport with no launch command is the mcp_oauth shape.
        transport="streamable_http",
        is_visible_in_connector=True,
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


def _add_catalog_builtin_oauth_app(db, app_id: str = "acme-crm") -> PublicMCPApp:
    """A provider-redirect catalog app. `transport == "oauth"` is what
    classify_app_auth keys on, and it takes its own branch in list_mcp_apps --
    the one the other catalog fixtures never reach."""
    app = PublicMCPApp(
        app_id=app_id,
        name="Acme CRM",
        description="A builtin-oauth catalog app",
        icon="",
        category="Productivity",
        transport="oauth",
        provider_name="acme",
        is_visible_in_connector=True,
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


def _entry(db, user: User, entry_id: str, *, location: str = "local") -> dict:
    return next(
        a
        for a in list_mcp_apps(location=location, current_user=user, db=db)
        if a["id"] == entry_id
    )


# --- AC: the picker's gate is emitted, not re-derived -----------------------


def test_an_mcp_oauth_connector_with_an_active_grant_is_attachable(db_session):
    db, owner, _member = db_session
    server = _add_oauth_server(db, owner)
    _grant(db, server, owner)

    entry = _entry(db, owner, "records")
    assert entry["can_attach"] is True
    # Consent already given is not consent that cannot be given again: the
    # picker suppresses the trigger on connected state itself.
    assert entry["can_authorize"] is True
    assert entry["is_connected"] is True


def test_a_never_authorized_mcp_oauth_connector_is_not_attachable_standalone(
    db_session,
):
    """Standalone installs no resolver, so nothing can supply this server's
    tokens: attaching it would fail at run time with "MCP server credentials
    are unavailable". Fail early instead, and keep consent advertised as the
    recovery."""
    db, owner, _member = db_session
    _add_oauth_server(db, owner)

    entry = _entry(db, owner, "records")
    assert entry["can_attach"] is False
    assert entry["can_authorize"] is True


def test_a_revoked_grant_does_not_make_a_connector_attachable(db_session):
    """The grant query filters status == "active"; a revoked row must not read
    as credentials."""
    db, owner, _member = db_session
    server = _add_oauth_server(db, owner)
    _grant(db, server, owner, status="revoked")

    assert _entry(db, owner, "records")["can_attach"] is False


def test_another_users_grant_does_not_make_a_connector_attachable(db_session):
    """Grants are per (server, user). The requesting user's own grant is the
    only one that resolves at run time."""
    db, owner, member = db_session
    server = _add_oauth_server(db, owner, is_active=True)
    db.add(
        UserMCPServer(
            user_id=member.id, mcpserver_id=server.id, is_owner=False, is_active=True
        )
    )
    db.commit()
    _grant(db, server, owner)

    assert _entry(db, owner, "records")["can_attach"] is True
    assert _entry(db, member, "records")["can_attach"] is False


def test_a_non_oauth_custom_connector_is_attachable_without_any_grant(db_session):
    """Only the mcp_oauth shape has a credential gate; a stdio server carries
    its credentials on the row and its env layers."""
    db, owner, _member = db_session
    _add_stdio_server(db, owner)

    entry = _entry(db, owner, "files")
    assert entry["can_attach"] is True
    assert entry["can_authorize"] is False


# --- AC: an inactive association is not attachable -------------------------


def test_an_inactive_association_is_not_attachable(db_session):
    """The runtime's server query filters UserMCPServer.is_active, so a
    deactivated connector would load zero tools. This is the case the old
    predicate got wrong in one direction: deactivating *after* consent leaves
    is_connected true, so it stayed attachable."""
    db, owner, _member = db_session
    server = _add_stdio_server(db, owner, is_active=False)
    assert _entry(db, owner, "files")["can_attach"] is False

    # Including when a completed grant exists: toggle_mcp_server never revokes
    # one, so is_connected stays true and cannot carry this decision.
    oauth = _add_oauth_server(db, owner, "records", is_active=False)
    _grant(db, oauth, owner)
    entry = _entry(db, owner, "records")
    assert entry["is_connected"] is True
    assert entry["can_attach"] is False
    assert entry["can_authorize"] is False
    assert server.id != oauth.id


def test_an_inactive_custom_api_association_is_not_attachable(db_session):
    """The Custom API half of the runtime query filters is_active identically,
    while this listing reports is_connected: True unconditionally."""
    db, owner, _member = db_session
    _add_custom_api(db, owner, is_active=False)

    entry = _entry(db, owner, "billing")
    assert entry["is_connected"] is True
    assert entry["can_attach"] is False


def test_an_active_custom_api_is_attachable_and_never_authorizable(db_session):
    db, owner, _member = db_session
    _add_custom_api(db, owner)

    entry = _entry(db, owner, "billing")
    assert entry["can_attach"] is True
    assert entry["can_authorize"] is False


# --- AC: catalog apps no longer depend on is_custom being absent ------------


def test_an_unconnected_catalog_oauth_app_is_not_attachable(db_session):
    """The entry the picker used to gate purely on `is_custom` being absent
    from this branch. It carries auth_type mcp_oauth exactly like the custom
    population, and is_connected: false here means no association row exists
    at all."""
    db, owner, _member = db_session
    _add_catalog_oauth_app(db)

    entry = _entry(db, owner, "granola", location="remote")
    # Load-bearing: without this the fixture could drift to some other shape
    # and the test would stop discriminating the case it exists for.
    assert entry["auth_type"] == "mcp_oauth"
    assert entry["is_connected"] is False
    assert entry["can_attach"] is False
    # Catalog entries connect through /apps/{app_id}/oauth/connect, dispatched
    # on auth_type; can_authorize deliberately does not restate that.
    assert entry["can_authorize"] is False


def test_an_unconnected_builtin_oauth_catalog_app_is_not_attachable(db_session):
    """The builtin_oauth branch resolves connected state through
    `_connected_oauth_server_for_app` and the user's UserOAuth accounts, not
    through the association lookups the other catalog fixtures exercise. Its
    `can_attach` is emitted by the same line, but nothing else in this file
    routes through that branch."""
    db, owner, _member = db_session
    _add_catalog_builtin_oauth_app(db)

    entry = _entry(db, owner, "acme-crm", location="remote")
    assert entry["auth_type"] == "builtin_oauth"
    assert entry["is_connected"] is False
    assert entry["can_attach"] is False
    assert entry["can_authorize"] is False


def test_a_catalog_app_stays_unattachable_even_with_a_resolver_installed(db_session):
    """The resolver relaxes the *credential* gate, never the association gate:
    a catalog app the user never connected is not their connector to attach,
    whatever can supply tokens for it."""
    db, owner, _member = db_session
    _add_catalog_oauth_app(db)
    _install_token_resolver()

    assert _entry(db, owner, "granola", location="remote")["can_attach"] is False


# --- AC: hook-supplied credentials -----------------------------------------


def test_a_resolver_hook_makes_a_never_authorized_connector_attachable(db_session):
    """#1332's population: tokens arrive through set_oauth_token_resolver_hook,
    so no MCPOAuthGrant is ever written and is_connected stays false forever.
    The connector is nonetheless attachable and the runtime resolves it."""
    db, owner, _member = db_session
    _add_oauth_server(db, owner)
    _install_token_resolver()

    entry = _entry(db, owner, "records")
    assert entry["is_connected"] is False
    assert entry["can_attach"] is True


def test_a_resolver_hook_suppresses_the_interactive_consent_flow(db_session):
    """There is no interactive consent and no identity the editor could sign
    in as, so advertising authorization asserts a step that does not exist."""
    db, owner, _member = db_session
    _add_oauth_server(db, owner)
    _install_token_resolver()

    assert _entry(db, owner, "records")["can_authorize"] is False


def test_a_resolver_hook_does_not_relax_the_association_gate(db_session):
    """Credentials resolving is not the same as the runtime seeing the server:
    its query still drops the inactive association."""
    db, owner, _member = db_session
    _add_oauth_server(db, owner, is_active=False)
    _install_token_resolver()

    assert _entry(db, owner, "records")["can_attach"] is False


# --- AC: the #1338 team overlay --------------------------------------------


def test_a_team_owned_mcp_oauth_connector_is_attachable_but_not_authorizable(
    db_session,
):
    """The case the two fields exist to separate. The member holds no personal
    association, so /{server_id}/oauth/connect would 404 — but the overlay
    listed the connector precisely so members could attach it, and the runtime
    overlays the same team ids. Withholding auth_type was the only way to say
    "not authorizable", and it said "not attachable" too."""
    db, _owner, member = db_session
    server = _add_oauth_server(db, None)
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )
    _install_token_resolver()

    entry = _entry(db, member, "records")
    assert entry["can_attach"] is True
    assert entry["can_authorize"] is False
    assert "auth_type" not in entry


def test_a_team_owned_mcp_oauth_connector_needs_credentials_like_any_other(db_session):
    """Without a resolver the member has no way to obtain tokens for it — a
    team link shares the server definition, not anyone's grant — so the
    credential gate applies to the overlay population too."""
    db, _owner, member = db_session
    server = _add_oauth_server(db, None)
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    assert _entry(db, member, "records")["can_attach"] is False


def test_a_team_owned_connector_is_never_authorizable_even_holding_a_grant(
    db_session,
):
    """The association half of can_authorize, pinned without a resolver
    installed so the resolver short-circuit cannot carry the assertion. A grant
    outlives the association row it was created under, so credentials can
    resolve for a connector the member now reaches only through the team link
    — and /{server_id}/oauth/connect would still 404 on the missing
    association."""
    db, _owner, member = db_session
    server = _add_oauth_server(db, None)
    _grant(db, server, member)
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    entry = _entry(db, member, "records")
    assert entry["can_attach"] is True
    assert entry["can_authorize"] is False


def test_a_team_link_survives_a_deactivated_personal_association(db_session):
    """`is_active` gates only the *personal* arm of the visibility rule.

    A member who holds both a (deactivated) personal association and a team
    link is carried into the listing as (server, inactive_user_mcp) -- the
    overlay skips ids already covered personally -- so a bare
    `user_mcp.is_active` check refuses it. The runtime does the opposite:
    `_load_visible_runtime_connectors` drops the inactive personal link and
    then re-adds the team ids, which `visible_mcp_server_clause` states
    outright. Refusing here would be a fail-early on a connector that loads
    fine."""
    db, owner, member = db_session
    server = _add_stdio_server(db, owner)
    db.add(
        UserMCPServer(
            user_id=member.id,
            mcpserver_id=server.id,
            is_owner=False,
            is_active=False,
        )
    )
    db.commit()
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    assert _entry(db, member, "files")["can_attach"] is True

    # Without the team link the same row is refused -- the personal arm still
    # gates, so this pins the team arm rather than the removal of the check.
    connector_team_scope.set_connector_team_hooks()
    assert _entry(db, member, "files")["can_attach"] is False


def test_a_team_link_survives_a_deactivated_custom_api_association(db_session):
    """The Custom API half of the same rule (`visible_custom_api_clause`)."""
    db, owner, member = db_session
    api = _add_custom_api(db, owner)
    db.add(
        UserCustomApi(
            user_id=member.id,
            custom_api_id=api.id,
            is_owner=False,
            is_active=False,
        )
    )
    db.commit()
    _install_visibility({int(member.id): {"mcp": set(), "custom_api": {int(api.id)}}})

    assert _entry(db, member, "billing")["can_attach"] is True

    connector_team_scope.set_connector_team_hooks()
    assert _entry(db, member, "billing")["can_attach"] is False


def test_auth_type_and_can_authorize_agree_on_the_consent_preconditions(db_session):
    """`auth_type` and `can_authorize` share one predicate and must not drift.

    They answer different questions -- which Connect flow the settings dialog
    dispatches, and whether the picker may advertise consent -- but the second
    is exactly the first plus the resolver term. Pinned as an algebraic
    relation so an edit to either condition alone fails here."""
    db, owner, _member = db_session
    _add_oauth_server(db, owner)

    entry = _entry(db, owner, "records")
    assert entry["auth_type"] == "mcp_oauth"
    assert entry["can_authorize"] is True

    # Installing the resolver must move can_authorize alone: auth_type keeps
    # answering the dispatch question, which the hook does not change.
    _install_token_resolver()
    entry = _entry(db, owner, "records")
    assert entry["auth_type"] == "mcp_oauth"
    assert entry["can_authorize"] is False


def test_a_team_owned_non_oauth_connector_is_attachable_standalone_of_any_hook(
    db_session,
):
    """The common team case: no credential gate, so the team link alone
    carries it."""
    db, owner, member = db_session
    server = _add_stdio_server(db, owner)
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )

    entry = _entry(db, member, "files")
    assert entry["can_attach"] is True
    assert entry["can_authorize"] is False


def test_a_team_owned_custom_api_is_attachable(db_session):
    db, owner, member = db_session
    api = _add_custom_api(db, owner)
    _install_visibility({int(member.id): {"mcp": set(), "custom_api": {int(api.id)}}})

    assert _entry(db, member, "billing")["can_attach"] is True


def test_a_connected_catalog_app_is_attachable(db_session):
    """The positive half of the catalog rule, and the branch's largest
    population by far. `is_connected` there means an active association to the
    server row the connect route writes under the catalog app id -- a stdio
    app, so no grant is involved and the credential gate cannot mask a
    regression in the association half. The app id must stay outside the
    builtin registry: `_app_to_dict` lets a builtin entry override the stored
    transport and launch_config, which would silently reshape this fixture."""
    db, owner, _member = db_session
    app = PublicMCPApp(
        app_id="acme-docs",
        name="Acme Docs",
        description="A catalog app",
        icon="",
        category="Productivity",
        transport="stdio",
        launch_config={"command": "acme-docs-mcp"},
        is_visible_in_connector=True,
    )
    db.add(app)
    db.commit()
    # The connect route stores the row under the catalog app_id, which is how
    # _connected_non_oauth_server_for_app resolves it back.
    server = MCPServer.from_config(
        {
            "name": "acme-docs",
            "managed": "external",
            "transport": "stdio",
            "command": "acme-docs-mcp",
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=owner.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()

    entry = _entry(db, owner, "acme-docs", location="remote")
    assert entry["is_connected"] is True
    assert entry["can_attach"] is True
    assert entry["can_authorize"] is False


# --- AC: can_configure, the picker's edit-route gate ------------------------


def test_a_hook_resolved_connector_is_configurable_by_its_owner(db_session):
    """#1332's population again: is_connected and can_authorize are both
    false for a hook-resolved connector, yet its owner holds the personal
    association the edit routes require, so the three fields must disagree."""
    db, owner, _member = db_session
    _add_oauth_server(db, owner)
    _install_token_resolver()

    entry = _entry(db, owner, "records")
    assert entry["is_connected"] is False
    assert entry["can_authorize"] is False
    assert entry["can_configure"] is True


def test_a_never_authorized_connector_is_both_configurable_and_authorizable(
    db_session,
):
    """A connector whose owner has an active association but never completed
    consent needs both: Configure to fix a misconfigured field, Authorize to
    complete consent. Neither implies the other."""
    db, owner, _member = db_session
    _add_oauth_server(db, owner)

    entry = _entry(db, owner, "records")
    assert entry["can_configure"] is True and entry["can_authorize"] is True


def test_a_deactivated_association_stays_configurable(db_session):
    """Neither GET nor PUT /api/mcp/servers/{id} filters is_active, so a
    deactivated connector's owner can still open and save its form -- unlike
    can_attach, which the runtime's own query does filter on."""
    db, owner, _member = db_session
    _add_stdio_server(db, owner, is_active=False)

    entry = _entry(db, owner, "files")
    assert entry["can_attach"] is False and entry["can_configure"] is True


def test_a_team_owned_connector_without_a_personal_association_is_not_configurable(
    db_session,
):
    """The overlay makes this connector attachable for the member, but the
    edit routes require a row of the member's own, which does not exist here
    -- so GET/PUT /api/mcp/servers/{id} would 404."""
    db, _owner, member = db_session
    server = _add_oauth_server(db, None)
    _install_visibility(
        {int(member.id): {"mcp": {int(server.id)}, "custom_api": set()}}
    )
    _install_token_resolver()

    entry = _entry(db, member, "records")
    assert entry["can_attach"] is True and entry["can_configure"] is False


def test_a_team_owned_custom_api_without_a_personal_association_is_not_configurable(
    db_session,
):
    """The Custom API half of the same rule. `is_connected` is hardcoded True
    for every Custom API entry, and must not be mistaken for configurability."""
    db, _owner, member = db_session
    api = _add_custom_api(db, None)
    _install_visibility({int(member.id): {"mcp": set(), "custom_api": {int(api.id)}}})

    entry = _entry(db, member, "billing")
    assert entry["is_connected"] is True and entry["can_configure"] is False


def test_an_own_custom_api_is_configurable(db_session):
    db, owner, _member = db_session
    _add_custom_api(db, owner)

    entry = _entry(db, owner, "billing")
    assert entry["can_configure"] is True


def test_an_unconnected_catalog_app_is_not_configurable(db_session):
    """The catalog branch's Configure equivalent is "manage my key" or
    "re-run OAuth", both of which only exist once connected -- guarding
    against the dead-button regression an unconditional True would reintroduce."""
    db, owner, _member = db_session
    _add_catalog_oauth_app(db)
    _add_catalog_builtin_oauth_app(db)

    granola = _entry(db, owner, "granola", location="remote")
    acme = _entry(db, owner, "acme-crm", location="remote")
    assert granola["can_configure"] is False
    assert acme["can_configure"] is False


def test_a_connected_catalog_app_is_configurable(db_session):
    db, owner, _member = db_session
    app = PublicMCPApp(
        app_id="acme-docs",
        name="Acme Docs",
        description="A catalog app",
        icon="",
        category="Productivity",
        transport="stdio",
        launch_config={"command": "acme-docs-mcp"},
        is_visible_in_connector=True,
    )
    db.add(app)
    db.commit()
    server = MCPServer.from_config(
        {
            "name": "acme-docs",
            "managed": "external",
            "transport": "stdio",
            "command": "acme-docs-mcp",
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=owner.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()

    entry = _entry(db, owner, "acme-docs", location="remote")
    assert entry["can_configure"] is True


def test_a_resolver_hook_does_not_make_a_catalog_app_configurable(db_session):
    """The resolver hook only relaxes the credential gate on the local mcp_oauth
    branch; the catalog branch's can_configure is the connection state, which
    the hook does not touch."""
    db, owner, _member = db_session
    _add_catalog_oauth_app(db)
    _install_token_resolver()

    entry = _entry(db, owner, "granola", location="remote")
    assert entry["can_configure"] is False


def test_a_non_owner_with_a_personal_association_is_configurable(db_session):
    """A non-owner's personal env override is still their own row to edit:
    the edit routes' first gate is association existence, not is_owner."""
    db, owner, member = db_session
    server = _add_stdio_server(db, owner)
    db.add(
        UserMCPServer(
            user_id=member.id,
            mcpserver_id=server.id,
            is_owner=False,
            is_active=True,
        )
    )
    db.commit()

    entry = _entry(db, member, "files")
    assert entry["can_configure"] is True


# --- Every entry carries all three decisions ---------------------------------


def test_every_listed_entry_carries_all_three_decisions(db_session):
    """A consumer reading `can_attach` must never see undefined and silently
    treat an entry as unattachable, whichever branch emitted it. Same for
    `can_authorize` and `can_configure`."""
    db, owner, _member = db_session
    _add_stdio_server(db, owner)
    _add_oauth_server(db, owner)
    _add_custom_api(db, owner)
    _add_catalog_oauth_app(db)
    _add_catalog_builtin_oauth_app(db)

    entries = list_mcp_apps(location="all", current_user=owner, db=db)
    assert {"files", "records", "billing", "granola", "acme-crm"} <= {
        e["id"] for e in entries
    }
    for entry in entries:
        assert isinstance(entry.get("can_attach"), bool), entry["id"]
        assert isinstance(entry.get("can_authorize"), bool), entry["id"]
        assert isinstance(entry.get("can_configure"), bool), entry["id"]
