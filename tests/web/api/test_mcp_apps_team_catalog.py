"""`/api/mcp/apps` must offer a team-shared *catalog* connector (#1387).

A connector shared with a team writes a team link row and no per-member
association, and before this change neither branch of the listing claimed the
catalog-backed ones: the catalog branch resolves connection state from personal
associations alone ("is this app connected *for me*"), and the local branch —
the only one applying the team overlay (#1321) — skips every row a catalog app
speaks for, which is what keeps it free of #1346's `is_custom` duplicate. The
member ended up with no entry in the Local tab and an unconnected catalog entry
in the Remote tab, while `/api/mcp/servers` listed the connector all along.

The fix gives the catalog branch a third state. `is_connected` still means
"connected for me"; `is_team_shared` means "reached me through a team link";
and `can_attach` unions the two, gated by shape. Keyless needs no credential
at all. `api_key` is deliberately not credential-gated: the key the runtime
resolves may live on the row's platform env, the application-injected shared
layer, or the governing agent's team row — none visible to this user-scoped
endpoint — so the answer is optimistic and the env-source flags carry the
needs-config signal. The OAuth shapes gate on the member's own credential
(`mcp_oauth`: an active grant; `builtin_oauth`: a usable provider account),
except where an installed token-resolver hook supplies the deployment's
tokens out of band, which satisfies both.

Fictional app ids (`acme*`) are admin-created apps, whose stored transport and
launch_config are used as-is — built-in ids would take theirs from
`builtin_mcp_registry` instead, leaving the shape under test unauthorable.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.mcp import get_mcp_servers, list_mcp_apps
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import connector_team_scope

_MCP_OAUTH_LAUNCH: dict[str, Any] = {
    "url": "https://mcp.example.com/mcp",
    "auth": {
        "type": "mcp_oauth",
        "resource": "https://mcp.example.com/mcp",
        "issuer": "https://auth.example.com",
    },
}


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "mcp-apps-team-catalog.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    creator = User(username="alice", password_hash="x", is_admin=False)
    member = User(username="bob", password_hash="x", is_admin=False)
    db.add_all([creator, member])
    db.commit()
    db.refresh(creator)
    db.refresh(member)

    try:
        yield db, creator, member
    finally:
        # Nested so the engine is disposed even if closing the session raises,
        # while the first failure still propagates.
        try:
            db.close()
        finally:
            engine.dispose()


@pytest.fixture(autouse=True)
def _reset_connector_team_hooks():
    """The hooks are process-global; never leak one into a sibling test."""
    yield
    connector_team_scope.set_connector_team_hooks()


@pytest.fixture()
def _token_resolver_installed():
    """Install a token-resolver hook for the test's duration. Presence is all
    the listing probes (`oauth_token_resolver_installed`), so the body never
    runs."""
    from xagent.web.tools.config import set_oauth_token_resolver_hook

    set_oauth_token_resolver_hook(lambda **_kwargs: None)
    yield
    set_oauth_token_resolver_hook(None)


def _install_visibility(
    user: User, server_ids: set[int], custom_api_ids: set[int] | None = None
) -> None:
    """Answer for ``user`` only, so a test can tell "the overlay ran" apart from
    "the overlay ignored who asked"."""

    def visibility(_db, user_id: int) -> dict[str, set[int]]:
        if int(user_id) != int(user.id):
            return {"mcp": set(), "custom_api": set()}
        return {"mcp": set(server_ids), "custom_api": set(custom_api_ids or set())}

    connector_team_scope.set_connector_team_hooks(visibility=visibility)


def _add_app(
    db,
    app_id: str,
    name: str,
    *,
    transport: str,
    launch_config: dict | None = None,
    provider_name: str | None = None,
    is_visible_in_connector: bool = True,
) -> PublicMCPApp:
    app = PublicMCPApp(
        app_id=app_id,
        name=name,
        description="A catalog app",
        icon="https://example.com/icon.png",
        category="Productivity",
        transport=transport,
        provider_name=provider_name,
        launch_config=launch_config,
        is_visible_in_connector=is_visible_in_connector,
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


def _add_keyless_app(db, app_id: str, name: str, **kwargs) -> PublicMCPApp:
    """The shape with no credential at all — the shared row's launch config is
    everything, so sharing the connector really does share a working one."""
    return _add_app(
        db,
        app_id,
        name,
        transport="stdio",
        launch_config={"command": "npx", "args": ["-y", f"{app_id}-mcp"]},
        **kwargs,
    )


def _add_api_key_app(db, app_id: str, name: str) -> PublicMCPApp:
    """The key-based shape: same shared stdio row as keyless, plus required_env
    the runtime resolves from the row/shared/team/user env layers."""
    return _add_app(
        db,
        app_id,
        name,
        transport="stdio",
        launch_config={
            "command": "npx",
            "args": ["-y", f"{app_id}-mcp"],
            "required_env": ["API_KEY"],
        },
    )


def _add_mcp_oauth_app(db, app_id: str, name: str) -> PublicMCPApp:
    return _add_app(
        db, app_id, name, transport="streamable_http", launch_config=_MCP_OAUTH_LAUNCH
    )


def _add_builtin_oauth_app(db, app_id: str, name: str) -> PublicMCPApp:
    return _add_app(db, app_id, name, transport="oauth", provider_name=app_id)


def _add_server_row(db, config: dict) -> MCPServer:
    server = MCPServer.from_config(config)
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _add_catalog_server(db, app_id: str) -> MCPServer:
    """The shared stdio row a key-based/keyless connect writes: named after the
    app_id (`_ensure_catalog_app_server`), with no association of its own."""
    return _add_server_row(
        db,
        {
            "name": app_id,
            "description": "A catalog app",
            "managed": "external",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", f"{app_id}-mcp"],
        },
    )


def _add_mcp_oauth_server(db, app_id: str) -> MCPServer:
    """The shared remote row `_ensure_catalog_mcp_oauth_server` writes."""
    return _add_server_row(
        db,
        {
            "name": app_id,
            "managed": "external",
            "transport": "streamable_http",
            "url": _MCP_OAUTH_LAUNCH["url"],
            "auth": dict(_MCP_OAUTH_LAUNCH["auth"]),
        },
    )


def _add_builtin_oauth_server(db, app_id: str, name: str) -> MCPServer:
    """The row `_ensure_user_mcp_server` writes: named after the *display name*,
    transport "oauth", carrying the app_id back-reference in `auth`."""
    return _add_server_row(
        db,
        {
            "name": name,
            "managed": "external",
            "transport": "oauth",
            "auth": {"app_id": app_id, "provider": app_id},
        },
    )


def _associate(db, user: User, server: MCPServer, *, is_active: bool = True) -> None:
    """The association a catalog connect writes: the connecting user never owns
    the shared row (`is_owner=False`)."""
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            can_edit=False,
            can_delete=True,
            is_active=is_active,
        )
    )
    db.commit()


def _add_active_grant(db, user: User, server: MCPServer) -> None:
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
            resource=_MCP_OAUTH_LAUNCH["url"],
            scope="",
            access_token=encrypt_value("runtime-token"),
            status="active",
        )
    )
    db.commit()


def _add_oauth_account(db, user: User, provider: str) -> None:
    db.add(
        UserOAuth(
            user_id=user.id,
            provider=provider,
            access_token="provider-token",
            refresh_token="provider-refresh",
            email="bob@example.com",
        )
    )
    db.commit()


def _entries(db, user: User, app_id: str, *, location: str = "all", **kwargs) -> list:
    return [
        a
        for a in list_mcp_apps(location=location, current_user=user, db=db, **kwargs)
        if a["id"] == app_id
    ]


def _entry(db, user: User, app_id: str, *, location: str = "all", **kwargs) -> dict:
    found = _entries(db, user, app_id, location=location, **kwargs)
    assert len(found) == 1, f"expected exactly one {app_id} entry, got {len(found)}"
    return found[0]


def test_a_team_shared_catalog_connector_is_offered_in_catalog_shape(db_session):
    """AC1: exactly once, in catalog shape, with no `is_custom` twin.

    The Local tab stays empty on purpose — the catalog skip that keeps #1346
    fixed is what removes the duplicate, and this is the entry that replaces it.
    """
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True
    assert entry.get("is_custom") is not True
    assert entry["name"] == "Acme Notes"
    assert entry["icon"] == "https://example.com/icon.png"
    assert entry["category"] == "Productivity"
    assert entry["auth_type"] == "keyless"

    assert _entries(db, member, "acme-notes", location="local") == []
    assert len(_entries(db, member, "acme-notes", location="remote")) == 1


def test_a_team_shared_catalog_connector_is_not_reported_connected(db_session):
    """`is_connected` keeps answering "connected *for me*", which only a personal
    association establishes. Claiming otherwise would show the member a
    Configure button for a connector they do not own, and hide the Connect route
    the per-member auth shapes still need."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_connected"] is False
    assert entry["can_authorize"] is False
    assert "server_id" not in entry


def test_the_picker_and_the_tools_page_agree_on_a_team_shared_connector(db_session):
    """AC2: `/api/mcp/servers` overlays team ids with no catalog skip, so it has
    always listed this connector. The disagreement that opened #1387 was the
    picker having no *usable* entry for what the Tools page showed — the catalog
    entry was always rendered, as an app the member would have to connect for
    themselves, so agreement is asserted on attachability rather than on the
    entry's mere presence."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _install_visibility(member, {int(server.id)})

    listed = get_mcp_servers(current_user=member, db=db)
    assert [s.name for s in listed] == ["acme-notes"]
    assert _entry(db, member, "acme-notes")["can_attach"] is True


def test_a_member_holding_both_a_personal_row_and_a_team_link_sees_one_entry(
    db_session,
):
    """AC3: the two states coexist on one entry rather than producing two."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _associate(db, member, server)
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_connected"] is True
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True
    assert entry["server_id"] == server.id


def test_a_deactivated_personal_row_does_not_revoke_team_visibility(db_session):
    """`is_active` gates the personal arm alone (`connector_visible_to_user`), so
    a member who turned their own connection off keeps the team's."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _associate(db, member, server, is_active=False)
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_connected"] is False
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True


def test_a_team_shared_api_key_connector_is_attachable_without_the_members_own_key(
    db_session,
):
    """Deliberate, not an oversight (#1403 review N2): api_key is not
    credential-gated, mirroring every existing path — a personally connected
    key app reports can_attach on the association alone, and the local branch
    gates no shape but mcp_oauth. The key the runtime resolves may live on the
    row's platform env, the shared layer, or the governing agent's team row,
    none of which this user-scoped endpoint can rule out; saying no on "the
    member set no key of their own" would refuse where the runtime succeeds,
    the one direction _local_mcp_can_attach's contract forbids. The env-source
    flags carry the needs-config signal instead."""
    db, _creator, member = db_session
    _add_api_key_app(db, "acme-crm", "Acme CRM")
    server = _add_catalog_server(db, "acme-crm")
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-crm")
    assert entry["auth_type"] == "api_key"
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True
    assert entry["is_connected"] is False
    # No key anywhere in this fixture — the flags say so; attachability doesn't.
    assert entry["user_env_configured"] is False
    assert entry["shared_env_available"] is False
    assert entry["platform_env_available"] is False


def test_env_source_flags_report_the_layer_that_covers_a_team_shared_key_app(
    db_session,
):
    """The needs-config signal the optimistic can_attach leans on: when a key
    layer this endpoint *can* see covers required_env, the matching flag says
    so. The platform layer is the row's own env; the shared layer is the
    application-injected per-user hook. (The governing agent's team layer is
    runtime-side and deliberately untested here — that basis question is
    #1366's.)"""
    from xagent.core.utils.encryption import encrypt_env_dict
    from xagent.web.services.mcp_runtime import set_mcp_shared_env_hook

    db, _creator, member = db_session
    _add_api_key_app(db, "acme-crm", "Acme CRM")
    server = _add_catalog_server(db, "acme-crm")
    _install_visibility(member, {int(server.id)})

    server.env = encrypt_env_dict({"API_KEY": "platform-key"})
    db.commit()
    entry = _entry(db, member, "acme-crm")
    assert entry["platform_env_available"] is True
    assert entry["shared_env_available"] is False
    assert entry["can_attach"] is True
    assert entry["is_connected"] is False

    set_mcp_shared_env_hook(lambda _db, _uid: {int(server.id): {"API_KEY": "team-key"}})
    try:
        entry = _entry(db, member, "acme-crm")
        assert entry["shared_env_available"] is True
    finally:
        set_mcp_shared_env_hook(None)


def test_a_members_own_key_flags_user_env_on_a_team_shared_key_app(db_session):
    """The user layer: a member who additionally connected the app with their
    own key sees user_env_configured, alongside — not instead of — the team
    provenance."""
    from xagent.core.utils.encryption import encrypt_env_dict

    db, _creator, member = db_session
    _add_api_key_app(db, "acme-crm", "Acme CRM")
    server = _add_catalog_server(db, "acme-crm")
    _associate(db, member, server)
    assoc = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == member.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    assoc.env = encrypt_env_dict({"API_KEY": "member-key"})
    db.commit()
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-crm")
    assert entry["user_env_configured"] is True
    assert entry["is_connected"] is True
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True


def test_a_partially_covered_required_env_does_not_flag_the_layer(db_session):
    """_env_covers_required is all-or-nothing: a layer that covers one of two
    required keys advertises nothing, so the picker never offers "use the
    platform key" for a key set that would fail at launch."""
    from xagent.core.utils.encryption import encrypt_env_dict

    db, _creator, member = db_session
    _add_app(
        db,
        "acme-crm",
        "Acme CRM",
        transport="stdio",
        launch_config={
            "command": "npx",
            "args": ["-y", "acme-crm-mcp"],
            "required_env": ["API_KEY", "REGION"],
        },
    )
    server = _add_server_row(
        db,
        {
            "name": "acme-crm",
            "description": "A catalog app",
            "managed": "external",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "acme-crm-mcp"],
        },
    )
    server.env = encrypt_env_dict({"API_KEY": "platform-key"})
    db.commit()
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-crm")
    assert entry["platform_env_available"] is False
    assert entry["is_team_shared"] is True
    # Attachability stays optimistic by contract; the flags carry the truth.
    assert entry["can_attach"] is True


def test_a_team_row_running_a_foreign_command_is_not_claimed(db_session):
    """The connect endpoints 409 on a same-named row whose launch config
    differs ("a victim would run a foreign command with their own key
    attached"), so the personal path can never bind catalog branding to a
    foreign row. The team path has no write-time gate — the hook shares
    whatever rows the application names — so the same comparison runs at read
    time (#1403 review N1): the mismatched row yields no team-shared entry,
    falling back to the pre-#1387 state (listed by /api/mcp/servers, never
    rendered in catalog shape)."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    foreign = _add_server_row(
        db,
        {
            "name": "acme-notes",
            "managed": "external",
            "transport": "stdio",
            "command": "evil-mcp",
        },
    )
    _install_visibility(member, {int(foreign.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_team_shared"] is False
    assert entry["can_attach"] is False
    # The local branch's catalog skip still suppresses the raw row, so the
    # foreign command never reaches the picker under any name.
    assert _entries(db, member, "acme-notes", location="local") == []


def test_a_team_row_pointing_at_a_foreign_url_is_not_claimed(db_session):
    """The mcp_oauth arm of the same guard: identity is the remote URL, the
    field _ensure_catalog_mcp_oauth_server refuses to adopt when it differs."""
    db, _creator, member = db_session
    _add_mcp_oauth_app(db, "acme-remote", "Acme Remote")
    foreign = _add_server_row(
        db,
        {
            "name": "acme-remote",
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.evil.example.com/mcp",
            "auth": dict(_MCP_OAUTH_LAUNCH["auth"]),
        },
    )
    _install_visibility(member, {int(foreign.id)})

    entry = _entry(db, member, "acme-remote")
    assert entry["is_team_shared"] is False
    assert entry["can_attach"] is False


@pytest.mark.parametrize(
    "auth",
    [{"type": "bearer", "bearer_token": "shared-token"}, None],
    ids=["bearer", "no-auth"],
)
def test_a_team_row_with_a_foreign_auth_shape_is_not_claimed(db_session, auth):
    """The same URL is not enough for the mcp_oauth identity: the auth shape is
    part of it. A same-URL row whose auth is bearer or absent reads as a
    non-oauth shape to _local_mcp_can_attach, which would skip the grant gate
    every real mcp_oauth row is subject to — catalog branding on different
    credential semantics (#1403 review, F1 round 2)."""
    db, _creator, member = db_session
    _add_mcp_oauth_app(db, "acme-remote", "Acme Remote")
    config = {
        "name": "acme-remote",
        "managed": "external",
        "transport": "streamable_http",
        "url": _MCP_OAUTH_LAUNCH["url"],
    }
    if auth is not None:
        config["auth"] = auth
    row = _add_server_row(db, config)
    _install_visibility(member, {int(row.id)})

    entry = _entry(db, member, "acme-remote")
    assert entry["is_team_shared"] is False
    assert entry["can_attach"] is False


def test_a_user_owned_row_is_not_rendered_under_catalog_branding(db_session):
    """Read-time analog of _reject_user_owned_catalog_squat (#1403 review
    round 4): the legitimate shared row for the non-oauth shapes never has an
    owner, so a user-owned row squatting a catalog identity — creatable only
    before the app was seeded — must not be officialized even while its launch
    happens to match: its owner keeps edit rights and could swap in a foreign
    command *after* a member attaches the branded entry. Falls back to the
    pre-#1387 state (Tools page only)."""
    db, creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    squatter = _add_catalog_server(db, "acme-notes")
    db.add(
        UserMCPServer(
            user_id=creator.id,
            mcpserver_id=squatter.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.commit()
    _install_visibility(member, {int(squatter.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_team_shared"] is False
    assert entry["can_attach"] is False
    assert _entries(db, member, "acme-notes", location="local") == []


def test_an_owned_builtin_oauth_row_keeps_its_team_entry(db_session):
    """The owner filter is scoped to the non-oauth arm on purpose:
    builtin_oauth provisioning makes the first connector an owner by design
    (_ensure_user_mcp_server writes is_owner=True), so filtering that arm
    would break every legitimately team-shared builtin_oauth connector."""
    db, creator, member = db_session
    _add_builtin_oauth_app(db, "acme-drive", "Acme Drive")
    server = _add_builtin_oauth_server(db, "acme-drive", "Acme Drive")
    db.add(
        UserMCPServer(
            user_id=creator.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    _add_oauth_account(db, member, "acme-drive")
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-drive")
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True


def test_a_colliding_foreign_row_does_not_mask_the_official_row(db_session):
    """Two team-visible rows can normalize to one (transport, name) key —
    "Acme Notes" and `acme-notes` — and the index keeps every candidate, so
    the earlier-sorted foreign row gets no veto: the team resolution scans
    candidates against the official launch config and claims the row that
    actually runs it (#1403 review round 2)."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    # Inserted first, so it sorts first in every candidate list.
    foreign = _add_server_row(
        db,
        {
            "name": "Acme Notes",
            "managed": "external",
            "transport": "stdio",
            "command": "evil-mcp",
        },
    )
    official = _add_catalog_server(db, "acme-notes")
    _install_visibility(member, {int(foreign.id), int(official.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True


def test_a_team_shared_mcp_oauth_connector_is_not_attachable_without_a_grant(
    db_session,
):
    """AC4: authorization is per user. The shared row carries no token for the
    member, so the entry is offered but not attachable — its card keeps the
    catalog Connect route ("connect it for yourself"), dispatched on
    `auth_type`, rather than the per-server route that would 404 for want of a
    personal association."""
    db, _creator, member = db_session
    _add_mcp_oauth_app(db, "acme-remote", "Acme Remote")
    server = _add_mcp_oauth_server(db, "acme-remote")
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-remote")
    assert entry["is_team_shared"] is True
    assert entry["is_connected"] is False
    assert entry["can_attach"] is False
    assert entry["can_authorize"] is False
    assert entry["auth_type"] == "mcp_oauth"


def test_a_team_shared_mcp_oauth_connector_is_attachable_once_the_member_consents(
    db_session,
):
    """The grant is the credential term, and it is the member's own."""
    db, _creator, member = db_session
    _add_mcp_oauth_app(db, "acme-remote", "Acme Remote")
    server = _add_mcp_oauth_server(db, "acme-remote")
    _associate(db, member, server)
    _add_active_grant(db, member, server)
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-remote")
    assert entry["is_connected"] is True
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True


def test_a_team_shared_builtin_oauth_connector_needs_the_members_own_account(
    db_session,
):
    """The provider login is held on `UserOAuth`, never on the shared row, so
    sharing the connector shares no credential for this shape either — the term
    `_local_mcp_can_attach` never needed, because a catalog-keyed row never
    reaches the local branch."""
    db, _creator, member = db_session
    _add_builtin_oauth_app(db, "acme-drive", "Acme Drive")
    server = _add_builtin_oauth_server(db, "acme-drive", "Acme Drive")
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-drive")
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is False

    _add_oauth_account(db, member, "acme-drive")
    entry = _entry(db, member, "acme-drive")
    assert entry["can_attach"] is True
    # Still not a personal connection: the member has an account but no
    # association, which is what is_connected reports on.
    assert entry["is_connected"] is False


def test_a_renamed_builtin_oauth_apps_row_still_reports_its_app_id(db_session):
    """/api/mcp/servers enrichment resolves by stable auth.app_id (with a name
    fallback for legacy rows), not by name alone: an admin renaming an app
    leaves the provisioned row under the old display name, and the name-only
    lookup reported app_id None for exactly the row the catalog entry resolves
    through auth.app_id. The picker's selector resolution keys on this app_id
    to map the entry back to its real row; without it, the persisted selector
    was the new display name of a row the runtime knows by the old one — zero
    tools, silently (#1403 review round 3)."""
    db, _creator, member = db_session
    _add_builtin_oauth_app(db, "acme-drive", "Acme Drive v2")
    server = _add_builtin_oauth_server(db, "acme-drive", "Acme Drive")
    _install_visibility(member, {int(server.id)})

    listed = get_mcp_servers(current_user=member, db=db)
    row = next(s for s in listed if s.name == "Acme Drive")
    assert row.app_id == "acme-drive"
    # The catalog entry keeps resolving the renamed row through auth.app_id.
    assert _entry(db, member, "acme-drive")["is_team_shared"] is True


def test_a_team_shared_builtin_oauth_connector_attaches_through_a_resolver_hook(
    db_session, _token_resolver_installed
):
    """The resolver hook outranks the account check: the runtime resolves every
    transport=="oauth" server's token through the installed hook first and
    falls back to UserOAuth only without one, so on a resolver deployment the
    row runs with no account at all. Refusing here would say no where the
    runtime succeeds — the one direction the attachability contract forbids
    (#1403 review round 2)."""
    db, _creator, member = db_session
    _add_builtin_oauth_app(db, "acme-drive", "Acme Drive")
    server = _add_builtin_oauth_server(db, "acme-drive", "Acme Drive")
    _install_visibility(member, {int(server.id)})

    entry = _entry(db, member, "acme-drive")
    assert entry["is_team_shared"] is True
    assert entry["can_attach"] is True
    # The hook supplies tokens, not a personal connection.
    assert entry["is_connected"] is False


def test_a_hidden_catalog_app_stays_hidden_when_team_shared(db_session):
    """Strong hide mode removes the app for everyone, connected or not. The
    local branch's catalog skip is deliberately broader than the catalog
    branch's claim so a hidden app cannot fall through to a custom entry."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes", is_visible_in_connector=False)
    server = _add_catalog_server(db, "acme-notes")
    _install_visibility(member, {int(server.id)})

    assert _entries(db, member, "acme-notes") == []


def test_the_verified_filter_keeps_a_team_shared_catalog_connector(db_session):
    """The filter narrows to connectors backing a real connection, which a
    team-shared one does. Withholding it would put the connector back out of
    reach for a member browsing under the filter."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    _add_keyless_app(db, "acme-other", "Acme Other")
    server = _add_catalog_server(db, "acme-notes")
    _install_visibility(member, {int(server.id)})

    ids = [
        a["id"]
        for a in list_mcp_apps(
            location="remote", status="verified", current_user=member, db=db
        )
    ]
    assert ids == ["acme-notes"]


def test_a_connector_shared_with_another_team_is_not_offered(db_session):
    """AC5's other half: the overlay is keyed on the requesting user, so a
    connector a live hook reports for someone else must not reach this member."""
    db, creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _install_visibility(creator, {int(server.id)})

    assert _entry(db, creator, "acme-notes")["is_team_shared"] is True
    entry = _entry(db, member, "acme-notes")
    assert entry["is_team_shared"] is False
    assert entry["can_attach"] is False


def test_a_hook_answering_string_ids_flags_nothing_it_cannot_attach(db_session):
    """Element types are the hook's contract (`dict[str, set[int]]`), and
    `visible_team_connector_ids` does not yet validate them. SQLite's numeric
    affinity makes `id IN ('1')` match an INTEGER primary key, so a string id
    still reaches the row query that fills the team indexes — the same fail-open
    `test_mcp_apps_team_visibility.py` pins for the local branch.

    What this pins is that the entry stays *coherent* under that answer:
    `is_team_shared` and `can_attach` both resolve through the in-Python
    predicate, which compares int to str and says no, so the response never
    labels a connector as the team's while refusing to attach it."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _install_visibility(member, {str(server.id)})  # type: ignore[arg-type]

    entry = _entry(db, member, "acme-notes")
    assert entry["is_team_shared"] is False
    assert entry["can_attach"] is False


def test_a_provider_only_row_does_not_brand_every_same_provider_app(db_session):
    """Providers are non-unique across apps, so the team oauth index carries no
    bare-provider fallback (#1403 review round 6): a row whose auth.app_id is
    blank must not satisfy every catalog app on its provider. Only the row
    whose stable app_id names an app claims that app; the malformed sibling
    claims nothing."""
    db, _creator, member = db_session
    _add_app(db, "acme-drive", "Acme Drive", transport="oauth", provider_name="acme")
    _add_app(db, "acme-mail", "Acme Mail", transport="oauth", provider_name="acme")
    malformed = _add_server_row(
        db,
        {
            "name": "legacy-acme",
            "managed": "external",
            "transport": "oauth",
            "auth": {"app_id": "   ", "provider": "acme"},
        },
    )
    proper = _add_server_row(
        db,
        {
            "name": "Acme Drive",
            "managed": "external",
            "transport": "oauth",
            "auth": {"app_id": "acme-drive", "provider": "acme"},
        },
    )
    _add_oauth_account(db, member, "acme")
    _install_visibility(member, {int(malformed.id), int(proper.id)})

    assert _entry(db, member, "acme-drive")["is_team_shared"] is True
    entry = _entry(db, member, "acme-mail")
    assert entry["is_team_shared"] is False
    assert entry["can_attach"] is False


def test_a_provider_named_row_does_not_brand_every_same_provider_app(db_session):
    """The collision the provider-key removal alone cannot close (#1403 review
    round 7): a malformed row *named* the provider string itself lands under
    that key through its own name. The team matcher therefore accepts exact
    identity only — a nonblank auth.app_id equal to the app's id, or a name
    among the app's own catalog keys — never the provider metadata the personal
    matcher tolerates, so this row brands nothing."""
    db, _creator, member = db_session
    _add_app(db, "acme-drive", "Acme Drive", transport="oauth", provider_name="acme")
    _add_app(db, "acme-mail", "Acme Mail", transport="oauth", provider_name="acme")
    row = _add_server_row(
        db,
        {
            "name": "acme",
            "managed": "external",
            "transport": "oauth",
            "auth": {"app_id": "   ", "provider": "acme"},
        },
    )
    _add_oauth_account(db, member, "acme")
    _install_visibility(member, {int(row.id)})

    for app_id in ("acme-drive", "acme-mail"):
        entry = _entry(db, member, app_id)
        assert entry["is_team_shared"] is False
        assert entry["can_attach"] is False


def test_a_raising_visibility_hook_yields_a_typed_503(db_session):
    """The hook is optional application code and the default remote picker now
    runs it on every load; a raising hook must surface as this seam's typed
    503 (mirroring resolve_team_connector_ids_or_raise), not a bare 500
    (#1403 review round 6)."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")

    def raising(_db, _user_id):
        raise RuntimeError("hook exploded")

    connector_team_scope.set_connector_team_hooks(visibility=raising)
    with pytest.raises(HTTPException) as exc_info:
        list_mcp_apps(location="remote", current_user=member, db=db)
    assert exc_info.value.status_code == 503


@pytest.mark.parametrize(
    "answer",
    [None, {"mcp": set()}, {"mcp": [1], "custom_api": set()}],
    ids=["none", "missing-key", "list-not-set"],
)
def test_a_malformed_outer_hook_answer_yields_a_typed_503(db_session, answer):
    """Outer shape only — element-level id validation stays with the accessor
    (#1244). A hook answering something that is not {mcp: set, custom_api: set}
    is malformed authorization input and fails loudly with the typed 503,
    never a KeyError-shaped 500."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    connector_team_scope.set_connector_team_hooks(visibility=lambda _db, _uid: answer)

    with pytest.raises(HTTPException) as exc_info:
        list_mcp_apps(location="remote", current_user=member, db=db)
    assert exc_info.value.status_code == 503


def test_a_standalone_deployment_is_unchanged(db_session):
    """AC5: no hook installed resolves empty, so every entry keeps the
    pre-#1387 answer — `can_attach` is exactly `is_connected`, and the
    is_team_shared field is absent entirely, keeping standalone payloads
    identical to what they were before this change. Present-false is reserved
    for deployments where the concept exists (hook installed, nothing shared
    with this user)."""
    db, _creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    _add_mcp_oauth_app(db, "acme-remote", "Acme Remote")
    server = _add_catalog_server(db, "acme-notes")
    _associate(db, member, server)

    remote = list_mcp_apps(location="remote", current_user=member, db=db)
    assert {a["id"] for a in remote} == {"acme-notes", "acme-remote"}
    for app in remote:
        assert "is_team_shared" not in app
        assert app["can_attach"] is app["is_connected"]

    local = list_mcp_apps(location="local", current_user=member, db=db)
    for app in local:
        assert "is_team_shared" not in app


def test_an_installed_hook_answering_empty_still_emits_the_field(db_session):
    """The field's absence means "this deployment has no team sharing", never
    "nothing is shared with you" — an installed hook answering empty is the
    latter, and consumers may rely on present-false to tell the two apart."""
    db, creator, member = db_session
    _add_keyless_app(db, "acme-notes", "Acme Notes")
    server = _add_catalog_server(db, "acme-notes")
    _associate(db, member, server)
    # Hook answers only for the creator; the member gets empty sets.
    _install_visibility(creator, {int(server.id)})

    entry = _entry(db, member, "acme-notes")
    assert entry["is_team_shared"] is False
    assert entry["is_connected"] is True


def _add_custom_api(db, name: str, *, owner: User | None = None) -> CustomApi:
    """A Custom API row; associated to ``owner`` when given, otherwise reachable
    only through the team overlay."""
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
                is_active=True,
            )
        )
        db.commit()
    return api


def test_the_local_branch_flags_a_team_shared_custom_connector(db_session):
    """The field means the same thing on both branches and both local loops
    (MCP servers and Custom APIs), so a consumer never has to read its absence
    as "no"."""
    db, _creator, member = db_session
    shared = _add_server_row(
        db,
        {
            "name": "records",
            "managed": "external",
            "transport": "stdio",
            "command": "records-mcp",
        },
    )
    personal = _add_server_row(
        db,
        {
            "name": "ledger",
            "managed": "external",
            "transport": "stdio",
            "command": "ledger-mcp",
        },
    )
    _associate(db, member, personal)
    shared_api = _add_custom_api(db, "reports")
    _add_custom_api(db, "billing", owner=member)
    _install_visibility(member, {int(shared.id)}, {int(shared_api.id)})

    local = {
        a["id"]: a for a in list_mcp_apps(location="local", current_user=member, db=db)
    }
    assert local["records"]["is_team_shared"] is True
    assert local["ledger"]["is_team_shared"] is False
    assert local["reports"]["is_team_shared"] is True
    assert local["reports"]["transport"] == "custom_api"
    assert local["billing"]["is_team_shared"] is False
