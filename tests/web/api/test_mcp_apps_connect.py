"""Tests for connecting key-based (non-oauth) catalog MCP apps.

Covers the connector "connect" path for apps like Google Maps that authenticate
with a static API key rather than OAuth: one shared server row, many users, each
with their own per-user env (their key). See PR #750 for the per-user env layer
this builds on.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import decrypt_env_dict
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    db.add(User(username="alice", id=1, password_hash="x"))
    db.add(User(username="bob", id=2, password_hash="x"))
    db.add(
        PublicMCPApp(
            app_id="google-maps",
            name="google-maps",
            description="Google Maps",
            transport="stdio",
            launch_config={
                "command": "npx",
                "args": ["-y", "@cablate/mcp-google-map", "--stdio"],
                "required_env": ["GOOGLE_MAPS_API_KEY"],
            },
        )
    )
    db.commit()
    yield db
    db.close()


def _user(db, uid):
    return db.query(User).filter(User.id == uid).first()


def test_connect_provisions_shared_server_with_per_user_env(test_db):
    """First connect creates the shared stdio server + a non-owner association
    whose per-user env holds the caller's key, encrypted at rest."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "alice-key"}),
        current_user=_user(test_db, 1),
        db=test_db,
    )

    server = test_db.query(MCPServer).filter(MCPServer.name == "google-maps").first()
    assert server is not None
    assert server.transport == "stdio"
    assert server.command == "npx"
    assert server.args == ["-y", "@cablate/mcp-google-map", "--stdio"]

    assoc = (
        test_db.query(UserMCPServer)
        .filter(UserMCPServer.user_id == 1, UserMCPServer.mcpserver_id == server.id)
        .first()
    )
    assert assoc is not None
    # Connect users never own the shared global config (they can't edit global env).
    assert assoc.is_owner is False
    # Key stored encrypted, decrypts back to plaintext.
    assert assoc.env != {"GOOGLE_MAPS_API_KEY": "alice-key"}
    assert decrypt_env_dict(assoc.env) == {"GOOGLE_MAPS_API_KEY": "alice-key"}


def test_second_user_joins_same_server_with_own_key(test_db):
    """A second user connecting reuses the one shared server row (no name clash)
    and gets an independent per-user key."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "alice-key"}),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "bob-key"}),
        current_user=_user(test_db, 2),
        db=test_db,
    )

    servers = test_db.query(MCPServer).filter(MCPServer.name == "google-maps").all()
    assert len(servers) == 1  # shared, not one row per user

    assocs = test_db.query(UserMCPServer).filter(
        UserMCPServer.mcpserver_id == servers[0].id
    )
    by_user = {a.user_id: decrypt_env_dict(a.env) for a in assocs}
    assert by_user == {
        1: {"GOOGLE_MAPS_API_KEY": "alice-key"},
        2: {"GOOGLE_MAPS_API_KEY": "bob-key"},
    }


def test_reconnect_updates_own_key(test_db):
    """Connecting again updates the caller's key rather than erroring."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    u = _user(test_db, 1)
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "old"}),
        current_user=u,
        db=test_db,
    )
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "new"}),
        current_user=u,
        db=test_db,
    )

    assocs = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).all()
    assert len(assocs) == 1
    assert decrypt_env_dict(assocs[0].env) == {"GOOGLE_MAPS_API_KEY": "new"}


async def test_disconnect_keeps_shared_server_for_other_users(test_db):
    """A connect user (non-owner) can disconnect their own association; the
    shared server row survives while another user is still connected, and is
    removed only when the last user leaves."""
    from xagent.web.api.mcp import (
        MCPAppConnectRequest,
        connect_mcp_app,
        delete_mcp_server,
    )

    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "a"}),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "b"}),
        current_user=_user(test_db, 2),
        db=test_db,
    )
    server_id = (
        test_db.query(MCPServer).filter(MCPServer.name == "google-maps").first().id
    )

    # Alice (non-owner) disconnects — allowed via can_delete, row must survive.
    await delete_mcp_server(server_id, current_user=_user(test_db, 1), db=test_db)
    assert (
        test_db.query(MCPServer).filter(MCPServer.id == server_id).first() is not None
    )
    assert (
        test_db.query(UserMCPServer)
        .filter(UserMCPServer.mcpserver_id == server_id)
        .count()
        == 1
    )

    # Last user leaves — shared row is cleaned up.
    await delete_mcp_server(server_id, current_user=_user(test_db, 2), db=test_db)
    assert test_db.query(MCPServer).filter(MCPServer.id == server_id).first() is None


def test_connect_with_blank_key_uses_global_fallback(test_db):
    """A user may connect without supplying a key (relying on the admin-set
    global env). No per-user override is stored, and a blank value never
    overrides the global key."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    # No key at all.
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env=None),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    assoc1 = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assert assoc1 is not None
    assert assoc1.env is None  # falls back to global at runtime

    # Blank value must not be persisted as an override that would blank the global.
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "   "}),
        current_user=_user(test_db, 2),
        db=test_db,
    )
    assoc2 = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 2).first()
    assert assoc2.env is None


def test_platform_and_shared_env_available_flags(test_db):
    """The catalog exposes two distinct shared-key flags: the platform-global env
    on the server row (platform_env_available) and an app-injected shared layer,
    e.g. a team key (shared_env_available), so the UI can offer either source."""
    from xagent.core.utils.encryption import encrypt_env_dict
    from xagent.web.api.mcp import (
        _app_platform_env_available,
        _app_shared_env_available,
    )

    app = {
        "id": "google-maps",
        "name": "google-maps",
        "transport": "stdio",
        "launch_config": {"required_env": ["GOOGLE_MAPS_API_KEY"]},
    }

    # No server row yet -> neither available.
    assert _app_platform_env_available(app, None) is False
    assert _app_shared_env_available(app, None, {}) is False

    server = MCPServer(
        name="google-maps", transport="stdio", managed="external", command="npx"
    )
    test_db.add(server)
    test_db.commit()

    # Server exists but no shared env anywhere -> neither available.
    assert _app_platform_env_available(app, server) is False
    assert _app_shared_env_available(app, server, {}) is False

    # Platform-global env on the row covers the required key -> platform available,
    # shared (injected layer) still not.
    server.env = encrypt_env_dict({"GOOGLE_MAPS_API_KEY": "platform"})
    test_db.commit()
    assert _app_platform_env_available(app, server) is True
    assert _app_shared_env_available(app, server, {}) is False

    # An injected shared layer (decrypted, keyed by server id) covers it ->
    # shared available.
    server.env = None
    test_db.commit()
    injected = {server.id: {"GOOGLE_MAPS_API_KEY": "team"}}
    assert _app_shared_env_available(app, server, injected) is True
    # Injected layer missing the key -> not available.
    assert _app_shared_env_available(app, server, {server.id: {"X": "y"}}) is False


def test_user_env_configured_reflects_own_key(test_db):
    """The catalog exposes whether the current user has their own per-user key
    (vs relying on the admin's global key), so the manage dialog can show it."""
    from xagent.web.api.mcp import (
        MCPAppConnectRequest,
        _app_user_env_configured,
        connect_mcp_app,
    )

    app = {
        "id": "google-maps",
        "name": "google-maps",
        "transport": "stdio",
        "launch_config": {"required_env": ["GOOGLE_MAPS_API_KEY"]},
    }

    def _server():
        return test_db.query(MCPServer).filter(MCPServer.name == "google-maps").first()

    def _um_by_id(uid):
        return {
            um.mcpserver_id: um
            for um in test_db.query(UserMCPServer)
            .filter(UserMCPServer.user_id == uid)
            .all()
        }

    # Connected with a blank key (using admin global) -> no own key.
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env=None),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    assert _app_user_env_configured(app, _server(), _um_by_id(1)) is False

    # Connected with own key -> configured.
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "mine"}),
        current_user=_user(test_db, 2),
        db=test_db,
    )
    assert _app_user_env_configured(app, _server(), _um_by_id(2)) is True


def test_connect_only_stores_required_env_keys(test_db):
    """Only the app's declared required_env keys are persisted; arbitrary keys
    (e.g. NODE_OPTIONS/LD_PRELOAD) must not reach the stdio subprocess."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(
            env={
                "GOOGLE_MAPS_API_KEY": "ok",
                "NODE_OPTIONS": "--require /evil.js",
                "LD_PRELOAD": "/evil.so",
            }
        ),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    assoc = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assert decrypt_env_dict(assoc.env) == {"GOOGLE_MAPS_API_KEY": "ok"}


def test_reconnect_preserves_disabled_state(test_db):
    """Reconnecting (e.g. to update the key) must not silently re-enable a
    connection the user has toggled off."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    u = _user(test_db, 1)
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "k"}),
        current_user=u,
        db=test_db,
    )
    assoc = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assoc.is_active = False
    test_db.commit()

    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "k2"}),
        current_user=u,
        db=test_db,
    )
    test_db.refresh(assoc)
    assert assoc.is_active is False


def test_reconnect_keeps_key_when_not_resupplied(test_db):
    """Reconnecting with a masked value keeps the stored key (manage dialog must
    not wipe the secret when the user doesn't retype it)."""
    from xagent.core.tools.core.mcp.model import MASKED_SECRET_VALUE
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    u = _user(test_db, 1)
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "secret"}),
        current_user=u,
        db=test_db,
    )
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": MASKED_SECRET_VALUE}),
        current_user=u,
        db=test_db,
    )
    assoc = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assert decrypt_env_dict(assoc.env) == {"GOOGLE_MAPS_API_KEY": "secret"}


def test_reconnect_without_env_keeps_key(test_db):
    """An is_active-only reconnect (env omitted) must preserve the stored key
    rather than wiping it to the global fallback."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    u = _user(test_db, 1)
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "secret"}),
        current_user=u,
        db=test_db,
    )
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(is_active=False),
        current_user=u,
        db=test_db,
    )
    assoc = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assert assoc.is_active is False
    assert decrypt_env_dict(assoc.env) == {"GOOGLE_MAPS_API_KEY": "secret"}


def test_connect_rejects_hijacked_server_with_foreign_config(test_db):
    """A pre-existing row under the catalog id with a different command must not
    be reused — otherwise a victim runs an attacker's command with their key."""
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    test_db.add(
        MCPServer(
            name="google-maps",
            managed="external",
            transport="stdio",
            command="/bin/evil",
            args=["--pwn"],
        )
    )
    test_db.commit()

    with pytest.raises(HTTPException) as exc:
        connect_mcp_app(
            "google-maps",
            MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "victim-key"}),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 409


def test_connect_rejects_user_owned_server_even_with_matching_config(test_db):
    """A row under the catalog id that a user OWNS is a custom server squatting
    the id (creatable only before the app was seeded). Even with a config that
    matches the official launch, it must not be adopted as the shared row: the
    owner keeps edit rights and could later swap in a foreign command that every
    connected user runs."""
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    # Matching official config, but owned by user 1 (the attacker).
    server = MCPServer(
        name="google-maps",
        managed="external",
        transport="stdio",
        command="npx",
        args=["-y", "@cablate/mcp-google-map", "--stdio"],
    )
    test_db.add(server)
    test_db.commit()
    test_db.add(
        UserMCPServer(user_id=1, mcpserver_id=server.id, is_owner=True, can_edit=True)
    )
    test_db.commit()

    # A different user connecting via the catalog must be rejected, not handed the
    # attacker-owned row.
    with pytest.raises(HTTPException) as exc:
        connect_mcp_app(
            "google-maps",
            MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "victim-key"}),
            current_user=_user(test_db, 2),
            db=test_db,
        )
    assert exc.value.status_code == 409


@pytest.mark.parametrize("name", ["google-maps", "Google-Maps", "google maps"])
def test_create_server_rejects_catalog_app_id(test_db, name):
    """Custom servers can't squat a catalog app id (the hijack precondition),
    including case/spacing variants the app-matching normalizes together."""
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPServerCreate, create_mcp_server

    with pytest.raises(HTTPException) as exc:
        create_mcp_server(
            MCPServerCreate(
                name=name,
                transport="stdio",
                config={"command": "/bin/evil", "args": ["--pwn"]},
            ),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 400


def test_connect_rejects_oauth_app(test_db):
    """OAuth apps must go through the OAuth flow, not this key-based path."""
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    test_db.add(
        PublicMCPApp(
            app_id="gmail", name="gmail", transport="oauth", provider_name="google"
        )
    )
    test_db.commit()

    with pytest.raises(HTTPException) as exc:
        connect_mcp_app(
            "gmail",
            MCPAppConnectRequest(env={"X": "y"}),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 400


def test_connect_keyless_app_creates_association_without_env(test_db):
    """A keyless entry (a stdio command with no required_env, e.g. Chrome)
    connects through the same endpoint: shared server row + non-owner
    association, but with no per-user env at all. Any env a caller does send is
    dropped — required_env is empty, so nothing is allowed through.
    """
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    test_db.add(
        PublicMCPApp(
            app_id="keyless",
            name="keyless",
            transport="stdio",
            launch_config={"command": "npx", "args": ["-y", "some-mcp"]},
        )
    )
    test_db.commit()

    connect_mcp_app(
        "keyless",
        MCPAppConnectRequest(env={"INJECTED": "nope"}),
        current_user=_user(test_db, 1),
        db=test_db,
    )

    server = test_db.query(MCPServer).filter(MCPServer.name == "keyless").first()
    assert server is not None
    assert server.transport == "stdio"
    assert server.command == "npx"
    assert server.args == ["-y", "some-mcp"]

    assoc = (
        test_db.query(UserMCPServer)
        .filter(UserMCPServer.user_id == 1, UserMCPServer.mcpserver_id == server.id)
        .first()
    )
    assert assoc is not None
    assert assoc.is_owner is False
    assert assoc.is_active is True
    # No declared keys -> nothing stored, not even the injected env.
    assert assoc.env is None


def test_connect_rejects_hidden_app(test_db):
    """is_visible_in_connector is a release gate, not just a display toggle:
    a hidden app must not be connectable by anyone who knows the app_id (the
    chrome connector ships hidden until persistent stdio sessions land). The
    gate returns 404 so a hidden app is indistinguishable from a nonexistent
    one, and it must fire before any shared server row is created.
    """
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    test_db.add(
        PublicMCPApp(
            app_id="hidden-keyless",
            name="hidden-keyless",
            transport="stdio",
            is_visible_in_connector=False,
            launch_config={"command": "npx", "args": ["-y", "some-mcp"]},
        )
    )
    test_db.commit()

    with pytest.raises(HTTPException) as exc:
        connect_mcp_app(
            "hidden-keyless",
            MCPAppConnectRequest(is_active=True),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 404

    # The gate fired before provisioning: no shared server row was created.
    assert (
        test_db.query(MCPServer).filter(MCPServer.name == "hidden-keyless").first()
        is None
    )


def test_connect_rejects_the_real_shipped_chrome_app(test_db):
    """End-to-end on the actual app_id="chrome", not a synthetic stand-in:
    shipping hidden is this PR's entire safety story, so it must be pinned
    against the real registry row, not just a hand-built fixture that happens
    to share the shape. The DB row only needs app_id/is_visible_in_connector
    to match the seed migration -- transport/launch_config are overlaid from
    builtin_mcp_registry.py by get_app_by_id regardless of what's stored.
    """
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    registry_row = get_builtin_public_mcp_app("chrome")
    assert registry_row is not None
    assert registry_row["is_visible_in_connector"] is False, (
        "chrome must ship hidden -- if this ever flips to True, this test's "
        "premise (and the PR's rollout safety story) no longer holds"
    )

    test_db.add(
        PublicMCPApp(
            app_id="chrome",
            name="Chrome",
            transport=registry_row["transport"],
            is_visible_in_connector=registry_row["is_visible_in_connector"],
            launch_config=registry_row["launch_config"],
        )
    )
    test_db.commit()

    with pytest.raises(HTTPException) as exc:
        connect_mcp_app(
            "chrome",
            MCPAppConnectRequest(is_active=True),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 404
    assert test_db.query(MCPServer).filter(MCPServer.name == "chrome").first() is None


def test_connect_reactivates_dormant_association_when_requested(test_db):
    """The keyless connect flow sends is_active=True explicitly; the backend
    must flip a dormant (is_active=False) association back on. Guards the
    reactivation semantics the frontend now depends on
    (connect-mcp-dialog.tsx submitKeylessConnect).
    """
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    test_db.add(
        PublicMCPApp(
            app_id="keyless-dormant",
            name="keyless-dormant",
            transport="stdio",
            launch_config={"command": "npx", "args": ["-y", "some-mcp"]},
        )
    )
    test_db.commit()

    # First connect creates the association, then the user turns it off.
    connect_mcp_app(
        "keyless-dormant",
        MCPAppConnectRequest(is_active=True),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    server = test_db.query(MCPServer).filter(MCPServer.name == "keyless-dormant").one()
    assoc = (
        test_db.query(UserMCPServer)
        .filter(UserMCPServer.user_id == 1, UserMCPServer.mcpserver_id == server.id)
        .one()
    )
    assoc.is_active = False
    test_db.commit()

    # Reconnect with is_active=True reactivates; omitting it must NOT
    # (a key-update reconnect may not silently re-enable a disabled row).
    connect_mcp_app(
        "keyless-dormant",
        MCPAppConnectRequest(),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    test_db.refresh(assoc)
    assert assoc.is_active is False

    connect_mcp_app(
        "keyless-dormant",
        MCPAppConnectRequest(is_active=True),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    test_db.refresh(assoc)
    assert assoc.is_active is True


def test_connect_rejects_unconnectable_app(test_db):
    """An entry that classifies as "unconnectable" (required_env but no launch
    command, so nothing to run) is rejected by the connect gate. Such a row can
    only reach the DB directly / pre-validator; the connect gate is the backstop.
    Guards the auth_type not in ("api_key", "keyless") branch of
    _ensure_catalog_app_server.
    """
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    test_db.add(
        PublicMCPApp(
            app_id="broken",
            name="broken",
            transport="stdio",
            launch_config={"required_env": ["KEY"]},
        )
    )
    test_db.commit()

    with pytest.raises(HTTPException) as exc:
        connect_mcp_app(
            "broken",
            MCPAppConnectRequest(env={"X": "y"}),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 400


async def test_last_disconnect_keeps_row_with_platform_key(test_db):
    """When a shared catalog row carries the admin's platform fallback key, the
    last user's disconnect must NOT hard-delete the row — that would silently
    wipe the platform key with no signal to the admin. (A row with no platform
    key still cascades away, see test_disconnect_keeps_shared_server_for_other_users.)
    """
    from xagent.core.utils.encryption import encrypt_env_dict
    from xagent.web.api.mcp import (
        MCPAppConnectRequest,
        connect_mcp_app,
        delete_mcp_server,
    )

    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "alice-key"}),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    server = test_db.query(MCPServer).filter(MCPServer.name == "google-maps").first()
    # Admin attaches a platform fallback key on the shared row.
    server.env = encrypt_env_dict({"GOOGLE_MAPS_API_KEY": "platform-fallback"})
    test_db.commit()
    server_id = server.id

    # The last (only) connected user disconnects.
    await delete_mcp_server(server_id, current_user=_user(test_db, 1), db=test_db)

    kept = test_db.query(MCPServer).filter(MCPServer.id == server_id).first()
    assert kept is not None  # row survives
    assert decrypt_env_dict(kept.env) == {"GOOGLE_MAPS_API_KEY": "platform-fallback"}
    assert (
        test_db.query(UserMCPServer)
        .filter(UserMCPServer.mcpserver_id == server_id)
        .count()
        == 0
    )


def test_concurrent_same_user_connect_is_idempotent(test_db):
    """Two racing connect requests from the same user both pass the initial
    "no association yet" SELECT and try to insert; the loser trips the
    (user_id, mcpserver_id) unique constraint. It must recover as an idempotent
    update, not surface an unhandled 500."""
    from sqlalchemy.exc import IntegrityError

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    # Shared row already exists so _ensure_catalog_app_server does not commit;
    # the only commit inside connect_mcp_app is then our association insert.
    test_db.add(
        MCPServer(
            name="google-maps",
            transport="stdio",
            managed="external",
            command="npx",
            args=["-y", "@cablate/mcp-google-map", "--stdio"],
        )
    )
    test_db.commit()
    server_id = (
        test_db.query(MCPServer).filter(MCPServer.name == "google-maps").first().id
    )

    real_commit = test_db.commit
    state = {"raced": False}

    def racing_commit():
        # On the association insert, simulate the winning concurrent request:
        # drop our own pending insert, commit an identical row so the recovery
        # re-query finds it, then signal that our insert lost the race.
        if not state["raced"]:
            state["raced"] = True
            for obj in list(test_db.new):
                test_db.expunge(obj)
            test_db.add(
                UserMCPServer(user_id=1, mcpserver_id=server_id, is_owner=False)
            )
            real_commit()
            raise IntegrityError("raced", {}, Exception("duplicate"))
        return real_commit()

    test_db.commit = racing_commit
    try:
        connect_mcp_app(
            "google-maps",
            MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "alice-key"}),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    finally:
        test_db.commit = real_commit

    assert state["raced"] is True  # the race path was actually exercised
    assocs = (
        test_db.query(UserMCPServer)
        .filter(UserMCPServer.user_id == 1, UserMCPServer.mcpserver_id == server_id)
        .all()
    )
    assert len(assocs) == 1  # idempotent: one row, no duplicate, no 500
    assert decrypt_env_dict(assocs[0].env) == {"GOOGLE_MAPS_API_KEY": "alice-key"}


def test_concurrent_connect_recovery_keeps_winners_key(test_db):
    """The dangerous direction of the race: an env=None request (e.g. activation
    toggle) loses to a concurrent request that stored a real key. Recovery must
    merge against the winner's *current* row, not overwrite it with the loser's
    stale pre-race merged_env (which is None on the insert path)."""
    from sqlalchemy.exc import IntegrityError

    from xagent.core.utils.encryption import encrypt_env_dict
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    test_db.add(
        MCPServer(
            name="google-maps",
            transport="stdio",
            managed="external",
            command="npx",
            args=["-y", "@cablate/mcp-google-map", "--stdio"],
        )
    )
    test_db.commit()
    server_id = (
        test_db.query(MCPServer).filter(MCPServer.name == "google-maps").first().id
    )

    real_commit = test_db.commit
    state = {"raced": False}

    def racing_commit():
        # The winner commits a real key; then our (env=None) insert loses.
        if not state["raced"]:
            state["raced"] = True
            for obj in list(test_db.new):
                test_db.expunge(obj)
            test_db.add(
                UserMCPServer(
                    user_id=1,
                    mcpserver_id=server_id,
                    is_owner=False,
                    env=encrypt_env_dict({"GOOGLE_MAPS_API_KEY": "winner-key"}),
                )
            )
            real_commit()
            raise IntegrityError("raced", {}, Exception("duplicate"))
        return real_commit()

    test_db.commit = racing_commit
    try:
        connect_mcp_app(
            "google-maps",
            MCPAppConnectRequest(env=None),  # "don't touch my key" toggle
            current_user=_user(test_db, 1),
            db=test_db,
        )
    finally:
        test_db.commit = real_commit

    assert state["raced"] is True
    assoc = (
        test_db.query(UserMCPServer)
        .filter(UserMCPServer.user_id == 1, UserMCPServer.mcpserver_id == server_id)
        .first()
    )
    # The winner's key must survive — not be wiped by the loser's stale env=None.
    assert decrypt_env_dict(assoc.env) == {"GOOGLE_MAPS_API_KEY": "winner-key"}


def test_own_source_with_blank_key_not_persisted_as_own(test_db):
    """Selecting env_source="own" but submitting no usable key must not persist a
    misleading "own" label: at runtime the connection falls back to the platform/
    global key, so the stored source should be cleared, not left saying "own".
    A real own key still persists "own"."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    # "own" + blank key -> no key stored, source not persisted as "own".
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "   "}, env_source="own"),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    assoc = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assert assoc.env is None
    assert assoc.env_source is None

    # "own" + a real key -> source legitimately persisted as "own".
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "mine"}, env_source="own"),
        current_user=_user(test_db, 2),
        db=test_db,
    )
    assoc2 = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 2).first()
    assert assoc2.env_source == "own"


def test_reconnect_clearing_own_key_drops_stale_own_source(test_db):
    """A reconnect that clears the key (explicit {}) without restating the source
    must not leave a stale env_source="own": the row would silently run on the
    global key while still claiming the user's own. The invariant is enforced on
    the resulting row state, not just the incoming request."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    u = _user(test_db, 1)
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "mine"}, env_source="own"),
        current_user=u,
        db=test_db,
    )
    assoc = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assert assoc.env_source == "own"

    # Clear the key without restating env_source (env_source left untouched).
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={}),
        current_user=u,
        db=test_db,
    )
    test_db.refresh(assoc)
    assert assoc.env is None
    assert assoc.env_source is None  # stale "own" dropped


def test_provision_surfaces_genuine_config_error_as_400(test_db, monkeypatch):
    """A non-race failure from add_server leaves no shared row behind, so it must
    surface as a 400 carrying the real message — not the race path's opaque 500."""
    from fastapi import HTTPException

    from xagent.web.api import mcp as mcp_api
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    def boom(self, config):
        raise ValueError("bad config")

    monkeypatch.setattr(mcp_api.DatabaseMCPServerManager, "add_server", boom)

    with pytest.raises(HTTPException) as exc:
        connect_mcp_app(
            "google-maps",
            MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": "alice-key"}),
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 400
    assert "bad config" in str(exc.value.detail)


def test_rename_to_catalog_id_is_rejected(test_db):
    """The catalog-namespace reservation must also apply on rename, not only on
    create — otherwise a user could create an arbitrary server and rename it to a
    reserved catalog id (e.g. "google-maps"), squatting the namespace."""
    from fastapi import HTTPException

    from xagent.web.api.mcp import (
        MCPServerCreate,
        MCPServerUpdate,
        create_mcp_server,
        update_mcp_server,
    )

    created = create_mcp_server(
        MCPServerCreate(
            name="my-custom",
            transport="stdio",
            config={"command": "npx", "args": ["--stdio"]},
        ),
        current_user=_user(test_db, 1),
        db=test_db,
    )

    with pytest.raises(HTTPException) as exc:
        update_mcp_server(
            created.id,
            MCPServerUpdate(name="Google-Maps"),  # case variant of the reserved id
            current_user=_user(test_db, 1),
            db=test_db,
        )
    assert exc.value.status_code == 400


def test_connect_coerces_scalar_env_values(test_db):
    """Numeric scalar env values are coerced to trimmed strings rather than
    silently dropped; bool is not coerced (storing "True" as a key is worse than
    dropping it, which falls back to the global key)."""
    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app

    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": 12345}),
        current_user=_user(test_db, 1),
        db=test_db,
    )
    assoc = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 1).first()
    assert decrypt_env_dict(assoc.env) == {"GOOGLE_MAPS_API_KEY": "12345"}

    # A bool is dropped (not stored as "True"), so it falls back to the global key.
    connect_mcp_app(
        "google-maps",
        MCPAppConnectRequest(env={"GOOGLE_MAPS_API_KEY": True}),
        current_user=_user(test_db, 2),
        db=test_db,
    )
    assoc2 = test_db.query(UserMCPServer).filter(UserMCPServer.user_id == 2).first()
    assert assoc2.env is None
