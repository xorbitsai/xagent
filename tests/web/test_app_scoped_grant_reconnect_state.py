"""A connector whose only grant the app-scoped policy rejects must say "reconnect".

For an app in ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT``, a bare provider-level
grant (``UserOAuth.provider == "microsoft"``) is not accepted. A user who
connected such an app through the bare batch login, before the app joined the
set, keeps an active ``UserMCPServer`` row backed only by that grant. This file
pins what each surface reports for that stale row (#2887):

* the catalog listing reports the app as not connected;
* ``GET /api/mcp/servers`` reports ``connection_status == "needs_reconnect"``
  rather than ``None``, which documents "no persisted grant exists";
* the runtime builds the server as unavailable with a reconnect hint.

Every case has a control: no grant at all, a working app-scoped grant, an app
outside the policy, and an operator-owned ``word`` row (which the policy only
covers when it carries the builtin provenance), so a change that reported
"reconnect" for every OAuth row cannot pass.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.mcp import get_mcp_server, get_mcp_servers, list_mcp_apps
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.mcp_apps import oauth_grant_keys_rejected_by_app_scope
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools.config import (
    UNAVAILABLE_MCP_APP_SCOPED_GRANT_MESSAGE,
    UNAVAILABLE_MCP_CREDENTIAL_MESSAGE,
    WebToolConfig,
    set_oauth_token_resolver_hook,
)

CUSTOM_LAUNCH_CONFIG = {
    "command": "python",
    "args": ["-m", "custom_microsoft_app"],
    "env_mapping": {"AUTH_TOKEN": "access_token"},
}


@pytest.fixture(autouse=True)
def no_oauth_token_resolver_hook():
    set_oauth_token_resolver_hook(None)
    yield
    set_oauth_token_resolver_hook(None)


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_local()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _builtin_app(db, app_id: str) -> str:
    row = get_builtin_public_mcp_app(app_id)
    assert row is not None
    db.add(PublicMCPApp(**row))
    return app_id


def _custom_microsoft_app(db, app_id: str) -> str:
    """An operator-created app without the builtin provenance stamp."""
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=f"Custom {app_id}",
            description="Operator-defined Microsoft connector",
            transport="oauth",
            provider_name="microsoft",
            category="Productivity",
            oauth_scopes=[],
            is_visible_in_connector=True,
            launch_config=dict(CUSTOM_LAUNCH_CONFIG),
        )
    )
    return app_id


def _connect(db, user: User, app_id: str, *grant_providers: str) -> MCPServer:
    """An active server row for ``app_id`` plus one grant per provider key,
    created in order (the last one is the newest row)."""
    server = MCPServer(
        name=app_id,
        description=f"{app_id} server",
        managed="external",
        transport="oauth",
        auth={"app_id": app_id, "provider": "microsoft"},
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=user.id, mcpserver_id=server.id, is_owner=True, is_active=True
        )
    )
    for provider in grant_providers:
        db.add(
            UserOAuth(
                user_id=user.id,
                provider=provider,
                provider_user_id=f"{provider}-user",
                access_token=f"{provider}-token",
                email=f"{provider}@example.com",
            )
        )
    db.commit()
    db.refresh(server)
    return server


def _catalog_entry(db, user: User, app_id: str) -> dict:
    [entry] = [
        app
        for app in list_mcp_apps(location="remote", current_user=user, db=db)
        if app["id"] == app_id
    ]
    return entry


def _server_listing(db, user: User, server: MCPServer):
    [response] = [
        r for r in get_mcp_servers(current_user=user, db=db) if r.id == server.id
    ]
    return response


async def _runtime_config(db, user: User) -> dict:
    cfg = WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user.id,
        workspace_config={"base_dir": "/tmp", "task_id": "task-1"},
    )
    [config] = await cfg.get_mcp_server_configs()
    return config


def _assert_credential_unavailable(config: dict, *, message: str) -> None:
    assert config["transport"] == "unavailable"
    assert config["config"]["reason"] == "oauth_token_required"
    assert config["config"]["failure_code"] == "oauth_token_required"
    assert config["config"]["message"] == message


# -- the pure policy helper ---------------------------------------------------


def test_rejected_keys_name_the_bare_provider_for_an_app_in_the_policy():
    assert oauth_grant_keys_rejected_by_app_scope(
        "onedrive", ["microsoft", "onedrive"]
    ) == ["microsoft"]


def test_rejected_keys_are_empty_for_an_app_outside_the_policy():
    assert (
        oauth_grant_keys_rejected_by_app_scope(
            "custom-microsoft-app", ["microsoft", "custom-microsoft-app"]
        )
        == []
    )


def test_rejected_keys_are_empty_for_an_operator_owned_word_row():
    app = {"id": "word", "launch_config": dict(CUSTOM_LAUNCH_CONFIG)}
    assert oauth_grant_keys_rejected_by_app_scope(app, ["microsoft", "word"]) == []


# -- a stale row: bare grant plus an active server row ------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("app_id", ["onedrive", "word"])
async def test_stale_row_reports_reconnect_on_every_surface(db_session, app_id):
    db, user = db_session
    _builtin_app(db, app_id)
    server = _connect(db, user, app_id, "microsoft")

    assert _catalog_entry(db, user, app_id)["is_connected"] is False

    response = _server_listing(db, user, server)
    assert response.connection_status == "needs_reconnect"
    assert response.connected_account is None

    _assert_credential_unavailable(
        await _runtime_config(db, user),
        message=UNAVAILABLE_MCP_APP_SCOPED_GRANT_MESSAGE,
    )


def test_stale_row_reports_reconnect_on_the_single_server_endpoint(db_session):
    """``GET /api/mcp/servers/{id}`` shares the listing's enrichment, and a
    bare grant whose token was also cleared still points at reconnecting."""
    db, user = db_session
    _builtin_app(db, "onedrive")
    server = _connect(db, user, "onedrive", "microsoft")
    db.query(UserOAuth).update({"access_token": "", "refresh_token": None})
    db.commit()

    response = get_mcp_server(server_id=server.id, current_user=user, db=db)

    assert response.connection_status == "needs_reconnect"
    assert response.connected_account is None


# -- controls ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_without_any_grant_keeps_the_generic_state(db_session):
    db, user = db_session
    _builtin_app(db, "onedrive")
    server = _connect(db, user, "onedrive")

    assert _catalog_entry(db, user, "onedrive")["is_connected"] is False

    response = _server_listing(db, user, server)
    assert response.connection_status is None
    assert response.connected_account is None

    _assert_credential_unavailable(
        await _runtime_config(db, user), message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE
    )


@pytest.mark.asyncio
async def test_app_scoped_grant_is_connected_even_beside_a_newer_bare_grant(
    db_session,
):
    db, user = db_session
    _builtin_app(db, "onedrive")
    server = _connect(db, user, "onedrive", "onedrive", "microsoft")

    entry = _catalog_entry(db, user, "onedrive")
    assert entry["is_connected"] is True
    assert entry["connected_account"] == "onedrive@example.com"

    response = _server_listing(db, user, server)
    assert response.connection_status == "connected"
    assert response.connected_account == "onedrive@example.com"

    assert (await _runtime_config(db, user))["transport"] == "stdio"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "app_id",
    [
        # Outside the policy entirely.
        "custom-microsoft-app",
        # "word" is in the policy only with the builtin provenance; an
        # operator-owned row keeps accepting the bare grant on every surface.
        "word",
    ],
)
async def test_bare_grant_stays_connected_where_the_policy_does_not_apply(
    db_session, app_id
):
    db, user = db_session
    _custom_microsoft_app(db, app_id)
    server = _connect(db, user, app_id, "microsoft")

    entry = _catalog_entry(db, user, app_id)
    assert entry["is_connected"] is True
    assert entry["connected_account"] == "microsoft@example.com"

    response = _server_listing(db, user, server)
    assert response.connection_status == "connected"
    assert response.connected_account == "microsoft@example.com"

    assert (await _runtime_config(db, user))["transport"] == "stdio"


@pytest.mark.asyncio
async def test_another_users_bare_grant_is_not_the_requesting_users_connection(
    db_session,
):
    """Bob's bare grant must not turn alice's grant-less row into "reconnect".
    Guards the user scoping of the rejected-grant lookup on both surfaces."""
    db, alice = db_session
    bob = User(username="bob", password_hash="x", is_admin=False)
    db.add(bob)
    db.commit()
    db.refresh(bob)

    _builtin_app(db, "onedrive")
    server = _connect(db, alice, "onedrive")
    db.add(
        UserOAuth(
            user_id=bob.id,
            provider="microsoft",
            provider_user_id="microsoft-bob",
            access_token="bob-token",
            email="bob@example.com",
        )
    )
    db.commit()

    assert _catalog_entry(db, alice, "onedrive")["is_connected"] is False

    response = _server_listing(db, alice, server)
    assert response.connection_status is None
    assert response.connected_account is None

    _assert_credential_unavailable(
        await _runtime_config(db, alice), message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "grant_order", [("onedrive", "microsoft"), ("microsoft", "onedrive")]
)
async def test_cleared_app_scoped_grant_beside_a_bare_grant_keeps_the_generic_state(
    db_session, grant_order
):
    """The app-scoped grant is the accepted one even when its token was cleared,
    so the user does have a grant for this app. The provider-level reconnect
    hint only applies when no accepted grant exists."""
    db, user = db_session
    _builtin_app(db, "onedrive")
    server = _connect(db, user, "onedrive", *grant_order)
    db.query(UserOAuth).filter(UserOAuth.provider == "onedrive").update(
        {"access_token": "", "refresh_token": None}
    )
    db.commit()

    assert _catalog_entry(db, user, "onedrive")["is_connected"] is False

    response = _server_listing(db, user, server)
    assert response.connection_status == "needs_reconnect"
    assert response.connected_account is None

    _assert_credential_unavailable(
        await _runtime_config(db, user), message=UNAVAILABLE_MCP_CREDENTIAL_MESSAGE
    )
