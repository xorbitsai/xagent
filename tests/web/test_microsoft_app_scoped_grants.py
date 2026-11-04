"""A bare Microsoft grant must not satisfy OneDrive, Outlook, or Teams.

The app_id-less Microsoft connect flow requests only the provider's
default_scopes (``User.Read``), never an app's own oauth_scopes, and stores the
grant under ``UserOAuth.provider == "microsoft"``. Each of these apps needs
scopes beyond that (``Files.ReadWrite``, ``Mail.*``, ``Chat.ReadWrite``, ...),
so treating the bare grant as sufficient reports the connector as connected
while every Graph call fails for insufficient scope.

The three surfaces named in #2503 are pinned here: the bare callback's batch
connect, the connected-state lookup, and runtime token resolution (both the
legacy ``UserOAuth`` read and the resolver-hook candidates). The server-list
enrichment and the disconnect paths consult the same policy but are not pinned
here. Every surface gets an app-scoped positive control so a change that simply
stopped resolving anything cannot pass vacuously.
"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.shared.oauth_browser_fixture import (
    create_access_token,
    generic_oauth_callback,
)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.api.mcp import _connected_oauth_server_for_app, _oauth_keys_for_app
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools.config import WebToolConfig, _oauth_token_provider_candidates

APP_IDS = ("onedrive", "outlook", "teams")
SERVER_ID = 7


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data


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


def _microsoft_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="microsoft",
        client_id=encrypt_value("microsoft-client-id"),
        client_secret=encrypt_value("microsoft-client-secret"),
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        redirect_uri="https://app.example.com/api/auth/microsoft/callback",
        userinfo_url="https://graph.microsoft.com/v1.0/me",
        user_id_path="id",
        email_path="userPrincipalName",
        default_scopes=["User.Read"],
    )


def _app(app_id: str) -> dict[str, str]:
    return {"id": app_id, "name": app_id, "provider": "microsoft"}


def _oauth_server(app_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=SERVER_ID,
        name=app_id,
        transport="oauth",
        auth={"app_id": app_id, "provider": "microsoft"},
    )


def _builtin_app_row(app_id: str) -> dict:
    row = get_builtin_public_mcp_app(app_id)
    assert row is not None
    return row


@pytest.mark.parametrize("app_id", APP_IDS)
def test_oauth_keys_exclude_bare_microsoft_provider(app_id):
    assert _oauth_keys_for_app(_app(app_id)) == [app_id]


@pytest.mark.parametrize("app_id", APP_IDS)
def test_resolver_hook_candidates_exclude_bare_microsoft_provider(app_id):
    assert _oauth_token_provider_candidates(_app(app_id)) == [app_id]


@pytest.mark.parametrize("app_id", APP_IDS)
def test_connected_state_ignores_bare_microsoft_grant(app_id):
    bare_account = SimpleNamespace(email="bare@example.com")

    assert _connected_oauth_server_for_app(
        _app(app_id), {app_id: [_oauth_server(app_id)]}, {"microsoft": bare_account}
    ) == (None, None)


@pytest.mark.parametrize("app_id", APP_IDS)
def test_connected_state_uses_app_scoped_grant(app_id):
    scoped_account = SimpleNamespace(email="scoped@example.com")

    assert _connected_oauth_server_for_app(
        _app(app_id), {app_id: [_oauth_server(app_id)]}, {app_id: scoped_account}
    ) == (SERVER_ID, "scoped@example.com")


def _resolve_legacy_token(db, user_id: int, app_id: str):
    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: db, user_id=user_id)
    return asyncio.run(
        cfg._resolve_legacy_oauth_access_token(provider_name="microsoft", app_id=app_id)
    )


@pytest.mark.parametrize("app_id", APP_IDS)
def test_legacy_token_resolution_ignores_bare_microsoft_grant(db_session, app_id):
    db, user = db_session
    db.add(UserOAuth(user_id=user.id, provider="microsoft", access_token="bare-token"))
    db.commit()

    assert _resolve_legacy_token(db, user.id, app_id).access_token is None


@pytest.mark.parametrize("app_id", APP_IDS)
def test_legacy_token_resolution_uses_app_scoped_grant(db_session, app_id):
    db, user = db_session
    # The bare grant is the newer row (higher id): before these apps joined the
    # policy, it would have won the most-recent-first lookup over the working
    # app-scoped grant.
    db.add(UserOAuth(user_id=user.id, provider=app_id, access_token="scoped-token"))
    db.add(UserOAuth(user_id=user.id, provider="microsoft", access_token="bare-token"))
    db.commit()

    assert _resolve_legacy_token(db, user.id, app_id).access_token == "scoped-token"


def _run_microsoft_callback(db, user, monkeypatch, *, app_id=None, access_token):
    state_data = {"type": "oauth_state", "user_id": user.id, "provider": "microsoft"}
    if app_id is not None:
        state_data["app_id"] = app_id
    state = create_access_token(data=state_data, expires_delta=timedelta(minutes=10))
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "User.Read",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"id": "ms-user-1", "userPrincipalName": "alice@example.com"}
            )
        ),
    )
    return generic_oauth_callback("microsoft", request, db, _microsoft_provider())


def test_bare_microsoft_login_skips_app_scoped_apps(db_session, monkeypatch):
    """The bare callback must not activate a UserMCPServer for an app it never
    requested scopes for: that row would be backed by a grant the runtime
    refuses to use for the app (see auth.py's bare batch-connect comment).

    A custom, unlisted Microsoft app is registered alongside as a positive
    control. Without it, a callback that stopped creating any UserMCPServer at
    all would make the absence assertions below pass vacuously.
    """
    db, user = db_session
    for app_id in APP_IDS:
        db.add(PublicMCPApp(**_builtin_app_row(app_id)))
    db.add(
        PublicMCPApp(
            app_id="custom-microsoft-app",
            name="Custom Microsoft App",
            description="Operator-defined Microsoft connector",
            transport="oauth",
            provider_name="microsoft",
            category="Productivity",
            oauth_scopes=[],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "custom_microsoft_app"],
                "env_mapping": {"AUTH_TOKEN": "access_token"},
            },
        )
    )
    db.commit()

    response = _run_microsoft_callback(db, user, monkeypatch, access_token="bare-token")
    assert response.status_code == 200

    # The bare grant is still created -- only the app-scoped servers aren't.
    bare_grant = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "microsoft")
        .one()
    )
    assert bare_grant.access_token == "bare-token"

    server_names = {server.name for server in db.query(MCPServer).all()}
    assert "Custom Microsoft App" in server_names
    for app_id in APP_IDS:
        assert _builtin_app_row(app_id)["name"] not in server_names


@pytest.mark.parametrize("app_id", APP_IDS)
def test_catalog_microsoft_login_stores_app_scoped_grant(
    db_session, monkeypatch, app_id
):
    """Connecting from the catalog entry (the reconnect path for a user whose
    bare grant is no longer accepted) stores the grant under the app id, and
    the runtime resolves it."""
    db, user = db_session
    app_row = _builtin_app_row(app_id)
    db.add(PublicMCPApp(**app_row))
    db.commit()

    response = _run_microsoft_callback(
        db, user, monkeypatch, app_id=app_id, access_token="scoped-token"
    )
    assert response.status_code == 200

    providers = {
        grant.provider
        for grant in db.query(UserOAuth).filter(UserOAuth.user_id == user.id)
    }
    assert providers == {app_id}
    assert app_row["name"] in {server.name for server in db.query(MCPServer).all()}
    assert _resolve_legacy_token(db, user.id, app_id).access_token == "scoped-token"
