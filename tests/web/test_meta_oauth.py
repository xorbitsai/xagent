from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import create_access_token, generic_oauth_callback
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools import config as tool_config


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data


class NonJsonResponse(MockResponse):
    def json(self):
        raise ValueError("response body is not JSON")


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.add(
        PublicMCPApp(
            app_id="facebook",
            name="Facebook Pages",
            description="Facebook connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["pages_show_list", "pages_manage_posts"],
            is_visible_in_connector=True,
            launch_config={
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.facebook"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _meta_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="meta",
        client_id=encrypt_value("meta-client-id"),
        client_secret=encrypt_value("meta-client-secret"),
        token_url="https://graph.facebook.com/v25.0/oauth/access_token",
        redirect_uri="https://app.example.com/api/auth/meta/callback",
        userinfo_url="https://graph.facebook.com/v25.0/me?fields=id,email",
        user_id_path="id",
        email_path="email",
        default_scopes=["public_profile"],
    )


def _google_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="google",
        client_id=encrypt_value("google-client-id"),
        client_secret=encrypt_value("google-client-secret"),
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri="https://app.example.com/api/auth/google/callback",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        user_id_path="sub",
        email_path="email",
        default_scopes=["https://www.googleapis.com/auth/gmail.modify"],
    )


def test_gmail_callback_best_effort_registers_watch_after_oauth_commit(
    db_session, monkeypatch
):
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "gmail",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "gmail-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "gmail-token",
                    "refresh_token": "gmail-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "https://www.googleapis.com/auth/gmail.modify",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-user-1", "email": "alice@gmail.com"}
            )
        ),
    )
    calls: list[int] = []

    def fake_best_effort_provision(_db, *, user_id: int, context: str):
        calls.append(user_id)

    monkeypatch.setattr(
        "xagent.web.services.gmail_provisioning."
        "best_effort_provision_gmail_watches_for_user",
        fake_best_effort_provision,
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "gmail")
        .one()
    )
    assert oauth_account.email == "alice@gmail.com"
    assert calls == [int(user.id)]


def test_meta_callback_exchanges_short_lived_token_and_connects_selected_app(
    db_session, monkeypatch
):
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "meta",
            "app_id": "facebook",
            "redirect": "https://app.example.com/tools",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "short-code", "state": state})

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "short-token",
                "token_type": "bearer",
                "expires_in": 3600,
            }
        )
    )

    def get(url, **kwargs):
        if url.endswith("/oauth/access_token"):
            assert kwargs["params"] == {
                "grant_type": "fb_exchange_token",
                "client_id": "meta-client-id",
                "client_secret": "meta-client-secret",
                "fb_exchange_token": "short-token",
            }
            return MockResponse(
                {
                    "access_token": "long-token",
                    "token_type": "bearer",
                    "expires_in": 5184000,
                }
            )

        assert url == "https://graph.facebook.com/v25.0/me?fields=id,email"
        assert kwargs["headers"] == {"Authorization": "Bearer long-token"}
        return MockResponse({"id": "meta-user-1", "email": "alice@example.com"})

    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", Mock(side_effect=get))

    response = generic_oauth_callback("meta", request, db, _meta_provider())

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "facebook")
        .one()
    )
    assert oauth_account.access_token == "long-token"
    assert oauth_account.provider_user_id == "meta-user-1"
    assert oauth_account.email == "alice@example.com"
    assert oauth_account.expires_at is not None

    server = db.query(MCPServer).filter(MCPServer.name == "Facebook Pages").one()
    assert server.transport == "oauth"
    assert server.auth == {"app_id": "facebook", "provider": "meta"}
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    assert user_mcp.is_active is True


def test_meta_callback_uses_short_lived_token_when_long_lived_exchange_is_not_json(
    db_session, monkeypatch, caplog
):
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "meta",
            "app_id": "facebook",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "short-code", "state": state})

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "short-token",
                    "token_type": "bearer",
                    "expires_in": 3600,
                }
            )
        ),
    )

    def get(url, **kwargs):
        if url.endswith("/oauth/access_token"):
            return NonJsonResponse(status_code=502, text="<html>bad gateway</html>")

        assert url == "https://graph.facebook.com/v25.0/me?fields=id,email"
        assert kwargs["headers"] == {"Authorization": "Bearer short-token"}
        return MockResponse({"id": "meta-user-1", "email": "alice@example.com"})

    monkeypatch.setattr(auth_api.requests, "get", Mock(side_effect=get))
    caplog.set_level(logging.WARNING, logger=auth_api.__name__)

    response = generic_oauth_callback("meta", request, db, _meta_provider())

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "facebook")
        .one()
    )
    assert oauth_account.access_token == "short-token"
    assert oauth_account.provider_user_id == "meta-user-1"
    assert "Meta long-lived token exchange failed" in caplog.text
    assert "response body is not JSON" in caplog.text


def test_meta_callback_uses_short_lived_token_when_long_lived_exchange_fails(
    db_session, monkeypatch, caplog
):
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "meta",
            "app_id": "facebook",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "short-code", "state": state})

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "short-token",
                    "token_type": "bearer",
                    "expires_in": 3600,
                }
            )
        ),
    )

    def get(url, **kwargs):
        if url.endswith("/oauth/access_token"):
            raise auth_api.requests.RequestException("meta token exchange timed out")

        assert url == "https://graph.facebook.com/v25.0/me?fields=id,email"
        assert kwargs["headers"] == {"Authorization": "Bearer short-token"}
        return MockResponse({"id": "meta-user-1", "email": "alice@example.com"})

    monkeypatch.setattr(auth_api.requests, "get", Mock(side_effect=get))
    caplog.set_level(logging.WARNING, logger=auth_api.__name__)

    response = generic_oauth_callback("meta", request, db, _meta_provider())

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "facebook")
        .one()
    )
    assert oauth_account.access_token == "short-token"
    assert oauth_account.provider_user_id == "meta-user-1"
    assert "Meta long-lived token exchange failed" in caplog.text
    assert "meta token exchange timed out" in caplog.text


def test_meta_long_lived_token_exchange_logs_rejected_response(monkeypatch, caplog):
    token_data = {
        "access_token": "short-token",
        "token_type": "bearer",
        "expires_in": 3600,
    }
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"error": {"message": "invalid short token"}},
                status_code=400,
            )
        ),
    )
    caplog.set_level(logging.WARNING, logger=auth_api.__name__)

    result = auth_api._exchange_meta_long_lived_token(
        "meta",
        "https://graph.facebook.com/v25.0/oauth/access_token",
        token_data,
        "meta-client-id",
        "meta-client-secret",
    )

    assert result == token_data
    assert "Meta long-lived token exchange returned unusable response" in caplog.text
    assert "status=400" in caplog.text
    assert "invalid short token" in caplog.text


@pytest.mark.asyncio
async def test_meta_expired_token_refresh_uses_fb_exchange_token(
    db_session, monkeypatch
):
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="meta",
            name="Meta",
            client_id=encrypt_value("meta-client-id"),
            client_secret=encrypt_value("meta-client-secret"),
            auth_url="https://www.facebook.com/v25.0/dialog/oauth",
            token_url="https://graph.facebook.com/v25.0/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/meta/callback",
            userinfo_url="https://graph.facebook.com/v25.0/me?fields=id,email",
            user_id_path="id",
            email_path="email",
            default_scopes=["public_profile"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="facebook",
        access_token="old-long-token",
        refresh_token=None,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="meta-user-1",
    )
    db.add(oauth_account)
    db.commit()

    captured_requests = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            captured_requests.append((url, kwargs))
            return MockResponse(
                {
                    "access_token": "new-long-token",
                    "token_type": "bearer",
                    "expires_in": 5184000,
                }
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "meta")
        is True
    )

    assert oauth_account.access_token == "new-long-token"
    assert oauth_account.expires_at is not None
    assert captured_requests == [
        (
            "https://graph.facebook.com/v25.0/oauth/access_token",
            {
                "params": {
                    "grant_type": "fb_exchange_token",
                    "client_id": "meta-client-id",
                    "client_secret": "meta-client-secret",
                    "fb_exchange_token": "old-long-token",
                },
                "timeout": 10.0,
            },
        )
    ]


def test_generic_oauth_batch_skips_non_oauth_app_and_connects_oauth_app(
    db_session, monkeypatch
):
    """Provider-only OAuth callback (no app_id) connects every catalog app under
    the provider. A mis-tagged non-oauth app must be skipped without aborting the
    batch, while the legitimate builtin_oauth app still connects (L1 + the
    narrowed AppNotOAuthError catch)."""
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="gmail",
            name="Gmail",
            transport="oauth",
            provider_name="google",
        )
    )
    db.add(
        PublicMCPApp(
            app_id="gmaps",
            name="GMaps",
            transport="stdio",
            provider_name="google",
            launch_config={"command": "npx", "required_env": ["KEY"]},
        )
    )
    db.commit()

    state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": "google"},
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "tok",
                    "token_type": "Bearer",
                    "scope": "",
                    "expires_in": 3600,
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": "u1", "email": "alice@gmail.com"})),
    )

    response = generic_oauth_callback("google", request, db, _google_provider())
    assert response.status_code == 200

    server_names = {s.name for s in db.query(MCPServer).all()}
    assert "Gmail" in server_names  # legitimate oauth app connected
    assert "GMaps" not in server_names  # mis-tagged key-based app skipped


def test_bare_meta_login_skips_facebook_but_still_connects_instagram(
    db_session, monkeypatch
):
    """Provider-only ("bare") Meta login (no app_id) only ever requests
    db_provider.default_scopes, never an app's own oauth_scopes — it can't
    carry pages_read_user_content. Creating a Facebook UserMCPServer row from
    this flow would be an orphan the agent runtime picks up directly (bypasses
    the connected-state check) and can never resolve a token for
    (APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT). Instagram's required scopes
    haven't changed, so it must still connect via this same bare flow."""
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="instagram",
            name="Instagram",
            description="Instagram connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["instagram_basic", "instagram_content_publish"],
            is_visible_in_connector=True,
            launch_config={
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.instagram"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()

    state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": "meta"},
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})

    post = Mock(
        return_value=MockResponse(
            {"access_token": "short-token", "token_type": "bearer", "expires_in": 3600}
        )
    )

    def get(url, **kwargs):
        if url.endswith("/oauth/access_token"):
            return MockResponse(
                {
                    "access_token": "long-token",
                    "token_type": "bearer",
                    "expires_in": 5184000,
                }
            )
        return MockResponse({"id": "meta-user-1", "email": "alice@example.com"})

    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", Mock(side_effect=get))

    response = generic_oauth_callback("meta", request, db, _meta_provider())
    assert response.status_code == 200

    # The bare grant is still created — only the Facebook MCP server isn't.
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "meta")
        .one()
    )
    assert oauth_account.access_token == "long-token"

    server_names = {s.name for s in db.query(MCPServer).all()}
    assert "Instagram" in server_names
    assert "Facebook Pages" not in server_names


async def test_disconnecting_facebook_preserves_shared_bare_meta_grant_for_instagram(
    db_session, monkeypatch
):
    """UserOAuth has no app_id column: a bare Meta grant (provider="meta")
    and an app-scoped Facebook grant (provider="facebook") are just two rows.
    _oauth_keys_for_app still lets Instagram rely on the shared "meta" row, so
    deleting the Facebook MCP server must not delete it out from under
    Instagram — the delete path has to stay symmetric with that app-scoped
    read-path policy."""
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="instagram",
            name="Instagram",
            description="Instagram connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["instagram_basic", "instagram_content_publish"],
            is_visible_in_connector=True,
            launch_config={
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.instagram"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()

    post = Mock(
        return_value=MockResponse(
            {"access_token": "short-token", "token_type": "bearer", "expires_in": 3600}
        )
    )

    def get(url, **kwargs):
        if url.endswith("/oauth/access_token"):
            return MockResponse(
                {
                    "access_token": "long-token",
                    "token_type": "bearer",
                    "expires_in": 5184000,
                }
            )
        return MockResponse({"id": "meta-user-1", "email": "alice@example.com"})

    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", Mock(side_effect=get))

    # 1. Bare login connects Instagram via the shared provider="meta" grant.
    bare_state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": "meta"},
        expires_delta=timedelta(minutes=10),
    )
    generic_oauth_callback(
        "meta",
        SimpleNamespace(query_params={"code": "bare-code", "state": bare_state}),
        db,
        _meta_provider(),
    )

    # 2. A separate app-specific login connects Facebook with its own grant.
    fb_state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "meta",
            "app_id": "facebook",
        },
        expires_delta=timedelta(minutes=10),
    )
    generic_oauth_callback(
        "meta",
        SimpleNamespace(query_params={"code": "fb-code", "state": fb_state}),
        db,
        _meta_provider(),
    )

    assert db.query(UserOAuth).filter(UserOAuth.provider == "meta").count() == 1
    assert db.query(UserOAuth).filter(UserOAuth.provider == "facebook").count() == 1

    from xagent.web.api.mcp import delete_mcp_server

    facebook_server = (
        db.query(MCPServer).filter(MCPServer.name == "Facebook Pages").one()
    )
    await delete_mcp_server(facebook_server.id, current_user=user, db=db)

    assert db.query(UserOAuth).filter(UserOAuth.provider == "facebook").count() == 0
    # The shared bare grant Instagram still relies on must survive.
    assert db.query(UserOAuth).filter(UserOAuth.provider == "meta").count() == 1


async def test_disconnecting_facebook_only_user_also_removes_orphaned_bare_meta_grant(
    db_session,
):
    """Mirror of the previous test's opposite case: no Instagram connection
    relies on the shared bare "meta" row (e.g. it predates this app-scoped
    policy, or was left over from a bare login that never connected
    anything). Excluding it from providers_to_delete is only correct while
    some sibling app still needs it; with none connected, it must still be
    cleaned up on disconnect instead of becoming a permanent orphan with no
    UI path to remove it."""
    db, user = db_session
    db.add(UserOAuth(user_id=user.id, provider="meta", access_token="bare-meta-token"))
    db.add(
        UserOAuth(user_id=user.id, provider="facebook", access_token="app-scoped-token")
    )
    server = MCPServer(name="Facebook Pages", transport="oauth", managed="external")
    db.add(server)
    db.commit()
    db.add(UserMCPServer(user_id=user.id, mcpserver_id=server.id, is_owner=True))
    db.commit()

    from xagent.web.api.mcp import delete_mcp_server

    await delete_mcp_server(server.id, current_user=user, db=db)

    assert db.query(UserOAuth).filter(UserOAuth.provider == "facebook").count() == 0
    assert db.query(UserOAuth).filter(UserOAuth.provider == "meta").count() == 0


def test_facebook_server_list_does_not_show_bare_meta_email_as_connected(db_session):
    """GET /servers must agree with the app-scoped-grant policy: showing a
    bare "meta" account's email as Facebook's connected_account would tell
    the user Facebook is connected when _oauth_keys_for_app (and the runtime
    token resolver) say it isn't."""
    db, user = db_session
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="meta",
            access_token="bare-meta-token",
            email="alice@example.com",
        )
    )
    server = MCPServer(name="Facebook Pages", transport="oauth", managed="external")
    db.add(server)
    db.commit()
    db.add(UserMCPServer(user_id=user.id, mcpserver_id=server.id, is_owner=True))
    db.commit()

    from xagent.web.api.mcp import get_mcp_servers

    responses = get_mcp_servers(current_user=user, db=db)

    assert len(responses) == 1
    assert responses[0].connected_account is None


def test_facebook_server_list_does_not_show_blanked_token_as_connected(db_session):
    """The reconnect migration blanks access_token but not email; a stale
    email must not read as "still connected" once the token is gone."""
    db, user = db_session
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="facebook",
            access_token="",
            email="alice@example.com",
        )
    )
    server = MCPServer(name="Facebook Pages", transport="oauth", managed="external")
    db.add(server)
    db.commit()
    db.add(UserMCPServer(user_id=user.id, mcpserver_id=server.id, is_owner=True))
    db.commit()

    from xagent.web.api.mcp import get_mcp_servers

    responses = get_mcp_servers(current_user=user, db=db)

    assert len(responses) == 1
    assert responses[0].connected_account is None


def test_instagram_server_list_still_shows_bare_meta_email_as_connected(db_session):
    """Sanity counterpart: Instagram's required scopes haven't changed, so its
    display must keep accepting the shared bare "meta" grant."""
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="instagram",
            name="Instagram",
            description="Instagram connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["instagram_basic", "instagram_content_publish"],
            is_visible_in_connector=True,
            launch_config={},
        )
    )
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="meta",
            access_token="bare-meta-token",
            email="alice@example.com",
        )
    )
    server = MCPServer(name="Instagram", transport="oauth", managed="external")
    db.add(server)
    db.commit()
    db.add(UserMCPServer(user_id=user.id, mcpserver_id=server.id, is_owner=True))
    db.commit()

    from xagent.web.api.mcp import get_mcp_servers

    responses = get_mcp_servers(current_user=user, db=db)

    assert len(responses) == 1
    assert responses[0].connected_account == "alice@example.com"


def test_generic_oauth_single_app_rejects_non_oauth_app_cleanly(
    db_session, monkeypatch
):
    """Single-app OAuth callback (app_id in state) pointing at a non-oauth app
    must fail with a clear error page instead of a generic 500, and must not
    create an MCP server. Symmetric with the batch branch's AppNotOAuthError
    handling (New Finding C)."""
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="gmaps",
            name="GMaps",
            transport="stdio",
            provider_name="google",
            launch_config={"command": "npx", "required_env": ["KEY"]},
        )
    )
    db.commit()

    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "gmaps",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "tok",
                    "token_type": "Bearer",
                    "scope": "",
                    "expires_in": 3600,
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": "u1", "email": "alice@gmail.com"})),
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 400
    assert "GMaps" not in {s.name for s in db.query(MCPServer).all()}
