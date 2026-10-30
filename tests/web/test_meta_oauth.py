from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from tests.shared.oauth_browser_fixture import (
    create_access_token,
    generic_oauth_callback,
)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.google_oauth_grant import (
    GoogleOAuthAppAssociation,
    GoogleOAuthGrant,
)
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import gmail_provisioning
from xagent.web.services import google_oauth_grants as google_oauth_grants_service
from xagent.web.services.google_oauth_grants import (
    create_google_oauth_grant_association,
)
from xagent.web.tools import config as tool_config
from xagent.web.tools.config import WebToolConfig


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
def db_session(tmp_path, monkeypatch):
    # Gmail callback tests exercise the enabled review environment.
    monkeypatch.setenv("XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED", "true")
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
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri="https://app.example.com/api/auth/google/callback",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        user_id_path="sub",
        email_path="email",
        default_scopes=["https://www.googleapis.com/auth/gmail.modify"],
    )


def _shared_google_state(
    user: User, app_id: str, scopes: list[str], *, forced_consent: bool = False
) -> str:
    data = {
        "type": "oauth_state",
        "user_id": user.id,
        "provider": "google",
        "app_id": app_id,
        "google_shared_mode": True,
        "google_shared_scopes": scopes,
    }
    if forced_consent:
        data["google_forced_consent"] = True
    return create_access_token(
        data=data,
        expires_delta=timedelta(minutes=10),
    )


@pytest.mark.asyncio
async def test_google_shared_callback_dual_writes_for_instant_rollback(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-drive",
            name="Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.drive"],
            is_visible_in_connector=True,
            launch_config={"command": "python", "args": ["-m", "drive"]},
        )
    )
    db.commit()
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "google-drive",
            "google_shared_mode": True,
            "google_shared_scopes": ["openid", "scope.drive"],
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
                    "access_token": "shared-access",
                    "refresh_token": "shared-refresh",
                    "token_type": "Bearer",
                    "scope": "openid scope.drive",
                    "expires_in": 3600,
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    shared = db.query(UserOAuth).filter_by(provider="google").one()
    legacy = db.query(UserOAuth).filter_by(provider="google-drive").one()
    association = db.query(GoogleOAuthAppAssociation).one()
    grant = db.query(GoogleOAuthGrant).one()
    assert shared.access_token == legacy.access_token == "shared-access"
    assert shared.refresh_token == legacy.refresh_token == "shared-refresh"
    assert grant.oauth_account_id == shared.id
    assert association.legacy_oauth_account_id == legacy.id

    # Exercise the real rollout fallback without first resolving the shared
    # row (which would itself mirror credentials and make this test vacuous).
    shared.access_token = "must-not-be-selected"
    db.commit()
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "false")
    isolated_session_factory = sessionmaker(
        bind=db.get_bind(), autoflush=False, autocommit=False
    )
    config = WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=int(user.id),
        workspace_config={"base_dir": "/tmp", "task_id": "task-1"},
        db_factory=isolated_session_factory,
    )
    resolved = await config._resolve_legacy_oauth_access_token(
        provider_name="google",
        app_id="google-drive",
        app_info={"oauth_scopes": ["scope.drive"]},
    )

    assert resolved.access_token == "shared-access"


def test_shared_sibling_callback_reprovisions_mirrored_gmail(db_session, monkeypatch):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    gmail_app = get_builtin_public_mcp_app("gmail")
    assert gmail_app is not None
    db.add(PublicMCPApp(**gmail_app))
    db.add(
        PublicMCPApp(
            app_id="test-google-drive",
            name="Test Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.drive"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "drive"],
                "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    gmail_scope = "https://www.googleapis.com/auth/gmail.modify"
    shared = UserOAuth(
        user_id=int(user.id),
        provider="google",
        provider_user_id="google-subject",
        access_token="expired-shared",
        refresh_token="expired-refresh",
        scope=f"{gmail_scope} scope.drive",
        email="alice@example.com",
    )
    gmail = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        provider_user_id="google-subject",
        access_token="",
        refresh_token=None,
        scope=gmail_scope,
        email="alice@example.com",
    )
    drive = UserOAuth(
        user_id=int(user.id),
        provider="test-google-drive",
        provider_user_id="google-subject",
        access_token="expired-drive",
        refresh_token="expired-refresh",
        scope="scope.drive",
        email="alice@example.com",
    )
    db.add_all([shared, gmail, drive])
    db.flush()
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="google-client-id",
        app_id="gmail",
        legacy_oauth_account=gmail,
    )
    db.commit()
    state = _shared_google_state(
        user,
        "test-google-drive",
        [gmail_scope, "scope.drive"],
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "reconnected-access",
                    "refresh_token": "reconnected-refresh",
                    "token_type": "Bearer",
                    "scope": f"{gmail_scope} scope.drive",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )
    provision = Mock()
    monkeypatch.setattr(
        auth_api, "_best_effort_ensure_gmail_watches_for_user", provision
    )

    response = generic_oauth_callback(
        "google",
        SimpleNamespace(query_params={"code": "code", "state": state}),
        db,
        _google_provider(),
    )

    assert response.status_code == 200
    db.refresh(gmail)
    assert gmail.access_token == "reconnected-access"
    assert gmail.refresh_token == "reconnected-refresh"
    provision.assert_called_once_with(db, user_id=int(user.id))


def test_google_shared_callback_retries_with_consent_for_new_subject_without_refresh(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-sheets",
            name="Google Sheets",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.sheets"],
            is_visible_in_connector=True,
            launch_config={"command": "python", "args": ["-m", "sheets"]},
        )
    )
    shared = UserOAuth(
        user_id=int(user.id),
        provider="google",
        provider_user_id="subject-a",
        access_token="existing-access",
        refresh_token="existing-refresh",
        scope="openid scope.drive",
    )
    legacy = UserOAuth(
        user_id=int(user.id),
        provider="google-sheets",
        provider_user_id="subject-a",
        access_token="existing-access",
        refresh_token="existing-refresh",
        scope="openid scope.sheets",
    )
    db.add_all([shared, legacy])
    db.flush()
    create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="test-client-id",
        app_id="google-sheets",
        legacy_oauth_account=legacy,
    )
    db.commit()
    shared_id = int(shared.id)
    request = SimpleNamespace(
        query_params={
            "code": "code",
            "state": _shared_google_state(
                user, "google-sheets", ["openid", "scope.sheets"]
            ),
        }
    )
    # The callback runs in a fresh request in production. Do not retain the
    # legacy row in this test session while the subject-switch path deletes it.
    db.expunge(legacy)
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "subject-b-access",
                    "token_type": "Bearer",
                    "scope": "openid scope.sheets",
                    "expires_in": 3600,
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": "subject-b", "email": "b@example.com"})),
    )
    lock_order: list[str] = []
    locked_account_ids: list[tuple[int, ...]] = []
    original_account_lock = google_oauth_grants_service._lock_google_oauth_accounts

    def observe_account_lock(lock_db, account_ids):
        normalized_ids = tuple(sorted({int(account_id) for account_id in account_ids}))
        lock_order.append("shared-lock")
        locked_account_ids.append(normalized_ids)
        return original_account_lock(lock_db, normalized_ids)

    def observe_sql(_conn, _cursor, statement, _parameters, _context, _many):
        if "delete from user_oauth" in statement.lower():
            lock_order.append("legacy-delete")

    monkeypatch.setattr(
        google_oauth_grants_service,
        "_lock_google_oauth_accounts",
        observe_account_lock,
    )
    event.listen(db.get_bind(), "before_cursor_execute", observe_sql)

    try:
        response = generic_oauth_callback("google", request, db, _google_provider())
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", observe_sql)

    assert response.status_code == 307
    assert lock_order.index("shared-lock") < lock_order.index("legacy-delete")
    assert (shared_id,) in locked_account_ids
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert params["prompt"] == ["consent"]
    retry_state = auth_api.verify_token(params["state"][0])
    assert retry_state["google_shared_mode"] is True
    assert retry_state["google_forced_consent"] is True
    assert retry_state["app_id"] == "google-sheets"
    assert (
        db.query(UserOAuth).filter(UserOAuth.provider_user_id == "subject-b").count()
        == 0
    )


def test_google_shared_callback_stops_retrying_after_forced_consent(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-sheets",
            name="Google Sheets",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.sheets"],
            is_visible_in_connector=True,
            launch_config={"command": "python", "args": ["-m", "sheets"]},
        )
    )
    db.commit()
    request = SimpleNamespace(
        query_params={
            "code": "code",
            "state": _shared_google_state(
                user,
                "google-sheets",
                ["openid", "scope.sheets"],
                forced_consent=True,
            ),
        }
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "subject-b-access",
                    "token_type": "Bearer",
                    "scope": "openid scope.sheets",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": "subject-b", "email": "b@example.com"})),
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 400
    assert "did not provide offline access" in response.body.decode()
    assert db.query(UserOAuth).count() == 0


def test_actor_google_shared_consent_retry_persists_replacement_flow(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    app = get_builtin_public_mcp_app("google-sheets")
    assert app is not None
    db.add(PublicMCPApp(**app))
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id="google-sheets")
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()

    start = auth_api.start_builtin_oauth_for_resource_owner(
        provider="google",
        app_id="google-sheets",
        user=user,
        resource_owner_key="toby:workspace:actor",
        redirect="https://app.example.com/settings",
        db=db,
        db_provider=_google_provider(),
    )
    db.commit()
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    cookies = SimpleCookie()
    cookies.load(start.headers["set-cookie"])
    actor_cookie = next(
        (name, morsel.value)
        for name, morsel in cookies.items()
        if name.startswith("xagent_actor_oauth_")
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "actor-access",
                    "token_type": "Bearer",
                    "scope": "openid scope.sheets",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "actor-subject", "email": "a@example.com"}
            )
        ),
    )

    response = auth_api.generic_oauth_callback(
        "google",
        SimpleNamespace(
            query_params={"code": "code", "state": state},
            cookies={actor_cookie[0]: actor_cookie[1]},
        ),
        db,
        _google_provider(),
    )

    assert response.status_code == 307
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert params["prompt"] == ["consent"]
    retry_state = auth_api.verify_token(params["state"][0])
    assert retry_state["google_forced_consent"] is True
    assert db.query(ActorOAuthFlowState).count() == 1


def test_actor_google_shared_callback_persists_only_actor_namespace(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    app = get_builtin_public_mcp_app("google-sheets")
    assert app is not None
    db.add(PublicMCPApp(**app))
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id="google-sheets")
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    owner_key = "toby:workspace:actor"
    start = auth_api.start_builtin_oauth_for_resource_owner(
        provider="google",
        app_id="google-sheets",
        user=user,
        resource_owner_key=owner_key,
        redirect="https://app.example.com/settings",
        db=db,
        db_provider=_google_provider(),
    )
    db.commit()
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    state_payload = auth_api.verify_token(state)
    granted_scope = " ".join(state_payload["google_shared_scopes"])
    cookies = SimpleCookie()
    cookies.load(start.headers["set-cookie"])
    actor_cookie = next(
        (name, morsel.value)
        for name, morsel in cookies.items()
        if name.startswith("xagent_actor_oauth_")
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "actor-access",
                    "refresh_token": "actor-refresh",
                    "token_type": "Bearer",
                    "scope": granted_scope,
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "actor-subject", "email": "actor@example.com"}
            )
        ),
    )

    response = auth_api.generic_oauth_callback(
        "google",
        SimpleNamespace(
            query_params={"code": "code", "state": state},
            cookies={actor_cookie[0]: actor_cookie[1]},
        ),
        db,
        _google_provider(),
    )

    assert response.status_code == 200
    accounts = db.query(UserOAuth).order_by(UserOAuth.id).all()
    assert {account.provider for account in accounts} == {
        "google",
        "google-sheets",
    }
    assert {account.resource_owner_key for account in accounts} == {owner_key}
    assert (
        db.query(UserOAuth).filter(UserOAuth.resource_owner_key.is_(None)).count() == 0
    )
    shared = next(account for account in accounts if account.provider == "google")
    legacy = next(
        account for account in accounts if account.provider == "google-sheets"
    )
    grant = db.query(GoogleOAuthGrant).one()
    association = db.query(GoogleOAuthAppAssociation).one()
    assert grant.oauth_account_id == shared.id
    assert grant.resource_owner_key == owner_key
    assert association.legacy_oauth_account_id == legacy.id
    assert db.query(ActorOAuthFlowState).count() == 0


def test_google_shared_callback_reports_missing_scope(db_session, monkeypatch):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-drive",
            name="Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.drive"],
            is_visible_in_connector=True,
            launch_config={"command": "python", "args": ["-m", "drive"]},
        )
    )
    db.commit()
    request = SimpleNamespace(
        query_params={
            "code": "code",
            "state": _shared_google_state(
                user, "google-drive", ["openid", "scope.drive"]
            ),
        }
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "shared-access",
                    "refresh_token": "shared-refresh",
                    "token_type": "Bearer",
                    "scope": "openid",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 400
    assert "Required Google permissions were not granted" in response.body.decode()
    assert "scope.drive" in response.body.decode()
    assert db.query(UserOAuth).count() == 0


def test_google_shared_callback_drops_restricted_scope_from_old_state(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    monkeypatch.setenv("XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED", "false")
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-drive",
            name="Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["openid"],
            is_visible_in_connector=True,
            launch_config={"command": "python", "args": ["-m", "drive"]},
        )
    )
    db.commit()
    request = SimpleNamespace(
        query_params={
            "code": "code",
            "state": _shared_google_state(user, "google-drive", ["openid"]),
        }
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "shared-access",
                    "refresh_token": "shared-refresh",
                    "token_type": "Bearer",
                    "scope": "openid https://www.googleapis.com/auth/drive",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    shared = db.query(UserOAuth).filter_by(provider="google").one()
    legacy = db.query(UserOAuth).filter_by(provider="google-drive").one()
    assert shared.scope == legacy.scope == "openid"


def test_shared_google_state_preserves_refresh_if_flag_changes_before_callback(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "false")
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-drive",
            name="Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.drive"],
            is_visible_in_connector=True,
            launch_config={"command": "python", "args": ["-m", "drive"]},
        )
    )
    legacy = UserOAuth(
        user_id=int(user.id),
        provider="google-drive",
        provider_user_id="google-subject",
        access_token="old-access",
        refresh_token="old-refresh",
        scope="openid scope.drive",
    )
    db.add(legacy)
    db.commit()
    request = SimpleNamespace(
        query_params={
            "code": "code",
            "state": _shared_google_state(
                user, "google-drive", ["openid", "scope.drive"]
            ),
        }
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "new-access",
                    "token_type": "Bearer",
                    "scope": "openid scope.drive",
                    "expires_in": 3600,
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    db.refresh(legacy)
    assert legacy.access_token == "new-access"
    assert legacy.refresh_token == "old-refresh"
    assert db.query(GoogleOAuthGrant).count() == 0


def test_shared_google_state_rejects_flag_change_without_refresh_fallback(
    db_session, monkeypatch
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "false")
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="google-drive",
            name="Google Drive",
            transport="oauth",
            provider_name="google",
            oauth_scopes=["scope.drive"],
            is_visible_in_connector=True,
            launch_config={"command": "python", "args": ["-m", "drive"]},
        )
    )
    db.commit()
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "new-access",
                    "token_type": "Bearer",
                    "scope": "openid scope.drive",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback(
        "google",
        SimpleNamespace(
            query_params={
                "code": "code",
                "state": _shared_google_state(
                    user, "google-drive", ["openid", "scope.drive"]
                ),
            }
        ),
        db,
        _google_provider(),
    )

    assert response.status_code == 409
    assert db.query(UserOAuth).count() == 0


def test_bare_google_callback_cannot_replace_grant_owned_shared_row(
    db_session, monkeypatch
):
    db, user = db_session
    shared = UserOAuth(
        user_id=int(user.id),
        provider="google",
        provider_user_id="google-subject",
        access_token="shared-access",
        refresh_token="shared-refresh",
        scope="openid scope.drive",
    )
    legacy = UserOAuth(
        user_id=int(user.id),
        provider="google-drive",
        provider_user_id="google-subject",
        access_token="shared-access",
        refresh_token="shared-refresh",
        scope="openid scope.drive",
    )
    db.add_all([shared, legacy])
    db.flush()
    association = create_google_oauth_grant_association(
        db,
        oauth_account=shared,
        oauth_client_id="google-client-id",
        app_id="google-drive",
        legacy_oauth_account=legacy,
    )
    db.commit()
    shared_id = int(shared.id)
    association_id = int(association.id)
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": int(user.id),
            "provider": "google",
            "app_id": None,
        },
        expires_delta=timedelta(minutes=10),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "bare-access",
                    "refresh_token": "bare-refresh",
                    "token_type": "Bearer",
                    "scope": "openid",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback(
        "google",
        SimpleNamespace(query_params={"code": "code", "state": state}),
        db,
        _google_provider(),
    )

    assert response.status_code == 409
    db.expire_all()
    preserved = db.get(UserOAuth, shared_id)
    assert preserved is not None
    assert preserved.access_token == "shared-access"
    assert db.get(GoogleOAuthAppAssociation, association_id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("drive_has_shared_association", [True, False])
async def test_disconnecting_one_shared_google_app_preserves_sibling_grant(
    db_session, monkeypatch, drive_has_shared_association
):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    db, user = db_session
    for app_id, name, scope in (
        ("google-drive", "Google Drive", "scope.drive"),
        ("google-sheets", "Google Sheets", "scope.sheets"),
    ):
        db.add(
            PublicMCPApp(
                app_id=app_id,
                name=name,
                transport="oauth",
                provider_name="google",
                oauth_scopes=[scope],
                is_visible_in_connector=True,
                launch_config={"command": "python", "args": ["-m", app_id]},
            )
        )
    db.commit()
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            side_effect=[
                MockResponse(
                    {
                        "access_token": "drive-access",
                        "refresh_token": "shared-refresh",
                        "token_type": "Bearer",
                        "scope": "openid scope.drive",
                    }
                ),
                MockResponse(
                    {
                        "access_token": "sheets-access",
                        "refresh_token": "shared-refresh",
                        "token_type": "Bearer",
                        "scope": "openid scope.drive scope.sheets",
                    }
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"sub": "google-subject", "email": "alice@example.com"}
            )
        ),
    )
    for app_id, scope in (
        ("google-drive", "scope.drive"),
        ("google-sheets", "scope.sheets"),
    ):
        response = generic_oauth_callback(
            "google",
            SimpleNamespace(
                query_params={
                    "code": f"{app_id}-code",
                    "state": _shared_google_state(user, app_id, ["openid", scope]),
                }
            ),
            db,
            _google_provider(),
        )
        assert response.status_code == 200

    if not drive_has_shared_association:
        # Reproduce a gradual-rollout mix: Drive is still legacy-only while
        # Sheets already relies on the shared row.
        drive_association = (
            db.query(GoogleOAuthAppAssociation)
            .filter(GoogleOAuthAppAssociation.app_id == "google-drive")
            .one()
        )
        db.delete(drive_association)
        db.commit()

    from xagent.web.api.mcp import delete_mcp_server

    drive_server = db.query(MCPServer).filter(MCPServer.name == "Google Drive").one()
    await delete_mcp_server(drive_server.id, current_user=user, db=db)

    assert db.query(UserOAuth).filter_by(provider="google-drive").count() == 0
    assert db.query(UserOAuth).filter_by(provider="google-sheets").count() == 1
    assert db.query(UserOAuth).filter_by(provider="google").count() == 1
    assert db.query(GoogleOAuthGrant).count() == 1
    association = db.query(GoogleOAuthAppAssociation).one()
    assert association.app_id == "google-sheets"


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


def test_gmail_reconnect_revives_same_identity_tombstone_in_place(
    db_session, monkeypatch
):
    db, user = db_session
    tombstone = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        provider_user_id="google-user-1",
        email="alice@gmail.com",
        access_token="",
    )
    db.add(tombstone)
    db.flush()
    watch = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(tombstone.id),
        email="alice@gmail.com",
        history_id="history-1",
        topic_name="projects/demo/topics/xagent-gmail-alice",
        status="failed",
        last_error=gmail_provisioning.GMAIL_RECONNECT_REQUIRED_ERROR,
    )
    db.add(watch)
    db.commit()
    tombstone_id = int(tombstone.id)
    watch_id = int(watch.id)

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
                    "access_token": "reconnected-token",
                    "refresh_token": "reconnected-refresh",
                    "expires_in": 3600,
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
    provisioned_account_ids: list[int] = []

    def capture_best_effort(_db, *, user_id: int, context: str):
        account = (
            _db.query(UserOAuth)
            .filter(UserOAuth.user_id == user_id, UserOAuth.provider == "gmail")
            .one()
        )
        provisioned_account_ids.append(int(account.id))

    monkeypatch.setattr(
        gmail_provisioning,
        "best_effort_provision_gmail_watches_for_user",
        capture_best_effort,
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    reconnected = db.query(UserOAuth).filter(UserOAuth.provider == "gmail").one()
    assert int(reconnected.id) == tombstone_id
    assert reconnected.access_token == "reconnected-token"
    assert reconnected.refresh_token == "reconnected-refresh"
    assert db.get(GmailWatchState, watch_id) is not None
    assert provisioned_account_ids == [tombstone_id]


def test_gmail_reconnect_does_not_reuse_a_different_identity_tombstone(db_session):
    db, user = db_session
    tombstone = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        provider_user_id="google-user-1",
        email="shared-address@gmail.com",
        access_token="",
    )
    db.add(tombstone)
    db.commit()

    matched = auth_api._matching_gmail_reconnect_tombstone(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        connector_key="gmail",
        provider_user_id="google-user-2",
        email="shared-address@gmail.com",
    )

    assert matched is None


def test_gmail_reconnect_does_not_reuse_a_verified_tombstone_when_the_new_callback_has_no_id(
    db_session,
):
    """A reconnect whose callback didn't yield a provider_user_id (e.g. a
    userinfo response missing the configured id field) must not fall back to
    matching a tombstone by email alone when that tombstone already has its
    own verified upstream id recorded -- that id proves it belongs to a
    specific Google identity, and this callback hasn't proven it's the same
    one.
    """
    db, user = db_session
    tombstone = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        provider_user_id="google-user-1",
        email="shared-address@gmail.com",
        access_token="",
    )
    db.add(tombstone)
    db.commit()

    matched = auth_api._matching_gmail_reconnect_tombstone(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        connector_key="gmail",
        provider_user_id=None,
        email="shared-address@gmail.com",
    )

    assert matched is None


def test_gmail_callback_succeeds_when_best_effort_watch_provisioning_raises(
    db_session, monkeypatch, caplog
):
    """A post-commit provisioning failure must not fail the connect itself."""
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

    def raising_best_effort_provision(_db, *, user_id: int, context: str):
        raise RuntimeError("account lookup lost its connection")

    monkeypatch.setattr(
        "xagent.web.services.gmail_provisioning."
        "best_effort_provision_gmail_watches_for_user",
        raising_best_effort_provision,
    )

    caplog.set_level(logging.WARNING, logger=auth_api.__name__)

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    assert "Authentication Failed" not in response.body.decode()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "gmail")
        .one()
    )
    assert oauth_account.email == "alice@gmail.com"
    assert "Best-effort Gmail watch provisioning failed" in caplog.text


def test_gmail_callback_survives_a_raising_account_lookup_during_provisioning(
    db_session, monkeypatch, caplog
):
    """Exercise the callback guard and the service guard as composed.

    The test above stubs provisioning out entirely, so it only proves the
    callback guard works. Here the real service runs and its own account
    lookup raises, which is the failure #1150 reproduced on staging. The
    watch flag must be on for the real service to reach that lookup.
    """
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
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

    real_provision = gmail_provisioning.best_effort_provision_gmail_watches_for_user

    def failing_query(*_args, **_kwargs):
        raise OperationalError("SELECT user_oauth", {}, Exception("connection lost"))

    def provision_with_failing_account_lookup(provision_db, **kwargs):
        # Fail only the lookups the service itself issues; the callback's own
        # queries, which run before the OAuth commit, must succeed. The revert
        # has to happen here rather than at teardown, because this test queries
        # the session again below to assert the OAuth row was committed. `pop`
        # rather than `del` so a refactor that skips the assignment surfaces as
        # its own failure instead of an AttributeError from this line.
        provision_db.query = failing_query
        try:
            return real_provision(provision_db, **kwargs)
        finally:
            provision_db.__dict__.pop("query", None)

    monkeypatch.setattr(
        gmail_provisioning,
        "best_effort_provision_gmail_watches_for_user",
        provision_with_failing_account_lookup,
    )

    caplog.set_level(logging.WARNING, logger=gmail_provisioning.__name__)

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    assert "Authentication Failed" not in response.body.decode()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "gmail")
        .one()
    )
    assert oauth_account.email == "alice@gmail.com"
    assert "Failed to resolve Gmail accounts for user" in caplog.text


def test_oauth_callback_survives_a_raising_post_commit_side_effect(
    db_session, monkeypatch, caplog
):
    """The post-commit region is guarded as a region, not per side effect.

    The tests above all raise from inside Gmail provisioning, which carries its
    own guard, so they cannot tell a per-call guard apart from a regional one.
    Here the guarded wrapper itself raises, which is what a newly added
    post-commit side effect would do before anyone remembers to guard it.
    """
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

    def raising_side_effect(_db, *, user_id: int):
        raise RuntimeError("an unguarded post-commit side effect blew up")

    monkeypatch.setattr(
        auth_api, "_best_effort_ensure_gmail_watches_for_user", raising_side_effect
    )

    caplog.set_level(logging.WARNING, logger=auth_api.__name__)

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    assert "Authentication Failed" not in response.body.decode()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "gmail")
        .one()
    )
    assert oauth_account.email == "alice@gmail.com"
    assert "Post-commit OAuth side effects failed" in caplog.text


def test_oauth_callback_still_fails_when_the_success_page_cannot_be_rendered(
    db_session, monkeypatch
):
    """The swallow covers post-commit side effects only, not response building.

    Rendering runs after the commit too, but a failure there leaves nothing to
    return, so it must keep reaching the outer handler instead of being
    swallowed into a response that does not exist.
    """
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
    monkeypatch.setattr(
        "xagent.web.services.gmail_provisioning."
        "best_effort_provision_gmail_watches_for_user",
        lambda _db, *, user_id, context: None,
    )

    real_html_response = auth_api.HTMLResponse

    def failing_success_page(*args, **kwargs):
        # Fail only the success page; the outer handler's own error page has to
        # keep rendering, otherwise the test could not observe the 500.
        content = kwargs.get("content", args[0] if args else "")
        if "Connected Successfully" in str(content):
            raise RuntimeError("template rendering failed")
        return real_html_response(*args, **kwargs)

    monkeypatch.setattr(auth_api, "HTMLResponse", failing_success_page)

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 500
    assert "Authentication Failed" in response.body.decode()


@pytest.mark.parametrize(
    "user_id_claim",
    [
        pytest.param("7abc", id="non-integer-string"),
        pytest.param(True, id="boolean"),
        pytest.param(7.9, id="fractional-float"),
        pytest.param(7.0, id="integral-float"),
        pytest.param(float("inf"), id="infinite-float"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(2**31, id="larger-than-database-integer"),
    ],
)
def test_oauth_callback_rejects_an_invalid_user_id_claim_before_exchange(
    db_session, monkeypatch, user_id_claim: object
):
    """An invalid state owner must fail before provider and database effects."""
    db, _user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user_id_claim,
            "provider": "google",
            "app_id": "gmail",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "gmail-code", "state": state})
    token_exchange = Mock(
        return_value=MockResponse(
            {
                "access_token": "gmail-token",
                "refresh_token": "gmail-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/gmail.modify",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", token_exchange)
    provision = Mock()
    monkeypatch.setattr(
        "xagent.web.services.gmail_provisioning."
        "best_effort_provision_gmail_watches_for_user",
        provision,
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 400
    assert "Invalid or expired state" in response.body.decode()
    token_exchange.assert_not_called()
    provision.assert_not_called()
    assert db.query(UserOAuth).count() == 0


def test_bare_google_callback_does_not_provision_gmail_watches(db_session, monkeypatch):
    """A bare ``google`` connect must not trigger Gmail watch provisioning.

    With no ``app_id`` claim the callback takes the batch branch and
    ``connector_key`` falls back to the provider name, so the
    ``connector_key == "gmail"`` gate stays closed even though the batch
    connects the Gmail catalog app. That is correct: the bare grant is stored
    with ``UserOAuth.provider == "google"``, which can never back a Gmail
    trigger, so provisioning would only be wasted work. Before this test,
    deleting the gate entirely left the whole suite green.
    """
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="gmail",
            name="Gmail",
            transport="oauth",
            provider_name="google",
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
    provision = Mock()
    monkeypatch.setattr(
        "xagent.web.services.gmail_provisioning."
        "best_effort_provision_gmail_watches_for_user",
        provision,
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    # The connect itself succeeded: the bare grant is stored under the
    # provider name and the Gmail catalog app is connected...
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "google")
        .one()
    )
    assert oauth_account.email == "alice@gmail.com"
    assert "Gmail" in {s.name for s in db.query(MCPServer).all()}
    # ...but Gmail watch provisioning stayed out of it.
    provision.assert_not_called()


def test_app_scoped_non_gmail_callback_does_not_provision_gmail_watches(
    db_session, monkeypatch
):
    """An app-scoped non-Gmail connect must not trigger Gmail provisioning.

    Complements the bare-connect test above with the other route through
    ``connector_key=(app_id or provider)``: here ``app_id`` is present and the
    gate compares it, not the provider name. ``google-calendar`` is Gmail's
    closest neighbor — same provider, different app — so this also pins that
    the gate keys on the app rather than on the provider.
    """
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "google-calendar",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "calendar-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "calendar-token",
                    "refresh_token": "calendar-refresh",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "https://www.googleapis.com/auth/calendar",
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
    provision = Mock()
    monkeypatch.setattr(
        "xagent.web.services.gmail_provisioning."
        "best_effort_provision_gmail_watches_for_user",
        provision,
    )

    response = generic_oauth_callback("google", request, db, _google_provider())

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "google-calendar")
        .one()
    )
    assert oauth_account.email == "alice@gmail.com"
    provision.assert_not_called()


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
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
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

    monkeypatch.setattr(tool_config, "create_public_async_client", FakeAsyncClient)

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


@pytest.mark.asyncio
async def test_meta_refresh_with_revoked_session_raises_permanent(
    db_session, monkeypatch
):
    """Meta nests its refresh error as an object instead of the standard
    top-level string `error` code; OAuthException + code 190 is Meta's
    documented "access token is invalid/expired" signal and must still be
    recognized as a permanent (not merely transient) failure.
    """
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
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
        provider_user_id="meta-user-1",
    )
    db.add(oauth_account)
    db.commit()

    class RevokedAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            return MockResponse(
                {
                    "error": {
                        "message": "Error validating access token: Session has expired.",
                        "type": "OAuthException",
                        "code": 190,
                    }
                },
                status_code=400,
            )

    monkeypatch.setattr(tool_config, "create_public_async_client", RevokedAsyncClient)

    with pytest.raises(tool_config._OAuthRefreshPermanentlyInvalid):
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "meta")


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


def test_bare_meta_login_skips_meta_ads(db_session, monkeypatch):
    """Meta Ads' ads_read scope is new and lives solely on the app row —
    same situation as Facebook's pages_read_user_content — so a bare Meta
    login must not activate its UserMCPServer either, or every ads_read tool
    call would fail against an under-scoped grant while reporting
    "connected" (APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT)."""
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="meta-ads",
            name="Meta Ads",
            description="Meta Ads connector",
            transport="oauth",
            provider_name="meta",
            category="Marketing",
            oauth_scopes=["ads_read"],
            is_visible_in_connector=True,
            launch_config={
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.meta_ads"],
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

    # The bare grant is still created — neither app-scoped server is.
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "meta")
        .one()
    )
    assert oauth_account.access_token == "long-token"

    server_names = {s.name for s in db.query(MCPServer).all()}
    assert "Meta Ads" not in server_names
    assert "Facebook Pages" not in server_names


def test_bare_meta_login_skips_whatsapp_but_still_connects_instagram(
    db_session, monkeypatch
):
    """None of WhatsApp's scopes (business_management, whatsapp_business_*)
    is in the meta provider's default_scopes; they live solely on the app row
    -- same situation as Facebook's pages_read_user_content -- so a bare Meta
    login must not activate its UserMCPServer, or every WhatsApp tool call
    would fail against an under-scoped grant while reporting "connected"
    (APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT).

    Instagram is registered alongside it as a positive control (mirroring
    test_bare_meta_login_skips_facebook_but_still_connects_instagram): its
    required scopes haven't changed, so it must still connect via this same
    bare flow. Without this, a callback that skipped *every* app-scoped and
    bare-eligible server alike (e.g. a bug that stopped creating any
    UserMCPServer at all) would make the assertions below pass vacuously.
    """
    db, user = db_session
    db.add(
        PublicMCPApp(
            app_id="whatsapp",
            name="WhatsApp Business",
            description="WhatsApp Business connector",
            transport="oauth",
            provider_name="meta",
            category="Communication",
            oauth_scopes=[
                "business_management",
                "whatsapp_business_management",
                "whatsapp_business_messaging",
            ],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.whatsapp"],
                "env_mapping": {"META_ACCESS_TOKEN": "access_token"},
                # Match the provenance-owned row seeded in production. An
                # unmarked app_id collision is intentionally treated as an
                # operator-owned custom connector and must not inherit the
                # builtin WhatsApp app-scoped OAuth policy.
                "builtin_provenance": {
                    "registry": "xagent",
                    "app_id": "whatsapp",
                    "version": 1,
                },
            },
        )
    )
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

    # The bare grant is still created -- neither app-scoped server is.
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "meta")
        .one()
    )
    assert oauth_account.access_token == "long-token"

    server_names = {s.name for s in db.query(MCPServer).all()}
    assert "WhatsApp Business" not in server_names
    assert "Instagram" in server_names


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
    db.add_all(
        [
            UserOAuth(
                user_id=user.id,
                provider="meta",
                provider_user_id="actor-meta",
                resource_owner_key="actor:meta",
                access_token="actor-meta-token",
            ),
            UserOAuth(
                user_id=user.id,
                provider="facebook",
                provider_user_id="actor-facebook",
                resource_owner_key="actor:facebook",
                access_token="actor-facebook-token",
            ),
        ]
    )
    db.commit()

    from xagent.web.api.mcp import delete_mcp_server

    facebook_server = (
        db.query(MCPServer).filter(MCPServer.name == "Facebook Pages").one()
    )
    await delete_mcp_server(facebook_server.id, current_user=user, db=db)

    assert (
        db.query(UserOAuth)
        .filter(
            UserOAuth.provider == "facebook",
            UserOAuth.resource_owner_key.is_(None),
        )
        .count()
        == 0
    )
    # The shared bare grant Instagram still relies on must survive.
    assert (
        db.query(UserOAuth)
        .filter(
            UserOAuth.provider == "meta",
            UserOAuth.resource_owner_key.is_(None),
        )
        .count()
        == 1
    )
    assert {
        row.access_token
        for row in db.query(UserOAuth)
        .filter(UserOAuth.resource_owner_key.is_not(None))
        .all()
    } == {"actor-meta-token", "actor-facebook-token"}


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
    db.add_all(
        [
            UserOAuth(
                user_id=user.id,
                provider="meta",
                provider_user_id="actor-meta",
                resource_owner_key="actor:meta",
                access_token="actor-meta-token",
            ),
            UserOAuth(
                user_id=user.id,
                provider="facebook",
                provider_user_id="actor-facebook",
                resource_owner_key="actor:facebook",
                access_token="actor-facebook-token",
            ),
        ]
    )
    server = MCPServer(name="Facebook Pages", transport="oauth", managed="external")
    db.add(server)
    db.commit()
    db.add(UserMCPServer(user_id=user.id, mcpserver_id=server.id, is_owner=True))
    db.commit()

    from xagent.web.api.mcp import delete_mcp_server

    await delete_mcp_server(server.id, current_user=user, db=db)

    assert (
        db.query(UserOAuth).filter(UserOAuth.resource_owner_key.is_(None)).count() == 0
    )
    assert {
        row.access_token
        for row in db.query(UserOAuth)
        .filter(UserOAuth.resource_owner_key.is_not(None))
        .all()
    } == {"actor-meta-token", "actor-facebook-token"}


def test_facebook_server_reads_ignore_actor_owned_email(db_session):
    """Personal server reads must ignore actor-owned OAuth accounts."""
    db, user = db_session
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="meta",
            access_token="bare-meta-token",
            email="alice@example.com",
        )
    )
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="facebook",
            provider_user_id="actor-facebook",
            resource_owner_key="actor:facebook",
            access_token="actor-facebook-token",
            email="actor@example.com",
        )
    )
    server = MCPServer(name="Facebook Pages", transport="oauth", managed="external")
    db.add(server)
    db.commit()
    db.add(UserMCPServer(user_id=user.id, mcpserver_id=server.id, is_owner=True))
    db.commit()

    from xagent.web.api.mcp import get_mcp_server, get_mcp_servers

    responses = get_mcp_servers(current_user=user, db=db)
    response = get_mcp_server(server.id, current_user=user, db=db)

    assert len(responses) == 1
    assert responses[0].connected_account is None
    assert response.connected_account is None


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
