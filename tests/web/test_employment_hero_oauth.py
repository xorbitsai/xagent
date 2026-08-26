from __future__ import annotations

from datetime import timedelta
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
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data


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
            app_id="employment-hero",
            name="Employment Hero",
            transport="oauth",
            provider_name="employment-hero",
            category="HR",
            oauth_scopes=[
                "organisations:list",
                "employees:list",
                "employees:show",
                "teams:list",
                "timesheet_entries:list",
            ],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.employment_hero"],
                "env_mapping": {"EMPLOYMENT_HERO_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _employment_hero_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="employment-hero",
        client_id=encrypt_value("eh-client-id"),
        client_secret=encrypt_value("eh-client-secret"),
        token_url="https://oauth.employmenthero.com/oauth2/token",
        redirect_uri="https://app.example.com/api/auth/employment-hero/callback",
        userinfo_url="",
        user_id_path="id",
        email_path="email",
        default_scopes=[],
    )


def _callback_request(user, *, code_verifier: str | None = None) -> SimpleNamespace:
    state_payload = {
        "type": "oauth_state",
        "user_id": user.id,
        "provider": "employment-hero",
        "app_id": "employment-hero",
    }
    if code_verifier is not None:
        state_payload["code_verifier"] = encrypt_value(code_verifier)
    state = create_access_token(data=state_payload, expires_delta=timedelta(minutes=10))
    return SimpleNamespace(query_params={"code": "eh-code", "state": state})


def test_callback_exchanges_code_and_skips_userinfo_lookup(db_session, monkeypatch):
    """Employment Hero has no identity endpoint this callback's flat userinfo
    lookup can use (see the provider row's comment) -- the callback must not
    attempt a GET at all, and must still persist a connected account with a
    NULL identity rather than erroring."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "eh-token",
                "refresh_token": "eh-refresh",
                "token_type": "bearer",
                "expires_in": 900,
            }
        )
    )
    get = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", get)

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    get.assert_not_called()

    data = post.call_args.kwargs["data"]
    assert data["grant_type"] == "authorization_code"
    assert data["client_id"] == "eh-client-id"
    assert data["client_secret"] == "eh-client-secret"
    assert data["code_verifier"] == "verifier-value"

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .one()
    )
    assert oauth_account.access_token == "eh-token"
    assert oauth_account.refresh_token == "eh-refresh"
    assert oauth_account.provider_user_id is None
    assert oauth_account.email is None

    server = db.query(MCPServer).filter(MCPServer.name == "Employment Hero").one()
    assert server.transport == "oauth"
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    assert user_mcp.is_active is True


def test_callback_without_code_verifier_omits_it_from_token_exchange(
    db_session, monkeypatch
):
    """Sanity companion to the PKCE test above -- if the login route never
    minted a verifier (e.g. an older signed state token), the exchange must
    not send a bogus code_verifier field."""
    db, user = db_session
    request = _callback_request(user, code_verifier=None)

    post = Mock(
        return_value=MockResponse(
            {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", Mock())

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    assert "code_verifier" not in post.call_args.kwargs["data"]
