from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import (
    _merge_oauth_scopes,
    _oauth_scope_separator,
    create_access_token,
    generic_oauth_callback,
)
from xagent.web.models.database import Base
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
            app_id="slack",
            name="Slack",
            transport="oauth",
            provider_name="slack",
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _slack_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="slack",
        client_id=encrypt_value("slack-client-id"),
        client_secret=encrypt_value("slack-client-secret"),
        token_url="https://slack.com/api/oauth.v2.access",
        redirect_uri="https://app.example.com/api/auth/slack/callback",
        userinfo_url="https://slack.com/api/auth.test",
        user_id_path="team_id",
        email_path="team",
        default_scopes=["chat:write", "chat:write.public", "channels:read"],
    )


def _callback_request(db, user) -> SimpleNamespace:
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "slack",
            "app_id": "slack",
        },
        expires_delta=timedelta(minutes=10),
    )
    return SimpleNamespace(query_params={"code": "slack-code", "state": state})


def test_slack_callback_persists_workspace_identity(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "ok": True,
                "access_token": "xoxb-token",
                "token_type": "bot",
                "scope": "chat:write,chat:write.public,channels:read",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"ok": True, "team_id": "T0123", "team": "acme-workspace"}
            )
        ),
    )

    response = generic_oauth_callback(
        "slack", _callback_request(db, user), db, _slack_provider()
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "slack")
        .one()
    )
    assert oauth_account.access_token == "xoxb-token"
    assert oauth_account.provider_user_id == "T0123"
    assert oauth_account.email == "acme-workspace"

    # Slack (unlike some OAuth providers) accepts client_id/client_secret in
    # the token-exchange POST body rather than requiring HTTP Basic auth —
    # this PR relies on that to avoid a Slack-specific token-exchange path,
    # so pin the request shape rather than leaving it unverified.
    assert mock_post.call_args.args[0] == "https://slack.com/api/oauth.v2.access"
    assert mock_post.call_args.kwargs["data"] == {
        "grant_type": "authorization_code",
        "code": "slack-code",
        "client_id": "slack-client-id",
        "client_secret": "slack-client-secret",
        "redirect_uri": "https://app.example.com/api/auth/slack/callback",
    }


def test_slack_callback_fails_when_auth_test_reports_ok_false(db_session, monkeypatch):
    """Slack answers HTTP 200 with {"ok": false} for a bad/revoked token; the
    callback must fail visibly instead of persisting a dead "connected"
    account with no identity."""
    db, user = db_session
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"ok": True, "access_token": "xoxb-token", "token_type": "bot"}
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"ok": False, "error": "invalid_auth"})),
    )

    response = generic_oauth_callback(
        "slack", _callback_request(db, user), db, _slack_provider()
    )

    assert response.status_code == 400
    assert "invalid_auth" in response.body.decode()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "slack")
        .first()
        is None
    )


def test_slack_scopes_join_with_spaces():
    """Regression pin for the scope-string form: Slack's docs describe
    comma-separated scopes, but the space-joined form produced by the generic
    separator was verified working end-to-end against a real workspace. A
    refactor of _oauth_scope_separator must not silently change this."""
    assert _oauth_scope_separator("slack") == " "
    provider = _slack_provider()
    joined = _oauth_scope_separator("slack").join(
        _merge_oauth_scopes(provider.default_scopes, None)
    )
    assert joined == "chat:write chat:write.public channels:read"
