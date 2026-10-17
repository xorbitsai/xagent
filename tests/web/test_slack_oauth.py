from __future__ import annotations

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
from xagent.web.api.auth import (
    _merge_oauth_scopes,
    _oauth_scope_separator,
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
            "session_epoch": user.session_epoch,
        },
        expires_delta=timedelta(minutes=10),
    )
    return SimpleNamespace(query_params={"code": "slack-code", "state": state})


@pytest.mark.parametrize("phase", ["before-callback", "during-exchange"])
def test_password_change_fences_pending_slack_connector_write(
    db_session, monkeypatch, phase
):
    db, user = db_session
    request = _callback_request(db, user)
    user_id = user.id
    bind = db.get_bind()

    def rotate():
        with sessionmaker(bind=bind)() as other:
            other.query(User).filter(User.id == user_id).update(
                {User.session_epoch: "revoked"}
            )
            other.commit()

    def exchange(*args, **kwargs):
        rotate()
        return MockResponse(
            {"ok": True, "access_token": "xoxb-token", "scope": "chat:write"}
        )

    post = Mock(side_effect=exchange)
    monkeypatch.setattr(auth_api.requests, "post", post)
    rejection = Mock()
    monkeypatch.setattr(auth_api, "_log_oauth_callback_rejection", rejection)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"ok": True, "team_id": "T1", "team": "Acme"})),
    )
    if phase == "before-callback":
        rotate()
    response = generic_oauth_callback("slack", request, db, _slack_provider())
    assert response.status_code == 400
    assert "Invalid or expired state" in response.body.decode()
    assert db.query(UserOAuth).count() == 0
    assert post.call_count == (1 if phase == "during-exchange" else 0)
    rejection.assert_called_once_with(
        "slack",
        "session_epoch_changed_after_exchange"
        if phase == "during-exchange"
        else "state_session_mismatch",
        app_id="slack",
        actor_flow=False,
        provider_error=None,
    )


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
