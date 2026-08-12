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
            app_id="intercom",
            name="Intercom",
            description="Intercom connector",
            transport="oauth",
            provider_name="intercom",
            category="Support",
            oauth_scopes=[],
            is_visible_in_connector=False,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.intercom"],
                "env_mapping": {"INTERCOM_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _intercom_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="intercom",
        client_id=encrypt_value("intercom-client-id"),
        client_secret=encrypt_value("intercom-client-secret"),
        token_url="https://api.intercom.io/auth/eagle/token",
        redirect_uri="https://app.example.com/api/auth/intercom/callback",
        userinfo_url="https://api.intercom.io/me",
        user_id_path="id",
        email_path="email",
        default_scopes=[],
    )


def test_normalize_intercom_token_response_maps_token_to_access_token():
    result = auth_api._normalize_intercom_token_response(
        "intercom", {"token": "raw-intercom-token"}
    )
    assert result == {
        "token": "raw-intercom-token",
        "access_token": "raw-intercom-token",
    }


def test_normalize_intercom_token_response_is_a_no_op_for_other_providers():
    token_data = {"token": "not-an-access-token"}
    result = auth_api._normalize_intercom_token_response("zoom", token_data)
    assert result == token_data
    assert "access_token" not in result


def test_normalize_intercom_token_response_prefers_existing_access_token():
    """If Intercom's response shape ever changes to include a real
    access_token, do not clobber it with the (now-stale) `token` field."""
    token_data = {"token": "legacy-token", "access_token": "real-access-token"}
    result = auth_api._normalize_intercom_token_response("intercom", token_data)
    assert result["access_token"] == "real-access-token"


def test_normalize_intercom_token_response_leaves_tokenless_response_untouched():
    token_data = {"type": "error.list", "errors": [{"message": "bad code"}]}
    result = auth_api._normalize_intercom_token_response("intercom", token_data)
    assert result == token_data


def test_intercom_callback_persists_access_token_from_non_standard_response(
    db_session, monkeypatch
):
    """Intercom's token endpoint returns {"token": ...}, not the standard
    {"access_token": ...} -- end-to-end proof that generic_oauth_callback still
    ends up persisting a usable access_token for it."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "intercom",
            "app_id": "intercom",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "intercom-code", "state": state})

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse({"token": "raw-intercom-token"})),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"type": "admin", "id": "admin-1", "email": "alice@example.com"}
            )
        ),
    )

    response = generic_oauth_callback("intercom", request, db, _intercom_provider())

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "intercom")
        .one()
    )
    assert oauth_account.access_token == "raw-intercom-token"
    assert oauth_account.provider_user_id == "admin-1"
    assert oauth_account.email == "alice@example.com"

    server = db.query(MCPServer).filter(MCPServer.name == "Intercom").one()
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


def test_intercom_callback_fails_cleanly_when_token_exchange_yields_no_token(
    db_session, monkeypatch
):
    """Intercom's error envelope ({"type": "error.list", ...}) does not match
    the `"error" in token_data` guard, so a failed exchange must still be
    caught by the access_token guard rather than falling through to a raw
    IntegrityError from UserOAuth.access_token's NOT NULL constraint."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "intercom",
            "app_id": "intercom",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "bad-code", "state": state})

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"type": "error.list", "errors": [{"message": "invalid code"}]}
            )
        ),
    )
    get_mock = Mock()
    monkeypatch.setattr(auth_api.requests, "get", get_mock)

    response = generic_oauth_callback("intercom", request, db, _intercom_provider())

    assert response.status_code == 400
    assert "did not return an access token" in response.body.decode()
    assert "IntegrityError" not in response.body.decode()
    get_mock.assert_not_called()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "intercom")
        .count()
        == 0
    )
