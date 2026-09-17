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
    create_access_token,
    generic_oauth_callback,
    generic_oauth_login,
)
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


class NonJsonResponse(MockResponse):
    def json(self):
        raise ValueError("response body is not JSON")


def _intercom_app(*, is_visible_in_connector: bool) -> PublicMCPApp:
    return PublicMCPApp(
        app_id="intercom",
        name="Intercom",
        description="Intercom connector",
        transport="oauth",
        provider_name="intercom",
        category="Support",
        oauth_scopes=[],
        is_visible_in_connector=is_visible_in_connector,
        launch_config={
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.intercom"],
            "env_mapping": {"INTERCOM_ACCESS_TOKEN": "access_token"},
        },
    )


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
    # Visible here: these tests exercise token normalization/persistence, a
    # concern orthogonal to the release-visibility gate. The gate itself
    # (production ships this app with is_visible_in_connector=False) is
    # covered separately below, against its own fixture.
    db.add(_intercom_app(is_visible_in_connector=True))
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


@pytest.fixture()
def hidden_db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.add(_intercom_app(is_visible_in_connector=False))
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
        auth_url="https://app.intercom.com/oauth",
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
    body = response.body.decode()
    assert "did not return an access token" in body
    # Pins _extract_provider_error_message: the error.list envelope's own
    # detail must actually reach the response, not just a generic message.
    assert "invalid code" in body
    assert "IntegrityError" not in body
    get_mock.assert_not_called()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "intercom")
        .count()
        == 0
    )


def test_intercom_error_list_message_is_length_capped(db_session, monkeypatch):
    """The error.list envelope's detail is echoed to the browser the same
    way the standard error/error_description branch is -- it must be
    capped the same way (500 chars), not echoed unbounded."""
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

    long_message = "x" * 10_000
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"type": "error.list", "errors": [{"message": long_message}]}
            )
        ),
    )
    monkeypatch.setattr(auth_api.requests, "get", Mock())

    response = generic_oauth_callback("intercom", request, db, _intercom_provider())

    assert response.status_code == 400
    body = response.body.decode()
    assert long_message not in body
    assert "x" * 500 in body
    assert "x" * 501 not in body


def test_hidden_intercom_app_rejects_single_app_oauth_connect(
    hidden_db_session, monkeypatch
):
    """Production ships Intercom with is_visible_in_connector=False pending
    live-workspace verification. generic_oauth_callback's builtin_oauth path
    used to be the one connect path _reject_hidden_catalog_app's docstring
    flagged as NOT enforcing that gate (#1203) -- an authenticated user who
    simply knew (or guessed) the app_id could still connect a hidden,
    unverified, customer-facing write connector. This asserts the gate now
    actually blocks it, symmetric with the existing mcp_oauth-path coverage
    in test_mcp_oauth_flow.py::test_connect_app_rejects_hidden_mcp_oauth_app.
    """
    db, user = hidden_db_session
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
    post_mock = Mock()
    get_mock = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post_mock)
    monkeypatch.setattr(auth_api.requests, "get", get_mock)

    response = generic_oauth_callback("intercom", request, db, _intercom_provider())

    assert response.status_code == 404
    # The gate fires before any token exchange is attempted, not just before
    # persistence -- a hidden app should not even reach the provider.
    post_mock.assert_not_called()
    get_mock.assert_not_called()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "intercom")
        .count()
        == 0
    )
    assert db.query(MCPServer).filter(MCPServer.name == "Intercom").count() == 0


def test_hidden_intercom_app_is_skipped_during_bare_provider_oauth_connect(
    hidden_db_session, monkeypatch
):
    """Mirror of the single-app case above, for the app_id-less ("bare
    provider") batch connect branch: a hidden app must be skipped like a
    mis-tagged non-oauth app, not abort the whole batch and not silently
    create its MCPServer association."""
    db, user = hidden_db_session
    state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": "intercom"},
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
    # The bare provider-level grant is still created (same as the
    # AppNotOAuthError skip case) -- only the app-specific MCPServer
    # association for the hidden app is withheld.
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "intercom")
        .count()
        == 1
    )
    assert db.query(MCPServer).filter(MCPServer.name == "Intercom").count() == 0


def test_hidden_intercom_app_rejects_oauth_login_redirect(hidden_db_session):
    """The gate must also cover the login hop, not just the callback --
    otherwise a hidden connector's real consent screen is still shown to the
    user (no security bypass, since the callback still blocks the connect,
    but confusing: the app doesn't feel hidden if you can watch it start
    connecting at the provider)."""
    db, user = hidden_db_session
    token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )

    response = generic_oauth_login(
        "intercom",
        token=token,
        app_id="intercom",
        redirect=None,
        db=db,
        db_provider=_intercom_provider(),
    )

    assert response.status_code == 404


def test_access_token_guard_applies_to_a_non_intercom_provider_too(monkeypatch):
    """The access_token guard added alongside the intercom fix (auth.py) is
    shared, provider-agnostic code -- it must not only be exercised through
    provider="intercom". Zoom stands in for "some other provider" here."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()
    user = User(username="bob", password_hash="x", is_admin=False)
    db.add(user)
    db.add(
        PublicMCPApp(
            app_id="zoom",
            name="Zoom",
            transport="oauth",
            provider_name="zoom",
            is_visible_in_connector=True,
        )
    )
    db.commit()
    db.refresh(user)

    zoom_provider = SimpleNamespace(
        provider_name="zoom",
        client_id=encrypt_value("zoom-client-id"),
        client_secret=encrypt_value("zoom-client-secret"),
        token_url="https://zoom.us/oauth/token",
        redirect_uri="https://app.example.com/api/auth/zoom/callback",
        userinfo_url="https://api.zoom.us/v2/users/me",
        user_id_path="id",
        email_path="email",
        default_scopes=[],
    )
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "zoom",
            "app_id": "zoom",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "zoom-code", "state": state})
    # No "error" key (would hit the earlier guard) and no access_token: an
    # atypical but real-world shape for a misbehaving/proxied token endpoint.
    monkeypatch.setattr(
        auth_api.requests, "post", Mock(return_value=MockResponse({"foo": "bar"}))
    )
    get_mock = Mock()
    monkeypatch.setattr(auth_api.requests, "get", get_mock)

    response = generic_oauth_callback("zoom", request, db, zoom_provider)

    assert response.status_code == 400
    assert "did not return an access token" in response.body.decode()
    get_mock.assert_not_called()
    assert db.query(UserOAuth).filter(UserOAuth.provider == "zoom").count() == 0
    db.close()
    engine.dispose()


def test_intercom_callback_rejects_non_json_token_response(db_session, monkeypatch):
    """token_response.json() is now guarded the same way for every provider
    (generic_oauth_callback, api/auth.py), not just GitHub: a non-JSON body
    gets the clean 400 error page instead of falling through to the generic
    exception handler as an unhandled-feeling 500. Previously pinned as a
    known 500 here; this is the fix, not a regression."""
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
        Mock(return_value=NonJsonResponse(status_code=502, text="<html>bad gateway")),
    )

    response = generic_oauth_callback("intercom", request, db, _intercom_provider())

    assert response.status_code == 400
    assert "could not be parsed" in response.body.decode()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "intercom")
        .count()
        == 0
    )


def test_intercom_callback_rejects_non_2xx_json_response_with_no_error_field(
    db_session, monkeypatch
):
    """A non-2xx status with a JSON body carrying neither "error" nor a
    token must be rejected on the status code itself, not only inferred
    from the body missing an access_token -- otherwise a non-2xx response
    that happens to carry an access_token-shaped field (a misbehaving
    proxy/gateway, a stale cached body) would be trusted as success purely
    because the body shape looked fine."""
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
        Mock(return_value=MockResponse({"foo": "bar"}, status_code=503)),
    )
    get_mock = Mock()
    monkeypatch.setattr(auth_api.requests, "get", get_mock)

    response = generic_oauth_callback("intercom", request, db, _intercom_provider())

    assert response.status_code == 400
    assert "status 503" in response.body.decode()
    get_mock.assert_not_called()
