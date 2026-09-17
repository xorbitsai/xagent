from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.auth import (
    _merge_oauth_scopes,
    _oauth_scope_separator,
    create_access_token,
    generic_oauth_callback,
    generic_oauth_login,
)
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.tools import config as tool_config


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        raise_on_json: bool = False,
    ):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not JSON")
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
            app_id="linear",
            name="Linear",
            transport="oauth",
            provider_name="linear",
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _linear_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="linear",
        client_id=encrypt_value("linear-client-id"),
        client_secret=encrypt_value("linear-client-secret"),
        auth_url="https://linear.app/oauth/authorize",
        token_url="https://api.linear.app/oauth/token",
        redirect_uri="https://app.example.com/api/auth/linear/callback",
        # Deliberately empty, matching the registry row: Linear's API is
        # GraphQL-only, so there is no compatible flat REST userinfo endpoint.
        userinfo_url="",
        user_id_path="id",
        email_path="email",
        # NOT the same as the registry provider row's default_scopes
        # (["read"] there -- the "write" scope lives on the app row and is
        # merged in via app_id at authorize time). This fixture bypasses
        # that merge (db_provider is passed in directly, app_id lookup is
        # separate), so it uses both scopes directly to exercise the ","
        # separator regardless of the merge path.
        default_scopes=["read", "write"],
    )


def _callback_request(db, user) -> SimpleNamespace:
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "linear",
            "app_id": "linear",
        },
        expires_delta=timedelta(minutes=10),
    )
    return SimpleNamespace(query_params={"code": "linear-code", "state": state})


def test_scope_separator_joins_linear_scopes_with_commas():
    """Linear's authorize endpoint documents scope as a comma-separated
    list, unlike most providers here — a refactor of _oauth_scope_separator
    must not silently change this."""
    assert _oauth_scope_separator("linear") == ","
    provider = _linear_provider()
    joined = _oauth_scope_separator("linear").join(
        _merge_oauth_scopes(provider.default_scopes, None)
    )
    assert joined == "read,write"


def test_login_sends_comma_separated_scope_param(db_session):
    db, user = db_session
    token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )

    response = generic_oauth_login(
        "linear", token=token, app_id="linear", db=db, db_provider=_linear_provider()
    )

    assert response.status_code == 307
    assert "scope=read%2Cwrite" in response.headers["location"]


def test_login_forces_consent_prompt(db_session):
    """Linear does not auto-reprompt when a later request asks for a
    broader scope set than a previously granted token (confirmed against
    Linear's own OAuth docs) -- without prompt=consent, a bare
    read-only connect followed by an app-scoped read+write connect could
    silently leave the user on the earlier, narrower grant."""
    db, user = db_session
    token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )

    response = generic_oauth_login(
        "linear", token=token, app_id="linear", db=db, db_provider=_linear_provider()
    )

    assert response.status_code == 307
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["prompt"] == ["consent"]


_LINEAR_VIEWER_RESPONSE = MockResponse(
    {"data": {"viewer": {"id": "linear-user-1", "email": "ada@example.com"}}}
)


def test_callback_fetches_identity_via_graphql_viewer_query(db_session, monkeypatch):
    """With userinfo_url left empty, the callback must skip the flat-REST
    lookup (no GET attempted) but still fetch identity via a `viewer`
    GraphQL query against Linear's own endpoint, and persist it."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "refresh_token": "linear-refresh",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    # Linear access tokens expire in 24 hours.
                    "expires_in": 86400,
                }
            ),
            _LINEAR_VIEWER_RESPONSE,
        ]
    )
    mock_get = Mock()
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)
    monkeypatch.setattr("xagent.web.api.auth.requests.get", mock_get)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 200
    mock_get.assert_not_called()
    assert mock_post.call_count == 2
    viewer_call = mock_post.call_args_list[1]
    assert viewer_call.args[0] == "https://api.linear.app/graphql"
    assert viewer_call.kwargs["headers"]["Authorization"] == "Bearer linear-token"
    assert "viewer" in viewer_call.kwargs["json"]["query"]

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "linear")
        .first()
    )
    assert oauth_account is not None
    assert oauth_account.access_token == "linear-token"
    assert oauth_account.refresh_token == "linear-refresh"
    assert oauth_account.expires_at is not None
    assert oauth_account.email == "ada@example.com"
    assert oauth_account.provider_user_id == "linear-user-1"


def test_callback_uses_graphql_identity_even_if_userinfo_url_is_set(
    db_session, monkeypatch
):
    """The Linear branch must win on provider name alone, not merely because
    userinfo_url happens to be empty today -- if a provider row's
    userinfo_url were ever populated for Linear (e.g. an admin edit via
    update_provider), the generic REST-GET branch would otherwise run
    instead, fail silently against Linear's GraphQL-only API, and persist
    the connection as "healthy" with no identity."""
    db, user = db_session
    provider = _linear_provider()
    provider.userinfo_url = "https://example.com/should-not-be-called"
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "refresh_token": "linear-refresh",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            _LINEAR_VIEWER_RESPONSE,
        ]
    )
    mock_get = Mock()
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)
    monkeypatch.setattr("xagent.web.api.auth.requests.get", mock_get)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, provider
    )

    assert response.status_code == 200
    mock_get.assert_not_called()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "linear")
        .first()
    )
    assert oauth_account.provider_user_id == "linear-user-1"
    assert oauth_account.email == "ada@example.com"


def test_callback_fails_when_viewer_query_is_rejected(db_session, monkeypatch):
    """A token Linear won't honour must be caught here (this doubles as
    post-exchange token verification) rather than persisted as a healthy
    connection that then fails opaquely from inside a later tool call."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "bad-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            MockResponse(
                {"errors": [{"message": "Authentication required"}]}, status_code=200
            ),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    assert "Authentication required" in response.body.decode()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "linear")
        .first()
    )
    assert oauth_account is None


def test_callback_surfaces_error_detail_on_non_200_viewer_response(
    db_session, monkeypatch
):
    """A non-200 response with a structured error body must not collapse
    to the opaque "status 401" -- the actual reason (e.g. an expired or
    wrong-scope token) is what lets an operator self-serve the fix."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "bad-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            MockResponse({"errors": [{"message": "Not authorized"}]}, status_code=401),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    assert "Not authorized" in response.body.decode()


def test_callback_surfaces_raw_body_when_non_200_error_is_not_graphql_shaped(
    db_session, monkeypatch
):
    """A non-200 error body that isn't the {"errors": [...]} GraphQL shape
    (e.g. a differently-keyed JSON error, or an HTML gateway/WAF page on a
    502/504) must still surface something, not silently collapse to a bare
    "status 502" with the actual body detail discarded."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            MockResponse(
                {"message": "Bad Gateway"},
                status_code=502,
                text='{"message": "Bad Gateway"}',
            ),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    assert "Bad Gateway" in response.body.decode()


def test_callback_surfaces_raw_body_when_non_200_response_is_not_json(
    db_session, monkeypatch
):
    """A non-200 error body that isn't JSON at all (e.g. a plain-text or
    HTML gateway page) must hit the `except ValueError` around parsing it as
    GraphQL errors and fall through to the raw-text fallback, not raise
    unhandled out of the callback."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            MockResponse(
                status_code=503,
                text="<html>Service Unavailable</html>",
                raise_on_json=True,
            ),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    assert "Service Unavailable" in response.body.decode()


def test_callback_fails_with_clear_message_when_200_viewer_response_is_not_json(
    db_session, monkeypatch
):
    """A 200 response whose body isn't valid JSON must raise a clear,
    truncated message rather than an unhandled exception escaping the
    callback."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            MockResponse(status_code=200, text="not json", raise_on_json=True),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    assert "non-JSON response" in response.body.decode()


def test_callback_fails_when_200_response_body_is_not_a_dict(db_session, monkeypatch):
    """A 200 response whose JSON body parses to something other than an
    object (e.g. a bare list) must raise a clear message instead of an
    unhandled AttributeError from treating it as a dict."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            MockResponse([1, 2, 3], status_code=200),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    assert "unexpected response body" in response.body.decode()


def test_callback_fails_with_generic_message_when_viewer_missing_and_no_errors(
    db_session, monkeypatch
):
    """A 200 response with neither a usable `viewer` nor an `errors` array
    (an empty-but-valid GraphQL response) must fall back to a generic
    "did not return a viewer" message rather than a blank/None detail."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            MockResponse({"data": {}}, status_code=200),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    assert "did not return a viewer" in response.body.decode()


def test_callback_distinguishes_network_failure_from_provider_rejection(
    db_session, monkeypatch
):
    """A network-level failure (Linear never actually responded) must not
    be worded as "the provider reported" -- that misattributes a
    client-side/connectivity problem to Linear."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "linear-token",
                    "token_type": "Bearer",
                    "scope": "read,write",
                    "expires_in": 86400,
                }
            ),
            requests.exceptions.ConnectionError("boom"),
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 400
    body = response.body.decode()
    assert "Could not reach Linear" in body
    assert "The provider reported" not in body


def test_callback_normalizes_legacy_array_scope_to_string(db_session, monkeypatch):
    """Linear OAuth applications created before December 1, 2023 return
    "scope" as a list of strings rather than a joined string. UserOAuth.scope
    is a plain String column, so persisting the list as-is would raise at
    flush time instead of saving a valid connection. The joined string is
    always space-separated regardless of provider (unlike the comma
    separator Linear's authorize request itself uses), matching the format
    every reader of this column already expects."""
    db, user = db_session
    mock_post = Mock(
        side_effect=[
            MockResponse(
                {
                    "access_token": "legacy-linear-token",
                    "refresh_token": "legacy-linear-refresh",
                    "token_type": "Bearer",
                    "scope": ["read", "write"],
                    "expires_in": 86400,
                }
            ),
            _LINEAR_VIEWER_RESPONSE,
        ]
    )
    monkeypatch.setattr("xagent.web.api.auth.requests.post", mock_post)
    monkeypatch.setattr("xagent.web.api.auth.requests.get", Mock())

    response = generic_oauth_callback(
        "linear", _callback_request(db, user), db, _linear_provider()
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "linear")
        .one()
    )
    assert oauth_account.scope == "read write"


async def test_linear_expired_token_refresh_uses_generic_form_body(
    db_session, monkeypatch
):
    """Linear access tokens expire in 24 hours and have no provider-specific
    branch in refresh_oauth_token_if_needed, so this exercises the fully
    generic refresh path: form-encoded body, grant_type=refresh_token, and
    client credentials sent in the body (not Basic Auth, not JSON)."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="linear",
            name="Linear",
            client_id=encrypt_value("linear-client-id"),
            client_secret=encrypt_value("linear-client-secret"),
            auth_url="https://linear.app/oauth/authorize",
            token_url="https://api.linear.app/oauth/token",
            redirect_uri="https://app.example.com/api/auth/linear/callback",
            userinfo_url="",
            user_id_path="id",
            email_path="email",
            default_scopes=["read"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="linear",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="linear-user-1",
    )
    db.add(oauth_account)
    db.commit()

    captured_requests = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            captured_requests.append((url, kwargs))
            return MockResponse(
                {
                    "access_token": "new-token",
                    "refresh_token": "new-refresh",
                    "token_type": "Bearer",
                    "expires_in": 86400,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "linear")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert oauth_account.expires_at > datetime.now(timezone.utc) + timedelta(hours=1)
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://api.linear.app/oauth/token"
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "linear-client-id",
        "client_secret": "linear-client-secret",
    }
    assert "json" not in kwargs
    assert "auth" not in kwargs
    assert kwargs["headers"] == {}
