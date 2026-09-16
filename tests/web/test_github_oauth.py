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
from xagent.web.api.auth import (
    create_access_token,
    generic_oauth_callback,
    generic_oauth_login,
)
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
            app_id="github",
            name="GitHub",
            description="GitHub connector",
            transport="oauth",
            provider_name="github",
            category="Development",
            # read:org was dropped from the canonical registry (no tool
            # calls an organization endpoint) -- kept in sync here too,
            # even though get_app_by_id overlays the canonical registry's
            # oauth_scopes for this builtin app_id regardless of this row's
            # value (see test_generic_oauth_login.py's
            # test_github_login_requests_exact_canonical_scope).
            oauth_scopes=["repo", "user:email"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.github"],
                "env_mapping": {"GITHUB_ACCESS_TOKEN": "access_token"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _github_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="github",
        client_id=encrypt_value("github-client-id"),
        client_secret=encrypt_value("github-client-secret"),
        auth_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        redirect_uri="https://app.example.com/api/auth/github/callback",
        userinfo_url="https://api.github.com/user",
        user_id_path="id",
        email_path="login",
        default_scopes=["read:user"],
    )


def test_github_callback_requests_json_and_sends_secret_in_body(
    db_session, monkeypatch
):
    """GitHub's token endpoint answers with a form-urlencoded body unless the
    request explicitly asks for JSON via Accept -- assert the exchange sets
    that header (unlike zoom, GitHub still takes client_id/secret in the body,
    not HTTP Basic Auth)."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "github-code", "state": state})
    actor_account = UserOAuth(
        user_id=user.id,
        provider="github",
        provider_user_id="actor-octocat",
        resource_owner_key="actor:github",
        access_token="actor-token",
        email="actor@example.com",
    )
    db.add_all(
        [
            UserOAuth(
                user_id=user.id,
                provider="github",
                provider_user_id="stale-ordinary",
                access_token="stale-ordinary-token",
            ),
            actor_account,
        ]
    )
    db.commit()
    actor_account_id = int(actor_account.id)

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "github-token",
                "token_type": "bearer",
                "scope": "repo,user:email",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"id": 42, "login": "octocat"})),
    )

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 200
    assert post.call_args.kwargs["headers"]["Accept"] == "application/json"
    assert post.call_args.kwargs["auth"] is None
    assert post.call_args.kwargs["data"]["client_id"] == "github-client-id"
    assert post.call_args.kwargs["data"]["client_secret"] == "github-client-secret"
    # Pins that the Employment Hero-specific query-param rewrite
    # (grant_type/redirect_uri moved out of the body into `params`) stays
    # scoped to that family and never leaks onto an unrelated provider's
    # token exchange, e.g. if matches_provider_family's matching were ever
    # accidentally widened.
    assert post.call_args.kwargs["params"] is None
    assert "grant_type" in post.call_args.kwargs["data"]

    oauth_account = (
        db.query(UserOAuth)
        .filter(
            UserOAuth.user_id == user.id,
            UserOAuth.provider == "github",
            UserOAuth.resource_owner_key.is_(None),
        )
        .one()
    )
    assert oauth_account.access_token == "github-token"
    assert oauth_account.provider_user_id == "42"
    assert oauth_account.email == "octocat"

    preserved_actor = db.get(UserOAuth, actor_account_id)
    assert preserved_actor is actor_account
    assert preserved_actor.resource_owner_key == "actor:github"
    assert preserved_actor.access_token == "actor-token"
    assert preserved_actor.provider_user_id == "actor-octocat"

    server = db.query(MCPServer).filter(MCPServer.name == "GitHub").one()
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


def test_bare_github_callback_is_rejected_even_for_a_pre_existing_state(
    db_session, monkeypatch
):
    """generic_oauth_login's own guard only stops a NEW bare state from being
    minted -- it can't protect a bare (app_id-less) state that was already
    signed before that guard was deployed and is still within its 10-minute
    TTL, nor a future internal caller that reaches this callback directly
    with one. The callback must therefore carry the same rejection, checked
    before any token exchange or UserOAuth write, so a fresh user has no
    identity-only grant persisted for a provider that requires an
    app-scoped one."""
    db, user = db_session
    # No app_id in the state -- simulates a bare state minted before the
    # callback-side guard existed (or a caller bypassing generic_oauth_login).
    state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": "github"},
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "github-code", "state": state})

    post = Mock(
        return_value=MockResponse(
            {
                "access_token": "bare-github-token",
                "token_type": "bearer",
                "scope": "read:user",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 404
    # Rejected before the token exchange -- no request was ever made.
    post.assert_not_called()
    assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).first() is None
    assert db.query(MCPServer).filter(MCPServer.name == "GitHub").first() is None


def test_bare_github_login_does_not_disturb_an_existing_scoped_connection(
    db_session, monkeypatch
):
    """The critical regression: a user with a fully-scoped, active GitHub
    connection must not have it silently downgraded by re-running the bare
    (app_id-less) login route. Before the login-time guard, this callback
    would delete-and-replace the existing UserOAuth row (same
    provider="github" key as the app-scoped connection) with an
    identity-only grant while leaving the MCPServer/UserMCPServer row
    active -- reporting "connected" against a token that can no longer
    actually use the connector's tools."""
    db, user = db_session

    # Establish a fully-scoped connection first, exactly as the app-scoped
    # callback test above does.
    scoped_state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    scoped_request = SimpleNamespace(
        query_params={"code": "github-code", "state": scoped_state}
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "scoped-github-token",
                    "token_type": "bearer",
                    "scope": "repo,user:email",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"id": 42, "login": "octocat"})),
    )
    setup_response = generic_oauth_callback(
        "github", scoped_request, db, _github_provider()
    )
    assert setup_response.status_code == 200

    # Now attempt the bare login route a malicious/confused re-connect
    # attempt would use.
    login_token = create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )
    login_response = generic_oauth_login(
        provider="github",
        token=login_token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=_github_provider(),
    )

    assert login_response.status_code == 404

    # The original scoped connection and its active server are untouched.
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .one()
    )
    assert oauth_account.access_token == "scoped-github-token"
    server = db.query(MCPServer).filter(MCPServer.name == "GitHub").one()
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    assert user_mcp.is_active is True


def test_bare_github_callback_does_not_disturb_an_existing_scoped_connection(
    db_session, monkeypatch
):
    """The residual the login-time guard alone couldn't close: a bare state
    minted before that guard existed (or reaching the callback through any
    other path than generic_oauth_login) must not be allowed to delete and
    replace an existing app-scoped grant, even though generic_oauth_login
    itself is never consulted here."""
    db, user = db_session

    scoped_state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    scoped_request = SimpleNamespace(
        query_params={"code": "github-code", "state": scoped_state}
    )
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "scoped-github-token",
                    "token_type": "bearer",
                    "scope": "repo,user:email",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"id": 42, "login": "octocat"})),
    )
    setup_response = generic_oauth_callback(
        "github", scoped_request, db, _github_provider()
    )
    assert setup_response.status_code == 200

    # A stale bare state (as if minted before this guard was deployed)
    # reaches the callback directly, bypassing generic_oauth_login entirely.
    bare_state = create_access_token(
        data={"type": "oauth_state", "user_id": user.id, "provider": "github"},
        expires_delta=timedelta(minutes=10),
    )
    bare_request = SimpleNamespace(
        query_params={"code": "bare-code", "state": bare_state}
    )
    bare_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "bare-github-token",
                "token_type": "bearer",
                "scope": "read:user",
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", bare_post)

    bare_response = generic_oauth_callback(
        "github", bare_request, db, _github_provider()
    )

    assert bare_response.status_code == 404
    bare_post.assert_not_called()

    # The original scoped connection and its active server are untouched.
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .one()
    )
    assert oauth_account.access_token == "scoped-github-token"
    server = db.query(MCPServer).filter(MCPServer.name == "GitHub").one()
    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .one()
    )
    assert user_mcp.is_active is True


def test_bounded_oauth_error_message_caps_at_500_chars():
    """Direct unit coverage for _bounded_oauth_error_message's cap -- the
    integration test above pins the end-to-end callback behavior, but
    nothing previously asserted the 500-char limit itself, or that an
    access_token field in the provider's error payload isn't echoed."""
    long_description = "x" * 10_000
    token_data = {
        "error": "invalid_grant",
        "error_description": long_description,
        "access_token": "should-never-be-echoed",
    }

    message = auth_api._bounded_oauth_error_message(token_data)

    assert len(message) <= 500
    assert "should-never-be-echoed" not in message
    assert message.startswith("invalid_grant: " + "x" * 10)


def test_bounded_oauth_error_message_escapes_html():
    token_data = {"error": "<script>alert(1)</script>"}

    message = auth_api._bounded_oauth_error_message(token_data)

    assert "<script>" not in message
    assert "&lt;script&gt;" in message


def test_redact_oauth_log_payload_keeps_only_allowlisted_fields():
    """The server-side log line for a failed token exchange must not put a
    live secret in the log just because the browser-facing message was
    already trimmed -- a malformed/partial response can carry an
    access_token (or client_secret, refresh_token, etc.) alongside an
    error field."""
    token_data = {
        "error": "invalid_grant",
        "error_description": "the code has expired",
        "access_token": "live-secret-token",
        "refresh_token": "live-secret-refresh",
        "client_secret": "live-secret-client",
        "unexpected_provider_field": "also-should-not-appear",
    }

    redacted = auth_api._redact_oauth_log_payload(token_data)

    assert redacted["error"] == "invalid_grant"
    assert redacted["error_description"] == "the code has expired"
    for secret in (
        "live-secret-token",
        "live-secret-refresh",
        "live-secret-client",
        "also-should-not-appear",
    ):
        assert secret not in str(redacted)


def test_redact_oauth_log_payload_recurses_into_a_dict_shaped_allowlisted_field():
    """Meta's "error" is itself an object ({"message": ..., "type": ...}),
    the same shape _bounded_oauth_error_message already handles for the
    browser-facing message -- an allowlisted key with a non-str value must
    still keep its diagnostic content, not be blanked to "<redacted>" just
    because it isn't a plain string."""
    token_data = {
        "error": {"message": "Invalid OAuth access token.", "type": "OAuthException"},
    }

    redacted = auth_api._redact_oauth_log_payload(token_data)

    assert redacted["error"] != "<redacted>"
    assert redacted["error"]["message"] == "Invalid OAuth access token."
    assert redacted["error"]["type"] == "OAuthException"


def test_redact_oauth_log_payload_redacts_a_secret_nested_inside_an_allowlisted_field():
    """A first version of this function serialized an allowlisted key's
    whole value verbatim (e.g. via json.dumps) once it wasn't a plain str
    -- that reopens the exact leak this function exists to close if a
    malformed/adversarial response nests a live secret *inside* an
    allowlisted key's object, e.g. {"error": {"access_token": "..."}}.
    The nested secret must be redacted the same as a top-level one."""
    token_data = {
        "access_token": "top-level-secret",
        "error": {
            "message": "safe to log",
            "access_token": "nested-secret",
        },
    }

    redacted = auth_api._redact_oauth_log_payload(token_data)

    assert redacted["access_token"] == "<redacted>"
    assert redacted["error"]["message"] == "safe to log"
    assert redacted["error"]["access_token"] == "<redacted>"
    for secret in ("top-level-secret", "nested-secret"):
        assert secret not in str(redacted)


def test_github_callback_does_not_log_token_alongside_error(
    db_session, monkeypatch, caplog
):
    """End-to-end: a malformed response carrying an access_token alongside
    an error field must not put that token into the server log via the
    warning added for debuggability."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "bad-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "error": "invalid_grant",
                    "access_token": "live-secret-token",
                }
            )
        ),
    )
    caplog.set_level(logging.WARNING, logger=auth_api.__name__)

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    assert "live-secret-token" not in response.body.decode()
    assert "live-secret-token" not in caplog.text


def test_github_callback_rejects_non_200_userinfo_response(db_session, monkeypatch):
    """A non-200 /user response (401/403/429/5xx) used to be silently
    skipped, leaving provider_user_id/email at None and still persisting
    a "connected" account -- it must fail the callback instead."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "good-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse({"access_token": "some-token"})),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({}, status_code=401)),
    )

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    assert "status 401" in response.body.decode()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .count()
        == 0
    )


def test_github_callback_userinfo_failure_does_not_replace_a_working_grant(
    db_session, monkeypatch
):
    """The concrete risk a fail-open userinfo check creates: reconnecting
    with a since-expired/insufficiently-scoped token whose /user call now
    401s must not delete the still-working prior grant."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="github",
            access_token="working-token",
            provider_user_id="42",
        )
    )
    db.commit()

    request = SimpleNamespace(query_params={"code": "good-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse({"access_token": "new-but-unverifiable"})),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({}, status_code=401)),
    )

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .one()
    )
    assert oauth_account.access_token == "working-token"
    assert oauth_account.provider_user_id == "42"


def test_github_callback_rejects_non_json_userinfo_response(db_session, monkeypatch):
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "good-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse({"access_token": "some-token"})),
    )
    non_json_response = Mock(status_code=200)
    non_json_response.json.side_effect = ValueError("Expecting value")
    monkeypatch.setattr(auth_api.requests, "get", Mock(return_value=non_json_response))

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    assert "could not be parsed" in response.body.decode()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .count()
        == 0
    )


def test_github_callback_rejects_non_object_userinfo_response(db_session, monkeypatch):
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "good-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse({"access_token": "some-token"})),
    )
    list_response = Mock(status_code=200)
    list_response.json.return_value = []
    monkeypatch.setattr(auth_api.requests, "get", Mock(return_value=list_response))

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .count()
        == 0
    )


def test_bounded_oauth_error_message_extracts_metas_nested_error_object():
    """Meta's OAuth error is itself an object, not a bare string -- str()
    on it directly would render a Python dict repr into the page."""
    token_data = {
        "error": {
            "message": "Error validating access token",
            "type": "OAuthException",
            "code": 190,
        }
    }

    message = auth_api._bounded_oauth_error_message(token_data)

    assert message == "Error validating access token"
    assert "OAuthException" not in message
    assert "{" not in message


def test_bounded_oauth_error_message_uses_zooms_reason_field():
    """Zoom's error shape carries the human-readable detail in a `reason`
    key, not the standard OAuth2 `error_description`."""
    token_data = {"error": "invalid_request", "reason": "Invalid Refresh Token"}

    message = auth_api._bounded_oauth_error_message(token_data)

    assert message == "invalid_request: Invalid Refresh Token"


def test_bounded_oauth_error_message_drops_non_string_description():
    """An object-valued error_description (or reason) must not be rendered
    as a Python dict repr -- the same defect class the dict-`error` branch
    prevents, one key over."""
    token_data = {
        "error": "invalid_request",
        "error_description": {"code": 4700, "trace_id": "abc"},
    }

    message = auth_api._bounded_oauth_error_message(token_data)

    assert message == "invalid_request"
    assert "4700" not in message
    assert "{" not in message


def test_github_callback_bounds_token_exchange_error_response(db_session, monkeypatch):
    """GitHub's JSON Accept header makes its JSON error path reachable
    through the generic `"error" in token_data` branch -- that branch must
    not echo the full decoded provider response (which could carry
    unexpected/unbounded fields), only the standard error/error_description
    fields, HTML-escaped."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "bad-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "error": "incorrect_client_credentials",
                    "error_description": (
                        "The client_id and/or client_secret passed are incorrect."
                    ),
                    "unexpected_field": "<script>alert(1)</script>",
                }
            )
        ),
    )

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    body = response.body.decode()
    assert "incorrect_client_credentials" in body
    assert "client_id and/or client_secret" in body
    assert "unexpected_field" not in body
    assert "<script>" not in body


def test_github_callback_rejects_non_json_token_response(db_session, monkeypatch):
    """The Accept-header quirk pushes GitHub toward a JSON body but doesn't
    guarantee one -- a non-JSON response (e.g. a proxy stripping the header)
    must surface the same clean, actionable error every other failure branch
    here gives, not a bare JSONDecodeError from an unguarded .json() call."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "bad-code", "state": state})
    non_json_response = Mock(status_code=200)
    non_json_response.json.side_effect = ValueError("Expecting value")
    monkeypatch.setattr(auth_api.requests, "post", Mock(return_value=non_json_response))

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    body = response.body.decode()
    assert "could not be parsed" in body
    assert "Expecting value" not in body


def test_github_callback_rejects_non_2xx_response_with_access_token_shaped_body(
    db_session, monkeypatch
):
    """A non-2xx response body that happens to carry an access_token field
    (a misbehaving proxy/gateway, a stale cached body) must not be trusted
    as a successful exchange just because it has neither an "error" key
    nor a missing access_token -- the HTTP status itself has to gate
    success, or this would persist a connection off a response GitHub
    never actually returned as successful."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "some-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "should-not-be-persisted"}, status_code=502
            )
        ),
    )

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    assert "status 502" in response.body.decode()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .count()
        == 0
    )


def test_github_callback_rejects_non_object_token_response(db_session, monkeypatch):
    """A JSON-parseable but non-object token response (a bare list/string/
    number) must not reach `"error" in token_data` -- which doesn't raise
    for a list/string, it does a membership/substring check, not a key
    check -- and then `token_data.get("access_token")`, which does raise
    (AttributeError) on anything but a dict. Both would otherwise escape to
    the outer handler as an opaque 500 instead of the same clear 400 every
    other malformed-body case here gives."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "some-code", "state": state})
    list_response = Mock(status_code=200)
    list_response.json.return_value = []
    monkeypatch.setattr(auth_api.requests, "post", Mock(return_value=list_response))

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "github")
        .count()
        == 0
    )


def test_github_callback_does_not_leak_token_on_db_commit_failure(
    db_session, monkeypatch
):
    """A DB error while persisting the OAuth account must not echo the
    just-obtained access_token back to the browser: SQLAlchemy's default
    StatementError.__str__ includes bound parameters, and the outer
    exception handler used to render str(e) -- and therefore the token --
    directly into the 500 response. hide_parameters=True on the engine
    (models/database.py) now hides it there too, but this test forces a
    synthetic error message to pin the handler's own behavior
    independently of that."""
    db, user = db_session
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "github",
            "app_id": "github",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "good-code", "state": state})
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(return_value=MockResponse({"access_token": "secret-token-xyz"})),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"login": "octocat", "id": 1})),
    )

    def failing_commit():
        raise RuntimeError(
            "(psycopg2.errors.UniqueViolation) duplicate key value "
            "[parameters: {'access_token': 'secret-token-xyz'}]"
        )

    monkeypatch.setattr(db, "commit", failing_commit)

    response = generic_oauth_callback("github", request, db, _github_provider())

    assert response.status_code == 500
    body = response.body.decode()
    assert "secret-token-xyz" not in body
    assert "Authentication Failed" in body


def test_non_github_callback_omits_accept_json_header(db_session, monkeypatch):
    """Guard the branch condition: a non-github provider must be unaffected by
    the GitHub-specific Accept header."""
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
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": "gmail",
        },
        expires_delta=timedelta(minutes=10),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    post = Mock(
        return_value=MockResponse(
            {"access_token": "tok", "token_type": "Bearer", "scope": ""}
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"sub": "u1", "email": "alice@gmail.com"})),
    )

    google_provider = SimpleNamespace(
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

    response = generic_oauth_callback("google", request, db, google_provider)

    assert response.status_code == 200
    assert "Accept" not in post.call_args.kwargs["headers"]


@pytest.mark.asyncio
async def test_github_expired_token_refresh_sends_accept_json_header(
    db_session, monkeypatch
):
    """Same form-urlencoded-response quirk as the initial code exchange
    applies to GitHub's refresh_token grant too -- without the Accept header,
    a successful refresh's response.json() call would raise instead of
    parsing the new access token."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
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
                    "token_type": "bearer",
                    "expires_in": 28800,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.refresh_token == "new-refresh"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://github.com/login/oauth/access_token"
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "github-client-id",
        "client_secret": "github-client-secret",
    }


@pytest.mark.asyncio
async def test_github_refresh_falls_back_to_env_credentials(db_session, monkeypatch):
    """The connect path already falls back to GITHUB_CLIENT_ID/SECRET when
    the DB row is blank (e.g. a migration that seeded the row before the
    app's env was fully populated) -- the refresh path used to read only
    the DB row, so connect would work but every refresh would fail this
    credential check instead of using the same fallback."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id="",
            client_secret="",
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    monkeypatch.setenv("GITHUB_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "env-client-secret")

    captured_requests = []

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            captured_requests.append((url, kwargs))
            return MockResponse(
                {"access_token": "new-token", "token_type": "bearer"},
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert len(captured_requests) == 1
    _, kwargs = captured_requests[0]
    assert kwargs["data"]["client_id"] == "env-client-id"
    assert kwargs["data"]["client_secret"] == "env-client-secret"


@pytest.mark.asyncio
async def test_github_refresh_surfaces_failure_instead_of_raising_on_form_body(
    db_session, monkeypatch
):
    """Guard against a regression of the bug this fix addresses: if GitHub
    ever answers form-urlencoded despite the Accept header, response.json()
    must fail cleanly (refresh reported as failed) rather than propagating an
    unhandled exception out of refresh_oauth_token_if_needed."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class FormBodyResponse:
        status_code = 200

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return FormBodyResponse()

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is False
    )
    assert oauth_account.access_token == "old-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ["invalid_client", "unauthorized_client"])
async def test_github_refresh_treats_bad_provider_credentials_as_transient(
    db_session, monkeypatch, error_code
):
    """invalid_client/unauthorized_client describe the OAuthProvider row's
    own client_id/client_secret or grant-type authorization -- an
    admin-fixable config problem, not evidence this user's refresh_token is
    dead -- so they must not raise _OAuthRefreshPermanentlyInvalid (which
    would delete every affected user's connection over one admin typo)."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class BadCredentialsResponse:
        status_code = 401

        def json(self):
            return {"error": error_code}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return BadCredentialsResponse()

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
        is False
    )
    assert oauth_account.access_token == "old-token"


@pytest.mark.asyncio
async def test_github_refresh_deletes_connection_on_200_bad_refresh_token(
    db_session, monkeypatch
):
    """GitHub's classic OAuth Apps token endpoint reports a dead
    refresh_token via a 200 response carrying {"error": "bad_refresh_token"}
    rather than a 4xx -- unlike a generic non-200 failure, this must still
    be classified as permanent (raising _OAuthRefreshPermanentlyInvalid),
    or a genuinely dead GitHub connection would be retried forever instead
    of being cleaned up."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="github",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class DeadRefreshTokenResponse:
        status_code = 200

        def json(self):
            return {
                "error": "bad_refresh_token",
                "error_description": "The refresh token is invalid or expired.",
            }

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return DeadRefreshTokenResponse()

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(tool_config._OAuthRefreshPermanentlyInvalid):
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "github")
    assert oauth_account.access_token == "old-token"


def _add_github_provider_row(db) -> None:
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id=encrypt_value("github-client-id"),
            client_secret=encrypt_value("github-client-secret"),
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    db.commit()


def test_revoke_builtin_oauth_grant_deletes_github_grant_with_decrypted_secret(
    db_session, monkeypatch
):
    """GitHub silently 302s through an existing grant instead of showing its
    consent screen again -- deleting the grant on disconnect is what makes
    the next connect re-prompt. The stored client_id/client_secret are
    encrypted at rest, so this must decrypt them (like generic_oauth_login/
    generic_oauth_callback do for this same row) rather than send the raw
    ciphertext to GitHub's Basic Auth header."""
    db, _user = db_session
    _add_github_provider_row(db)

    delete = Mock(return_value=MockResponse(status_code=204))
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    auth_api.revoke_builtin_oauth_grant(
        db,
        provider="github",
        access_token="live-access-token",
        provider_user_id="42",
    )

    delete.assert_called_once()
    args, kwargs = delete.call_args
    assert args[0] == "https://api.github.com/applications/github-client-id/grant"
    assert kwargs["auth"] == ("github-client-id", "github-client-secret")
    assert kwargs["json"] == {"access_token": "live-access-token"}


def test_revoke_builtin_oauth_grant_ignores_non_github_providers(
    db_session, monkeypatch
):
    """Only GitHub needs this today; every other provider is a no-op and
    must not even query the database, let alone call out to it."""
    db, _user = db_session

    class ExplodingQuery:
        def query(self, *args, **kwargs):
            raise AssertionError("non-github provider must not query the database")

    delete = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    auth_api.revoke_builtin_oauth_grant(
        ExplodingQuery(), provider="slack", access_token="tok", provider_user_id="99"
    )

    delete.assert_not_called()


def test_revoke_builtin_oauth_grant_swallows_network_failure(db_session, monkeypatch):
    """Called only after the local disconnect has already committed -- a
    dead network or an unreachable GitHub must not surface as a failed
    disconnect."""
    db, _user = db_session
    _add_github_provider_row(db)

    monkeypatch.setattr(
        auth_api.requests,
        "delete",
        Mock(side_effect=auth_api.requests.ConnectionError("network down")),
    )

    auth_api.revoke_builtin_oauth_grant(
        db, provider="github", access_token="tok", provider_user_id="42"
    )


def test_revoke_builtin_oauth_grant_treats_404_as_success_without_a_warning(
    db_session, monkeypatch
):
    """GitHub answers 404 when the token or grant is already gone -- a
    normal outcome of a stale disconnect retry, not an error to log loudly
    or raise."""
    db, _user = db_session
    _add_github_provider_row(db)

    delete = Mock(return_value=MockResponse(status_code=404))
    monkeypatch.setattr(auth_api.requests, "delete", delete)
    warning = Mock()
    monkeypatch.setattr(auth_api.logger, "warning", warning)

    auth_api.revoke_builtin_oauth_grant(
        db, provider="github", access_token="tok", provider_user_id="42"
    )

    delete.assert_called_once()
    warning.assert_not_called()


def test_revoke_builtin_oauth_grant_logs_422_as_a_real_failure(db_session, monkeypatch):
    """422 is NOT the same outcome as 404/204: GitHub documents it for a
    malformed request or an abuse/rate-limit block, neither of which means
    the grant is actually gone. Silently bucketing it with "already revoked"
    would hide a real, still-live grant behind a clean-looking disconnect."""
    db, _user = db_session
    _add_github_provider_row(db)

    delete = Mock(return_value=MockResponse(status_code=422))
    monkeypatch.setattr(auth_api.requests, "delete", delete)
    warning = Mock()
    monkeypatch.setattr(auth_api.logger, "warning", warning)

    auth_api.revoke_builtin_oauth_grant(
        db, provider="github", access_token="tok", provider_user_id="42"
    )

    delete.assert_called_once()
    warning.assert_called_once()


def test_revoke_builtin_oauth_grant_is_noop_without_a_provider_row(
    db_session, monkeypatch
):
    """No configured GitHub OAuthProvider row (e.g. the connector was never
    set up) -- nothing to authenticate the revoke call with, so skip it."""
    db, _user = db_session

    delete = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    auth_api.revoke_builtin_oauth_grant(
        db, provider="github", access_token="tok", provider_user_id="42"
    )

    delete.assert_not_called()


def test_revoke_builtin_oauth_grant_skips_when_another_user_shares_the_account(
    db_session, monkeypatch
):
    """The regression this finding is about: GitHub's grant-delete revokes
    every token issued to that GitHub account for this app, not just the
    disconnecting user's own token. A second XAgent user (or actor
    namespace) still connected to the same upstream GitHub account must
    keep working -- so the app-wide DELETE must not fire while that
    reference still exists."""
    db, user = db_session
    _add_github_provider_row(db)
    other_user = User(username="bob", password_hash="x", is_admin=False)
    db.add(other_user)
    db.flush()
    db.add(
        UserOAuth(
            user_id=other_user.id,
            provider="github",
            provider_user_id="42",
            access_token="bobs-still-live-token",
        )
    )
    db.commit()

    delete = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    auth_api.revoke_builtin_oauth_grant(
        db, provider="github", access_token="alices-old-token", provider_user_id="42"
    )

    delete.assert_not_called()


def test_resolve_builtin_oauth_revocation_decrypts_credentials(db_session):
    """An async caller (mcp.py's disconnect endpoints) resolves this record
    while still holding its transaction's locks, before ever touching the
    network -- it must come back decrypted and ready to use, exactly like
    revoke_builtin_oauth_grant's own resolution, carrying provider_user_id
    through unchanged for the later reference check."""
    db, _user = db_session
    _add_github_provider_row(db)

    resolved = auth_api.resolve_builtin_oauth_revocation(
        db,
        provider="github",
        access_token="live-access-token",
        provider_user_id="42",
    )

    assert resolved == auth_api.BuiltinOAuthRevocation(
        access_token="live-access-token",
        client_id="github-client-id",
        client_secret="github-client-secret",
        provider_user_id="42",
    )


def test_resolve_builtin_oauth_revocation_is_none_when_nothing_to_revoke(db_session):
    """Every reason to skip revocation must come back as a plain ``None`` --
    the async caller decides what to do with that, not this function."""
    db, _user = db_session

    assert (
        auth_api.resolve_builtin_oauth_revocation(
            db, provider="slack", access_token="tok", provider_user_id="42"
        )
        is None
    )
    assert (
        auth_api.resolve_builtin_oauth_revocation(
            db, provider="github", access_token="", provider_user_id="42"
        )
        is None
    )
    assert (
        auth_api.resolve_builtin_oauth_revocation(
            db, provider="github", access_token="tok", provider_user_id="42"
        )
        is None
    )  # no OAuthProvider row seeded yet


def test_resolve_builtin_oauth_revocation_warns_on_missing_provider_row(
    db_session, monkeypatch
):
    """Every other skip reason logs (a network failure, an unexpected
    status, a still-referenced sibling); silence here would hide a real
    misconfiguration (the connector was never set up) behind an
    indistinguishable "disconnect worked cleanly" outcome."""
    db, _user = db_session
    warning = Mock()
    monkeypatch.setattr(auth_api.logger, "warning", warning)

    auth_api.resolve_builtin_oauth_revocation(
        db, provider="github", access_token="tok", provider_user_id="42"
    )

    warning.assert_called_once()


def test_resolve_builtin_oauth_revocation_warns_on_unresolvable_secret(
    db_session, monkeypatch
):
    """Same reasoning as the missing-provider-row case, for a
    partially-configured provider row (e.g. only one of
    GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET set)."""
    db, _user = db_session
    db.add(
        OAuthProvider(
            provider_name="github",
            name="GitHub",
            client_id="",
            client_secret="",
            auth_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/github/callback",
            userinfo_url="https://api.github.com/user",
            user_id_path="id",
            email_path="login",
            default_scopes=["read:user"],
        )
    )
    db.commit()
    warning = Mock()
    monkeypatch.setattr(auth_api.logger, "warning", warning)

    result = auth_api.resolve_builtin_oauth_revocation(
        db, provider="github", access_token="tok", provider_user_id="42"
    )

    assert result is None
    warning.assert_called_once()


def test_has_other_builtin_oauth_reference_true_for_a_live_sibling_row(db_session):
    """Any current row -- any user, any owner namespace -- sharing the same
    upstream (provider, provider_user_id) counts as a live reference."""
    db, user = db_session
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="github",
            provider_user_id="42",
            resource_owner_key="actor:github",
            access_token="actor-token",
        )
    )
    db.commit()

    assert (
        auth_api.has_other_builtin_oauth_reference(
            db, provider="github", provider_user_id="42"
        )
        is True
    )


def test_has_other_builtin_oauth_reference_false_without_a_sibling_row(db_session):
    db, _user = db_session

    assert (
        auth_api.has_other_builtin_oauth_reference(
            db, provider="github", provider_user_id="42"
        )
        is False
    )


def test_has_other_builtin_oauth_reference_false_when_id_is_unknown(db_session):
    """No provider_user_id to check against can't prove anything either
    way -- treated as "no known other reference" so a provider without a
    stable upstream id still gets revoked, matching the pre-existing
    behavior from before this reference check was added."""
    db, _user = db_session

    assert (
        auth_api.has_other_builtin_oauth_reference(
            db, provider="github", provider_user_id=None
        )
        is False
    )


def test_revoke_resolved_github_oauth_grant_calls_github_directly(monkeypatch):
    """The database-free half of the split: given an already-resolved
    triple, it must call GitHub with exactly those credentials and never
    touch a database."""
    delete = Mock(return_value=MockResponse(status_code=204))
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    auth_api.revoke_resolved_github_oauth_grant("tok", "cid", "secret")

    delete.assert_called_once()
    args, kwargs = delete.call_args
    assert args[0] == "https://api.github.com/applications/cid/grant"
    assert kwargs["auth"] == ("cid", "secret")
    assert kwargs["json"] == {"access_token": "tok"}


def test_revoke_builtin_oauth_grant_if_unreferenced_skips_with_a_live_sibling(
    db_session, monkeypatch
):
    db, user = db_session
    other_user = User(username="bob", password_hash="x", is_admin=False)
    db.add(other_user)
    db.flush()
    db.add(
        UserOAuth(
            user_id=other_user.id,
            provider="github",
            provider_user_id="42",
            access_token="bobs-still-live-token",
        )
    )
    db.commit()

    delete = Mock()
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    auth_api.revoke_builtin_oauth_grant_if_unreferenced(
        db,
        auth_api.BuiltinOAuthRevocation(
            access_token="tok",
            client_id="cid",
            client_secret="secret",
            provider_user_id="42",
        ),
    )

    delete.assert_not_called()


def test_revoke_builtin_oauth_grant_if_unreferenced_revokes_without_a_sibling(
    db_session, monkeypatch
):
    db, _user = db_session

    delete = Mock(return_value=MockResponse(status_code=204))
    monkeypatch.setattr(auth_api.requests, "delete", delete)

    auth_api.revoke_builtin_oauth_grant_if_unreferenced(
        db,
        auth_api.BuiltinOAuthRevocation(
            access_token="tok",
            client_id="cid",
            client_secret="secret",
            provider_user_id="42",
        ),
    )

    delete.assert_called_once()


def test_revoke_builtin_oauth_grant_if_unreferenced_rolls_back_a_failed_check():
    """The reference check does real database work, unlike the pure network
    call -- unlike revoke_resolved_github_oauth_grant, this function's own
    "Never raises" promise depends on catching that failure itself, and on
    rolling the session back first so a poisoned Postgres transaction
    doesn't leak into whatever the caller does with `db` next."""

    class FakeDB:
        def __init__(self):
            self.rolled_back = False

        def query(self, *args, **kwargs):
            raise RuntimeError("connection dropped")

        def rollback(self):
            self.rolled_back = True

    db = FakeDB()

    # Must not raise.
    auth_api.revoke_builtin_oauth_grant_if_unreferenced(
        db,
        auth_api.BuiltinOAuthRevocation(
            access_token="tok",
            client_id="cid",
            client_secret="secret",
            provider_user_id="42",
        ),
    )

    assert db.rolled_back is True


def test_revoke_resolved_github_oauth_grant_swallows_a_non_request_exception(
    monkeypatch,
):
    """ "Never raises" is a real contract other functions depend on, not
    just a description of the common (requests.RequestException) case --
    any exception from the network call must be swallowed the same way."""
    monkeypatch.setattr(
        auth_api.requests, "delete", Mock(side_effect=RuntimeError("boom"))
    )

    auth_api.revoke_resolved_github_oauth_grant("tok", "cid", "secret")
