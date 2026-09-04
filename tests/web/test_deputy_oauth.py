from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
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


@pytest.mark.parametrize(
    "raw_endpoint,expected",
    [
        ("acme.au.deputy.com", "https://acme.au.deputy.com"),
        ("https://acme.au.deputy.com", "https://acme.au.deputy.com"),
        # Trailing slash / embedded path must not survive into the value
        # UserOAuth.instance_url stores -- _fetch_deputy_identity and
        # tools/config.py's refresh-URL builder both concatenate this value
        # with a path via plain string formatting, with no re-parsing of
        # their own.
        ("acme.au.deputy.com/", "https://acme.au.deputy.com"),
        ("https://acme.au.deputy.com/", "https://acme.au.deputy.com"),
        # An embedded path is stripped, not rejected -- same reconstruction
        # behavior as deputy.py's own _instance_url(), which discards
        # anything past scheme://host[:port] rather than treating it as
        # invalid.
        ("https://acme.au.deputy.com/evil/path", "https://acme.au.deputy.com"),
        # Non-https must be rejected outright, not silently upgraded --
        # deputy.py's own _instance_url() rejects it at tool-call time, so
        # accepting it here would let a connection show "Connected
        # Successfully" while every deputy_*.py tool call then fails.
        ("http://acme.au.deputy.com", None),
        ("ftp://acme.au.deputy.com", None),
        ("acme.notdeputy.com", None),
        ("evil.com", None),
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),
        # The same equivalence cases deputy.py's own _instance_url() is
        # tested against (tests/web/tools/test_deputy_mcp.py) -- these two
        # functions are meant to accept/reject the same inputs.
        ("https://acme.au.deputy.com:8443", "https://acme.au.deputy.com:8443"),
        ("https://acme.au.deputy.com.", "https://acme.au.deputy.com"),
        (
            "https://user:pw@acme.au.deputy.com/evil/path?x=1",
            "https://acme.au.deputy.com",
        ),
        ("https://acme.au.deputy.com:abc", None),
        # urlparse() itself (not just the .port access) can raise
        # ValueError on an IPv6-literal-like host -- must return None, not
        # let the ValueError escape uncaught.
        ("https://[::1].deputy.com", None),
    ],
)
def test_normalize_deputy_endpoint(raw_endpoint, expected):
    assert auth_api._normalize_deputy_endpoint(raw_endpoint) == expected


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
            app_id="deputy",
            name="Deputy",
            description=(
                "Connect to Deputy to look up employees, view rosters/shifts, "
                "and read timesheets."
            ),
            transport="oauth",
            provider_name="deputy",
            category="Scheduling",
            oauth_scopes=["longlife_refresh_token"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.deputy"],
                "env_mapping": {
                    "DEPUTY_ACCESS_TOKEN": "access_token",
                    "DEPUTY_INSTANCE_URL": "instance_url",
                },
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _deputy_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="deputy",
        client_id=encrypt_value("deputy-client-id"),
        client_secret=encrypt_value("deputy-client-secret"),
        auth_url="https://once.deputy.com/my/oauth/login",
        token_url="https://once.deputy.com/my/oauth/access_token",
        redirect_uri="https://app.example.com/api/auth/deputy/callback",
        userinfo_url="",
        user_id_path="",
        email_path="",
        default_scopes=["longlife_refresh_token"],
    )


def _callback_request(db, user, provider: str = "deputy") -> SimpleNamespace:
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": provider,
            "app_id": provider,
        },
        expires_delta=timedelta(minutes=10),
    )
    return SimpleNamespace(query_params={"code": "deputy-code", "state": state})


def test_callback_persists_normalized_endpoint_scope_and_identity(
    db_session, monkeypatch
):
    """A successful callback must: send scope=longlife_refresh_token in the
    token-exchange body, normalize the bare-host `endpoint` field into a
    full origin stored as UserOAuth.instance_url, and populate
    provider_user_id/email from the dedicated GET .../api/v1/me identity
    fetch."""
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "deputy-token",
                "refresh_token": "deputy-refresh",
                "token_type": "Bearer",
                "endpoint": "acme.au.deputy.com",
            }
        )
    )
    mock_get = Mock(
        return_value=MockResponse(
            {"Id": 42, "Email": "alice@acme.example", "FirstName": "Alice"}
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)
    monkeypatch.setattr(auth_api.requests, "get", mock_get)

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 200
    assert mock_post.call_args.kwargs["data"]["scope"] == "longlife_refresh_token"

    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "deputy")
        .one()
    )
    assert oauth_account.access_token == "deputy-token"
    assert oauth_account.refresh_token == "deputy-refresh"
    assert oauth_account.instance_url == "https://acme.au.deputy.com"
    assert oauth_account.provider_user_id == "42"
    assert oauth_account.email == "alice@acme.example"

    mock_get.assert_called_once()
    assert mock_get.call_args.args[0] == "https://acme.au.deputy.com/api/v1/me"
    assert (
        mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer deputy-token"
    )


def test_callback_scope_filters_empty_entries_in_default_scopes(
    db_session, monkeypatch
):
    """A stray empty or whitespace-only entry in the provider row's
    default_scopes (e.g. from an admin edit via PUT /admin/mcp/providers
    with no element-level validation) must not produce a scope value with
    a leading/trailing/doubled space -- filtered out, not just joined
    as-is."""
    db, user = db_session
    provider = _deputy_provider()
    provider.default_scopes = ["longlife_refresh_token", "", "   "]
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "deputy-token",
                "refresh_token": "deputy-refresh",
                "token_type": "Bearer",
                "endpoint": "acme.au.deputy.com",
            }
        )
    )
    mock_get = Mock(return_value=MockResponse({"Id": 42}))
    monkeypatch.setattr(auth_api.requests, "post", mock_post)
    monkeypatch.setattr(auth_api.requests, "get", mock_get)

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, provider
    )

    assert response.status_code == 200
    assert mock_post.call_args.kwargs["data"]["scope"] == "longlife_refresh_token"

    server = db.query(MCPServer).filter(MCPServer.name == "Deputy").one()
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


def test_callback_reports_error_in_token_data_for_deputy(db_session, monkeypatch):
    """The generic `"error" in token_data` guard must still fire correctly
    for Deputy despite the Deputy-specific `data["scope"] = ...` code
    injected into the outbound request just before this POST -- that
    injection must not interfere with how the (unrelated) response is
    handled."""
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {"error": "invalid_client", "error_description": "bad credentials"}
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "deputy")
        .first()
        is None
    )


def test_callback_rejects_non_2xx_status_for_deputy(db_session, monkeypatch):
    """A non-2xx response with no explicit "error" key (e.g. a proxy/
    gateway artifact) must not be trusted as a successful Deputy
    exchange."""
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {"access_token": "deputy-token", "endpoint": "acme.au.deputy.com"},
            status_code=502,
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "deputy")
        .first()
        is None
    )


@pytest.mark.parametrize(
    "token_data",
    [
        {"access_token": "deputy-token", "token_type": "Bearer"},
        {
            "access_token": "deputy-token",
            "token_type": "Bearer",
            "endpoint": 12345,
        },
        {
            "access_token": "deputy-token",
            "token_type": "Bearer",
            "endpoint": "",
        },
        {
            "access_token": "deputy-token",
            "token_type": "Bearer",
            "endpoint": "   ",
        },
    ],
    ids=["missing", "non-string", "empty-string", "whitespace-only"],
)
def test_callback_rejects_missing_endpoint_without_touching_prior_grant(
    db_session, monkeypatch, token_data
):
    """A token response with a missing/non-string/blank endpoint must be
    rejected before the delete-then-recreate persistence step runs --
    letting it through would destroy any prior *working* grant while still
    reporting success, since instance_url is required for the connector to
    launch at all (launch_config.env_mapping)."""
    db, user = db_session
    existing = UserOAuth(
        user_id=user.id,
        provider="deputy",
        access_token="old-working-token",
        instance_url="https://old.deputy.com",
    )
    db.add(existing)
    db.commit()

    mock_post = Mock(return_value=MockResponse(token_data))
    monkeypatch.setattr(auth_api.requests, "post", mock_post)
    monkeypatch.setattr(auth_api.requests, "get", Mock())

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 400
    assert "endpoint" in response.body.decode()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "deputy")
        .one()
    )
    assert oauth_account.access_token == "old-working-token"
    assert oauth_account.instance_url == "https://old.deputy.com"


def test_callback_identity_fetch_non_200_returns_400(db_session, monkeypatch):
    db, user = db_session
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "deputy-token",
                    "endpoint": "acme.au.deputy.com",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse(status_code=500, text="server error")),
    )

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 400
    assert "The provider reported" in response.body.decode()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "deputy")
        .first()
        is None
    )


def test_callback_identity_fetch_network_error_returns_400(db_session, monkeypatch):
    db, user = db_session
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "deputy-token",
                    "endpoint": "acme.au.deputy.com",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(side_effect=requests.exceptions.ConnectionError("connection refused")),
    )

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 400
    assert "Could not reach Deputy" in response.body.decode()


def test_callback_succeeds_when_identity_fields_are_absent(db_session, monkeypatch):
    """A deliberately lenient design: a 200 response from GET /api/v1/me
    with no recognizable id/email key must still succeed the connect,
    leaving provider_user_id/email as None rather than failing the whole
    connect."""
    db, user = db_session
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "deputy-token",
                    "endpoint": "acme.au.deputy.com",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"SomeOtherField": "value"})),
    )

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "deputy")
        .one()
    )
    assert oauth_account.provider_user_id is None
    assert oauth_account.email is None


def test_callback_identity_fetch_falls_back_through_email_keys(db_session, monkeypatch):
    """Email, then CompanyEmail, then DeputyEmail -- the first non-empty
    string match wins."""
    db, user = db_session
    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {
                    "access_token": "deputy-token",
                    "endpoint": "acme.au.deputy.com",
                }
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"Id": 7, "CompanyEmail": "alice@company.example"}
            )
        ),
    )

    response = generic_oauth_callback(
        "deputy", _callback_request(db, user), db, _deputy_provider()
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "deputy")
        .one()
    )
    assert oauth_account.email == "alice@company.example"


@pytest.mark.asyncio
async def test_deputy_refresh_posts_to_stored_instance_url_with_scope_and_redirect(
    db_session, monkeypatch
):
    """Deputy's refresh must go to the per-install host stored on the
    connection (`{instance_url}/oauth/access_token`), never the generic
    once.deputy.com token_url on the provider row, and must include
    scope=longlife_refresh_token and redirect_uri in the body."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="deputy",
            name="Deputy",
            client_id=encrypt_value("deputy-client-id"),
            client_secret=encrypt_value("deputy-client-secret"),
            auth_url="https://once.deputy.com/my/oauth/login",
            token_url="https://once.deputy.com/my/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/deputy/callback",
            userinfo_url="",
            user_id_path="",
            email_path="",
            default_scopes=["longlife_refresh_token"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="deputy",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url="https://acme.au.deputy.com",
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
                {"access_token": "new-token", "token_type": "Bearer"},
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "deputy")
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://acme.au.deputy.com/oauth/access_token"
    assert kwargs["data"]["scope"] == "longlife_refresh_token"
    assert kwargs["data"]["redirect_uri"] == (
        "https://app.example.com/api/auth/deputy/callback"
    )
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "old-refresh"
    # Deputy relies on httpx's default form-urlencoded Content-Type for a
    # `data=` body (no `json=` body, no explicit Content-Type/Accept
    # override, unlike jira/linkedin/github's branches in the same
    # function) -- locks in that Deputy takes neither of those branches.
    assert "json" not in kwargs
    assert kwargs["headers"] == {}
    # Pins that Employment Hero's query-param refresh rewrite (grant_type/
    # refresh_token moved out of the body into `params`) stays scoped to
    # that family and never leaks onto an unrelated provider's refresh.
    assert "params" not in kwargs


@pytest.mark.asyncio
async def test_deputy_refresh_scope_filters_empty_entries_in_default_scopes(
    db_session, monkeypatch
):
    """See the matching callback-side test's docstring -- same reasoning
    applies to the refresh leg's scope-building."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="deputy",
            name="Deputy",
            client_id=encrypt_value("deputy-client-id"),
            client_secret=encrypt_value("deputy-client-secret"),
            auth_url="https://once.deputy.com/my/oauth/login",
            token_url="https://once.deputy.com/my/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/deputy/callback",
            userinfo_url="",
            user_id_path="",
            email_path="",
            default_scopes=["longlife_refresh_token", "", "   "],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="deputy",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url="https://acme.au.deputy.com",
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
                {"access_token": "new-token", "token_type": "Bearer"},
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "deputy")
        is True
    )

    assert captured_requests[0][1]["data"]["scope"] == "longlife_refresh_token"


@pytest.mark.asyncio
async def test_deputy_refresh_without_instance_url_raises_permanent_and_skips_network(
    db_session, monkeypatch
):
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="deputy",
            name="Deputy",
            client_id=encrypt_value("deputy-client-id"),
            client_secret=encrypt_value("deputy-client-secret"),
            auth_url="https://once.deputy.com/my/oauth/login",
            token_url="https://once.deputy.com/my/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/deputy/callback",
            userinfo_url="",
            user_id_path="",
            email_path="",
            default_scopes=["longlife_refresh_token"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="deputy",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url=None,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class ExplodingAsyncClient:
        def __call__(self, *args, **kwargs):
            raise AssertionError(
                "refresh_oauth_token_if_needed must not open an HTTP client "
                "when instance_url is missing"
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", ExplodingAsyncClient())

    # No instance_url on this account is a per-account data problem that no
    # retry can fix, so it's signalled as permanent (see
    # _OAuthRefreshPermanentlyInvalid), not returned as a plain False.
    with pytest.raises(tool_config._OAuthRefreshPermanentlyInvalid):
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "deputy")
    assert oauth_account.access_token == "old-token"


@pytest.mark.asyncio
async def test_deputy_refresh_updates_instance_url_from_new_endpoint(
    db_session, monkeypatch
):
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="deputy",
            name="Deputy",
            client_id=encrypt_value("deputy-client-id"),
            client_secret=encrypt_value("deputy-client-secret"),
            auth_url="https://once.deputy.com/my/oauth/login",
            token_url="https://once.deputy.com/my/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/deputy/callback",
            userinfo_url="",
            user_id_path="",
            email_path="",
            default_scopes=["longlife_refresh_token"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="deputy",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url="https://acme.au.deputy.com",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return MockResponse(
                {
                    "access_token": "new-token",
                    "token_type": "Bearer",
                    "endpoint": "acme2.au.deputy.com",
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "deputy")
        is True
    )
    assert oauth_account.instance_url == "https://acme2.au.deputy.com"


@pytest.mark.asyncio
async def test_deputy_refresh_keeps_prior_instance_url_when_endpoint_malformed(
    db_session, monkeypatch, caplog
):
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="deputy",
            name="Deputy",
            client_id=encrypt_value("deputy-client-id"),
            client_secret=encrypt_value("deputy-client-secret"),
            auth_url="https://once.deputy.com/my/oauth/login",
            token_url="https://once.deputy.com/my/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/deputy/callback",
            userinfo_url="",
            user_id_path="",
            email_path="",
            default_scopes=["longlife_refresh_token"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="deputy",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url="https://acme.au.deputy.com",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return MockResponse(
                {
                    "access_token": "new-token",
                    "token_type": "Bearer",
                    "endpoint": 12345,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    with caplog.at_level("WARNING"):
        assert (
            await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "deputy")
            is True
        )

    assert oauth_account.access_token == "new-token"
    assert oauth_account.instance_url == "https://acme.au.deputy.com"
    assert any("malformed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_deputy_refresh_returns_false_on_non_200_response(
    db_session, monkeypatch
):
    """Mirrors the equivalent coverage for other providers in this file
    (e.g. Meta) -- a non-200 refresh response must leave the stored token
    untouched and report failure, not raise or silently treat it as
    success."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="deputy",
            name="Deputy",
            client_id=encrypt_value("deputy-client-id"),
            client_secret=encrypt_value("deputy-client-secret"),
            auth_url="https://once.deputy.com/my/oauth/login",
            token_url="https://once.deputy.com/my/oauth/access_token",
            redirect_uri="https://app.example.com/api/auth/deputy/callback",
            userinfo_url="",
            user_id_path="",
            email_path="",
            default_scopes=["longlife_refresh_token"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="deputy",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url="https://acme.au.deputy.com",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="42",
    )
    db.add(oauth_account)
    db.commit()

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            return MockResponse(status_code=401, text="invalid_grant")

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "deputy")
        is False
    )
    assert oauth_account.access_token == "old-token"


@pytest.mark.asyncio
async def test_non_deputy_refresh_still_posts_to_provider_token_url(
    db_session, monkeypatch
):
    """Regression check: the refresh_token_url variable this connector
    introduced is shared by every provider's refresh call site -- a
    non-Deputy provider must still post to provider_config.token_url
    unchanged."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="google",
            name="Google",
            client_id=encrypt_value("google-client-id"),
            client_secret=encrypt_value("google-client-secret"),
            auth_url="https://accounts.google.com/o/oauth2/auth",
            token_url="https://oauth2.googleapis.com/token",
            redirect_uri="https://app.example.com/api/auth/google/callback",
            userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
            user_id_path="sub",
            email_path="email",
            default_scopes=["https://www.googleapis.com/auth/gmail.modify"],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="google",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="google-user-1",
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
                    "expires_in": 3600,
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "google")
        is True
    )

    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://oauth2.googleapis.com/token"
    assert "redirect_uri" not in kwargs["data"]
    assert "scope" not in kwargs["data"]
