from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

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
    "raw_business_id,expected",
    [
        (
            "11111111-2222-3333-4444-555555555555",
            "11111111-2222-3333-4444-555555555555",
        ),
        # Case-insensitivity: MYOB's own GUIDs are typically uppercase.
        (
            "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
            "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
        ),
        # Surrounding whitespace is stripped before validating, unlike
        # require_clean_identifier's stricter rejection elsewhere -- this
        # value comes straight off the raw callback query string, not a
        # caller-typed id, so tolerating incidental whitespace here is safe.
        (
            "  11111111-2222-3333-4444-555555555555  ",
            "11111111-2222-3333-4444-555555555555",
        ),
        ("not-a-guid", None),
        ("11111111-2222-3333-4444-55555555555", None),  # one hex digit short
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),
    ],
)
def test_normalize_myob_business_id(raw_business_id, expected):
    assert auth_api._normalize_myob_business_id(raw_business_id) == expected


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
            app_id="myob",
            name="MYOB",
            description="Connect to MYOB AccountRight.",
            transport="oauth",
            provider_name="myob",
            category="Operations",
            oauth_scopes=["sme-company-file", "sme-contacts-customer"],
            is_visible_in_connector=True,
            launch_config={
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.myob"],
                "env_mapping": {
                    "MYOB_ACCESS_TOKEN": "access_token",
                    "MYOB_BUSINESS_ID": "instance_url",
                },
                "static_env": {"MYOB_API_KEY": "MYOB_CLIENT_ID"},
            },
        )
    )
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


def _myob_provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="myob",
        client_id=encrypt_value("myob-client-id"),
        client_secret=encrypt_value("myob-client-secret"),
        auth_url="https://secure.myob.com/oauth2/account/authorize/",
        token_url="https://secure.myob.com/oauth2/v1/authorize/",
        redirect_uri="https://app.example.com/api/auth/myob/callback",
        userinfo_url="",
        user_id_path="",
        email_path="",
        default_scopes=[],
    )


def _token_for(user: User) -> str:
    return create_access_token(
        data={"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=5),
    )


def _callback_request(
    db, user, provider: str = "myob", *, business_id: str | None = None
):
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": provider,
            "app_id": provider,
        },
        expires_delta=timedelta(minutes=10),
    )
    query_params = {"code": "myob-code", "state": state}
    if business_id is not None:
        query_params["businessId"] = business_id
    return SimpleNamespace(query_params=query_params)


BUSINESS_ID = "11111111-2222-3333-4444-555555555555"


def test_login_sets_prompt_consent_for_myob(db_session):
    """Without prompt=consent, MYOB never appends businessId to the
    authorization redirect at all -- the callback's businessId guard would
    then always trigger, so this must hold for every MYOB login, not just
    the resource-owner flow. Uses app_id="myob" (the only reachable path,
    now that bare app_id-less logins are rejected -- see
    test_bare_myob_login_is_rejected_once_authenticated_and_configured
    below), matching how the catalog UI actually connects this app."""
    db, user = db_session
    token = _token_for(user)

    provider = SimpleNamespace(
        client_id=encrypt_value("myob-client-id"),
        auth_url="https://secure.myob.com/oauth2/account/authorize/",
        redirect_uri="https://app.example.com/api/auth/myob/callback",
        default_scopes=[],
    )

    resp = generic_oauth_login(
        provider="myob",
        token=token,
        app_id="myob",
        redirect=None,
        db=db,
        db_provider=provider,
    )
    url = resp.headers["location"]
    qs = parse_qs(urlparse(url).query)

    assert qs.get("prompt") == ["consent"], f"myob prompt missing: {url}"


def test_bare_myob_login_is_rejected_once_authenticated_and_configured(db_session):
    """MYOB is in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT (see mcp_apps.py):
    its oauth_providers row has no default_scopes at all, so a bare
    (app_id-less) login would request zero sme-* scopes yet still complete
    and report success -- this guard must reject it before any state token
    is minted, the same way it already does for github."""
    db, user = db_session
    token = _token_for(user)

    provider = SimpleNamespace(
        client_id=encrypt_value("myob-client-id"),
        auth_url="https://secure.myob.com/oauth2/account/authorize/",
        redirect_uri="https://app.example.com/api/auth/myob/callback",
        default_scopes=[],
    )

    resp = generic_oauth_login(
        provider="myob",
        token=token,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=provider,
    )

    assert resp.status_code == 404


def test_callback_persists_business_id_and_identity(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "myob-token",
                "refresh_token": "myob-refresh",
                "token_type": "bearer",
                # MYOB's documented response returns this as a JSON *string*
                # (e.g. "1200"), not a number -- pinned here, not just the
                # more forgiving int the fixture used to carry, so a
                # regression back to an unguarded int(...)/timedelta(...)
                # cast would be caught the same way it would against the
                # real API.
                "expires_in": "3600",
                "user": {"uid": "42", "username": "alice@acme.example"},
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 200
    # MYOB's provider row carries no default_scopes of its own -- every
    # sme-* scope sent on the exchange must come from the app's own
    # oauth_scopes, matching what MYOB's own docs require on this leg.
    # get_app_by_id resolves a builtin app_id's oauth_scopes from the
    # builtin_mcp_registry.py definition, not whatever the db_session
    # fixture's PublicMCPApp row happens to carry (a builtin app's DB row
    # is a catalog/display shell; the registry is the source of truth for
    # its actual scopes) -- so this is the real production scope list,
    # sorted per _merge_oauth_scopes' app-scope ordering.
    assert mock_post.call_args.kwargs["data"]["scope"] == (
        "sme-company-file sme-contacts-customer sme-contacts-supplier "
        "sme-general-ledger sme-purchases sme-sales"
    )
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .one()
    )
    assert oauth_account.access_token == "myob-token"
    assert oauth_account.refresh_token == "myob-refresh"
    assert oauth_account.instance_url == BUSINESS_ID
    assert oauth_account.provider_user_id == "42"
    assert oauth_account.email == "alice@acme.example"
    assert oauth_account.expires_at is not None


def test_callback_rejects_missing_business_id_without_touching_prior_grant(
    db_session, monkeypatch
):
    """A callback with no businessId (prompt=consent missing/ignored) must
    be rejected before the delete-then-recreate persistence step -- letting
    it through would destroy a prior *working* grant while still reporting
    success, since instance_url is required for the connector to launch at
    all (launch_config.env_mapping). Rejected before the token exchange even
    starts (mock_post is never called): myob_business_id is already fully
    known from the raw callback request, so there's nothing to gain by
    burning a network round trip and the single-use authorization code on an
    outcome that's already decided."""
    db, user = db_session
    existing = UserOAuth(
        user_id=user.id,
        provider="myob",
        access_token="old-working-token",
        instance_url="99999999-8888-7777-6666-555555555555",
    )
    db.add(existing)
    db.commit()

    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "myob-token",
                "user": {"uid": "42", "username": "alice@acme.example"},
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob", _callback_request(db, user, business_id=None), db, _myob_provider()
    )

    assert response.status_code == 400
    assert "businessId" in response.body.decode()
    mock_post.assert_not_called()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .one()
    )
    assert oauth_account.access_token == "old-working-token"
    assert oauth_account.instance_url == "99999999-8888-7777-6666-555555555555"


def test_callback_rejects_malformed_business_id(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(return_value=MockResponse({"access_token": "myob-token"}))
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id="not-a-guid"),
        db,
        _myob_provider(),
    )

    assert response.status_code == 400
    mock_post.assert_not_called()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .first()
        is None
    )


def test_callback_reports_error_in_token_data_for_myob(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {"error": "invalid_grant", "error_description": "bad code"}
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .first()
        is None
    )


def test_callback_rejects_non_2xx_status_for_myob(db_session, monkeypatch):
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {"access_token": "myob-token"},
            status_code=502,
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 400
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .first()
        is None
    )


def test_callback_succeeds_when_identity_fields_are_absent(db_session, monkeypatch):
    """A deliberately lenient design, matching every other identity branch
    in this file: a token response with no recognizable `user` shape must
    still succeed the connect, leaving provider_user_id/email as None
    rather than failing the whole connect."""
    db, user = db_session
    mock_post = Mock(return_value=MockResponse({"access_token": "myob-token"}))
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .one()
    )
    assert oauth_account.provider_user_id is None
    assert oauth_account.email is None


def test_callback_rejects_boolean_uid_instead_of_stringifying_it(
    db_session, monkeypatch
):
    """`bool` is a subclass of `int` in Python, so `isinstance(uid, (str,
    int))` alone would let a boolean `uid` through and stringify it to the
    nonsensical "True"/"False" -- must be treated the same as no usable uid
    at all (None), not silently accepted."""
    db, user = db_session
    mock_post = Mock(
        return_value=MockResponse(
            {
                "access_token": "myob-token",
                "user": {"uid": True, "username": "alice@acme.example"},
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", mock_post)

    response = generic_oauth_callback(
        "myob",
        _callback_request(db, user, business_id=BUSINESS_ID),
        db,
        _myob_provider(),
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "myob")
        .one()
    )
    assert oauth_account.provider_user_id is None
    assert oauth_account.email == "alice@acme.example"


@pytest.mark.asyncio
async def test_myob_refresh_handles_string_expires_in(db_session, monkeypatch):
    """MYOB's documented refresh response returns expires_in as a JSON
    *string* (e.g. "1200"), not a number -- timedelta()'s seconds kwarg
    raises TypeError on a bare string, and refresh_oauth_token_if_needed's
    outer except swallows that as a silent False, which is exactly what
    would otherwise make every MYOB connection stop refreshing ~20 minutes
    after connect with no visible error."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="myob",
            name="MYOB",
            client_id=encrypt_value("myob-client-id"),
            client_secret=encrypt_value("myob-client-secret"),
            auth_url="https://secure.myob.com/oauth2/account/authorize/",
            token_url="https://secure.myob.com/oauth2/v1/authorize/",
            redirect_uri="https://app.example.com/api/auth/myob/callback",
            userinfo_url="",
            user_id_path="",
            email_path="",
            default_scopes=[],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="myob",
        access_token="old-token",
        refresh_token="old-refresh",
        instance_url=BUSINESS_ID,
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
                    "refresh_token": "new-refresh",
                    "token_type": "bearer",
                    "expires_in": "1200",
                },
                status_code=200,
            )

    monkeypatch.setattr(tool_config.httpx, "AsyncClient", FakeAsyncClient)

    assert (
        await tool_config.refresh_oauth_token_if_needed(db, oauth_account, "myob")
        is True
    )
    assert oauth_account.access_token == "new-token"
    assert oauth_account.expires_at > datetime.now(timezone.utc) + timedelta(minutes=15)
