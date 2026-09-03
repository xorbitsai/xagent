from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
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


def test_callback_exchanges_code_and_backfills_identity_from_organisations(
    db_session, monkeypatch
):
    """Employment Hero has no flat userinfo endpoint this callback's generic
    lookup can use (see the provider row's comment) -- the callback must not
    attempt that generic GET, but it must still call GET /organisations to
    derive a provider_user_id (see _fetch_employment_hero_identity's
    docstring for why a permanently-NULL provider_user_id would otherwise
    reopen the same duplicate-UserOAuth-row race Salesforce's token-id
    fallback exists to close)."""
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
    get = Mock(
        return_value=MockResponse(
            {"data": {"items": [{"id": "org-2"}, {"id": "org-1"}]}}
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", get)

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    get.assert_called_once()
    assert (
        get.call_args.args[0] == "https://api.employmenthero.com/api/v1/organisations"
    )
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer eh-token"

    params = post.call_args.kwargs["params"]
    assert params == {
        "grant_type": "authorization_code",
        "redirect_uri": "https://app.example.com/api/auth/employment-hero/callback",
    }
    data = post.call_args.kwargs["data"]
    assert "grant_type" not in data
    assert "redirect_uri" not in data
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
    assert oauth_account.provider_user_id == json.dumps(["org-1", "org-2"])
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


def test_callback_persists_null_identity_when_no_organisations_accessible(
    db_session, monkeypatch
):
    """A token scoped to zero organisations is an edge case no tool in this
    connector could do anything useful with -- the connect must still
    succeed with a NULL provider_user_id rather than erroring."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"data": {"items": []}})),
    )

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .one()
    )
    assert oauth_account.provider_user_id is None


def test_callback_backfills_identity_across_multiple_organisation_pages(
    db_session, monkeypatch
):
    """A grant with more organisations than fit on one page must have every
    page's ids folded into provider_user_id, not just the first -- otherwise
    ids past page 1 are silently excluded from the uniqueness key."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )

    # Short numeric ids, not "org-N", and few enough of them (51, not 101)
    # that the JSON-encoded set stays under _PROVIDER_USER_ID_SAFE_LENGTH --
    # this test checks pagination completeness, not the separate
    # length-safety hashing covered by
    # test_callback_hashes_provider_user_id_when_too_long_for_index.
    page_1_items = [{"id": str(i)} for i in range(50)]
    page_2_items = [{"id": "50"}]
    responses = [
        MockResponse({"data": {"items": page_1_items, "total_pages": 2}}),
        MockResponse({"data": {"items": page_2_items, "total_pages": 2}}),
    ]
    get = Mock(side_effect=responses)
    monkeypatch.setattr(auth_api.requests, "get", get)

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    assert get.call_count == 2
    assert get.call_args_list[0].kwargs["params"]["page_index"] == 1
    assert get.call_args_list[1].kwargs["params"]["page_index"] == 2
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .one()
    )
    expected = json.dumps(sorted(str(i) for i in range(51)))
    assert len(expected) <= auth_api._PROVIDER_USER_ID_SAFE_LENGTH
    assert oauth_account.provider_user_id == expected


def test_callback_keeps_paginating_when_server_returns_short_pages(
    db_session, monkeypatch
):
    """The Employment Hero API's documented max item_per_page (100) is a
    ceiling on what a caller may request, not a guarantee the server always
    returns that many when more results exist -- termination must be driven
    by the response envelope's own total_pages, not by comparing the
    returned page size against what this function itself requested. A
    server that clamps its effective page size below 100 while still
    reporting more pages via total_pages must not have the fetch stop
    early."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )

    # Each page returns only 20 items (a server-side clamp below the 100
    # this function requests), but total_pages says there are 3 pages --
    # the old "len(items) < item_per_page" heuristic would have stopped
    # after page 1, since 20 < 100.
    responses = [
        MockResponse(
            {
                "data": {
                    "items": [{"id": f"p1-{i}"} for i in range(5)],
                    "total_pages": 3,
                }
            }
        ),
        MockResponse(
            {
                "data": {
                    "items": [{"id": f"p2-{i}"} for i in range(5)],
                    "total_pages": 3,
                }
            }
        ),
        MockResponse(
            {
                "data": {
                    "items": [{"id": f"p3-{i}"} for i in range(5)],
                    "total_pages": 3,
                }
            }
        ),
    ]
    get = Mock(side_effect=responses)
    monkeypatch.setattr(auth_api.requests, "get", get)

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    assert get.call_count == 3
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .one()
    )
    expected_ids = [f"p{page}-{i}" for page in (1, 2, 3) for i in range(5)]
    expected = json.dumps(sorted(expected_ids))
    assert len(expected) <= auth_api._PROVIDER_USER_ID_SAFE_LENGTH
    assert oauth_account.provider_user_id == expected


def test_callback_paginates_across_multiple_full_pages_without_total_pages(
    db_session, monkeypatch
):
    """When a response never carries total_pages at all, the fallback
    short-page heuristic must still correctly paginate across MULTIPLE full
    pages before it finally sees a short page -- not just stop-after-one or
    stop-after-two by coincidence."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )

    # No total_pages anywhere in any response -- every page must fall back
    # to the short-page heuristic. Pages 1 and 2 are full (100 items, not
    # < item_per_page), page 3 is short (5 items), so the fetch must run
    # exactly 3 times, not stop after page 1 or 2.
    responses = [
        MockResponse({"data": {"items": [{"id": f"p1-{i}"} for i in range(100)]}}),
        MockResponse({"data": {"items": [{"id": f"p2-{i}"} for i in range(100)]}}),
        MockResponse({"data": {"items": [{"id": f"p3-{i}"} for i in range(5)]}}),
    ]
    get = Mock(side_effect=responses)
    monkeypatch.setattr(auth_api.requests, "get", get)

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    assert get.call_count == 3


def test_callback_hashes_provider_user_id_when_too_long_for_index(
    db_session, monkeypatch
):
    """provider_user_id participates in a real unique Postgres btree index
    (~2704-byte row-size limit). UUID-shaped organisation ids at real scale
    (a partner/bookkeeper account with many accessible organisations) can
    exceed it -- past a safe length, the joined id set must be hashed to a
    fixed-size, still-deterministic value instead of stored raw."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )
    # UUID-shaped ids, enough of them to exceed _PROVIDER_USER_ID_SAFE_LENGTH.
    org_ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(20)]
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(
            return_value=MockResponse(
                {"data": {"items": [{"id": oid} for oid in org_ids], "total_pages": 1}}
            )
        ),
    )

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .one()
    )
    joined = json.dumps(sorted(org_ids))
    assert len(joined) > auth_api._PROVIDER_USER_ID_SAFE_LENGTH
    assert oauth_account.provider_user_id == hashlib.sha256(joined.encode()).hexdigest()


def test_callback_warns_and_uses_partial_set_when_pagination_hits_page_cap(
    db_session, monkeypatch, caplog
):
    """A response that never reports total_pages reached and never shrinks
    below a full page must not hang the callback forever -- it stops at
    _EMPLOYMENT_HERO_IDENTITY_MAX_PAGES, logs a warning (an incomplete
    organisation set is otherwise indistinguishable from a complete one),
    and still completes the connect with whatever it collected."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )

    def _full_page(*args, **kwargs):
        page_index = kwargs["params"]["page_index"]
        return MockResponse(
            {"data": {"items": [{"id": f"p{page_index}-{i}"} for i in range(100)]}}
        )

    monkeypatch.setattr(auth_api.requests, "get", Mock(side_effect=_full_page))

    with caplog.at_level("WARNING"):
        response = generic_oauth_callback(
            "employment-hero", request, db, _employment_hero_provider()
        )

    assert response.status_code == 200
    assert any("safety cap" in record.message for record in caplog.records), (
        "expected a warning when the pagination page cap is hit"
    )
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .one()
    )
    assert oauth_account.provider_user_id is not None


def test_callback_reports_error_when_organisations_lookup_returns_non_json(
    db_session, monkeypatch
):
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )
    bad_response = Mock(status_code=200, text="<html>not json</html>")
    bad_response.json.side_effect = ValueError("no JSON object could be decoded")
    monkeypatch.setattr(auth_api.requests, "get", Mock(return_value=bad_response))

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 400
    assert "non-JSON response" in response.body.decode()


def test_callback_reports_error_when_organisations_body_shape_is_unexpected(
    db_session, monkeypatch
):
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"data": {"items": "not-a-list"}})),
    )

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 400
    assert "unexpected response body" in response.body.decode()


def test_callback_reports_error_when_organisations_lookup_fails(
    db_session, monkeypatch
):
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse(status_code=401, text="invalid token")),
    )

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 400
    assert "provider reported" in response.body.decode()
    assert (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .count()
        == 0
    )


def test_callback_reports_error_when_organisations_lookup_unreachable(
    db_session, monkeypatch
):
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    monkeypatch.setattr(
        auth_api.requests,
        "post",
        Mock(
            return_value=MockResponse(
                {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
            )
        ),
    )
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(side_effect=ConnectionError("timed out")),
    )

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 400
    assert "Could not reach Employment Hero" in response.body.decode()


def test_callback_skips_identity_fetch_when_token_url_is_not_employment_hero(
    db_session, monkeypatch
):
    """An "employment-hero"-family provider row's token_url is admin-
    editable (matches_provider_family matches by name alone, by design, so
    an admin-created "employment-hero-sandbox" row is expected) -- a row
    whose token_url doesn't actually belong to Employment Hero must not
    have this callback forward its access token to the real
    api.employmenthero.com. The connect must still succeed, just with no
    identity (the same NULL-provider_user_id path a provider with no
    userinfo_url and no family match takes)."""
    db, user = db_session
    request = _callback_request(user, code_verifier="verifier-value")

    post = Mock(
        return_value=MockResponse(
            {"access_token": "eh-token", "token_type": "bearer", "expires_in": 900}
        )
    )
    get = Mock()
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(auth_api.requests, "get", get)

    provider = _employment_hero_provider()
    provider.token_url = "https://attacker.example.com/oauth2/token"

    response = generic_oauth_callback("employment-hero", request, db, provider)

    assert response.status_code == 200
    get.assert_not_called()
    oauth_account = (
        db.query(UserOAuth)
        .filter(UserOAuth.user_id == user.id, UserOAuth.provider == "employment-hero")
        .one()
    )
    assert oauth_account.provider_user_id is None


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
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=MockResponse({"data": {"items": []}})),
    )

    response = generic_oauth_callback(
        "employment-hero", request, db, _employment_hero_provider()
    )

    assert response.status_code == 200
    assert "code_verifier" not in post.call_args.kwargs["data"]


@pytest.mark.asyncio
async def test_employment_hero_refresh_sends_grant_type_and_refresh_token_as_query(
    db_session, monkeypatch
):
    """Employment Hero's partner guide requires grant_type and refresh_token
    as query parameters on the refresh request, with only client_id/
    client_secret in the form body -- unlike every other provider's refresh,
    which sends the full RFC 6749 shape (everything in the body)."""
    db, user = db_session
    db.add(
        OAuthProvider(
            provider_name="employment-hero",
            name="Employment Hero",
            client_id=encrypt_value("eh-client-id"),
            client_secret=encrypt_value("eh-client-secret"),
            auth_url="https://oauth.employmenthero.com/oauth2/authorize",
            token_url="https://oauth.employmenthero.com/oauth2/token",
            redirect_uri="https://app.example.com/api/auth/employment-hero/callback",
            userinfo_url="",
            user_id_path="id",
            email_path="email",
            default_scopes=[],
        )
    )
    oauth_account = UserOAuth(
        user_id=user.id,
        provider="employment-hero",
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        provider_user_id="org-1",
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
        await tool_config.refresh_oauth_token_if_needed(
            db, oauth_account, "employment-hero"
        )
        is True
    )

    assert oauth_account.access_token == "new-token"
    assert len(captured_requests) == 1
    url, kwargs = captured_requests[0]
    assert url == "https://oauth.employmenthero.com/oauth2/token"
    assert kwargs["params"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
    }
    assert "grant_type" not in kwargs["data"]
    assert "refresh_token" not in kwargs["data"]
    assert kwargs["data"]["client_id"] == "eh-client-id"
    assert kwargs["data"]["client_secret"] == "eh-client-secret"
