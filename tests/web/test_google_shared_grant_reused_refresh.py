"""A shared Google grant may reuse its stored refresh token only if Google
confirms the token covers the scopes the new authorization asked for.

When the user connects another Google app, Google can answer the code exchange
without a new refresh token. The callback used to keep the stored one and write
the new access token's scopes next to it. A refresh then returned the older,
narrower scopes, while the row kept claiming the new ones.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.shared.oauth_browser_fixture import (
    create_access_token,
    generic_oauth_callback,
)
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api import auth as auth_api
from xagent.web.models.database import Base
from xagent.web.models.google_oauth_grant import GoogleOAuthAppAssociation
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import google_oauth_grants as grants_service
from xagent.web.services.google_oauth_grants import (
    GoogleOAuthRefreshTokenRequired,
    google_scopes_confirmed_by_refresh,
    stored_google_refresh_token_for_callback,
    write_google_shared_oauth_callback,
)

SUBJECT = "google-subject"
CLIENT_ID = "google-client-id"
DRIVE = "test-google-drive"
DOCS = "test-google-docs"
ALL_SCOPES = "openid scope.drive scope.docs"


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _NotJson(_Response):
    def __init__(self) -> None:
        super().__init__({})

    def json(self) -> dict[str, Any]:
        raise ValueError("not json")


def _refresh_response(scope: str) -> _Response:
    return _Response(
        {
            "access_token": "refreshed",
            "token_type": "Bearer",
            "scope": scope,
            "expires_in": 3600,
        }
    )


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_GOOGLE_SHARED_OAUTH_GRANTS_ENABLED", "true")
    monkeypatch.delenv("XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED", raising=False)
    engine = create_engine(
        f"sqlite:///{tmp_path / 'google-shared.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    session.add(user)
    session.add_all(
        [
            PublicMCPApp(
                app_id=app_id,
                name=app_id,
                transport="oauth",
                provider_name="google",
                oauth_scopes=[scope],
                is_visible_in_connector=True,
                launch_config={"command": "python", "args": ["-m", app_id]},
            )
            for app_id, scope in ((DRIVE, "scope.drive"), (DOCS, "scope.docs"))
        ]
    )
    session.commit()
    session.refresh(user)
    yield session, user
    session.close()
    engine.dispose()


def _provider() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="google",
        client_id=encrypt_value(CLIENT_ID),
        client_secret=encrypt_value("google-client-secret"),
        auth_url="https://accounts.google.com/o/oauth2/auth",
        token_url="https://oauth2.googleapis.com/token",
        redirect_uri="https://app.example.com/api/auth/google/callback",
        userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
        user_id_path="id",
        email_path="email",
        default_scopes=["openid"],
    )


def _connect_drive(db, user, *, owner: str | None = None) -> UserOAuth:
    """Drive is connected first: one shared credential with its own refresh token."""
    legacy = UserOAuth(
        user_id=int(user.id),
        provider=DRIVE,
        resource_owner_key=owner,
        provider_user_id=SUBJECT,
        access_token="drive-access",
    )
    db.add(legacy)
    db.flush()
    shared, _, _ = write_google_shared_oauth_callback(
        db,
        user_id=int(user.id),
        resource_owner_key=owner,
        oauth_client_id=CLIENT_ID,
        app_id=DRIVE,
        google_subject=SUBJECT,
        access_token="drive-access",
        refresh_token="stored-refresh",
        granted_scopes="openid scope.drive",
        requested_scopes=["openid", "scope.drive"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=legacy,
    )
    db.commit()
    return shared


def _docs_callback(
    db,
    user,
    monkeypatch,
    *,
    refresh_response: Any = None,
    requested: tuple[str, ...] = ("openid", "scope.drive", "scope.docs"),
    granted: str = ALL_SCOPES,
    forced_consent: bool = False,
):
    """Run the callback of a docs authorization whose code exchange carries no
    refresh token. The second token-endpoint call, when there is one, is the
    refresh check."""
    state_data: dict[str, Any] = {
        "type": "oauth_state",
        "user_id": user.id,
        "provider": "google",
        "app_id": DOCS,
        "google_shared_mode": True,
        "google_shared_scopes": list(requested),
    }
    if forced_consent:
        state_data["google_forced_consent"] = True
    state = create_access_token(data=state_data, expires_delta=timedelta(minutes=10))
    responses: list[Any] = [
        _Response(
            {
                "access_token": "docs-access",
                "token_type": "Bearer",
                "scope": granted,
                "expires_in": 3600,
            }
        )
    ]
    if refresh_response is not None:
        responses.append(refresh_response)
    post = Mock(side_effect=responses)
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=_Response({"id": SUBJECT, "email": "alice@example.com"})),
    )
    request = SimpleNamespace(query_params={"code": "code", "state": state})
    response = generic_oauth_callback("google", request, db, _provider())
    return response, post


def _snapshot(db, shared_id: int) -> tuple[str, str]:
    db.expire_all()
    shared = db.get(UserOAuth, shared_id)
    return str(shared.refresh_token), str(shared.scope)


def _asks_for_consent(response) -> bool:
    return response.status_code in (302, 307) and parse_qs(
        urlparse(response.headers["location"]).query
    )["prompt"] == ["consent"]


def test_a_reused_refresh_token_that_does_not_cover_the_new_scopes_asks_for_consent(
    db_session, monkeypatch
):
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)

    response, post = _docs_callback(
        db,
        user,
        monkeypatch,
        # Google answers the stored token's refresh with the old scopes only.
        refresh_response=_refresh_response("openid scope.drive"),
    )

    assert _asks_for_consent(response), response.body
    assert post.call_count == 2
    refresh_call = post.call_args_list[1]
    assert refresh_call.kwargs["data"]["grant_type"] == "refresh_token"
    assert refresh_call.kwargs["data"]["refresh_token"] == "stored-refresh"
    # Nothing was written: the row still says what Google confirmed.
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")
    assert db.query(GoogleOAuthAppAssociation).count() == 1


def test_a_reused_refresh_token_that_covers_the_new_scopes_is_kept(
    db_session, monkeypatch
):
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)

    response, _post = _docs_callback(
        db, user, monkeypatch, refresh_response=_refresh_response(ALL_SCOPES)
    )

    assert response.status_code == 200, response.body
    refresh_token, scope = _snapshot(db, shared_id)
    assert refresh_token == "stored-refresh"
    assert set(scope.split()) == set(ALL_SCOPES.split())
    assert db.query(GoogleOAuthAppAssociation).count() == 2


@pytest.mark.parametrize(
    "refresh_response",
    [
        _Response({"error": "invalid_grant"}, 400),
        _Response({"error": "backend_error"}, 503),
        _Response({"access_token": "refreshed", "token_type": "Bearer"}),
        _NotJson(),
        auth_api.requests.RequestException("timed out"),
    ],
    ids=["revoked", "unavailable", "no-scope", "not-json", "network-error"],
)
def test_a_stored_refresh_token_google_cannot_vouch_for_is_not_reused(
    db_session, monkeypatch, refresh_response
):
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)

    response, _post = _docs_callback(
        db, user, monkeypatch, refresh_response=refresh_response
    )

    assert _asks_for_consent(response), response.body
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")


def test_a_forced_consent_that_still_cannot_be_verified_ends_instead_of_looping(
    db_session, monkeypatch
):
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)

    response, _post = _docs_callback(
        db,
        user,
        monkeypatch,
        refresh_response=_refresh_response("openid scope.drive"),
        forced_consent=True,
    )

    assert response.status_code == 400
    assert b"did not provide offline access" in response.body
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")


def test_a_forced_consent_that_returns_a_refresh_token_needs_no_check(
    db_session, monkeypatch
):
    db, user = db_session
    _connect_drive(db, user)
    state = create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "google",
            "app_id": DOCS,
            "google_shared_mode": True,
            "google_shared_scopes": ["openid", "scope.drive", "scope.docs"],
            "google_forced_consent": True,
        },
        expires_delta=timedelta(minutes=10),
    )
    post = Mock(
        return_value=_Response(
            {
                "access_token": "docs-access",
                "refresh_token": "consented-refresh",
                "token_type": "Bearer",
                "scope": ALL_SCOPES,
                "expires_in": 3600,
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=_Response({"id": SUBJECT, "email": "alice@example.com"})),
    )

    response = generic_oauth_callback(
        "google",
        SimpleNamespace(query_params={"code": "code", "state": state}),
        db,
        _provider(),
    )

    assert response.status_code == 200, response.body
    assert post.call_count == 1


def test_a_request_with_only_identity_scopes_makes_no_refresh_call(
    db_session, monkeypatch
):
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)

    response, post = _docs_callback(
        db,
        user,
        monkeypatch,
        requested=("openid",),
        granted="openid scope.drive",
    )

    assert response.status_code == 200, response.body
    assert post.call_count == 1
    assert _snapshot(db, shared_id)[0] == "stored-refresh"


def test_only_the_scopes_the_refresh_confirmed_are_recorded(db_session, monkeypatch):
    """With include_granted_scopes the exchange can report more than was asked
    for; the stored token is proven only for what its refresh listed."""
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)

    response, _post = _docs_callback(
        db,
        user,
        monkeypatch,
        refresh_response=_refresh_response(ALL_SCOPES),
        granted=f"{ALL_SCOPES} scope.sheets",
    )

    assert response.status_code == 200, response.body
    _refresh, scope = _snapshot(db, shared_id)
    assert set(scope.split()) == set(ALL_SCOPES.split())


def test_the_refresh_check_runs_before_any_callback_lock(db_session, monkeypatch):
    """The refresh is a network call with a timeout; no lock may be held for it."""
    db, user = db_session
    _connect_drive(db, user)
    events: list[str] = []
    original_lock = grants_service.lock_google_shared_accounts_for_app_replacement
    monkeypatch.setattr(
        grants_service,
        "lock_google_shared_accounts_for_app_replacement",
        lambda *args, **kwargs: (events.append("lock"), original_lock(*args, **kwargs))[
            1
        ],
    )
    original_check = auth_api._verify_stored_google_refresh_token
    monkeypatch.setattr(
        auth_api,
        "_verify_stored_google_refresh_token",
        lambda **kwargs: (events.append("refresh"), original_check(**kwargs))[1],
    )

    response, _post = _docs_callback(
        db, user, monkeypatch, refresh_response=_refresh_response(ALL_SCOPES)
    )

    assert response.status_code == 200, response.body
    assert events[0] == "refresh"
    assert "lock" in events


def _docs_writer_arguments(db, user, **overrides: Any) -> dict[str, Any]:
    docs = UserOAuth(
        user_id=int(user.id),
        provider=DOCS,
        provider_user_id=SUBJECT,
        access_token="docs-access",
    )
    db.add(docs)
    db.flush()
    arguments: dict[str, Any] = dict(
        user_id=int(user.id),
        resource_owner_key=None,
        oauth_client_id=CLIENT_ID,
        app_id=DOCS,
        google_subject=SUBJECT,
        access_token="docs-access",
        refresh_token=None,
        granted_scopes=ALL_SCOPES,
        requested_scopes=["openid", "scope.drive", "scope.docs"],
        email="alice@example.com",
        token_type="Bearer",
        expires_at=None,
        legacy_oauth_account=docs,
    )
    arguments.update(overrides)
    return arguments


def test_the_writer_refuses_to_reuse_a_refresh_token_nobody_verified(db_session):
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)
    arguments = _docs_writer_arguments(db, user)

    with pytest.raises(GoogleOAuthRefreshTokenRequired):
        write_google_shared_oauth_callback(db, **arguments)
    db.rollback()
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")

    updated, _legacy, _association = write_google_shared_oauth_callback(
        db, verified_refresh_token="stored-refresh", **arguments
    )
    assert updated.refresh_token == "stored-refresh"


def test_the_writer_refuses_a_refresh_token_other_than_the_verified_one(db_session):
    db, user = db_session
    shared_id = int(_connect_drive(db, user).id)
    arguments = _docs_writer_arguments(db, user)

    with pytest.raises(GoogleOAuthRefreshTokenRequired):
        write_google_shared_oauth_callback(
            db, verified_refresh_token="some-other-refresh", **arguments
        )
    db.rollback()
    assert _snapshot(db, shared_id) == ("stored-refresh", "openid scope.drive")


def test_a_new_refresh_token_needs_no_verification(db_session):
    db, user = db_session
    _connect_drive(db, user)

    updated, _legacy, _association = write_google_shared_oauth_callback(
        db, **_docs_writer_arguments(db, user, refresh_token="new-refresh")
    )

    assert updated.refresh_token == "new-refresh"


def test_the_stored_refresh_token_is_the_one_the_writer_would_reuse(db_session):
    db, user = db_session

    def stored(
        *,
        owner: str | None = None,
        client_id: str = CLIENT_ID,
        legacy: str | None = "legacy-refresh",
    ) -> str | None:
        return stored_google_refresh_token_for_callback(
            db,
            user_id=int(user.id),
            resource_owner_key=owner,
            oauth_client_id=client_id,
            google_subject=SUBJECT,
            legacy_refresh_token=legacy,
        )

    # No shared grant yet: the same-subject legacy row is what a callback reuses.
    assert stored() == "legacy-refresh"
    assert stored(legacy=None) is None

    _connect_drive(db, user)
    # The shared credential wins over the legacy row.
    assert stored() == "stored-refresh"
    # A rotated OAuth client needs a fresh token, so nothing is reusable.
    assert stored(client_id="another-client") is None
    # A shared grant belongs to its resource owner, not to another one.
    assert stored(owner="actor:other", legacy=None) is None


def test_a_bare_google_row_without_a_grant_is_reused(db_session):
    db, user = db_session
    db.add(
        UserOAuth(
            user_id=int(user.id),
            provider="google",
            provider_user_id=SUBJECT,
            access_token="bare-access",
            refresh_token="bare-refresh",
        )
    )
    db.commit()

    assert (
        stored_google_refresh_token_for_callback(
            db,
            user_id=int(user.id),
            resource_owner_key=None,
            oauth_client_id=CLIENT_ID,
            google_subject=SUBJECT,
            legacy_refresh_token=None,
        )
        == "bare-refresh"
    )


def test_an_actor_owned_shared_grant_is_found_only_for_its_owner(db_session):
    db, user = db_session
    _connect_drive(db, user, owner="actor:one")

    def stored(owner: str | None) -> str | None:
        return stored_google_refresh_token_for_callback(
            db,
            user_id=int(user.id),
            resource_owner_key=owner,
            oauth_client_id=CLIENT_ID,
            google_subject=SUBJECT,
            legacy_refresh_token=None,
        )

    assert stored("actor:one") == "stored-refresh"
    assert stored("actor:two") is None
    assert stored(None) is None


@pytest.mark.parametrize(
    ("granted", "confirmed", "expected"),
    [
        ("openid scope.a scope.b", frozenset({"scope.a"}), "openid scope.a"),
        ("openid scope.a", frozenset({"scope.a", "scope.b"}), "openid scope.a"),
        ("openid scope.a scope.b", None, "openid scope.a scope.b"),
        ("openid scope.a", frozenset(), "openid"),
    ],
)
def test_only_confirmed_connector_scopes_are_kept(granted, confirmed, expected):
    assert google_scopes_confirmed_by_refresh(granted, confirmed) == expected
