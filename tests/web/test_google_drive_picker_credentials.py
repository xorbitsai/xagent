"""Google Drive credentials and Picker config for a resource owner's connection.

The ``/api/cloud`` routes read the user's own connection. Trusted in-process
callers can read the credential stored for a resource owner key instead; the
routes' status codes and details must stay exactly as they are.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi import HTTPException, Response
from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport import requests as google_auth_requests
from google.oauth2.credentials import Credentials
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from xagent.web.api import auth as auth_api
from xagent.web.api import cloud_storage
from xagent.web.api.cloud_storage import (
    GOOGLE_TOKEN_URI,
    GoogleDriveCredentialError,
    GoogleDriveCredentialReason,
    _GoogleTokenRequest,
    get_google_credentials,
    get_google_drive_picker_config,
    issue_google_drive_picker_config,
)
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import google_picker

OWNER = "delegated:7:member-3"
OTHER_OWNER = "delegated:7:member-4"
DRIVE = "https://www.googleapis.com/auth/drive"
DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
USERINFO = (
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile"
)
GMAIL = "https://www.googleapis.com/auth/gmail.modify"
DB_CLIENT_ID = "123456789012-db.apps.googleusercontent.com"
ENV_CLIENT_ID = "999999999999-env.apps.googleusercontent.com"

RECONNECT_DETAIL = "Google Drive session expired. Please reconnect."
SCOPE_DETAIL = (
    "This Google Drive connection uses an outdated permission. "
    "Reconnect it before opening Google Drive Picker."
)
PICKER_NOT_CONFIGURED_DETAIL = (
    "Google Drive Picker is not configured. Set the dedicated, "
    "referrer-restricted GOOGLE_PICKER_API_KEY and either "
    "GOOGLE_PICKER_APP_ID or a numeric Google OAuth client_id. "
    "The access token and Picker key are sent to the browser."
)


def _future(minutes: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


class _Store:
    def __init__(self, tmp_path) -> None:
        self.engine = create_engine(f"sqlite:///{tmp_path / 'picker.db'}")
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False
        )
        self.db = self.sessions()
        user = User(username="picker-user", password_hash="hash")
        self.db.add(user)
        self.db.commit()
        self.user_id = int(user.id)

    def add_provider(self, *, client_id: str, client_secret: str) -> None:
        self.db.add(
            OAuthProvider(
                provider_name="google",
                name="Google",
                client_id=client_id,
                client_secret=client_secret,
                auth_url="https://accounts.google.com/o/oauth2/auth",
                token_url="https://oauth2.googleapis.com/token",
            )
        )
        self.db.commit()

    def add_drive(
        self,
        *,
        owner: str | None,
        token: str,
        provider_user_id: str = "google-user",
        scope: str | None = f"{USERINFO} {DRIVE_FILE}",
        refresh_token: str | None = "refresh-token",
        expires_at: datetime | None = None,
    ) -> int:
        row = UserOAuth(
            user_id=self.user_id,
            provider="google-drive",
            resource_owner_key=owner,
            provider_user_id=provider_user_id,
            access_token=token,
            refresh_token=refresh_token,
            scope=scope,
            expires_at=expires_at if expires_at is not None else _future(60),
        )
        self.db.add(row)
        self.db.commit()
        return int(row.id)

    def stored(self, row_id: int) -> tuple[Any, Any, Any]:
        fresh = self.sessions()
        try:
            row = fresh.get(UserOAuth, row_id)
            assert row is not None
            return row.access_token, row.refresh_token, row.expires_at
        finally:
            fresh.close()

    def snapshot(self, row_id: int) -> tuple[Any, ...]:
        fresh = self.sessions()
        try:
            row = fresh.get(UserOAuth, row_id)
            assert row is not None
            return (
                row.access_token,
                row.refresh_token,
                row.expires_at,
                row.scope,
                row.provider_user_id,
                row.resource_owner_key,
            )
        finally:
            fresh.close()

    def close(self) -> None:
        self.db.close()
        self.engine.dispose()


@pytest.fixture
def store(tmp_path, monkeypatch):
    for name in (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_API_KEY",
        "GOOGLE_PICKER_API_KEY",
        "GOOGLE_PICKER_APP_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    created = _Store(tmp_path)
    created.add_provider(client_id=DB_CLIENT_ID, client_secret="db-secret")
    try:
        yield created
    finally:
        created.close()


@pytest.fixture
def picker_key(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_API_KEY", "picker-api-key")


def _refreshing(
    *,
    token: str = "refreshed-token",
    refresh_token: str | None = None,
    error: Exception | None = None,
):
    calls: list[str] = []

    def _refresh(self: Credentials, request: Any) -> None:
        del request
        calls.append(str(self.token))
        if error is not None:
            raise error
        self.token = token
        self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            hours=1
        )
        if refresh_token is not None:
            self._refresh_token = refresh_token

    return patch.object(Credentials, "refresh", _refresh), calls


# --- contract ------------------------------------------------------------


def test_credential_reasons_are_stable() -> None:
    assert set(get_args(GoogleDriveCredentialReason)) == {
        "picker_not_configured",
        "account_not_connected",
        "account_not_found",
        "reauth_required",
        "refresh_unavailable",
        "oauth_unconfigured",
        "scope_full_drive",
        "scope_mismatch",
        "scope_drive_missing",
    }


def test_credential_error_is_the_routes_http_error_plus_a_reason() -> None:
    error = GoogleDriveCredentialError(
        401, RECONNECT_DETAIL, reason="reauth_required", oauth_account_id=5
    )
    assert isinstance(error, HTTPException)
    assert (error.status_code, error.detail) == (401, RECONNECT_DETAIL)
    assert (error.reason, error.oauth_account_id) == ("reauth_required", 5)
    rowless = GoogleDriveCredentialError(
        503, PICKER_NOT_CONFIGURED_DETAIL, reason="picker_not_configured"
    )
    assert rowless.oauth_account_id is None


def test_contract_functions_take_their_options_by_keyword(store, picker_key) -> None:
    row_id = store.add_drive(owner=OWNER, token="owned")

    creds = get_google_credentials(
        user_id=store.user_id,
        db=store.db,
        account_id=row_id,
        resource_owner_key=OWNER,
        min_ttl=timedelta(minutes=5),
    )
    issued = issue_google_drive_picker_config(
        db=store.db,
        user_id=store.user_id,
        resource_owner_key=OWNER,
        account_id=row_id,
        minimal_scopes=True,
        min_ttl=timedelta(minutes=5),
    )

    assert creds.token == issued["access_token"] == "owned"
    with pytest.raises(TypeError):
        get_google_credentials(store.user_id, store.db, row_id, OWNER)
    with pytest.raises(TypeError):
        issue_google_drive_picker_config(store.db, store.user_id)


def _credentials_token(store: "_Store") -> str:
    return get_google_credentials(
        store.user_id, store.db, resource_owner_key=OWNER
    ).token


def _issued_token(store: "_Store") -> str:
    return issue_google_drive_picker_config(
        store.db, user_id=store.user_id, resource_owner_key=OWNER
    )["access_token"]


@pytest.mark.parametrize(
    "read_token", [_credentials_token, _issued_token], ids=["credentials", "issue"]
)
@pytest.mark.parametrize(
    ("minutes_left", "expected"), [(4, "refreshed-token"), (6, "stored")]
)
def test_default_refresh_threshold_is_five_minutes(
    store, picker_key, read_token, minutes_left, expected
) -> None:
    store.add_drive(owner=OWNER, token="stored", expires_at=_future(minutes_left))
    patcher, _calls = _refreshing()

    with patcher:
        assert read_token(store) == expected


# --- owner namespace -----------------------------------------------------


def test_credentials_read_only_the_requested_owner_namespace(store) -> None:
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")
    store.add_drive(owner=OTHER_OWNER, token="other", provider_user_id="other")

    assert get_google_credentials(store.user_id, store.db).token == "ordinary"
    owned = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)
    assert owned.token == "owned"


def test_owner_namespace_cannot_select_an_ordinary_row_by_id(store) -> None:
    ordinary_id = store.add_drive(owner=None, token="ordinary")
    store.add_drive(owner=OWNER, token="owned")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(
            store.user_id, store.db, ordinary_id, resource_owner_key=OWNER
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Selected Google Drive account not found"
    assert exc_info.value.reason == "account_not_found"


def test_missing_owner_connection_is_not_connected(store) -> None:
    store.add_drive(owner=None, token="ordinary")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Google Drive account not connected"
    assert exc_info.value.reason == "account_not_connected"
    assert exc_info.value.oauth_account_id is None


def test_ordinary_lookup_ignores_owner_rows(store) -> None:
    store.add_drive(owner=OWNER, token="owned")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db)

    assert exc_info.value.status_code == 401
    assert exc_info.value.reason == "account_not_connected"


def test_owner_branch_prefers_the_newest_row(store) -> None:
    store.add_drive(owner=OWNER, token="older", provider_user_id="first")
    store.add_drive(owner=OWNER, token="newer", provider_user_id="second")

    creds = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert creds.token == "newer"


def test_owner_branch_rereads_a_row_already_loaded_in_the_session(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="before")
    loaded = store.db.get(UserOAuth, row_id)
    assert loaded is not None and loaded.access_token == "before"
    # Another writer replaced the token after this session loaded the row.
    store.db.execute(
        text("UPDATE user_oauth SET access_token = 'after' WHERE id = :id"),
        {"id": row_id},
    )

    creds = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert creds.token == "after"


def test_owner_token_cleared_row_requires_reauth(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "reauth_required"
    assert exc_info.value.oauth_account_id == row_id


@pytest.mark.parametrize("owner", [None, OWNER])
def test_due_refresh_without_refresh_token_requires_reauth(store, owner) -> None:
    expires_at = _future(1)
    row_id = store.add_drive(
        owner=owner, token="old", refresh_token=None, expires_at=expires_at
    )
    patcher, calls = _refreshing()

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "reauth_required"
    assert exc_info.value.oauth_account_id == row_id
    assert calls == []
    access_token, refresh_token, _expires = store.stored(row_id)
    assert (access_token, refresh_token) == ("old", None)


_INVALID_OWNER_KEYS = pytest.mark.parametrize(
    "key", ["", "   ", "k" * 513, 7], ids=["empty", "blank", "oversized", "not-a-str"]
)
_INVALID_MIN_TTLS = pytest.mark.parametrize(
    "min_ttl", [timedelta(seconds=-1), 300, None], ids=["negative", "int", "none"]
)


def _nothing_may_be_read():
    return (
        patch(
            "xagent.web.api.cloud_storage.scoped_user_oauth_query",
            side_effect=AssertionError("no credential may be read"),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_picker_config",
            side_effect=AssertionError("the configuration may not be checked"),
        ),
    )


@_INVALID_OWNER_KEYS
def test_an_invalid_owner_key_is_rejected_before_anything_is_read(store, key) -> None:
    store.add_drive(owner=OWNER, token="owned")
    no_query, no_config = _nothing_may_be_read()

    with no_query, no_config:
        with pytest.raises(ValueError, match="resource_owner_key"):
            get_google_credentials(store.user_id, store.db, resource_owner_key=key)
        # Not 503 even though the Picker is not configured here.
        with pytest.raises(ValueError, match="resource_owner_key"):
            issue_google_drive_picker_config(
                store.db, user_id=store.user_id, resource_owner_key=key
            )


@_INVALID_MIN_TTLS
def test_an_invalid_min_ttl_is_rejected_before_anything_is_read(store, min_ttl) -> None:
    store.add_drive(owner=OWNER, token="owned")
    no_query, no_config = _nothing_may_be_read()

    with no_query, no_config:
        for owner in (None, OWNER):
            with pytest.raises(ValueError, match="min_ttl"):
                get_google_credentials(
                    store.user_id, store.db, resource_owner_key=owner, min_ttl=min_ttl
                )
            with pytest.raises(ValueError, match="min_ttl"):
                issue_google_drive_picker_config(
                    store.db,
                    user_id=store.user_id,
                    resource_owner_key=owner,
                    min_ttl=min_ttl,
                )


def test_an_owner_key_is_stripped(store) -> None:
    store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    patcher, calls = _refreshing()

    with patcher:
        creds = get_google_credentials(
            store.user_id, store.db, resource_owner_key=f"  {OWNER}\t"
        )

    assert calls == ["old"]
    assert creds.token == "refreshed-token"


def test_a_zero_min_ttl_leaves_the_refresh_to_google_auth(store) -> None:
    # Inside the default five minutes, but outside the few minutes before
    # expiry in which google-auth itself considers a token expired.
    store.add_drive(
        owner=OWNER,
        token="four-and-a-half-minutes",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=4, seconds=30),
    )
    patcher, calls = _refreshing()

    with patcher:
        kept = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER, min_ttl=timedelta(0)
        )
        refreshed = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )

    assert kept.token == "four-and-a-half-minutes"
    assert calls == ["four-and-a-half-minutes"]
    assert refreshed.token == "refreshed-token"


# --- OAuth client resolution -------------------------------------------


def test_owner_branch_resolves_the_oauth_client_per_field(store, monkeypatch) -> None:
    store.db.query(OAuthProvider).update({"client_secret": ""})
    store.db.commit()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", ENV_CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")

    with patch.object(
        auth_api,
        "_resolve_oauth_client_per_field",
        wraps=auth_api._resolve_oauth_client_per_field,
    ) as resolver:
        owned = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )
    ordinary = get_google_credentials(store.user_id, store.db)

    # Per field, with the helper the connector runtime refreshes it with.
    assert (owned.client_id, owned.client_secret) == (DB_CLIENT_ID, "env-secret")
    assert [call.args[0] for call in resolver.call_args_list] == ["google"]
    # The ordinary branch keeps replacing the incomplete pair as a whole.
    assert (ordinary.client_id, ordinary.client_secret) == (
        ENV_CLIENT_ID,
        "env-secret",
    )


def test_owner_branch_needs_the_google_provider_row(store, monkeypatch) -> None:
    store.db.query(OAuthProvider).delete()
    store.db.commit()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", ENV_CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    owned_id = store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        500,
        "Google OAuth configuration missing",
    )
    assert exc_info.value.reason == "oauth_unconfigured"
    assert exc_info.value.oauth_account_id == owned_id
    assert get_google_credentials(store.user_id, store.db).client_id == ENV_CLIENT_ID


# --- refresh -------------------------------------------------------------


def test_refresh_honours_min_ttl(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="ten-minutes", expires_at=_future(10))
    patcher, calls = _refreshing()

    with patcher:
        kept = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)
        refreshed = get_google_credentials(
            store.user_id,
            store.db,
            resource_owner_key=OWNER,
            min_ttl=timedelta(minutes=15),
        )

    assert kept.token == "ten-minutes"
    assert calls == ["ten-minutes"]
    assert refreshed.token == "refreshed-token"
    assert store.stored(row_id)[0] == "refreshed-token"


def test_owner_refresh_persists_only_the_owner_row(store) -> None:
    expired = _future(-5)
    ordinary_id = store.add_drive(
        owner=None, token="ordinary", provider_user_id="ordinary", expires_at=expired
    )
    owned_id = store.add_drive(
        owner=OWNER, token="owned", provider_user_id="owned", expires_at=expired
    )
    ordinary_before = store.stored(ordinary_id)
    patcher, calls = _refreshing(refresh_token="rotated-refresh-token")

    with patcher:
        creds = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )

    assert calls == ["owned"]
    assert creds.token == "refreshed-token"
    access_token, refresh_token, expires_at = store.stored(owned_id)
    assert (access_token, refresh_token) == (
        "refreshed-token",
        "rotated-refresh-token",
    )
    assert expires_at is not None
    assert expires_at.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)
    assert store.stored(ordinary_id) == ordinary_before


def test_refresh_keeps_an_unrotated_refresh_token(store) -> None:
    row_id = store.add_drive(owner=None, token="old", expires_at=_future(-5))
    patcher, _calls = _refreshing()

    with patcher:
        get_google_credentials(store.user_id, store.db)

    assert store.stored(row_id)[:2] == ("refreshed-token", "refresh-token")


_INVALID_GRANT_PAYLOAD = {
    "error": "invalid_grant",
    "error_description": "Token has been expired or revoked.",
}


@pytest.mark.parametrize("owner", [None, OWNER])
@pytest.mark.parametrize(
    "error",
    [
        TransportError("connection reset"),
        RefreshError("temporarily_unavailable", retryable=True),
        # The token endpoint never answered, so the payload is not trusted.
        RefreshError(
            "invalid_grant: Token has been expired or revoked.",
            _INVALID_GRANT_PAYLOAD,
        ),
        RuntimeError("unexpected"),
    ],
    ids=["transport", "retryable", "invalid-grant-without-an-answer", "unexpected"],
)
def test_refresh_failures_without_an_answer_are_unavailable(
    store, owner, error
) -> None:
    expires_at = _future(-5)
    row_id = store.add_drive(owner=owner, token="old", expires_at=expires_at)
    before = store.stored(row_id)
    patcher, calls = _refreshing(error=error)

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert calls == ["old"]
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "refresh_unavailable"
    assert exc_info.value.oauth_account_id == row_id
    assert store.stored(row_id) == before


@pytest.mark.parametrize("owner", [None, OWNER])
def test_failed_commit_after_refresh_rolls_back(store, owner, caplog) -> None:
    row_id = store.add_drive(owner=owner, token="old", expires_at=_future(-5))
    before = store.stored(row_id)
    rollbacks: list[bool] = []
    original_rollback = store.db.rollback

    def _failing_commit() -> None:
        # A database error's text includes the statement's parameters, which
        # here are the refreshed tokens.
        raise OperationalError(
            "UPDATE user_oauth SET access_token=?, refresh_token=? "
            "WHERE user_oauth.id = ?",
            ("refreshed-token", "rotated-refresh-token", row_id),
            Exception("database is locked"),
        )

    def _recording_rollback() -> None:
        rollbacks.append(True)
        original_rollback()

    patcher, _calls = _refreshing(refresh_token="rotated-refresh-token")
    with (
        caplog.at_level(logging.ERROR, logger="xagent.web.api.cloud_storage"),
        patcher,
        patch.object(store.db, "commit", _failing_commit),
        patch.object(store.db, "rollback", _recording_rollback),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert rollbacks == [True]
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "refresh_unavailable"
    assert exc_info.value.oauth_account_id == row_id
    assert store.stored(row_id) == before
    # The error itself carries both new tokens; the log carries neither.
    assert "rotated-refresh-token" in str(exc_info.value.__cause__)
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Failed to store refreshed Google token: OperationalError"]
    for secret in ("refreshed-token", "rotated-refresh-token"):
        assert all(secret not in message for message in messages)


def _replacement_row(
    store: "_Store",
    *,
    replaced_id: int,
    owner: str | None,
    expires_at: datetime,
    scope: str = f"{USERINFO} {DRIVE_FILE}",
) -> int:
    """Replace a stored row the way a reconnect does, from another session.

    Callers keep a later row in the table, so that SQLite does not hand the
    replacement the deleted row's id again.
    """
    other = store.sessions()
    try:
        other.query(UserOAuth).filter(UserOAuth.id == replaced_id).delete()
        replacement = UserOAuth(
            user_id=store.user_id,
            provider="google-drive",
            resource_owner_key=owner,
            provider_user_id="google-user",
            access_token="replacement-token",
            refresh_token="replacement-refresh-token",
            scope=scope,
            expires_at=expires_at,
        )
        other.add(replacement)
        other.commit()
        return int(replacement.id)
    finally:
        other.close()


def test_row_replaced_during_a_website_refresh_is_read_on_retry(store) -> None:
    row_id = store.add_drive(owner=None, token="old", expires_at=_future(-5))
    store.add_drive(owner=OTHER_OWNER, token="other")
    replacement_ids: list[int] = []

    def _replace_row_then_refresh(self: Credentials, request: Any) -> None:
        del request
        replacement_ids.append(
            _replacement_row(
                store, replaced_id=row_id, owner=None, expires_at=_future(60)
            )
        )
        self.token = "refreshed-token"

    with (
        patch.object(Credentials, "refresh", _replace_row_then_refresh),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db)

    assert exc_info.value.status_code == 401
    assert exc_info.value.reason == "refresh_unavailable"
    assert exc_info.value.oauth_account_id == row_id
    # The session was rolled back, so a retry reads the replacement.
    [replacement_id] = replacement_ids
    assert replacement_id != row_id
    retried = get_google_credentials(store.user_id, store.db)
    assert retried.token == "replacement-token"
    assert store.stored(replacement_id)[0] == "replacement-token"


# --- refresh lock for a resource owner's connection -------------------------
#
# The resource-owner branch locks the stored row before it refreshes and reads
# it again under the lock. On SQLite the lock is the database write lock.


def _locking_with(monkeypatch, before_lock) -> None:
    """Run ``before_lock`` each time a refresher is about to take the lock."""
    original = cloud_storage._lock_resource_owner_google_drive_row

    def _lock(db, **kwargs):
        before_lock()
        return original(db, **kwargs)

    monkeypatch.setattr(cloud_storage, "_lock_resource_owner_google_drive_row", _lock)


@pytest.mark.parametrize(
    ("replacement_minutes", "refreshed"),
    [(60, False), (-5, True)],
    ids=["fresh-replacement", "expired-replacement"],
)
def test_owner_refresh_uses_a_row_replaced_while_it_waited(
    store, monkeypatch, replacement_minutes, refreshed
) -> None:
    """The replacement commits just before the lock is taken.

    On SQLite a replacement that commits while the lock statement waits ends
    up the same way: the row is read only once the lock is held, and from
    then on no writer can commit. On PostgreSQL the locking statement reads
    the row itself; a replacement it cannot see is covered by
    ``test_owner_lock_selects_for_update_outside_sqlite``.
    """
    old_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    store.add_drive(owner=OTHER_OWNER, token="other")
    replacement_ids: list[int] = []
    _locking_with(
        monkeypatch,
        lambda: replacement_ids.append(
            _replacement_row(
                store,
                replaced_id=old_id,
                owner=OWNER,
                expires_at=_future(replacement_minutes),
            )
        ),
    )
    patcher, calls = _refreshing(refresh_token="rotated-refresh-token")

    with patcher:
        creds = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )

    [replacement_id] = replacement_ids
    assert replacement_id != old_id
    if refreshed:
        assert calls == ["replacement-token"]
        assert creds.token == "refreshed-token"
        assert store.stored(replacement_id)[:2] == (
            "refreshed-token",
            "rotated-refresh-token",
        )
    else:
        assert calls == []
        assert creds.token == "replacement-token"
        assert store.stored(replacement_id)[:2] == (
            "replacement-token",
            "replacement-refresh-token",
        )
    # The lock was released.
    assert not store.db.in_transaction()


def _delete_row(store: "_Store", row_id: int) -> None:
    """Delete a stored row the way a disconnect does, from another session."""
    other = store.sessions()
    try:
        other.query(UserOAuth).filter(UserOAuth.id == row_id).delete()
        other.commit()
    finally:
        other.close()


@pytest.mark.parametrize(
    ("select_row", "status", "detail", "reason"),
    [
        (False, 401, "Google Drive account not connected", "account_not_connected"),
        (True, 404, "Selected Google Drive account not found", "account_not_found"),
    ],
    ids=["newest-row", "selected-row"],
)
def test_owner_refresh_stops_when_the_row_was_deleted_while_it_waited(
    store, monkeypatch, select_row, status, detail, reason
) -> None:
    row_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    _locking_with(monkeypatch, lambda: _delete_row(store, row_id))
    patcher, calls = _refreshing()

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(
            store.user_id,
            store.db,
            row_id if select_row else None,
            resource_owner_key=OWNER,
        )

    assert calls == []
    assert (exc_info.value.status_code, exc_info.value.detail) == (status, detail)
    assert exc_info.value.reason == reason
    # The row it was about to refresh is gone, so the error names none.
    assert exc_info.value.oauth_account_id is None
    # The lock was released.
    assert not store.db.in_transaction()


def _read_owner_credentials(
    store: "_Store", min_ttl: timedelta = timedelta(minutes=5)
) -> tuple[str, str | None, bool]:
    session = store.sessions()
    try:
        creds = get_google_credentials(
            store.user_id, session, resource_owner_key=OWNER, min_ttl=min_ttl
        )
        return str(creds.token), creds.refresh_token, session.in_transaction()
    finally:
        session.close()


def _start(name: str, target) -> tuple[threading.Thread, dict[str, Any]]:
    outcome: dict[str, Any] = {}

    def _run() -> None:
        try:
            outcome["result"] = target()
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    thread = threading.Thread(target=_run, name=name, daemon=True)
    thread.start()
    return thread, outcome


@pytest.mark.parametrize(
    ("waiter_min_ttl", "waiter_refreshes"),
    [(timedelta(minutes=5), False), (timedelta(hours=2), True)],
    ids=["reuses-the-winners-token", "refreshes-with-the-rotated-token"],
)
def test_a_refresher_waiting_on_the_lock_never_uses_a_stale_refresh_token(
    store, monkeypatch, waiter_min_ttl, waiter_refreshes
) -> None:
    row_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    winner_refreshing = threading.Event()
    waiter_locking = threading.Event()
    refreshes: list[tuple[str, str, str | None]] = []

    def _before_lock() -> None:
        if threading.current_thread().name == "waiter":
            waiter_locking.set()

    def _refresh(self: Credentials, request: Any) -> None:
        del request
        name = threading.current_thread().name
        refreshes.append((name, str(self.token), self.refresh_token))
        if name == "winner":
            winner_refreshing.set()
            # Keep the lock until the other refresher waits for it.
            assert waiter_locking.wait(timeout=5)
            time.sleep(0.2)
        self.token = f"{name}-token"
        self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            hours=1
        )
        self._refresh_token = f"{name}-refresh-token"

    _locking_with(monkeypatch, _before_lock)
    monkeypatch.setattr(Credentials, "refresh", _refresh)

    winner, winner_outcome = _start("winner", lambda: _read_owner_credentials(store))
    assert winner_refreshing.wait(timeout=5)
    # The waiter's first read still sees the expired token.
    waiter, waiter_outcome = _start(
        "waiter", lambda: _read_owner_credentials(store, waiter_min_ttl)
    )
    winner.join(timeout=10)
    waiter.join(timeout=10)

    assert not winner.is_alive() and not waiter.is_alive()
    assert "error" not in winner_outcome, winner_outcome
    assert "error" not in waiter_outcome, waiter_outcome
    assert winner_outcome["result"] == ("winner-token", "winner-refresh-token", False)
    if waiter_refreshes:
        # The waiter refreshed with the token the winner stored.
        assert refreshes == [
            ("winner", "old", "refresh-token"),
            ("waiter", "winner-token", "winner-refresh-token"),
        ]
        assert waiter_outcome["result"] == (
            "waiter-token",
            "waiter-refresh-token",
            False,
        )
        assert store.stored(row_id)[:2] == ("waiter-token", "waiter-refresh-token")
    else:
        assert refreshes == [("winner", "old", "refresh-token")]
        assert waiter_outcome["result"] == (
            "winner-token",
            "winner-refresh-token",
            False,
        )
        assert store.stored(row_id)[:2] == ("winner-token", "winner-refresh-token")


def _quick_writer(store: "_Store", row_id: int) -> None:
    """Touch the row from another connection, giving up after 0.1 seconds."""
    engine = create_engine(store.engine.url, connect_args={"timeout": 0.1})
    try:
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE user_oauth SET scope = scope WHERE id = :id"),
                {"id": row_id},
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize("fails", [False, True], ids=["refreshed", "refresh-failed"])
def test_owner_refresh_holds_the_lock_until_it_is_done(store, fails) -> None:
    row_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    writes_during_refresh: list[str] = []

    def _refresh(self: Credentials, request: Any) -> None:
        del request
        try:
            _quick_writer(store, row_id)
            writes_during_refresh.append("written")
        except OperationalError:
            writes_during_refresh.append("locked")
        if fails:
            raise TransportError("connection reset")
        self.token = "refreshed-token"

    with patch.object(Credentials, "refresh", _refresh):
        if fails:
            with pytest.raises(GoogleDriveCredentialError):
                get_google_credentials(
                    store.user_id, store.db, resource_owner_key=OWNER
                )
        else:
            get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert writes_during_refresh == ["locked"]
    # Released afterwards, also after a failed refresh.
    assert not store.db.in_transaction()
    _quick_writer(store, row_id)


def test_website_refresh_takes_no_lock(store) -> None:
    row_id = store.add_drive(owner=None, token="old", expires_at=_future(-5))
    writes_during_refresh: list[str] = []

    def _refresh(self: Credentials, request: Any) -> None:
        del request
        _quick_writer(store, row_id)
        writes_during_refresh.append("written")
        self.token = "refreshed-token"

    with patch.object(Credentials, "refresh", _refresh):
        get_google_credentials(store.user_id, store.db)

    assert writes_during_refresh == ["written"]


def test_owner_lock_failure_is_unavailable_and_skips_the_refresh(
    store, monkeypatch
) -> None:
    row_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))

    def _locked(db, **kwargs):
        raise OperationalError("UPDATE user_oauth", {}, Exception("locked"))

    monkeypatch.setattr(cloud_storage, "_lock_resource_owner_google_drive_row", _locked)
    patcher, calls = _refreshing()

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert calls == []
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "refresh_unavailable"
    assert exc_info.value.oauth_account_id == row_id
    assert not store.db.in_transaction()


@pytest.mark.parametrize(
    ("reads", "locked_row"),
    [
        (["row"], "row"),
        # In READ COMMITTED, PostgreSQL skips a row deleted while FOR UPDATE
        # waits for it, and the replacement is not in that statement's
        # snapshot: only a second statement finds it.
        ([None, "replacement"], "replacement"),
        ([None, None], None),
    ],
    ids=["found", "replaced-while-it-waited", "deleted"],
)
def test_owner_lock_selects_for_update_outside_sqlite(reads, locked_row) -> None:
    query = MagicMock()
    for method in ("filter", "order_by", "with_for_update", "populate_existing"):
        getattr(query, method).return_value = query
    query.first.side_effect = reads
    db = MagicMock()
    db.get_bind.return_value.dialect.name = "postgresql"

    with patch(
        "xagent.web.api.cloud_storage.scoped_user_oauth_query", return_value=query
    ):
        row = cloud_storage._lock_resource_owner_google_drive_row(
            db, user_id=1, account_id=None, resource_owner_key=OWNER
        )

    assert row == locked_row
    assert query.first.call_count == len(reads)
    query.with_for_update.assert_called_once_with()
    query.populate_existing.assert_called_once_with()
    db.execute.assert_not_called()


# --- refresh through google-auth ------------------------------------------
#
# These tests run google-auth's own refresh and retry code; only the HTTP call
# under its ``requests`` transport is replaced.


@dataclass(frozen=True)
class _Answer:
    status: int = 200
    body: str = ""
    content_type: str = "application/json"
    error: Exception | None = None


def _json_answer(status: int, payload: dict[str, Any]) -> _Answer:
    return _Answer(status, json.dumps(payload))


def _page(status: int, html: str) -> _Answer:
    return _Answer(status, html, content_type="text/html")


def _no_answer(error: Exception) -> _Answer:
    return _Answer(error=error)


class _TokenEndpoint:
    """Stands in for ``requests.Session.request`` during a token refresh.

    It gives its answers in order and then keeps repeating the last one.
    """

    def __init__(self, *answers: _Answer) -> None:
        self.answers = answers or (_Answer(),)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        answer = self.answers[min(len(self.calls), len(self.answers) - 1)]
        self.calls.append(
            {"method": method, "url": url, "timeout": kwargs.get("timeout")}
        )
        if answer.error is not None:
            raise answer.error
        response = requests.Response()
        response.status_code = answer.status
        response._content = answer.body.encode()
        response.headers["Content-Type"] = answer.content_type
        response.url = url
        return response


@pytest.fixture
def backoff_sleeps(monkeypatch) -> list[float]:
    """Record google-auth's backoff between attempts instead of sleeping.

    google-auth waits with ``time.sleep``; only this thread's waits are
    recorded and skipped.
    """
    sleeps: list[float] = []
    real_sleep = time.sleep
    test_thread = threading.get_ident()

    def _sleep(seconds: float) -> None:
        if threading.get_ident() == test_thread:
            sleeps.append(seconds)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", _sleep)
    return sleeps


_GATEWAY_HTML = "<html><body><h1>502 Bad Gateway</h1></body></html>"
_UNAVAILABLE_HTML = "<html><body><h1>503 Service Unavailable</h1></body></html>"
_BACKEND_ERROR = {"error": "backend_error", "error_description": "Unavailable"}
_FRESH_TOKEN = {"access_token": "fresh", "expires_in": 3600, "token_type": "Bearer"}


# (answer, reason, token endpoint calls). google-auth retries only the
# answers it considers retryable, up to three attempts.
_REFRESH_RESPONSES = [
    pytest.param(_page(502, _GATEWAY_HTML), "refresh_unavailable", 1, id="502-html"),
    pytest.param(
        _json_answer(502, {"error": "bad_gateway", "error_description": "Bad Gateway"}),
        "refresh_unavailable",
        1,
        id="502-json",
    ),
    pytest.param(
        _json_answer(502, _INVALID_GRANT_PAYLOAD),
        "refresh_unavailable",
        1,
        id="502-json-invalid-grant",
    ),
    pytest.param(
        _page(503, _UNAVAILABLE_HTML),
        "refresh_unavailable",
        3,
        id="503-html-after-retries",
    ),
    pytest.param(
        _json_answer(503, _BACKEND_ERROR),
        "refresh_unavailable",
        3,
        id="503-json-after-retries",
    ),
    pytest.param(
        _json_answer(503, _INVALID_GRANT_PAYLOAD),
        "refresh_unavailable",
        3,
        id="503-json-invalid-grant-after-retries",
    ),
    pytest.param(
        _page(400, "<html>Bad Request</html>"),
        "refresh_unavailable",
        1,
        id="400-html",
    ),
    pytest.param(
        _json_answer(400, _INVALID_GRANT_PAYLOAD),
        "reauth_required",
        1,
        id="400-invalid-grant",
    ),
    pytest.param(
        # google-auth turns this answer into a "Reauthentication is needed."
        # error without the payload; the recorded answer still says
        # invalid_grant.
        _json_answer(
            400,
            {
                "error": "invalid_grant",
                "error_subtype": "invalid_rapt",
                "error_description": "reauth related error (invalid_rapt)",
            },
        ),
        "reauth_required",
        1,
        id="400-invalid-grant-reauth-subtype",
    ),
    pytest.param(
        _json_answer(
            401,
            {
                "error": "invalid_client",
                "error_description": "The OAuth client was not found.",
            },
        ),
        "refresh_unavailable",
        1,
        id="401-invalid-client",
    ),
    pytest.param(
        _json_answer(
            400, {"error": "unauthorized_client", "error_description": "Unauthorized"}
        ),
        "refresh_unavailable",
        1,
        id="400-unauthorized-client",
    ),
    pytest.param(
        _json_answer(400, {"error": "invalid_scope", "error_description": "Bad"}),
        "refresh_unavailable",
        1,
        id="400-invalid-scope",
    ),
    pytest.param(
        _json_answer(400, {"error": {"code": 400, "message": "Bad Request"}}),
        "refresh_unavailable",
        1,
        id="400-structured-error",
    ),
    pytest.param(
        _no_answer(requests.exceptions.ReadTimeout("Read timed out.")),
        "refresh_unavailable",
        1,
        id="transport-timeout",
    ),
]


@pytest.mark.parametrize("owner", [None, OWNER])
@pytest.mark.parametrize(("answer", "reason", "attempts"), _REFRESH_RESPONSES)
def test_token_endpoint_answers_are_classified(
    store, backoff_sleeps, owner, answer, reason, attempts
) -> None:
    endpoint = _TokenEndpoint(answer)
    row_id = store.add_drive(owner=owner, token="old", expires_at=_future(-5))
    before = store.snapshot(row_id)

    with (
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    # The website routes keep answering exactly as before.
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == reason
    assert exc_info.value.oauth_account_id == row_id
    assert isinstance(exc_info.value.__cause__, (RefreshError, TransportError))
    assert [(call["method"], call["url"]) for call in endpoint.calls] == [
        ("POST", GOOGLE_TOKEN_URI)
    ] * attempts
    assert len(backoff_sleeps) == attempts - 1
    assert store.snapshot(row_id) == before


@pytest.mark.parametrize("owner", [None, OWNER])
@pytest.mark.parametrize(
    "body",
    ["[]", '"invalid_grant"', "null", "[" * 100_000],
    ids=["array", "string", "null", "nested-too-deep"],
)
def test_a_token_endpoint_body_that_is_not_a_json_object_is_unavailable(
    store, backoff_sleeps, caplog, owner, body
) -> None:
    """google-auth itself may fail on such a body; it is still classified."""
    endpoint = _TokenEndpoint(_Answer(400, body))
    row_id = store.add_drive(owner=owner, token="old", expires_at=_future(-5))
    before = store.snapshot(row_id)

    with (
        caplog.at_level(logging.ERROR, logger="xagent.web.api.cloud_storage"),
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "refresh_unavailable"
    assert exc_info.value.oauth_account_id == row_id
    assert exc_info.value.__cause__ is not None
    assert len(endpoint.calls) == 1
    assert backoff_sleeps == []
    [message] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "xagent.web.api.cloud_storage"
    ]
    assert "token endpoint status 400, error None," in message
    assert store.snapshot(row_id) == before


@pytest.mark.parametrize(
    ("body", "error_code"),
    [
        (None, None),
        (b"", None),
        (
            b'{"error": "invalid_grant", "error_subtype": "invalid_rapt"}',
            "invalid_grant",
        ),
        (b'{"error": {"code": 400, "message": "Bad Request"}}', None),
        (b'{"error_description": "no code"}', None),
        (b"<html>Bad Request</html>", None),
        (b"[]", None),
        (b'"invalid_grant"', None),
        (b"null", None),
        (b"[" * 100_000, None),
    ],
    ids=[
        "none",
        "empty",
        "object",
        "structured-error",
        "no-error",
        "html",
        "array",
        "string",
        "null",
        "nested-too-deep",
    ],
)
def test_oauth_error_code_reads_only_a_json_object(body, error_code) -> None:
    assert cloud_storage._oauth_error_code(body) == error_code


# (answers, reason, last status, last error code, token endpoint calls)
_REFRESH_SEQUENCES = [
    pytest.param(
        (_page(503, _UNAVAILABLE_HTML), _json_answer(400, _INVALID_GRANT_PAYLOAD)),
        "reauth_required",
        400,
        "invalid_grant",
        2,
        id="503-then-400-invalid-grant",
    ),
    pytest.param(
        (
            _json_answer(503, _BACKEND_ERROR),
            _json_answer(503, _BACKEND_ERROR),
            _json_answer(400, _INVALID_GRANT_PAYLOAD),
        ),
        "reauth_required",
        400,
        "invalid_grant",
        3,
        id="503-503-then-400-invalid-grant",
    ),
    pytest.param(
        (_json_answer(503, _INVALID_GRANT_PAYLOAD), _page(502, _GATEWAY_HTML)),
        "refresh_unavailable",
        502,
        None,
        2,
        id="503-invalid-grant-then-502-html",
    ),
    pytest.param(
        (
            _page(503, _UNAVAILABLE_HTML),
            _no_answer(requests.exceptions.ReadTimeout("Read timed out.")),
        ),
        "refresh_unavailable",
        None,
        None,
        2,
        id="503-then-timeout",
    ),
    pytest.param(
        # invalid_grant is final: google-auth never asks again.
        (_json_answer(400, _INVALID_GRANT_PAYLOAD), _json_answer(200, _FRESH_TOKEN)),
        "reauth_required",
        400,
        "invalid_grant",
        1,
        id="400-invalid-grant-is-not-retried",
    ),
]


@pytest.mark.parametrize("owner", [None, OWNER])
@pytest.mark.parametrize(
    ("answers", "reason", "last_status", "error_code", "attempts"),
    _REFRESH_SEQUENCES,
)
def test_the_last_token_endpoint_answer_decides(
    store,
    backoff_sleeps,
    caplog,
    owner,
    answers,
    reason,
    last_status,
    error_code,
    attempts,
) -> None:
    endpoint = _TokenEndpoint(*answers)
    row_id = store.add_drive(owner=owner, token="old", expires_at=_future(-5))
    before = store.snapshot(row_id)

    with (
        caplog.at_level(logging.ERROR, logger="xagent.web.api.cloud_storage"),
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == reason
    assert len(endpoint.calls) == attempts
    assert len(backoff_sleeps) == attempts - 1
    [message] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "xagent.web.api.cloud_storage"
    ]
    assert f"token endpoint status {last_status}, error {error_code}," in message
    assert store.snapshot(row_id) == before


def test_refresh_failure_log_has_no_secrets_or_response_body(
    store, backoff_sleeps, caplog
) -> None:
    row_id = store.add_drive(owner=OWNER, token="old-access", expires_at=_future(-5))
    endpoint = _TokenEndpoint(_page(502, _GATEWAY_HTML))

    with (
        caplog.at_level(logging.ERROR, logger="xagent.web.api.cloud_storage"),
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError),
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Failed to refresh Google token (refresh_unavailable): RefreshError, "
        "token endpoint status 502, error None, retryable False"
    ]
    for secret in ("old-access", "refresh-token", "db-secret", "Bad Gateway"):
        assert all(secret not in message for message in messages)
    assert store.snapshot(row_id)[0] == "old-access"


def test_refresh_timeout_is_set_only_for_a_resource_owner(store) -> None:
    owned_id = store.add_drive(
        owner=OWNER, token="old", provider_user_id="owned", expires_at=_future(-5)
    )
    ordinary_id = store.add_drive(
        owner=None, token="old", provider_user_id="ordinary", expires_at=_future(-5)
    )
    endpoint = _TokenEndpoint(_json_answer(200, _FRESH_TOKEN))

    with patch.object(requests.Session, "request", endpoint.request):
        owned = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )
        owner_timeouts = [call["timeout"] for call in endpoint.calls]
        endpoint.calls.clear()
        ordinary = get_google_credentials(store.user_id, store.db)
        website_timeouts = [call["timeout"] for call in endpoint.calls]
        endpoint.calls.clear()
        # What google-auth's own transport sends when nothing sets a timeout.
        google_auth_requests.Request()(GOOGLE_TOKEN_URI, method="POST")
        default_timeouts = [call["timeout"] for call in endpoint.calls]

    # The same timeout as the connector runtime's own refresh.
    assert owner_timeouts == [10.0]
    # The website branch keeps google-auth's default.
    assert website_timeouts == default_timeouts
    assert default_timeouts != [10.0]
    assert (owned.token, ordinary.token) == ("fresh", "fresh")
    assert store.stored(owned_id)[0] == "fresh"
    assert store.stored(ordinary_id)[0] == "fresh"


def test_token_request_timeout_replaces_any_requested_timeout() -> None:
    endpoint = _TokenEndpoint(_json_answer(200, {"access_token": "fresh"}))

    with patch.object(requests.Session, "request", endpoint.request):
        _GoogleTokenRequest()(GOOGLE_TOKEN_URI, method="POST", timeout=3)
        _GoogleTokenRequest(timeout=10.0)(GOOGLE_TOKEN_URI, method="POST", timeout=3)
        _GoogleTokenRequest(timeout=10.0)(GOOGLE_TOKEN_URI, method="POST")

    assert [call["timeout"] for call in endpoint.calls] == [3, 10.0, 10.0]


def test_token_request_records_only_the_last_answer() -> None:
    endpoint = _TokenEndpoint(
        _json_answer(503, _BACKEND_ERROR),
        _no_answer(requests.exceptions.ConnectionError("connection reset")),
    )
    request = _GoogleTokenRequest()

    with patch.object(requests.Session, "request", endpoint.request):
        request(GOOGLE_TOKEN_URI, method="POST")
        assert request.last_status == 503
        assert request.last_body is not None
        assert json.loads(request.last_body) == _BACKEND_ERROR
        with pytest.raises(TransportError):
            request(GOOGLE_TOKEN_URI, method="POST")

    assert (request.last_status, request.last_body) == (None, None)


def test_owner_refresh_timeout_is_classified_as_unavailable(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    before = store.snapshot(row_id)
    endpoint = _TokenEndpoint(
        _no_answer(requests.exceptions.ConnectTimeout("Connection timed out."))
    )

    with (
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert [call["timeout"] for call in endpoint.calls] == [10.0]
    assert exc_info.value.reason == "refresh_unavailable"
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert store.snapshot(row_id) == before


# --- issue_google_drive_picker_config ------------------------------------


def test_issue_picker_config_for_owner_returns_three_fields(store, picker_key) -> None:
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")

    result = issue_google_drive_picker_config(
        store.db,
        user_id=store.user_id,
        resource_owner_key=OWNER,
        minimal_scopes=True,
    )

    assert result == {
        "access_token": "owned",
        "developer_key": "picker-api-key",
        "app_id": "123456789012",
    }


def test_issue_picker_config_derives_app_id_from_the_per_field_client(
    store, picker_key, monkeypatch
) -> None:
    store.db.query(OAuthProvider).update({"client_id": "", "client_secret": "s"})
    store.db.commit()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", ENV_CLIENT_ID)
    store.add_drive(owner=OWNER, token="owned")

    result = issue_google_drive_picker_config(
        store.db, user_id=store.user_id, resource_owner_key=OWNER
    )

    assert result["app_id"] == "999999999999"


def test_issue_picker_config_resolves_the_owner_client_once(store, picker_key) -> None:
    store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    patcher, calls = _refreshing()

    with (
        patcher,
        patch.object(
            auth_api,
            "_resolve_oauth_client_per_field",
            wraps=auth_api._resolve_oauth_client_per_field,
        ) as resolver,
    ):
        result = issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=OWNER
        )

    assert calls == ["old"]
    assert result["access_token"] == "refreshed-token"
    assert [call.args[0] for call in resolver.call_args_list] == ["google"]


def test_issue_picker_config_passes_min_ttl(store, picker_key) -> None:
    store.add_drive(owner=OWNER, token="ten-minutes", expires_at=_future(10))
    patcher, calls = _refreshing()

    with patcher:
        result = issue_google_drive_picker_config(
            store.db,
            user_id=store.user_id,
            resource_owner_key=OWNER,
            minimal_scopes=True,
            min_ttl=timedelta(minutes=15),
        )

    assert calls == ["ten-minutes"]
    assert result["access_token"] == "refreshed-token"


@pytest.mark.parametrize("owner", [None, OWNER])
def test_issue_picker_config_unconfigured_is_503_before_credentials(
    store, owner
) -> None:
    store.add_drive(owner=owner, token="token")

    with (
        patch(
            "xagent.web.api.cloud_storage.scoped_user_oauth_query",
            side_effect=AssertionError("credentials must not be read"),
        ),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=owner
        )

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        503,
        PICKER_NOT_CONFIGURED_DETAIL,
    )
    assert exc_info.value.reason == "picker_not_configured"


@pytest.mark.parametrize(
    ("scope", "minimal", "reason"),
    [
        (f"{USERINFO} {DRIVE}", False, "scope_full_drive"),
        (f"{USERINFO} {DRIVE}", True, "scope_full_drive"),
        (f"{DRIVE_FILE} {DRIVE_READONLY}", True, "scope_full_drive"),
        (
            f"{DRIVE_FILE} https://www.googleapis.com/auth/drive.metadata.readonly",
            False,
            "scope_mismatch",
        ),
        (f"{USERINFO} {DRIVE_FILE} {GMAIL}", True, "scope_mismatch"),
        (USERINFO, False, "scope_drive_missing"),
        (None, True, "scope_drive_missing"),
    ],
    ids=[
        "full-drive",
        "full-drive-minimal",
        "readonly-mixed",
        "mixed-drive",
        "extra-scope-minimal",
        "drive-missing",
        "no-scope",
    ],
)
def test_issue_picker_config_rejects_scopes(
    store, picker_key, scope, minimal, reason
) -> None:
    row_id = store.add_drive(owner=OWNER, token="owned", scope=scope)

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        issue_google_drive_picker_config(
            store.db,
            user_id=store.user_id,
            resource_owner_key=OWNER,
            minimal_scopes=minimal,
        )

    assert (exc_info.value.status_code, exc_info.value.detail) == (409, SCOPE_DETAIL)
    assert exc_info.value.reason == reason
    assert exc_info.value.oauth_account_id == row_id


def test_issue_picker_config_names_the_refreshed_row_on_a_scope_error(
    store, picker_key
) -> None:
    store.add_drive(
        owner=OWNER,
        token="older",
        provider_user_id="first",
        scope=f"{USERINFO} {DRIVE}",
    )
    row_id = store.add_drive(
        owner=OWNER,
        token="old",
        provider_user_id="second",
        scope=f"{USERINFO} {DRIVE}",
        expires_at=_future(-5),
    )
    patcher, calls = _refreshing()

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=OWNER
        )

    assert calls == ["old"]
    assert exc_info.value.reason == "scope_full_drive"
    assert exc_info.value.oauth_account_id == row_id


def test_issue_picker_config_names_the_row_read_under_the_lock_on_a_scope_error(
    store, picker_key, monkeypatch
) -> None:
    full_drive = f"{USERINFO} {DRIVE}"
    old_id = store.add_drive(
        owner=OWNER, token="old", scope=full_drive, expires_at=_future(-5)
    )
    store.add_drive(owner=OTHER_OWNER, token="other")
    replacement_ids: list[int] = []
    _locking_with(
        monkeypatch,
        lambda: replacement_ids.append(
            _replacement_row(
                store,
                replaced_id=old_id,
                owner=OWNER,
                expires_at=_future(60),
                scope=full_drive,
            )
        ),
    )
    patcher, calls = _refreshing()

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=OWNER
        )

    [replacement_id] = replacement_ids
    assert replacement_id != old_id
    # The replacement is fresh, so nothing was refreshed.
    assert calls == []
    assert exc_info.value.reason == "scope_full_drive"
    assert exc_info.value.oauth_account_id == replacement_id


def test_issue_picker_config_allows_extra_scopes_unless_minimal(
    store, picker_key
) -> None:
    store.add_drive(owner=OWNER, token="owned", scope=f"{DRIVE_FILE} {GMAIL}")

    result = issue_google_drive_picker_config(
        store.db, user_id=store.user_id, resource_owner_key=OWNER
    )

    assert result["access_token"] == "owned"


def test_issue_picker_config_reports_reasons_from_credentials(
    store, picker_key
) -> None:
    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=OWNER
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.reason == "account_not_connected"


_PICKER_LOGGER = "xagent.web.services.google_picker"


@pytest.fixture
def app_id_reports(monkeypatch, caplog):
    """Return the app id mismatch warnings logged so far, none reported yet."""
    monkeypatch.setattr(google_picker, "_REPORTED_APP_ID_MISMATCHES", set())
    caplog.set_level(logging.WARNING, logger=_PICKER_LOGGER)
    return lambda: [
        record.getMessage()
        for record in caplog.records
        if record.name == _PICKER_LOGGER
    ]


def test_issue_picker_config_warns_once_about_an_app_id_from_another_project(
    store, picker_key, monkeypatch, app_id_reports
) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_APP_ID", "555555555555")
    store.add_drive(owner=None, token="ordinary")

    first = issue_google_drive_picker_config(store.db, user_id=store.user_id)
    second = issue_google_drive_picker_config(store.db, user_id=store.user_id)

    assert first["app_id"] == second["app_id"] == "555555555555"
    [report] = app_id_reports()
    assert "555555555555" in report
    assert "123456789012" in report


@pytest.mark.parametrize(
    ("app_id", "warned"),
    [("999999999999", False), ("123456789012", True)],
    ids=["matches-the-env-client", "matches-only-the-row"],
)
def test_issue_picker_config_checks_the_app_id_against_the_tokens_client(
    store, picker_key, monkeypatch, app_id_reports, app_id, warned
) -> None:
    # The provider row has a client id but no secret, so the user's own
    # connection uses the environment's client pair.
    store.db.query(OAuthProvider).update({"client_secret": ""})
    store.db.commit()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", ENV_CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    monkeypatch.setenv("GOOGLE_PICKER_APP_ID", app_id)
    store.add_drive(owner=None, token="ordinary")

    result = issue_google_drive_picker_config(store.db, user_id=store.user_id)

    assert result["app_id"] == app_id
    reports = app_id_reports()
    assert len(reports) == (1 if warned else 0)
    if warned:
        assert "999999999999" in reports[0]


@pytest.mark.asyncio
async def test_picker_route_keeps_its_detail_for_a_grant_without_drive(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_API_KEY", "picker-api-key")
    monkeypatch.setenv("GOOGLE_PICKER_APP_ID", "1234567890")

    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=SimpleNamespace(token="access-token", scopes=USERINFO.split()),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=("1234567890-client.apps.googleusercontent.com", "secret"),
        ),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        await get_google_drive_picker_config(
            db=object(),
            user=SimpleNamespace(id=1),
            response=Response(),
        )

    assert (exc_info.value.status_code, exc_info.value.detail) == (409, SCOPE_DETAIL)
    assert exc_info.value.reason == "scope_drive_missing"


@pytest.mark.asyncio
async def test_picker_route_reads_the_users_own_connection(store, picker_key) -> None:
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    response = Response()

    result = await get_google_drive_picker_config(
        account_id=None,
        db=store.db,
        user=SimpleNamespace(id=store.user_id),
        response=response,
    )

    assert result["access_token"] == "ordinary"
    assert response.headers["cache-control"] == "no-store"
