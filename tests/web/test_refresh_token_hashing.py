"""users.refresh_token holds a digest of the refresh JWT, never the JWT."""

from __future__ import annotations

import hashlib

import pytest

from tests.web import test_auth_api as auth_tests
from tests.web.test_auth_api import test_db as test_db
from xagent.web.api import auth
from xagent.web.auth_sessions import (
    hash_refresh_token,
    refresh_token_lookup_values,
    refresh_token_matches,
)
from xagent.web.models.user import User

pytestmark = pytest.mark.usefixtures("test_db")

PASSWORD = "old-password-123"


def _login() -> dict:
    auth_tests.setup_first_admin(password=PASSWORD)
    return auth._login_in_worker(
        auth.LoginRequest(username="administrator", password=PASSWORD),
        auth_tests.TestingSessionLocal,
    )


def _stored() -> str | None:
    with auth_tests.TestingSessionLocal() as session:
        return session.query(User).one().refresh_token


def _refresh(token: str) -> auth.RefreshTokenResponse:
    return auth._refresh_in_worker(
        auth.RefreshTokenRequest(refresh_token=token),
        auth.verify_refresh_token(token)["user_id"],
        auth_tests.TestingSessionLocal,
    )


def test_hash_is_sha256_hex():
    assert hash_refresh_token("a.b.c") == hashlib.sha256(b"a.b.c").hexdigest()


def test_login_stores_only_the_digest():
    tokens = _login()

    stored = _stored()
    assert stored == hash_refresh_token(tokens["refresh_token"])
    assert stored != tokens["refresh_token"]
    assert "." not in stored


def test_refresh_rotates_and_stores_only_the_new_digest():
    tokens = _login()

    refreshed = _refresh(tokens["refresh_token"])

    assert refreshed.refresh_token != tokens["refresh_token"]
    assert _stored() == hash_refresh_token(refreshed.refresh_token)
    with pytest.raises(auth.HTTPException) as stale:
        _refresh(tokens["refresh_token"])
    assert stale.value.status_code == 401
    # The rotated token is still redeemable.
    assert _refresh(refreshed.refresh_token).success is True


def test_the_stored_digest_cannot_be_redeemed_as_a_refresh_token():
    _login()

    response = auth_tests.client.post(
        "/api/auth/refresh", json={"refresh_token": _stored()}
    )

    assert response.status_code == 401


def test_a_plaintext_row_left_by_an_older_release_is_still_redeemed_and_converted():
    tokens = _login()
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).one()
        user.refresh_token = tokens["refresh_token"]
        session.commit()

    refreshed = _refresh(tokens["refresh_token"])

    assert _stored() == hash_refresh_token(refreshed.refresh_token)


def test_logout_clears_the_digest_for_the_matching_token_only():
    tokens = _login()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    client = auth_tests.client
    digest = hash_refresh_token(tokens["refresh_token"])

    stale = client.post(
        "/api/auth/logout", json={"refresh_token": "some.other.token"}, headers=headers
    )
    assert stale.status_code == 200
    assert _stored() == digest

    # The digest itself is not a credential for revoking the session.
    as_digest = client.post(
        "/api/auth/logout", json={"refresh_token": digest}, headers=headers
    )
    assert as_digest.status_code == 200
    assert _stored() == digest

    matching = client.post(
        "/api/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
        headers=headers,
    )
    assert matching.status_code == 200
    assert _stored() is None


def test_matching_helpers():
    token = "header.payload.signature"
    digest = hash_refresh_token(token)

    assert refresh_token_matches(digest, token)
    assert not refresh_token_matches(digest, digest)
    assert not refresh_token_matches(digest, "other.token.value")
    assert refresh_token_matches(token, token)  # legacy plaintext row
    assert not refresh_token_matches(None, token)
    assert not refresh_token_matches("", token)
    assert refresh_token_lookup_values(token) == [digest, token]
