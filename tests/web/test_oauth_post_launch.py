"""The public connector launch transports credentials only in HTTP headers."""

import hashlib
import secrets
from datetime import timedelta
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xagent.web.api import auth
from xagent.web.auth_sessions import user_session_claims
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User


@pytest.fixture
def launch(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'launch.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(username="alice", password_hash="hash", session_epoch="generation")
        provider = OAuthProvider(
            provider_name="custom",
            name="Custom",
            client_id="client",
            client_secret="secret",
            auth_url="https://provider.example/authorize",
            token_url="https://provider.example/token",
        )
        db.add_all([user, provider])
        db.commit()
        token = auth.create_access_token(user_session_claims(user))
        app = FastAPI()
        app.include_router(auth.auth_router)
        app.dependency_overrides[auth.get_db] = lambda: db
        with TestClient(app) as client:
            yield client, db, user, provider, token
    engine.dispose()


def _post(client, credential, **options):
    return client.post(
        "/api/auth/custom/login",
        headers={"Authorization": f"Bearer {credential}"},
        json=options,
    )


def test_launch_preserves_browser_binding_and_session_without_disclosing_access_jwt(
    launch,
):
    client, _, user, _, token = launch
    response = _post(client, token, redirect="https://frontend.example/tools")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert token not in response.text
    assert "set-cookie" not in response.headers
    launch_url = response.json()["authorization_url"]
    ticket = parse_qs(urlsplit(launch_url).query)["ticket"][0]
    ticket_payload = auth.verify_token(ticket)
    assert ticket_payload["type"] == "oauth_launch"
    assert ticket_payload["provider"] == "custom"
    response = client.get(launch_url, follow_redirects=False)
    assert response.status_code == 307
    url = urlsplit(response.headers["location"])
    assert url.netloc == "provider.example"
    params = parse_qs(url.query)
    assert "token" not in params
    state = auth.verify_token(params["state"][0])
    assert state["type"] == "oauth_state"
    assert state["user_id"] == user.id
    assert state["session_epoch"] == "generation"
    assert state["redirect"] == "https://frontend.example/tools"
    cookie_name = auth._ordinary_oauth_cookie_name(state["browser_nonce_hash"])
    nonce = response.cookies[cookie_name]
    assert hashlib.sha256(nonce.encode()).hexdigest() == state["browser_nonce_hash"]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    assert token not in response.headers["set-cookie"]


def test_each_launch_keeps_a_distinct_browser_cookie(launch):
    client, _, _, _, token = launch
    first = client.get(
        _post(client, token).json()["authorization_url"], follow_redirects=False
    )
    second = client.get(
        _post(client, token).json()["authorization_url"], follow_redirects=False
    )
    assert set(first.cookies) != set(second.cookies)
    assert set(first.cookies).issubset(set(client.cookies))
    assert set(second.cookies).issubset(set(client.cookies))


@pytest.mark.parametrize("kind", ["refresh", "oauth_state", "file_stream"])
def test_non_access_tokens_cannot_start_authorization(launch, kind):
    client, _, user, _, _ = launch
    token = auth.create_access_token({**user_session_claims(user), "type": kind})
    assert _post(client, token).status_code == 401


def test_launch_requires_bearer_not_query_or_cookie_auth(launch):
    client, _, _, _, token = launch
    assert client.post("/api/auth/custom/login", json={}).status_code in {401, 403}
    assert client.post(
        f"/api/auth/custom/login?token={token}", json={}
    ).status_code in {401, 403}
    assert client.post(
        "/api/auth/custom/login", cookies={"token": token}, json={}
    ).status_code in {401, 403}


def test_expired_and_revoked_credentials_cannot_launch(launch):
    client, db, user, _, token = launch
    expired = auth.create_access_token(
        user_session_claims(user), expires_delta=timedelta(seconds=-1)
    )
    assert _post(client, expired).status_code == 401
    user.session_epoch = "rotated"
    db.commit()
    assert _post(client, token).status_code == 401


@pytest.mark.parametrize("field", ["token", "access_token"])
def test_even_authenticated_posts_reject_credentials_in_query(launch, field):
    client, _, _, _, token = launch
    response = client.post(
        f"/api/auth/custom/login?{field}=secret",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert response.status_code == 400
    assert "secret" not in response.text
    assert "set-cookie" not in response.headers


def test_body_does_not_accept_a_login_credential(launch):
    client, _, _, _, token = launch
    assert _post(client, token, token=token).status_code == 422


@pytest.mark.parametrize("field", ["token", "access_token"])
def test_deprecated_get_never_resolves_or_redirects_a_query_token(
    launch, monkeypatch, field
):
    client, _, _, _, token = launch
    resolve = Mock(side_effect=AssertionError("Query tokens must not be resolved"))
    monkeypatch.setattr(auth, "get_user_from_token", resolve)
    response = client.get(
        f"/api/auth/custom/login?{field}={token}", follow_redirects=False
    )
    assert response.status_code == 410
    assert "location" not in response.headers
    assert "set-cookie" not in response.headers
    assert response.headers["cache-control"] == "no-store"
    assert token not in response.text
    resolve.assert_not_called()


def test_provider_errors_are_not_reported_as_a_successful_json_url(launch):
    client, db, _, provider, token = launch
    provider.client_id = ""
    db.commit()
    response = client.get(
        _post(client, token).json()["authorization_url"], follow_redirects=False
    )
    assert response.status_code >= 400
    assert response.headers["cache-control"] == "no-store"


def test_launch_ticket_is_single_use_after_rollback_and_for_other_clients(launch):
    client, db, _, _, token = launch
    url = _post(client, token).json()["authorization_url"]
    assert client.get(url, follow_redirects=False).status_code == 307
    db.rollback()
    client.cookies.clear()
    assert client.get(url, follow_redirects=False).status_code == 401


def test_ticket_is_bound_to_provider_and_signed_options(launch):
    client, _, _, _, token = launch
    url = _post(client, token, redirect="https://original.example/tools").json()[
        "authorization_url"
    ]
    assert (
        client.get(
            url.replace("/custom/", "/other/"), follow_redirects=False
        ).status_code
        == 401
    )
    # Query options cannot replace the signed options or steal the completion.
    response = client.get(
        url + "&redirect=https://attacker.example&account=attacker",
        follow_redirects=False,
    )
    state = parse_qs(urlsplit(response.headers["location"]).query)["state"][0]
    assert auth.verify_token(state)["redirect"] == "https://original.example/tools"


@pytest.mark.parametrize("kind", ["access", "refresh", "oauth_state", "file_stream"])
def test_other_token_purposes_cannot_be_redeemed_as_launch_tickets(launch, kind):
    client, _, user, _, _ = launch
    ticket = auth.create_access_token({**user_session_claims(user), "type": kind})
    assert (
        client.get("/api/auth/custom/login", params={"ticket": ticket}).status_code
        == 401
    )


def test_password_change_between_issue_and_redeem_revokes_ticket(launch):
    client, db, user, _, token = launch
    url = _post(client, token).json()["authorization_url"]
    user.session_epoch = "changed"
    db.commit()
    assert client.get(url, follow_redirects=False).status_code == 401


@pytest.mark.parametrize("change", ["subject", "deleted"])
def test_redeem_requires_existing_matching_subject(launch, change):
    client, db, user, _, token = launch
    url = _post(client, token).json()["authorization_url"]
    if change == "subject":
        user.username = "renamed"
    else:
        db.delete(user)
    db.commit()
    response = client.get(url, follow_redirects=False)
    assert response.status_code == 401
    assert "location" not in response.headers
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("change", ["session_epoch", "username"])
def test_redeem_rechecks_identity_after_consumption_commit(launch, monkeypatch, change):
    from xagent.web.services import oauth_launch

    client, _, _, _, token = launch
    url = _post(client, token).json()["authorization_url"]
    original = oauth_launch.consume_oauth_launch_ticket

    def consume_and_change(db, claims):
        consumed = original(db, claims)
        user = db.get(User, claims["user_id"])
        setattr(user, change, "changed-after-commit")
        db.commit()
        return consumed

    monkeypatch.setattr(oauth_launch, "consume_oauth_launch_ticket", consume_and_change)
    response = client.get(url, follow_redirects=False)
    assert response.status_code == 401
    assert "location" not in response.headers
    assert "set-cookie" not in response.headers


def test_expired_ticket_is_rejected_without_redirect_or_cookie(launch):
    client, _, user, _, _ = launch
    ticket = auth.create_access_token(
        {
            **user_session_claims(user),
            "type": "oauth_launch",
            "provider": "custom",
            "jti": secrets.token_urlsafe(24),
            "options": {},
        },
        expires_delta=timedelta(seconds=-1),
    )
    response = client.get(
        "/api/auth/custom/login", params={"ticket": ticket}, follow_redirects=False
    )
    assert response.status_code == 401
    assert "location" not in response.headers
    assert "set-cookie" not in response.headers


def test_database_outage_fails_closed_without_leaking_ticket(launch, monkeypatch):
    from sqlalchemy.exc import OperationalError

    from xagent.web.services import oauth_launch

    client, _, _, _, token = launch
    url = _post(client, token).json()["authorization_url"]

    def fail(*_):
        raise OperationalError("statement", {}, Exception("database unavailable"))

    monkeypatch.setattr(oauth_launch, "consume_oauth_launch_ticket", fail)
    response = client.get(url, follow_redirects=False)
    assert response.status_code == 503
    assert "location" not in response.headers
    assert "ticket=" not in response.text
    assert "set-cookie" not in response.headers


def test_unknown_provider_is_a_private_error(launch):
    client, _, _, _, token = launch
    response = client.post(
        "/api/auth/unknown/login", headers={"Authorization": f"Bearer {token}"}, json={}
    )
    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"


def test_manifest_post_options_are_bound_into_ticket_without_a_query_credential(launch):
    client, _, _, _, token = launch
    response = client.post(
        "/api/auth/custom/login?app_id=custom-app&redirect=https://frontend.example/tools",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    ticket = parse_qs(urlsplit(response.json()["authorization_url"]).query)["ticket"][0]
    claims = auth.verify_token(ticket)
    assert claims["options"]["app_id"] == "custom-app"
    assert claims["options"]["redirect"] == "https://frontend.example/tools"


def test_launch_reports_the_configured_callback_origin_not_frontend_origin(launch):
    client, db, _, provider, token = launch
    provider.redirect_uri = "https://callback.example:8443/api/auth/custom/callback"
    db.commit()
    response = _post(client, token, redirect="https://frontend.example")
    assert response.status_code == 200
    assert response.json()["callback_origin"] == "https://callback.example:8443"


def test_conflicting_or_oversized_manifest_options_are_rejected(launch):
    client, _, _, _, token = launch
    headers = {"Authorization": f"Bearer {token}"}
    assert (
        client.post(
            "/api/auth/custom/login?app_id=one", headers=headers, json={"app_id": "two"}
        ).status_code
        == 400
    )

    assert (
        client.post(
            "/api/auth/custom/login",
            params={"account": "a" * 256},
            headers=headers,
            json={},
        ).status_code
        == 400
    )


@pytest.mark.parametrize("proof", ["", "invitation-proof", "邀请证明"])
def test_existing_invitation_password_hash_helper_remains_importable(proof):
    assert (
        auth.hash_password_reset_token(proof)
        == hashlib.sha256(proof.encode()).hexdigest()
    )
