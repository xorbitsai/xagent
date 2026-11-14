"""Drive the three web OAuth entries the way a browser reaches them.

``oauth_login`` is the in-process adapter, ``start_oauth_authorization`` issues
one short-lived launch ticket, and ``redeem_oauth_launch`` redeems it. Each
leads to the redirect to an app's Composio begin location when the app's
Composio switch is on.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

from xagent.web import composio_auth
from xagent.web.api import auth as auth_api
from xagent.web.auth_sessions import user_session_claims
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.user import User

WEB_ENTRIES = ("oauth_login", "redeem_oauth_launch", "start_oauth_authorization")


def access_token(user: User) -> str:
    return auth_api.create_access_token(
        data={**user_session_claims(user), "type": "access"},
        expires_delta=timedelta(minutes=5),
    )


def launch_ticket(user: User, provider: str, app_id: str) -> str:
    """One launch ticket, minted the way ``start_oauth_authorization`` does."""
    return auth_api.create_access_token(
        {
            **user_session_claims(user),
            "type": "oauth_launch",
            "provider": provider,
            "jti": secrets.token_urlsafe(24),
            "options": {"app_id": app_id, "redirect": None, "account": None},
        },
        expires_delta=timedelta(seconds=30),
    )


def redeem(db: Any, provider: str, ticket: str) -> Any:
    return auth_api.redeem_oauth_launch(
        provider, SimpleNamespace(query_params={}), ticket=ticket, db=db
    )


def start_authorization(
    db: Any, user: User, provider: str, app_id: str
) -> tuple[Any, str]:
    response = auth_api.start_oauth_authorization(
        provider,
        SimpleNamespace(query_params={}, headers={}),
        auth_api.OAuthLaunchRequest(app_id=app_id),
        db,
        user,
    )
    return response, json.loads(response.body)["callback_origin"]


def web_begin(
    db: Any, user: User, provider: str, app_id: str, entry: str
) -> tuple[Any, str | None]:
    """The begin-location redirect one web entry leads to, and its origin."""
    if entry == "oauth_login":
        return (
            auth_api.oauth_login(
                provider, token=access_token(user), app_id=app_id, db=db
            ),
            None,
        )
    if entry == "redeem_oauth_launch":
        return redeem(db, provider, launch_ticket(user, provider, app_id)), None
    response, callback_origin = start_authorization(db, user, provider, app_id)
    assert response.status_code == 200, response.body
    ticket = parse_qs(urlsplit(json.loads(response.body)["authorization_url"]).query)[
        "ticket"
    ][0]
    return redeem(db, provider, ticket), callback_origin


def begin_redirect(response: Any) -> tuple[str, dict[str, Any]]:
    """The begin-location redirect and the signed state it carries."""
    assert response.status_code in (302, 307), response.body
    location = response.headers["location"]
    state = auth_api.verify_token(parse_qs(urlsplit(location).query)["state"][0])
    assert state is not None
    return location, state


def assert_ordinary_state(
    db: Any, state: dict[str, Any], *, provider: str, app_id: str, session_epoch: Any
) -> None:
    assert state["type"] == "oauth_state"
    assert state["provider"] == provider
    assert state["app_id"] == app_id
    assert state[composio_auth.AUTH_BACKEND_CLAIM] == composio_auth.AUTH_BACKEND
    assert state.get("resource_owner_key") is None
    assert state["session_epoch"] == session_epoch
    assert re.fullmatch(r"[0-9a-f]{64}", state["browser_nonce_hash"]) is not None
    # Keys the flow row, which begin only checks while verify or denied spends
    # it; it also hydrates the ordinary proof cookie, as an actor nonce does.
    assert state["actor_flow_nonce"] == state["browser_nonce_hash"]
    assert db.query(ActorOAuthFlowState).count() == 1


def assert_proof_cookie(response: Any, browser: Any, state: dict[str, Any], path: str):
    name = auth_api._ordinary_oauth_cookie_name(state["browser_nonce_hash"])
    browser.receive(response)
    assert browser.cookies[name][1] == path
    (header,) = [
        value
        for value in response.headers.getlist("set-cookie")
        if value.startswith(f"{name}=")
    ]
    assert f"Path={path}" in header
    assert "HttpOnly" in header
    assert "SameSite=lax" in header
    assert "Secure" in header
