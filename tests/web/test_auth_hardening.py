"""Focused regression tests for the authentication hardening review fixes."""

from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from xagent.web.api import auth as auth_api


def test_login_budgets_normalize_unknown_identifiers() -> None:
    assert auth_api._login_budgets(
        "203.0.113.7", " Alice ", None
    ) == auth_api._login_budgets("203.0.113.7", "alice", None)


def test_unknown_login_pays_bcrypt_timing_floor(monkeypatch) -> None:
    dummy = Mock()
    monkeypatch.setattr(auth_api, "verify_dummy", dummy)

    with pytest.raises(HTTPException) as exc_info:
        auth_api._prepare_login_response(
            None, auth_api.LoginRequest(username="missing", password="secret")
        )

    assert exc_info.value.status_code == 401
    dummy.assert_called_once_with()


def test_oauth_state_cookie_follows_redirect_transport(monkeypatch) -> None:
    db_provider = SimpleNamespace(
        client_id="client-id",
        auth_url="https://provider.example/authorize",
        redirect_uri="https://app.example/api/auth/custom/callback",
        default_scopes=[],
    )
    db = SimpleNamespace()
    monkeypatch.setattr(auth_api, "_resolve_oauth_secret", lambda *args: "client-id")

    response = auth_api._generic_oauth_login(
        "custom",
        token=None,
        app_id=None,
        redirect=None,
        db=db,
        db_provider=db_provider,
        trusted_user_id=1,
        resource_owner_key=None,
        actor_flow_nonce=None,
        account=None,
    )
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    names = [
        name
        for name in cookie
        if name.startswith(auth_api._ORDINARY_OAUTH_STATE_COOKIE_PREFIX)
    ]
    assert len(names) == 1
    assert cookie[names[0]]["secure"]
