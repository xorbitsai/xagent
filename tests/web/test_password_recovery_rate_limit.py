import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from xagent.web.api import auth
from xagent.web.api.auth import (
    _PASSWORD_RECOVERY_LIMIT,
    _password_recovery_attempt_key,
    _reserve_password_recovery_attempt,
)


def _request(client_host: str = "198.51.100.10") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/forgot-password",
            "headers": [],
            "client": (client_host, 1234),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_password_recovery_limiter_caps_one_identifier():
    request = _request()
    key = _password_recovery_attempt_key(request, "email", "unique@example.com")

    assert all(
        _reserve_password_recovery_attempt(key, now=float(i))
        for i in range(_PASSWORD_RECOVERY_LIMIT)
    )
    assert not _reserve_password_recovery_attempt(key, now=10.0)


def test_password_recovery_limiter_expires_attempts():
    request = _request("198.51.100.11")
    key = _password_recovery_attempt_key(request, "token", "unique-token")

    for offset in range(_PASSWORD_RECOVERY_LIMIT):
        assert _reserve_password_recovery_attempt(key, now=float(offset))
    assert _reserve_password_recovery_attempt(key, now=901.0)


def test_recovery_limiter_refuses_new_buckets_when_full(monkeypatch):
    monkeypatch.setattr(auth, "_PASSWORD_RECOVERY_MAX_KEYS", 2)
    monkeypatch.setattr(auth, "_password_recovery_attempts", {})
    assert _reserve_password_recovery_attempt("a", now=1)
    assert _reserve_password_recovery_attempt("b", now=2)
    assert not _reserve_password_recovery_attempt("c", now=3)
    assert len(auth._password_recovery_attempts) == 2


def test_login_account_limit_cannot_be_bypassed_by_rotating_ips(monkeypatch):
    monkeypatch.setattr(auth, "_login_failures", {})
    worker = AsyncMock(side_effect=HTTPException(401, "Incorrect username or password"))
    monkeypatch.setattr(auth, "run_auth_db_worker", worker)
    for index in range(6):
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                auth.login(
                    auth.LoginRequest(
                        username="distributed-target", password="incorrect"
                    ),
                    db=object(),
                    http_request=_request(f"198.51.100.{index + 1}"),
                )
            )
        assert error.value.status_code == (401 if index < 5 else 429)
    assert worker.await_count == 5


def test_filling_login_table_does_not_unlock_an_active_account(monkeypatch):
    monkeypatch.setattr(auth, "_LOGIN_FAILURE_MAX_KEYS", 2)
    monkeypatch.setattr(auth, "_login_failures", {})
    monkeypatch.setattr(auth, "_login_failure_last_sweep", 0)
    for _ in range(5):
        assert auth._reserve_login_attempt("victim", now=1)
    assert not auth._reserve_login_attempt("victim", now=2)
    assert auth._reserve_login_attempt("another", now=3)
    for i in range(5):
        assert not auth._reserve_login_attempt(f"flood-{i}", now=4)
    assert not auth._reserve_login_attempt("victim", now=5)
    assert auth._reserve_login_attempt("after-expiry", now=905)


def test_missing_client_ip_does_not_charge_login_twice(monkeypatch):
    monkeypatch.setattr(auth, "_login_failures", {})
    worker = AsyncMock(side_effect=HTTPException(401, "Incorrect username or password"))
    monkeypatch.setattr(auth, "run_auth_db_worker", worker)
    for _ in range(5):
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                auth.login(
                    auth.LoginRequest(username="same", password="incorrect"), object()
                )
            )
        assert error.value.status_code == 401
    assert worker.await_count == 5
