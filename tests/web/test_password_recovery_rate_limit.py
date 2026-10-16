from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.web.api import auth
from xagent.web.models.database import Base
from xagent.web.models.user import User


@pytest.fixture(autouse=True)
def clean_budgets(monkeypatch):
    monkeypatch.setattr(auth, "_login_failures", {})
    monkeypatch.setattr(auth, "_password_recovery_attempts", {})


def _request(client_host="198.51.100.10"):
    return Request({"type": "http", "headers": [], "client": (client_host, 1234)})


def test_local_admission_expires_and_evicts_only_unlocked_buckets(monkeypatch):
    monkeypatch.setattr(auth, "_LOGIN_FAILURE_MAX_KEYS", 2)
    for _ in range(5):
        assert auth._reserve_login_attempt("victim", now=1)
    assert auth._reserve_login_attempt("unlocked", now=2)
    assert auth._reserve_login_attempt("replacement", now=3)
    assert "unlocked" not in auth._login_failures
    assert not auth._reserve_login_attempt("victim", now=4)
    assert len(auth._login_failures) == 2
    assert auth._reserve_login_attempt("after-expiry", now=905)
    assert "victim" not in auth._login_failures


def test_recovery_evicts_unlocked_buckets(monkeypatch):
    monkeypatch.setattr(auth, "_PASSWORD_RECOVERY_MAX_KEYS", 2)
    assert auth._reserve_password_recovery_attempt("a", now=1)
    assert auth._reserve_password_recovery_attempt("b", now=2)
    assert auth._reserve_password_recovery_attempt("c", now=3)
    assert len(auth._password_recovery_attempts) == 2


def test_multi_bucket_rejection_does_not_charge_another_bucket():
    for _ in range(5):
        assert auth._reserve_login_attempt("locked", now=1)
    assert (
        auth._reserve_auth_budgets("login", (("new-ip", 200), ("locked", 5)), now=2) > 0
    )
    assert "new-ip" not in auth._login_failures


def test_username_email_aliases_share_resolved_account_budget():
    user = SimpleNamespace(id=7)
    a = auth._login_budgets("ip-a", "alice", user)
    b = auth._login_budgets("ip-b", "alice@example.com", user)
    assert a[1] == b[1]
    assert a[1][1] >= 50
    assert a[0] != b[0]
    assert a[0][1] > a[1][1] > a[2][1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'auth.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    with sessions() as db:
        db.add(
            User(
                username="alice",
                email="alice@example.com",
                password_hash=auth.hash_password("correct-password"),
            )
        )
        db.commit()

    @contextmanager
    def session_factory():
        with sessions() as db:
            yield db

    def get_db():
        with sessions() as db:
            yield db

    app = FastAPI()
    app.include_router(auth.auth_router)
    app.dependency_overrides[auth.get_db] = get_db
    app.dependency_overrides[auth.get_auth_db] = lambda: session_factory
    monkeypatch.setattr(
        auth,
        "remote_ip_from_request",
        lambda request: request.headers.get("x-test-ip", "ip-a")
        if request
        else "unknown",
    )
    monkeypatch.setattr(auth, "verify_dummy", lambda: None)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def test_forgot_password_endpoint_429_and_retry_header_without_global_email_lock(
    client,
):
    for _ in range(5):
        assert (
            client.post(
                "/api/auth/forgot-password", json={"email": "missing@example.com"}
            ).status_code
            == 200
        )
    response = client.post(
        "/api/auth/forgot-password", json={"email": "missing@example.com"}
    )
    assert response.status_code == 429
    assert 0 < int(response.headers["Retry-After"]) <= 900
    # Five requests from one browser must not globally deny the target account.
    assert (
        client.post(
            "/api/auth/forgot-password",
            headers={"x-test-ip": "ip-b"},
            json={"email": "missing@example.com"},
        ).status_code
        == 200
    )


def test_reset_guesses_share_ip_budget_and_short_password_does_not_charge(
    client, monkeypatch
):
    monkeypatch.setattr(auth, "_PASSWORD_RECOVERY_IP_LIMIT", 3)
    short = client.post(
        "/api/auth/reset-password", json={"token": "guess", "new_password": "x"}
    )
    assert short.status_code == 200
    assert auth._password_recovery_attempts == {}
    for i in range(3):
        assert (
            client.post(
                "/api/auth/reset-password",
                json={"token": f"guess-{i}", "new_password": "new-password"},
            ).status_code
            == 200
        )
    response = client.post(
        "/api/auth/reset-password",
        json={"token": "fresh-guess", "new_password": "new-password"},
    )
    assert response.status_code == 429
    assert 0 < int(response.headers["Retry-After"]) <= 900
    assert len(auth._password_recovery_attempts) == 1


def test_ip_churn_cannot_deny_other_ips(client, monkeypatch):
    monkeypatch.setattr(auth, "_LOGIN_IP_LIMIT", 3)
    for i in range(3):
        assert (
            client.post(
                "/api/auth/login",
                json={"username": f"unknown-{i}", "password": "wrong"},
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/api/auth/login", json={"username": "new", "password": "wrong"}
        ).status_code
        == 429
    )
    assert (
        client.post(
            "/api/auth/login",
            headers={"x-test-ip": "ip-b"},
            json={"username": "other", "password": "wrong"},
        ).status_code
        == 401
    )


def test_successful_login_clears_identity_budgets_but_not_ip(client):
    assert (
        client.post(
            "/api/auth/login", json={"username": "alice", "password": "wrong"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login",
            json={"username": "alice@example.com", "password": "correct-password"},
        ).status_code
        == 200
    )
    keys = auth._login_budgets("ip-a", "alice", SimpleNamespace(id=1))
    assert keys[0][0] in auth._login_failures
    assert all(key not in auth._login_failures for key, _ in keys[1:])


def test_clear_outage_does_not_fail_successful_login(client, monkeypatch):
    def outage(*args):
        raise HTTPException(503, "unavailable")

    monkeypatch.setattr(auth, "clear_shared_attempt", outage)
    assert (
        client.post(
            "/api/auth/login",
            json={"username": "alice", "password": "correct-password"},
        ).status_code
        == 200
    )


def test_rotating_ips_share_account_limit_without_alias_bypass(client, monkeypatch):
    monkeypatch.setattr(auth, "_LOGIN_ACCOUNT_LIMIT", 6)
    monkeypatch.setattr(auth, "verify_password", lambda *args: False)
    for i in range(6):
        assert (
            client.post(
                "/api/auth/login",
                headers={"x-test-ip": f"ip-{i}"},
                json={
                    "username": "alice" if i % 2 else "alice@example.com",
                    "password": "wrong",
                },
            ).status_code
            == 401
        )
    response = client.post(
        "/api/auth/login",
        headers={"x-test-ip": "new-ip"},
        json={"username": "alice", "password": "wrong"},
    )
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
