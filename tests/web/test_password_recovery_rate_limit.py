from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
    monkeypatch.setenv("XAGENT_AUTH_RATE_LIMIT_MAX_LOCAL_KEYS", "2")
    for _ in range(5):
        assert auth._reserve_auth_budgets("login", (("victim", 5),), now=1) == 0
    assert auth._reserve_auth_budgets("login", (("unlocked", 5),), now=2) == 0
    assert auth._reserve_auth_budgets("login", (("replacement", 5),), now=3) == 0
    assert "unlocked" not in auth._login_failures
    assert not (auth._reserve_auth_budgets("login", (("victim", 5),), now=4) == 0)
    assert len(auth._login_failures) == 2
    assert auth._reserve_auth_budgets("login", (("after-expiry", 5),), now=905) == 0
    assert "victim" not in auth._login_failures


def test_recovery_evicts_unlocked_buckets(monkeypatch):
    monkeypatch.setenv("XAGENT_AUTH_RATE_LIMIT_MAX_LOCAL_KEYS", "2")
    assert auth._reserve_auth_budgets("recovery", (("a", 5),), now=1) == 0
    assert auth._reserve_auth_budgets("recovery", (("b", 5),), now=2) == 0
    assert auth._reserve_auth_budgets("recovery", (("c", 5),), now=3) == 0
    assert len(auth._password_recovery_attempts) == 2


def test_multi_bucket_rejection_does_not_charge_another_bucket():
    for _ in range(5):
        assert auth._reserve_auth_budgets("login", (("locked", 5),), now=1) == 0
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
    monkeypatch.setenv("XAGENT_AUTH_RECOVERY_IP_LIMIT", "3")
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
    monkeypatch.setenv("XAGENT_AUTH_LOGIN_IP_LIMIT", "3")
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


def test_successful_login_refunds_own_ip_but_preserves_other_failures(client):
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
    assert len(auth._login_failures[keys[0][0]].attempts) == 1
    assert all(key not in auth._login_failures for key, _ in keys[1:])


def test_shared_egress_allows_more_successes_than_ip_budget(client, monkeypatch):
    monkeypatch.setenv("XAGENT_AUTH_LOGIN_IP_LIMIT", "1")
    for _ in range(6):
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "alice", "password": "correct-password"},
            ).status_code
            == 200
        )
    assert auth._login_ip_budget("ip-a")[0] not in auth._login_failures


def test_ip_rejection_precedes_auth_database_worker(client, monkeypatch):
    monkeypatch.setenv("XAGENT_AUTH_LOGIN_IP_LIMIT", "2")
    worker = AsyncMock(side_effect=HTTPException(401, "bad credentials"))
    monkeypatch.setattr(auth, "run_auth_db_worker", worker)
    for expected in (401, 401, 429):
        assert (
            client.post(
                "/api/auth/login", json={"username": "alice", "password": "wrong"}
            ).status_code
            == expected
        )
    assert worker.await_count == 2


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/auth/login", {"username": "alice", "password": "wrong"}),
        ("/api/auth/forgot-password", {"email": "alice@example.com"}),
        (
            "/api/auth/reset-password",
            {"token": "guess", "new_password": "long-enough-password"},
        ),
    ],
)
def test_auth_endpoint_storage_outage_returns_sanitized_503(
    client, monkeypatch, path, payload
):
    from redis.exceptions import ConnectionError

    from xagent.web.services import auth_attempts

    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://private.invalid")
    backend = Mock()
    backend.eval.side_effect = ConnectionError("private-infrastructure")
    monkeypatch.setattr(auth_attempts, "_client", lambda _: backend)
    worker = AsyncMock()
    monkeypatch.setattr(auth, "run_auth_db_worker", worker)
    response = client.post(path, json=payload)
    assert response.status_code == 503
    assert "private" not in response.text
    worker.assert_not_called()


def test_forgot_and_reset_intentionally_share_recovery_ip_budget(client, monkeypatch):
    monkeypatch.setenv("XAGENT_AUTH_RECOVERY_IP_LIMIT", "1")
    assert (
        client.post(
            "/api/auth/forgot-password", json={"email": "missing@example.com"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/auth/reset-password",
            json={"token": "guess", "new_password": "long-password"},
        ).status_code
        == 429
    )


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
    monkeypatch.setenv("XAGENT_AUTH_LOGIN_ACCOUNT_LIMIT", "6")
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
