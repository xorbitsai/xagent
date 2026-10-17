from datetime import datetime, timedelta, timezone

import pytest

from tests.web import test_auth_api as auth_tests
from tests.web.test_auth_api import test_db as test_db
from xagent.web.api import auth
from xagent.web.models.user import User

pytestmark = pytest.mark.usefixtures("test_db")


@pytest.fixture
def email_change(monkeypatch):
    auth_tests.setup_first_admin()
    token = auth_tests.login_and_get_token("administrator", "admin123")
    headers = {"Authorization": f"Bearer {token}"}
    sent = []
    monkeypatch.setattr(
        auth,
        "send_email_change_confirmation",
        lambda email, code, app: sent.append((email, code)),
    )
    result = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={
            "email": "new@example.com",
            "current_password": "admin123",
        },
    )
    assert result.json()["pending_confirmation"]
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).first()
        assert user.email == "administrator@example.com"
        assert user.email_confirmation_hash != sent[-1][1]
    return headers, sent


def test_email_confirmation_is_single_use(email_change):
    headers, sent = email_change
    payload = {"token": sent[-1][1]}
    confirmed = auth_tests.client.post(
        "/api/auth/email/confirm", headers=headers, json=payload
    )
    assert confirmed.json()["user"]["email"] == "new@example.com"
    assert not auth_tests.client.post(
        "/api/auth/email/confirm", headers=headers, json=payload
    ).json()["success"]


@pytest.mark.parametrize(
    "invalidity",
    [
        "wrong-code",
        "expired",
        "other-user",
        "replaced",
        "password-changed",
        "admin-password-reset",
    ],
)
def test_email_confirmation_rejects_invalid_proof(email_change, invalidity):
    headers, sent = email_change
    code = sent[-1][1]
    if invalidity == "wrong-code":
        code = "invalid-confirmation-code"
    elif invalidity == "expired":
        with auth_tests.TestingSessionLocal() as db:
            user = db.query(User).first()
            user.email_confirmation_expires_at = datetime.now(timezone.utc) - timedelta(
                seconds=1
            )
            db.commit()
    elif invalidity == "other-user":
        auth_tests.client.post(
            "/api/auth/register",
            json={
                "username": "other",
                "email": "other@example.com",
                "password": "other-password",
            },
        )
        headers = {
            "Authorization": f"Bearer {auth_tests.login_and_get_token('other', 'other-password')}"
        }
    elif invalidity == "password-changed":
        changed = auth_tests.client.post(
            "/api/auth/change-password",
            headers=headers,
            json={
                "current_password": "admin123",
                "new_password": "new-password-123",
            },
        )
        assert changed.json()["success"]
        headers = {
            "Authorization": f"Bearer {auth_tests.login_and_get_token('administrator', 'new-password-123')}"
        }
    elif invalidity == "admin-password-reset":
        from xagent.web.reset_admin_password import reset_admin_password

        reset_admin_password("administrator", "new-password-123")
        with auth_tests.TestingSessionLocal() as db:
            user = db.query(User).filter_by(username="administrator").one()
            assert user.pending_email is None
            assert user.email_confirmation_hash is None
            assert user.email_confirmation_expires_at is None
        headers = {
            "Authorization": f"Bearer {auth_tests.login_and_get_token('administrator', 'new-password-123')}"
        }
    else:
        assert auth_tests.client.patch(
            "/api/auth/email",
            headers=headers,
            json={
                "email": "second@example.com",
                "current_password": "admin123",
            },
        ).json()["success"]
    response = auth_tests.client.post(
        "/api/auth/email/confirm", headers=headers, json={"token": code}
    )
    assert not response.json()["success"]
    with auth_tests.TestingSessionLocal() as db:
        assert (
            db.query(User).filter_by(username="administrator").one().email
            == "administrator@example.com"
        )


def test_delivery_failure_keeps_old_email_and_discards_proof(email_change, monkeypatch):
    headers, sent = email_change

    def fail(*args):
        raise RuntimeError("SMTP unavailable")

    monkeypatch.setattr(auth, "send_email_change_confirmation", fail)
    response = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={
            "email": "second@example.com",
            "current_password": "admin123",
        },
    )
    assert not response.json()["success"]
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).first()
        assert user.email == "administrator@example.com"
        assert user.email_confirmation_hash is None
