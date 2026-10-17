from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query

from tests.web import test_auth_api as auth_tests
from tests.web.test_auth_api import test_db as test_db
from xagent.web.api import auth
from xagent.web.models.user import User

pytestmark = pytest.mark.usefixtures("test_db")


@pytest.fixture(autouse=True)
def configured_mail(monkeypatch):
    monkeypatch.setenv("XAGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("XAGENT_SMTP_FROM_EMAIL", "no-reply@example.com")
    monkeypatch.setattr(auth, "send_email_changed_notice", Mock())
    monkeypatch.setattr(
        auth,
        "remote_ip_from_request",
        lambda request: request.headers.get("x-test-ip", "ip-a"),
    )


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
        assert user.email_confirmation_hash == auth.hash_auth_token(sent[-1][1])
        assert user.pending_email == "new@example.com"
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
        "password-recovered",
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
    elif invalidity == "password-recovered":
        with auth_tests.TestingSessionLocal() as db:
            user = db.query(User).one()
            user.password_reset_token_hash = auth.hash_auth_token("recover-secret")
            user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=5
            )
            db.commit()
        assert auth_tests.client.post(
            "/api/auth/reset-password",
            json={"token": "recover-secret", "new_password": "new-password-123"},
        ).json()["success"]
        headers = {
            "Authorization": f"Bearer {auth_tests.login_and_get_token('administrator', 'new-password-123')}"
        }
        with auth_tests.TestingSessionLocal() as db:
            user = db.query(User).one()
            assert user.pending_email is None
            assert user.email_confirmation_hash is None
            assert user.email_confirmation_expires_at is None
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


def test_missing_smtp_is_actionable_and_does_not_replace_pending_proof(
    email_change, monkeypatch
):
    headers, sent = email_change
    monkeypatch.delenv("XAGENT_SMTP_HOST")
    response = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={
            "email": "second@example.com",
            "current_password": "admin123",
        },
    ).json()
    assert not response["success"]
    assert response["error_code"] == "smtp_not_configured"
    assert "administrator" in response["message"]
    assert len(sent) == 1
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).one()
        assert user.pending_email == "new@example.com"
        assert user.email_confirmation_hash == auth.hash_auth_token(sent[0][1])
    profile = auth_tests.client.get("/api/auth/me", headers=headers).json()["user"]
    assert profile["email_change_available"] is False
    assert profile["pending_email"] == "new@example.com"
    assert "email_confirmation_hash" not in profile


def test_current_email_is_noop_without_mail_or_proof_replacement(email_change):
    headers, sent = email_change
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).one()
        pending_proof = (
            user.pending_email,
            user.email_confirmation_hash,
            user.email_confirmation_expires_at,
        )
    result = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={
            "email": "administrator@example.com",
            "current_password": "admin123",
        },
    ).json()
    assert result["success"] and not result["pending_confirmation"]
    assert result["unchanged"] is True
    assert "unchanged" in result["message"]
    assert result["user"]["pending_email"] == "new@example.com"
    assert len(sent) == 1
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).one()
        assert (
            user.pending_email,
            user.email_confirmation_hash,
            user.email_confirmation_expires_at,
        ) == pending_proof
    assert auth_tests.client.post(
        "/api/auth/email/confirm", headers=headers, json={"token": sent[0][1]}
    ).json()["success"]


def _other_user_headers(username="other"):
    with auth_tests.TestingSessionLocal() as db:
        password_hash = (
            db.query(User).filter_by(username="administrator").one().password_hash
        )
        user = User(
            username=username,
            email=f"{username}@example.com",
            password_hash=password_hash,
        )
        db.add(user)
        db.commit()
        token = auth.create_access_token(auth.user_session_claims(user))
    return {"Authorization": f"Bearer {token}"}


def test_current_email_without_pending_proof_is_explicitly_unchanged(email_change):
    headers, sent = email_change
    assert auth_tests.client.post(
        "/api/auth/email/confirm", headers=headers, json={"token": sent[0][1]}
    ).json()["success"]
    result = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={"email": "new@example.com", "current_password": "admin123"},
    ).json()
    assert result["success"] and result["unchanged"]
    assert not result["pending_confirmation"]
    assert result["user"]["pending_email"] is None
    assert len(sent) == 1


@pytest.mark.parametrize("dimension", ["user", "target", "ip"])
def test_confirmation_requests_have_user_target_and_source_budgets(
    email_change, monkeypatch, dimension
):
    headers, sent = email_change
    settings = {
        "user": "XAGENT_AUTH_RECOVERY_PAIR_LIMIT",
        "target": "XAGENT_EMAIL_CHANGE_TARGET_LIMIT",
        "ip": "XAGENT_AUTH_RECOVERY_IP_LIMIT",
    }
    for name, setting in settings.items():
        monkeypatch.setenv(setting, "1" if name == dimension else "100")
    # Keep only the tested dimension identical to the fixture's first request.
    if dimension != "user":
        headers = _other_user_headers()
    headers = {**headers, "x-test-ip": "ip-a" if dimension == "ip" else "ip-b"}
    response = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={
            "email": "NEW@EXAMPLE.COM"
            if dimension == "target"
            else "second@example.com",
            "current_password": "admin123",
        },
    )
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
    assert len(sent) == 1


@pytest.mark.parametrize("password", ["wrong-password", None])
def test_bad_password_attempts_consume_user_budget(email_change, monkeypatch, password):
    headers, sent = email_change
    monkeypatch.setenv("XAGENT_AUTH_RECOVERY_PAIR_LIMIT", "2")
    response = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={"email": "second@example.com", "current_password": password},
    )
    assert response.status_code == 200
    assert not response.json()["success"]
    limited = auth_tests.client.patch(
        "/api/auth/email",
        headers={**headers, "x-test-ip": "ip-b"},
        json={"email": "third@example.com", "current_password": "admin123"},
    )
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0
    assert len(sent) == 1


def test_default_target_budget_is_shared_across_users_and_ips(
    email_change, monkeypatch
):
    _, sent = email_change
    monkeypatch.delenv("XAGENT_EMAIL_CHANGE_TARGET_LIMIT", raising=False)
    monkeypatch.setenv("XAGENT_AUTH_RECOVERY_EMAIL_LIMIT", "100")
    for index in range(1, 6):
        response = auth_tests.client.patch(
            "/api/auth/email",
            headers={
                **_other_user_headers(f"other{index}"),
                "x-test-ip": f"ip-{index}",
            },
            json={"email": "new@example.com", "current_password": "admin123"},
        )
        assert response.status_code == (200 if index < 5 else 429)
        if index < 5:
            assert response.json()["pending_confirmation"]
    assert len(sent) == 5


@pytest.mark.parametrize("expiry", ["expired", "missing"])
def test_profile_hides_inactive_pending_target(email_change, expiry):
    headers, _ = email_change
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).one()
        user.email_confirmation_expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
            if expiry == "expired"
            else None
        )
        db.commit()
    profile = auth_tests.client.get("/api/auth/me", headers=headers).json()["user"]
    assert profile["pending_email"] is None
    assert profile["email"] == "administrator@example.com"
    assert "email_confirmation_hash" not in profile


@pytest.mark.parametrize("notice_fails", [False, True])
def test_confirmation_clears_credentials_and_notifies_old_address(
    email_change, monkeypatch, notice_fails
):
    headers, sent = email_change
    notice = Mock(
        side_effect=RuntimeError("SMTP unavailable") if notice_fails else None
    )
    monkeypatch.setattr(auth, "send_email_changed_notice", notice)
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).one()
        user.password_reset_token_hash = "old-reset-digest"
        user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=5
        )
        assert user.refresh_token is not None
        db.commit()
    result = auth_tests.client.post(
        "/api/auth/email/confirm", headers=headers, json={"token": sent[0][1]}
    ).json()
    assert result["success"]
    notice.assert_called_once_with("administrator@example.com", auth.get_app_name())
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).one()
        assert user.email == "new@example.com"
        for field in (
            "refresh_token",
            "refresh_token_expires_at",
            "password_reset_token_hash",
            "password_reset_expires_at",
            "pending_email",
            "email_confirmation_hash",
            "email_confirmation_expires_at",
        ):
            assert getattr(user, field) is None


def test_reissued_code_can_confirm_and_old_code_cannot(email_change):
    headers, sent = email_change
    assert auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={
            "email": "second@example.com",
            "current_password": "admin123",
        },
    ).json()["success"]
    for code, success in ((sent[0][1], False), (sent[1][1], True)):
        result = auth_tests.client.post(
            "/api/auth/email/confirm", headers=headers, json={"token": code}
        ).json()
        assert result["success"] is success


@pytest.mark.parametrize("conflict", ["email", "username"])
def test_confirmation_rechecks_new_namespace_conflicts(email_change, conflict):
    headers, sent = email_change
    with auth_tests.TestingSessionLocal() as db:
        db.add(
            User(
                username="new@example.com" if conflict == "username" else "other",
                email="new@example.com" if conflict == "email" else "other@example.com",
                password_hash="unused",
            )
        )
        db.commit()
    response = auth_tests.client.post(
        "/api/auth/email/confirm", headers=headers, json={"token": sent[0][1]}
    ).json()
    assert not response["success"]
    assert "no longer available" in response["message"]


def test_expiry_configuration_is_serialized_and_applied(email_change, monkeypatch):
    headers, _ = email_change
    monkeypatch.setenv("XAGENT_EMAIL_CONFIRMATION_EXPIRE_MINUTES", "7")
    result = auth_tests.client.patch(
        "/api/auth/email",
        headers=headers,
        json={
            "email": "second@example.com",
            "current_password": "admin123",
        },
    ).json()
    assert result["user"]["email_confirmation_expire_minutes"] == 7
    with auth_tests.TestingSessionLocal() as db:
        expires = (
            db.query(User)
            .one()
            .email_confirmation_expires_at.replace(tzinfo=timezone.utc)
        )
        assert 410 < (expires - datetime.now(timezone.utc)).total_seconds() <= 420


@pytest.mark.parametrize("phase", ["request", "confirm", "integrity-error"])
def test_conditional_write_failure_preserves_recovery_address(
    email_change, monkeypatch, phase
):
    headers, sent = email_change

    def failed_write(*args, **kwargs):
        if phase == "integrity-error":
            raise IntegrityError("UPDATE users", {}, RuntimeError("duplicate email"))
        return 0

    monkeypatch.setattr(Query, "update", failed_write)
    if phase == "request":
        response = auth_tests.client.patch(
            "/api/auth/email",
            headers=headers,
            json={
                "email": "second@example.com",
                "current_password": "admin123",
            },
        )
    else:
        response = auth_tests.client.post(
            "/api/auth/email/confirm", headers=headers, json={"token": sent[0][1]}
        )
    assert response.status_code == 200
    assert not response.json()["success"]
    assert len(sent) == 1
    with auth_tests.TestingSessionLocal() as db:
        user = db.query(User).one()
        assert user.email == "administrator@example.com"
        assert user.pending_email == "new@example.com"
