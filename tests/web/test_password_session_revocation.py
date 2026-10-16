"""Real password changes must invalidate live credentials, not just refresh."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from tests.web import test_auth_api as auth_tests
from tests.web.test_auth_api import test_db as test_db
from xagent.web.api import auth, files
from xagent.web.auth_dependencies import (
    get_user_from_token,
    get_user_from_websocket_token,
)
from xagent.web.models.user import User

pytestmark = pytest.mark.usefixtures("test_db")


@pytest.mark.parametrize("operation", ["change", "reset"])
def test_password_change_revokes_access_refresh_and_file_tickets(operation):
    auth_tests.setup_first_admin(password="old-password-123")
    client = auth_tests.client
    before = client.post(
        "/api/auth/login",
        json={
            "username": "administrator",
            "password": "old-password-123",
        },
    ).json()
    headers = {"Authorization": f"Bearer {before['access_token']}"}
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="administrator").one()
        ticket = asyncio.run(files.issue_preview_stream_ticket("example-file", user))
        raw_ticket = ticket["path"].split("ticket=", 1)[1]
        assert get_user_from_token(before["access_token"], session) is not None
        if operation == "reset":
            user.password_reset_token_hash = auth.hash_password_reset_token(
                "reset-secret"
            )
            user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=5
            )
            session.commit()
    if operation == "change":
        result = client.post(
            "/api/auth/change-password",
            headers=headers,
            json={
                "current_password": "old-password-123",
                "new_password": "new-password-456",
            },
        )
    else:
        result = client.post(
            "/api/auth/reset-password",
            json={
                "token": "reset-secret",
                "new_password": "new-password-456",
            },
        )
    assert result.status_code == 200
    assert result.json()["success"]
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    assert (
        client.post(
            "/api/auth/refresh",
            json={
                "refresh_token": before["refresh_token"],
            },
        ).status_code
        == 401
    )
    with auth_tests.TestingSessionLocal() as session:
        assert get_user_from_token(before["access_token"], session) is None
        assert get_user_from_websocket_token(before["access_token"], session) is None
        with pytest.raises(HTTPException) as error:
            files._user_from_stream_ticket(session, raw_ticket, "example-file")
        assert error.value.status_code == 401
    after = client.post(
        "/api/auth/login",
        json={
            "username": "administrator",
            "password": "new-password-456",
        },
    ).json()
    assert (
        client.get(
            "/api/auth/me",
            headers={
                "Authorization": f"Bearer {after['access_token']}",
            },
        ).status_code
        == 200
    )
    refreshed = client.post(
        "/api/auth/refresh",
        json={
            "refresh_token": after["refresh_token"],
        },
    )
    assert refreshed.status_code == 200
    assert (
        client.get(
            "/api/auth/me",
            headers={
                "Authorization": f"Bearer {refreshed.json()['access_token']}",
            },
        ).status_code
        == 200
    )


def test_reset_token_cannot_be_reused():
    auth_tests.setup_first_admin(password="old-password-123")
    with auth_tests.TestingSessionLocal() as session:
        user = session.query(User).first()
        user.password_reset_token_hash = auth.hash_password_reset_token("one-use")
        user.password_reset_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=5
        )
        session.commit()
    request = {"token": "one-use", "new_password": "new-password-456"}
    assert auth_tests.client.post("/api/auth/reset-password", json=request).json()[
        "success"
    ]
    assert not auth_tests.client.post("/api/auth/reset-password", json=request).json()[
        "success"
    ]
