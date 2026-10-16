from starlette.requests import Request

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
