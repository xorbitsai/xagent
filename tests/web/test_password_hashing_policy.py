"""Exercise password policy with unmodified production bcrypt settings."""

import hashlib
from unittest.mock import Mock

import bcrypt
import pytest
from fastapi import HTTPException

from xagent.web.api import auth as auth_api
from xagent.web.models.user import User


@pytest.fixture(autouse=True)
def fast_password_hashing() -> None:
    """Override the web fixture so these tests retain production hashing cost."""


def test_password_hash_uses_production_cost_and_random_salt() -> None:
    first = auth_api.hash_password("correct-password")
    second = auth_api.hash_password("correct-password")

    assert first.startswith("$2b$12$")
    assert second.startswith("$2b$12$")
    assert first != second
    assert auth_api.verify_password("correct-password", first)
    assert not auth_api.verify_password("wrong-password", first)


def test_password_hash_preserves_characters_beyond_bcrypt_input_limit() -> None:
    prefix = "a" * 72
    stored = auth_api.hash_password(prefix + "correct-suffix")

    assert auth_api.verify_password(prefix + "correct-suffix", stored)
    assert not auth_api.verify_password(prefix + "wrong-suffix", stored)


def test_login_upgrades_legacy_password_to_production_bcrypt() -> None:
    password = "legacy-password"
    user = User(
        id=1,
        username="legacy-user",
        password_hash=hashlib.sha256(password.encode()).hexdigest(),
    )

    response = auth_api._prepare_login_response(
        user, auth_api.LoginRequest(username="legacy-user", password=password)
    )

    assert response["success"] is True
    assert str(user.password_hash).startswith("$2b$12$")
    assert auth_api.verify_password(password, str(user.password_hash))


@pytest.mark.parametrize("legacy_user", [False, True])
def test_rejected_login_pays_production_dummy_cost(
    monkeypatch: pytest.MonkeyPatch, legacy_user: bool
) -> None:
    user = (
        User(
            id=1,
            username="legacy-user",
            password_hash=hashlib.sha256(b"correct-password").hexdigest(),
        )
        if legacy_user
        else None
    )
    checkpw = Mock(wraps=bcrypt.checkpw)
    monkeypatch.setattr(bcrypt, "checkpw", checkpw)

    with pytest.raises(HTTPException) as caught:
        auth_api._prepare_login_response(
            user, auth_api.LoginRequest(username="legacy-user", password="wrong")
        )

    assert caught.value.status_code == 401
    checkpw.assert_called_once()
    assert checkpw.call_args.args[1].startswith(b"$2b$12$")
