import pytest

from xagent.core.utils.encryption import (
    _get_encryption_key,
    decrypt_value,
    encrypt_value,
)


def test_encrypt_decrypt_roundtrip():
    original_value = "my_super_secret_value"
    encrypted = encrypt_value(original_value)

    assert encrypted != original_value
    assert isinstance(encrypted, str)

    decrypted = decrypt_value(encrypted)
    assert decrypted == original_value


def test_encrypt_empty_value():
    assert encrypt_value("") == ""
    assert encrypt_value(None) is None


def test_decrypt_empty_value():
    assert decrypt_value("") == ""
    assert decrypt_value(None) is None


def test_decrypt_invalid_token():
    # Provide an invalid token, should catch InvalidToken and return ""
    invalid_encrypted = "invalid_token_value"
    result = decrypt_value(invalid_encrypted)
    assert result == ""


def test_get_encryption_key_no_env(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    key = _get_encryption_key()
    assert key == "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="


def test_get_encryption_key_production_missing_key(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(
        ValueError, match="ENCRYPTION_KEY environment variable is not set"
    ):
        _get_encryption_key()


def test_get_encryption_key_with_env(monkeypatch):
    test_key = "some_test_key_base64_encoded="
    monkeypatch.setenv("ENCRYPTION_KEY", test_key)
    key = _get_encryption_key()
    assert key == test_key
