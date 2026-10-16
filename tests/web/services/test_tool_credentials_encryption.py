import base64
import hashlib

import pytest
from cryptography.fernet import Fernet

from xagent.web.services.tool_credentials import _build_fernet_key, _decrypt, _encrypt


def test_development_uses_the_local_fallback(monkeypatch):
    monkeypatch.delenv("XAGENT_SECRET_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("XAGENT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("XAGENT_REQUIRE_SECURE_SECRETS", raising=False)

    assert _build_fernet_key() == base64.urlsafe_b64encode(
        hashlib.sha256(b"xagent-dev-key").digest()
    )


@pytest.mark.parametrize("environment", ["production", "staging", "qa"])
def test_deployed_environment_rejects_missing_encryption_key(monkeypatch, environment):
    monkeypatch.delenv("XAGENT_SECRET_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("XAGENT_ENVIRONMENT", environment)

    with pytest.raises(RuntimeError, match="XAGENT_SECRET_ENCRYPTION_KEY"):
        _build_fernet_key()


def test_secure_secret_override_rejects_the_development_fallback(monkeypatch):
    monkeypatch.delenv("XAGENT_SECRET_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "development")
    monkeypatch.setenv("XAGENT_REQUIRE_SECURE_SECRETS", "true")

    with pytest.raises(RuntimeError, match="XAGENT_SECRET_ENCRYPTION_KEY"):
        _build_fernet_key()


def test_configured_encryption_key_is_used(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv(
        "XAGENT_SECRET_ENCRYPTION_KEY", "deployment-secret-01234567890123456789"
    )
    monkeypatch.delenv("SECRET_KEY", raising=False)

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(b"deployment-secret-01234567890123456789").digest()
    )
    assert _build_fernet_key() == expected
    ciphertext = _encrypt("private credential")
    assert Fernet(expected).decrypt(ciphertext.encode()) == b"private credential"
    assert _decrypt(ciphertext) == "private credential"
