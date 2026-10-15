import pytest

from xagent.web import auth_config


@pytest.fixture(autouse=True)
def _clear_auth_environment(monkeypatch):
    for name in (
        "ENVIRONMENT",
        "XAGENT_ENVIRONMENT",
        "XAGENT_JWT_SECRET",
        "XAGENT_REQUIRE_SECURE_SECRETS",
        "XAGENT_SESSION_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)


def test_development_allows_the_development_secret(monkeypatch):
    monkeypatch.delenv("XAGENT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("XAGENT_REQUIRE_SECURE_SECRETS", raising=False)
    monkeypatch.delenv("XAGENT_JWT_SECRET", raising=False)
    auth_config.validate_auth_config()


@pytest.mark.parametrize("environment", ["production", "staging", "qa"])
def test_non_development_rejects_blank_and_placeholder_secrets(
    monkeypatch, environment
):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", environment)
    monkeypatch.setenv("XAGENT_JWT_SECRET", "   ")
    with pytest.raises(RuntimeError, match="XAGENT_JWT_SECRET"):
        auth_config.validate_auth_config()

    for placeholder in ("change-me", "replace-with-a-long-random-secret"):
        monkeypatch.setenv("XAGENT_JWT_SECRET", placeholder)
        with pytest.raises(RuntimeError, match="XAGENT_JWT_SECRET"):
            auth_config.validate_auth_config()


def test_non_development_rejects_unset_secret(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="XAGENT_JWT_SECRET"):
        auth_config.validate_auth_config()


def test_non_development_accepts_a_configured_secret(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("XAGENT_ENVIRONMENT", raising=False)
    monkeypatch.setenv(
        "XAGENT_JWT_SECRET", "a-real-secret-from-the-deployment-1234567890"
    )
    auth_config.validate_auth_config()


def test_secure_secret_override_is_enforced_in_development(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "development")
    monkeypatch.setenv("XAGENT_REQUIRE_SECURE_SECRETS", "true")
    monkeypatch.setenv("XAGENT_JWT_SECRET", "change-me")
    with pytest.raises(RuntimeError, match="XAGENT_JWT_SECRET"):
        auth_config.validate_auth_config()


def test_non_development_rejects_explicit_session_placeholder(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv(
        "XAGENT_JWT_SECRET", "a-real-secret-from-the-deployment-1234567890"
    )
    monkeypatch.setenv("XAGENT_SESSION_SECRET", "change-me-to-a-random-session-secret")
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        auth_config.validate_auth_config()


def test_non_development_rejects_short_secret(monkeypatch):
    monkeypatch.setenv("XAGENT_ENVIRONMENT", "production")
    monkeypatch.setenv("XAGENT_JWT_SECRET", "a" * 31)
    with pytest.raises(RuntimeError, match="XAGENT_JWT_SECRET"):
        auth_config.validate_auth_config()
