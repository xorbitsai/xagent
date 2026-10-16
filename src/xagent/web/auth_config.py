import os

from ..config import (
    INSECURE_SECRET_PLACEHOLDER,
    get_session_secret,
    requires_secure_secrets,
)

MIN_SECRET_LENGTH = 32

_INSECURE_JWT_SECRET_VALUES = {
    INSECURE_SECRET_PLACEHOLDER,
    "change-me",
    "secret",
    "replace-with-a-long-random-secret",
    "change-me-to-a-random-session-secret",
}
JWT_SECRET_KEY = os.getenv("XAGENT_JWT_SECRET", INSECURE_SECRET_PLACEHOLDER)
JWT_ALGORITHM = os.getenv("XAGENT_JWT_ALGORITHM", "HS256")


def _is_insecure_secret(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        len(normalized) < MIN_SECRET_LENGTH or normalized in _INSECURE_JWT_SECRET_VALUES
    )


def validate_auth_config() -> None:
    """Fail production startup when the JWT signing secret is not configured."""
    requires_secure_secret = requires_secure_secrets()
    # Read the environment again here rather than trusting the import-time JWT
    # constant. Startup validation must catch a missing deployment setting even
    # when another test or an embedding process imported this module earlier.
    configured_secret = os.getenv(
        "XAGENT_JWT_SECRET", INSECURE_SECRET_PLACEHOLDER
    ).strip()
    if requires_secure_secret and _is_insecure_secret(configured_secret):
        raise RuntimeError("XAGENT_JWT_SECRET must be set to a unique deployment value")
    session_secret = get_session_secret().strip()
    if requires_secure_secret and _is_insecure_secret(session_secret):
        raise RuntimeError(
            "XAGENT_SESSION_SECRET or XAGENT_JWT_SECRET must provide a unique "
            "session signing value"
        )


def _get_positive_int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
        if parsed <= 0:
            return default
        return parsed
    except ValueError:
        return default


ACCESS_TOKEN_EXPIRE_MINUTES = _get_positive_int_from_env(
    "XAGENT_ACCESS_TOKEN_EXPIRE_MINUTES", 120
)
REFRESH_TOKEN_EXPIRE_DAYS = _get_positive_int_from_env(
    "XAGENT_REFRESH_TOKEN_EXPIRE_DAYS", 7
)
PASSWORD_MIN_LENGTH = _get_positive_int_from_env("XAGENT_PASSWORD_MIN_LENGTH", 8)
