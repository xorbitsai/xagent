import os

from ..config import (
    INSECURE_SECRET_PLACEHOLDER,
    get_environment,
    get_require_secure_secrets,
    get_session_secret,
)

_INSECURE_JWT_SECRET = INSECURE_SECRET_PLACEHOLDER
_INSECURE_JWT_SECRET_VALUES = {
    _INSECURE_JWT_SECRET,
    "change-me",
    "secret",
    "replace-with-a-long-random-secret",
    "change-me-to-a-random-session-secret",
}
JWT_SECRET_KEY = os.getenv("XAGENT_JWT_SECRET", _INSECURE_JWT_SECRET)
JWT_ALGORITHM = os.getenv("XAGENT_JWT_ALGORITHM", "HS256")


def validate_auth_config() -> None:
    """Fail production startup when the JWT signing secret is not configured."""
    environment = get_environment()
    requires_secure_secret = environment != "development"
    requires_secure_secret = requires_secure_secret or get_require_secure_secrets()
    configured_secret = os.getenv("XAGENT_JWT_SECRET", JWT_SECRET_KEY).strip()
    if requires_secure_secret and (
        not configured_secret
        or configured_secret.lower() in _INSECURE_JWT_SECRET_VALUES
    ):
        raise RuntimeError("XAGENT_JWT_SECRET must be set to a unique deployment value")
    session_secret = get_session_secret().strip()
    if requires_secure_secret and (
        not session_secret or session_secret.lower() in _INSECURE_JWT_SECRET_VALUES
    ):
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
