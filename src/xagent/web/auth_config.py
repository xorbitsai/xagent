import os

_INSECURE_JWT_SECRET = "your-secret-key-change-in-production"
JWT_SECRET_KEY = os.getenv("XAGENT_JWT_SECRET", _INSECURE_JWT_SECRET)
JWT_ALGORITHM = os.getenv("XAGENT_JWT_ALGORITHM", "HS256")


def validate_auth_config() -> None:
    """Fail production startup when the JWT signing secret is not configured."""
    environment = (
        os.getenv("XAGENT_ENVIRONMENT") or os.getenv("ENVIRONMENT") or "development"
    ).strip().lower()
    requires_secure_secret = environment in {"prod", "production", "stage", "staging"}
    requires_secure_secret = requires_secure_secret or os.getenv(
        "XAGENT_REQUIRE_SECURE_SECRETS", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if requires_secure_secret and (
        not JWT_SECRET_KEY or JWT_SECRET_KEY == _INSECURE_JWT_SECRET
    ):
        raise RuntimeError(
            "XAGENT_JWT_SECRET must be set to a unique value before starting "
            f"the {environment} environment"
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
