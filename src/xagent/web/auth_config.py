import os

JWT_SECRET_KEY = os.getenv("XAGENT_JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("XAGENT_JWT_ALGORITHM", "HS256")


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
PASSWORD_MIN_LENGTH = _get_positive_int_from_env("XAGENT_PASSWORD_MIN_LENGTH", 6)
