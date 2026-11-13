import os
from dataclasses import dataclass

from ..config import (
    INSECURE_SECRET_PLACEHOLDER,
    get_session_secret,
    requires_secure_secrets,
    validate_encryption_config,
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
    """Validate deployed signing, session and encryption keys at startup."""
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
    validate_encryption_config()
    # An enabled idle policy with an unusable value must stop the deployment
    # rather than fall back to a weaker default.
    get_browser_idle_policy()


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


BROWSER_IDLE_TIMEOUT_ENABLED_ENV = "XAGENT_BROWSER_IDLE_TIMEOUT_ENABLED"
BROWSER_IDLE_TIMEOUT_MINUTES_ENV = "XAGENT_BROWSER_IDLE_TIMEOUT_MINUTES"
BROWSER_IDLE_TIMEOUT_ADMIN_MINUTES_ENV = "XAGENT_BROWSER_IDLE_TIMEOUT_ADMIN_MINUTES"
_ENABLED_VALUES = {"1", "true", "yes", "on"}
# One year: far above any sensible idle threshold, far below datetime overflow.
_MAX_IDLE_MINUTES = 365 * 24 * 60
_DISABLED_VALUES = {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class BrowserIdlePolicy:
    """Idle limits for interactive browser logins; see docs/browser-idle-timeout.md."""

    enabled: bool
    ordinary_minutes: int
    admin_minutes: int

    def timeout_seconds(self, is_platform_admin: bool) -> int:
        minutes = self.admin_minutes if is_platform_admin else self.ordinary_minutes
        return minutes * 60


def _idle_minutes_from_env(name: str, default: int, *, strict: bool) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if 0 < parsed <= _MAX_IDLE_MINUTES:
        return parsed
    if strict:
        raise RuntimeError(
            f"{name} must be a whole number of minutes between 1 and "
            f"{_MAX_IDLE_MINUTES}"
        )
    return default


def get_browser_idle_policy() -> BrowserIdlePolicy:
    """Read the policy on each call so tests and startup checks see the current env.

    Raises RuntimeError when the switch is unrecognised, or when it is on and a
    threshold is unusable: an enabled control must not silently weaken.
    """
    switch = os.getenv(BROWSER_IDLE_TIMEOUT_ENABLED_ENV, "").strip().lower()
    if switch in _ENABLED_VALUES:
        enabled = True
    elif switch in _DISABLED_VALUES:
        enabled = False
    else:
        raise RuntimeError(f"{BROWSER_IDLE_TIMEOUT_ENABLED_ENV} must be true or false")
    return BrowserIdlePolicy(
        enabled=enabled,
        ordinary_minutes=_idle_minutes_from_env(
            BROWSER_IDLE_TIMEOUT_MINUTES_ENV, 30, strict=enabled
        ),
        admin_minutes=_idle_minutes_from_env(
            BROWSER_IDLE_TIMEOUT_ADMIN_MINUTES_ENV, 15, strict=enabled
        ),
    )
