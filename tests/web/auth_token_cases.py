"""Shared access-token rejection cases for authentication contract tests."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from jose import jwt

from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

_DEFAULT_LIFETIME = timedelta(minutes=5)


def _build_access_token(
    claims: dict[str, object],
    *,
    remove_claims: tuple[str, ...] = (),
    lifetime: timedelta = _DEFAULT_LIFETIME,
    signing_secret: str = JWT_SECRET_KEY,
) -> str:
    payload: dict[str, object] = {
        "type": "access",
        "sub": "existing-user",
        "user_id": 1,
        "exp": datetime.now(timezone.utc) + lifetime,
    }
    payload.update(claims)
    for claim in remove_claims:
        payload.pop(claim, None)
    return jwt.encode(payload, signing_secret, algorithm=JWT_ALGORITHM)


def build_access_token(*, remove_claims: tuple[str, ...] = (), **claims: object) -> str:
    """Build an access token whose default expiry starts at call time."""
    return _build_access_token(claims, remove_claims=remove_claims)


@dataclass(frozen=True, slots=True)
class RejectedAccessTokenCase:
    """One stable credential-rejection reason and its public HTTP contract."""

    id: str
    expected_detail: str
    expected_header_items: tuple[tuple[str, str], ...]
    claim_items: tuple[tuple[str, object], ...] = ()
    remove_claims: tuple[str, ...] = ()
    lifetime: timedelta = _DEFAULT_LIFETIME
    signing_secret: str = JWT_SECRET_KEY

    @property
    def expected_headers(self) -> dict[str, str]:
        return dict(self.expected_header_items)

    def build_token(self) -> str:
        """Build this rejection token with expiry relative to current time."""
        return _build_access_token(
            dict(self.claim_items),
            remove_claims=self.remove_claims,
            lifetime=self.lifetime,
            signing_secret=self.signing_secret,
        )


_BEARER_HEADERS = (("WWW-Authenticate", "Bearer"),)

REJECTED_ACCESS_TOKEN_CASES = (
    RejectedAccessTokenCase(
        "expired",
        "Token expired",
        _BEARER_HEADERS + (("Error-Type", "TokenExpired"),),
        lifetime=-_DEFAULT_LIFETIME,
    ),
    RejectedAccessTokenCase(
        "invalid-signature",
        "Invalid token",
        _BEARER_HEADERS + (("Error-Type", "InvalidToken"),),
        signing_secret="different-signing-secret",
    ),
    RejectedAccessTokenCase(
        "wrong-type",
        "Invalid token type",
        _BEARER_HEADERS,
        (("type", "refresh"),),
    ),
    RejectedAccessTokenCase(
        "invalid-claims",
        "Invalid token payload",
        _BEARER_HEADERS,
        (("user_id", "1"),),
    ),
    RejectedAccessTokenCase(
        "user-not-found",
        "User not found",
        _BEARER_HEADERS,
        (("sub", "missing-user"),),
    ),
)

WRONG_TYPE_ACCESS_TOKEN_CASE = REJECTED_ACCESS_TOKEN_CASES[2]
