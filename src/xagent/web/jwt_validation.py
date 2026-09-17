"""Side-effect-free JWT claim predicates shared by authentication owners."""

from typing import TypeGuard

from jose import JWTError, jwt

_SQLITE_INTEGER_MIN = -(2**63)
_SQLITE_INTEGER_MAX = 2**63 - 1
_POSTGRESQL_INTEGER_MIN = -(2**31)
_POSTGRESQL_INTEGER_MAX = 2**31 - 1


def has_matching_temporal_claim_conversion_failure(
    token: str, original_error: TypeError | OverflowError
) -> bool:
    """Whether unverified temporal data proves the decoder's exact failure."""
    try:
        claims = jwt.get_unverified_claims(token)
    except JWTError:
        return False

    for claim_name in ("exp", "nbf", "iat"):
        if claim_name not in claims:
            continue
        try:
            int(claims[claim_name])
        except ValueError:
            continue
        except (TypeError, OverflowError) as reproduced_error:
            if type(reproduced_error) is type(original_error):
                return True
    return False


def is_exact_integer_bindable(value: object, dialect_name: str) -> TypeGuard[int]:
    """Whether an exact integer fits a supported database parameter domain."""
    if type(value) is not int:
        return False
    if dialect_name == "sqlite":
        return _SQLITE_INTEGER_MIN <= value <= _SQLITE_INTEGER_MAX
    if dialect_name == "postgresql":
        return _POSTGRESQL_INTEGER_MIN <= value <= _POSTGRESQL_INTEGER_MAX
    return True
