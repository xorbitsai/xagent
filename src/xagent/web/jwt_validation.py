"""Side-effect-free JWT claim predicates shared by authentication owners."""

from typing import TypeGuard

_SQLITE_INTEGER_MIN = -(2**63)
_SQLITE_INTEGER_MAX = 2**63 - 1
_POSTGRESQL_INTEGER_MIN = -(2**31)
_POSTGRESQL_INTEGER_MAX = 2**31 - 1


def is_exact_integer_bindable(value: object, dialect_name: str) -> TypeGuard[int]:
    """Whether an exact integer fits a supported database parameter domain."""
    if type(value) is not int:
        return False
    if dialect_name == "sqlite":
        return _SQLITE_INTEGER_MIN <= value <= _SQLITE_INTEGER_MAX
    if dialect_name == "postgresql":
        return _POSTGRESQL_INTEGER_MIN <= value <= _POSTGRESQL_INTEGER_MAX
    return True
