"""Single owner of the web sandbox lifecycle key format.

A user sandbox lifecycle key is ``user:{owner_id}`` for unscoped execution
and ``user:{owner_id}:{suffix}`` when an :class:`ExecutionScope` with a
``sandbox_key_suffix`` is active — one container family per scope under the
same platform user. The corresponding sandbox-manager lifecycle pair is
``("user", "{owner_id}")`` / ``("user", "{owner_id}:{suffix}")``.

Every composition and parse of this format must go through these helpers.
Keys are recorded once at sandbox build time and read back verbatim — never
re-derived from an owner id, because a reconstructed owner-only key would
silently miss a scope-suffixed sandbox.
"""

from __future__ import annotations

from typing import Optional

from ..core.execution_scope import validate_scope_component

USER_LIFECYCLE_TYPE = "user"
DURABLE_CHROME_LIFECYCLE_TYPE = "chrome-execution"


def make_user_lifecycle_id(owner_id: int, suffix: Optional[str] = None) -> str:
    """Compose the sandbox-manager lifecycle id for a user (+ scope suffix).

    Raises:
        InvalidScopeComponentError: ``suffix`` fails scope-component
            validation (which also guarantees it cannot contain ``:``).
    """
    if suffix is None:
        return str(int(owner_id))
    validate_scope_component(suffix, field_name="sandbox_key_suffix")
    return f"{int(owner_id)}:{suffix}"


def make_user_sandbox_key(owner_id: int, suffix: Optional[str] = None) -> str:
    """Compose the recorded sandbox lifecycle key: ``user:{owner}[:{suffix}]``."""
    return f"{USER_LIFECYCLE_TYPE}:{make_user_lifecycle_id(owner_id, suffix)}"


def parse_user_lifecycle_id(lifecycle_id: str) -> tuple[int, Optional[str]]:
    """Parse ``{owner_id}[:{suffix}]`` into ``(owner_id, suffix)``.

    One split; only the owner segment is ``int()``'d, so scoped ids parse
    instead of raising.

    Raises:
        ValueError: the owner segment is not an integer, or the suffix is
            empty (``"7:"``).
    """
    owner_segment, sep, suffix = lifecycle_id.partition(":")
    if sep and not suffix:
        raise ValueError(f"invalid user lifecycle id {lifecycle_id!r}: empty suffix")
    try:
        owner_id = int(owner_segment)
    except ValueError:
        raise ValueError(
            f"invalid user lifecycle id {lifecycle_id!r}: "
            f"owner segment {owner_segment!r} is not an integer"
        ) from None
    return owner_id, (suffix if sep else None)


def parse_user_sandbox_key(key: str) -> tuple[int, Optional[str]]:
    """Parse ``user:{owner_id}[:{suffix}]`` into ``(owner_id, suffix)``.

    At most two splits; only the owner segment is ``int()``'d.

    Raises:
        ValueError: the key does not have the ``user:`` prefix or its
            lifecycle id part is malformed.
    """
    lifecycle_type, sep, lifecycle_id = key.partition(":")
    if not sep or lifecycle_type != USER_LIFECYCLE_TYPE:
        raise ValueError(f"invalid user sandbox key {key!r}")
    try:
        return parse_user_lifecycle_id(lifecycle_id)
    except ValueError as exc:
        raise ValueError(f"invalid user sandbox key {key!r}: {exc}") from None
