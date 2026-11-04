"""Shared invariants used by startup and Alembic owner-aware migrations."""

from __future__ import annotations

OWNER_AWARE_UNIQUE_INDEX_DIALECTS = frozenset({"sqlite", "postgresql"})


def require_owner_aware_unique_index_dialect(dialect: object) -> str:
    """Return a supported dialect name or reject before schema inspection.

    Actor/ordinary builtin OAuth identity relies on partial unique indexes.
    Both application startup and the revision itself call this neutral helper
    so their supported-dialect contract and diagnostic cannot drift.
    """
    if not isinstance(dialect, str) or dialect not in OWNER_AWARE_UNIQUE_INDEX_DIALECTS:
        raise RuntimeError(
            "actor-owned builtin OAuth requires partial unique indexes; "
            f"database dialect {dialect!r} is unsupported"
        )
    return dialect
