"""Stable normalization for built-in catalog collision detection."""

from __future__ import annotations

from typing import Any


def canonicalize_builtin_identity(value: object) -> str | None:
    """Normalize an identity only for collision checks, never persistence."""
    if value is None:
        return None
    normalized = "-".join(str(value).strip().casefold().split())
    return normalized or None


def builtin_provenance_identity(value: Any) -> tuple[str, str] | None:
    """Return stable ownership identity, excluding schema/version metadata."""
    if not isinstance(value, dict):
        return None
    registry = value.get("registry")
    app_id = value.get("app_id")
    if not isinstance(registry, str) or not isinstance(app_id, str):
        return None
    if not registry or not app_id:
        return None
    return registry, app_id
