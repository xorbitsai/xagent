"""Stable normalization for built-in catalog collision detection."""

from __future__ import annotations

from collections.abc import Mapping
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


def owned_catalog_marker(app_info: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the catalog app's provenance marker when it names this exact app.

    The marker is ``launch_config["builtin_provenance"]``. It counts only when it
    is a dict whose identity is ``("xagent", app_info["id"])``; anything else,
    including a marker for another app or registry, yields ``None``. The catalog
    OAuth callback copies this marker into the server's auth, and the canonical
    builtin OAuth server check accepts a stored marker only against it, so both
    sides share one definition of the app's own marker.
    """
    launch_config = app_info.get("launch_config")
    if not isinstance(launch_config, dict):
        return None
    marker = launch_config.get("builtin_provenance")
    if not isinstance(marker, dict):
        return None
    if builtin_provenance_identity(marker) != ("xagent", str(app_info["id"])):
        return None
    return marker
