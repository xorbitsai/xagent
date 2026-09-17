"""Small shared datetime helpers for service modules."""

from __future__ import annotations

from datetime import datetime, timezone


def coerce_utc(value: datetime | None) -> datetime | None:
    """Read a DB datetime as aware UTC (SQLite returns naive UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
