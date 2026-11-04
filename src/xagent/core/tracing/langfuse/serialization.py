"""Serialization helpers for Langfuse payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def serialize_for_langfuse(value: Any) -> Any:
    """Best-effort conversion of trace payloads into JSON-safe data."""
    if hasattr(value, "model_dump"):
        return serialize_for_langfuse(value.model_dump())

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): serialize_for_langfuse(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [serialize_for_langfuse(v) for v in value]

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes:{len(value)}>"

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def coerce_usage_details(value: Any) -> Optional[dict[str, int]]:
    """Convert model usage payloads to Langfuse integer counters."""
    if not isinstance(value, dict):
        return None

    usage_details: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            usage_details[str(key)] = int(raw)

    return usage_details or None
