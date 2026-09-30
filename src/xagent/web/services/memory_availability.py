"""Caller-safe persistent-memory availability metadata."""

from __future__ import annotations

from typing import Any

from ..memory_lifecycle import MemoryLifecycleState

# These lifecycle states are already part of the public store-info contract.
PUBLIC_MEMORY_AVAILABILITY_REASONS = frozenset(
    state.value for state in MemoryLifecycleState
)
GENERIC_MEMORY_AVAILABILITY_REASON = "unavailable"

MEMORY_AVAILABLE_METADATA_KEY = "memory_available"
MEMORY_AVAILABILITY_REASON_METADATA_KEY = "memory_availability_reason"


def public_memory_availability_reason(reason: Any) -> str | None:
    """Fold an availability reason onto something safe to publish."""
    if reason is None:
        return None
    if isinstance(reason, str) and reason in PUBLIC_MEMORY_AVAILABILITY_REASONS:
        return reason
    return GENERIC_MEMORY_AVAILABILITY_REASON


def caller_facing_execution_metadata(metadata: Any) -> dict[str, Any]:
    """Return a caller-safe copy of one execution-metadata mapping."""
    if not isinstance(metadata, dict):
        return {}
    folded = dict(metadata)
    if MEMORY_AVAILABILITY_REASON_METADATA_KEY in folded:
        folded[MEMORY_AVAILABILITY_REASON_METADATA_KEY] = (
            public_memory_availability_reason(
                folded[MEMORY_AVAILABILITY_REASON_METADATA_KEY]
            )
        )
    return folded


def caller_facing_trace_data(data: Any) -> Any:
    """Recursively fold memory reasons in data leaving the input untouched.

    Historical checkpoint events can carry execution metadata at several
    nesting levels. Public trace APIs therefore cannot safely sanitize only a
    known current shape: every mapping is copied and any server-owned memory
    reason is folded before the payload leaves the operator trace boundary.
    """
    if isinstance(data, dict):
        folded = {key: caller_facing_trace_data(value) for key, value in data.items()}
        if MEMORY_AVAILABILITY_REASON_METADATA_KEY in folded:
            folded[MEMORY_AVAILABILITY_REASON_METADATA_KEY] = (
                public_memory_availability_reason(
                    folded[MEMORY_AVAILABILITY_REASON_METADATA_KEY]
                )
            )
        return folded
    if isinstance(data, list):
        return [caller_facing_trace_data(value) for value in data]
    if isinstance(data, tuple):
        return tuple(caller_facing_trace_data(value) for value in data)
    return data
