"""Helpers for collision-aware tag mapping registration."""

from __future__ import annotations

import logging
from typing import Callable, Dict, TypeVar

ValueT = TypeVar("ValueT")


def register_tag_mapping(
    mapping: Dict[str, ValueT],
    tag: str,
    value: ValueT,
    *,
    get_identity: Callable[[ValueT], str],
    logger: logging.Logger,
) -> None:
    """Register a normalized tag mapping and warn on identity collisions.

    Args:
        mapping: Destination mapping keyed by normalized tag.
        tag: Normalized tag key.
        value: Value to store for the tag.
        get_identity: Function returning the logical identity used to detect
            collisions. For example, for ``tuple[str, Optional[int]]`` values it
            can return the first element (Hub model ID).
        logger: Logger used to emit collision warnings.
    """
    existing = mapping.get(tag)
    if existing is not None:
        existing_id = get_identity(existing)
        value_id = get_identity(value)
        if existing_id != value_id:
            logger.warning("Tag collision: %s -> %s vs %s", tag, existing_id, value_id)
            return
    mapping[tag] = value
