"""Shared filter that turns raw values into a ``prompt_context`` value."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from ..schemas.agent import (
    PROMPT_CONTEXT_FIELD_NAME_PATTERN,
    PROMPT_CONTEXT_PRIMITIVES,
    PromptContextFieldList,
)
from .file_turn import (
    PROMPT_CONTEXT_MAX_DATA_BYTES,
    build_prompt_context_block,
    prompt_context_data_size,
)

_DECLARATIONS: TypeAdapter[Any] = TypeAdapter(PromptContextFieldList)
_FIELD_NAME_RE = re.compile(PROMPT_CONTEXT_FIELD_NAME_PATTERN)


def validate_signed_prompt_context(values: Any) -> None:
    """Reject a ``/v1`` prompt_context whose shape the filter would not store.

    Runs for every key. Signed (runtime key) values get no declarations
    whitelist, so this is what turns a value the filter would drop into a
    4xx. Unsigned (personal key) values are then also checked against the
    agent's declarations.
    """
    if not isinstance(values, Mapping):
        raise ValueError("prompt_context must be an object.")
    for name, value in values.items():
        if not isinstance(name, str) or not _FIELD_NAME_RE.fullmatch(name):
            raise ValueError(
                "prompt_context keys must match "
                f"{PROMPT_CONTEXT_FIELD_NAME_PATTERN} (at most 64 characters)."
            )
        if not isinstance(value, PROMPT_CONTEXT_PRIMITIVES):
            raise ValueError(
                "prompt_context values must be strings, numbers, or booleans."
            )
    if not values:
        return
    size = prompt_context_data_size(dict(values))
    if size is None:
        raise ValueError("prompt_context numbers must be finite.")
    if size > PROMPT_CONTEXT_MAX_DATA_BYTES:
        raise ValueError(
            f"prompt_context must be at most {PROMPT_CONTEXT_MAX_DATA_BYTES} bytes."
        )


def _validated_declarations(declarations: Any) -> list[Any] | None:
    try:
        return cast(
            "list[Any]", _DECLARATIONS.validate_python(declarations, strict=True)
        )
    except ValidationError:
        return None


def filter_prompt_context(
    values: Any, declarations: Any, *, signed: bool
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Filter raw values into a prompt_context value and name what it dropped.

    Returns the value (None if no field survives) and, in input order, every
    input field name the value does not carry, whatever the reason.
    """
    if not isinstance(values, Mapping):
        return None, ()
    validated = _validated_declarations(declarations) or []
    limits = None if signed else {item.name: item.max_length for item in validated}
    descriptions = {
        item.name: item.description
        for item in validated
        if item.description is not None
    }
    kept: dict[str, Any] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not isinstance(
            value, PROMPT_CONTEXT_PRIMITIVES
        ):
            continue
        if limits is not None:
            if name not in limits:
                continue
            try:
                length = len(str(value))
            except ValueError:
                continue
            if length > limits[name]:
                continue
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeEncodeError):
            continue
        kept[name] = value
    result: dict[str, Any] | None = None
    if kept:
        result = {"data": kept, "signed": signed}
        # Signed values stay unrestricted; declarations only supply meanings.
        meanings = {name: descriptions[name] for name in kept if name in descriptions}
        if meanings:
            result["descriptions"] = meanings
        if build_prompt_context_block(result) is None:
            result = None
    carried = result["data"] if result is not None else {}
    return result, tuple(str(name) for name in values if name not in carried)


def build_prompt_context_value(
    values: Any, declarations: Any, *, signed: bool
) -> dict[str, Any] | None:
    """Filter raw values into a prompt_context value, or None if none survive."""
    return filter_prompt_context(values, declarations, signed=signed)[0]
