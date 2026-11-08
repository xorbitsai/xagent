"""The copy of a schema shown to the model, built the long way for tests.

Every reference and nullable union is expanded recursively, straight from the
caller's schema. Nothing here comes from the code under test (the module
imports only the marker constant from the shared random-schema helpers), so a
test that compares against it does not share a mistake with that code.
"""

from __future__ import annotations

import json
from typing import Any

from tests.shared.structured_schema_random import MARKER


def _allowed_values(values: list[Any]) -> str:
    text = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return " Allowed values: " + (text if len(text) <= 500 else text[:500] + "…")


def _expanded(
    node: dict[str, Any],
    defs: dict[str, Any],
    *,
    root: bool = False,
    any_of: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The copy handed to the model, built the long way: every node except the
    root is made nullable, references and nullable unions are expanded, and an
    enum lists its values in the description."""
    if "$ref" in node:
        return _expanded(defs[node["$ref"].removeprefix("#/$defs/")], defs)
    if "anyOf" in node:
        value_branch = next(b for b in node["anyOf"] if b != {"type": "null"})
        target = value_branch
        while "$ref" in target:
            target = defs[target["$ref"].removeprefix("#/$defs/")]
        return _expanded(target, defs, any_of=node)
    copy_: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties":
            copy_[key] = {k: _expanded(v, defs) for k, v in value.items()}
        elif key == "items":
            copy_[key] = _expanded(value, defs)
        elif key not in ("$defs", "const"):
            copy_[key] = value
    enum = [node["const"]] if "const" in node else node.get("enum")
    if root:
        if enum is not None:
            copy_["enum"] = enum
        return copy_
    if any_of is not None:
        if "title" in node or "title" in any_of:
            copy_["title"] = node.get("title", any_of.get("title"))
        descriptions = [d["description"] for d in (any_of, node) if "description" in d]
        if descriptions:
            copy_["description"] = " ".join(descriptions)
    if isinstance(copy_["type"], str):
        copy_["type"] = [copy_["type"], "null"]
    if enum is not None:
        values = list(enum)
        if None not in values:
            values.append(None)
        node_type = node["type"]
        if node_type == "string" or (
            isinstance(node_type, list) and "string" in node_type
        ):
            values.append(MARKER)
        copy_["enum"] = values
        copy_["description"] = copy_.get("description", "") + _allowed_values(values)
    return copy_


def expanded_copy(schema: dict[str, Any]) -> dict[str, Any]:
    """The copy of ``schema`` shown to the model."""
    return _expanded(schema, schema.get("$defs", {}), root=True)


def copy_size(schema: dict[str, Any]) -> int:
    """The size in bytes of the compact JSON text of the copy of ``schema``."""
    text = json.dumps(
        expanded_copy(schema),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return len(text.encode("utf-8"))
