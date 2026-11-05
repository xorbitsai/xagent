"""Random schemas inside the structured-output subset, and an independent view
of them for tests.

The generators draw from a seeded ``random.Random`` and use only the standard
library. :class:`RNode` is built here from the caller's schema without any help
from the code under test, so a test that reads it does not share a mistake
with that code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

STR_POOL = ["open", "closed", "approved", "rejected", "a", "b", "x"]


def gen_schema(rng: random.Random, max_depth: int = 4) -> dict[str, Any]:
    """A schema that passes the structural checks.

    Entries of ``$defs`` the root cannot reach are dropped, because every entry
    must be reachable.
    """
    defs: dict[str, Any] = {}
    plain_defs: list[str] = []  # usable as the value branch of a nullable union
    for i in range(rng.randint(0, 3)):
        name = f"d{i}"
        node = _gen_node(rng, 2, defs, plain_defs, allow_nullable=rng.random() < 0.3)
        defs[name] = node
        if isinstance(node.get("type"), str):
            plain_defs.append(name)
    root = _gen_object(rng, max_depth, defs, plain_defs, nprops=rng.randint(1, 5))
    reachable = _reachable_defs(root, defs)
    kept = {name: node for name, node in defs.items() if name in reachable}
    if kept:
        root["$defs"] = kept
    return root


def _refs_in(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        if "$ref" in node:
            found.append(node["$ref"].removeprefix("#/$defs/"))
        for value in node.values():
            found.extend(_refs_in(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_refs_in(value))
    return found


def _reachable_defs(root: dict[str, Any], defs: dict[str, Any]) -> set[str]:
    reached: set[str] = set()
    pending = _refs_in(root)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        pending.extend(_refs_in(defs[name]))
    return reached


def _maybe_desc(rng: random.Random, node: dict[str, Any]) -> dict[str, Any]:
    if rng.random() < 0.3:
        node["description"] = rng.choice(["Some field.", "Format: date.", "值"])
    if rng.random() < 0.1:
        node["title"] = "T"
    return node


def _gen_object(
    rng: random.Random,
    depth: int,
    defs: dict[str, Any],
    plain_defs: list[str],
    nprops: int | None = None,
) -> dict[str, Any]:
    if nprops is None:
        nprops = rng.choice([0, 1, 2, 2, 3])
    props = {
        f"p{i}": _gen_node(rng, depth - 1, defs, plain_defs, allow_nullable=True)
        for i in range(nprops)
    }
    return _maybe_desc(
        rng,
        {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        },
    )


def _gen_plain(
    rng: random.Random, depth: int, defs: dict[str, Any], plain_defs: list[str]
) -> dict[str, Any]:
    """A node with a single non-null type."""
    kinds = [
        "string",
        "integer",
        "number",
        "boolean",
        "senum",
        "ienum",
        "const",
        "bconst",
    ]
    if depth > 0:
        kinds += ["object", "object", "array", "array"]
    k = rng.choice(kinds)
    if k == "object":
        return _gen_object(rng, depth, defs, plain_defs)
    if k == "array":
        items = _gen_node(rng, depth - 1, defs, plain_defs, rng.random() < 0.3)
        return _maybe_desc(rng, {"type": "array", "items": items})
    if k == "senum":
        values = rng.sample(STR_POOL, rng.randint(1, 3))
        return _maybe_desc(rng, {"type": "string", "enum": values})
    if k == "ienum":
        values = rng.sample([1, 2, 3, 5], rng.randint(1, 3))
        return _maybe_desc(rng, {"type": "integer", "enum": values})
    if k == "const":
        return _maybe_desc(rng, {"type": "string", "const": rng.choice(STR_POOL)})
    if k == "bconst":
        return {"type": "boolean", "const": True}
    return _maybe_desc(rng, {"type": k})


def _gen_node(
    rng: random.Random,
    depth: int,
    defs: dict[str, Any],
    plain_defs: list[str],
    allow_nullable: bool,
) -> dict[str, Any]:
    if defs and rng.random() < 0.2:
        name = rng.choice(list(defs))
        ref = {"$ref": f"#/$defs/{name}"}
        if allow_nullable and name in plain_defs and rng.random() < 0.5:
            return _maybe_desc(rng, {"anyOf": [ref, {"type": "null"}]})
        return ref
    node = _gen_plain(rng, depth, defs, plain_defs)
    if allow_nullable and rng.random() < 0.4:
        if rng.random() < 0.5:
            node["type"] = [node["type"], "null"]
            if "enum" in node:
                node["enum"] = node["enum"] + [None]
            if "const" in node:
                node["enum"] = [node.pop("const"), None]
        else:
            return {"anyOf": [node, {"type": "null"}]}
    return node


# --- an independent view of a schema ------------------------------------------------


@dataclass
class RNode:
    """What a position of a schema allows, read straight from the schema."""

    kind: str
    nullable: bool = False
    enum_list: list[Any] | None = None  # the non-null enum values
    props: list[tuple[str, RNode]] = field(default_factory=list)
    items: RNode | None = None

    @property
    def single(self) -> bool:
        """Whether the node is not nullable and allows exactly one value."""
        return (
            not self.nullable
            and self.enum_list is not None
            and len(self.enum_list) == 1
        )


def build_rnode(
    raw: dict[str, Any], defs: dict[str, Any], nullable: bool = False
) -> RNode:
    """The :class:`RNode` of a position; ``defs`` is the root's ``$defs``."""
    while "$ref" in raw:
        raw = defs[raw["$ref"].removeprefix("#/$defs/")]
    if "anyOf" in raw:
        value = next(b for b in raw["anyOf"] if b != {"type": "null"})
        return build_rnode(value, defs, True)
    declared = raw["type"]
    types = declared if isinstance(declared, list) else [declared]
    node = RNode(
        kind=next(t for t in types if t != "null"),
        nullable=nullable or "null" in types,
    )
    values = [raw["const"]] if "const" in raw else raw.get("enum")
    if values is not None:
        node.enum_list = [v for v in values if v is not None]
    if node.kind == "object":
        node.props = [
            (name, build_rnode(child, defs))
            for name, child in raw["properties"].items()
        ]
    if node.kind == "array":
        node.items = build_rnode(raw["items"], defs)
    return node


def root_rnode(schema: dict[str, Any]) -> RNode:
    """The :class:`RNode` of a whole schema."""
    return build_rnode(schema, schema.get("$defs", {}))


# --- instances, decisions and writings ------------------------------------------------

MARKER = "__NOT_EXPRESSIBLE__"


def gen_value(rng: random.Random, n: RNode) -> Any:
    """A value the node allows."""
    if n.enum_list is not None:
        return rng.choice(n.enum_list)
    pools: dict[str, list[Any]] = {
        "string": ["Alice", "1 Main St", "2025-03-03", "x y"],
        "integer": [0, 1, 7, -3],
        "number": [0, 1.5, 2, -0.25],
        "boolean": [True, False],
    }
    return rng.choice(pools[n.kind])


def gen_instance(
    rng: random.Random, n: RNode, is_root: bool = True, p_bad: float = 0.04
) -> Any:
    """An instance that is mostly valid for the copy, sometimes broken."""
    r = rng.random()
    if not is_root:
        if r < 0.12:
            return None
        if r < 0.17:
            return rng.choice(["", "  ", "\t"])
        if r < 0.20 and n.kind == "string":
            return MARKER
    if rng.random() < p_bad:
        wrong = [MARKER, "x" + MARKER, 3, "zz", True, [], {}, None, 2.5, 1.0, 1]
        return rng.choice(wrong)
    if n.kind == "object":
        d = {k: gen_instance(rng, c, False, p_bad) for k, c in n.props}
        if n.props and rng.random() < p_bad:
            d.pop(rng.choice([k for k, _ in n.props]))
        if rng.random() < p_bad:
            d["zz_extra"] = 1
        return d
    if n.kind == "array":
        assert n.items is not None
        return [
            gen_instance(rng, n.items, False, p_bad) for _ in range(rng.randint(0, 3))
        ]
    return gen_value(rng, n)


# A decision is what the final answer says, whatever way it is written:
#   ("val", v) stated      ("ns",) not stated      ("nx",) stated, cannot be written
#   ("obj", [..]) ("arr", [..])      ("nonelist",) the answer says there are none
Decision = tuple[Any, ...]


def gen_decision(rng: random.Random, n: RNode, is_root: bool = False) -> Decision:
    if n.kind == "object":
        if not is_root and rng.random() < 0.25:
            return ("ns",)
        return ("obj", [gen_decision(rng, c) for _, c in n.props])
    if n.kind == "array":
        assert n.items is not None
        r = rng.random()
        if r < 0.2:
            return ("ns",)
        if r < 0.3:
            return ("nonelist",)
        return ("arr", [gen_decision(rng, n.items) for _ in range(rng.randint(1, 3))])
    r = rng.random()
    if r < 0.5:
        return ("val", gen_value(rng, n))
    return ("ns",) if r < 0.85 else ("nx",)


def render(
    rng: random.Random,
    d: Decision,
    n: RNode,
    in_object: bool = False,
    is_root: bool = False,
) -> Any:
    """One way of writing decision ``d``."""
    tag = d[0]
    if tag == "val":
        return d[1]
    if tag == "nx":
        return MARKER if n.kind == "string" else None
    if tag == "nonelist":
        return []
    if tag == "obj":
        return {k: render(rng, cd, c, True) for (k, c), cd in zip(n.props, d[1])}
    if tag == "arr":
        assert n.items is not None
        return [render(rng, cd, n.items) for cd in d[1]]
    return skeleton(rng, n) if is_root else render_ns(rng, n, in_object)


def render_ns(rng: random.Random, n: RNode, in_object: bool) -> Any:
    """Every writing of "the answer says nothing here"; inside an object a
    property with one allowed value may also be written as that value."""
    if n.kind == "object":
        return None if rng.random() < 0.5 else skeleton(rng, n)
    if n.kind == "array":
        assert n.items is not None
        if rng.random() < 0.5:
            return None
        return [render_ns(rng, n.items, False) for _ in range(rng.randint(1, 3))]
    choices: list[Any] = [None, None, "", "  "]
    if in_object and n.single:
        assert n.enum_list is not None
        choices.append(n.enum_list[0])
    return rng.choice(choices)


def skeleton(rng: random.Random, n: RNode) -> dict[str, Any]:
    return {k: render_ns(rng, c, True) for k, c in n.props}


def lookup(root: RNode, path: str) -> RNode:
    """The node at a JSON Pointer made of property names and indices."""
    n = root
    for seg in path.split("/")[1:]:
        n = dict(n.props)[seg] if n.kind == "object" else n.items  # type: ignore[assignment]
    return n


def synth_valid(n: RNode) -> Any:
    """Some value the node allows."""
    if n.enum_list is not None:
        return n.enum_list[0]
    if n.kind == "object":
        return {k: synth_valid(c) for k, c in n.props}
    if n.kind == "array":
        return []
    return {"string": "s", "integer": 1, "number": 1.5, "boolean": True}[n.kind]
