"""Structural checks for the JSON Schema subset accepted for structured output.

:func:`check_schema_structure` is the single entry point. It runs the schema
rules S1 to S13 in a fixed order and returns the first violation as a
:class:`SchemaViolation` (a rule id and the path of the offending value), or
``None`` when the schema is acceptable. It never raises for any JSON value and
never echoes a value of the schema: callers render the path with
:func:`render_error_path`.

Every judgement about a JSON value goes through :func:`json_kind` first. A
bool is never a number, ``0`` is never ``False``, and a tuple or a dict
subclass is never an array or an object. Name rules always use
``re.fullmatch`` so that a trailing newline cannot slip through. All walks use
an explicit stack; only the serialization in S1 relies on the interpreter's
recursion limit, and it turns a ``RecursionError`` into a violation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

ERROR_PATH_MAX_CHARS = 2560

SCHEMA_MAX_BYTES = 16384
MAX_SCHEMA_NODES = 2000
MAX_SCHEMA_DEPTH = 32
MAX_ENUM_VALUES = 500
MAX_ENUM_STRING_CHARS = 256
MAX_TEXT_CHARS = 1024

# Rewritten schemas carry this marker as one more allowed value of a string
# enum, so a caller's own enum value can never be the marker.
NOT_EXPRESSIBLE_MARKER = "__NOT_EXPRESSIBLE__"

# Name rules are matched with ``re.fullmatch``; the patterns carry no anchors
# because ``$`` would also match before a trailing newline.
NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
PROPERTY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
DIGITS_PATTERN = re.compile(r"[0-9]+")
REF_PREFIX = "#/$defs/"
REF_PATTERN = re.compile(r"#/\$defs/[A-Za-z0-9_-]{1,64}")

_FORBIDDEN_CHAR = re.compile("[\x00\ud800-\udfff]")

_SCALAR_KINDS = frozenset({"null", "boolean", "integer", "float", "string"})
_TYPE_NAMES = frozenset({"string", "number", "integer", "boolean", "object", "array"})
_KEYWORDS = frozenset(
    {
        "type",
        "description",
        "title",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "enum",
        "const",
        "$ref",
        "$defs",
    }
)

PathSegment = str | int


@dataclass(frozen=True)
class SchemaViolation:
    """The first rule a schema breaks and where.

    ``path`` holds raw segments (property names, keywords, list indices); it
    is empty when the schema root itself is at fault. Render it with
    :func:`render_error_path` before it leaves the process.
    """

    rule: str
    path: tuple[PathSegment, ...] = ()


def json_kind(v: object) -> str:
    """The exact JSON type of a decoded value; a bool is never a number."""
    if v is None:
        return "null"
    t = type(v)
    if t is bool:
        return "boolean"
    if t is int:
        return "integer"
    if t is float:
        return "float"
    if t is str:
        return "string"
    if t is list:
        return "array"
    if t is dict:
        return "object"
    return "other"


def canon(v: object) -> tuple[Any, ...]:
    """The key two enum or const values are compared by.

    Equal JSON values have equal keys, ``1`` and ``1.0`` are equal, and a bool
    never equals a number. Arrays and objects have no key.
    """
    kind = json_kind(v)
    if kind == "null":
        return ("null",)
    if kind == "boolean":
        return ("bool", v)
    if kind == "integer":
        return ("num", str(v))
    if kind == "float":
        assert isinstance(v, float)
        if v.is_integer():
            return ("num", str(int(v)))
        return ("num", repr(v))
    if kind == "string":
        return ("str", v)
    raise ValueError(f"canon is not defined for a JSON {kind}")


def has_forbidden_char(s: str) -> bool:
    """Whether ``s`` holds U+0000 or a surrogate code point.

    Neither can be stored in a JSON column or encoded as UTF-8.
    """
    return _FORBIDDEN_CHAR.search(s) is not None


def render_error_path(segments: list[PathSegment]) -> str:
    """Render path segments as a JSON Pointer made of ASCII characters only.

    The path is capped at ``ERROR_PATH_MAX_CHARS`` characters, keeping its end
    and starting with ``...`` when it was cut. Each string segment is cut to
    its last ``ERROR_PATH_MAX_CHARS`` characters before it is escaped, and
    rendering stops once the segments already rendered are enough, so the
    memory used does not depend on the length of a caller's property name.
    """
    parts: list[str] = []
    total = 0
    for seg in reversed(segments):
        if isinstance(seg, int):
            part = str(seg)
        else:
            seg = seg[-ERROR_PATH_MAX_CHARS:]
            seg = seg.replace("~", "~0").replace("/", "~1")
            part = json.dumps(seg, ensure_ascii=True)[1:-1]
        parts.append(part)
        total += 1 + len(part)
        if total > ERROR_PATH_MAX_CHARS:
            break
    path = "".join("/" + p for p in reversed(parts))
    if len(path) > ERROR_PATH_MAX_CHARS:
        path = "..." + path[-(ERROR_PATH_MAX_CHARS - 3) :]
    return path


def normalized_enum(node: dict[str, Any]) -> list[Any] | None:
    """The enum of a node, with ``const: c`` read as ``enum: [c]``.

    ``None`` when the node has neither. A node that has both is rejected by
    S4 before anything reads this.
    """
    if "const" in node:
        return [node["const"]]
    enum = node.get("enum")
    return enum if type(enum) is list else None


def conforms_to_type(value: object, types: list[str]) -> bool:
    """Whether a value is accepted by a node whose ``type`` names ``types``."""
    kind = json_kind(value)
    if kind == "null":
        return "null" in types
    if kind == "boolean":
        return "boolean" in types
    if kind == "integer":
        return "integer" in types or "number" in types
    if kind == "float":
        assert isinstance(value, float)
        return "number" in types or ("integer" in types and value.is_integer())
    if kind == "string":
        return "string" in types
    if kind == "array":
        return "array" in types
    if kind == "object":
        return "object" in types
    return False


def check_schema_structure(schema: object) -> SchemaViolation | None:
    """Run S1 to S13 on the value of ``json_schema.schema``.

    The rules run in the order S2, S1, S3 to S6, S7 (names and placement), S8,
    S9, S7 (every ``$defs`` entry is reachable), S10, S12a and S13. A rule scans
    the whole schema before the next rule runs, and within a rule the first
    violation in document order (pre-order, keys in the order they were
    written) is reported.
    """
    root_dict = schema if type(schema) is dict else None
    violation = (
        _check_node_limits(schema)
        or _check_serialization(schema)
        or _check_keywords(schema)
        or _check_shapes(schema)
        or _check_object_and_array_keys(schema)
        or _check_root(schema)
    )
    if violation is not None or root_dict is None:
        return violation
    violation = (
        _check_defs_placement(root_dict)
        or _check_refs(root_dict)
        or _check_no_recursion(root_dict)
        or _check_defs_reachable(root_dict)
        or _check_nullable_anyof(root_dict)
        or _check_enums(root_dict)
        or _check_property_names(root_dict)
    )
    return violation


# --- traversal ---------------------------------------------------------------


def _walk(schema: object) -> Iterator[tuple[object, tuple[PathSegment, ...], bool]]:
    """Schema-position values in pre-order: ``(value, path, is_any_of_element)``.

    Children are the values of ``properties`` and ``$defs`` when those are
    objects, the value of ``items``, and the elements of ``anyOf`` when it is
    an array, visited in the order their keywords were written. A value that
    is not an object is yielded but has no children.
    """
    stack: list[tuple[object, tuple[PathSegment, ...], bool]] = [(schema, (), False)]
    while stack:
        value, path, in_any_of = stack.pop()
        yield value, path, in_any_of
        if type(value) is not dict:
            continue
        children: list[tuple[object, tuple[PathSegment, ...], bool]] = []
        for key, sub in value.items():
            if key in ("properties", "$defs") and type(sub) is dict:
                children.extend((v, path + (key, k), False) for k, v in sub.items())
            elif key == "items":
                children.append((sub, path + ("items",), False))
            elif key == "anyOf" and type(sub) is list:
                children.extend(
                    (v, path + ("anyOf", i), True) for i, v in enumerate(sub)
                )
        stack.extend(reversed(children))


def _dict_nodes(
    schema: object,
) -> Iterator[tuple[dict[str, Any], tuple[PathSegment, ...], bool]]:
    for value, path, in_any_of in _walk(schema):
        if type(value) is dict:
            yield value, path, in_any_of


def _node_types(node: dict[str, Any]) -> list[str]:
    declared = node.get("type")
    if type(declared) is str:
        return [declared]
    if type(declared) is list:
        return declared
    return []


# --- S2 ----------------------------------------------------------------------


def _check_node_limits(schema: object) -> SchemaViolation | None:
    """S2: at most 2000 schema nodes, none nested deeper than 32 levels.

    Only values in schema positions count. A value that is in a schema
    position but is not an object counts as one node and is not entered. The
    value of ``properties``, ``$defs`` or ``anyOf`` is not itself in a schema
    position, so when it has the wrong shape it is not counted at all and is
    left to S3. The contents of ``enum``, ``const``, ``description``,
    ``title`` and of unsupported keywords are never entered.
    """
    count = 0
    stack: list[tuple[object, tuple[PathSegment, ...], int]] = [(schema, (), 0)]
    while stack:
        value, path, depth = stack.pop()
        count += 1
        if count > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
            return SchemaViolation("S2", path)
        if type(value) is not dict:
            continue
        children: list[tuple[object, tuple[PathSegment, ...], int]] = []
        for key, sub in value.items():
            if key in ("properties", "$defs"):
                if type(sub) is dict:
                    children.extend(
                        (v, path + (key, k), depth + 1) for k, v in sub.items()
                    )
            elif key == "anyOf":
                if type(sub) is list:
                    children.extend(
                        (v, path + (key, i), depth + 1) for i, v in enumerate(sub)
                    )
            elif key == "items":
                children.append((sub, path + (key,), depth + 1))
        stack.extend(reversed(children))
    return None


# --- S1 ----------------------------------------------------------------------


def _check_serialization(schema: object) -> SchemaViolation | None:
    """S1: serializable, at most 16 KiB, and free of NUL and surrogates.

    The size is measured before the characters are checked, so the character
    check never walks more than 16 KiB of text. Unencodable lone surrogates
    are counted as three bytes by ``surrogatepass``, so measuring never fails.
    """
    try:
        text = json.dumps(
            schema, allow_nan=False, separators=(",", ":"), ensure_ascii=False
        )
    except (ValueError, RecursionError):
        return SchemaViolation("S1")
    if len(text.encode("utf-8", "surrogatepass")) > SCHEMA_MAX_BYTES:
        return SchemaViolation("S1")

    # Every key and string value, whether or not it is in a schema position.
    # A path is kept as a linked list so that a deep value costs nothing until
    # it is reported.
    stack: list[tuple[object, Any]] = [(schema, None)]
    while stack:
        value, link = stack.pop()
        if type(value) is str:
            if has_forbidden_char(value):
                return SchemaViolation("S1", _unlink(link))
        elif type(value) is dict:
            for key in value:
                if has_forbidden_char(key):
                    return SchemaViolation("S1", _unlink(link))
            stack.extend((v, (k, link)) for k, v in reversed(value.items()))
        elif type(value) is list:
            stack.extend((v, (i, link)) for i, v in reversed(list(enumerate(value))))
    return None


def _unlink(link: Any) -> tuple[PathSegment, ...]:
    segments: list[PathSegment] = []
    while link is not None:
        segment, link = link
        segments.append(segment)
    segments.reverse()
    return tuple(segments)


# --- S3 ----------------------------------------------------------------------


def _keyword_value_ok(key: str, value: object, in_any_of: bool) -> bool:
    if key == "type":
        if type(value) is str:
            return value in _TYPE_NAMES or (value == "null" and in_any_of)
        if type(value) is list:
            return (
                len(value) == 2
                and all(type(t) is str for t in value)
                and value.count("null") == 1
                and any(t in _TYPE_NAMES for t in value)
            )
        return False
    if key in ("description", "title"):
        return type(value) is str and len(value) <= MAX_TEXT_CHARS
    if key in ("properties", "$defs"):
        return type(value) is dict
    if key == "required":
        return type(value) is list and all(type(name) is str for name in value)
    if key == "additionalProperties":
        return value is False
    if key in ("anyOf", "enum"):
        return type(value) is list
    if key == "const":
        return json_kind(value) in _SCALAR_KINDS
    # ``items`` and ``$ref`` are judged at the child node and by S8.
    return key in _KEYWORDS


def _check_keywords(schema: object) -> SchemaViolation | None:
    """S3: only supported keywords, each with a value of the right type.

    A value in a schema position that is not an object is reported at its own
    path, after the keywords of its parent. The root is left to S6.
    """
    for value, path, in_any_of in _walk(schema):
        if type(value) is not dict:
            if path:
                return SchemaViolation("S3", path)
            continue
        for key, keyword_value in value.items():
            if not _keyword_value_ok(key, keyword_value, in_any_of):
                return SchemaViolation("S3", path + (key,))
    return None


# --- S4 ----------------------------------------------------------------------


def _shape_ok(node: dict[str, Any]) -> bool:
    if "anyOf" in node:
        return not any(k in node for k in ("type", "$ref", "const", "enum"))
    if "$ref" in node:
        return not any(k in node for k in ("type", "const", "enum"))
    if "type" not in node:
        return False
    if "const" in node and "enum" in node:
        return False
    enum = normalized_enum(node)
    if enum is None:
        return True
    types = _node_types(node)
    if not all(conforms_to_type(v, types) for v in enum):
        return False
    return "null" not in types or any(v is None for v in enum)


def _check_shapes(schema: object) -> SchemaViolation | None:
    """S4: every node is one of the supported shapes.

    The shapes are a lone ``anyOf``, a lone ``$ref``, a ``type`` (optionally
    with ``enum``), or a ``type`` with ``const``. Enum and const values must
    match the type, and a nullable type with an enum must list null.
    """
    for node, path, _ in _dict_nodes(schema):
        if not _shape_ok(node):
            return SchemaViolation("S4", path)
    return None


# --- S5 ----------------------------------------------------------------------


def _keys_ok(node: dict[str, Any]) -> bool:
    types = _node_types(node)
    if "object" in types:
        if not all(
            k in node for k in ("properties", "additionalProperties", "required")
        ):
            return False
        required = node["required"]
        if len(set(required)) != len(required):
            return False
        if set(required) != set(node["properties"]):
            return False
    elif any(k in node for k in ("properties", "required", "additionalProperties")):
        return False
    if "array" in types:
        return "items" in node
    return "items" not in node


def _check_object_and_array_keys(schema: object) -> SchemaViolation | None:
    """S5: an object node lists each property once in ``required``; an array
    node has ``items``; other nodes have neither set of keys."""
    for node, path, _ in _dict_nodes(schema):
        if not _keys_ok(node):
            return SchemaViolation("S5", path)
    return None


# --- S6 ----------------------------------------------------------------------


def _check_root(schema: object) -> SchemaViolation | None:
    """S6: the root is an object node whose ``type`` is the string "object"."""
    if type(schema) is not dict or schema.get("type") != "object":
        return SchemaViolation("S6")
    return None


# --- S7, S8, S9 --------------------------------------------------------------


def _defs_of(root: dict[str, Any]) -> dict[str, Any]:
    defs = root.get("$defs")
    return defs if type(defs) is dict else {}


def _check_defs_placement(root: dict[str, Any]) -> SchemaViolation | None:
    """S7, first part: ``$defs`` only at the root, with well-formed keys."""
    for node, path, _ in _dict_nodes(root):
        if "$defs" not in node:
            continue
        if path:
            return SchemaViolation("S7", path + ("$defs",))
        for key in node["$defs"]:
            if NAME_PATTERN.fullmatch(key) is None:
                return SchemaViolation("S7", ("$defs", key))
    return None


def _check_refs(root: dict[str, Any]) -> SchemaViolation | None:
    """S8: a ``$ref`` is the only key of its node and names an existing entry."""
    defs = _defs_of(root)
    for node, path, _ in _dict_nodes(root):
        if "$ref" not in node:
            continue
        ref = node["$ref"]
        if (
            len(node) != 1
            or type(ref) is not str
            or REF_PATTERN.fullmatch(ref) is None
            or ref[len(REF_PREFIX) :] not in defs
        ):
            return SchemaViolation("S8", path + ("$ref",))
    return None


def _reference_graph(root: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    """The entries referenced from outside ``$defs``, and the entries each
    entry references. Only valid after S8 has passed."""
    from_root: list[str] = []
    edges: dict[str, list[str]] = {name: [] for name in _defs_of(root)}
    for node, path, _ in _dict_nodes(root):
        if "$ref" not in node:
            continue
        target = node["$ref"][len(REF_PREFIX) :]
        if path[:1] == ("$defs",):
            edges[str(path[1])].append(target)
        else:
            from_root.append(target)
    return from_root, edges


def _entries_on_a_cycle(edges: dict[str, list[str]]) -> set[str]:
    """Entries that can reach themselves (strongly connected components of
    size above one, and self references), found with Tarjan's algorithm."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    component_stack: list[str] = []
    cyclic: set[str] = set()
    counter = 0
    for start in edges:
        if start in index:
            continue
        index[start] = low[start] = counter
        counter += 1
        component_stack.append(start)
        on_stack.add(start)
        work = [(start, iter(edges[start]))]
        while work:
            name, successors = work[-1]
            advanced = False
            for succ in successors:
                if succ not in index:
                    index[succ] = low[succ] = counter
                    counter += 1
                    component_stack.append(succ)
                    on_stack.add(succ)
                    work.append((succ, iter(edges[succ])))
                    advanced = True
                    break
                if succ in on_stack:
                    low[name] = min(low[name], index[succ])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[name])
            if low[name] == index[name]:
                component: list[str] = []
                while True:
                    member = component_stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == name:
                        break
                if len(component) > 1 or name in edges[name]:
                    cyclic.update(component)
    return cyclic


def _check_no_recursion(root: dict[str, Any]) -> SchemaViolation | None:
    """S9: the references between ``$defs`` entries contain no cycle."""
    _, edges = _reference_graph(root)
    cyclic = _entries_on_a_cycle(edges)
    for name in edges:
        if name in cyclic:
            return SchemaViolation("S9", ("$defs", name))
    return None


def _check_defs_reachable(root: dict[str, Any]) -> SchemaViolation | None:
    """S7, second part: every ``$defs`` entry is reached from the root through
    ``$ref``, so the size of the expansion is the size of the root's."""
    from_root, edges = _reference_graph(root)
    reached: set[str] = set()
    pending = list(from_root)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        pending.extend(edges[name])
    for name in edges:
        if name not in reached:
            return SchemaViolation("S7", ("$defs", name))
    return None


# --- S10 ----------------------------------------------------------------------


def _is_null_branch(branch: object) -> bool:
    return type(branch) is dict and len(branch) == 1 and branch.get("type") == "null"


def resolve_alias(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Follow ``$ref`` nodes to the first node that is not a reference."""
    while "$ref" in node:
        node = defs[node["$ref"][len(REF_PREFIX) :]]
    return node


def _check_nullable_anyof(root: dict[str, Any]) -> SchemaViolation | None:
    """S10: ``anyOf`` only spells "nullable": a ``{"type": "null"}`` branch and
    one branch whose node has a single non-null type."""
    defs = _defs_of(root)
    for node, path, _ in _dict_nodes(root):
        if "anyOf" not in node:
            continue
        branches = node["anyOf"]
        if len(branches) != 2 or sum(_is_null_branch(b) for b in branches) != 1:
            return SchemaViolation("S10", path)
        value_branch = branches[1] if _is_null_branch(branches[0]) else branches[0]
        value_type = resolve_alias(value_branch, defs).get("type")
        if type(value_type) is not str or value_type == "null":
            return SchemaViolation("S10", path)
    return None


# --- S12a, S13 ----------------------------------------------------------------


def _enum_ok(values: list[Any]) -> bool:
    if not 1 <= len(values) <= MAX_ENUM_VALUES:
        return False
    for value in values:
        kind = json_kind(value)
        if kind not in _SCALAR_KINDS:
            return False
        if kind == "float" and not math.isfinite(value):
            return False
        if kind == "string" and (
            len(value) > MAX_ENUM_STRING_CHARS
            or value.strip() == ""
            or NOT_EXPRESSIBLE_MARKER in value
        ):
            return False
    return len({canon(v) for v in values}) == len(values)


def _check_enums(root: dict[str, Any]) -> SchemaViolation | None:
    """S12a: each enum (a const counts as a one-value enum) has 1 to 500
    distinct scalar values, and its strings are neither blank nor the
    reserved marker."""
    for node, path, _ in _dict_nodes(root):
        values = normalized_enum(node)
        if values is not None and not _enum_ok(values):
            return SchemaViolation("S12a", path)
    return None


def _check_property_names(root: dict[str, Any]) -> SchemaViolation | None:
    """S13: property names are 1 to 64 characters of ``A-Za-z0-9_.-``, and are
    neither ``definitions``, ``-`` nor only digits."""
    for node, path, _ in _dict_nodes(root):
        properties = node.get("properties")
        if type(properties) is not dict:
            continue
        for name in properties:
            if (
                PROPERTY_NAME_PATTERN.fullmatch(name) is None
                or name == "definitions"
                or name == "-"
                or DIGITS_PATTERN.fullmatch(name) is not None
            ):
                return SchemaViolation("S13", path + ("properties", name))
    return None
