"""The JSON Schema subset accepted for structured output: checking, compiling
and judging.

:func:`check_schema_structure` runs the schema rules S1 to S13 in a fixed
order and returns the first violation as a :class:`SchemaViolation` (a rule id
and the path of the offending value), or ``None`` when the schema is
acceptable. It never raises for any JSON value and never echoes a value of the
schema: callers render the path with :func:`render_error_path`.

:func:`compile_schema` turns a schema that passed those rules into the nodes
used for judging, and :func:`model_facing_schema` writes the nullable copy
of the schema that the model is shown. :func:`judge_instance` and
:func:`judge_empty_answer` judge a value against the compiled schema and return
the outcome as a value: :class:`Ready`, :class:`Incomplete`, :class:`Err` or
:class:`TooComplex`. Compiling, copying and judging, like every other function
here, are pure: they neither mutate their arguments nor read or write anything
outside them.

Every judgement about a JSON value goes through :func:`json_kind` first. A
bool is never a number, ``0`` is never ``False``, and a tuple or a dict
subclass is never an array or an object. Name rules always use
``re.fullmatch`` so that a trailing newline cannot slip through. All walks use
an explicit stack; only the serialization in S1 and in
:func:`canonical_json_size` relies on the interpreter's recursion limit, and
S1 turns a ``RecursionError`` into a violation.
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
MAX_EXPANDED_NODES = 2000
MAX_EXPANDED_DEPTH = 32
MAX_EXPANDED_BYTES = 65536
MAX_EXPANDED_ENUM_VALUES = 5000
ALLOWED_VALUES_MAX_CHARS = 500
# The most positions one judgement may count: the instance positions it visits
# plus the positions that expanding empty required objects adds to the result.
MAX_JUDGED_STEPS = 200_000
# The most bytes of canonical JSON a judged result may take.
MAX_RESULT_BYTES = 2_097_152
# The most field paths a judgement lists for one reason.
MAX_LISTED_FIELDS = 20

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
    S9, S7 (every ``$defs`` entry is reachable), S10, S11, S12a, S12b and S13.
    A rule scans
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
    )
    if violation is not None:
        return violation
    expansion = measure_expansion(root_dict)
    if (
        expansion.nodes > MAX_EXPANDED_NODES
        or expansion.depth > MAX_EXPANDED_DEPTH
        or expansion.rewritten_bytes > MAX_EXPANDED_BYTES
    ):
        return SchemaViolation("S11")
    violation = _check_enums(root_dict)
    if violation is not None:
        return violation
    if expansion.enum_values > MAX_EXPANDED_ENUM_VALUES:
        return SchemaViolation("S12b")
    return _check_property_names(root_dict)


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


# --- S11, S12b: the size of the schema after $ref references are expanded ------


@dataclass(frozen=True)
class ExpansionMetrics:
    """What the schema measures once every ``$ref`` is replaced by its target.

    ``depth`` counts one level per nesting step and one more per ``$ref`` hop,
    with the root at 0. ``enum_values`` counts the values as the caller wrote
    them. ``rewritten_bytes`` is the size, as compact UTF-8 JSON, of the copy
    shown to the model.
    """

    nodes: int
    depth: int
    enum_values: int
    rewritten_bytes: int


def allowed_values_suffix(values: list[Any]) -> str:
    """The text appended to a description to list the values of its enum.

    Only the JSON text of the list is cut: past 500 characters (code points)
    it keeps the first 500 and adds an ellipsis. The leading space and the
    label are never counted.
    """
    text = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    if len(text) > ALLOWED_VALUES_MAX_CHARS:
        text = text[:ALLOWED_VALUES_MAX_CHARS] + "\u2026"
    return " Allowed values: " + text


def is_string_typed(node: dict[str, Any]) -> bool:
    """Whether a node's ``type`` is ``"string"`` or an array naming it."""
    declared = node.get("type")
    return declared == "string" or (type(declared) is list and "string" in declared)


def _children_of(node: dict[str, Any]) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    properties = node.get("properties")
    if type(properties) is dict:
        children.extend(properties.values())
    if "items" in node:
        children.append(node["items"])
    return children


def _expansion_dependencies(
    node: dict[str, Any], defs: dict[str, Any]
) -> list[dict[str, Any]]:
    if "$ref" in node:
        return [defs[node["$ref"][len(REF_PREFIX) :]]]
    if "anyOf" in node:
        return list(node["anyOf"])
    return _children_of(node)


def make_nullable(
    node: dict[str, Any], any_of: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The copy of ``node`` that also allows null; ``node`` is not changed.

    A single type ``T`` becomes ``[T, "null"]`` and an enum that lacks ``null``
    gains it, so a node that is nullable already comes back equal. A node must
    list an enum as ``enum``, never as ``const`` (see :func:`normalized_enum`).
    Given the ``anyOf`` node whose value branch ``node`` is, the union supplies
    the title unless ``node`` has one, and the two descriptions are joined by a
    space.
    """
    nullable = dict(node)
    if any_of is not None:
        if "title" not in node and "title" in any_of:
            nullable["title"] = any_of["title"]
        descriptions = [d["description"] for d in (any_of, node) if "description" in d]
        if descriptions:
            nullable["description"] = " ".join(descriptions)
    if type(nullable["type"]) is str:
        nullable["type"] = [nullable["type"], "null"]
    if "enum" in nullable:
        values = list(nullable["enum"])
        if not any(v is None for v in values):
            values.append(None)
        nullable["enum"] = values
    return nullable


def _rewritten_node(
    node: dict[str, Any], *, root: bool, any_of: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One node of the copy shown to the model, with every child replaced by
    the placeholder ``0``.

    Every node but the root is made nullable by :func:`make_nullable`. A string
    enum then gains the reserved marker, and a node with an enum lists its
    values at the end of its description. ``$defs`` is dropped and ``const``
    is written as an enum.
    """
    own: dict[str, Any] = {}
    for key, value in node.items():
        if key == "properties":
            own[key] = dict.fromkeys(value, 0)
        elif key == "items":
            own[key] = 0
        elif key not in ("$defs", "const"):
            own[key] = value
    enum = normalized_enum(node)
    if enum is not None:
        own["enum"] = enum
    if root:
        return own
    own = make_nullable(own, any_of)
    if enum is not None:
        if is_string_typed(own):
            own["enum"].append(NOT_EXPRESSIBLE_MARKER)
        own["description"] = own.get("description", "") + allowed_values_suffix(
            own["enum"]
        )
    return own


def _own_size(own: dict[str, Any]) -> int:
    """The size of a node of the copy with every child replaced by a single
    byte, minus those bytes: the node's own contribution to the copy."""
    children = len(own.get("properties", ())) + ("items" in own)
    return canonical_json_size(own) - children


def _own_bytes(
    node: dict[str, Any], *, root: bool, any_of: dict[str, Any] | None = None
) -> int:
    """The node's own contribution to the size of the copy shown to the model,
    which :func:`model_facing_schema` writes from the same rewritten node."""
    return _own_size(_rewritten_node(node, root=root, any_of=any_of))


def measure_expansion(schema: dict[str, Any]) -> ExpansionMetrics:
    """Measure the expansion of a schema that passed S1 to S10.

    One bottom-up pass over the nodes, each measured once however many times
    it is referenced: the node count, the depth and the enum values of a node
    follow from its children, and so does the size of its copy, because a
    node's own bytes are known without building the copy. Each node is
    rewritten once, on its own and with its children left out, and no
    reference is expanded, so a schema that expands to millions of nodes costs
    as much as its original size.
    """
    defs = _defs_of(schema)
    order: list[dict[str, Any]] = []
    seen = {id(schema)}
    stack = [(schema, iter(_expansion_dependencies(schema, defs)))]
    while stack:
        node, pending = stack[-1]
        for dependency in pending:
            if id(dependency) not in seen:
                seen.add(id(dependency))
                stack.append(
                    (dependency, iter(_expansion_dependencies(dependency, defs)))
                )
                break
        else:
            stack.pop()
            order.append(node)

    # id(node) -> (nodes, height, enum values, bytes of the copy)
    measured: dict[int, tuple[int, int, int, int]] = {}
    for node in order:
        if "$ref" in node:
            nodes, height, enums, size = measured[
                id(defs[node["$ref"][len(REF_PREFIX) :]])
            ]
            measured[id(node)] = (nodes, height + 1, enums, size)
        elif "anyOf" in node:
            branches = node["anyOf"]
            value = branches[1] if _is_null_branch(branches[0]) else branches[0]
            nodes, height, enums, _ = measured[id(value)]
            target = resolve_alias(value, defs)
            size = _own_bytes(target, root=False, any_of=node) + sum(
                measured[id(child)][3] for child in _children_of(target)
            )
            measured[id(node)] = (nodes + 2, height + 1, enums, size)
        else:
            child_measures = [measured[id(child)] for child in _children_of(node)]
            enum = normalized_enum(node)
            measured[id(node)] = (
                1 + sum(m[0] for m in child_measures),
                max((m[1] + 1 for m in child_measures), default=0),
                (len(enum) if enum is not None else 0)
                + sum(m[2] for m in child_measures),
                _own_bytes(node, root=node is schema)
                + sum(m[3] for m in child_measures),
            )
    nodes, height, enums, size = measured[id(schema)]
    return ExpansionMetrics(
        nodes=nodes, depth=height, enum_values=enums, rewritten_bytes=size
    )


# --- compiling ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledNode:
    """One position of the caller's schema, ready for judging.

    ``$ref`` is resolved (an alias chain to its end) and a nullable union is
    folded into its value node, so the same target used directly and inside a
    union gives two nodes that share their children. ``types`` is the node's
    ``type`` as written, as a list. ``enum_keys`` holds the canonical keys of
    the non-null enum values. ``single`` is the only allowed value of a node
    that is not nullable and lists exactly one non-null value (an integer node
    stores an integral float as an int), and ``single_bytes`` its canonical
    JSON size.

    The remaining fields describe what the node stands for when its value is
    empty. A nullable node is one null (1 node, 4 bytes) and a node with a
    ``single`` is that value (1 node, ``single_bytes``); neither has a
    ``template``. Otherwise ``missing_total`` counts the positions that stay
    unfilled below the node and ``missing_paths`` holds the first
    ``MAX_LISTED_FIELDS`` of them relative to the node, in document order. An
    object with nothing missing has a ``template``: its node count, its
    canonical JSON size and the entries that build it, ``(name, mode, value)``
    where mode 0 is a null, mode 1 the only allowed value ``value`` and mode 2
    the template of node number ``value``. Templates refer to each other by
    node number, so their storage grows with the number of nodes, not with the
    size of the expansion.
    """

    kind: str
    types: list[str]
    nullable: bool
    enum_keys: frozenset[tuple[Any, ...]] | None
    single: Any
    single_bytes: int
    properties: tuple[tuple[str, int, int], ...]
    items: int | None
    missing_total: int
    missing_paths: tuple[str, ...]
    template_nodes: int
    template_bytes: int
    template: tuple[tuple[str, int, Any], ...] | None


@dataclass(frozen=True, slots=True)
class CompiledSchema:
    """A schema that passed the structural checks, compiled for judging.

    ``schema`` is the caller's schema itself, kept for the copy shown to the
    model; ``nodes`` lists every compiled node with the children of a node
    before the node, and ``root`` is the number of the root node.
    """

    schema: dict[str, Any]
    nodes: tuple[CompiledNode, ...]
    root: int


def canonical_json_size(value: Any) -> int:
    """The size in bytes of the canonical JSON text of ``value``.

    The text is compact, keeps the key order, writes non-ASCII characters as
    they are and is encoded as UTF-8. It raises ``ValueError`` for a NaN or an
    infinity; the value must also hold no NUL and no lone surrogate.
    """
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return len(text.encode("utf-8"))


def _pointer_segment(name: str) -> str:
    return name.replace("~", "~0").replace("/", "~1")


def _position_of(
    node: dict[str, Any], defs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The node a schema position stands for and the nullable union that made
    it nullable, if any: references and aliases are followed and a union is
    read as its value node."""
    node = resolve_alias(node, defs)
    if "anyOf" in node:
        branches = node["anyOf"]
        value = branches[1] if _is_null_branch(branches[0]) else branches[0]
        return resolve_alias(value, defs), node
    return node, None


def _compile_node(
    target: dict[str, Any], from_union: bool, kids: list[int], nodes: list[CompiledNode]
) -> CompiledNode:
    declared = target["type"]
    types = list(declared) if type(declared) is list else [declared]
    kind = next(t for t in types if t != "null")
    nullable = from_union or "null" in types

    enum_keys = None
    single = None
    single_bytes = 0
    enum = normalized_enum(target)
    if enum is not None:
        values = [v for v in enum if v is not None]
        enum_keys = frozenset(canon(v) for v in values)
        if not nullable and len(values) == 1:
            single = values[0]
            if kind == "integer" and type(single) is float:
                single = int(single)
            single_bytes = canonical_json_size(single)

    declared_properties = target.get("properties")
    names = list(declared_properties) if type(declared_properties) is dict else []
    properties = tuple((name, len(name) + 2, kids[i]) for i, name in enumerate(names))
    items = kids[len(names)] if "items" in target else None

    missing_total = 0
    missing_paths: tuple[str, ...] = ()
    template_nodes = 0
    template_bytes = 0
    template = None
    if nullable:
        template_nodes, template_bytes = 1, 4
    elif kind == "object":
        paths: list[str] = []
        for name, _, index in properties:
            child = nodes[index]
            missing_total += child.missing_total
            for relative in child.missing_paths:
                if len(paths) < MAX_LISTED_FIELDS:
                    paths.append("/" + _pointer_segment(name) + relative)
        missing_paths = tuple(paths)
        if missing_total == 0:
            entries: list[tuple[str, int, Any]] = []
            template_nodes = 1
            template_bytes = 2 + max(len(properties) - 1, 0)
            for name, key_bytes, index in properties:
                child = nodes[index]
                template_nodes += child.template_nodes
                template_bytes += key_bytes + 1 + child.template_bytes
                if child.nullable:
                    entries.append((name, 0, None))
                elif child.kind == "object":
                    entries.append((name, 2, index))
                else:
                    entries.append((name, 1, child.single))
            template = tuple(entries)
    elif single is not None:
        template_nodes, template_bytes = 1, single_bytes
    else:
        missing_total, missing_paths = 1, ("",)

    return CompiledNode(
        kind=kind,
        types=types,
        nullable=nullable,
        enum_keys=enum_keys,
        single=single,
        single_bytes=single_bytes,
        properties=properties,
        items=items,
        missing_total=missing_total,
        missing_paths=missing_paths,
        template_nodes=template_nodes,
        template_bytes=template_bytes,
        template=template,
    )


def compile_schema(schema: dict[str, Any]) -> CompiledSchema:
    """Compile a schema that passed :func:`check_schema_structure`.

    Every position the root reaches becomes one node, built bottom-up so that
    a node is built once however often it is referenced. The schema is read,
    never changed.
    """
    defs = _defs_of(schema)

    def kids_of(target: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
        positions = [_position_of(child, defs) for child in _children_of(target)]
        return [(found, union is not None) for found, union in positions]

    nodes: list[CompiledNode] = []
    index_of: dict[tuple[int, bool], int] = {}
    root_target, root_node = _position_of(schema, defs)
    root_union = root_node is not None
    stack: list[list[Any]] = [[root_target, root_union, kids_of(root_target), 0]]
    while stack:
        frame = stack[-1]
        target, from_union, kids, position = frame
        if position < len(kids):
            frame[3] = position + 1
            kid_target, kid_union = kids[position]
            if (id(kid_target), kid_union) not in index_of:
                stack.append([kid_target, kid_union, kids_of(kid_target), 0])
            continue
        stack.pop()
        indexes = [index_of[(id(t), u)] for t, u in kids]
        index_of[(id(target), from_union)] = len(nodes)
        nodes.append(_compile_node(target, from_union, indexes, nodes))
    return CompiledSchema(
        schema=schema,
        nodes=tuple(nodes),
        root=index_of[(id(root_target), root_union)],
    )


# --- the copy shown to the model --------------------------------------------------


def model_facing_schema(compiled: CompiledSchema) -> dict[str, Any]:
    """The schema the model is shown: the caller's schema with every node but
    the root made nullable.

    The copy holds no ``$ref``, ``$defs`` or ``anyOf``; each reference is
    expanded with its own copy of the target, so no two positions share a list
    or a dict with each other or with the caller's schema, and a string enum
    lists the reserved marker. It only tells the model what it may write; a
    value is always judged against the caller's schema. The copy is built node
    by node, and ``ValueError`` is raised as soon as it passes
    ``MAX_EXPANDED_BYTES``, whatever the schema.
    """
    schema = compiled.schema
    defs = _defs_of(schema)
    limit = MAX_EXPANDED_BYTES
    spent = 0
    result: list[Any] = [None]
    work: list[tuple[Any, Any, dict[str, Any], bool]] = [(result, 0, schema, True)]
    while work:
        holder, key, position, is_root = work.pop()
        target, union = _position_of(position, defs)
        node = _rewritten_node(target, root=is_root, any_of=union)
        spent += _own_size(node)
        if spent > limit:
            raise ValueError("the copy of the schema is over the size limit")
        for name, value in node.items():
            if type(value) is list:
                node[name] = list(value)
        holder[key] = node
        if "items" in target:
            work.append((node, "items", target["items"], False))
        properties = target.get("properties")
        if type(properties) is dict:
            for name in reversed(list(properties)):
                work.append((node["properties"], name, properties[name], False))
    copied: dict[str, Any] = result[0]
    return copied


# --- judging ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ready:
    """A value that fills the caller's schema.

    ``output`` is built from the value, the schema's only allowed values and
    the templates of empty required objects; its keys follow the order of the
    schema's properties. The counts hold what was seen on the way: the gaps
    (always 0 here), the blank strings read as null, the instance positions
    visited and the positions added by expanding empty required objects.
    ``output_bytes`` is the size of ``output`` as canonical JSON, counted while
    the output is built.
    """

    output: Any
    missing_count: int
    unexpressible_count: int
    blank_as_null_count: int
    r1_steps: int
    expansion_nodes: int
    output_bytes: int


@dataclass(frozen=True, slots=True)
class Incomplete:
    """A value with gaps. ``missing_fields`` lists the positions with nothing
    written and ``unexpressible_fields`` those marked as not writable, each at
    most ``MAX_LISTED_FIELDS`` long and in document order, with the true totals
    in the counts. No output is kept."""

    missing_fields: list[str]
    unexpressible_fields: list[str]
    missing_count: int
    unexpressible_count: int
    blank_as_null_count: int
    r1_steps: int
    expansion_nodes: int


@dataclass(frozen=True, slots=True)
class Err:
    """A value that breaks the schema's shape: ``path`` is a JSON Pointer made
    from the schema's property names and list indices, ``rule`` one of
    ``type``, ``enum``, ``missing_property``, ``extra_property`` or ``marker``.
    The value itself is never reported."""

    path: str
    rule: str


@dataclass(frozen=True, slots=True)
class TooComplex:
    """A value too big to judge or to return. ``stage`` is ``"r1"`` when the
    shape check passed the step limit and ``"r4"`` when a result that is
    complete would pass ``MAX_JUDGED_STEPS`` positions or ``MAX_RESULT_BYTES``
    of canonical JSON; ``output_bytes`` is 0 for ``"r1"``."""

    stage: str
    r1_steps: int
    expansion_nodes: int
    output_bytes: int


class _Empty:
    """Nothing is written at a position."""


class _Unexpressible:
    """The marker is written at a position."""


class _Val:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _Kids:
    """An object or a list that is not empty: the reduced value of each
    property, in the order of the node's properties, or of each item."""

    __slots__ = ("kids",)

    def __init__(self, kids: list[Any]) -> None:
        self.kids = kids


_EMPTY = _Empty()
_UNEXPRESSIBLE = _Unexpressible()


class _Stop(Exception):
    """Ends the shape check with a verdict; it never leaves this module."""

    def __init__(self, verdict: Err | TooComplex) -> None:
        self.verdict = verdict


def _pointer_of(link: Any) -> str:
    """The JSON Pointer of a position kept as a chain of ``(segment, parent)``."""
    return "".join(
        "/" + (_pointer_segment(s) if type(s) is str else str(s)) for s in _unlink(link)
    )


def _reduce(
    nodes: tuple[CompiledNode, ...], root: int, instance: Any
) -> tuple[Any, int, int]:
    """The shape check and the emptiness of every position in one walk.

    Returns the reduced tree, the number of positions visited and the number of
    blank strings read as null, or raises :class:`_Stop` at the first error in
    document order. Each position reduces to ``_EMPTY``, ``_UNEXPRESSIBLE`` or a
    value; an object whose properties all reduce to ``_EMPTY`` is ``_EMPTY``,
    and so is a list with at least one item when every item is.
    """
    steps = 0
    blanks = 0

    def visit(index: int, value: Any, link: Any, is_root: bool) -> Any:
        nonlocal steps, blanks
        steps += 1
        if steps > MAX_JUDGED_STEPS:
            raise _Stop(TooComplex("r1", steps, 0, 0))
        if value is None:
            if is_root:
                raise _Stop(Err(_pointer_of(link), "type"))
            return _EMPTY
        node = nodes[index]
        if type(value) is str:
            if value.strip() == "":
                if is_root:
                    raise _Stop(Err(_pointer_of(link), "type"))
                blanks += 1
                return _EMPTY
            if NOT_EXPRESSIBLE_MARKER in value:
                if (
                    value == NOT_EXPRESSIBLE_MARKER
                    and node.kind == "string"
                    and not is_root
                ):
                    return _UNEXPRESSIBLE
                raise _Stop(Err(_pointer_of(link), "marker"))
        # Neither can be stored in a JSON column. The argument parser refuses
        # both before judging; this keeps any other caller from getting them
        # back in an output.
        if (type(value) is float and not math.isfinite(value)) or (
            type(value) is str and has_forbidden_char(value)
        ):
            raise _Stop(Err(_pointer_of(link), "type"))
        if not conforms_to_type(value, node.types):
            raise _Stop(Err(_pointer_of(link), "type"))
        if node.enum_keys is not None and canon(value) not in node.enum_keys:
            raise _Stop(Err(_pointer_of(link), "enum"))
        if node.kind == "object":
            for name, _, _ in node.properties:
                if name not in value:
                    raise _Stop(Err(_pointer_of((name, link)), "missing_property"))
            if len(value) != len(node.properties):
                raise _Stop(Err(_pointer_of(link), "extra_property"))
            return [node, value, link, 0, len(node.properties), [], True]
        if node.kind == "array":
            return [node, value, link, 0, len(value), [], False]
        return _Val(value)

    first = visit(root, instance, None, True)
    stack = [first]
    done: Any = None
    while stack:
        frame = stack[-1]
        node, value, link, position, count, results, is_object = frame
        if position < count:
            frame[3] = position + 1
            if is_object:
                name, _, child = node.properties[position]
                reduced = visit(child, value[name], (name, link), False)
            else:
                child = node.items
                reduced = visit(child, value[position], (position, link), False)
            if type(reduced) is list:
                stack.append(reduced)
                continue
            if is_object and type(reduced) is _Val and nodes[child].single is not None:
                reduced = _EMPTY
            results.append(reduced)
            continue
        stack.pop()
        if all(r is _EMPTY for r in results) and (is_object or count):
            done = _EMPTY
        else:
            done = _Kids(results)
        if stack:
            stack[-1][5].append(done)
    return done, steps, blanks


def _scalar_bytes(value: Any) -> int:
    """The canonical JSON size of a scalar of an output."""
    if value is None or value is True:
        return 4
    if value is False:
        return 5
    if type(value) is int:
        return len(str(value))
    if type(value) is float:
        return len(repr(value))
    return canonical_json_size(value)


def _copy_template(compiled: CompiledSchema, index: int) -> dict[str, Any]:
    """A fresh output for node ``index`` when it is empty, built from its
    template without recursion."""
    nodes = compiled.nodes
    built: dict[str, Any] = {}
    work = [(index, built)]
    while work:
        node_index, target = work.pop()
        template = nodes[node_index].template
        assert template is not None
        for name, mode, value in template:
            if mode == 2:
                child: dict[str, Any] = {}
                target[name] = child
                work.append((value, child))
            else:
                target[name] = value
    return built


def _decide(
    compiled: CompiledSchema, reduced: Any, steps: int, blanks: int
) -> Ready | Incomplete | TooComplex:
    """Turn a reduced tree into the output and the lists of gaps, then decide
    the status. An empty position is null when its node is nullable, an object
    of empty positions when it is an object, its only allowed value when it has
    one, and a gap otherwise; a marker is a position that cannot be written.
    The canonical JSON size of the output is added up position by position, a
    template counting for its precomputed size, so the output is never
    serialized."""
    nodes = compiled.nodes
    missing: list[str] = []
    unexpressible: list[str] = []
    missing_count = unexpressible_count = expansion_nodes = output_bytes = 0
    holder: list[Any] = [None]
    slots: list[tuple[Any, Any, int]] = []
    work: list[tuple[int, Any, Any, Any, Any]] = [
        (compiled.root, reduced, None, holder, 0)
    ]
    while work:
        index, item, link, parent, key = work.pop()
        node = nodes[index]
        if item is _EMPTY:
            if node.nullable:
                output_bytes += 4
                continue
            if node.kind == "object":
                if node.missing_total:
                    missing_count += node.missing_total
                    if len(missing) < MAX_LISTED_FIELDS:
                        prefix = _pointer_of(link)
                        room = MAX_LISTED_FIELDS - len(missing)
                        missing.extend(
                            prefix + rel for rel in node.missing_paths[:room]
                        )
                else:
                    expansion_nodes += node.template_nodes
                    output_bytes += node.template_bytes
                    slots.append((parent, key, index))
            elif node.single is not None:
                parent[key] = node.single
                output_bytes += node.single_bytes
            else:
                missing_count += 1
                if len(missing) < MAX_LISTED_FIELDS:
                    missing.append(_pointer_of(link))
        elif item is _UNEXPRESSIBLE:
            unexpressible_count += 1
            if len(unexpressible) < MAX_LISTED_FIELDS:
                unexpressible.append(_pointer_of(link))
        elif type(item) is _Kids and node.kind == "object":
            out = dict.fromkeys(name for name, _, _ in node.properties)
            parent[key] = out
            output_bytes += 2 + max(len(out) - 1, 0)
            output_bytes += sum(key_bytes + 1 for _, key_bytes, _ in node.properties)
            for (name, _, child), kid in reversed(
                list(zip(node.properties, item.kids))
            ):
                work.append((child, kid, (name, link), out, name))
        elif type(item) is _Kids:
            assert node.items is not None
            items_out: list[Any] = [None] * len(item.kids)
            parent[key] = items_out
            output_bytes += 2 + max(len(items_out) - 1, 0)
            for position in reversed(range(len(item.kids))):
                work.append(
                    (
                        node.items,
                        item.kids[position],
                        (position, link),
                        items_out,
                        position,
                    )
                )
        else:
            value = item.value
            if node.kind == "integer" and type(value) is float:
                value = int(value)
            parent[key] = value
            output_bytes += _scalar_bytes(value)
    if missing_count or unexpressible_count:
        return Incomplete(
            missing_fields=missing,
            unexpressible_fields=unexpressible,
            missing_count=missing_count,
            unexpressible_count=unexpressible_count,
            blank_as_null_count=blanks,
            r1_steps=steps,
            expansion_nodes=expansion_nodes,
        )
    if steps + expansion_nodes > MAX_JUDGED_STEPS or output_bytes > MAX_RESULT_BYTES:
        return TooComplex("r4", steps, expansion_nodes, output_bytes)
    for target, key, index in slots:
        target[key] = _copy_template(compiled, index)
    return Ready(
        output=holder[0],
        missing_count=0,
        unexpressible_count=0,
        blank_as_null_count=blanks,
        r1_steps=steps,
        expansion_nodes=expansion_nodes,
        output_bytes=output_bytes,
    )


def judge_instance(
    compiled: CompiledSchema, instance: Any
) -> Ready | Incomplete | Err | TooComplex:
    """Judge the value a model wrote against the caller's schema.

    The shape is checked first, in one walk, and the first error in document
    order is returned as :class:`Err`. A position holding null or a blank
    string counts as empty, an object is empty when every property is empty or
    holds its only allowed value, and a list is empty when it has items and all
    are empty. What each empty position stands for, and whether the result is
    :class:`Ready`, :class:`Incomplete` or :class:`TooComplex`, follows from
    the compiled nodes. A NaN, an infinity, or a string holding U+0000 or a
    surrogate code point is an :class:`Err` with rule ``type`` wherever it
    appears, since none of them can be stored as JSON. The instance must hold
    no integer too long for the interpreter to turn into text (``str`` raises
    ``ValueError`` past the interpreter's digit limit, 4300 by default). An
    integer decoded from JSON text under that limit never is, because the
    decoder refuses a longer literal, so only an integer built in Python can
    break this. The instance is read, never changed.
    """
    try:
        reduced, steps, blanks = _reduce(compiled.nodes, compiled.root, instance)
    except _Stop as stop:
        return stop.verdict
    return _decide(compiled, reduced, steps, blanks)


def judge_empty_answer(compiled: CompiledSchema) -> Ready | Incomplete | TooComplex:
    """Judge an answer that says nothing: the root is taken as empty, so the
    outcome is that of the same object written one level down as null."""
    return _decide(compiled, _EMPTY, 0, 0)
