"""Compiling a checked schema and judging a value against it."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from xagent.core.agent import structured_schema
from xagent.core.agent.structured_schema import check_schema_structure

STRING: dict[str, Any] = {"type": "string"}


def _obj(properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    node.update(extra)
    return node


def _ref(name: str) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{name}"}


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [node, {"type": "null"}]}


def _compile(schema: dict[str, Any]) -> Any:
    assert check_schema_structure(schema) is None
    return structured_schema.compile_schema(schema)


def _root(compiled: Any) -> Any:
    return compiled.nodes[compiled.root]


def _child(compiled: Any, node: Any, name: str) -> Any:
    for property_name, _key_bytes, index in node.properties:
        if property_name == name:
            return compiled.nodes[index]
    raise KeyError(name)


def _items(compiled: Any, node: Any) -> Any:
    return compiled.nodes[node.items]


# --- the node of each position ---------------------------------------------------


def test_a_node_records_its_kind_types_and_nullability() -> None:
    compiled = _compile(
        _obj(
            {
                "a": STRING,
                "b": {"type": ["string", "null"]},
                "c": {"type": ["null", "integer"]},
                "d": _nullable({"type": "number"}),
                "e": {"type": "array", "items": STRING},
            }
        )
    )
    root = _root(compiled)

    assert (root.kind, root.nullable) == ("object", False)
    a = _child(compiled, root, "a")
    assert (a.kind, a.types, a.nullable) == ("string", ["string"], False)
    b = _child(compiled, root, "b")
    assert (b.kind, b.nullable) == ("string", True)
    c = _child(compiled, root, "c")
    assert (c.kind, c.nullable) == ("integer", True)
    d = _child(compiled, root, "d")
    assert (d.kind, d.types, d.nullable) == ("number", ["number"], True)
    e = _child(compiled, root, "e")
    assert (e.kind, e.nullable) == ("array", False)
    assert _items(compiled, e).kind == "string"


def test_properties_keep_their_written_order_and_the_size_of_their_quoted_key() -> None:
    compiled = _compile(_obj({"zeta": STRING, "a": STRING, "mid.k": STRING}))
    root = _root(compiled)

    assert [(name, key_bytes) for name, key_bytes, _ in root.properties] == [
        ("zeta", 6),
        ("a", 3),
        ("mid.k", 7),
    ]


def test_an_enum_is_stored_as_canonical_keys_without_null() -> None:
    compiled = _compile(
        _obj(
            {
                "a": {"type": "integer", "enum": [1, 2.0, 3]},
                "b": {"type": ["string", "null"], "enum": ["x", None]},
                "c": STRING,
            }
        )
    )
    root = _root(compiled)

    assert _child(compiled, root, "a").enum_keys == frozenset(
        {("num", "1"), ("num", "2"), ("num", "3")}
    )
    assert _child(compiled, root, "b").enum_keys == frozenset({("str", "x")})
    assert _child(compiled, root, "c").enum_keys is None


# --- the only allowed value ------------------------------------------------------


@pytest.mark.parametrize(
    ("node", "value", "size"),
    [
        ({"type": "boolean", "const": False}, False, 5),
        ({"type": "boolean", "const": True}, True, 4),
        ({"type": "integer", "const": 0}, 0, 1),
        ({"type": "number", "const": 0.0}, 0.0, 3),
        ({"type": "string", "const": "x"}, "x", 3),
        ({"type": "string", "enum": ["é"]}, "é", 4),
    ],
)
def test_a_falsy_only_allowed_value_is_still_recorded(
    node: dict[str, Any], value: Any, size: int
) -> None:
    compiled = _compile(_obj({"a": node}))
    single = _child(compiled, _root(compiled), "a")

    assert single.single == value
    assert type(single.single) is type(value)
    assert single.single_bytes == size


def test_a_nullable_node_has_no_only_allowed_value() -> None:
    compiled = _compile(
        _obj(
            {
                "a": {"type": ["string", "null"], "enum": ["x", None]},
                "b": _nullable({"type": "string", "const": "x"}),
                "c": {"type": ["null", "string"], "enum": [None, "x"]},
            }
        )
    )
    root = _root(compiled)

    for name in ("a", "b", "c"):
        assert _child(compiled, root, name).single is None


def test_a_node_with_two_values_has_no_only_allowed_value() -> None:
    compiled = _compile(_obj({"a": {"type": "string", "enum": ["x", "y"]}}))

    assert _child(compiled, _root(compiled), "a").single is None


def test_an_integer_only_allowed_value_written_as_a_float_is_stored_as_an_int() -> None:
    compiled = _compile(
        _obj(
            {
                "a": {"type": "integer", "const": 1.0},
                "b": {"type": "integer", "const": 1e308},
                "c": {"type": "number", "const": 1.0},
                "d": {"type": "integer", "const": -0.0},
            }
        )
    )
    root = _root(compiled)

    a, b, c, d = (_child(compiled, root, name) for name in "abcd")
    assert (type(a.single), a.single, a.single_bytes) == (int, 1, 1)
    assert type(b.single) is int
    assert b.single == int(1e308)
    assert b.single_bytes == len(str(int(1e308))) == 309
    assert (type(c.single), c.single, c.single_bytes) == (float, 1.0, 3)
    assert (type(d.single), d.single, d.single_bytes) == (int, 0, 1)


# --- references ------------------------------------------------------------------


def test_an_alias_chain_resolves_to_its_end_and_gets_no_node_of_its_own() -> None:
    schema = _obj(
        {"a": _ref("A"), "b": _ref("B")},
        **{"$defs": {"A": _ref("B"), "B": _ref("C"), "C": STRING}},
    )
    compiled = _compile(schema)
    root = _root(compiled)

    assert len(compiled.nodes) == 2
    assert _child(compiled, root, "a") is _child(compiled, root, "b")


def test_a_target_used_directly_and_as_a_nullable_union_gets_two_nodes() -> None:
    target = _obj({"x": STRING, "y": {"type": "integer"}})
    compiled = _compile(
        _obj(
            {"direct": _ref("T"), "maybe": _nullable(_ref("T"))},
            **{"$defs": {"T": target}},
        )
    )
    root = _root(compiled)
    direct = _child(compiled, root, "direct")
    maybe = _child(compiled, root, "maybe")

    assert (direct.nullable, maybe.nullable) == (False, True)
    assert direct is not maybe
    assert [i for _, _, i in direct.properties] == [i for _, _, i in maybe.properties]
    assert len(compiled.nodes) == 5


def test_two_unions_over_the_same_target_share_one_node() -> None:
    compiled = _compile(
        _obj(
            {"a": _nullable(_ref("T")), "b": _nullable(_ref("T"))},
            **{"$defs": {"T": STRING}},
        )
    )
    root = _root(compiled)

    assert _child(compiled, root, "a") is _child(compiled, root, "b")


def test_a_union_over_an_alias_chain_reads_the_end_of_the_chain() -> None:
    compiled = _compile(
        _obj(
            {"a": _nullable(_ref("A"))},
            **{"$defs": {"A": _ref("B"), "B": {"type": "string", "const": "k"}}},
        )
    )
    node = _child(compiled, _root(compiled), "a")

    assert (node.kind, node.nullable, node.single) == ("string", True, None)
    assert node.enum_keys == frozenset({("str", "k")})


def test_a_target_that_is_itself_a_nullable_union_is_a_nullable_node() -> None:
    compiled = _compile(
        _obj(
            {"a": _ref("N"), "b": {"type": "array", "items": _ref("N")}},
            **{"$defs": {"N": _nullable(STRING)}},
        )
    )
    root = _root(compiled)

    assert _child(compiled, root, "a").nullable is True
    assert _items(compiled, _child(compiled, root, "b")).nullable is True


# --- what an empty position expands to ---------------------------------------------


def test_a_nullable_child_is_one_node_of_four_bytes_and_is_not_expanded() -> None:
    inner = _obj({"x": STRING})
    compiled = _compile(
        _obj(
            {
                "n": {**inner, "type": ["object", "null"]},
                "u": _nullable(_ref("T")),
            },
            **{"$defs": {"T": inner}},
        )
    )
    root = _root(compiled)

    assert root.missing_total == 0
    assert root.template_nodes == 3
    assert root.template_bytes == len('{"n":null,"u":null}')
    for name in ("n", "u"):
        node = _child(compiled, root, name)
        assert (node.template_nodes, node.template_bytes, node.missing_total) == (
            1,
            4,
            0,
        )


def test_an_only_allowed_value_child_is_one_node_of_its_own_size() -> None:
    compiled = _compile(
        _obj(
            {
                "kind": {"type": "string", "const": "person"},
                "ok": {"type": "boolean", "const": False},
            }
        )
    )
    root = _root(compiled)

    assert root.missing_total == 0
    assert root.template_nodes == 3
    assert root.template_bytes == len('{"kind":"person","ok":false}')
    assert _child(compiled, root, "kind").template_nodes == 1
    assert _child(compiled, root, "kind").template_bytes == len('"person"')


def test_a_required_object_child_keeps_expanding() -> None:
    compiled = _compile(
        _obj({"a": _obj({"b": _obj({"c": {"type": ["integer", "null"]}})})})
    )
    root = _root(compiled)

    assert root.template_nodes == 4
    assert root.template_bytes == len('{"a":{"b":{"c":null}}}')
    assert _child(compiled, root, "a").template_nodes == 3


def test_an_object_without_properties_expands_to_an_empty_object() -> None:
    compiled = _compile(_obj({"e": _obj({})}))

    assert _child(compiled, _root(compiled), "e").template_nodes == 1
    assert _child(compiled, _root(compiled), "e").template_bytes == 2


@pytest.mark.parametrize(
    "node",
    [
        STRING,
        {"type": "integer"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "string", "enum": ["a", "b"]},
        {"type": "array", "items": STRING},
    ],
)
def test_a_required_scalar_or_array_child_is_one_missing_value_at_its_own_path(
    node: dict[str, Any],
) -> None:
    compiled = _compile(_obj({"a": node}))
    child = _child(compiled, _root(compiled), "a")

    assert (child.missing_total, child.missing_paths) == (1, ("",))
    assert child.template is None
    assert _root(compiled).missing_paths == ("/a",)


def test_missing_paths_follow_document_order_through_nested_objects() -> None:
    compiled = _compile(
        _obj(
            {
                "a": _obj(
                    {"x": STRING, "y": {"type": ["string", "null"]}, "z": STRING}
                ),
                "b": {"type": "integer"},
                "c": _ref("T"),
            },
            **{"$defs": {"T": _obj({"deep": STRING})}},
        )
    )
    root = _root(compiled)

    assert root.missing_total == 4
    assert root.missing_paths == ("/a/x", "/a/z", "/b", "/c/deep")
    assert root.template is None


def test_twenty_five_missing_fields_are_counted_and_only_the_first_twenty_kept() -> (
    None
):
    compiled = _compile(_obj({f"f{i:02d}": STRING for i in range(25)}))
    root = _root(compiled)

    assert root.missing_total == 25
    assert root.missing_paths == tuple(f"/f{i:02d}" for i in range(20))


def test_the_total_keeps_counting_after_the_kept_paths_are_full() -> None:
    inner = _obj({f"f{i:02d}": STRING for i in range(25)})
    compiled = _compile(
        _obj({"a": _ref("D"), "b": _ref("D"), "z": STRING}, **{"$defs": {"D": inner}})
    )
    root = _root(compiled)

    assert root.missing_total == 51
    assert len(root.missing_paths) == 20
    assert root.missing_paths[0] == "/a/f00"
    assert root.missing_paths[-1] == "/a/f19"


def test_a_node_with_a_missing_value_has_no_template() -> None:
    compiled = _compile(_obj({"a": _obj({"s": STRING})}))

    assert _child(compiled, _root(compiled), "a").template is None
    assert _root(compiled).template is None


def test_a_node_without_missing_values_has_a_template() -> None:
    compiled = _compile(_obj({"a": _obj({"s": {"type": ["string", "null"]}})}))

    assert _child(compiled, _root(compiled), "a").template is not None


def test_a_template_is_shared_by_node_number_not_copied_into_its_parents() -> None:
    layers = 9
    defs: dict[str, Any] = {f"D{layers}": {"type": ["string", "null"]}}
    for i in reversed(range(layers)):
        defs[f"D{i}"] = _obj({"a": _ref(f"D{i + 1}"), "b": _ref(f"D{i + 1}")})
    compiled = _compile(_obj({"d": _ref("D0")}, **{"$defs": defs}))

    assert len(compiled.nodes) == layers + 2
    assert (
        _child(compiled, _root(compiled), "d").template_nodes == 2 ** (layers + 1) - 1
    )
    assert _root(compiled).template_nodes == 2 ** (layers + 1)


def test_the_size_of_a_small_template_is_the_size_of_its_json() -> None:
    options = _obj(
        {
            "verbose": {"type": ["boolean", "null"]},
            "limit": {"type": ["integer", "null"]},
        }
    )
    compiled = _compile(
        _obj(
            {"name": {"type": ["string", "null"]}, "opt": _ref("Options")},
            **{"$defs": {"Options": options}},
        )
    )
    root = _root(compiled)

    assert _child(compiled, root, "opt").template_bytes == 29
    assert root.template_bytes == 49


def test_the_template_sizes_of_a_chain_of_seven_doubled_layers() -> None:
    long_a, long_b = "a" * 64, "b" * 64
    defs: dict[str, Any] = {
        f"D{i}": _obj({long_a: _ref(f"D{i + 1}"), long_b: _ref(f"D{i + 1}")})
        for i in range(7)
    }
    defs["D7"] = {"type": ["string", "null"]}
    compiled = _compile(
        _obj(
            {
                "items": {
                    "type": "array",
                    "items": _obj({"i": {"type": "integer"}, "b": _ref("D0")}),
                }
            },
            **{"$defs": defs},
        )
    )
    element = _items(compiled, _child(compiled, _root(compiled), "items"))
    d0 = _child(compiled, element, "b")

    assert d0.template_bytes == 17_911
    assert d0.template_nodes == 255


def test_compiling_does_not_change_the_schema() -> None:
    schema = _obj(
        {
            "a": _nullable(_ref("T")),
            "b": {"type": "array", "items": _ref("E")},
            "c": {"type": "integer", "const": 1.0},
        },
        **{
            "$defs": {
                "T": _obj({"x": {"type": "string", "enum": ["p", "q"]}}),
                "E": _nullable({"type": "string", "description": "d"}),
            }
        },
    )
    before = copy.deepcopy(schema)

    structured_schema.compile_schema(schema)

    assert schema == before
    assert json.dumps(schema) == json.dumps(before)


def test_the_compiled_schema_keeps_the_schema_it_was_built_from() -> None:
    schema = _obj({"a": STRING})

    assert _compile(schema).schema is schema


# --- the size of canonical JSON ----------------------------------------------------


def test_canonical_json_is_measured_in_bytes_not_characters() -> None:
    assert structured_schema.canonical_json_bytes("é中\U0001f600") == 2 + 2 + 3 + 4
    assert structured_schema.canonical_json_bytes("é") == 4


def test_canonical_json_writes_compact_separators_and_keeps_key_order() -> None:
    value = {"b": [1, 2.5, True, None], "a": {"k": "v"}}
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    assert structured_schema.canonical_json_bytes(value) == len(text)
    assert text.startswith('{"b":')


def test_canonical_json_writes_the_smallest_float_and_the_largest() -> None:
    assert structured_schema.canonical_json_bytes(5e-324) == len("5e-324")
    assert structured_schema.canonical_json_bytes(1e308) == len("1e+308")
    assert structured_schema.canonical_json_bytes(-0.0) == len("-0.0")


def test_canonical_json_escapes_quotes_backslashes_and_control_characters() -> None:
    assert structured_schema.canonical_json_bytes('a"b\\c\n') == len('"a\\"b\\\\c\\n"')


def test_canonical_json_refuses_a_value_that_is_not_finite() -> None:
    with pytest.raises(ValueError):
        structured_schema.canonical_json_bytes(float("nan"))
    with pytest.raises(ValueError):
        structured_schema.canonical_json_bytes([float("inf")])
