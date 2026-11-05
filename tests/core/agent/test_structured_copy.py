"""The schema copy shown to the model: every node but the root may be null."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from xagent.core.agent import structured_schema
from xagent.core.agent.structured_schema import (
    allowed_values_suffix,
    canonical_json_bytes,
    check_schema_structure,
    measure_expansion,
)
from xagent.core.model.chat.basic import claude, gemini

MARKER = "__NOT_EXPRESSIBLE__"
STRING: dict[str, Any] = {"type": "string"}
NULLABLE_STRING: dict[str, Any] = {"type": ["string", "null"]}


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


def _union(node: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"anyOf": [node, {"type": "null"}], **extra}


def _suffix(*values: Any) -> str:
    return " Allowed values: " + json.dumps(list(values), separators=(",", ":"))


def _copy_of(schema: dict[str, Any]) -> dict[str, Any]:
    assert check_schema_structure(schema) is None
    return structured_schema.model_facing_schema(
        structured_schema.compile_schema(schema)
    )


# --- the nine shapes, each with its copy written out in full -------------------------

COPY_CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "plain-property": (
        _obj({"a": STRING}),
        _obj({"a": NULLABLE_STRING}),
    ),
    "array-element": (
        _obj({"xs": {"type": "array", "items": {"type": "integer"}}}),
        _obj(
            {
                "xs": {
                    "type": ["array", "null"],
                    "items": {"type": ["integer", "null"]},
                }
            }
        ),
    ),
    "reference": (
        _obj(
            {"a": _ref("T"), "b": _ref("T")},
            **{"$defs": {"T": {"type": "string", "description": "d"}}},
        ),
        _obj(
            {
                "a": {"type": ["string", "null"], "description": "d"},
                "b": {"type": ["string", "null"], "description": "d"},
            }
        ),
    ),
    "union-at-a-property": (
        _obj(
            {
                "a": _union(
                    {"type": "string", "description": "inner"}, description="outer"
                )
            }
        ),
        _obj({"a": {"type": ["string", "null"], "description": "outer inner"}}),
    ),
    "union-at-an-array-element": (
        _obj(
            {
                "xs": {
                    "type": "array",
                    "items": _union(
                        {"type": "string", "description": "inner", "title": "X"},
                        description="outer",
                    ),
                }
            }
        ),
        _obj(
            {
                "xs": {
                    "type": ["array", "null"],
                    "items": {
                        "type": ["string", "null"],
                        "title": "X",
                        "description": "outer inner",
                    },
                }
            }
        ),
    ),
    "nullable-type-with-an-enum": (
        _obj({"a": {"type": ["string", "null"], "enum": ["a", None]}}),
        _obj(
            {
                "a": {
                    "type": ["string", "null"],
                    "enum": ["a", None, MARKER],
                    "description": _suffix("a", None, MARKER),
                }
            }
        ),
    ),
    "const": (
        _obj({"a": {"type": "string", "const": "x"}}),
        _obj(
            {
                "a": {
                    "type": ["string", "null"],
                    "enum": ["x", None, MARKER],
                    "description": _suffix("x", None, MARKER),
                }
            }
        ),
    ),
    "root": (
        _obj({"a": STRING}, description="root text", title="R"),
        _obj({"a": NULLABLE_STRING}, description="root text", title="R"),
    ),
    "enum-that-is-not-text": (
        _obj({"a": {"type": "integer", "enum": [1, 2]}}),
        _obj(
            {
                "a": {
                    "type": ["integer", "null"],
                    "enum": [1, 2, None],
                    "description": _suffix(1, 2, None),
                }
            }
        ),
    ),
    "union-over-an-alias-chain": (
        _obj(
            {"a": _union(_ref("A"), description="o")},
            **{
                "$defs": {
                    "A": _ref("B"),
                    "B": {"type": "string", "description": "i", "title": "Inner"},
                }
            },
        ),
        _obj(
            {"a": {"type": ["string", "null"], "description": "o i", "title": "Inner"}}
        ),
    ),
    "union-title-conflict-takes-the-value-title": (
        _obj({"a": _union({"type": "string", "title": "Inner"}, title="Outer")}),
        _obj({"a": {"type": ["string", "null"], "title": "Inner"}}),
    ),
    "union-title-only-on-the-union": (
        _obj({"a": _union(STRING, title="Outer")}),
        _obj({"a": {"type": ["string", "null"], "title": "Outer"}}),
    ),
    "union-description-only-on-the-union": (
        _obj({"a": _union(STRING, description="outer")}),
        _obj({"a": {"type": ["string", "null"], "description": "outer"}}),
    ),
    "union-description-only-on-the-value": (
        _obj({"a": _union({"type": "string", "description": "inner"})}),
        _obj({"a": {"type": ["string", "null"], "description": "inner"}}),
    ),
    "union-over-an-enum-adds-one-null": (
        _obj({"a": _union({"type": "string", "enum": ["a", "b"]})}),
        _obj(
            {
                "a": {
                    "type": ["string", "null"],
                    "enum": ["a", "b", None, MARKER],
                    "description": _suffix("a", "b", None, MARKER),
                }
            }
        ),
    ),
    "target-that-is-a-nullable-union": (
        _obj(
            {"a": _ref("N"), "xs": {"type": "array", "items": _ref("N")}},
            **{
                "$defs": {
                    "N": _union(
                        {"type": "string", "description": "x"},
                        description="y",
                        title="T",
                    )
                }
            },
        ),
        _obj(
            {
                "a": {"type": ["string", "null"], "title": "T", "description": "y x"},
                "xs": {
                    "type": ["array", "null"],
                    "items": {
                        "type": ["string", "null"],
                        "title": "T",
                        "description": "y x",
                    },
                },
            }
        ),
    ),
    "null-listed-first-keeps-its-order": (
        _obj({"a": {"type": ["null", "string"]}}),
        _obj({"a": {"type": ["null", "string"]}}),
    ),
    "object-reached-through-a-reference": (
        _obj({"a": _ref("T")}, **{"$defs": {"T": _obj({"k": STRING})}}),
        _obj({"a": {**_obj({"k": NULLABLE_STRING}), "type": ["object", "null"]}}),
    ),
    "boolean-and-number-enums": (
        _obj(
            {
                "a": {"type": "boolean", "enum": [True]},
                "b": {"type": "number", "enum": [1.5, 2]},
            }
        ),
        _obj(
            {
                "a": {
                    "type": ["boolean", "null"],
                    "enum": [True, None],
                    "description": _suffix(True, None),
                },
                "b": {
                    "type": ["number", "null"],
                    "enum": [1.5, 2, None],
                    "description": _suffix(1.5, 2, None),
                },
            }
        ),
    ),
}


@pytest.mark.parametrize(
    ("schema", "expected"), COPY_CASES.values(), ids=COPY_CASES.keys()
)
def test_the_copy_of_each_shape_is_the_hand_written_schema(
    schema: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert _copy_of(schema) == expected


def test_make_nullable_turns_a_single_type_and_an_enum_nullable() -> None:
    assert structured_schema.make_nullable({"type": "string"}) == {
        "type": ["string", "null"]
    }
    assert structured_schema.make_nullable({"type": "integer", "enum": [1, 2]}) == {
        "type": ["integer", "null"],
        "enum": [1, 2, None],
    }


def test_make_nullable_leaves_a_nullable_node_as_it_is() -> None:
    node = {"type": ["null", "string"], "enum": [None, "x"], "description": "d"}

    assert structured_schema.make_nullable(node) == node
    assert structured_schema.make_nullable({"type": ["string", "null"]}) == {
        "type": ["string", "null"]
    }


def test_make_nullable_merges_the_union_into_the_node_without_changing_either() -> None:
    node = {"type": "string", "description": "inner"}
    union = {"anyOf": [node, {"type": "null"}], "description": "outer", "title": "U"}
    node_before, union_before = copy.deepcopy(node), copy.deepcopy(union)

    merged = structured_schema.make_nullable(node, any_of=union)

    assert merged == {
        "type": ["string", "null"],
        "description": "outer inner",
        "title": "U",
    }
    assert (node, union) == (node_before, union_before)


def test_the_schema_is_not_changed_and_the_copy_shares_nothing_with_it() -> None:
    schema = _obj(
        {
            "a": _ref("T"),
            "b": {"type": "array", "items": _ref("T")},
            "c": {"type": "string", "enum": ["x", "y"]},
            "d": _union(_ref("T")),
        },
        **{"$defs": {"T": _obj({"k": {"type": "integer", "enum": [1]}})}},
    )
    before = copy.deepcopy(schema)

    result = _copy_of(schema)

    assert schema == before
    owned: set[int] = set()

    def collect(value: Any, into: set[int]) -> None:
        if isinstance(value, dict | list):
            assert id(value) not in into, "the same container appears twice"
            into.add(id(value))
            for child in value.values() if isinstance(value, dict) else value:
                collect(child, into)

    collect(result, owned)
    original: set[int] = set()
    collect(schema, original)
    assert owned.isdisjoint(original)


# --- a null anywhere but the root ------------------------------------------------------

RICH = _obj(
    {
        "name": STRING,
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": STRING},
        "addr": _ref("Addr"),
        "nick": _union(STRING),
        "alts": {"type": "array", "items": _union(STRING)},
        "status": {"type": "string", "enum": ["open", "closed"]},
        "kind": {"type": "string", "const": "person"},
        "level": {"type": "integer", "enum": [1, 2]},
        "opt": {"type": ["string", "null"], "enum": ["x", None]},
        "flag": {"type": "boolean"},
        "ratio": {"type": "number"},
    },
    **{
        "$defs": {
            "Addr": _obj({"street": STRING, "geo": _obj({"lat": {"type": "number"}})})
        }
    },
)
RICH_FULL = {
    "name": "Ann",
    "age": 3,
    "tags": ["t", "u"],
    "addr": {"street": "s", "geo": {"lat": 1.5}},
    "nick": "n",
    "alts": ["p", "q"],
    "status": "open",
    "kind": "person",
    "level": 2,
    "opt": "x",
    "flag": True,
    "ratio": 0.5,
}


def _paths(value: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    found = [prefix]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_paths(child, prefix + (key,)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_paths(child, prefix + (index,)))
    return found


def _with_null_at(instance: Any, path: tuple[Any, ...]) -> Any:
    changed = copy.deepcopy(instance)
    target = changed
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = None
    return changed


def test_the_copy_accepts_a_null_at_every_position_but_the_root() -> None:
    result = _copy_of(RICH)
    validator = Draft202012Validator(result)
    assert Draft202012Validator(RICH).is_valid(RICH_FULL)
    assert validator.is_valid(RICH_FULL)

    positions = [p for p in _paths(RICH_FULL) if p]
    assert len(positions) > 15
    for path in positions:
        assert validator.is_valid(_with_null_at(RICH_FULL, path)), path
    assert not validator.is_valid(None)


def test_the_copy_accepts_the_marker_in_a_text_position_only_when_it_is_listed() -> (
    None
):
    validator = Draft202012Validator(_copy_of(RICH))

    assert validator.is_valid({**RICH_FULL, "status": MARKER})
    assert validator.is_valid({**RICH_FULL, "name": MARKER})
    assert not validator.is_valid({**RICH_FULL, "level": MARKER})
    assert not validator.is_valid({**RICH_FULL, "level": 3})


# --- the size of the copy (I-25) ------------------------------------------------------


def _walk_nodes(copy_node: dict[str, Any]) -> list[dict[str, Any]]:
    found = [copy_node]
    for child in copy_node.get("properties", {}).values():
        found.extend(_walk_nodes(child))
    if "items" in copy_node:
        found.extend(_walk_nodes(copy_node["items"]))
    return found


def _keys_anywhere(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys |= _keys_anywhere(child)
    elif isinstance(value, list):
        for child in value:
            keys |= _keys_anywhere(child)
    return keys


def test_the_copy_has_no_reference_no_definitions_and_stays_inside_the_limits() -> None:
    schema = _obj(
        {
            "a": _ref("A"),
            "b": {"type": "array", "items": _ref("A")},
            "c": _union(_ref("A")),
        },
        **{
            "$defs": {
                "A": _obj({"x": {"type": "string", "enum": ["p", "q"]}, "y": STRING})
            }
        },
    )
    result = _copy_of(schema)

    assert not _keys_anywhere(result) & {"$ref", "$defs", "anyOf"}
    assert len(_walk_nodes(result)) <= measure_expansion(schema).nodes
    assert canonical_json_bytes(result) <= 65536


def test_a_schema_of_3_kb_whose_copy_is_290_kb_is_refused_at_request_time() -> None:
    shared = _obj({f"k{i:02d}": {"type": "string", "const": "v"} for i in range(43)})
    schema = _obj(
        {f"r{j:02d}": _ref("D") for j in range(45)}, **{"$defs": {"D": shared}}
    )

    found = check_schema_structure(schema)

    assert len(json.dumps(schema, separators=(",", ":"))) == 3454
    assert measure_expansion(schema).rewritten_bytes == 292_709
    assert found is not None and found.rule == "S11"


def test_a_copy_of_exactly_64_kib_of_chinese_text_is_accepted() -> None:
    props: dict[str, Any] = {f"p{i:02d}": _ref("A") for i in range(21)}
    props["x"] = {"type": "integer"}
    schema = _obj(
        props,
        **{"$defs": {"A": {"type": "string", "description": "中" * 1000}}},
        description="文" * 411,
    )

    assert check_schema_structure(schema) is None
    result = _copy_of(schema)

    assert canonical_json_bytes(result) == 65536
    ascii_size = len(json.dumps(result, separators=(",", ":")).encode("utf-8"))
    assert ascii_size > 65536 * 1.5


def test_an_integer_constant_written_as_1e308_stays_a_float_in_the_copy() -> None:
    result = _copy_of(_obj({"n": {"type": "integer", "const": 1e308}}))
    node = result["properties"]["n"]

    assert node["enum"] == [1e308, None]
    assert type(node["enum"][0]) is float
    assert json.dumps(node["enum"]) == "[1e+308, null]"
    assert "1e+308" in node["description"]


# --- the constant and the list of allowed values (I-32) ----------------------------------


def test_a_constant_becomes_an_enum_and_its_values_end_the_description() -> None:
    schema = _obj({"a": {"type": "string", "const": "x", "description": "Pick it."}})

    node = _copy_of(schema)["properties"]["a"]

    assert "const" not in node
    assert node["enum"] == ["x", None, MARKER]
    assert node["description"] == "Pick it." + _suffix("x", None, MARKER)


def test_a_cut_list_of_allowed_values_matches_the_function_that_writes_it() -> None:
    values = [f"value-{i:03d}" for i in range(60)]

    node = _copy_of(_obj({"a": {"type": "string", "enum": values}}))["properties"]["a"]

    assert node["enum"] == [*values, None, MARKER]
    assert node["description"] == allowed_values_suffix(node["enum"])
    assert node["description"].endswith("…")
    listing = node["description"].removeprefix(" Allowed values: ")
    assert len(listing) == 501


# --- through the real provider rewriters (J-14) -------------------------------------------

ADAPTER_SCHEMA = _obj(
    {
        "status": {"type": "string", "enum": ["open", "closed"]},
        "statuses": {
            "type": "array",
            "items": {"type": "string", "enum": ["open", "closed"]},
        },
        "level": {"type": "integer", "enum": [1, 2]},
        "maybe": _union({"type": "string", "description": "d"}),
        "alts": {"type": "array", "items": _union(STRING)},
        "kind": {"type": "string", "const": "person"},
    }
)


def test_claude_leaves_the_copy_as_it_is() -> None:
    result = _copy_of(ADAPTER_SCHEMA)

    assert claude._fix_pydantic_schema_for_claude(result) == result


def _through_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    parameters = schema
    if gemini._contains_refs_or_defs(parameters):
        parameters = gemini._flatten_json_schema(parameters)
    sanitized: dict[str, Any] = gemini._sanitize_schema_for_gemini(parameters)
    return sanitized


def test_gemini_keeps_the_marker_and_the_null_of_a_text_enum() -> None:
    result = _through_gemini(_copy_of(ADAPTER_SCHEMA))
    status = result["properties"]["status"]
    element = result["properties"]["statuses"]["items"]

    for node in (status, element, result["properties"]["kind"]):
        assert MARKER in node["enum"]
        assert node["nullable"] is True
        assert node["type"] == "string"
    assert result["properties"]["alts"]["items"]["nullable"] is True
    assert result["properties"]["maybe"]["nullable"] is True


def test_gemini_drops_an_enum_that_is_not_text_but_the_description_keeps_it() -> None:
    result = _through_gemini(_copy_of(ADAPTER_SCHEMA))
    level = result["properties"]["level"]

    assert "enum" not in level
    assert level["type"] == "integer"
    assert "Allowed values: [1,2,null]" in level["description"]


# --- the budget on the copy -------------------------------------------------------------------


def _layered_schema(layers: int, leaf: dict[str, Any]) -> dict[str, Any]:
    defs: dict[str, Any] = {"L0": leaf}
    for layer in range(1, layers):
        defs[f"L{layer}"] = _obj(
            {"x": _ref(f"L{layer - 1}"), "y": _ref(f"L{layer - 1}")}
        )
    return _obj(
        {"x": _ref(f"L{layers - 1}"), "y": _ref(f"L{layers - 1}")}, **{"$defs": defs}
    )


def test_a_copy_over_the_byte_limit_stops_with_a_value_error() -> None:
    schema = _obj({"a": {"type": "string", "description": "d" * 200}})
    compiled = structured_schema.compile_schema(schema)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(structured_schema, "MAX_EXPANDED_BYTES", 100)
        with pytest.raises(ValueError) as caught:
            structured_schema.model_facing_schema(compiled)

    assert "d" * 20 not in str(caught.value)
    assert structured_schema.model_facing_schema(compiled)


def test_a_copy_that_would_be_megabytes_is_cut_off_before_it_is_built() -> None:
    values = [f"{i:04d}-" + "v" * 14 for i in range(500)]
    schema = _layered_schema(9, {"type": "string", "enum": values})
    compiled = structured_schema.compile_schema(schema)

    start = time.perf_counter()
    with pytest.raises(ValueError):
        structured_schema.model_facing_schema(compiled)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, f"took {elapsed * 1000:.1f} ms"
