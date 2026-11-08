"""Structural checks on the schema subset accepted for structured output."""

from __future__ import annotations

import collections
import copy
import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from tests.shared.structured_path_reference import reference_render
from tests.shared.structured_schema_random import gen_schema
from tests.shared.structured_schema_reference import copy_size, expanded_copy
from xagent.core.agent import structured_schema
from xagent.core.agent.structured_schema import (
    ERROR_PATH_MAX_CHARS,
    SchemaViolation,
    allowed_values_suffix,
    canon,
    check_schema_structure,
    json_kind,
    measure_expansion,
    normalized_enum,
    render_error_path,
)

CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "structured_output"
    / "structured_schema_corpus.json"
)
CORPUS = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

SCHEMA_REJECTS = [
    entry for entry in CORPUS["reject"] if str(entry["rule"]).startswith("S")
]
SCHEMA_ACCEPTS = [
    entry for entry in CORPUS["accept"] if entry["response_format"]["type"] != "text"
]

STRING: dict[str, Any] = {"type": "string"}


def _schema_of(entry: dict[str, Any]) -> Any:
    return entry["response_format"]["json_schema"]["schema"]


def _obj(properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    node.update(extra)
    return node


def _items_chain(arrays: int) -> dict[str, Any]:
    node: dict[str, Any] = STRING
    for _ in range(arrays):
        node = {"type": "array", "items": node}
    return node


def _violation(schema: Any) -> tuple[str, str] | None:
    found = check_schema_structure(schema)
    if found is None:
        return None
    return found.rule, render_error_path(list(found.path))


# --- the shared corpus -----------------------------------------------------------


@pytest.mark.parametrize("entry", SCHEMA_REJECTS, ids=lambda e: e["id"])
def test_every_corpus_rejection_reports_its_rule_and_path(
    entry: dict[str, Any],
) -> None:
    assert _violation(_schema_of(entry)) == (entry["rule"], entry["path"])


@pytest.mark.parametrize("entry", SCHEMA_ACCEPTS, ids=lambda e: e["id"])
def test_every_corpus_acceptance_passes_and_leaves_the_schema_alone(
    entry: dict[str, Any],
) -> None:
    schema = _schema_of(entry)
    before = copy.deepcopy(schema)

    assert check_schema_structure(schema) is None
    assert schema == before


def test_every_rule_checked_here_has_a_rejection_in_the_corpus() -> None:
    covered = {entry["rule"] for entry in SCHEMA_REJECTS}

    assert covered >= {
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
        "S11",
        "S12a",
        "S12b",
        "S13",
    }


def test_each_unsupported_json_schema_keyword_has_its_own_rejection() -> None:
    keywords = {
        entry["id"].removeprefix("s3-keyword-")
        for entry in SCHEMA_REJECTS
        if entry["id"].startswith("s3-keyword-")
    }

    assert {"minLength", "pattern", "format", "oneOf", "$dynamicRef", "default"} <= (
        keywords
    )
    assert not keywords & set(structured_schema._KEYWORDS)


# --- exact JSON types ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (None, "null"),
        (True, "boolean"),
        (False, "boolean"),
        (1, "integer"),
        (1.0, "float"),
        ("x", "string"),
        ([], "array"),
        ({}, "object"),
        ((1,), "other"),
        (b"x", "other"),
    ],
)
def test_json_kind_is_exact(value: object, kind: str) -> None:
    assert json_kind(value) == kind


def test_json_kind_does_not_treat_subclasses_as_their_base() -> None:
    class Flag(int):
        pass

    class Mapping(dict):  # type: ignore[type-arg]
        pass

    assert json_kind(Flag(1)) == "other"
    assert json_kind(Mapping()) == "other"


def test_canon_keeps_booleans_apart_from_numbers() -> None:
    assert canon(True) != canon(1)
    assert canon(False) != canon(0)
    assert canon(1) == canon(1.0)
    assert canon(0) == canon(-0.0)
    assert canon(1.5) != canon(1)
    assert canon("1") != canon(1)
    assert canon(None) == canon(None)


def test_canon_is_not_defined_for_containers() -> None:
    with pytest.raises(ValueError):
        canon([1])
    with pytest.raises(ValueError):
        canon({"a": 1})


def test_canon_of_a_huge_integer_valued_float_is_exact() -> None:
    assert canon(1e300) == canon(int(1e300))


def test_boolean_is_not_accepted_where_a_number_or_an_integer_is_required() -> None:
    assert _violation(_obj({"a": {"type": "integer", "enum": [True]}})) == (
        "S4",
        "/properties/a",
    )
    assert _violation(_obj({"a": {"type": "number", "const": False}})) == (
        "S4",
        "/properties/a",
    )


@pytest.mark.parametrize(
    "node",
    [
        {"type": "boolean", "enum": [True]},
        {"type": "integer", "enum": [1, 2.0]},
        {"type": "number", "const": 1},
        {"type": "string", "const": "x"},
    ],
)
def test_matching_enum_and_const_values_are_accepted(node: dict[str, Any]) -> None:
    assert check_schema_structure(_obj({"a": node})) is None


# --- const is an enum of one value -----------------------------------------------


def test_normalized_enum_reads_const_as_a_one_value_enum() -> None:
    assert normalized_enum({"type": "string", "const": "x"}) == ["x"]
    assert normalized_enum({"type": "string", "enum": ["a", "b"]}) == ["a", "b"]
    assert normalized_enum({"type": "string"}) is None
    assert normalized_enum({"type": "null", "const": None}) == [None]


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"const": "x"}, "S4"),
        ({"type": "string", "const": "x", "enum": ["x"]}, "S4"),
        ({"type": ["string", "null"], "enum": ["a"]}, "S4"),
        ({"type": ["string", "null"], "const": "x"}, "S4"),
    ],
)
def test_const_and_nullable_enum_rules(node: dict[str, Any], expected: str) -> None:
    found = check_schema_structure(_obj({"a": node}))

    assert found == SchemaViolation(expected, ("properties", "a"))


# --- enum values (S12a) ----------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        {"type": "string", "enum": [""]},
        {"type": "string", "enum": [" "]},
        {"type": "string", "enum": ["__NOT_EXPRESSIBLE__"]},
        {"type": "string", "enum": ["a__NOT_EXPRESSIBLE__"]},
        {"type": "string", "const": ""},
        {"type": "string", "enum": ["a", "a"]},
        {"type": "number", "enum": [1, 1.0]},
        {"type": ["string", "null"], "enum": ["x", None, None]},
    ],
)
def test_enum_values_that_cannot_be_told_apart_or_are_blank_are_rejected(
    node: dict[str, Any],
) -> None:
    assert _violation(_obj({"a": node})) == ("S12a", "/properties/a")


def test_true_and_one_are_distinct_enum_values() -> None:
    schema = _obj({"a": {"type": ["boolean", "null"], "enum": [True, None]}})
    other = _obj({"a": {"type": ["integer", "null"], "enum": [1, None]}})

    assert check_schema_structure(schema) is None
    assert check_schema_structure(other) is None
    assert canon(True) != canon(1)


def test_enum_limits_are_inclusive() -> None:
    values = [f"v{i}" for i in range(500)]

    assert (
        check_schema_structure(_obj({"a": {"type": "string", "enum": values}})) is None
    )
    assert _violation(_obj({"a": {"type": "string", "enum": values + ["extra"]}})) == (
        "S12a",
        "/properties/a",
    )
    assert (
        check_schema_structure(_obj({"a": {"type": "string", "enum": ["x" * 256]}}))
        is None
    )
    assert _violation(_obj({"a": {"type": "string", "enum": ["x" * 257]}})) == (
        "S12a",
        "/properties/a",
    )


# --- node count and depth (S2) ---------------------------------------------------


def test_the_2001st_node_in_document_order_is_reported() -> None:
    properties = {f"p{i}": STRING for i in range(2000)}

    assert _violation(_obj(properties)) == ("S2", "/properties/p1999")
    # 2000 nodes pass S2; the schema is then turned back by S1 for its size.
    assert _violation(_obj({f"p{i}": STRING for i in range(1999)})) == ("S1", "")


def test_a_node_nested_33_levels_deep_is_reported_and_32_is_accepted() -> None:
    # The leaf sits at depth 1 (the property) plus one per array level.
    assert check_schema_structure(_obj({"a": _items_chain(31)})) is None
    found = _violation(_obj({"a": _items_chain(32)}))
    assert found == ("S2", "/properties/a" + "/items" * 32)


def test_whichever_limit_is_met_first_in_document_order_is_reported() -> None:
    nodes_first = {f"p{i}": STRING for i in range(1999)}
    nodes_first["deep"] = _items_chain(40)
    depth_first = {"deep": _items_chain(40)}
    depth_first.update({f"p{i}": STRING for i in range(2000)})

    assert _violation(_obj(nodes_first)) == ("S2", "/properties/deep")
    assert _violation(_obj(depth_first)) == (
        "S2",
        "/properties/deep" + "/items" * 32,
    )


def test_a_properties_list_is_not_a_node_and_is_left_to_s3() -> None:
    schema = {
        "type": "object",
        "properties": [1] * 3000,
        "required": [],
        "additionalProperties": False,
    }

    assert _violation(schema) == ("S3", "/properties")


def test_a_properties_list_on_the_deepest_allowed_node_is_s3_not_s2() -> None:
    node: dict[str, Any] = {
        "type": "object",
        "properties": [],
        "required": [],
        "additionalProperties": False,
    }
    for _ in range(31):
        node = {"type": "array", "items": node}

    assert _violation(_obj({"a": node})) == (
        "S3",
        "/properties/a" + "/items" * 31 + "/properties",
    )


def test_values_of_unsupported_keywords_are_not_counted_as_nodes() -> None:
    schema = _obj({"a": {"type": "string", "format": [[1]] * 3000}})

    assert _violation(schema) == ("S3", "/properties/a/format")


# --- serialization, size and characters (S1) -------------------------------------


def test_nan_and_infinity_are_rejected_as_unserializable() -> None:
    for number in (float("nan"), float("inf"), float("-inf")):
        schema = _obj({"a": {"type": "number", "enum": [number]}})
        assert _violation(schema) == ("S1", "")


def test_a_nul_in_the_value_of_an_unsupported_keyword_is_reported_as_s1() -> None:
    schema = _obj({"p": {"type": "string", "format": "a\u0000"}})

    assert _violation(schema) == ("S1", "/properties/p/format")


def test_nesting_that_cannot_be_serialized_is_rejected_not_raised() -> None:
    deep: list[Any] = []
    for _ in range(10000):
        deep = [deep]
    schema = _obj({"a": {"type": "string", "unknown_keyword": deep}})

    assert _violation(schema) == ("S1", "")


def test_the_size_is_measured_before_the_characters_are_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real = structured_schema.has_forbidden_char

    def spy(text: str) -> bool:
        nonlocal calls
        calls += 1
        return real(text)

    monkeypatch.setattr(structured_schema, "has_forbidden_char", spy)
    schema = _obj({"a": {"type": "string", "unknown_keyword": "x" * 10_000_000}})

    assert _violation(schema) == ("S1", "")
    assert calls == 0


def test_the_size_limit_is_16_kib_of_utf8() -> None:
    def schema_of_size(target: int) -> dict[str, Any]:
        schema = _obj({"a": {"type": "string", "note": ""}})
        size = len(json.dumps(schema, separators=(",", ":")).encode())
        schema["properties"]["a"]["note"] = "n" * (target - size)
        assert len(json.dumps(schema, separators=(",", ":")).encode()) == target
        return schema

    # The unsupported keyword is only reached once the size is accepted.
    assert _violation(schema_of_size(16384)) == ("S3", "/properties/a/note")
    assert _violation(schema_of_size(16385)) == ("S1", "")


def test_a_lone_surrogate_in_an_oversized_schema_is_reported_as_size() -> None:
    properties = {
        f"p{i}": {"type": "string", "description": "d" * 1000} for i in range(17)
    }
    properties["p3"]["description"] += "\ud800"

    assert _violation(_obj(properties)) == ("S1", "")


def test_a_key_with_a_forbidden_character_is_reported_at_its_object() -> None:
    assert _violation(_obj({"a": {"type": "string", "x\u0000": 1}})) == (
        "S1",
        "/properties/a",
    )


def test_the_first_forbidden_character_in_document_order_is_reported() -> None:
    schema = _obj(
        {
            "a": {"type": "string", "description": "ok", "title": "t\u0000"},
            "b": {"type": "string", "description": "\ud800"},
        }
    )

    assert _violation(schema) == ("S1", "/properties/a/title")


def test_forbidden_characters_inside_enum_values_are_found() -> None:
    schema = _obj({"a": {"type": "string", "enum": ["fine", "bad\u0000"]}})

    assert _violation(schema) == ("S1", "/properties/a/enum/1")


# --- the path in a message -------------------------------------------------------


def test_a_path_is_ascii_json_pointer_text() -> None:
    rendered = render_error_path(
        ["properties", "a/b~c", 3, "café", "\ud800", 'q"\\', "x\n", "\u0000"]
    )

    assert rendered == (
        '/properties/a~1b~0c/3/caf\\u00e9/\\ud800/q\\"\\\\/x\\n/\\u0000'
    )
    assert rendered.isascii()


def test_an_empty_path_renders_as_an_empty_string() -> None:
    assert render_error_path([]) == ""


def test_a_property_name_of_5000_characters_cuts_the_path_to_its_end() -> None:
    schema = _obj({"x" * 5000: _items_chain(40)})

    rule, path = _violation(schema) or ("", "")

    assert rule == "S2"
    assert len(path) == ERROR_PATH_MAX_CHARS
    assert path.startswith("...")
    assert path.endswith("/items")
    assert path == reference_render(["properties", "x" * 5000] + ["items"] * 32)


def test_33_levels_of_64_character_names_are_not_cut() -> None:
    names = [chr(ord("a") + i % 26) * 64 for i in range(33)]
    node: dict[str, Any] = STRING
    for name in reversed(names[1:]):
        node = _obj({name: node})
    schema = _obj({names[0]: node})

    rule, path = _violation(schema) or ("", "")

    assert rule == "S2"
    assert path == "".join(f"/properties/{name}" for name in names)
    assert len(path) <= ERROR_PATH_MAX_CHARS
    assert not path.startswith("...")


def test_a_lone_surrogate_name_with_33_levels_renders_as_ascii() -> None:
    schema = _obj({"\ud800": _items_chain(40)})

    rule, path = _violation(schema) or ("", "")

    assert rule == "S2"
    assert path.startswith("/properties/\\ud800/items/")
    assert path.isascii()


def test_rendering_a_huge_segment_matches_escaping_everything_then_cutting() -> None:
    segment = "\u0080" * 5_000_000
    expected = reference_render([segment])
    expected_with_tail = reference_render(["properties", segment, "items", 2])

    tracemalloc.start()
    try:
        rendered = render_error_path([segment])
        rendered_with_tail = render_error_path(["properties", segment, "items", 2])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert rendered == expected
    assert rendered_with_tail == expected_with_tail
    assert len(rendered) == ERROR_PATH_MAX_CHARS
    assert rendered.startswith("...")
    assert peak <= 1024 * 1024


@pytest.mark.parametrize(
    "segments",
    [
        ["a" * 3000],
        ["a" * 3000, "b"],
        ["a", "b" * 2559],
        ["\U0001f600" * 2000, 1, "x"],
        ["p", 7, "q" * 2558],
        ["~/" * 2000],
    ],
)
def test_rendering_matches_the_reference_for_long_paths(
    segments: list[str | int],
) -> None:
    assert render_error_path(segments) == reference_render(segments)


# --- names are matched as whole strings ------------------------------------------


def test_names_with_a_trailing_newline_are_rejected_everywhere() -> None:
    assert _violation(_obj({"a\n": STRING})) == ("S13", "/properties/a\\n")
    assert _violation(_obj({"a": STRING}, **{"$defs": {"a\n": STRING}})) == (
        "S7",
        "/$defs/a\\n",
    )
    assert _violation(
        _obj(
            {"p": {"$ref": "#/$defs/a\n"}, "q": {"$ref": "#/$defs/a"}},
            **{"$defs": {"a": STRING}},
        )
    ) == ("S8", "/properties/p/$ref")


def test_the_same_names_without_the_newline_are_accepted() -> None:
    schema = _obj(
        {"a": STRING, "q": {"$ref": "#/$defs/a"}},
        **{"$defs": {"a": STRING}},
    )

    assert check_schema_structure(schema) is None


# --- scale attacks are refused quickly -------------------------------------------

# These checks are pinned by what they return and, for S11, by how much text
# they serialize. This bound only catches a slowdown by an order of magnitude.
BACKSTOP_SECONDS = 10.0


def _nested_any_of(levels: int) -> dict[str, Any]:
    node: dict[str, Any] = STRING
    for _ in range(levels):
        node = {"anyOf": [node, {"type": "null"}]}
    return node


def _doubling_defs(entries: int) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    for i in range(entries - 1):
        defs[f"d{i}"] = _obj(
            {"x": {"$ref": f"#/$defs/d{i + 1}"}, "y": {"$ref": f"#/$defs/d{i + 1}"}}
        )
    defs[f"d{entries - 1}"] = STRING
    return defs


SCALE_ATTACKS = [
    pytest.param(_obj({"a": _nested_any_of(400)}), "S2", id="400-nested-any-of"),
    pytest.param(
        _obj(
            {"a": {"$ref": "#/$defs/A", "anyOf": [STRING, {"type": "null"}]}},
            **{"$defs": {"A": STRING}},
        ),
        "S4",
        id="ref-beside-any-of",
    ),
    pytest.param(
        _obj({"a": {"$ref": "#/%24defs/A"}}, **{"$defs": {"A": STRING}}),
        "S8",
        id="percent-encoded-fragment",
    ),
    pytest.param(
        _obj({"a": {"anyOf": [STRING, {"type": "integer"}, {"type": "null"}]}}),
        "S10",
        id="three-branch-any-of",
    ),
    pytest.param(
        _obj({"a": STRING}, **{"$defs": _doubling_defs(41)}),
        "S7",
        id="41-unreferenced-doubling-entries",
    ),
]


@pytest.mark.parametrize(("schema", "rule"), SCALE_ATTACKS)
def test_scale_attacks_are_rejected(schema: dict[str, Any], rule: str) -> None:
    start = time.perf_counter()
    found = check_schema_structure(schema)
    elapsed = time.perf_counter() - start

    assert found is not None
    assert found.rule == rule
    assert elapsed < BACKSTOP_SECONDS


def test_an_unreferenced_entry_is_reported_with_the_first_one_in_key_order() -> None:
    found = check_schema_structure(_obj({"a": STRING}, **{"$defs": _doubling_defs(41)}))

    assert found == SchemaViolation("S7", ("$defs", "d0"))


# --- several violations: the rule order decides, then document order --------------


def test_an_earlier_rule_wins_over_an_earlier_node() -> None:
    schema = _obj(
        {
            "a": {"type": "object", "properties": {}, "additionalProperties": False},
            "b": {"type": "string", "minLength": 1},
        }
    )

    assert _violation(schema) == ("S3", "/properties/b/minLength")


def test_the_node_s_own_keywords_are_judged_before_its_children() -> None:
    schema = _obj({"x": 1}, minLength=1)

    assert _violation(schema) == ("S3", "/minLength")


def test_a_cycle_is_reported_at_the_first_entry_on_it_in_key_order() -> None:
    schema = _obj(
        {"p": {"$ref": "#/$defs/a"}},
        **{
            "$defs": {
                "b": _obj({"x": {"$ref": "#/$defs/a"}}),
                "a": _obj({"y": {"$ref": "#/$defs/b"}}),
            }
        },
    )

    assert _violation(schema) == ("S9", "/$defs/b")


def test_the_check_never_raises_for_a_root_that_is_not_an_object() -> None:
    for root in ("x", 1, 1.5, True, None, [], [{"type": "object"}], (1, 2)):
        found = check_schema_structure(root)
        assert found is not None
        assert found.rule in {"S6", "S3", "S2", "S1"}


# --- the size of the schema after $ref references are expanded (S11, S12b) --------


def _ref(name: str) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{name}"}


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [node, {"type": "null"}]}


def _count_nodes(schema: dict[str, Any]) -> int:
    """Nodes of the expanded tree: a reference and an alias count for nothing and
    a nullable union counts for itself plus its two branches."""
    defs = schema.get("$defs", {})

    def count(node: dict[str, Any]) -> int:
        if "$ref" in node:
            return count(defs[node["$ref"].removeprefix("#/$defs/")])
        if "anyOf" in node:
            return 2 + sum(count(b) for b in node["anyOf"] if b != {"type": "null"})
        total = 1
        for child in (node.get("properties") or {}).values():
            total += count(child)
        if "items" in node:
            total += count(node["items"])
        return total

    return count(schema)


def test_a_nullable_union_counts_as_three_nodes_and_an_alias_as_none() -> None:
    nullable = _obj({"a": _nullable(STRING)})
    aliased = _obj({"a": _ref("A")}, **{"$defs": {"A": _ref("B"), "B": STRING}})

    assert measure_expansion(nullable).nodes == 4
    assert measure_expansion(aliased).nodes == 2


def test_the_root_is_at_depth_zero_and_each_reference_adds_a_level() -> None:
    assert measure_expansion(_obj({})).depth == 0
    assert measure_expansion(_obj({"a": STRING})).depth == 1
    assert measure_expansion(_obj({"a": _nullable(STRING)})).depth == 2
    aliased = _obj({"a": _ref("A")}, **{"$defs": {"A": _ref("B"), "B": STRING}})
    # The property is at level 1; the reference to A and the alias A to B are
    # two hops, so the string sits at level 3.
    assert measure_expansion(aliased).depth == 3


def test_enum_values_are_counted_as_written_without_the_added_null_and_marker() -> None:
    schema = _obj(
        {
            "a": {"type": "string", "enum": ["x", "y", "z"]},
            "b": {"type": ["string", "null"], "enum": ["x", None]},
            "c": {"type": "string", "const": "k"},
        }
    )

    assert measure_expansion(schema).enum_values == 6


def test_a_reference_repeats_the_enum_of_its_target() -> None:
    schema = _obj(
        {"a": _ref("E"), "b": _ref("E"), "c": _ref("E")},
        **{"$defs": {"E": {"type": "string", "enum": ["x", "y"]}}},
    )

    assert measure_expansion(schema).enum_values == 6


SIZE_SCHEMAS = {
    "plain": _obj({"a": STRING, "b": {"type": "integer"}, "c": {"type": "boolean"}}),
    "described": _obj(
        {"a": {"type": "string", "title": "T", "description": "the a"}},
        description="root",
    ),
    "string-enum": _obj({"a": {"type": "string", "enum": ["x", "y"]}}),
    "integer-enum": _obj({"a": {"type": "integer", "enum": [1, 2, 3.0]}}),
    "const": _obj({"a": {"type": "string", "const": "k"}}),
    "nullable-enum": _obj({"a": {"type": ["string", "null"], "enum": ["x", None]}}),
    "nullable-type-array": _obj({"a": {"type": ["integer", "null"]}}),
    "non-ascii": _obj(
        {"a": {"type": "string", "description": "\u4e2d\u6587", "enum": ["\u00e9"]}}
    ),
    "array": _obj({"a": {"type": "array", "items": {"type": "number"}}}),
    "array-of-nullable": _obj({"a": {"type": "array", "items": _nullable(STRING)}}),
    "nullable-union": _obj({"a": _nullable({"type": "integer"})}),
    "union-with-both-descriptions": _obj(
        {
            "a": {
                "description": "outer",
                "title": "Outer",
                "anyOf": [
                    {"type": "string", "description": "inner", "enum": ["x"]},
                    {"type": "null"},
                ],
            }
        }
    ),
    "union-with-outer-description-only": _obj(
        {
            "a": {
                "description": "outer",
                "anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}],
            }
        }
    ),
    "union-over-a-reference": _obj(
        {"a": {"description": "d", "anyOf": [_ref("A"), {"type": "null"}]}},
        **{"$defs": {"A": _ref("B"), "B": {"type": "string", "enum": ["p", "q"]}}},
    ),
    "union-over-an-object": _obj(
        {"a": _nullable(_obj({"x": STRING, "y": _nullable({"type": "integer"})}))}
    ),
    "references-and-aliases": _obj(
        {"a": _ref("A"), "b": _ref("O"), "c": {"type": "array", "items": _ref("O")}},
        **{
            "$defs": {
                "A": _ref("S"),
                "S": {"type": "string", "enum": ["m", "n"]},
                "O": _obj({"s": _ref("S"), "t": _nullable(_ref("S"))}),
            }
        },
    ),
    "alias-of-an-alias-of-an-object": _obj(
        {"a": _ref("A")},
        **{"$defs": {"A": _ref("B"), "B": _obj({"x": STRING})}},
    ),
    "long-enum-with-truncated-suffix": _obj(
        {"a": {"type": "string", "enum": [f"value-{i:03d}" for i in range(120)]}}
    ),
    "surrogate-pair-and-escapes": _obj(
        {"a": {"type": "string", "enum": ["\U0001f600", 'q"\\', "tab\t"]}}
    ),
    "target-that-is-a-nullable-union": _obj(
        {"a": _ref("N"), "b": {"type": "array", "items": _ref("N")}},
        **{
            "$defs": {
                "N": {
                    "anyOf": [
                        {"type": "string", "description": "x"},
                        {"type": "null"},
                    ],
                    "description": "y",
                    "title": "T",
                }
            }
        },
    ),
    "title-conflict": _obj(
        {
            "a": {
                "title": "Outer",
                "anyOf": [{"type": "string", "title": "Inner"}, {"type": "null"}],
            }
        }
    ),
    "ninety-nullable-references-to-one-enum": _obj(
        {f"f{i:03d}": _nullable(_ref("E")) for i in range(90)},
        **{
            "$defs": {"E": {"type": "string", "enum": [f"v{i:02d}" for i in range(40)]}}
        },
    ),
}


@pytest.mark.parametrize("schema", SIZE_SCHEMAS.values(), ids=SIZE_SCHEMAS.keys())
def test_the_estimate_and_the_copy_match_the_reference_copy(
    schema: dict[str, Any],
) -> None:
    assert check_schema_structure(schema) is None

    measured = measure_expansion(schema)
    built = structured_schema.model_facing_schema(
        structured_schema.compile_schema(schema)
    )

    assert measured.rewritten_bytes == copy_size(schema)
    assert built == expanded_copy(schema)
    assert measured.nodes == _count_nodes(schema)


def test_the_estimate_and_the_copy_of_random_schemas_match_the_reference_copy() -> None:
    rng = random.Random(20261104)
    for _ in range(400):
        schema = gen_schema(rng)
        assert check_schema_structure(schema) is None, schema
        built = structured_schema.model_facing_schema(
            structured_schema.compile_schema(schema)
        )

        assert measure_expansion(schema).rewritten_bytes == copy_size(schema), schema
        assert built == expanded_copy(schema), schema


def _count_shapes(node: Any, counts: collections.Counter[str]) -> None:
    if isinstance(node, dict):
        declared = node.get("type")
        types = declared if isinstance(declared, list) else [declared]
        if "anyOf" in node and node["anyOf"][0] == {"type": "null"}:
            counts["null-first-union"] += 1
        if isinstance(declared, list) and declared[0] == "null":
            counts["null-first-type"] += 1
        if "enum" in node and "number" in types:
            counts["number-enum"] += 1
        if "const" in node and declared == "number":
            counts["number-const"] += 1
        if "const" in node and declared == "integer":
            counts["integer-const"] += 1
        for value in node.values():
            _count_shapes(value, counts)
    elif isinstance(node, list):
        for value in node:
            _count_shapes(value, counts)


def test_the_random_schemas_hold_every_shape_the_compiler_branches_on() -> None:
    counts: collections.Counter[str] = collections.Counter()
    rng = random.Random(20261104)
    for _ in range(400):
        _count_shapes(gen_schema(rng), counts)

    shapes = (
        "null-first-union",
        "null-first-type",
        "number-enum",
        "number-const",
        "integer-const",
    )
    assert all(counts[shape] >= 10 for shape in shapes), counts


def test_the_estimate_does_not_change_the_schema() -> None:
    schema = SIZE_SCHEMAS["references-and-aliases"]
    before = copy.deepcopy(schema)

    measure_expansion(schema)

    assert schema == before


def test_the_allowed_values_text_is_cut_at_500_characters() -> None:
    short = ["a", None]
    assert allowed_values_suffix(short) == ' Allowed values: ["a",null]'

    def enum_text_of_length(length: int) -> list[Any]:
        # ["xxx..."] is the characters plus four: two quotes and two brackets.
        return ["x" * (length - 4)]

    exactly = enum_text_of_length(500)
    over = enum_text_of_length(501)
    assert len(json.dumps(exactly, separators=(",", ":"))) == 500
    assert allowed_values_suffix(exactly) == (
        " Allowed values: " + json.dumps(exactly, separators=(",", ":"))
    )
    cut = allowed_values_suffix(over)
    assert (
        cut
        == " Allowed values: "
        + json.dumps(over, separators=(",", ":"))[:500]
        + "\u2026"
    )


def test_the_allowed_values_text_is_cut_by_characters_not_bytes() -> None:
    values = ["\u4e2d" * 600]

    suffix = allowed_values_suffix(values)

    text = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    assert suffix == " Allowed values: " + text[:500] + "\u2026"
    assert len(suffix) == len(" Allowed values: ") + 500 + 1


def test_the_allowed_values_text_keeps_non_ascii_characters() -> None:
    assert allowed_values_suffix(["\u00e9"]) == ' Allowed values: ["\u00e9"]'


def test_the_expansion_limits_are_inclusive() -> None:
    assert measure_expansion(_alias_schema(31)).depth == 32
    assert check_schema_structure(_alias_schema(31)) is None
    assert measure_expansion(_alias_schema(32)).depth == 33
    assert _violation(_alias_schema(32)) == ("S11", "")


def _alias_schema(aliases: int) -> dict[str, Any]:
    defs: dict[str, Any] = {f"A{i}": _ref(f"A{i + 1}") for i in range(1, aliases)}
    defs[f"A{aliases}"] = STRING
    return _obj({"p": _ref("A1")}, **{"$defs": defs})


def _counted_schema(extra_plain: int) -> dict[str, Any]:
    defs: dict[str, Any] = {
        "N": _nullable(STRING),
        "G": _obj({f"a{i}": _ref("N") for i in range(10)}),
    }
    defs["H"] = _obj({f"a{i}": _ref("G") for i in range(10)})
    properties: dict[str, Any] = {f"h{i}": _ref("H") for i in range(6)}
    properties.update({f"g{i}": _ref("G") for i in range(4)})
    properties.update({f"n{i}": _ref("N") for i in range(3)})
    properties.update({f"s{i}": STRING for i in range(extra_plain)})
    return _obj(properties, **{"$defs": defs})


def test_2000_expanded_nodes_pass_and_2001_do_not() -> None:
    assert measure_expansion(_counted_schema(0)).nodes == 2000
    assert check_schema_structure(_counted_schema(0)) is None
    assert measure_expansion(_counted_schema(1)).nodes == 2001
    assert _violation(_counted_schema(1)) == ("S11", "")


def test_the_nullable_value_of_a_33_hop_alias_chain_is_too_deep() -> None:
    defs: dict[str, Any] = {f"A{i}": _ref(f"A{i + 1}") for i in range(1, 32)}
    defs["A32"] = STRING
    schema = _obj({"p": {"anyOf": [_ref("A1"), {"type": "null"}]}}, **{"$defs": defs})

    assert _violation(schema) == ("S11", "")


def _layered_schema(layers: int, leaf: dict[str, Any]) -> dict[str, Any]:
    """A leaf under ``layers`` objects, each with two properties that both refer
    to the next layer down; the root is the top layer."""
    defs: dict[str, Any] = {"L0": leaf}
    for layer in range(1, layers):
        defs[f"L{layer}"] = _obj(
            {"x": _ref(f"L{layer - 1}"), "y": _ref(f"L{layer - 1}")}
        )
    return _obj(
        {"x": _ref(f"L{layers - 1}"), "y": _ref(f"L{layers - 1}")}, **{"$defs": defs}
    )


def test_a_small_schema_whose_copy_would_be_over_a_megabyte_is_refused() -> None:
    leaf = {"type": "string", "title": "t" * 1024, "description": "d" * 1000}
    schema = _layered_schema(9, leaf)
    original = len(json.dumps(schema, separators=(",", ":")).encode())

    assert original < 16384
    assert measure_expansion(schema).nodes == 1023
    assert measure_expansion(schema).rewritten_bytes > 1_000_000
    assert _violation(schema) == ("S11", "")


def test_a_500_value_enum_expanded_nine_times_is_refused_without_building_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [f"{i:04d}-" + "v" * 14 for i in range(500)]
    schema = _layered_schema(9, {"type": "string", "enum": values})
    original = len(json.dumps(schema, separators=(",", ":")).encode())
    assert 8000 < original < 16384
    real_dumps = json.dumps
    written: list[int] = []

    def counting_dumps(value: Any, *args: Any, **kwargs: Any) -> str:
        text = real_dumps(value, *args, **kwargs)
        written.append(len(text.encode("utf-8", "surrogatepass")))
        return text

    monkeypatch.setattr(json, "dumps", counting_dumps)
    start = time.perf_counter()
    found = _violation(schema)
    elapsed = time.perf_counter() - start
    monkeypatch.undo()

    assert found == ("S11", "")
    # S11 refuses this schema because its copy is over MAX_EXPANDED_BYTES, so a
    # check that serialized less than that in total never serialized the copy.
    assert sum(written) <= structured_schema.MAX_EXPANDED_BYTES
    assert elapsed < BACKSTOP_SECONDS


def test_an_unreferenced_entry_is_refused_before_any_expansion_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_schema: object) -> None:
        raise AssertionError("the expansion was measured")

    monkeypatch.setattr(structured_schema, "measure_expansion", fail)
    schema = _obj({"a": STRING}, **{"$defs": _doubling_defs(41)})

    assert _violation(schema) == ("S7", "/$defs/d0")


def test_enum_values_over_5000_after_expansion_are_refused_by_s12b() -> None:
    values = [f"v{i}" for i in range(500)]
    defs = {"E": {"type": "string", "enum": values}}

    ten = _obj({f"p{i}": _ref("E") for i in range(10)}, **{"$defs": defs})
    eleven = _obj({f"p{i}": _ref("E") for i in range(11)}, **{"$defs": defs})

    assert check_schema_structure(ten) is None
    assert measure_expansion(ten).enum_values == 5000
    assert _violation(eleven) == ("S12b", "")


def test_the_expanded_size_is_judged_before_the_enum_rules() -> None:
    schema = _obj(
        {
            "a": {"type": "string", "enum": [f"v{i}" for i in range(501)]},
            "p": _ref("d0"),
        },
        **{"$defs": _doubling_defs(60)},
    )

    assert _violation(schema) == ("S11", "")


def test_a_500_layer_alias_chain_is_refused_quickly() -> None:
    defs: dict[str, Any] = {f"a{i}": _ref(f"a{i + 1}") for i in range(499)}
    defs["a499"] = STRING
    schema = _obj({"p": _ref("a0")}, **{"$defs": defs})

    start = time.perf_counter()
    found = _violation(schema)
    elapsed = time.perf_counter() - start

    assert found == ("S11", "")
    assert elapsed < BACKSTOP_SECONDS


def test_a_100_layer_doubling_chain_is_refused_quickly() -> None:
    schema = _obj({"p": _ref("d0")}, **{"$defs": _doubling_defs(100)})

    start = time.perf_counter()
    found = _violation(schema)
    elapsed = time.perf_counter() - start

    assert found == ("S11", "")
    assert elapsed < BACKSTOP_SECONDS
