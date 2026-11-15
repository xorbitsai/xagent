"""Compiling a checked schema and judging a value against it."""

from __future__ import annotations

import copy
import dataclasses
import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tests.shared.structured_schema_random import (
    MARKER,
    RNode,
    gen_decision,
    gen_instance,
    gen_schema,
    lookup,
    render,
    root_rnode,
    synth_valid,
)
from tests.shared.structured_schema_reference import expanded_copy
from xagent.core.agent import structured_schema
from xagent.core.agent.structured_schema import check_schema_structure

ss = structured_schema
CORPUS = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "structured_output"
        / "structured_schema_corpus.json"
    ).read_text(encoding="utf-8")
)
INSTANCES = CORPUS["instances"]

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


def test_properties_keep_their_written_order_and_the_size_of_their_quoted_key() -> None:
    compiled = _compile(_obj({"zeta": STRING, "a": STRING, "mid.k": STRING}))
    root = _root(compiled)

    assert [(name, key_bytes) for name, key_bytes, _ in root.properties] == [
        ("zeta", 6),
        ("a", 3),
        ("mid.k", 7),
    ]


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


def test_a_template_is_shared_by_node_number_not_copied_into_its_parents() -> None:
    layers = 18
    defs: dict[str, Any] = {f"D{layers}": {"type": ["string", "null"]}}
    for i in reversed(range(layers)):
        defs[f"D{i}"] = _obj({"a": _ref(f"D{i + 1}"), "b": _ref(f"D{i + 1}")})
    schema = _obj({"d": _ref("D0")}, **{"$defs": defs})

    tracemalloc.start()
    try:
        compiled = structured_schema.compile_schema(schema)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 1024 * 1024
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


def _p4b_schema() -> dict[str, Any]:
    """A list whose elements each hold a required object that expands to a
    template of 17,911 bytes."""
    defs: dict[str, Any] = {
        f"D{i}": _obj({"a" * 64: _ref(f"D{i + 1}"), "b" * 64: _ref(f"D{i + 1}")})
        for i in range(7)
    }
    defs["D7"] = {"type": ["string", "null"]}
    element = _obj({"i": {"type": "integer"}, "b": _ref("D0")})
    return _obj({"items": {"type": "array", "items": element}}, **{"$defs": defs})


def test_the_template_sizes_of_a_chain_of_seven_doubled_layers() -> None:
    compiled = _compile(_p4b_schema())
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


def test_changing_the_schema_after_compiling_reaches_neither_the_copy_nor_the_judgement() -> (
    None
):
    schema = _obj({"a": {"type": "string", "enum": ["p", "q"]}})
    compiled = _compile(schema)

    schema["properties"]["a"]["enum"] = ["z"]
    schema["properties"]["a"]["description"] = "changed"

    copied = ss.model_facing_schema(compiled)
    assert copied["properties"]["a"]["enum"] == ["p", "q", None, MARKER]
    assert "changed" not in json.dumps(copied)
    assert isinstance(ss.judge_instance(compiled, {"a": "p"}), ss.Ready)
    assert ss.judge_instance(compiled, {"a": "z"}) == ss.Err(path="/a", rule="enum")


def test_no_field_of_a_compiled_node_can_be_changed_in_place() -> None:
    compiled = _compile(
        _obj(
            {
                "a": _nullable(_ref("T")),
                "b": {"type": "array", "items": {"type": ["string", "null"]}},
                "c": {"type": "integer", "const": 1},
                "o": _obj({"x": {"type": ["string", "null"]}}),
            },
            **{"$defs": {"T": _obj({"x": {"type": "string", "enum": ["p", "q"]}})}},
        )
    )

    for node in compiled.nodes:
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            assert not isinstance(value, (list, dict, set)), field.name


# --- the size of canonical JSON ----------------------------------------------------


def test_canonical_json_is_measured_in_bytes_not_characters() -> None:
    assert structured_schema.canonical_json_size("é中\U0001f600") == 2 + 2 + 3 + 4
    assert structured_schema.canonical_json_size("é") == 4


def test_canonical_json_writes_compact_separators() -> None:
    value = {"b": [1, 2.5, True, None], "a": {"k": "v"}}

    assert structured_schema.canonical_json_size(value) == len(
        '{"b":[1,2.5,true,null],"a":{"k":"v"}}'
    )


def test_canonical_json_writes_the_smallest_float_and_the_largest() -> None:
    assert structured_schema.canonical_json_size(5e-324) == len("5e-324")
    assert structured_schema.canonical_json_size(1e308) == len("1e+308")
    assert structured_schema.canonical_json_size(-0.0) == len("-0.0")


def test_canonical_json_escapes_quotes_backslashes_and_control_characters() -> None:
    assert structured_schema.canonical_json_size('a"b\\c\n') == len('"a\\"b\\\\c\\n"')
    assert structured_schema.canonical_json_size("\x00") == len('"\\u0000"')
    with pytest.raises(UnicodeEncodeError):
        structured_schema.canonical_json_size("\ud800")


def test_canonical_json_refuses_a_value_that_is_not_finite() -> None:
    with pytest.raises(ValueError):
        structured_schema.canonical_json_size(float("nan"))
    with pytest.raises(ValueError):
        structured_schema.canonical_json_size([float("inf")])


# --- judging ---------------------------------------------------------------------------


CYCLES = {
    "property": _obj({"a": _ref("A")}, **{"$defs": {"A": _obj({"x": _ref("A")})}}),
    "nullable-property": _obj(
        {"a": _ref("A")},
        **{"$defs": {"A": _obj({"x": _nullable(_ref("A"))})}},
    ),
    "list-item": _obj(
        {"a": _ref("L")},
        **{"$defs": {"L": {"type": "array", "items": _ref("L")}}},
    ),
    "alias-pair": _obj({"a": _ref("A")}, **{"$defs": {"A": _ref("B"), "B": _ref("A")}}),
    "alias-to-itself": _obj({"a": _ref("A")}, **{"$defs": {"A": _ref("A")}}),
    "union-of-itself": _obj({"a": _ref("A")}, **{"$defs": {"A": _nullable(_ref("A"))}}),
}


@pytest.mark.timeout(10)
@pytest.mark.parametrize("schema", CYCLES.values(), ids=CYCLES.keys())
def test_compiling_a_schema_that_refers_to_itself_raises(
    schema: dict[str, Any],
) -> None:
    violation = check_schema_structure(schema)
    assert violation is not None and violation.rule == "S9"

    with pytest.raises(ValueError, match="refers to itself"):
        structured_schema.compile_schema(schema)


def _judge(schema: dict[str, Any], instance: Any) -> Any:
    return ss.judge_instance(_compile(schema), instance)


def _same(a: Any, b: Any) -> bool:
    """Equal as JSON text, so ``1``, ``1.0`` and ``true`` differ, and so do the
    same keys in another order."""
    return json.dumps(a) == json.dumps(b)


def _failed(result: Any) -> bool:
    return isinstance(result, ss.Err | ss.TooComplex)


@pytest.mark.parametrize("entry", INSTANCES, ids=lambda e: e["id"])
def test_every_corpus_instance_has_the_outcome_written_for_it(
    entry: dict[str, Any],
) -> None:
    result = _judge(entry["schema"], entry["instance"])
    want = entry["expected"]

    if want["kind"] == "ready":
        assert isinstance(result, ss.Ready)
        assert _same(result.output, want["output"])
        assert (result.missing_count, result.unexpressible_count) == (0, 0)
    elif want["kind"] == "incomplete":
        assert isinstance(result, ss.Incomplete)
        code = (
            "information_missing" if result.missing_fields else "value_not_expressible"
        )
        assert code == want["code"]
        assert result.missing_fields == want["missing_fields"]
        assert result.unexpressible_fields == want["unexpressible_fields"]
        for name in ("missing_count", "unexpressible_count"):
            if name in want:
                assert getattr(result, name) == want[name]
    else:
        assert result == ss.Err(path=want["path"], rule=want["rule"])


def _blanks_to_null(v: Any) -> Any:
    if isinstance(v, str) and v.strip() == "":
        return None
    if isinstance(v, dict):
        return {k: _blanks_to_null(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_blanks_to_null(x) for x in v]
    return v


def _has_misplaced_marker(v: Any) -> bool:
    if isinstance(v, str):
        return MARKER in v and v != MARKER
    if isinstance(v, dict):
        return any(_has_misplaced_marker(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_misplaced_marker(x) for x in v)
    return False


def _agrees_with_jsonschema(
    validator: Draft202012Validator, compiled: Any, instance: Any
) -> bool:
    oracle = validator.is_valid(_blanks_to_null(instance))
    oracle = oracle and not _has_misplaced_marker(instance)
    return (not _failed(ss.judge_instance(compiled, instance))) == oracle


@pytest.mark.parametrize("entry", INSTANCES, ids=lambda e: e["id"])
def test_the_shape_check_agrees_with_jsonschema_on_every_corpus_instance(
    entry: dict[str, Any],
) -> None:
    compiled = _compile(entry["schema"])
    validator = Draft202012Validator(expanded_copy(entry["schema"]))

    assert _agrees_with_jsonschema(validator, compiled, entry["instance"])


@pytest.fixture(scope="module")
def random_cases() -> list[tuple[dict[str, Any], Any, dict[str, Any], RNode]]:
    rng = random.Random(20261104)
    cases = []
    for _ in range(200):
        schema = gen_schema(rng)
        compiled = _compile(schema)
        cases.append((schema, compiled, expanded_copy(schema), root_rnode(schema)))
    return cases


def test_the_shape_check_agrees_with_jsonschema_on_random_instances(
    random_cases: Any,
) -> None:
    rng = random.Random(3)
    accepted = rejected = 0
    for schema, compiled, copy_, rnode in random_cases:
        validator = Draft202012Validator(copy_)
        for _ in range(8):
            instance = gen_instance(rng, rnode)
            assert _agrees_with_jsonschema(validator, compiled, instance), (
                schema,
                instance,
            )
            failed = _failed(ss.judge_instance(compiled, instance))
            accepted += not failed
            rejected += failed
    assert accepted > 600 and rejected > 200


def _integers_are_ints(n: RNode, v: Any) -> bool:
    if v is None:
        return True
    if n.kind == "integer":
        return type(v) is int
    if n.kind == "object":
        return all(_integers_are_ints(c, v[k]) for k, c in n.props)
    if n.kind == "array":
        assert n.items is not None
        return all(_integers_are_ints(n.items, x) for x in v)
    return True


def test_every_ready_output_is_valid_for_the_original_schema(random_cases: Any) -> None:
    rng = random.Random(4)
    ready = 0
    for schema, compiled, _, rnode in random_cases:
        validator = Draft202012Validator(schema)
        for _ in range(8):
            instance = gen_instance(rng, rnode)
            before = copy.deepcopy(instance)
            result = ss.judge_instance(compiled, instance)
            assert _same(instance, before)
            if isinstance(result, ss.Ready):
                ready += 1
                assert validator.is_valid(result.output), (schema, instance)
                assert _integers_are_ints(rnode, result.output)
    assert ready > 400


def test_an_integer_written_as_a_float_is_an_int_in_the_output() -> None:
    schema = _obj({"n": {"type": "integer"}})

    for written, wanted in ((1.0, 1), (1e20, 10**20), (-0.0, 0)):
        result = _judge(schema, {"n": written})
        assert isinstance(result, ss.Ready)
        assert type(result.output["n"]) is int and result.output["n"] == wanted


def _repair(instance: Any, paths: list[str], root: RNode) -> Any:
    """Write a valid value at each listed path, giving an empty object on the
    way the keys it needs."""
    fixed = [copy.deepcopy(instance)]
    for path in paths:
        holder: Any = fixed
        key: Any = 0
        n = root
        for seg in path.split("/")[1:]:
            if n.kind == "object":
                if not isinstance(holder[key], dict):
                    holder[key] = {k: None for k, _ in n.props}
                holder, key, n = holder[key], seg, dict(n.props)[seg]
            else:
                assert n.items is not None
                holder, key, n = holder[key], int(seg), n.items
        holder[key] = synth_valid(n)
    return fixed[0]


def test_the_lists_hold_every_gap_and_only_gaps(random_cases: Any) -> None:
    rng = random.Random(5)
    checked = 0
    for schema, compiled, _, rnode in random_cases:
        validator = Draft202012Validator(schema)
        for _ in range(8):
            instance = gen_instance(rng, rnode, p_bad=0)
            result = ss.judge_instance(compiled, instance)
            if not isinstance(result, ss.Incomplete):
                continue
            if max(result.missing_count, result.unexpressible_count) > 20:
                continue
            checked += 1
            paths = result.missing_fields + result.unexpressible_fields
            again = ss.judge_instance(compiled, _repair(instance, paths, rnode))
            assert isinstance(again, ss.Ready), (schema, instance)
            assert validator.is_valid(again.output)
            for path in result.missing_fields:
                n = lookup(rnode, path)
                assert not n.nullable and n.kind != "object" and not n.single
            for path in result.unexpressible_fields:
                assert lookup(rnode, path).kind == "string"
    assert checked > 250


def _outcome(result: Any) -> tuple[Any, ...]:
    if isinstance(result, ss.Ready):
        return ("ready", json.dumps(result.output))
    assert isinstance(result, ss.Incomplete)
    return (
        "incomplete",
        tuple(result.missing_fields),
        tuple(result.unexpressible_fields),
        result.missing_count,
        result.unexpressible_count,
    )


def test_different_ways_of_writing_one_answer_give_one_outcome(
    random_cases: Any,
) -> None:
    rng = random.Random(6)
    compared = 0
    for schema, compiled, _, rnode in random_cases:
        for _ in range(4):
            decision = gen_decision(rng, rnode, is_root=True)
            results = [
                ss.judge_instance(compiled, render(rng, decision, rnode, is_root=True))
                for _ in range(6)
            ]
            assert not any(_failed(r) for r in results), (schema, decision)
            if max(r.r1_steps + r.expansion_nodes for r in results) > 100_000:
                continue
            if max(getattr(r, "output_bytes", 0) for r in results) > 2**20:
                continue
            compared += 1
            assert len({_outcome(r) for r in results}) == 1, (schema, decision)
    assert compared > 700


ALL_NULLABLE = _obj(
    {
        "verbose": {"type": ["boolean", "null"]},
        "limit": {"type": ["integer", "null"]},
        "empty": _obj({}),
    }
)


def test_the_root_and_the_next_level_down_give_the_same_output() -> None:
    root_result = ss.judge_empty_answer(_compile(ALL_NULLABLE))
    nested = _judge(_obj({"o": ALL_NULLABLE}), {"o": None})

    assert isinstance(root_result, ss.Ready) and isinstance(nested, ss.Ready)
    assert (
        root_result.output
        == nested.output["o"]
        == {"verbose": None, "limit": None, "empty": {}}
    )


def test_an_empty_answer_with_a_required_field_lists_it() -> None:
    schema = _obj(
        {"a": STRING, "b": {"type": ["string", "null"]}, "c": {"type": "integer"}}
    )

    result = ss.judge_empty_answer(_compile(schema))

    assert isinstance(result, ss.Incomplete)
    assert (result.missing_fields, result.missing_count) == (["/a", "/c"], 2)
    assert (result.unexpressible_fields, result.unexpressible_count) == ([], 0)


def test_the_counts_cover_blanks_at_any_type_markers_and_gaps() -> None:
    schema = _obj(
        {
            "a": {"type": "integer"},
            "b": STRING,
            "c": {"type": ["string", "null"]},
            "d": STRING,
        }
    )

    result = _judge(schema, {"a": "", "b": None, "c": " ", "d": MARKER})

    assert isinstance(result, ss.Incomplete)
    assert result.missing_fields == ["/a", "/b"]
    assert result.unexpressible_fields == ["/d"]
    assert (result.missing_count, result.unexpressible_count) == (2, 1)
    assert result.blank_as_null_count == 2


UNSTORABLE = _obj(
    {
        "n": {"type": "number"},
        "m": {"type": ["number", "null"]},
        "s": STRING,
        "e": {"type": "string", "enum": ["a"]},
        "xs": {"type": "array", "items": {"type": "number"}},
    }
)
STORABLE = {"n": 1.5, "m": None, "s": "x", "e": "a", "xs": [1.0]}


@pytest.mark.parametrize(
    ("changed", "path", "rule"),
    [
        ({"n": float("nan")}, "/n", "type"),
        ({"n": float("inf")}, "/n", "type"),
        ({"m": float("-inf")}, "/m", "type"),
        ({"xs": [1.0, float("nan")]}, "/xs/1", "type"),
        ({"s": "a\u0000b"}, "/s", "type"),
        ({"s": "\ud800"}, "/s", "type"),
        ({"e": "a\u0000"}, "/e", "type"),
        ({"s": "\u0000" + MARKER}, "/s", "marker"),
        ({"\u0000": 1}, "", "extra_property"),
    ],
    ids=[
        "nan",
        "infinity",
        "negative-infinity-on-a-nullable-number",
        "nan-in-a-list",
        "nul",
        "lone-surrogate",
        "nul-at-an-enum",
        "marker-is-checked-first",
        "forbidden-character-in-a-key",
    ],
)
def test_values_a_json_column_cannot_store_are_type_errors(
    changed: dict[str, Any], path: str, rule: str
) -> None:
    assert _judge(UNSTORABLE, STORABLE | changed) == ss.Err(path=path, rule=rule)


def test_finite_extremes_and_non_ascii_text_are_still_values() -> None:
    result = _judge(UNSTORABLE, STORABLE | {"n": 1e308, "s": "é中\U0001f600"})

    assert isinstance(result, ss.Ready)
    assert (result.output["n"], result.output["s"]) == (1e308, "é中\U0001f600")


def test_the_output_follows_the_order_of_the_properties_not_the_instance() -> None:
    result = _judge(
        _obj({"a": STRING, "b": STRING, "c": STRING}), {"c": "3", "a": "1", "b": "2"}
    )

    assert isinstance(result, ss.Ready)
    assert list(result.output) == ["a", "b", "c"]


# --- what the worst instances cost ----------------------------------------------------------

LIMIT = 256 * 1024


def _fit(item: str) -> tuple[Any, int]:
    """An instance ``{"xs":[item,item,...]}`` that fills the argument limit."""
    count = (LIMIT - 20) // (len(item) + 1)
    return json.loads('{"xs":[' + ",".join([item] * count) + "]}"), count


def _nodes(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_nodes(v) for v in value.values())
    if isinstance(value, list):
        return 1 + sum(_nodes(v) for v in value)
    return 1


# The cost of a judgement is pinned by the step, node and byte counts each test
# asserts. This bound only catches a slowdown by an order of magnitude; the
# slowest case takes well under two seconds on a loaded CI runner.
BACKSTOP_SECONDS = 10.0


def _judge_with_backstop(schema: dict[str, Any], instance: Any) -> Any:
    start = time.perf_counter()
    compiled = _compile(schema)
    assert time.perf_counter() - start < BACKSTOP_SECONDS
    start = time.perf_counter()
    result = ss.judge_instance(compiled, instance)
    assert time.perf_counter() - start < BACKSTOP_SECONDS
    return result


def _layered(leaf: dict[str, Any]) -> dict[str, Any]:
    """1000 leaves behind three layers of ten, as the ``big`` property of the
    elements of ``xs``."""
    defs = {
        "L3": _obj({f"c{i}": _ref("L2") for i in range(10)}),
        "L2": _obj({f"b{i}": _ref("L1") for i in range(10)}),
        "L1": _obj({f"a{i}": _ref("Leaf") for i in range(10)}),
        "Leaf": leaf,
    }
    element = _obj({"a": {"type": "integer"}, "big": _ref("L3")})
    return _obj({"xs": {"type": "array", "items": element}}, **{"$defs": defs})


def test_a_flat_list_of_integers_costs_one_step_per_node() -> None:
    schema = _obj({"xs": {"type": "array", "items": {"type": "integer"}}})
    instance, _ = _fit("0")

    result = _judge_with_backstop(schema, instance)

    assert isinstance(result, ss.Ready)
    assert result.r1_steps == _nodes(instance)


@pytest.mark.parametrize(
    ("items", "item", "first"),
    [
        ({"type": ["integer", "null"]}, "null", 1),
        ({"type": "array", "items": {"type": "integer"}}, "[]", None),
    ],
    ids=["nullable-items", "empty-lists"],
)
def test_lists_of_nulls_and_of_empty_lists_cost_one_step_per_node(
    items: dict[str, Any], item: str, first: Any
) -> None:
    instance, _ = _fit(item)
    if first is not None:
        instance["xs"][0] = first

    result = _judge_with_backstop(
        _obj({"xs": {"type": "array", "items": items}}), instance
    )

    assert isinstance(result, ss.Ready)
    assert result.r1_steps == _nodes(instance)


def test_a_thousand_empty_leaves_per_element_cost_the_size_of_the_template_only() -> (
    None
):
    nullable = _layered({"type": ["string", "null"]})
    required = _layered(STRING)
    instance, count = _fit('{"a":1,"big":null}')

    result = _judge_with_backstop(nullable, instance)
    assert isinstance(result, ss.TooComplex) and result.stage == "r4"
    assert result.r1_steps == _nodes(instance)

    result = _judge_with_backstop(required, instance)
    assert isinstance(result, ss.Incomplete)
    assert result.missing_count == count * 1000 == 13_796_000
    assert result.r1_steps == _nodes(instance)


def test_a_small_template_per_element_is_still_ready() -> None:
    small = _obj(
        {"p": {"type": ["string", "null"]}, "q": {"type": ["integer", "null"]}}
    )
    element = _obj({"a": {"type": "integer"}, "o": small})
    instance, count = _fit('{"a":1,"o":null}')

    result = _judge_with_backstop(
        _obj({"xs": {"type": "array", "items": element}}), instance
    )

    assert isinstance(result, ss.Ready)
    assert result.r1_steps == _nodes(instance)
    assert result.expansion_nodes == 3 * count


def _elements(
    count: int, leaves: dict[str, Any], *, with_id: bool
) -> tuple[dict[str, Any], Any]:
    sub = _obj({f"f{i}": leaves for i in range(1, 101)})
    element = _obj(
        ({"id": {"type": "integer"}} if with_id else {}) | {"sub": _ref("Sub")}
    )
    schema = _obj(
        {"items": {"type": "array", "items": element}}, **{"$defs": {"Sub": sub}}
    )
    one = {"id": 7, "sub": None} if with_id else {"sub": None}
    return schema, {"items": [dict(one) for _ in range(count)]}


def test_elements_whose_only_property_is_empty_make_the_whole_list_empty() -> None:
    schema, instance = _elements(2000, STRING, with_id=False)
    result = _judge_with_backstop(schema, instance)
    assert isinstance(result, ss.Incomplete)
    assert result.missing_fields == ["/items"] and result.r1_steps == 4002

    nullable = copy.deepcopy(schema)
    nullable["properties"]["items"]["type"] = ["array", "null"]
    result = _judge_with_backstop(nullable, instance)
    assert isinstance(result, ss.Ready) and result.output == {"items": None}
    assert result.r1_steps == 4002


def test_elements_with_a_gap_in_their_empty_part_are_incomplete_and_cheap() -> None:
    leaves = {"type": ["string", "null"]}
    schema, instance = _elements(2000, leaves, with_id=True)
    schema["$defs"]["Sub"]["properties"]["f1"] = STRING

    result = _judge_with_backstop(schema, instance)

    assert isinstance(result, ss.Incomplete) and result.r1_steps == 6002
    assert (result.missing_count, result.expansion_nodes) == (2000, 0)


def test_only_the_template_of_an_empty_required_object_adds_expansion_nodes() -> None:
    schema = _obj(
        {
            "a": {"type": ["string", "null"]},
            "o": _obj({"s": STRING}),
            "k": {"type": "string", "const": "k"},
            "b": STRING,
        }
    )

    result = _judge(schema, {"a": None, "o": None, "k": None, "b": "x"})

    assert isinstance(result, ss.Incomplete) and result.missing_fields == ["/o/s"]
    assert result.expansion_nodes == 0


def test_elements_whose_empty_part_is_too_big_to_return_are_too_complex() -> None:
    schema, instance = _elements(2000, {"type": ["string", "null"]}, with_id=True)
    result = _judge_with_backstop(schema, instance)
    assert isinstance(result, ss.TooComplex) and result.stage == "r4"
    assert result.r1_steps + result.expansion_nodes == 208_002


def test_the_template_is_not_copied_when_the_result_is_too_complex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object) -> None:
        raise AssertionError("the template was copied")

    monkeypatch.setattr(ss, "_copy_template", fail)
    schema, instance = _elements(2000, {"type": ["string", "null"]}, with_id=True)

    assert isinstance(_judge(schema, instance), ss.TooComplex)


def test_a_result_that_is_incomplete_is_not_checked_for_size() -> None:
    schema, instance = _elements(2000, {"type": ["string", "null"]}, with_id=True)
    schema["properties"]["z"] = STRING
    schema["required"].append("z")
    instance["z"] = None

    result = _judge_with_backstop(schema, instance)

    assert isinstance(result, ss.Incomplete) and result.missing_fields == ["/z"]
    assert result.r1_steps + result.expansion_nodes == 208_003


@pytest.mark.parametrize(("count", "ready"), [(199_998, True), (199_999, False)])
def test_the_shape_check_stops_just_past_200_000_steps(count: int, ready: bool) -> None:
    schema = _obj({"a": {"type": "array", "items": {"type": "integer"}}})
    result = _judge_with_backstop(schema, {"a": [0] * count})

    if ready:
        assert isinstance(result, ss.Ready) and result.r1_steps == 200_000
    else:
        assert isinstance(result, ss.TooComplex) and result.stage == "r1"
        assert result.r1_steps == 200_001


def test_the_densest_instance_within_256_kib_is_under_the_step_limit() -> None:
    schema = _obj({"a": {"type": "array", "items": {"type": "integer"}}})
    instance = {"a": [0] * 131_068}
    assert len(json.dumps(instance, separators=(",", ":"))) <= LIMIT

    result = _judge_with_backstop(schema, instance)

    assert isinstance(result, ss.Ready) and result.r1_steps == 131_070


def _fill(text: str) -> int:
    return (LIMIT - 300) // (len(text) + 1)


def _worst_schema() -> dict[str, Any]:
    defs = {
        "E": {"type": "integer", "enum": list(range(500))},
        "N": _nullable(STRING),
        "G": _obj({f"a{i}": _ref("N") for i in range(10)}),
    }
    defs["H"] = _obj({f"a{i}": _ref("G") for i in range(10)})
    props = {f"e{i}": _ref("E") for i in range(9)}
    props["xs"] = {"type": "array", "items": _ref("E")}
    props |= {f"h{i}": _ref("H") for i in range(6)} | {
        f"g{i}": _ref("G") for i in range(3)
    }
    return _obj(props, **{"$defs": defs})


def _worst_instance(last: int) -> dict[str, Any]:
    instance: dict[str, Any] = {f"e{i}": i for i in range(9)}
    instance["xs"] = [499 - i % 400 for i in range(_fill("499"))]
    instance["xs"][-1] = last
    instance |= {f"h{i}": None for i in range(6)} | {f"g{i}": None for i in range(3)}
    return instance


def test_the_worst_schema_meets_a_256_kib_instance_in_one_step_per_node() -> None:
    schema = _worst_schema()
    metrics = structured_schema.measure_expansion(schema)
    assert (metrics.nodes, metrics.enum_values) == (1971, 5000)
    for last in (7, 500):
        text = json.dumps(_worst_instance(last), separators=(",", ":"))
        assert 250_000 < len(text.encode()) <= LIMIT
    valid = _judge_with_backstop(schema, _worst_instance(7))
    assert isinstance(valid, ss.Ready)
    assert valid.r1_steps == _nodes(_worst_instance(7))
    assert valid.expansion_nodes == 6 * 111 + 3 * 11

    invalid = _judge_with_backstop(schema, _worst_instance(500))
    assert invalid == ss.Err(path=f"/xs/{_fill('499') - 1}", rule="enum")


@pytest.mark.parametrize(
    ("schema", "instance", "expected"),
    [
        (
            _obj({"xs": {"type": "array", "items": {"type": "integer"}}}),
            {"xs": list(range(1000))},
            "one-step-per-node",
        ),
        (
            _obj({"xs": {"type": "array", "items": {"type": "integer"}}}),
            {"xs": ["x" * 100_000] * 3},
            ss.Err(path="/xs/0", rule="type"),
        ),
        (
            _obj({"xs": {"type": "array", "items": _obj({})}}),
            {"xs": [{f"k{i}": 1 for i in range(10_000)}]},
            ss.Err(path="/xs/0", rule="extra_property"),
        ),
    ],
    ids=["many-integers", "long-strings", "many-keys"],
)
def test_other_worst_instances_stop_at_the_first_error_or_cost_one_step_per_node(
    schema: dict[str, Any], instance: Any, expected: Any
) -> None:
    result = _judge_with_backstop(schema, instance)

    if expected == "one-step-per-node":
        assert isinstance(result, ss.Ready) and result.r1_steps == _nodes(instance)
    else:
        assert result == expected


def test_a_256_kib_list_of_values_of_a_500_value_enum_costs_one_step_per_node() -> None:
    values = [f"{i:03d}-" + "v" * 8 for i in range(500)]
    schema = _obj(
        {"xs": {"type": "array", "items": {"type": "string", "enum": values}}}
    )
    count = (LIMIT - 20) // (len(values[-1]) + 3)

    instance = {"xs": [values[-1 - i % 500] for i in range(count)]}

    result = _judge_with_backstop(schema, instance)

    assert isinstance(result, ss.Ready)
    assert result.r1_steps == _nodes(instance)


def test_integers_with_one_hash_get_enum_keys_with_as_many_hashes() -> None:
    same_hash = [7 + i * (2**61 - 1) for i in range(500)]
    assert len({hash(v) for v in same_hash}) == 1
    schema = _obj(
        {"xs": {"type": "array", "items": {"type": "integer", "enum": same_hash}}}
    )
    compiled = _compile(schema)
    items = _items(compiled, _child(compiled, _root(compiled), "xs"))

    assert items.enum_keys is not None
    assert len({hash(key) for key in items.enum_keys}) == 500

    result = _judge_with_backstop(schema, {"xs": same_hash[-100:] * 8})
    assert isinstance(result, ss.Ready) and result.r1_steps == 802


# --- the size of the result ----------------------------------------------------------------

SPECIAL = _obj(
    {
        "s": {"type": ["string", "null"]},
        "n": {"type": "number"},
        "i": {"type": "integer"},
        "kind": {"type": "string", "const": "k\u00e9"},
        "flag": {"type": "boolean", "const": False},
        "opts": _obj(
            {"v": {"type": ["boolean", "null"]}, "w": {"type": ["integer", "null"]}}
        ),
    }
)
SPECIAL_WRITINGS = [
    {"s": "\u00e9\u4e2d\U0001f600", "n": 5e-324, "i": 1e20},
    {"s": 'q"\\\n\t\x01', "n": 1e308, "i": 2.0},
    {"s": None, "n": -0.0, "i": -3},
    {"s": "plain", "n": 1.5, "i": 0, "kind": None, "flag": None, "opts": None},
]


@pytest.mark.parametrize("writing", SPECIAL_WRITINGS)
def test_the_counted_bytes_equal_the_bytes_of_the_serialized_output(
    writing: dict[str, Any],
) -> None:
    instance = {
        "kind": "k\u00e9",
        "flag": None,
        "opts": {"v": True, "w": None},
    } | writing
    instance = {k: instance.get(k) for k in SPECIAL["properties"]}

    result = _judge(SPECIAL, instance)

    assert isinstance(result, ss.Ready)
    assert result.output_bytes == ss.canonical_json_size(result.output)


def test_the_counted_bytes_equal_the_serialized_bytes_of_random_results(
    random_cases: Any,
) -> None:
    rng = random.Random(7)
    ready = 0
    for _, compiled, _, rnode in random_cases:
        for _ in range(8):
            result = ss.judge_instance(compiled, gen_instance(rng, rnode))
            if isinstance(result, ss.Ready):
                ready += 1
                assert result.output_bytes == ss.canonical_json_size(result.output)
    assert ready > 400


def test_the_judgement_serializes_no_object_or_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compile(SPECIAL)
    instance = SPECIAL_WRITINGS[0] | {"kind": None, "flag": None, "opts": None}
    real_dumps = json.dumps

    def scalar_only_dumps(value: Any, *args: Any, **kwargs: Any) -> str:
        assert not isinstance(value, (dict, list)), "the output was serialized"
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", scalar_only_dumps)
    result = ss.judge_instance(compiled, instance)
    monkeypatch.undo()

    assert isinstance(result, ss.Ready)


def test_an_empty_answer_counts_the_bytes_of_its_output() -> None:
    result = ss.judge_empty_answer(
        _compile(_obj({"o": ALL_NULLABLE, "k": SPECIAL["properties"]["kind"]}))
    )

    assert isinstance(result, ss.Ready)
    assert result.output_bytes == ss.canonical_json_size(result.output)


def test_a_small_result_is_counted_to_the_byte() -> None:
    options = _obj(
        {
            "verbose": {"type": ["boolean", "null"]},
            "limit": {"type": ["integer", "null"]},
        }
    )
    result = _judge(_obj({"name": STRING, "opt": options}), {"name": "Bo", "opt": None})

    assert isinstance(result, ss.Ready) and result.output_bytes == 49


@pytest.mark.parametrize("count", [1, 2, 117])
def test_a_list_of_big_empty_objects_is_counted_to_the_byte(count: int) -> None:
    result = _judge(_p4b_schema(), {"items": [{"i": 1, "b": None}] * count})

    assert isinstance(result, ss.Ready)
    assert (
        result.output_bytes
        == 17_924 * count + 11
        == ss.canonical_json_size(result.output)
    )
    if count > 1:
        first, second = result.output["items"][:2]
        assert first["b"] is not second["b"]
    assert count != 117 or result.output_bytes == 2_097_119


def test_a_result_over_2_mib_is_too_complex_even_when_its_nodes_are_few() -> None:
    result = _judge(_p4b_schema(), {"items": [{"i": 1, "b": None}] * 118})

    assert result == ss.TooComplex(
        stage="r4",
        r1_steps=2 + 118 * 3,
        expansion_nodes=118 * 255,
        output_bytes=2_115_043,
    )


def test_775_such_elements_pass_the_node_check_and_fail_the_byte_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object) -> None:
        raise AssertionError("the template was copied")

    monkeypatch.setattr(ss, "_copy_template", fail)

    result = _judge(_p4b_schema(), {"items": [{"i": 1, "b": None}] * 775})

    assert isinstance(result, ss.TooComplex) and result.stage == "r4"
    assert result.r1_steps + result.expansion_nodes == 199_952
    assert result.output_bytes == 13_891_111


@pytest.mark.parametrize(("extra", "ready"), [(2, True), (3, False)])
def test_the_node_check_alone_turns_a_result_away_at_200_001_positions(
    extra: int, ready: bool
) -> None:
    element = _obj(
        {"id": {"type": "integer"}, "sub": _obj({"leaf": {"type": ["string", "null"]}})}
    )
    schema = _obj(
        {
            "items": {"type": "array", "items": element},
            "ys": {"type": "array", "items": {"type": "integer"}},
        }
    )
    instance = {"items": [{"id": 1, "sub": None}] * 39_999, "ys": [0] * extra}

    result = _judge(schema, instance)

    if ready:
        assert isinstance(result, ss.Ready)
    else:
        assert isinstance(result, ss.TooComplex) and result.stage == "r4"
        assert 1_000_000 < result.output_bytes < ss.MAX_RESULT_BYTES
    assert result.r1_steps + result.expansion_nodes == 199_998 + extra


@pytest.mark.parametrize(("size", "ready"), [(2_097_144, True), (2_097_145, False)])
def test_the_byte_check_alone_turns_a_result_away_past_2_mib(
    size: int, ready: bool
) -> None:
    result = _judge(_obj({"s": STRING}), {"s": "x" * size})

    assert isinstance(result, ss.Ready if ready else ss.TooComplex)
    assert result.output_bytes == 8 + size


def test_a_tuple_in_the_instance_is_a_type_error() -> None:
    schema = _obj({"xs": {"type": "array", "items": STRING}})

    result = _judge(schema, {"xs": ("a\u0000",)})

    assert result == ss.Err(path="/xs", rule="type")


def test_an_extra_property_counts_the_keys_beyond_the_schema() -> None:
    schema = _obj({"a": STRING, "o": _obj({"b": STRING})})
    instance = {"a": "x", "o": {"b": "y", "p": 1, "q": 2, "r": 3}}

    result = _judge(schema, instance)

    assert result == ss.Err(path="/o", rule="extra_property")
    assert result.extra_count == 3
    assert _judge(schema, {**instance, "o": {"b": "y"}, "z": 1}).extra_count == 1
    assert _judge(schema, {"a": 5, "o": None}).extra_count == 0


# --- keeping the value of an incomplete result ----------------------------------

_NEW_INCOMPLETE_FIELDS = {"output", "output_bytes", "output_dropped"}


def _without_new_fields(result: Any) -> Any:
    return dataclasses.replace(
        result, output=None, output_bytes=0, output_dropped=False
    )


def test_keep_output_changes_only_partial_fields() -> None:
    incomplete = 0
    for entry in INSTANCES:
        compiled = _compile(entry["schema"])
        plain = ss.judge_instance(compiled, entry["instance"])
        kept = ss.judge_instance(compiled, entry["instance"], keep_output=True)
        assert type(kept) is type(plain)
        if isinstance(plain, ss.Incomplete):
            incomplete += 1
            assert plain.output is None and plain.output_bytes == 0
            assert plain.output_dropped is False
            assert _without_new_fields(kept) == plain
        else:
            assert kept == plain
    assert incomplete > 0


def _walk_nulls(node: Any, value: Any, pointer: str = "") -> Any:
    """Every ``(pointer, node)`` at which ``value`` holds null, read from the
    node tree built straight from the schema."""
    if value is None:
        yield pointer, node
    elif node.kind == "object":
        for name, child in node.props:
            yield from _walk_nulls(
                child,
                value[name],
                pointer + "/" + name.replace("~", "~0").replace("/", "~1"),
            )
    elif node.kind == "array":
        for position, item in enumerate(value):
            yield from _walk_nulls(node.items, item, f"{pointer}/{position}")


def _is_null_or_below_null(output: Any, path: str) -> bool:
    current = output
    for segment in path.split("/")[1:]:
        if current is None:
            return True
        current = (
            current[int(segment)]
            if isinstance(current, list)
            else current[segment.replace("~1", "/").replace("~0", "~")]
        )
    return current is None


def _kept_cases(random_cases: Any) -> Any:
    """``(schema, rnode, compiled, instance)`` of every corpus instance and
    every random instance that judges to an incomplete result with a value."""
    rng = random.Random(11)
    for entry in INSTANCES:
        compiled = _compile(entry["schema"])
        yield entry["schema"], root_rnode(entry["schema"]), compiled, entry["instance"]
    for schema, compiled, _, rnode in random_cases:
        for _ in range(8):
            yield schema, rnode, compiled, gen_instance(rng, rnode, p_bad=0)


def test_partial_output_gap_paths_size_and_nulls(
    random_cases: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked = 0
    for schema, rnode, compiled, instance in _kept_cases(random_cases):
        result = ss.judge_instance(compiled, instance, keep_output=True)
        if not isinstance(result, ss.Incomplete) or result.output is None:
            continue
        checked += 1
        output = result.output
        # Every listed path is null, or lies below a null.
        for path in result.missing_fields + result.unexpressible_fields:
            assert _is_null_or_below_null(output, path), (schema, instance, path)
        # The size is that of the canonical JSON of the value.
        assert ss.canonical_json_size(output) == result.output_bytes
        # A null sits where the schema allows null, or at a gap, listed or not.
        with monkeypatch.context() as patched:
            patched.setattr(ss, "MAX_LISTED_FIELDS", 10**6)
            every = ss.judge_instance(
                ss.compile_schema(schema), instance, keep_output=True
            )
        assert isinstance(every, ss.Incomplete)
        assert every.output == output
        gaps = every.missing_fields + every.unexpressible_fields
        for pointer, node in _walk_nulls(rnode, output):
            assert node.nullable or any(
                gap == pointer or gap.startswith(pointer + "/") for gap in gaps
            ), (schema, instance, pointer)
    assert checked > 100


_ITEM = _obj(
    {
        "key": STRING,
        "level": {"type": "string", "enum": ["low", "high"]},
        "note": _nullable(STRING),
    }
)
_ITEMS = _obj({"items": {"type": "array", "items": _ITEM}})
_FULL = {"key": "k", "level": "low", "note": None}
_GAP_SCHEMA = _obj(
    {
        "a": STRING,
        "g": STRING,
        "n": _nullable(STRING),
        "o": _obj({"p": STRING, "q": STRING}),
        "t": _obj({"x": _nullable(STRING)}),
        "k": {"type": "string", "enum": ["only"]},
        "l": {"type": "array", "items": STRING},
        "m": {"type": "string"},
        "i": {"type": "integer"},
    }
)
_GAP_BASE = {
    "a": "w",
    "g": None,
    "n": None,
    "o": {"p": "p", "q": "q"},
    "t": {"x": "x"},
    "k": "only",
    "l": ["s"],
    "m": "m",
    "i": 3,
}

# (id, schema, instance, whether the root is a gap). One cell per row of the
# table of positions in ``_decide``, then the shapes around them.
_PARTIAL_CELLS: list[tuple[str, dict[str, Any], Any, bool]] = [
    ("empty-nullable", _GAP_SCHEMA, _GAP_BASE, False),
    ("empty-object-with-missing-values", _GAP_SCHEMA, {**_GAP_BASE, "o": None}, False),
    (
        "empty-object-without-missing-values",
        _GAP_SCHEMA,
        {**_GAP_BASE, "t": None},
        False,
    ),
    ("empty-with-one-allowed-value", _GAP_SCHEMA, {**_GAP_BASE, "k": None}, False),
    (
        "empty-scalar-and-list",
        _GAP_SCHEMA,
        {**_GAP_BASE, "m": None, "l": None},
        False,
    ),
    ("marker", _GAP_SCHEMA, {**_GAP_BASE, "m": MARKER}, False),
    (
        "object-and-list-of-values",
        _GAP_SCHEMA,
        {**_GAP_BASE, "l": ["s", "u"]},
        False,
    ),
    ("integer-written-as-float", _GAP_SCHEMA, {**_GAP_BASE, "i": 3.0}, False),
    (
        "nested-list",
        _ITEMS,
        {"items": [_FULL, {"key": "k", "level": None, "note": "n"}, _FULL]},
        False,
    ),
    ("root-empty", _ITEMS, {"items": None}, True),
    (
        "root-with-another-property",
        _obj({"batch": STRING, "items": {"type": "array", "items": _ITEM}}),
        {"batch": "b", "items": None},
        False,
    ),
    (
        "one-allowed-value-in-a-list-item",
        _obj(
            {
                "items": {
                    "type": "array",
                    "items": _obj(
                        {
                            "kind": {"type": "string", "enum": ["a"]},
                            "v": STRING,
                            "w": STRING,
                        }
                    ),
                }
            }
        ),
        {
            "items": [
                {"kind": None, "v": "y", "w": "w"},
                {"kind": "a", "v": None, "w": "w"},
            ]
        },
        False,
    ),
    (
        "template-in-a-list-item",
        _obj(
            {
                "items": {
                    "type": "array",
                    "items": _obj(
                        {"v": STRING, "extra": _obj({"n": _nullable(STRING)})}
                    ),
                }
            }
        ),
        {"items": [{"v": "y", "extra": None}, {"v": None, "extra": {"n": "z"}}]},
        False,
    ),
]
assert len(_PARTIAL_CELLS) == 13


@pytest.mark.parametrize(
    ("schema", "instance", "root_is_gap"),
    [cell[1:] for cell in _PARTIAL_CELLS],
    ids=[cell[0] for cell in _PARTIAL_CELLS],
)
def test_partial_output_matches_filled_ready(
    schema: dict[str, Any], instance: Any, root_is_gap: bool
) -> None:
    compiled = _compile(schema)
    result = ss.judge_instance(compiled, instance, keep_output=True)

    assert isinstance(result, ss.Incomplete)
    assert result.output_dropped is False
    if root_is_gap:
        assert result.output is None and result.output_bytes == 0
        return
    assert result.output is not None
    gaps = result.missing_fields + result.unexpressible_fields
    rnode = root_rnode(schema)
    filled = ss.judge_instance(compiled, _repair(instance, gaps, rnode))
    assert isinstance(filled, ss.Ready)

    def same_outside_gaps(kept: Any, whole: Any, pointer: str) -> None:
        if kept is None:
            below = any(g == pointer or g.startswith(pointer + "/") for g in gaps)
            assert below or whole is None, pointer
        elif isinstance(kept, dict):
            for name in kept:
                same_outside_gaps(kept[name], whole[name], f"{pointer}/{name}")
        elif isinstance(kept, list):
            assert len(kept) == len(whole)
            for position, item in enumerate(kept):
                same_outside_gaps(item, whole[position], f"{pointer}/{position}")
        else:
            assert kept == whole and type(kept) is type(whole), pointer

    same_outside_gaps(result.output, filled.output, "")


_TEMPLATED = _obj({"e": _obj({"n": _nullable(STRING)}), "w": STRING})


def _padded_schema() -> dict[str, Any]:
    return _obj(
        {
            "pad": STRING,
            "g": STRING,
            "es": {"type": "array", "items": _TEMPLATED},
            "xs": {"type": "array", "items": {"type": "integer"}},
        }
    )


def _padded(pad: int, templated: int, filler: int) -> dict[str, Any]:
    return {
        "pad": "x" * pad,
        "g": None,
        "es": [{"e": None, "w": "x"}] * templated,
        "xs": [1] * filler,
    }


def _lists_of_gaps(count: int, marker: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = _obj({"a": STRING, **{f"f{i}": STRING for i in range(count)}})
    instance: dict[str, Any] = {"a": "w"}
    instance.update({f"f{i}": MARKER if marker else None for i in range(count)})
    return schema, instance


def test_partial_output_keep_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _padded_schema()
    compiled = _compile(schema)
    probe = ss.judge_instance(compiled, _padded(1, 1, 0), keep_output=True)
    assert isinstance(probe, ss.Incomplete) and probe.output is not None
    pad = ss.MAX_RESULT_BYTES - probe.output_bytes + 1
    # A templated element costs steps and expansion nodes; plain integers
    # (one step each) make up what is left to reach the limit exactly.
    two = ss.judge_instance(compiled, _padded(1, 2, 0), keep_output=True)
    cost = (two.r1_steps + two.expansion_nodes) - (
        probe.r1_steps + probe.expansion_nodes
    )
    spare = ss.MAX_JUDGED_STEPS - (probe.r1_steps + probe.expansion_nodes)
    templated = 1 + spare // cost
    filler = spare % cost

    cells = [
        ("bytes-at-limit", _padded(pad, 1, 0), False),
        ("bytes-over-limit", _padded(pad + 1, 1, 0), True),
        ("nodes-at-limit", _padded(1, templated, filler), False),
        ("nodes-over-limit", _padded(1, templated, filler + 1), True),
    ]
    for name, instance, dropped in cells:
        kept = ss.judge_instance(compiled, instance, keep_output=True)
        plain = ss.judge_instance(compiled, instance)
        assert isinstance(kept, ss.Incomplete), name
        assert kept.output_dropped is dropped, name
        assert (kept.output is None) is dropped, name
        assert _without_new_fields(kept) == plain, name
        assert (kept.missing_count, kept.missing_fields) == (
            plain.missing_count,
            plain.missing_fields,
        )
    over = ss.judge_instance(compiled, _padded(pad + 1, 1, 0), keep_output=True)
    assert over.output_bytes == ss.MAX_RESULT_BYTES + 1

    for marker in (False, True):
        many_schema, many = _lists_of_gaps(21, marker)
        kept = ss.judge_instance(_compile(many_schema), many, keep_output=True)
        listed = kept.unexpressible_fields if marker else kept.missing_fields
        counted = kept.unexpressible_count if marker else kept.missing_count
        assert (len(listed), counted) == (20, 21)
        assert kept.output is not None and kept.output_dropped is False
