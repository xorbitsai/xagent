"""When a run needs a structured answer, and how the model's arguments are read."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import pytest

from xagent.core.agent import structured_output as so
from xagent.core.agent import structured_schema as ss

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


def _fmt(schema: dict[str, Any] | None = None) -> Any:
    return so.StructuredOutputFormat(
        name="shape",
        description=None,
        schema=schema or _obj({"a": STRING, "b": NULLABLE_STRING}),
    )


def _result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "completed",
        "output": "The answer is 42.",
        "success": True,
        "completion_outcome": "completed",
    }
    result.update(overrides)
    return result


def _settled(status: str, code: str | None = None, **extra: Any) -> dict[str, Any]:
    if status == "ready":
        return {"status": "ready", "output": extra["output"], "reason": None}
    reason: dict[str, Any] = {"code": code}
    reason.update({k: v for k, v in extra.items() if v})
    return {"status": status, "output": None, "reason": reason}


# --- the format and its two sentinels ----------------------------------------


def test_the_invalid_and_read_failed_sentinels_are_formats_told_apart_by_identity():
    for sentinel in (
        so.StructuredOutputFormat.INVALID,
        so.StructuredOutputFormat.READ_FAILED,
    ):
        assert isinstance(sentinel, so.StructuredOutputFormat)
    assert (
        so.StructuredOutputFormat.INVALID is not so.StructuredOutputFormat.READ_FAILED
    )
    assert so.StructuredOutputFormat.INVALID != so.StructuredOutputFormat.READ_FAILED
    assert so.StructuredOutputFormat.INVALID is not _fmt()


# --- the answer text -----------------------------------------------------------


def test_a_string_output_is_used_as_it_is():
    assert so.answer_text({"output": " keep  this\n"}) == " keep  this\n"


def test_a_dict_or_list_output_is_written_compactly_without_escaping_non_ascii():
    assert so.answer_text({"output": {"a": "中", "b": [1, 2]}}) == (
        '{"a":"中","b":[1,2]}'
    )
    assert so.answer_text({"output": ["x", {"y": None}]}) == '["x",{"y":null}]'


def test_a_missing_or_null_output_is_the_empty_string():
    assert so.answer_text({}) == ""
    assert so.answer_text({"output": None}) == ""


def test_any_other_output_is_turned_into_text():
    assert so.answer_text({"output": 42}) == "42"


# --- the eligibility table -------------------------------------------------------

SKIPPED = [
    pytest.param(_result(success=False), id="not-successful"),
    pytest.param(_result(success=None), id="success-missing"),
    pytest.param(_result(status="waiting_for_user"), id="waiting"),
    pytest.param(_result(status="interrupted"), id="interrupted"),
    pytest.param(_result(status="paused"), id="paused"),
    pytest.param(_result(injection_outcome_unknown=True), id="injection-unknown"),
]


@pytest.mark.parametrize("result", SKIPPED)
def test_a_run_that_did_not_finish_normally_is_skipped(result):
    assert so.structured_output_eligibility(result, _fmt()) is so.SKIP


def test_no_format_is_skipped():
    assert so.structured_output_eligibility(_result(), None) is so.SKIP


def test_an_invalid_stored_format_fails_without_a_model():
    assert so.structured_output_eligibility(
        _result(), so.StructuredOutputFormat.INVALID
    ) == _settled("failed", "invalid_stored_format")


def test_a_failed_read_of_the_format_is_an_internal_error():
    assert so.structured_output_eligibility(
        _result(), so.StructuredOutputFormat.READ_FAILED
    ) == _settled("failed", "internal_error")


def test_a_partial_run_is_incomplete_without_a_model():
    assert so.structured_output_eligibility(
        _result(completion_outcome="partial"), _fmt()
    ) == _settled("incomplete", "agent_partial")


def test_a_blocked_run_is_incomplete_without_a_model():
    assert so.structured_output_eligibility(
        _result(completion_outcome="blocked"), _fmt()
    ) == _settled("incomplete", "agent_blocked")


@pytest.mark.parametrize("outcome", ["completed", None])
def test_a_completed_run_with_an_answer_is_extracted(outcome):
    result = _result(completion_outcome=outcome)
    assert so.structured_output_eligibility(result, _fmt()) is so.EXTRACT


def test_a_dict_answer_is_extracted():
    assert (
        so.structured_output_eligibility(_result(output={"k": 1}), _fmt()) is so.EXTRACT
    )


def test_the_rows_are_taken_from_the_top():
    invalid = so.StructuredOutputFormat.INVALID
    read_failed = so.StructuredOutputFormat.READ_FAILED
    assert so.structured_output_eligibility(
        _result(completion_outcome="partial"), invalid
    ) == _settled("failed", "invalid_stored_format")
    assert so.structured_output_eligibility(
        _result(completion_outcome="blocked"), read_failed
    ) == _settled("failed", "internal_error")
    assert so.structured_output_eligibility(_result(success=False), invalid) is so.SKIP
    assert so.structured_output_eligibility(
        _result(completion_outcome="partial", output=""), _fmt()
    ) == _settled("incomplete", "agent_partial")


@pytest.mark.parametrize("output", ["", "  \n", None, " No output provided\n"])
def test_an_empty_answer_is_judged_without_a_model(output):
    fmt = _fmt(_obj({"a": STRING, "c": {"type": "integer"}, "b": NULLABLE_STRING}))
    assert so.structured_output_eligibility(_result(output=output), fmt) == _settled(
        "incomplete", "information_missing", missing_fields=["/a", "/c"]
    )


def test_an_empty_answer_with_only_nullable_properties_is_ready_with_nulls():
    fmt = _fmt(_obj({"a": NULLABLE_STRING, "b": {"type": ["integer", "null"]}}))
    assert so.structured_output_eligibility(_result(output=""), fmt) == _settled(
        "ready", output={"a": None, "b": None}
    )


def test_an_empty_answer_fills_the_only_allowed_value():
    fmt = _fmt(
        _obj({"kind": {"type": "string", "const": "person"}, "a": NULLABLE_STRING})
    )
    assert so.structured_output_eligibility(_result(output=""), fmt) == _settled(
        "ready", output={"kind": "person", "a": None}
    )


def test_a_too_complex_empty_answer_is_a_result_too_large_failure(monkeypatch):
    monkeypatch.setattr(
        so, "judge_empty_answer", lambda compiled: ss.TooComplex("r4", 1, 2, 3)
    )
    assert so.structured_output_eligibility(_result(output=""), _fmt()) == _settled(
        "failed", "result_too_large"
    )


def test_the_schema_of_the_format_is_not_changed():
    schema = _obj({"a": STRING})
    before = json.dumps(schema)
    so.structured_output_eligibility(_result(output=""), _fmt(schema))
    assert json.dumps(schema) == before


# --- reading the arguments ---------------------------------------------------------


def _bad(arguments: Any) -> str:
    parsed = so._parse_arguments(arguments)
    assert isinstance(parsed, so._BadArguments), parsed
    return parsed.reason


def _good(arguments: Any) -> Any:
    parsed = so._parse_arguments(arguments)
    assert isinstance(parsed, so._ParsedArguments), parsed
    return parsed.value


def test_a_json_text_and_the_object_it_holds_are_both_accepted():
    assert _good('{"a":"x"}') == {"a": "x"}
    assert _good({"a": "x"}) == {"a": "x"}


@pytest.mark.parametrize("text", ["{", "", "{'a': 1}", "[1,", "1" * 4301])
def test_text_that_is_not_json_is_invalid_json(text):
    assert _bad(text) == "invalid_json"


def test_nesting_too_deep_is_invalid_json_on_both_paths():
    text = "[" * 100_000 + "]" * 100_000
    assert _bad(text) == "invalid_json"
    deep: Any = []
    for _ in range(100_000):
        deep = [deep]
    assert _bad({"a": deep}) == "invalid_json"


NON_FINITE = ["NaN", "Infinity", "-Infinity", "1e400"]


@pytest.mark.parametrize("token", NON_FINITE)
def test_a_non_finite_number_is_named_on_both_paths(token):
    text = '{"a":' + token + "}"
    assert _bad(text) == "non_finite_number"
    assert _bad(json.loads(text)) == "non_finite_number"


def test_a_float_nan_in_a_dict_is_non_finite():
    assert _bad({"a": float("nan")}) == "non_finite_number"
    assert _bad({"a": [float("-inf")]}) == "non_finite_number"


@pytest.mark.parametrize("escape", ["\\u0000", "\\ud800"])
def test_nul_and_a_lone_surrogate_are_invalid_characters_on_both_paths(escape):
    text = '{"a":"x' + escape + '"}'
    assert _bad(text) == "invalid_character"
    assert _bad(json.loads(text)) == "invalid_character"
    key_text = '{"x' + escape + '":1}'
    assert _bad(key_text) == "invalid_character"
    assert _bad(json.loads(key_text)) == "invalid_character"


def test_an_invalid_character_is_reported_before_a_non_finite_number():
    text = '{"a":NaN,"b":"\\u0000"}'
    assert _bad(text) == "invalid_character"
    assert _bad(json.loads(text)) == "invalid_character"


def test_a_non_finite_number_is_reported_before_a_type_that_json_has_not():
    assert _bad({"a": float("nan"), "b": {1, 2}}) == "non_finite_number"


def test_one_thousand_han_characters_are_measured_the_same_on_both_paths():
    text = json.dumps({"a": "汉" * 1000}, ensure_ascii=False)
    assert _good(text) == _good(json.loads(text))


def _padded(total_bytes: int) -> dict[str, str]:
    base = ss.canonical_json_bytes({"a": ""})
    return {"a": "x" * (total_bytes - base)}


def test_exactly_the_argument_limit_passes_and_one_byte_more_is_too_long():
    at_limit = _padded(so.MAX_ARGUMENTS_BYTES)
    assert ss.canonical_json_bytes(at_limit) == 262_144
    assert _good(json.dumps(at_limit)) == at_limit
    assert _good(at_limit) == at_limit
    over = _padded(so.MAX_ARGUMENTS_BYTES + 1)
    for arguments in (json.dumps(over), over):
        parsed = so._parse_arguments(arguments)
        assert isinstance(parsed, so._BadArguments)
        assert (parsed.reason, parsed.arguments_bytes) == ("too_long", 262_145)


def test_the_underflow_of_a_tiny_number_is_accepted_as_zero_on_both_paths():
    assert _good('{"a":1e-400}') == {"a": 0.0}
    assert _good(json.loads('{"a":1e-400}')) == {"a": 0.0}


def test_a_tuple_key_and_a_set_value_are_invalid_json():
    assert _bad({("a", "b"): 1}) == "invalid_json"
    assert _bad({"a": {1, 2}}) == "invalid_json"


def test_an_int_key_is_left_for_the_shape_check():
    assert _good({1: "x"}) == {1: "x"}


def test_a_tuple_hides_nothing_from_the_walk():
    assert _bad({"a": ("a\u0000",)}) == "invalid_json"


def test_bytes_and_an_ordered_dict_are_not_json_types():
    assert _bad({"a": b"x"}) == "invalid_json"
    assert _bad({"a": OrderedDict(b=1)}) == "invalid_json"


def test_a_str_subclass_is_not_a_json_type():
    class Text(str):
        pass

    assert _bad({"a": Text("x")}) == "invalid_json"


@pytest.mark.parametrize("arguments", [None, 5, 1.5, ["a"], True])
def test_arguments_that_are_not_text_or_an_object_skip_the_text_step(arguments):
    assert _good(arguments) == arguments


def test_the_walk_does_not_recurse():
    deep: Any = "x"
    for _ in range(100_000):
        deep = {"a": deep}
    assert _bad(deep) == "invalid_json"
