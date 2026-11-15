"""Checking, normalizing and storing the ``response_format`` of a task."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import logging
import textwrap
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from tests.shared.structured_path_reference import reference_render
from xagent.core.agent.structured_output import StructuredOutputFormat
from xagent.core.agent.structured_schema import json_kind
from xagent.web.services import response_format
from xagent.web.services.response_format import (
    CLEAR,
    RESPONSE_FORMAT_CONFIG_KEY,
    RULE_TEXTS,
    ResponseFormat,
    ResponseFormatError,
    echo_for_change,
    echo_for_config,
    normalize_response_format,
    store_response_format,
    violation_message,
)
from xagent.web.services.task_runtime import (
    CLIENT_RESERVED_AGENT_CONFIG_KEYS,
    sanitize_client_agent_config,
)

CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "structured_output"
    / "structured_schema_corpus.json"
)
CORPUS = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

# Changing the corpus means changing this digest in the same commit.
CORPUS_SHA256 = "a5ba545a1031f39ec50a1232350c7ee6c2be9ac383acd1e81b376ae4cf49139c"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": False,
}

EXPECTED_TEXTS = {
    "N1": "response_format must be an object or null.",
    "N2": 'type must be "text" or "json_schema".',
    "N3": (
        "A response_format may only have the key type, "
        'plus json_schema and accept_partial when type is "json_schema".'
    ),
    "N4": (
        "json_schema must be an object with a schema key and no keys other "
        "than name, schema, strict and description."
    ),
    "N5": ('name must be 1 to 64 characters, each one of A-Z, a-z, 0-9, "_" or "-".'),
    "N6": (
        "description must be a string of at most 1024 characters "
        "with no NUL or surrogate characters."
    ),
    "N7": (
        "strict, when present, must be true; "
        "non-strict structured output is not supported."
    ),
    "N8": "Structured output is not supported for workforce keys.",
    "N9": "accept_partial, when present, must be true or false.",
    "S1": (
        "The schema must be JSON of at most 16384 bytes, with no NaN or "
        "Infinity, no NUL or surrogate characters, and no excessively "
        "deep nesting."
    ),
    "S2": (
        "The schema has more than 2000 schema nodes, "
        "or a schema node nested deeper than 32 levels."
    ),
    "S3": (
        "This keyword or keyword value is not supported. Put constraints "
        "such as format, length, range or pattern in description instead."
    ),
    "S4": (
        "Each schema node needs one supported shape: type (optionally with "
        "enum or const), anyOf, or $ref; enum and const values must match "
        "type, and a nullable type with enum must list null."
    ),
    "S5": (
        'An object node needs properties, "additionalProperties": false and '
        "a required list that names every property exactly once; "
        "an array node needs items."
    ),
    "S6": 'The root of the schema must have "type": "object".',
    "S7": (
        "$defs is allowed only at the root; its keys must be 1 to 64 "
        'characters, each one of A-Z, a-z, 0-9, "_" or "-"; '
        "every entry must be referenced from the root."
    ),
    "S8": (
        "$ref must be the only key of its node and must name an existing "
        'entry as "#/$defs/<name>".'
    ),
    "S9": "Recursive references are not supported.",
    "S10": (
        "anyOf is supported only as a nullable value: exactly one "
        '{"type": "null"} branch and one branch with a single non-null '
        "type. Use several nullable fields instead of a union."
    ),
    "S11": (
        "The schema is too large after $ref references are expanded: "
        "more than 2000 nodes, depth above 32, or more than 65536 bytes "
        "as sent to the model."
    ),
    "S12a": (
        "Each enum needs 1 to 500 distinct values; each value must be a "
        "string of at most 256 characters, a finite number, a boolean or "
        'null; strings must not be blank or contain "__NOT_EXPRESSIBLE__".'
    ),
    "S12b": (
        "The schema has more than 5000 enum values in total "
        "after $ref references are expanded."
    ),
    "S13": (
        "Property names must be 1 to 64 characters, each one of A-Z, a-z, "
        '0-9, "_", "." or "-", and must not be "definitions", "-" or '
        "only digits."
    ),
}


def _format(**json_schema: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"name": "result", "schema": copy.deepcopy(SCHEMA)}
    body.update(json_schema)
    return {"type": "json_schema", "json_schema": body}


def _rejection(raw: Any) -> ResponseFormatError:
    with pytest.raises(ResponseFormatError) as caught:
        normalize_response_format(raw)
    return caught.value


def _expected_message(rule: str, path: str) -> str:
    where = path or ("response_format" if rule.startswith("N") else "the schema root")
    return f"{rule} at {where}: {EXPECTED_TEXTS[rule]}"


# --- the message --------------------------------------------------------------


def test_the_rule_texts_are_the_twenty_three_published_sentences() -> None:
    assert RULE_TEXTS == EXPECTED_TEXTS
    assert len(RULE_TEXTS) == 23


def test_a_message_names_the_rule_the_place_and_the_fixed_text() -> None:
    assert violation_message("S3", ["properties", "a", "minLength"]) == (
        f"S3 at /properties/a/minLength: {EXPECTED_TEXTS['S3']}"
    )
    assert (
        violation_message("N1", []) == f"N1 at response_format: {EXPECTED_TEXTS['N1']}"
    )
    assert (
        violation_message("S6", []) == f"S6 at the schema root: {EXPECTED_TEXTS['S6']}"
    )
    assert (
        violation_message("N8", []) == f"N8 at response_format: {EXPECTED_TEXTS['N8']}"
    )


# --- the corpus through normalize ---------------------------------------------


@pytest.mark.parametrize("entry", CORPUS["reject"], ids=lambda e: e["id"])
def test_every_corpus_rejection_has_the_exact_message(entry: dict[str, Any]) -> None:
    raw = copy.deepcopy(entry["response_format"])

    error = _rejection(raw)

    assert error.rule == entry["rule"]
    assert str(error) == _expected_message(entry["rule"], entry["path"])
    assert raw == entry["response_format"]


@pytest.mark.parametrize("entry", CORPUS["accept"], ids=lambda e: e["id"])
def test_every_corpus_acceptance_keeps_the_schema_as_given(
    entry: dict[str, Any],
) -> None:
    raw = copy.deepcopy(entry["response_format"])

    result = normalize_response_format(raw)

    assert raw == entry["response_format"]
    if raw["type"] == "text":
        assert result is CLEAR
        return
    assert isinstance(result, ResponseFormat)
    body = entry["response_format"]["json_schema"]
    assert result.name == body["name"]
    assert result.schema == body["schema"]
    assert result.description == body.get("description")
    assert result.accept_partial is (
        entry["response_format"].get("accept_partial") is True
    )


def test_the_corpus_digest_is_pinned() -> None:
    assert hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest() == CORPUS_SHA256


# --- the outer shape -----------------------------------------------------------


def test_absent_and_null_change_nothing_and_text_clears() -> None:
    assert normalize_response_format(None) is None
    assert normalize_response_format({"type": "text"}) is CLEAR


def test_strict_true_is_accepted() -> None:
    result = normalize_response_format(_format(strict=True))

    assert isinstance(result, ResponseFormat)


def _json_kind_names() -> set[str]:
    """Every string ``json_kind`` can return, read from its return statements."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(json_kind)))
    return {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


# Every kind of value the switch can be given, plus the values the contract
# names as easy to confuse with a boolean.
_SWITCH_NOT_BOOLEAN: list[Any] = [None, 0, 1, 1.0, "true", "", [], {}]


def _with_switch(value: Any) -> dict[str, Any]:
    raw = _format()
    raw["accept_partial"] = value
    return raw


def test_the_switch_samples_cover_every_json_kind() -> None:
    kinds = {json_kind(sample) for sample in [True, *_SWITCH_NOT_BOOLEAN]}
    corpus_kinds = {
        json_kind(entry["response_format"]["accept_partial"])
        for entry in CORPUS["reject"]
        if entry["rule"] == "N9"
    }

    assert kinds == _json_kind_names() - {"other"}
    assert corpus_kinds == kinds - {"boolean"}


def test_n9_runs_after_n3_and_before_n4() -> None:
    bad_outer = _with_switch("true")
    bad_outer["zzz"] = 1
    bad_body = _with_switch("true")
    bad_body["json_schema"] = "x"

    assert str(_rejection(bad_outer)) == _expected_message("N3", "/zzz")
    assert str(_rejection(bad_body)) == _expected_message("N9", "/accept_partial")


@pytest.mark.parametrize(
    ("raw", "rule", "path"),
    [
        (_format(name="a b", strict=1), "N5", "/json_schema/name"),
        (_format(description="x" * 1025, strict=1), "N6", "/json_schema/description"),
        (_format(strict=False, schema={"type": "string"}), "N7", "/json_schema/strict"),
        (_format(foo=1, name="a b"), "N4", "/json_schema/foo"),
        ({"type": "x", "foo": 1}, "N2", "/type"),
        (
            {"type": "json_schema", "json_schema": {"name": "a b"}},
            "N4",
            "/json_schema/schema",
        ),
    ],
    ids=[
        "n5-before-n7",
        "n6-before-n7",
        "n7-before-s6",
        "n4-before-n5",
        "n2-before-n3",
        "n4-before-n5-missing",
    ],
)
def test_the_outer_rules_run_in_their_numbered_order(
    raw: Any, rule: str, path: str
) -> None:
    assert str(_rejection(raw)) == _expected_message(rule, path)


def test_description_is_measured_before_it_is_searched_for_bad_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searched: list[str] = []
    monkeypatch.setattr(
        response_format,
        "has_forbidden_char",
        lambda text: searched.append(text) or False,
    )

    error = _rejection(_format(description="\x00" + "a" * 1999))

    assert str(error) == _expected_message("N6", "/json_schema/description")
    assert searched == []


def test_description_with_a_bad_character_is_searched_when_short() -> None:
    error = _rejection(_format(description="a\ud800b"))

    assert str(error) == _expected_message("N6", "/json_schema/description")


def test_a_schema_that_is_a_string_is_the_root_rule() -> None:
    error = _rejection(_format(schema="x"))

    assert str(error) == _expected_message("S6", "")


def test_nesting_too_deep_to_serialize_is_reported_as_the_size_rule() -> None:
    deep: list[Any] = []
    for _ in range(10000):
        deep = [deep]
    schema = copy.deepcopy(SCHEMA)
    schema["x-deep"] = deep

    error = _rejection(_format(schema=schema))

    assert str(error) == _expected_message("S1", "")


def test_an_extra_key_of_five_million_characters_is_cut_in_bounded_memory() -> None:
    key = "\u0080" * 5_000_000
    raw = {"type": "text", key: 1}

    tracemalloc.start()
    try:
        error = _rejection(raw)
        message = str(error)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert error.rule == "N3"
    assert message.isascii()
    assert message == _expected_message("N3", reference_render([key]))
    assert peak <= 1024 * 1024


def test_the_schema_is_not_rewritten() -> None:
    raw = _format(description="a note")
    before = copy.deepcopy(raw)

    result = normalize_response_format(raw)

    assert isinstance(result, ResponseFormat)
    assert result.schema == before["json_schema"]["schema"]
    assert result.schema is raw["json_schema"]["schema"]
    assert raw == before


# --- storing -------------------------------------------------------------------


def test_storing_a_format_writes_the_four_keys_and_leaves_the_input_alone() -> None:
    config = {"prompt_context": {"data": {"a": "b"}}}
    change = ResponseFormat(name="result", schema=SCHEMA, description=None)

    stored = store_response_format(config, change)

    assert stored is not config
    assert config == {"prompt_context": {"data": {"a": "b"}}}
    assert stored == {
        "prompt_context": {"data": {"a": "b"}},
        RESPONSE_FORMAT_CONFIG_KEY: {
            "type": "json_schema",
            "name": "result",
            "schema": SCHEMA,
            "description": None,
        },
    }
    assert list(stored[RESPONSE_FORMAT_CONFIG_KEY]) == [
        "type",
        "name",
        "schema",
        "description",
    ]


def test_storing_keeps_the_description_when_there_is_one() -> None:
    change = normalize_response_format(_format(description="the answer"))
    assert isinstance(change, ResponseFormat)

    stored = store_response_format({}, change)

    assert stored[RESPONSE_FORMAT_CONFIG_KEY]["description"] == "the answer"


def test_storing_none_changes_nothing_and_clear_removes_the_key() -> None:
    config = {RESPONSE_FORMAT_CONFIG_KEY: {"type": "json_schema"}, "keep": 1}

    unchanged = store_response_format(config, None)
    cleared = store_response_format(config, CLEAR)

    assert unchanged == config
    assert unchanged is not config
    assert cleared == {"keep": 1}
    assert config == {RESPONSE_FORMAT_CONFIG_KEY: {"type": "json_schema"}, "keep": 1}
    assert store_response_format({}, CLEAR) == {}


# The stored value of a format without the switch, as written before the switch
# existed.
_STORED_WITHOUT_SWITCH = (
    '{"type": "json_schema", "name": "result", "schema": '
    '{"type": "object", "properties": {"a": {"type": "string"}}, '
    '"required": ["a"], "additionalProperties": false}, "description": %s}'
)


@pytest.mark.parametrize("switch", [None, False], ids=["absent", "false"])
@pytest.mark.parametrize("description", [None, "the answer"], ids=["bare", "described"])
def test_store_without_switch_is_byte_identical(
    switch: bool | None, description: str | None
) -> None:
    extra: dict[str, Any] = {} if switch is None else {"accept_partial": switch}
    raw = {**_format(), **extra}
    if description is not None:
        raw["json_schema"]["description"] = description
    change = normalize_response_format(raw)

    stored = store_response_format({}, change)[RESPONSE_FORMAT_CONFIG_KEY]

    expected = _STORED_WITHOUT_SWITCH % (
        "null" if description is None else json.dumps(description)
    )
    assert json.dumps(stored) == expected


def test_storing_the_switch_adds_a_fifth_key_after_the_four() -> None:
    change = normalize_response_format(_with_switch(True))

    stored = store_response_format({}, change)[RESPONSE_FORMAT_CONFIG_KEY]

    assert list(stored) == ["type", "name", "schema", "description", "accept_partial"]
    assert stored["accept_partial"] is True


def test_storing_replaces_an_earlier_format() -> None:
    first = ResponseFormat(name="one", schema=SCHEMA, description=None)
    second = ResponseFormat(name="two", schema=SCHEMA, description="d")

    stored = store_response_format(store_response_format({}, first), second)

    assert stored[RESPONSE_FORMAT_CONFIG_KEY]["name"] == "two"
    assert stored[RESPONSE_FORMAT_CONFIG_KEY]["description"] == "d"


# --- the echo ------------------------------------------------------------------


def test_the_echo_of_a_change() -> None:
    change = ResponseFormat(name="result", schema=SCHEMA, description=None)

    assert echo_for_change(change) == ("json_schema", "result")
    assert echo_for_change(None) == ("text", "")
    assert echo_for_change(CLEAR) == ("text", "")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, ("text", "")),
        ("not a dict", ("text", "")),
        ({}, ("text", "")),
        ({RESPONSE_FORMAT_CONFIG_KEY: None}, ("text", "")),
        ({RESPONSE_FORMAT_CONFIG_KEY: "x"}, ("text", "")),
        ({RESPONSE_FORMAT_CONFIG_KEY: {"type": "text"}}, ("text", "")),
        ({RESPONSE_FORMAT_CONFIG_KEY: {"type": "json_schema"}}, ("text", "")),
        (
            {RESPONSE_FORMAT_CONFIG_KEY: {"type": "json_schema", "name": 1}},
            ("text", ""),
        ),
        (
            {RESPONSE_FORMAT_CONFIG_KEY: {"type": "json_schema", "name": "answer"}},
            ("json_schema", "answer"),
        ),
    ],
)
def test_the_echo_of_a_stored_config(config: Any, expected: tuple[str, str]) -> None:
    assert echo_for_config(config) == expected


def test_what_is_stored_is_echoed_back() -> None:
    change = normalize_response_format(_format(name="answer"))

    stored = store_response_format({}, change)

    assert (
        echo_for_config(stored) == echo_for_change(change) == ("json_schema", "answer")
    )


# --- only this module writes the key ---------------------------------------------


def _mentions_key(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and node.value == "response_format"
        or isinstance(node, ast.Name)
        and node.id == "RESPONSE_FORMAT_CONFIG_KEY"
    )


def _is_agent_config(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "agent_config" in node.id
    if isinstance(node, ast.Attribute):
        return "agent_config" in node.attr
    return False


def _keyed_writes(source: str) -> list[int]:
    """Lines that assign, delete, pop or update the format key of something
    whose name says it is an ``agent_config``."""
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and _is_agent_config(target.value)
                and _mentions_key(target.slice)
            ):
                found.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"pop", "update", "setdefault"}
            and _is_agent_config(node.func.value)
        ):
            keyed = any(_mentions_key(arg) for arg in node.args)
            keyed = keyed or any(
                isinstance(arg, ast.Dict)
                and any(k and _mentions_key(k) for k in arg.keys)
                for arg in node.args
            )
            keyed = keyed or any(kw.arg == "response_format" for kw in node.keywords)
            if keyed:
                found.append(node.lineno)
    return found


@pytest.mark.parametrize(
    "line",
    [
        'agent_config["response_format"] = 1',
        "task.agent_config[RESPONSE_FORMAT_CONFIG_KEY] = 1",
        'del agent_config["response_format"]',
        'agent_config.pop("response_format", None)',
        'agent_config.update({"response_format": 1})',
        "agent_config.update(response_format=1)",
        "agent_config.setdefault(RESPONSE_FORMAT_CONFIG_KEY, {})",
    ],
)
def test_the_scan_sees_each_way_of_writing_the_key(line: str) -> None:
    assert _keyed_writes(line) == [1]


@pytest.mark.parametrize(
    "line",
    [
        'agent_config["other"] = 1',
        'value = agent_config["response_format"]',
        'agent_config.get("response_format")',
        'other["response_format"] = 1',
    ],
)
def test_the_scan_ignores_reads_and_other_keys(line: str) -> None:
    assert _keyed_writes(line) == []


def test_only_the_format_module_writes_the_key_in_the_web_package() -> None:
    web_root = Path(response_format.__file__).resolve().parents[1]
    owner = Path(response_format.__file__).resolve()

    outside: dict[str, list[int]] = {}
    inside: list[int] = []
    for path in sorted(web_root.rglob("*.py")):
        lines = _keyed_writes(path.read_text(encoding="utf-8"))
        if path.resolve() == owner:
            inside = lines
        elif lines:
            outside[str(path.relative_to(web_root))] = lines

    assert outside == {}
    assert inside, "the scan no longer sees the one function that writes the key"


def test_a_client_cannot_send_the_key_through_the_sanitizer() -> None:
    assert RESPONSE_FORMAT_CONFIG_KEY in CLIENT_RESERVED_AGENT_CONFIG_KEYS

    sanitized = sanitize_client_agent_config(
        {RESPONSE_FORMAT_CONFIG_KEY: {"type": "json_schema"}, "keep": 1}
    )

    assert sanitized == {"keep": 1}


# --- the format stored on a task ---------------------------------------------

INVALID = StructuredOutputFormat.INVALID
STORED_WARNING = "stored response_format is not valid"


def parse_stored_response_format(
    value: Any, *, task_id: object = None
) -> StructuredOutputFormat:
    return response_format.parse_stored_response_format(value, task_id=task_id)


def _stored(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "json_schema",
        "name": "result",
        "schema": copy.deepcopy(SCHEMA),
        "description": "A result.",
    }
    value.update(changes)
    return value


def _stored_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if STORED_WARNING in r.getMessage()]


def _parse_invalid(
    value: Any, caplog: pytest.LogCaptureFixture, task_id: object = 7
) -> str:
    """Parse ``value``, assert it is refused with one WARNING, return its text."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=response_format.__name__):
        result = parse_stored_response_format(value, task_id=task_id)

    assert result is INVALID
    records = _stored_warnings(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    return records[0].getMessage()


def test_a_format_written_by_store_is_read_back_as_it_was_stored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    change = normalize_response_format(_format(description="A result."))
    stored = store_response_format({}, change)[RESPONSE_FORMAT_CONFIG_KEY]

    with caplog.at_level(logging.WARNING, logger=response_format.__name__):
        fmt = parse_stored_response_format(stored, task_id=1)

    assert fmt is not INVALID
    assert isinstance(fmt, StructuredOutputFormat)
    assert (fmt.name, fmt.description, fmt.schema) == (
        stored["name"],
        stored["description"],
        stored["schema"],
    )
    assert fmt.schema is stored["schema"]
    assert _stored_warnings(caplog) == []


def test_a_stored_format_without_a_description_is_accepted() -> None:
    fmt = parse_stored_response_format(_stored(description=None))

    assert fmt is not INVALID
    assert fmt.description is None


@pytest.mark.parametrize("value", ["x", [], 1], ids=["str", "list", "int"])
def test_a_stored_value_that_is_not_an_object_is_refused(
    value: Any, caplog: pytest.LogCaptureFixture
) -> None:
    message = _parse_invalid(value, caplog)

    assert message == (
        "task_id=7 stored response_format is not valid: "
        "rule=stored_shape at response_format"
    )


def test_an_extra_key_is_refused_and_named(caplog: pytest.LogCaptureFixture) -> None:
    message = _parse_invalid(_stored(strict=True), caplog)

    assert message.endswith("rule=stored_shape at /strict")


def test_a_missing_key_is_refused_and_named(caplog: pytest.LogCaptureFixture) -> None:
    value = _stored()
    del value["description"]

    assert _parse_invalid(value, caplog).endswith("rule=stored_shape at /description")


def test_the_first_missing_key_in_the_fixed_order_is_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    value = _stored()
    del value["description"]
    del value["name"]

    assert _parse_invalid(value, caplog).endswith("rule=stored_shape at /name")


@pytest.mark.parametrize("bad_type", ["text", 1, True], ids=["text", "int", "bool"])
def test_a_stored_type_other_than_json_schema_is_refused(
    bad_type: Any, caplog: pytest.LogCaptureFixture
) -> None:
    message = _parse_invalid(_stored(type=bad_type), caplog)

    assert message.endswith("rule=stored_shape at /type")


@pytest.mark.parametrize(
    "name", [1, "a\n", "a" * 65], ids=["int", "newline", "65-chars"]
)
def test_a_stored_name_is_checked_by_rule_n5(
    name: Any, caplog: pytest.LogCaptureFixture
) -> None:
    assert _parse_invalid(_stored(name=name), caplog).endswith("rule=N5 at /name")


@pytest.mark.parametrize(
    "description",
    [1, "d" * 1025, "a\u0000b"],
    ids=["int", "1025-chars", "nul"],
)
def test_a_stored_description_is_checked_by_rule_n6(
    description: Any, caplog: pytest.LogCaptureFixture
) -> None:
    message = _parse_invalid(_stored(description=description), caplog)

    assert message.endswith("rule=N6 at /description")


def test_a_stored_schema_that_is_not_an_object_breaks_rule_s6(
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = _parse_invalid(_stored(schema="x"), caplog)

    assert message.endswith("rule=S6 at the schema root")


def test_a_stored_schema_violation_names_the_path_of_the_violation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    schema = copy.deepcopy(SCHEMA)
    schema["properties"]["a"]["minLength"] = 1

    message = _parse_invalid(_stored(schema=schema), caplog)

    assert message.endswith("rule=S3 at /properties/a/minLength")


# A stored switch is read as a boolean (True or False) or refused: the value
# is the boolean read, or the tail of the WARNING that names the fault.
_STORED_SWITCH_CASES: list[tuple[dict[str, Any], bool | str]] = [
    ({}, False),
    ({"accept_partial": True}, True),
    ({"accept_partial": False}, False),
    *[
        ({"accept_partial": value}, "stored_shape at /accept_partial")
        for value in _SWITCH_NOT_BOOLEAN
    ],
    ({"accept_partial": True, "strict": True}, "stored_shape at /strict"),
]


@pytest.mark.parametrize(
    ("extra", "expected"),
    _STORED_SWITCH_CASES,
    ids=lambda v: repr(v) if isinstance(v, dict) else None,
)
def test_parse_stored_accept_partial(
    extra: dict[str, Any],
    expected: bool | str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if isinstance(expected, str):
        message = _parse_invalid(_stored(**extra), caplog)
        assert message.endswith(f"rule={expected}")
        return
    fmt = parse_stored_response_format(_stored(**extra))

    assert fmt is not INVALID
    assert fmt.accept_partial is expected


def test_the_switch_is_checked_after_the_type_and_before_the_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _parse_invalid(_stored(type="text", accept_partial=1, name=1), caplog)
    second = _parse_invalid(_stored(accept_partial=1, name=1), caplog)

    assert first.endswith("rule=stored_shape at /type")
    assert second.endswith("rule=stored_shape at /accept_partial")


def test_the_invalid_format_never_has_the_switch() -> None:
    assert INVALID.accept_partial is False


def test_the_checks_stop_at_the_first_step_that_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    value = _stored(type="text", name=1, description=1, schema="x")
    value["strict"] = True

    message = _parse_invalid(value, caplog)

    assert message.endswith("rule=stored_shape at /strict")


@pytest.mark.parametrize(
    "entry",
    [e for e in CORPUS["reject"] if e["rule"].startswith("S")],
    ids=lambda e: e["id"],
)
def test_every_corpus_schema_rejection_is_refused_when_stored(
    entry: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    schema = copy.deepcopy(entry["response_format"]["json_schema"]["schema"])

    message = _parse_invalid(_stored(schema=schema), caplog)

    assert f"rule={entry['rule']} at " in message


@pytest.mark.parametrize(
    "entry",
    [e for e in CORPUS["accept"] if e["response_format"]["type"] == "json_schema"],
    ids=lambda e: e["id"],
)
def test_every_corpus_acceptance_is_accepted_once_stored(
    entry: dict[str, Any],
) -> None:
    change = normalize_response_format(copy.deepcopy(entry["response_format"]))
    stored = store_response_format({}, change)[RESPONSE_FORMAT_CONFIG_KEY]

    fmt = parse_stored_response_format(stored)

    assert fmt is not INVALID
    assert fmt.schema is stored["schema"]


def test_the_warning_carries_neither_the_name_nor_the_description(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=response_format.__name__):
        for value in (
            _stored(name="SECRET\nX"),
            _stored(description="SECRET" * 300),
        ):
            assert parse_stored_response_format(value, task_id=3) is INVALID

    assert len(_stored_warnings(caplog)) == 2
    assert "SECRET" not in caplog.text


def test_reading_a_stored_format_does_not_go_through_normalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(raw: Any) -> None:
        raise AssertionError("normalize_response_format must not be called")

    monkeypatch.setattr(response_format, "normalize_response_format", boom)

    assert parse_stored_response_format(_stored()) is not INVALID


def test_reading_a_stored_format_does_not_modify_it() -> None:
    value = _stored()
    before = copy.deepcopy(value)

    parse_stored_response_format(value)
    parse_stored_response_format(_stored(strict=True))

    assert value == before
