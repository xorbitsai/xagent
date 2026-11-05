"""Checking, normalizing and storing the ``response_format`` of a task."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from tests.shared.structured_path_reference import reference_render
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
CORPUS_SHA256 = "eacdd1438976f20acc740ee116511eda61bc74c8bfe39b77ea48e79e34bcdaec"

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
        'plus json_schema when type is "json_schema".'
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


def test_the_rule_texts_are_the_twenty_two_published_sentences() -> None:
    assert RULE_TEXTS == EXPECTED_TEXTS
    assert len(RULE_TEXTS) == 22


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


def test_the_corpus_digest_is_pinned() -> None:
    assert hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest() == CORPUS_SHA256


# --- the outer shape -----------------------------------------------------------


def test_absent_and_null_change_nothing_and_text_clears() -> None:
    assert normalize_response_format(None) is None
    assert normalize_response_format({"type": "text"}) is CLEAR


def test_strict_true_is_accepted() -> None:
    result = normalize_response_format(_format(strict=True))

    assert isinstance(result, ResponseFormat)


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
