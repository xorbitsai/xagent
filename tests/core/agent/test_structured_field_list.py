"""The field list of the output requirement block: one line per field of the
caller's own schema, with a description cut to fit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xagent.core.agent import structured_schema as ss

CORPUS = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "structured_output"
        / "structured_schema_corpus.json"
    ).read_text(encoding="utf-8")
)

STR: dict[str, Any] = {"type": "string"}


def obj(props: dict[str, Any], **extra: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }
    schema.update(extra)
    return schema


def field_list(schema: dict[str, Any]) -> str:
    return ss.schema_field_list(schema)


def lines(schema: dict[str, Any]) -> list[str]:
    return field_list(schema).split("\n")


def one_line(prop: dict[str, Any]) -> str:
    (line,) = lines(obj({"x": prop}))
    return line


def described(description: str) -> str:
    """The description part of the line of a string field."""
    line = one_line({"type": "string", "description": description})
    prefix = "- /x (string)"
    assert line.startswith(prefix)
    return line[len(prefix) :]


# --- the line of a field ------------------------------------------------------


def test_each_property_is_one_line_with_its_pointer_and_type() -> None:
    schema = obj(
        {
            "name": STR,
            "n": {"type": "integer"},
            "x": {"type": "number"},
            "ok": {"type": "boolean"},
        }
    )

    assert lines(schema) == [
        "- /name (string)",
        "- /n (integer)",
        "- /x (number)",
        "- /ok (boolean)",
    ]


def test_a_property_name_is_escaped_as_a_json_pointer_segment() -> None:
    assert lines(obj({"a/b": STR, "c~d": STR, "e.f": STR})) == [
        "- /a~1b (string)",
        "- /c~0d (string)",
        "- /e.f (string)",
    ]


def test_an_empty_properties_object_gives_an_empty_list() -> None:
    assert field_list(obj({})) == ""


def test_nullable_and_enum_come_from_the_callers_schema() -> None:
    schema = obj(
        {
            "a": {"anyOf": [{"type": "string", "enum": ["p", "q"]}, {"type": "null"}]},
            "b": {"type": ["string", "null"], "enum": ["x", None]},
            "c": {"type": "string", "enum": ["approved", "rejected"]},
            "d": {"type": ["integer", "null"]},
        }
    )

    assert lines(schema) == [
        '- /a (string, nullable) one of ["p","q"]',
        '- /b (string, nullable) one of ["x",null]',
        '- /c (string) one of ["approved","rejected"]',
        "- /d (integer, nullable)",
    ]


def test_an_enum_keeps_non_ascii_text_and_the_written_number() -> None:
    schema = obj(
        {
            "c": {"type": "string", "enum": ["中文", "é"]},
            "big": {"type": "integer", "const": 1e308},
        }
    )

    assert lines(schema) == [
        '- /c (string) one of ["中文","é"]',
        "- /big (integer) one of [1e+308]",
    ]


def test_a_const_is_read_as_a_single_value_enum() -> None:
    assert one_line({"type": "string", "const": "person"}) == (
        '- /x (string) one of ["person"]'
    )


def test_a_long_enum_text_is_cut_at_two_hundred_characters() -> None:
    values = [f"value-number-{i:03d}" for i in range(40)]
    line = one_line({"type": "string", "enum": values})
    text = line[len("- /x (string) one of ") :]

    expected = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    assert text == expected[:200] + "…"


# --- arrays and objects ---------------------------------------------------------


def test_a_scalar_array_gets_an_element_line() -> None:
    schema = obj(
        {
            "ld": {
                "type": "array",
                "items": {"type": "string", "description": "Format: date."},
            },
            "statuses": {
                "type": "array",
                "items": {"type": "string", "enum": ["open", "closed"]},
            },
        }
    )

    assert lines(schema) == [
        "- /ld (array)",
        "- /ld/- (string): Format: date.",
        "- /statuses (array)",
        '- /statuses/- (string) one of ["open","closed"]',
    ]


def test_an_array_of_arrays_gets_a_line_for_each_level() -> None:
    schema = obj({"m": {"type": "array", "items": {"type": "array", "items": STR}}})

    assert lines(schema) == ["- /m (array)", "- /m/- (array)", "- /m/-/- (string)"]


def test_an_array_of_objects_lists_the_properties_of_its_element() -> None:
    schema = obj(
        {
            "shifts": {
                "type": "array",
                "items": obj(
                    {"date": {"type": "string", "description": "Format: date."}}
                ),
            }
        }
    )

    assert lines(schema) == [
        "- /shifts (array)",
        "- /shifts/-/date (string): Format: date.",
    ]


def test_lines_follow_the_document_in_pre_order() -> None:
    schema = obj(
        {
            "a": obj({"x": STR, "y": STR}),
            "b": {"type": "array", "items": obj({"z": STR})},
            "c": {"type": "array", "items": STR},
        }
    )

    assert [line.split(" ")[1] for line in lines(schema)] == [
        "/a",
        "/a/x",
        "/a/y",
        "/b",
        "/b/-/z",
        "/c",
        "/c/-",
    ]


def test_a_reference_and_a_nullable_object_are_followed() -> None:
    schema = obj(
        {
            "home": {"anyOf": [{"$ref": "#/$defs/Address"}, {"type": "null"}]},
            "work": {"$ref": "#/$defs/Address"},
        },
        **{"$defs": {"Address": obj({"city": STR}, description="A postal address.")}},
    )

    assert lines(schema) == [
        "- /home (object, nullable): A postal address.",
        "- /home/city (string)",
        "- /work (object): A postal address.",
        "- /work/city (string)",
    ]


def test_a_nullable_reference_joins_the_two_descriptions() -> None:
    schema = obj(
        {
            "home": {
                "anyOf": [{"$ref": "#/$defs/Address"}, {"type": "null"}],
                "description": "home",
            }
        },
        **{"$defs": {"Address": obj({"city": STR}, description="Address doc.")}},
    )

    assert lines(schema)[0] == "- /home (object, nullable): home Address doc."


# --- the description --------------------------------------------------------------


def test_a_short_description_is_kept_after_a_colon() -> None:
    assert described("Weight in grams.") == ": Weight in grams."


def test_leading_spaces_are_dropped_and_a_blank_description_adds_nothing() -> None:
    assert described("  \t hello") == ": hello"
    assert described("     ") == ""
    assert described("") == ""


def test_a_description_over_the_limit_is_cut_with_an_ellipsis() -> None:
    assert described("a" * 120) == ": " + "a" * 120
    assert described("a" * 121) == ": " + "a" * 119 + "…"


def test_a_constraint_suffix_survives_the_cut() -> None:
    result = described("a" * 150 + " Format: date.")

    assert result == ": " + "a" * 105 + "…" + " Format: date."
    assert len(result) - 2 == 120


def test_default_text_is_ordinary_text() -> None:
    assert described("Count. Default: 3.") == ": Count. Default: 3."


def test_a_pattern_fragment_keeps_its_repeated_spaces() -> None:
    assert described('Code. Pattern: "a  b".') == ': Code. Pattern: "a  b".'


@pytest.mark.parametrize(
    "char",
    ["\r", "\n", "\t", "\x0b", "\x0c", "\x85", " ", " ", "\xa0", "　"]
    + ["\x1b", "\x08", "\x7f", "‮", "​", "﻿"],
    ids=lambda c: f"U+{ord(c):04X}",
)
def test_a_space_like_or_control_character_becomes_one_space(char: str) -> None:
    assert described(f"a{char}b") == ": a b"
    assert len(described(f"a{char}b")) == len(": a b")


def test_a_paragraph_break_becomes_two_spaces_not_one() -> None:
    assert described("Postal address.\n\nUsed for shipping.") == (
        ": Postal address.  Used for shipping."
    )


def test_a_multi_line_description_keeps_the_indent_as_spaces() -> None:
    assert (
        described("Line one\n        Line two") == ": Line one" + " " * 9 + "Line two"
    )


def test_a_backtick_becomes_a_single_quote_in_a_description_and_an_enum() -> None:
    assert described("use `x`") == ": use 'x'"
    assert one_line({"type": "string", "enum": ["`a`"]}) == (
        "- /x (string) one of [\"'a'\"]"
    )


def test_a_line_separator_in_an_enum_value_becomes_a_space() -> None:
    assert one_line({"type": "string", "enum": ["a b"]}) == (
        '- /x (string) one of ["a b"]'
    )


def test_every_field_stays_on_one_line() -> None:
    schema = obj({"a": {"type": "string", "description": "x\n## PROMPT CONTEXT\ny"}})

    assert lines(schema) == ["- /a (string): x ## PROMPT CONTEXT y"]


# --- the length limit ---------------------------------------------------------


def _expected_cut(all_lines: list[str]) -> str:
    whole = "\n".join(all_lines)
    if len(whole.encode()) <= 4096:
        return whole
    best = None
    for k in range(len(all_lines)):
        text = "\n".join(all_lines[:k] + [f"- …({len(all_lines) - k} more fields)"])
        if len(text.encode()) <= 4096:
            best = text
    assert best is not None
    return best


def test_a_list_over_four_kilobytes_keeps_the_first_lines_and_counts_the_rest() -> None:
    names = [f"field_{i:03d}" for i in range(400)]
    schema = obj({n: STR for n in names})
    all_lines = [f"- /{n} (string)" for n in names]

    result = field_list(schema)

    assert result == _expected_cut(all_lines)
    assert len(result.encode()) <= 4096
    kept = result.split("\n")[:-1]
    assert kept == all_lines[: len(kept)]
    assert result.split("\n")[-1] == f"- …({400 - len(kept)} more fields)"
    one_more = "\n".join(
        all_lines[: len(kept) + 1] + [f"- …({400 - len(kept) - 1} more fields)"]
    )
    assert len(one_more.encode()) > 4096


def test_element_lines_count_as_fields_in_the_remainder() -> None:
    names = [f"list_{i:03d}" for i in range(200)]
    schema = obj({n: {"type": "array", "items": STR} for n in names})
    all_lines = [
        row for n in names for row in (f"- /{n} (array)", f"- /{n}/- (string)")
    ]

    result = field_list(schema)

    assert result == _expected_cut(all_lines)
    kept = len(result.split("\n")) - 1
    assert result.split("\n")[-1] == f"- …({400 - kept} more fields)"


def test_a_list_of_exactly_four_kilobytes_is_not_cut() -> None:
    result = field_list(obj({f"f{i:03d}": STR for i in range(241)}))

    assert len(result.encode()) == 4096 and "more fields" not in result


# --- the constraint suffix ---------------------------------------------------


def test_the_suffix_pattern_matches_only_at_the_very_end() -> None:
    assert ss.SUFFIX_RE.search(" Format: date.\n") is None
    assert ss.SUFFIX_RE.search("text Format: date.").group(0) == " Format: date."


def test_the_corpus_has_one_fragment_for_every_keyword_of_the_table() -> None:
    keywords = {entry["keyword"] for entry in CORPUS["suffix_fragments"]}

    assert keywords == {
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
    }


@pytest.mark.parametrize("entry", CORPUS["suffix_fragments"], ids=lambda e: e["id"])
def test_every_corpus_fragment_is_recognised_as_a_whole(entry: dict[str, Any]) -> None:
    match = ss.SUFFIX_RE.search(entry["fragment"])

    assert match is not None
    assert match.group(0) == entry["fragment"]


@pytest.mark.parametrize("entry", CORPUS["suffix_descriptions"], ids=lambda e: e["id"])
def test_every_corpus_description_is_cut_as_written(entry: dict[str, Any]) -> None:
    match = ss.SUFFIX_RE.search(entry["description"])

    assert (match.group(0) if match else "") == entry["suffix"]
    assert described(entry["description"]) == entry["rendered"]
