"""The assertion evaluator: payloads, paths and each assertion kind."""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.web.connector_checks.assertions import judge_result, resolve_expected
from xagent.web.connector_checks.declarations import (
    EachHasKeys,
    Equals,
    HasKeys,
    IsType,
    L3Envelope,
    Length,
    PayloadSource,
    StatusSuccess,
)
from xagent.web.connector_checks.fixtures import FixtureMissing, FixtureRef
from xagent.web.connector_checks.results import FailureClass
from xagent.web.services.connector_probe import _single_json_object

PASS = (None, None)


def text_result(*texts: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text} for text in texts],
        "structured_content": None,
        "is_error": is_error,
    }


def json_result(*objects: Any) -> dict[str, Any]:
    return text_result(*(json.dumps(item) for item in objects))


class Resolver:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def resolve_fixture(self, key: str) -> Any:
        return self.values.get(key)


# The call itself


@pytest.mark.parametrize("payload", ["json_text", "json_text_items"])
def test_a_flagged_error_fails_the_call_before_any_assertion(
    payload: PayloadSource,
) -> None:
    raw = json_result({"status": "success"})
    raw["is_error"] = True

    assert judge_result(raw, payload, (StatusSuccess(),)) == (
        FailureClass.CALL_FAILED,
        "tool_error",
    )


@pytest.mark.parametrize("raw", [None, "text", ["content"]])
def test_a_result_that_is_not_a_mapping_fails_the_call(raw: Any) -> None:
    assert judge_result(raw, "json_text", (StatusSuccess(),)) == (
        FailureClass.CALL_FAILED,
        "invalid_result",
    )


# Payloads


def test_json_text_reads_the_one_text_item_as_an_object() -> None:
    raw = json_result({"status": "success", "messages": []})

    assert judge_result(raw, "json_text", (HasKeys((), ("messages",)),)) == PASS


@pytest.mark.parametrize(
    "raw",
    [
        json_result({"a": 1}, {"b": 2}),
        text_result("not json"),
        json_result([1, 2]),
        {"content": [], "is_error": False},
        {"content": [{"type": "image", "data": "..."}], "is_error": False},
        {"is_error": False},
    ],
    ids=["two_items", "not_json", "array", "empty", "no_text", "no_content"],
)
def test_json_text_needs_exactly_one_text_item_holding_an_object(raw: Any) -> None:
    assert judge_result(raw, "json_text", (StatusSuccess(),)) == (
        FailureClass.ASSERTION_FAILED,
        "not_a_json_object",
    )


def test_json_text_items_parses_every_text_item_and_skips_other_content() -> None:
    raw = json_result({"id": 1}, [2, 3], "four")
    raw["content"].insert(1, {"type": "image", "data": "..."})

    assertions = (
        Length((), min=3, max=3),
        IsType((1,), "array"),
        Equals((2,), "four"),
    )
    assert judge_result(raw, "json_text_items", assertions) == PASS


def test_json_text_items_fails_on_a_text_item_that_is_not_json() -> None:
    raw = text_result(json.dumps({"id": 1}), "plain words")

    assert judge_result(raw, "json_text_items", (Length((), min=1),)) == (
        FailureClass.ASSERTION_FAILED,
        "not_json_text",
    )


def test_structured_content_reads_the_structured_object() -> None:
    raw = text_result("summary")
    raw["structured_content"] = {"user_id": "u-1"}

    assertions = (HasKeys((), ("user_id",)),)
    assert judge_result(raw, "structured_content", assertions) == PASS


@pytest.mark.parametrize("structured", [None, ["user"], "user"])
def test_structured_content_must_be_an_object(structured: Any) -> None:
    raw = text_result(json.dumps({"user_id": "u-1"}))
    raw["structured_content"] = structured

    assert judge_result(raw, "structured_content", (HasKeys((), ("user_id",)),)) == (
        FailureClass.ASSERTION_FAILED,
        "no_structured_content",
    )


# Assertions

_DOCUMENT = {
    "status": "success",
    "count": 2,
    "ratio": 0.5,
    "enabled": True,
    "owner": None,
    "site": {"id": "s-1", "name": "Main"},
    "items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    "empty": [],
}


@pytest.mark.parametrize(
    "assertion",
    [
        StatusSuccess(),
        HasKeys((), ("site", "items")),
        HasKeys(("site",), ("id", "name")),
        EachHasKeys(("items",), ("id", "name")),
        EachHasKeys(("empty",), ("id",)),
        IsType(("site",), "object"),
        IsType(("items",), "array"),
        IsType(("site", "id"), "string"),
        IsType(("count",), "number"),
        IsType(("ratio",), "number"),
        IsType(("enabled",), "boolean"),
        IsType(("owner",), "null"),
        Length(("items",), min=1, max=2),
        Length(("empty",)),
        Equals(("items", 1, "name"), "b"),
        Equals(("count",), 2),
        Equals(("count",), 2.0),
        Equals(("enabled",), True),
        Equals(("owner",), None),
    ],
)
def test_an_assertion_that_holds_passes(assertion: Any) -> None:
    assert judge_result(json_result(_DOCUMENT), "json_text", (assertion,)) == PASS


@pytest.mark.parametrize(
    ("assertion", "reason"),
    [
        (HasKeys(("site",), ("id", "url")), "has_keys"),
        (HasKeys(("items",), ("id",)), "has_keys"),
        (HasKeys(("missing",), ("id",)), "has_keys"),
        (EachHasKeys(("items",), ("id", "url")), "each_has_keys"),
        (EachHasKeys(("site",), ("id",)), "each_has_keys"),
        (IsType(("count",), "string"), "is_type"),
        (IsType(("enabled",), "number"), "is_type"),
        (IsType(("count",), "boolean"), "is_type"),
        (IsType(("items", 5), "object"), "is_type"),
        (Length(("items",), min=3), "length"),
        (Length(("items",), max=1), "length"),
        (Length(("site",)), "length"),
        (Equals(("site", "id"), "s-2"), "equals"),
        (Equals(("enabled",), 1), "equals"),
        (Equals(("count",), True), "equals"),
        (Equals(("owner",), "none"), "equals"),
        (Equals(("absent",), None), "equals"),
        (Equals(("site", 0), "s-1"), "equals"),
        (Equals(("items", "0"), 1), "equals"),
    ],
)
def test_an_assertion_that_does_not_hold_fails_with_its_kind(
    assertion: Any, reason: str
) -> None:
    assert judge_result(json_result(_DOCUMENT), "json_text", (assertion,)) == (
        FailureClass.ASSERTION_FAILED,
        reason,
    )


@pytest.mark.parametrize(
    "document", [{"status": "error"}, {"ok": True}, {"status": True}]
)
def test_status_success_needs_the_success_envelope(document: Any) -> None:
    assert judge_result(json_result(document), "json_text", (StatusSuccess(),)) == (
        FailureClass.ASSERTION_FAILED,
        "status_success",
    )


def test_status_success_fails_on_a_payload_that_is_not_an_object() -> None:
    raw = json_result({"status": "success"})

    assert judge_result(raw, "json_text_items", (StatusSuccess(),)) == (
        FailureClass.ASSERTION_FAILED,
        "status_success",
    )


def test_the_first_assertion_that_fails_is_reported() -> None:
    assertions = (
        StatusSuccess(),
        Length(("items",), max=1),
        HasKeys((), ("absent",)),
    )

    assert judge_result(json_result(_DOCUMENT), "json_text", assertions) == (
        FailureClass.ASSERTION_FAILED,
        "length",
    )


def test_an_unresolved_fixture_never_reaches_the_evaluator() -> None:
    with pytest.raises(TypeError):
        judge_result(
            json_result(_DOCUMENT),
            "json_text",
            (Equals(("site", "id"), FixtureRef("jira.site_id")),),
        )


# L3Envelope


@pytest.mark.parametrize(
    ("raw", "expect_keys", "expected"),
    [
        (json_result({"status": "success"}), (), PASS),
        (json_result({"status": "success", "sub": "x"}), ("sub",), PASS),
        (json_result({"sub": "x"}), ("sub",), PASS),
        (
            json_result({"status": "error", "message": "token expired"}),
            (),
            (FailureClass.ASSERTION_FAILED, "status_not_success"),
        ),
        (
            json_result({"sub": "x"}),
            (),
            (FailureClass.ASSERTION_FAILED, "no_status"),
        ),
        (
            json_result({"status": "success"}),
            ("sub",),
            (FailureClass.ASSERTION_FAILED, "missing_keys"),
        ),
        (
            text_result("Error: unauthorized"),
            ("sub",),
            (FailureClass.ASSERTION_FAILED, "not_a_json_object"),
        ),
        (
            text_result("{}", is_error=True),
            (),
            (FailureClass.CALL_FAILED, "tool_error"),
        ),
        (None, (), (FailureClass.CALL_FAILED, "invalid_result")),
    ],
)
def test_l3_envelope_judges_as_the_l3_probe(
    raw: Any, expect_keys: tuple[str, ...], expected: Any
) -> None:
    assert judge_result(raw, "json_text", (L3Envelope(expect_keys),)) == expected


# Expected values


def test_resolve_expected_fills_each_fixture_in_an_equals_value() -> None:
    assertions = (
        StatusSuccess(),
        Equals(("site", "id"), FixtureRef("jira.site_id")),
        Equals(("count",), 2),
    )

    resolved = resolve_expected(assertions, Resolver({"jira.site_id": "s-1"}))

    assert resolved == (
        StatusSuccess(),
        Equals(("site", "id"), "s-1"),
        Equals(("count",), 2),
    )
    assert judge_result(json_result(_DOCUMENT), "json_text", resolved) == PASS


def test_resolve_expected_raises_for_a_fixture_without_a_value() -> None:
    assertions = (Equals(("site", "id"), FixtureRef("jira.site_id")),)

    with pytest.raises(FixtureMissing) as raised:
        resolve_expected(assertions, Resolver({}))

    assert raised.value.key == "jira.site_id"


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": json.dumps({"status": "success"})}],
        [{"type": "text", "text": json.dumps({"a": 1})}] * 2,
        [{"type": "text", "text": json.dumps([1])}],
        [{"type": "text", "text": "not json"}],
        [],
        [{"type": "image", "data": "..."}],
        [{"type": "text", "text": "{}"}, {"type": "image", "data": "..."}],
        [{"type": "text", "text": 7}],
        ["{}"],
        None,
        "{}",
    ],
)
def test_json_text_takes_the_object_the_l3_probe_takes(content: Any) -> None:
    raw = {"content": content, "structured_content": None, "is_error": False}

    rejected = judge_result(raw, "json_text", (HasKeys((), ()),)) == (
        FailureClass.ASSERTION_FAILED,
        "not_a_json_object",
    )

    assert rejected == (_single_json_object(content) is None)
