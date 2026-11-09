"""Judge one tool call's result against a check's assertions.

The result is the tool adapter's normalized result,
``{"content": [...], "structured_content": ..., "is_error": ...}``. A result
that is not a mapping, or that is flagged ``is_error``, fails as
``call_failed`` before any assertion runs. Otherwise the check's payload is
taken from it (``declarations.PayloadSource``) and the assertions run in
order; the first that does not hold fails the check as ``assertion_failed``
with its kind as the reason (``has_keys``, ``equals``, …). A payload that
cannot be taken fails the same way, with a reason naming the payload rule.

``L3Envelope`` hands the whole result to the L3 probe's own judgement
(``evaluate_probe_result``) and keeps its reasons, so a check derived from
``connector_probes.py`` passes and fails exactly as the probe route does.

Reasons are fixed codes: nothing from the result reaches them.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..connector_probes import ConnectorProbe
from ..services.connector_probe import ConnectorProbeOutcome, evaluate_probe_result
from .declarations import (
    Assertion,
    CreatedRef,
    EachHasKeys,
    Equals,
    HasKeys,
    IsType,
    L3Envelope,
    Length,
    PayloadPath,
    PayloadSource,
    RunScopedName,
    StatusSuccess,
)
from .fixtures import FixtureRef, FixtureResolver, resolve_fixtures
from .results import FailureClass

# What a judgement returns: (None, None) when every assertion holds.
Judgement = tuple[FailureClass | None, str | None]

_PASS: Judgement = (None, None)

_L3_CLASSES = {
    ConnectorProbeOutcome.PROBE_FAILED: FailureClass.CALL_FAILED,
    ConnectorProbeOutcome.PROBE_ASSERTION_FAILED: FailureClass.ASSERTION_FAILED,
}

_KINDS: Mapping[type, str] = {
    StatusSuccess: "status_success",
    HasKeys: "has_keys",
    EachHasKeys: "each_has_keys",
    IsType: "is_type",
    Length: "length",
    Equals: "equals",
}

_MISSING = object()
# What ``value_at`` returns where the result holds no value.
MISSING: Any = _MISSING


class _NoPayload(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def judge_result(
    raw: Any, payload: PayloadSource, assertions: Sequence[Assertion]
) -> Judgement:
    """The failure class and reason of ``raw`` under ``assertions``, or
    ``(None, None)`` when the check passes.

    Every fixture, run-scoped name and created-object reference in an
    ``Equals`` value must already be resolved (``resolve_expected`` resolves
    fixtures); an unresolved one raises ``TypeError``.
    """
    for assertion in assertions:
        if isinstance(assertion, Equals) and isinstance(
            assertion.value, (FixtureRef, RunScopedName, CreatedRef)
        ):
            raise TypeError("an expected value must be resolved before judging")
    if len(assertions) == 1 and isinstance(assertions[0], L3Envelope):
        return _judge_l3(raw, assertions[0])
    if not isinstance(raw, Mapping):
        return FailureClass.CALL_FAILED, "invalid_result"
    if raw.get("is_error"):
        return FailureClass.CALL_FAILED, "tool_error"
    try:
        value = _payload(raw, payload)
    except _NoPayload as missing:
        return FailureClass.ASSERTION_FAILED, missing.reason
    for assertion in assertions:
        if not _holds(assertion, value):
            return FailureClass.ASSERTION_FAILED, _KINDS[type(assertion)]
    return _PASS


def resolve_expected(
    assertions: Sequence[Assertion], resolver: FixtureResolver
) -> tuple[Assertion, ...]:
    """``assertions`` with every ``Equals`` fixture replaced by its value.

    Raises ``fixtures.FixtureMissing`` for the first key without a value.
    """
    resolved: list[Assertion] = []
    for assertion in assertions:
        if isinstance(assertion, Equals) and isinstance(assertion.value, FixtureRef):
            assertion = dataclasses.replace(
                assertion, value=resolve_fixtures(assertion.value, resolver)
            )
        resolved.append(assertion)
    return tuple(resolved)


def value_at(raw: Any, payload: PayloadSource, path: PayloadPath) -> Any:
    """The value at ``path`` in ``raw``'s ``payload``, or ``MISSING``.

    A write check reads its tenant id and a created object's id this way. A
    result that is not a mapping, is flagged ``is_error`` or has no such
    payload holds no value.
    """
    if not isinstance(raw, Mapping) or raw.get("is_error"):
        return MISSING
    try:
        value = _payload(raw, payload)
    except _NoPayload:
        return MISSING
    return _at(value, path)


def json_equal(actual: Any, expected: Any) -> bool:
    """Whether two values are equal as JSON compares them (``Equals``)."""
    return _json_equal(actual, expected)


def _judge_l3(raw: Any, assertion: L3Envelope) -> Judgement:
    # evaluate_probe_result reads only the probe's expect_keys.
    probe = ConnectorProbe("", expect_keys=assertion.expect_keys)
    outcome, reason = evaluate_probe_result(raw, probe)
    if outcome is ConnectorProbeOutcome.OK:
        return _PASS
    return _L3_CLASSES[outcome], reason


def _payload(raw: Mapping[str, Any], source: PayloadSource) -> Any:
    if source == "structured_content":
        structured = raw.get("structured_content")
        if not isinstance(structured, Mapping):
            raise _NoPayload("no_structured_content")
        return structured
    content = raw.get("content")
    if source == "json_text":
        # One content item, a text holding a JSON object.
        if not isinstance(content, list) or len(content) != 1:
            raise _NoPayload("not_a_json_object")
        texts = _texts(content)
        parsed = _parse(texts[0]) if texts else _MISSING
        if not isinstance(parsed, dict):
            raise _NoPayload("not_a_json_object")
        return parsed
    texts = _texts(content)
    if texts is None:
        raise _NoPayload("not_json_text")
    items = [_parse(text) for text in texts]
    if any(item is _MISSING for item in items):
        raise _NoPayload("not_json_text")
    return items


def _texts(content: Any) -> list[str] | None:
    """The text of every text item, in order; None when there is no list."""
    if not isinstance(content, list):
        return None
    return [
        item["text"]
        for item in content
        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
    ]


def _parse(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        # Text nested past the parser's depth limit raises RecursionError.
        return _MISSING


def _at(value: Any, path: PayloadPath) -> Any:
    for segment in path:
        if isinstance(segment, str):
            if not isinstance(value, Mapping) or segment not in value:
                return _MISSING
        elif not isinstance(value, list) or segment >= len(value):
            return _MISSING
        value = value[segment]
    return value


def _holds(assertion: Assertion, payload: Any) -> bool:
    if isinstance(assertion, StatusSuccess):
        return isinstance(payload, Mapping) and payload.get("status") == "success"
    if isinstance(assertion, L3Envelope):
        # The declaration rules make it a check's only assertion.
        raise TypeError("L3Envelope must be a check's only assertion")
    value = _at(payload, assertion.path)
    if value is _MISSING:
        return False
    if isinstance(assertion, HasKeys):
        return _has_keys(value, assertion.keys)
    if isinstance(assertion, EachHasKeys):
        return isinstance(value, list) and all(
            _has_keys(item, assertion.keys) for item in value
        )
    if isinstance(assertion, IsType):
        return _json_type(value) == assertion.type
    if isinstance(assertion, Length):
        return (
            isinstance(value, list)
            and len(value) >= assertion.min
            and (assertion.max is None or len(value) <= assertion.max)
        )
    return _json_equal(value, assertion.value)


def _has_keys(value: Any, keys: tuple[str, ...]) -> bool:
    return isinstance(value, Mapping) and all(key in value for key in keys)


def _json_type(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return None


def _json_equal(actual: Any, expected: Any) -> bool:
    """JSON equality: a boolean never equals a number, unlike in Python."""
    if expected is None or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and actual == expected
        )
    return isinstance(actual, str) and actual == expected
