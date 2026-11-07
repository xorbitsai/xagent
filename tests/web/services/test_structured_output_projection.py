"""project_structured_output: what a read path reports for a task, given the
effective status, the run it is showing, the stored record and whether the
task has a format configured."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from xagent.web.models.task import TaskStatus
from xagent.web.services.structured_output_projection import (
    project_structured_output,
    response_format_configured,
)

RUN = "run-2"
OLD_RUN = "run-1"

# The fixed message of every code, character for character.
MESSAGES = {
    "agent_partial": "The agent completed only part of the task, so no structured result was extracted.",
    "agent_blocked": "The agent stopped without completing the task, so no structured result was extracted.",
    "information_missing": "The final answer does not state some required values. missing_fields lists up to 20 of them.",
    "value_not_expressible": "The final answer states some values that cannot be written in the form the schema requires. unexpressible_fields lists up to 20 of them.",
    "not_produced": "No structured result was produced for the run shown in this response.",
    "schema_mismatch": "The extraction model did not return a result that matches the schema within 3 attempts.",
    "result_too_large": "The structured result is too large to return: it exceeds a size limit.",
    "model_error": "The extraction model was unavailable or its call failed.",
    "invalid_stored_format": "The response_format stored on this task is no longer valid, so no structured result was extracted.",
    "extraction_timeout": "Structured result extraction did not finish within 120 seconds.",
    "internal_error": "Structured result extraction failed because of an internal error.",
}
INCOMPLETE_CODES = (
    "agent_partial",
    "agent_blocked",
    "information_missing",
    "value_not_expressible",
)
FAILED_CODES = (
    "schema_mismatch",
    "result_too_large",
    "model_error",
    "invalid_stored_format",
    "extraction_timeout",
    "internal_error",
)
NULLS = {
    "structured_output": None,
    "structured_output_status": None,
    "structured_output_reason": None,
}


def ready(**over: Any) -> dict[str, Any]:
    return {
        "run_id": RUN,
        "status": "ready",
        "output": {"a": 1},
        "reason": None,
        **over,
    }


def incomplete(code: str = "agent_partial", **reason: Any) -> dict[str, Any]:
    return {
        "run_id": RUN,
        "status": "incomplete",
        "output": None,
        "reason": {"code": code, **reason},
    }


def failed(code: str = "model_error", **reason: Any) -> dict[str, Any]:
    return {
        "run_id": RUN,
        "status": "failed",
        "output": None,
        "reason": {"code": code, **reason},
    }


def project(record: Any, *, status=TaskStatus.COMPLETED, run_id=RUN, configured=True):
    return project_structured_output(
        status=status, run_id=run_id, record=record, format_configured=configured
    ).as_dict()


def as_fields(status: str, reason: dict[str, Any]) -> dict[str, Any]:
    return {
        "structured_output": None,
        "structured_output_status": status,
        "structured_output_reason": reason,
    }


NOT_PRODUCED = as_fields(
    "incomplete", {"code": "not_produced", "message": MESSAGES["not_produced"]}
)
INTERNAL_ERROR = as_fields(
    "failed", {"code": "internal_error", "message": MESSAGES["internal_error"]}
)


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.WAITING_FOR_USER,
        TaskStatus.FAILED,
    ],
)
def test_a_task_that_is_not_completed_reports_nothing(status):
    assert project(ready(), status=status) == NULLS
    assert project(None, status=status) == NULLS
    assert project("malformed", status=status) == NULLS


def test_a_completed_task_reports_the_record_of_the_run_it_shows():
    assert project(ready()) == {
        "structured_output": {"a": 1},
        "structured_output_status": "ready",
        "structured_output_reason": None,
    }


def test_the_output_keeps_the_key_order_it_was_stored_with():
    fields = project(ready(output={"z": 1, "a": {"y": 1, "b": 2}}))

    assert list(fields["structured_output"]) == ["z", "a"]
    assert list(fields["structured_output"]["a"]) == ["y", "b"]


@pytest.mark.parametrize("configured", [True, False])
def test_a_malformed_record_is_internal_error_with_a_format_and_nothing_without(
    configured, caplog
):
    with caplog.at_level(logging.WARNING):
        fields = project("malformed", configured=configured)

    assert fields == (INTERNAL_ERROR if configured else NULLS)
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_a_completed_task_without_a_record_has_nothing_to_report_without_a_format():
    assert project(None, configured=False) == NULLS


def test_a_completed_task_without_a_record_was_not_produced_with_a_format():
    assert project(None, configured=True) == NOT_PRODUCED


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize(
    "stale",
    [ready(run_id=OLD_RUN), {"run_id": OLD_RUN, "status": 7}, {"run_id": OLD_RUN}],
    ids=["valid", "bad status", "missing keys"],
)
def test_a_record_of_an_earlier_run_is_no_record_whatever_else_is_wrong_with_it(
    stale, configured, caplog
):
    with caplog.at_level(logging.WARNING):
        fields = project(stale, configured=configured)

    assert fields == (NOT_PRODUCED if configured else NULLS)
    assert caplog.records == []


@pytest.mark.parametrize(
    "record",
    [
        ["run_id", RUN],
        {"status": "ready", "output": {}, "reason": None},
        {"run_id": 5, "status": "ready", "output": {}, "reason": None},
        {"run_id": None, "status": "ready", "output": {}, "reason": None},
    ],
    ids=["not a dict", "no run_id", "run_id not a string", "run_id None"],
)
def test_a_record_that_does_not_say_which_run_it_belongs_to_is_malformed(record):
    assert project(record) == INTERNAL_ERROR


def drop(record: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != key}


MALFORMED = {
    "status missing": drop(ready(), "status"),
    "output missing": drop(ready(), "output"),
    "reason missing": drop(ready(), "reason"),
    "status unknown": ready(status="done"),
    "status not a string": ready(status=["ready"]),
    "ready output not a dict": ready(output=[1]),
    "ready with a reason": ready(reason={"code": "agent_partial"}),
    "incomplete with an output": incomplete() | {"output": {"a": 1}},
    "incomplete without a reason": incomplete() | {"reason": None},
    "reason not a dict": incomplete() | {"reason": "agent_partial"},
    "code missing": incomplete() | {"reason": {}},
    "code not a string": incomplete() | {"reason": {"code": ["agent_partial"]}},
    "failed code on incomplete": incomplete("model_error"),
    "incomplete code on failed": failed("agent_partial"),
    "failed information_missing": failed("information_missing", missing_fields=["/a"]),
    "not_produced stored": incomplete("not_produced"),
    "unknown code": incomplete("agent_exploded"),
    "information_missing without missing_fields": incomplete("information_missing"),
    "missing_fields empty": incomplete("information_missing", missing_fields=[]),
    "missing_fields 21 items": incomplete(
        "information_missing", missing_fields=[f"/f{i}" for i in range(21)]
    ),
    "missing_fields not a list": incomplete("information_missing", missing_fields="/a"),
    "missing_fields tuple": incomplete("information_missing", missing_fields=("/a",)),
    "missing_fields non-string": incomplete("information_missing", missing_fields=[1]),
    "missing_fields on another code": incomplete(
        "agent_partial", missing_fields=["/a"]
    ),
    "missing_fields on value_not_expressible": incomplete(
        "value_not_expressible", unexpressible_fields=["/a"], missing_fields=["/b"]
    ),
    "value_not_expressible without unexpressible_fields": incomplete(
        "value_not_expressible"
    ),
    "unexpressible_fields empty": incomplete(
        "value_not_expressible", unexpressible_fields=[]
    ),
    "unexpressible_fields 21 items": incomplete(
        "value_not_expressible",
        unexpressible_fields=[f"/f{i}" for i in range(21)],
    ),
    "unexpressible_fields non-string": incomplete(
        "information_missing", missing_fields=["/a"], unexpressible_fields=["/b", 2]
    ),
    "unexpressible_fields on another code": incomplete(
        "agent_blocked", unexpressible_fields=["/a"]
    ),
    "fields on a failed code": failed("model_error", unexpressible_fields=["/a"]),
}


@pytest.mark.parametrize("record", MALFORMED.values(), ids=MALFORMED.keys())
def test_every_sentence_of_the_record_shape_is_enforced(record, caplog):
    with caplog.at_level(logging.WARNING):
        fields = project(record)

    assert fields == INTERNAL_ERROR
    assert len(caplog.records) == 1


def reported(status: str, code: str, **fields: Any) -> dict[str, Any]:
    return as_fields(status, {"code": code, "message": MESSAGES[code], **fields})


VALID = {
    "extra record key": (
        ready(message="free text"),
        {
            "structured_output": {"a": 1},
            "structured_output_status": "ready",
            "structured_output_reason": None,
        },
    ),
    "extra reason key": (
        incomplete("agent_partial", message="free text", note="x"),
        reported("incomplete", "agent_partial"),
    ),
    "information_missing": (
        incomplete("information_missing", missing_fields=["/a", "/b"]),
        reported("incomplete", "information_missing", missing_fields=["/a", "/b"]),
    ),
    "information_missing with both lists": (
        incomplete(
            "information_missing", missing_fields=["/a"], unexpressible_fields=["/b"]
        ),
        reported(
            "incomplete",
            "information_missing",
            missing_fields=["/a"],
            unexpressible_fields=["/b"],
        ),
    ),
    "value_not_expressible": (
        incomplete("value_not_expressible", unexpressible_fields=["/a"]),
        reported("incomplete", "value_not_expressible", unexpressible_fields=["/a"]),
    ),
    "twenty fields": (
        incomplete("information_missing", missing_fields=[f"/f{i}" for i in range(20)]),
        reported(
            "incomplete",
            "information_missing",
            missing_fields=[f"/f{i}" for i in range(20)],
        ),
    ),
    "result_too_large": (
        failed("result_too_large"),
        reported("failed", "result_too_large"),
    ),
    "stored internal_error": (
        failed("internal_error"),
        reported("failed", "internal_error"),
    ),
}


@pytest.mark.parametrize(("record", "expected"), VALID.values(), ids=VALID.keys())
def test_a_record_of_the_right_shape_is_reported_whatever_extra_keys_it_has(
    record, expected, caplog
):
    with caplog.at_level(logging.WARNING):
        fields = project(record)

    assert fields == expected
    assert caplog.records == []
    assert "free text" not in json.dumps(fields)


def test_the_reason_lists_come_out_in_the_stored_order_after_code_and_message():
    fields = project(
        incomplete(
            "information_missing",
            unexpressible_fields=["/z", "/a"],
            missing_fields=["/y", "/b"],
        )
    )

    assert list(fields["structured_output_reason"]) == [
        "code",
        "message",
        "missing_fields",
        "unexpressible_fields",
    ]
    assert fields["structured_output_reason"]["missing_fields"] == ["/y", "/b"]
    assert fields["structured_output_reason"]["unexpressible_fields"] == ["/z", "/a"]


def test_a_reason_has_no_key_for_a_list_the_record_does_not_carry():
    reason = project(incomplete("agent_partial"))["structured_output_reason"]

    assert set(reason) == {"code", "message"}


def test_the_fixed_messages_cover_every_code_the_projection_can_report():
    from xagent.web.services.structured_output_projection import REASON_MESSAGES

    assert REASON_MESSAGES == MESSAGES


@pytest.mark.parametrize(
    ("agent_config", "expected"),
    [
        (None, False),
        ({}, False),
        ({"other": 1}, False),
        ({"response_format": None}, False),
        ({"response_format": {"type": "text"}}, True),
        ({"response_format": {}}, True),
        ({"response_format": 0}, True),
        ("response_format", False),
        (["response_format"], False),
    ],
)
def test_a_format_is_configured_when_the_key_is_present_and_not_none(
    agent_config, expected
):
    assert response_format_configured(agent_config) is expected
