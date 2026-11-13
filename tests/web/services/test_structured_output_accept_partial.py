"""The accept_partial switch from the structured outcome of a run to the fields
a task response carries."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, NamedTuple

import pytest

from tests.core.agent.test_structured_output import (
    MARKER,
    NULLABLE_STRING,
    STRING,
    ScriptedModel,
    _args_reply,
    _fmt,
    _obj,
    _text_reply,
)
from tests.web.services.test_structured_output_projection import MESSAGES
from xagent.core.agent import structured_output as so
from xagent.core.agent import structured_schema as ss
from xagent.core.agent.result import COMPLETION_OUTCOMES
from xagent.core.model.chat.exceptions import LLMRetryableError
from xagent.web.models.task import TaskStatus
from xagent.web.services.structured_output_projection import (
    _FAILED_CODES,
    _GAP_CODES,
    _INCOMPLETE_CODES,
    _SELF_RATED_CODES,
    MAX_REASON_FIELDS,
    REASON_MESSAGES,
    REASON_MESSAGES_WITH_RESULT,
    _partial_problem,
    build_record,
    project_structured_output,
)

RUN_ID = "run-1"
REQUIRED = _obj({"a": STRING, "g": STRING})
ALL_NULLABLE = _obj({"a": NULLABLE_STRING, "b": NULLABLE_STRING})


class SlowModel(ScriptedModel):
    async def chat(self, **kwargs: Any) -> Any:
        await asyncio.sleep(5)


class Outcome(NamedTuple):
    """One way an extraction can end: the schema, the answer, the model script
    and what to patch."""

    name: str
    script: list[Any]
    schema: dict[str, Any] = REQUIRED
    answer: str = "The answer."
    patches: tuple[tuple[Any, str, Any], ...] = ()
    model: type[ScriptedModel] = ScriptedModel


OUTCOMES = [
    Outcome("ready", [_args_reply({"a": "x", "g": "y"})]),
    Outcome("information-missing", [_args_reply({"a": "x", "g": None})]),
    Outcome("value-not-expressible", [_args_reply({"a": "x", "g": MARKER})]),
    Outcome("root-is-a-gap", [_args_reply({"a": None, "g": None})]),
    Outcome(
        "dropped-for-size",
        [_args_reply({"a": "x", "g": None})],
        patches=((ss, "MAX_RESULT_BYTES", 1),),
    ),
    Outcome("schema-mismatch", [_text_reply()] * 3),
    Outcome("model-error", [LLMRetryableError("boom")]),
    Outcome(
        "extraction-timeout",
        [],
        patches=((so, "EXTRACTION_TIMEOUT_SECONDS", 0.05),),
        model=SlowModel,
    ),
    Outcome(
        "result-too-large",
        [_args_reply({"a": "x" * 40, "g": "y"})],
        patches=((so, "MAX_ARGUMENTS_BYTES", 20),),
    ),
    Outcome("internal-error", [_args_reply({"a": "x", "g": "y"})]),
    Outcome("empty-answer-all-nullable", [], ALL_NULLABLE, ""),
    Outcome("empty-answer-required-root", [], REQUIRED, ""),
]
RATINGS: list[Any] = [*sorted(COMPLETION_OUTCOMES), None]


async def _read_back(
    outcome: Outcome, rating: Any, accept_partial: bool, monkeypatch: Any
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run the extraction, store the record and project it as a response."""
    result = {
        "success": True,
        "status": "completed",
        "output": outcome.answer,
        "completion_outcome": rating,
    }
    with monkeypatch.context() as scoped:
        for module, name, value in outcome.patches:
            scoped.setattr(module, name, value)
        if outcome.name == "internal-error":

            async def boom(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("boom")

            scoped.setattr(so, "run_structured_output_step", boom)
        attached = await so._attach_structured_output(
            result,
            _fmt(outcome.schema, accept_partial),
            llm=outcome.model(copy.deepcopy(outcome.script)),
            task_id="t-1",
            run_id=RUN_ID,
            tracer=None,
        )
    assert attached is not None
    record = build_record(RUN_ID, attached.get("structured_output"))
    fields = project_structured_output(
        status=TaskStatus.COMPLETED,
        run_id=RUN_ID,
        record=record,
        format_configured=True,
    )
    return record, fields.as_dict()


PARTIAL_KEYS = ("partial_output", "partial_gaps")


async def test_status_and_code_independent_of_switch(monkeypatch) -> None:
    assert len(OUTCOMES) == 12
    cells = 0
    for outcome in OUTCOMES:
        for rating in RATINGS:
            record_off, off = await _read_back(outcome, rating, False, monkeypatch)
            record_on, on = await _read_back(outcome, rating, True, monkeypatch)
            cells += 1
            reason_off = off["structured_output_reason"]
            reason_on = on["structured_output_reason"]
            assert on["structured_output_status"] == off["structured_output_status"], (
                outcome.name,
                rating,
            )
            assert (reason_on or {}).get("code") == (reason_off or {}).get("code"), (
                outcome.name,
                rating,
            )
            # Without its two new keys the record is the one the switch off writes.
            assert record_off is not None and record_on is not None
            stripped = {k: v for k, v in record_on.items() if k not in PARTIAL_KEYS}
            assert json.dumps(stripped, sort_keys=True) == json.dumps(
                record_off, sort_keys=True
            ), (outcome.name, rating)
    assert cells == 48


# --- the record and what is read back from it ------------------------------------------

BY_NAME = {outcome.name: outcome for outcome in OUTCOMES}

# The record of a request without the switch, for each way a run can end:
# (outcome, the agent's rating, the record).
_SELF_RATED_RECORD = {"status": "incomplete", "output": None}
DEFAULT_RECORDS = {
    "ready": (
        "ready",
        "completed",
        {"status": "ready", "output": {"a": "x", "g": "y"}, "reason": None},
    ),
    "information-missing": (
        "information-missing",
        "completed",
        {
            "status": "incomplete",
            "output": None,
            "reason": {"code": "information_missing", "missing_fields": ["/g"]},
        },
    ),
    "value-not-expressible": (
        "value-not-expressible",
        "completed",
        {
            "status": "incomplete",
            "output": None,
            "reason": {"code": "value_not_expressible", "unexpressible_fields": ["/g"]},
        },
    ),
    "schema-mismatch": (
        "schema-mismatch",
        "completed",
        {"status": "failed", "output": None, "reason": {"code": "schema_mismatch"}},
    ),
    "agent-partial": (
        "ready",
        "partial",
        {**_SELF_RATED_RECORD, "reason": {"code": "agent_partial"}},
    ),
    "agent-blocked": (
        "ready",
        "blocked",
        {**_SELF_RATED_RECORD, "reason": {"code": "agent_blocked"}},
    ),
}
assert len(DEFAULT_RECORDS) == 6


@pytest.mark.parametrize("cell", list(DEFAULT_RECORDS))
async def test_default_request_record_unchanged(cell: str, monkeypatch) -> None:
    name, rating, expected = DEFAULT_RECORDS[cell]
    record, _ = await _read_back(BY_NAME[name], rating, False, monkeypatch)

    assert record == {"run_id": RUN_ID, **expected}
    assert list(record) == ["run_id", "status", "output", "reason"]


def _record(status: str, code: str | None, **extra: Any) -> dict[str, Any]:
    reason: dict[str, Any] | None = None
    if status != "ready":
        reason = {"code": code}
        if code == "information_missing":
            reason["missing_fields"] = ["/g"]
        if code == "value_not_expressible":
            reason["unexpressible_fields"] = ["/g"]
    return {
        "run_id": RUN_ID,
        "status": status,
        "output": {"a": "x"} if status == "ready" else None,
        "reason": reason,
        **extra,
    }


def _partial_cells() -> list[tuple[str, dict[str, Any], str | None]]:
    """``(row, record, problem)`` for every row of the table of ``_partial_problem``."""
    cells: list[tuple[str, dict[str, Any], str | None]] = []
    codes = sorted(_INCOMPLETE_CODES) + sorted(_FAILED_CODES)

    def status_of(code: str) -> str:
        return "incomplete" if code in _INCOMPLETE_CODES else "failed"

    for code in codes:
        cells.append(("1", _record(status_of(code), code), None))
        cells.append(
            (
                "2",
                _record(status_of(code), code, partial_gaps={"missing_fields": ["/g"]}),
                "partial_gaps",
            )
        )
    cells.append(("1", _record("ready", None), None))
    cells.append(
        (
            "2",
            _record("ready", None, partial_gaps={"missing_fields": ["/g"]}),
            "partial_gaps",
        )
    )
    for bad in (["x"], None, "x", 1):
        cells.append(
            (
                "3",
                _record("incomplete", "agent_partial", partial_output=bad),
                "partial_output",
            )
        )
    cells.append(
        ("4", _record("ready", None, partial_output={"a": None}), "partial_output")
    )
    for code in sorted(_FAILED_CODES):
        cells.append(
            ("4", _record("failed", code, partial_output={"a": None}), "partial_output")
        )
    for code in sorted(_GAP_CODES):
        cells.append(
            (
                "5",
                _record(
                    "incomplete",
                    code,
                    partial_output={"a": None},
                    partial_gaps={"missing_fields": ["/g"]},
                ),
                "partial_gaps",
            )
        )
        cells.append(
            ("6", _record("incomplete", code, partial_output={"a": None}), None)
        )
    for code in sorted(_SELF_RATED_CODES):
        cells.append(
            ("7", _record("incomplete", code, partial_output={"a": None}), None)
        )
        bad_gaps: list[Any] = [
            [],
            "x",
            {},
            {"other": ["/g"]},
            {"missing_fields": ["/g"], "other": ["/g"]},
            {"missing_fields": []},
            {"missing_fields": "/g"},
            {"missing_fields": [1]},
            {"unexpressible_fields": ["/g"] * (MAX_REASON_FIELDS + 1)},
            {"missing_fields": ["/g"], "unexpressible_fields": []},
        ]
        for gaps in bad_gaps:
            cells.append(
                (
                    "8",
                    _record(
                        "incomplete",
                        code,
                        partial_output={"a": None},
                        partial_gaps=gaps,
                    ),
                    "partial_gaps",
                )
            )
        good_gaps: list[Any] = [
            {"missing_fields": ["/g"]},
            {"unexpressible_fields": ["/g"]},
            {"missing_fields": ["/g"], "unexpressible_fields": ["/a"]},
            {"missing_fields": ["/g"] * MAX_REASON_FIELDS},
        ]
        for gaps in good_gaps:
            cells.append(
                (
                    "9",
                    _record(
                        "incomplete",
                        code,
                        partial_output={"a": None},
                        partial_gaps=gaps,
                    ),
                    None,
                )
            )
    return cells


def test_partial_problem() -> None:
    cells = _partial_cells()

    assert {row for row, _, _ in cells} == {str(n) for n in range(1, 10)}
    assert _GAP_CODES | _SELF_RATED_CODES == _INCOMPLETE_CODES
    assert not _GAP_CODES & _SELF_RATED_CODES
    for row, record, problem in cells:
        assert _partial_problem(record) == problem, (row, record)
        fields = project_structured_output(
            status=TaskStatus.COMPLETED,
            run_id=RUN_ID,
            record=record,
            format_configured=True,
        )
        if problem is None:
            assert fields.structured_output_status == record["status"], (row, record)
        else:
            assert fields.structured_output_status == "failed"
            assert (fields.structured_output_reason or {})["code"] == "internal_error"


KNOWN_TOP_KEYS = {
    "run_id",
    "status",
    "output",
    "reason",
    "partial_output",
    "partial_gaps",
}


def _legal_records() -> list[dict[str, Any]]:
    records = [_record("ready", None)]
    records += [_record("incomplete", code) for code in sorted(_INCOMPLETE_CODES)]
    records += [_record("failed", code) for code in sorted(_FAILED_CODES)]
    return records


def _project(record: Any) -> dict[str, Any]:
    return project_structured_output(
        status=TaskStatus.COMPLETED,
        run_id=RUN_ID,
        record=record,
        format_configured=True,
    ).as_dict()


def test_projection_ignores_unknown_extra_keys() -> None:
    names = ("zz", "future_key")
    values: list[Any] = [{"k": 1}, ["x"], None]
    assert not KNOWN_TOP_KEYS & set(names)
    cells = 0
    for record in _legal_records():
        baseline = _project(record)
        for name in names:
            for value in values:
                top = {**record, name: copy.deepcopy(value)}
                assert _project(top) == baseline
                cells += 1
                if record["reason"] is not None:
                    inner = {
                        **record,
                        "reason": {**record["reason"], name: copy.deepcopy(value)},
                    }
                    assert _project(inner) == baseline
                    cells += 1
    assert cells == (1 + 4 + 6) * 6 + (4 + 6) * 6


def test_build_record_writes_the_two_keys_and_keeps_the_output_null() -> None:
    gaps = {
        "missing_fields": [f"/g/{i}" for i in range(30)],
        "unexpressible_fields": ["/u"],
    }
    outcome = {
        "status": "incomplete",
        "output": None,
        "reason": {"code": "agent_partial"},
        "partial_output": {"a": None},
        "partial_gaps": gaps,
    }

    record = build_record(RUN_ID, outcome)

    assert record == {
        "run_id": RUN_ID,
        "status": "incomplete",
        "output": None,
        "reason": {"code": "agent_partial"},
        "partial_output": {"a": None},
        "partial_gaps": {
            "missing_fields": [f"/g/{i}" for i in range(MAX_REASON_FIELDS)],
            "unexpressible_fields": ["/u"],
        },
    }
    assert outcome["partial_gaps"] == gaps


@pytest.mark.parametrize("bad", [None, ["x"], "x", 1])
def test_build_record_ignores_keys_that_are_not_mappings(bad: Any) -> None:
    outcome = {
        "status": "incomplete",
        "output": None,
        "reason": {"code": "agent_partial"},
        "partial_output": bad,
        "partial_gaps": bad,
    }

    assert set(build_record(RUN_ID, outcome) or {}) == {
        "run_id",
        "status",
        "output",
        "reason",
    }


# The two messages of a response that holds values, character for character.
WITH_RESULT = {
    "agent_partial": "The agent completed only part of the task. structured_output holds the values extracted from its answer, with null where a value is missing or cannot be written; missing_fields and unexpressible_fields list up to 20 such positions each.",
    "agent_blocked": "The agent stopped without completing the task. structured_output holds the values extracted from its answer, with null where a value is missing or cannot be written; missing_fields and unexpressible_fields list up to 20 such positions each.",
}


@pytest.mark.parametrize("with_values", [False, True])
@pytest.mark.parametrize("code", sorted(_SELF_RATED_CODES))
def test_self_rated_message_variants(code: str, with_values: bool) -> None:
    extra = {"partial_output": {"a": None}} if with_values else {}
    fields = _project(_record("incomplete", code, **extra))
    reason = fields["structured_output_reason"]

    assert reason["message"] == (WITH_RESULT if with_values else MESSAGES)[code]
    assert fields["structured_output"] == ({"a": None} if with_values else None)
    assert fields["structured_output_status"] == "incomplete"
    assert REASON_MESSAGES == MESSAGES
    assert REASON_MESSAGES_WITH_RESULT == WITH_RESULT
    assert set(REASON_MESSAGES_WITH_RESULT) == _SELF_RATED_CODES


def test_a_self_rated_response_carries_the_lists_of_partial_gaps_in_order() -> None:
    gaps = {"unexpressible_fields": ["/u"], "missing_fields": ["/m"]}
    reason = _project(
        _record(
            "incomplete", "agent_blocked", partial_output={"a": None}, partial_gaps=gaps
        )
    )["structured_output_reason"]

    assert list(reason) == ["code", "message", "missing_fields", "unexpressible_fields"]
    assert reason["missing_fields"] == ["/m"] and reason["unexpressible_fields"] == [
        "/u"
    ]
