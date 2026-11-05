"""The structured output format as the web layer sees it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from xagent.core.agent import structured_output
from xagent.core.agent.structured_output import StructuredOutputFormat
from xagent.core.model.chat.exceptions import LLMRetryableError
from xagent.web.models.task import TaskStatus
from xagent.web.services import structured_output_projection as projection

STRING = {"type": "string"}


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


SHAPE = _obj({"name": STRING, "unit": STRING, "note": {"type": ["string", "null"]}})


class Model:
    model_name = "scripted"

    def __init__(self, reply: Any) -> None:
        self.reply = reply

    async def chat(self, **kwargs: Any) -> Any:
        if isinstance(self.reply, BaseException):
            raise self.reply
        if callable(self.reply):
            return await self.reply()
        return {
            "type": "tool_call",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "submit_result",
                        "arguments": json.dumps(self.reply),
                    },
                }
            ],
        }


async def _sleep_forever() -> Any:
    await asyncio.sleep(5)


def test_the_result_key_is_the_one_the_projection_reads() -> None:
    assert structured_output.STRUCTURED_OUTPUT_RESULT_KEY == projection._RESULT_KEY


OUTCOMES = [
    ("ready", None, {}, Model({"name": "a", "unit": "kg", "note": None})),
    (
        "incomplete",
        "information_missing",
        {},
        Model({"name": None, "unit": "kg", "note": None}),
    ),
    (
        "incomplete",
        "value_not_expressible",
        {},
        Model({"name": "a", "unit": "__NOT_EXPRESSIBLE__", "note": None}),
    ),
    ("incomplete", "agent_partial", {"completion_outcome": "partial"}, Model(None)),
    ("incomplete", "agent_blocked", {"completion_outcome": "blocked"}, Model(None)),
    ("failed", "schema_mismatch", {}, Model({"x": 1})),
    ("failed", "model_error", {}, Model(LLMRetryableError("boom"))),
    ("failed", "extraction_timeout", {}, Model(_sleep_forever)),
    ("failed", "internal_error", {"output": "x"}, Model(None)),
]


@pytest.mark.parametrize(("status", "code", "result", "model"), OUTCOMES)
async def test_every_outcome_reads_back_as_the_same_status_and_code(
    status: str, code: str | None, result: dict[str, Any], model: Model, monkeypatch
) -> None:
    monkeypatch.setattr(structured_output, "EXTRACTION_TIMEOUT_SECONDS", 0.05)
    if code == "internal_error":
        monkeypatch.setattr(structured_output, "compile_schema", lambda schema: 1 / 0)
    run = {"status": "completed", "output": "an answer", "success": True, **result}
    fmt = StructuredOutputFormat("shape", None, SHAPE)

    await structured_output._attach_structured_output(
        run, fmt, llm=model, task_id="t", run_id="r1", tracer=None
    )

    record = projection.build_record(
        "r1", run[structured_output.STRUCTURED_OUTPUT_RESULT_KEY]
    )
    fields = projection.project_structured_output(
        status=TaskStatus.COMPLETED,
        run_id="r1",
        record=record,
        format_configured=True,
    )
    assert fields.structured_output_status == status
    if code is None:
        assert fields.structured_output_reason is None
        assert fields.structured_output == {"name": "a", "unit": "kg", "note": None}
    else:
        assert fields.structured_output_reason["code"] == code


@pytest.mark.parametrize(
    "fmt", [StructuredOutputFormat.INVALID, StructuredOutputFormat.READ_FAILED]
)
async def test_a_format_that_could_not_be_used_reads_back_as_a_failure(fmt) -> None:
    run = {"status": "completed", "output": "x", "success": True}

    await structured_output._attach_structured_output(
        run, fmt, llm=None, task_id="t", run_id="r1", tracer=None
    )

    fields = projection.project_structured_output(
        status=TaskStatus.COMPLETED,
        run_id="r1",
        record=projection.build_record("r1", run["structured_output"]),
        format_configured=True,
    )
    expected = (
        "invalid_stored_format"
        if fmt is StructuredOutputFormat.INVALID
        else "internal_error"
    )
    assert fields.structured_output_status == "failed"
    assert fields.structured_output_reason["code"] == expected
