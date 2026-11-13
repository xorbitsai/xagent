"""The accept_partial switch from the structured outcome of a run to the fields
a task response carries."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from xagent.core.agent import structured_output as so
from xagent.core.agent import structured_schema as ss
from xagent.core.agent.result import COMPLETION_OUTCOMES
from xagent.core.model.chat.exceptions import LLMRetryableError
from xagent.web.models.task import TaskStatus
from xagent.web.services.structured_output_projection import (
    build_record,
    project_structured_output,
)

STRING: dict[str, Any] = {"type": "string"}
NULLABLE: dict[str, Any] = {"type": ["string", "null"]}
MARKER = "__NOT_EXPRESSIBLE__"
RUN_ID = "run-1"


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


REQUIRED = _obj({"a": STRING, "g": STRING})
ALL_NULLABLE = _obj({"a": NULLABLE, "b": NULLABLE})


def _fmt(schema: dict[str, Any], accept_partial: bool) -> Any:
    return so.StructuredOutputFormat(
        name="shape", description=None, schema=schema, accept_partial=accept_partial
    )


def _args(arguments: Any) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "submit_result",
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


class ScriptedModel:
    model_name = "fake-model"

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)

    async def chat(self, **kwargs: Any) -> Any:
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return item


async def _sleep_forever() -> Any:
    await asyncio.sleep(5)


_TEXT = {"type": "text", "content": "no tool call"}


class Outcome:
    """One way an extraction can end: the schema, the answer, the model script
    and what to patch."""

    def __init__(
        self,
        name: str,
        script: list[Any],
        schema: dict[str, Any] = REQUIRED,
        answer: str = "The answer.",
        patches: dict[tuple[Any, str], Any] | None = None,
    ) -> None:
        self.name = name
        self.script = script
        self.schema = schema
        self.answer = answer
        self.patches = patches or {}


OUTCOMES = [
    Outcome("ready", [_args({"a": "x", "g": "y"})]),
    Outcome("information-missing", [_args({"a": "x", "g": None})]),
    Outcome("value-not-expressible", [_args({"a": "x", "g": MARKER})]),
    Outcome("root-is-a-gap", [_args({"a": None, "g": None})]),
    Outcome(
        "dropped-for-size",
        [_args({"a": "x", "g": None})],
        patches={(ss, "MAX_RESULT_BYTES"): 1},
    ),
    Outcome("schema-mismatch", [_TEXT] * 3),
    Outcome("model-error", [LLMRetryableError("boom")]),
    Outcome(
        "extraction-timeout",
        [_sleep_forever],
        patches={(so, "EXTRACTION_TIMEOUT_SECONDS"): 0.05},
    ),
    Outcome(
        "result-too-large",
        [_args({"a": "x" * 40, "g": "y"})],
        patches={(so, "MAX_ARGUMENTS_BYTES"): 20},
    ),
    Outcome("internal-error", [_args({"a": "x", "g": "y"})]),
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
        for (module, name), value in outcome.patches.items():
            scoped.setattr(module, name, value)
        if outcome.name == "internal-error":

            async def boom(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("boom")

            scoped.setattr(so, "run_structured_output_step", boom)
        attached = await so._attach_structured_output(
            result,
            _fmt(outcome.schema, accept_partial),
            llm=ScriptedModel(copy.deepcopy(outcome.script)),
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


async def test_status_and_code_independent_of_switch(monkeypatch) -> None:
    assert len(OUTCOMES) == 12
    cells = 0
    for outcome in OUTCOMES:
        for rating in RATINGS:
            _, off = await _read_back(outcome, rating, False, monkeypatch)
            _, on = await _read_back(outcome, rating, True, monkeypatch)
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
    assert cells == 48
