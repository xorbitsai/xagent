"""A tool call cut off at the model's output limit (#2850).

The fake models here answer with ``finish_reason="length"`` and a
``write_file`` call whose JSON arguments stop mid-string, the shape seen in
production when a long report was written in one call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from pydantic import BaseModel

from tests.core.agent.test_dag import _provider_error, build_plan
from xagent.core.agent import (
    DAGPattern,
    ExecutionContext,
    PatternRuntime,
    PlanStep,
    ReActPattern,
)
from xagent.core.agent.pattern.dag.failure_handoff import step_results_handoff
from xagent.core.agent.pattern.partial_delivery import request_partial_delivery
from xagent.core.agent.pattern.react.truncation import is_truncated_tool_call
from xagent.core.model.chat.basic.deepseek_tool_protocol import (
    normalize_deepseek_response,
)
from xagent.core.model.chat.tool_protocol import (
    TOOL_PROTOCOL_ERROR_KEY,
    ToolProtocolViolation,
    tool_protocol_error_response,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk

TRUNCATED_WRITE_ARGUMENTS = '{"file_path": "report.md", "content": "# Report\\n\\nLong'


class WriteFileArgs(BaseModel):
    file_path: str
    content: str


class FileTool:
    def __init__(self, name: str) -> None:
        self.calls: list[dict[str, Any]] = []

        class Metadata:
            pass

        Metadata.name = name  # type: ignore[attr-defined]
        Metadata.description = f"{name} in the workspace."  # type: ignore[attr-defined]
        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return WriteFileArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return {"success": True, "file_path": args["file_path"]}


class ChatLLM:
    """Answers each ``chat`` call with the next scripted response."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self, event_type: Any, *, data: dict[str, Any] | None = None, **_: Any
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "data": data or {},
            }
        )
        return str(len(self.events))

    def llm_ends(self) -> list[dict[str, Any]]:
        return [e["data"] for e in self.events if e["event_type"] == "action_end_llm"]


def truncated_write(arguments: str = TRUNCATED_WRITE_ARGUMENTS) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call_write",
                "type": "function",
                "function": {"name": "write_file", "arguments": arguments},
            }
        ],
        "finish_reason": "length",
    }


def tool_call(name: str, call_id: str = "call", **args: Any) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    }


def react_context() -> ExecutionContext:
    context = ExecutionContext(system_prompt="You are helpful.", execution_id="t-1")
    context.add_user_message("Write the full report to report.md.")
    return context


@pytest.mark.asyncio
async def test_react_steers_a_truncated_write_to_split_instead_of_resending() -> None:
    write_tool = FileTool("write_file")
    append_tool = FileTool("append_file")
    llm = ChatLLM(
        [
            truncated_write(),
            tool_call("write_file", file_path="report.md", content="# Report"),
            {"content": "Report written.", "done": True},
        ]
    )
    tracer = TraceRecorder()
    runtime = PatternRuntime(execution_id="t-1", tracer=tracer)

    result = await ReActPattern(max_iterations=5).run(
        context=react_context(),
        tools=[write_tool, append_tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    # The cut-off call never ran, not even with its arguments passed through.
    assert write_tool.calls == [{"file_path": "report.md", "content": "# Report"}]
    retry_prompt = llm.calls[1]["messages"][0]["content"]
    assert retry_prompt != llm.calls[0]["messages"][0]["content"]
    assert "output length limit" in retry_prompt
    assert "append_file" in retry_prompt
    phases = [e["data"].get("phase") for e in tracer.events]
    assert "truncated_tool_arguments_recovery" in phases


@pytest.mark.asyncio
async def test_react_fails_with_a_truncation_code_when_the_retry_is_cut_off_too() -> (
    None
):
    write_tool = FileTool("write_file")
    llm = ChatLLM([truncated_write(), truncated_write()])

    result = await ReActPattern(max_iterations=5).run(
        context=react_context(),
        tools=[write_tool],
        llm=llm,
        runtime=PatternRuntime(execution_id="t-1"),
    )

    assert result["success"] is False
    # The status stays the one failure classification already resumes on.
    assert result["status"] == "invalid_tool_protocol"
    assert result["protocol_code"] == "truncated_tool_arguments"
    assert result["finish_reason"] == "length"
    assert "output length limit" in result["error"]
    assert write_tool.calls == []
    assert len(llm.calls) == 2
    # Without an append tool the retry still asks for smaller calls.
    assert "smaller" in llm.calls[1]["messages"][0]["content"]


class DeepSeekTruncatedStreamLLM:
    """DeepSeek's stream adapter: a protocol-error chunk, then the length stop."""

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        raise AssertionError("the step call stays streaming")

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        if len(self.stream_calls) == 1:
            yield StreamChunk(
                type=ChunkType.PROTOCOL_ERROR,
                protocol_error={
                    "provider": "deepseek",
                    "code": "malformed_tool_arguments",
                    "message": "DeepSeek returned malformed arguments for 'write_file'.",
                    "details": {"repair_status": "skipped_incomplete"},
                },
            )
            yield StreamChunk(type=ChunkType.END, finish_reason="length")
            return
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "id": "call_final",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps({"answer": "Done in parts."}),
                    },
                }
            ],
        )
        yield StreamChunk(type=ChunkType.END, finish_reason="tool_calls")


@pytest.mark.asyncio
async def test_react_reads_a_deepseek_length_stop_as_truncation() -> None:
    llm = DeepSeekTruncatedStreamLLM()

    result = await ReActPattern(max_iterations=5).run(
        context=react_context(),
        tools=[FileTool("write_file"), FileTool("append_file")],
        llm=llm,
        runtime=PatternRuntime(execution_id="t-1"),
    )

    assert result["success"] is True
    assert "output length limit" in llm.stream_calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_react_gives_a_returned_malformed_arguments_error_its_own_repair() -> (
    None
):
    """Without a length stop, a returned error gets the malformed-JSON repair."""
    violation = ToolProtocolViolation(
        provider="deepseek",
        code="malformed_tool_arguments",
        message="DeepSeek returned malformed arguments for 'write_file'.",
    )
    llm = ChatLLM(
        [
            tool_protocol_error_response(violation),
            {"content": "Done.", "done": True},
        ]
    )

    result = await ReActPattern(max_iterations=5).run(
        context=react_context(),
        tools=[FileTool("write_file")],
        llm=llm,
        runtime=PatternRuntime(execution_id="t-1"),
    )

    assert result["success"] is True
    retry_prompt = llm.calls[1]["messages"][0]["content"]
    assert "malformed JSON arguments" in retry_prompt
    assert "output length limit" not in retry_prompt


@pytest.mark.parametrize(
    ("finish_reason", "truncated"),
    [("length", True), ("max_tokens", True), ("stop", False), (None, False)],
)
def test_truncation_needs_a_length_stop(finish_reason: Any, truncated: bool) -> None:
    response = truncated_write()
    response["finish_reason"] = finish_reason

    assert is_truncated_tool_call(response) is truncated


def test_protocol_error_envelope_keeps_the_finish_reason() -> None:
    response = normalize_deepseek_response(
        {
            "type": "tool_call",
            "tool_calls": truncated_write()["tool_calls"],
            "finish_reason": "length",
        },
        tools=[
            {
                "type": "function",
                "function": {"name": "write_file", "parameters": {"type": "object"}},
            }
        ],
    )

    assert response[TOOL_PROTOCOL_ERROR_KEY]["code"] == "malformed_tool_arguments"
    assert response["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_partial_delivery_trace_records_finish_reason_and_protocol_code() -> None:
    violation = ToolProtocolViolation(
        provider="deepseek", code="malformed_tool_arguments", message="cut off"
    )
    envelope = tool_protocol_error_response(violation, finish_reason="length")
    tracer = TraceRecorder()
    runtime = PatternRuntime(execution_id="t-1", tracer=tracer)
    schema = {
        "type": "function",
        "function": {
            "name": "final_answer",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    llm = ChatLLM([envelope])
    llm.supports_forced_tool_choice = False

    args = await request_partial_delivery(
        context=react_context(),
        llm=llm,
        runtime=runtime,
        messages=[{"role": "user", "content": "hand over"}],
        schema=schema,
        parse_response=lambda _: {"answer": "never parsed"},
        metadata={"phase": "dag_failure_delivery"},
    )

    assert args is None
    assert llm.calls[0]["tool_choice"] == "auto"
    (end,) = tracer.llm_ends()
    assert end["finish_reason"] == "length"
    assert end["protocol_code"] == "malformed_tool_arguments"
    assert end["success"] is False


def research_dag() -> tuple[DAGPattern, Any]:
    plan = build_plan(
        PlanStep(id="s1", task="Research market size"),
        PlanStep(id="s2", task="Research competitors"),
        PlanStep(id="s3", task="Write the report", dependencies=["s1", "s2"]),
    )
    return (
        DAGPattern(lambda **_: plan, react_max_iterations=5, max_concurrency=1),
        plan,
    )


def dag_context() -> ExecutionContext:
    context = ExecutionContext(execution_id="task-2850")
    context.add_user_message("Research the market and write a full report.")
    return context


@pytest.mark.asyncio
async def test_dag_hands_over_completed_results_when_delivery_returns_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pattern, plan = research_dag()
    write_tool = FileTool("write_file")
    empty_delivery = tool_protocol_error_response(
        ToolProtocolViolation(
            provider="deepseek", code="malformed_tool_arguments", message="cut off"
        ),
        finish_reason="length",
    )
    llm = ChatLLM(
        [
            tool_call("final_answer", answer="Market size is 4.2B USD."),
            tool_call("final_answer", answer="Competitors: Acme and Globex."),
            truncated_write(),
            truncated_write(),
            empty_delivery,
        ]
    )

    with caplog.at_level(logging.ERROR, logger="xagent.core.agent.pattern.dag"):
        result = await pattern.run(context=dag_context(), tools=[write_tool], llm=llm)

    assert [step.status for step in plan.steps] == ["completed", "completed", "failed"]
    assert result["success"] is True
    assert result["completion_outcome"] == "partial"
    assert result["termination_reason"] == "step_failed"
    assert result["failed_step_id"] == "s3"
    output = result["output"]
    assert "not a completed task" in output
    assert "Research market size" in output
    assert "Market size is 4.2B USD." in output
    assert "Research competitors" in output
    assert "Competitors: Acme and Globex." in output
    assert "- Write the report (failed)" in output
    # Raw error text stays out of the user-facing handoff.
    assert "invalid tool protocol" not in output
    assert "output length limit" not in output
    assert write_tool.calls == []

    (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    message = record.getMessage()
    assert "task_id=task-2850" in message
    assert "step_id=s3" in message
    assert "code=truncated_tool_arguments" in message
    assert "finish_reason=length" in message


@pytest.mark.asyncio
async def test_dag_keeps_the_bare_failure_when_no_step_completed() -> None:
    plan = build_plan(PlanStep(id="s1", task="Write the report"))
    pattern = DAGPattern(lambda **_: plan, react_max_iterations=5)
    llm = ChatLLM([truncated_write(), truncated_write()])

    result = await pattern.run(
        context=dag_context(), tools=[FileTool("write_file")], llm=llm
    )

    assert result["success"] is False
    assert result["failure_reason"] == "step_failed"
    assert "output" not in result


class ProviderDownAfterLLM(ChatLLM):
    """Answers its scripted responses, then fails every call on the provider."""

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        raise _provider_error()


@pytest.mark.asyncio
async def test_dag_keeps_the_provider_failure_instead_of_a_handoff() -> None:
    """A provider outage fails the delivery call too; the result must still
    carry ``model_error`` rather than become a completed handoff."""
    pattern, _plan = research_dag()
    llm = ProviderDownAfterLLM(
        [tool_call("final_answer", answer="Market size is 4.2B USD.")]
    )

    result = await pattern.run(context=dag_context(), tools=[], llm=llm)

    assert result["success"] is False
    assert result["failure_reason"] == "step_failed"
    assert result["model_error"]["kind"] == "access_denied"
    assert "output" not in result


def test_handoff_keeps_results_of_steps_a_replan_dropped() -> None:
    current = [
        PlanStep(id="n1", task="Draft the summary"),
        PlanStep(id="n2", task="Publish"),
    ]

    args = step_results_handoff(
        current, {"old1": "Earlier finding", "n1": "Summary draft"}, "n2"
    )

    assert args is not None
    answer = args["answer"]
    assert "### old1\n\nEarlier finding" in answer
    assert "### Draft the summary\n\nSummary draft" in answer
    assert "- Publish (failed)" in answer


def test_handoff_lists_results_in_plan_order_with_one_line_names() -> None:
    steps = [
        PlanStep(id="a", task="Collect\n  market data"),
        PlanStep(id="b", task="Collect pricing"),
        PlanStep(id="c", task="Write\nthe report"),
    ]

    # Parallel steps finish out of plan order; a replan dropped "old".
    args = step_results_handoff(
        steps, {"b": "Pricing", "old": "Earlier", "a": "Market"}, "c"
    )

    assert args is not None
    answer = args["answer"]
    assert "### Collect market data\n\nMarket" in answer
    assert answer.index("### Collect market data") < answer.index("### Collect pricing")
    assert answer.index("### Collect pricing") < answer.index("### old")
    assert "- Write the report (failed)" in answer
