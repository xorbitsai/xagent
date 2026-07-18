from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime
from xagent.core.agent.pattern.final_answer_stream import (
    ToolCallStringFieldStreamer,
    _JsonStringFieldReader,
)
from xagent.core.agent.runtime import (
    LLMCallInterrupted,
    prepare_llm_for_context,
    resolved_llm_metadata,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk


class SlowLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def chat(self, **_: Any) -> str:
        self.started.set()
        await asyncio.sleep(60)
        return "never"


@pytest.mark.asyncio
async def test_prepare_llm_for_context_uses_resolved_model_window(monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_COMPACT_THRESHOLD_RATIO", raising=False)

    class PreparedLLM:
        model_name = "deepseek/deepseek-v4-flash"
        context_window = 1_048_576

    class VirtualLLM:
        async def prepare_for_call(self, messages: list[dict[str, Any]]) -> Any:
            assert messages[-1]["content"] == "make a podcast"
            return PreparedLLM()

    context = ExecutionContext()
    prepared = await prepare_llm_for_context(
        llm=VirtualLLM(),
        messages=[{"role": "user", "content": "make a podcast"}],
        context=context,
    )

    assert isinstance(prepared, PreparedLLM)
    assert context.compact_config.threshold == 786_432
    assert resolved_llm_metadata(prepared) == {
        "selected_model": "deepseek/deepseek-v4-flash",
        "context_window": 1_048_576,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("context_window", [None, 128_000])
async def test_prepare_llm_for_context_preserves_plain_llm_threshold(
    context_window: int | None,
) -> None:
    class PlainLLM:
        pass

    llm = PlainLLM()
    if context_window is not None:
        llm.context_window = context_window
    context = ExecutionContext()
    context.compact_config.threshold = 12_345

    prepared = await prepare_llm_for_context(
        llm=llm,
        messages=[{"role": "user", "content": "continue"}],
        context=context,
    )

    assert prepared is llm
    assert context.compact_config.threshold == 12_345


class CancelledLLM:
    async def chat(self, **_: Any) -> str:
        raise asyncio.CancelledError


class StreamingLLM:
    async def stream_chat(self, **_: Any) -> Any:
        yield StreamChunk(type=ChunkType.TOKEN, delta="hello")
        yield StreamChunk(type=ChunkType.TOKEN, delta=" world")
        yield StreamChunk(type=ChunkType.END)


class StreamingLLMWithUsage:
    async def stream_chat(self, **_: Any) -> Any:
        yield StreamChunk(type=ChunkType.TOKEN, delta="hello")
        yield StreamChunk(
            type=ChunkType.USAGE,
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        )
        yield StreamChunk(type=ChunkType.END)


class EmptyStreamingLLM:
    async def chat(self, **_: Any) -> str:
        return "fallback answer"

    async def stream_chat(self, **_: Any) -> Any:
        yield StreamChunk(type=ChunkType.END)


class UsageOnlyStreamingLLM:
    async def chat(self, **_: Any) -> str:
        return "fallback answer"

    async def stream_chat(self, **_: Any) -> Any:
        yield StreamChunk(
            type=ChunkType.USAGE,
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        )
        yield StreamChunk(type=ChunkType.END)


class StreamingToolDeltaLLM:
    async def stream_chat(self, **_: Any) -> Any:
        for arguments in ['{"expression"', ':"2 + ', '2"}']:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-1",
                        "function": {
                            "name": "calculator",
                            "arguments": arguments,
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


class StreamingToolDeltaWithReasoningLLM:
    async def stream_chat(self, **_: Any) -> Any:
        yield StreamChunk(
            type=ChunkType.TOOL_CALL,
            tool_calls=[
                {
                    "index": 0,
                    "id": "call-1",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2 + 2"}',
                    },
                }
            ],
            raw={
                "reasoning_content": "",
                "_xagent_provider_state": {"provider": {"field": ""}},
            },
        )
        yield StreamChunk(
            type=ChunkType.END,
            raw={
                "reasoning_content": "",
                "_xagent_provider_state": {"provider": {"field": ""}},
            },
        )


class StreamingFinalAnswerToolDeltaLLM:
    async def stream_chat(self, **_: Any) -> Any:
        for arguments in ['{"action":"final_answer"', ',"answer":"Hi', ' there"}']:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-final",
                        "function": {
                            "name": "route",
                            "arguments": arguments,
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


class StreamingToolDeltaWithLeadingBraceLLM:
    async def stream_chat(self, **_: Any) -> Any:
        for arguments in ['{"answer":"', "{hi", '"}']:
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-1",
                        "function": {
                            "name": "final_answer",
                            "arguments": arguments,
                        },
                    }
                ],
            )
        yield StreamChunk(type=ChunkType.END)


class ErrorAfterTokenLLM:
    async def stream_chat(self, **_: Any) -> Any:
        yield StreamChunk(type=ChunkType.TOKEN, delta="partial")
        raise RuntimeError("provider disconnected")


class ErrorBeforePayloadLLM:
    def __init__(self) -> None:
        self.stream_kwargs: dict[str, Any] | None = None
        self.chat_calls = 0

    async def chat(self, **kwargs: Any) -> str:
        self.chat_calls += 1
        return "fallback answer"

    async def stream_chat(self, **kwargs: Any) -> Any:
        self.stream_kwargs = kwargs
        raise RuntimeError("peer closed connection")
        yield StreamChunk(type=ChunkType.END)


class ChatOnlyLLM:
    async def chat(self, **_: Any) -> str:
        return "complete answer"


class OutboundCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


class CheckpointTracer:
    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    async def checkpoint(self, **payload: Any) -> None:
        self.checkpoints.append(payload)

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "step_id": step_id,
                "data": data or {},
            }
        )


class TraceOnlyTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "data": data or {},
            }
        )


class FailingTraceOnlyTracer:
    async def trace_event(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("trace failed")


class PatternWithState:
    status = "running"

    def get_state(self) -> dict[str, Any]:
        return {"step": 1}


@pytest.mark.asyncio
async def test_runtime_interrupt_converts_active_llm_cancel() -> None:
    runtime = PatternRuntime()
    llm = SlowLLM()
    task = asyncio.create_task(runtime.run_llm_call(llm))

    await llm.started.wait()
    runtime.request_interrupt("stop now")

    with pytest.raises(LLMCallInterrupted, match="stop now"):
        await task


@pytest.mark.asyncio
async def test_should_interrupt_string_result_becomes_reason() -> None:
    # A checker returning a string interrupts AND supplies the reason (used by
    # the mid-run quota gate so the run surfaces *why* it was stopped).
    runtime = PatternRuntime(interrupt_checker=lambda: "Monthly quota reached")
    assert await runtime.should_interrupt() is True
    assert runtime.interrupt_reason == "Monthly quota reached"


@pytest.mark.asyncio
async def test_should_interrupt_falsey_checker_does_not_interrupt() -> None:
    runtime = PatternRuntime(interrupt_checker=lambda: None)
    assert await runtime.should_interrupt() is False
    assert runtime.interrupt_reason is None


@pytest.mark.asyncio
async def test_should_interrupt_empty_string_neither_interrupts_nor_sets_reason() -> (
    None
):
    # "" is falsey: it must not interrupt, and must not clobber interrupt_reason.
    runtime = PatternRuntime(interrupt_checker=lambda: "")
    runtime.interrupt_reason = "prior"
    assert await runtime.should_interrupt() is False
    assert runtime.interrupt_reason == "prior"


@pytest.mark.asyncio
async def test_runtime_preserves_non_interrupt_cancelled_error() -> None:
    runtime = PatternRuntime()

    with pytest.raises(asyncio.CancelledError):
        await runtime.run_llm_call(CancelledLLM())


@pytest.mark.asyncio
async def test_runtime_stream_final_answer_emits_ui_events() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-123", outbound_message_handler=outbound)

    result = await runtime.stream_final_answer(
        StreamingLLM(), messages=[{"role": "user", "content": "Say hi"}]
    )

    assert result == "hello world"
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[0]["task_id"] == "task-123"
    assert outbound.events[1]["delta"] == "hello"
    assert outbound.events[2]["delta"] == " world"
    assert outbound.events[3]["content"] == "hello world"
    assert len({event["message_id"] for event in outbound.events}) == 1
    assert (
        runtime.last_final_answer_stream_message_id == outbound.events[0]["message_id"]
    )


@pytest.mark.asyncio
async def test_runtime_send_message_includes_active_step_id() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-123", outbound_message_handler=outbound)
    runtime.active_react_step_id = "react-step-1"

    payload = await runtime.send_message(
        message="Still working",
        message_type="progress",
        expect_response=False,
    )

    assert payload["step_id"] == "react-step-1"
    assert payload["metadata"]["step_id"] == "react-step-1"
    assert outbound.events[0]["step_id"] == "react-step-1"


@pytest.mark.asyncio
async def test_runtime_send_message_metadata_step_id_takes_precedence() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-123", outbound_message_handler=outbound)
    runtime.active_react_step_id = "react-step-1"

    payload = await runtime.send_message(
        message="Still working",
        metadata={"step_id": "dag-step-1"},
    )

    assert payload["step_id"] == "dag-step-1"
    assert payload["metadata"]["step_id"] == "dag-step-1"
    assert outbound.events[0]["step_id"] == "dag-step-1"


@pytest.mark.asyncio
async def test_runtime_stream_final_answer_preserves_usage_metadata() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-123", outbound_message_handler=outbound)
    context = ExecutionContext(execution_id="task-123")

    result = await runtime.stream_final_answer(StreamingLLMWithUsage(), messages=[])
    await runtime.on_llm_end(context=context, response=result)

    assert result == {
        "content": "hello",
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        },
    }
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[-1]["content"] == "hello"
    usage = context.get_total_token_usage()
    assert usage == {"total": 10, "input": 7, "output": 3, "call_count": 1}


@pytest.mark.asyncio
async def test_runtime_stream_final_answer_falls_back_to_chat_without_events() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(outbound_message_handler=outbound)

    result = await runtime.stream_final_answer(ChatOnlyLLM(), messages=[])

    assert result == "complete answer"
    assert outbound.events == []


@pytest.mark.asyncio
async def test_runtime_stream_final_answer_emits_error_terminal_event() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-123", outbound_message_handler=outbound)

    with pytest.raises(RuntimeError, match="provider disconnected"):
        await runtime.stream_final_answer(ErrorAfterTokenLLM(), messages=[])

    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_error",
    ]
    assert outbound.events[1]["delta"] == "partial"
    assert outbound.events[2]["error"] == "provider disconnected"
    assert len({event["message_id"] for event in outbound.events}) == 1
    assert runtime.last_final_answer_stream_message_id is None


@pytest.mark.asyncio
async def test_runtime_streaming_llm_call_merges_tool_call_argument_deltas() -> None:
    runtime = PatternRuntime()

    result = await runtime.run_streaming_llm_call(
        StreamingToolDeltaLLM(),
        messages=[],
        tools=[],
    )

    assert result == {
        "content": "",
        "tool_calls": [
            {
                "index": 0,
                "id": "call-1",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"2 + 2"}',
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_runtime_streaming_llm_call_preserves_empty_reasoning_content() -> None:
    runtime = PatternRuntime()

    result = await runtime.run_streaming_llm_call(
        StreamingToolDeltaWithReasoningLLM(),
        messages=[],
        tools=[],
    )

    assert result["tool_calls"][0]["function"]["name"] == "calculator"
    assert result["reasoning_content"] == ""
    assert result["_xagent_provider_state"] == {"provider": {"field": ""}}


@pytest.mark.asyncio
async def test_runtime_streaming_llm_call_falls_back_when_stream_is_empty() -> None:
    runtime = PatternRuntime()

    result = await runtime.run_streaming_llm_call(EmptyStreamingLLM(), messages=[])

    assert result == "fallback answer"


@pytest.mark.asyncio
async def test_runtime_streaming_llm_call_falls_back_when_stream_has_only_usage() -> (
    None
):
    runtime = PatternRuntime()

    result = await runtime.run_streaming_llm_call(UsageOnlyStreamingLLM(), messages=[])

    assert result == "fallback answer"


@pytest.mark.asyncio
async def test_runtime_streaming_llm_call_does_not_fallback_when_stream_fails() -> None:
    runtime = PatternRuntime()
    llm = ErrorBeforePayloadLLM()

    with pytest.raises(RuntimeError, match="peer closed connection"):
        await runtime.run_streaming_llm_call(
            llm,
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "noop"}}],
        )

    assert llm.stream_kwargs == {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "noop"}}],
    }
    assert llm.chat_calls == 0


@pytest.mark.asyncio
async def test_runtime_streaming_llm_call_does_not_fallback_after_payload() -> None:
    runtime = PatternRuntime()

    with pytest.raises(RuntimeError, match="provider disconnected"):
        await runtime.run_streaming_llm_call(ErrorAfterTokenLLM(), messages=[])


@pytest.mark.asyncio
async def test_runtime_streaming_llm_call_preserves_leading_brace_delta() -> None:
    runtime = PatternRuntime()

    result = await runtime.run_streaming_llm_call(
        StreamingToolDeltaWithLeadingBraceLLM(),
        messages=[],
        tools=[],
    )

    assert result["tool_calls"][0]["function"]["arguments"] == '{"answer":"{hi"}'


@pytest.mark.asyncio
async def test_tool_call_string_field_streamer_reads_argument_deltas() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-123", outbound_message_handler=outbound)
    streamer = ToolCallStringFieldStreamer(
        runtime=runtime,
        tool_name="route",
        field_name="answer",
        guard_field="action",
        guard_value="final_answer",
    )

    result = await runtime.run_streaming_llm_call(
        StreamingFinalAnswerToolDeltaLLM(),
        messages=[],
        tools=[],
        on_chunk=streamer.handle_chunk,
    )
    await streamer.finish("Hi there")

    assert result["tool_calls"][0]["function"]["arguments"] == (
        '{"action":"final_answer","answer":"Hi there"}'
    )
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[1]["delta"] == "Hi"
    assert outbound.events[2]["delta"] == " there"


@pytest.mark.asyncio
async def test_tool_call_string_field_streamer_preserves_leading_brace_delta() -> None:
    outbound = OutboundCollector()
    runtime = PatternRuntime(execution_id="task-123", outbound_message_handler=outbound)
    streamer = ToolCallStringFieldStreamer(
        runtime=runtime,
        tool_name="final_answer",
        field_name="answer",
    )

    result = await runtime.run_streaming_llm_call(
        StreamingToolDeltaWithLeadingBraceLLM(),
        messages=[],
        tools=[],
        on_chunk=streamer.handle_chunk,
    )
    await streamer.finish("{hi")

    assert result["tool_calls"][0]["function"]["arguments"] == '{"answer":"{hi"}'
    assert [event["type"] for event in outbound.events] == [
        "final_answer_start",
        "final_answer_delta",
        "final_answer_end",
    ]
    assert outbound.events[1]["delta"] == "{hi"


def test_json_string_field_reader_handles_unicode_surrogate_pairs() -> None:
    fields = _JsonStringFieldReader('{"answer":"hello \\ud83d\\ude00"}').read(
        {"answer"}
    )

    assert fields["answer"].complete is True
    assert fields["answer"].value == f"hello {chr(0x1F600)}"


def test_json_string_field_reader_rejects_invalid_escape_sequences() -> None:
    fields = _JsonStringFieldReader('{"answer":"bad \\z escape"}').read({"answer"})

    assert fields["answer"].complete is False
    assert fields["answer"].value == "bad "


@pytest.mark.asyncio
async def test_runtime_checkpoint_prefers_checkpoint_api() -> None:
    tracer = CheckpointTracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="exec-runtime")
    context = ExecutionContext(execution_id="exec-runtime")

    payload = await runtime.checkpoint(
        "before_llm",
        context=context,
        pattern=PatternWithState(),
        status="running",
    )

    assert payload["label"] == "before_llm"
    assert tracer.checkpoints[0]["execution_id"] == "exec-runtime"
    assert tracer.checkpoints[0]["pattern_state"] == {"step": 1}


@pytest.mark.asyncio
async def test_runtime_checkpoint_trace_event_fallback_is_task_scoped() -> None:
    tracer = TraceOnlyTracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="exec-runtime")
    context = ExecutionContext(execution_id="exec-runtime")

    await runtime.checkpoint("fallback", context=context, pattern=PatternWithState())

    assert tracer.events[0]["event_type"] == "task_update_general"
    assert tracer.events[0]["task_id"] == "exec-runtime"
    assert tracer.events[0]["data"]["label"] == "fallback"


@pytest.mark.asyncio
async def test_runtime_trace_events_are_best_effort() -> None:
    runtime = PatternRuntime(
        tracer=FailingTraceOnlyTracer(), execution_id="exec-runtime"
    )

    await runtime.on_llm_start(context=ExecutionContext(), messages=[], tools=[])


@pytest.mark.asyncio
async def test_on_llm_start_emits_context_usage_fields() -> None:
    """The LLM-start event must carry context_tokens + context_threshold so the
    frontend usage gauge has data; the tokens come from the same estimate that
    drives compaction."""

    class CapturingTracer:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def trace_event(self, event_type: Any, **kwargs: Any) -> None:
            self.events.append(
                {
                    "type": getattr(event_type, "value", str(event_type)),
                    "data": kwargs.get("data") or {},
                }
            )

    tracer = CapturingTracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-1")
    ctx = ExecutionContext()
    ctx.compact_config.threshold = 96000
    ctx.add_message("user", "x" * 400)

    await runtime.on_llm_start(
        context=ctx, messages=[{"role": "user", "content": "x" * 400}]
    )

    usage = [e["data"] for e in tracer.events if "context_threshold" in e["data"]]
    assert usage, tracer.events
    assert usage[0]["context_threshold"] == 96000
    assert isinstance(usage[0]["context_tokens"], int)
    assert usage[0]["context_tokens"] > 0


@pytest.mark.asyncio
async def test_tool_invocation_counts_one_action_each() -> None:
    """Each tool invocation increments tool_calls at start time.

    Billing on invocation (not self-reported success) is intentional: success
    comes from the tool's own return value, and user-controlled MCP tools could
    otherwise dodge billing by wrapping real output in {"success": false}.
    """
    from xagent.core.model.chat.token_context import (
        TokenUsage,
        get_token_usage,
        set_token_usage,
    )

    set_token_usage(TokenUsage())
    runtime = PatternRuntime(execution_id="task-actions")

    await runtime.on_tool_start(tool_call={"name": "calc", "args": {}, "id": "t1"})
    await runtime.on_tool_start(tool_call={"name": "search", "args": {}, "id": "t2"})
    assert get_token_usage().tool_calls == 2

    # Even a tool that will report failure was still invoked → billed.
    await runtime.on_tool_end(
        tool_call={"name": "search", "id": "t2"},
        result={"success": False, "error": "boom"},
    )
    assert get_token_usage().tool_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_failure_code", "expected_failure_code"),
    [
        ("oauth_token_required", "oauth_token_required"),
        ("other_valid_code", None),
        (" oauth_token_required", None),
        ({"failure_code": "oauth_token_required"}, None),
    ],
)
async def test_on_tool_end_emits_only_allowlisted_top_level_failure_code(
    raw_failure_code,
    expected_failure_code,
) -> None:
    class CapturingTracer:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def trace_event(self, event_type: Any, **kwargs: Any) -> None:
            self.events.append(
                {
                    "type": getattr(event_type, "value", str(event_type)),
                    "data": kwargs.get("data") or {},
                }
            )

    tracer = CapturingTracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-failure-code")
    result = {
        "is_error": True,
        "error": "MCP server credentials are unavailable.",
        "failure_code": raw_failure_code,
    }

    await runtime.on_tool_end(
        tool_call={"name": "mcp_unavailable", "id": "call-1"},
        result=result,
    )

    assert [event["type"] for event in tracer.events] == ["action_error_tool"]
    event_data = tracer.events[0]["data"]
    assert event_data["result"] == result
    if expected_failure_code is None:
        assert "failure_code" not in event_data
    else:
        assert event_data["failure_code"] == expected_failure_code


@pytest.mark.asyncio
async def test_concurrent_tool_calls_all_count() -> None:
    """A concurrent batch (asyncio.gather) increments the shared counter once per tool.

    Covers the PR's claim that counting is safe when react runs tools
    concurrently via _run_concurrent_batch.
    """
    from xagent.core.model.chat.token_context import (
        TokenUsage,
        get_token_usage,
        set_token_usage,
    )

    set_token_usage(TokenUsage())
    runtime = PatternRuntime(execution_id="task-batch")

    await asyncio.gather(
        *[
            runtime.on_tool_start(
                tool_call={"name": f"t{i}", "args": {}, "id": f"t{i}"}
            )
            for i in range(8)
        ]
    )

    assert get_token_usage().tool_calls == 8


class StreamingLLMWithCachedUsage:
    async def stream_chat(self, **_: Any) -> Any:
        yield StreamChunk(type=ChunkType.TOKEN, delta="hello")
        yield StreamChunk(
            type=ChunkType.USAGE,
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        )
        yield StreamChunk(type=ChunkType.END)


@pytest.mark.asyncio
async def test_runtime_surfaces_cached_tokens_in_usage_and_trace() -> None:
    """Provider cache telemetry reaches the merged usage payload and the
    LLM end trace event as a normalized cached_input_tokens count."""
    events: list[dict[str, Any]] = []

    class _CaptureTracer:
        async def trace_event(
            self,
            event_type: Any,
            task_id: Any = None,
            step_id: Any = None,
            data: Any = None,
            parent_id: Any = None,
        ) -> str:
            events.append({"data": dict(data or {})})
            return "evt"

    outbound = OutboundCollector()
    runtime = PatternRuntime(
        execution_id="task-123",
        outbound_message_handler=outbound,
        tracer=_CaptureTracer(),
    )
    context = ExecutionContext(execution_id="task-123")

    result = await runtime.stream_final_answer(
        StreamingLLMWithCachedUsage(), messages=[]
    )
    assert result["usage"]["cached_input_tokens"] == 4

    await runtime.on_llm_end(context=context, response=result)
    assert events[-1]["data"]["cached_input_tokens"] == 4
    assert events[-1]["data"]["input_tokens"] == 7
