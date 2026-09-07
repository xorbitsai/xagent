from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime
from xagent.core.agent import runtime as runtime_module
from xagent.core.agent.context.execution import (
    COMPACT_SUMMARY_METADATA_KEY,
    COMPACT_WATERMARK_METADATA_KEY,
    TRANSCRIPT_WATERMARK_METADATA_KEY,
)
from xagent.core.agent.pattern.final_answer_stream import (
    ToolCallStringFieldStreamer,
    _JsonStringFieldReader,
)
from xagent.core.agent.runtime import (
    LLMCallInterrupted,
    ToolCallInterrupted,
    prepare_llm_for_context,
    resolved_llm_metadata,
)
from xagent.core.model.chat.types import (
    CONTENT_SOURCE_KEY,
    CONTENT_SOURCE_REASONING_FALLBACK,
    ChunkType,
    StreamChunk,
)
from xagent.core.task_runtime import PREFERRED_INPUT_MODALITIES_METADATA_KEY


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
        async def prepare_for_call(
            self,
            messages: list[dict[str, Any]],
            *,
            preferred_input_modalities: tuple[str, ...] = (),
        ) -> Any:
            assert messages[-1]["content"] == "make a podcast"
            assert preferred_input_modalities == ()
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
async def test_prepare_llm_for_context_passes_runtime_modality_preferences() -> None:
    captured: list[tuple[str, ...]] = []

    class PreparedLLM:
        context_window = 128_000

    class VirtualLLM:
        async def prepare_for_call(
            self,
            messages: list[dict[str, Any]],
            *,
            preferred_input_modalities: tuple[str, ...] = (),
        ) -> Any:
            captured.append(preferred_input_modalities)
            return PreparedLLM()

    context = ExecutionContext()
    context.metadata[PREFERRED_INPUT_MODALITIES_METADATA_KEY] = [
        "IMAGE",
        "image",
        "audio",
    ]

    await prepare_llm_for_context(
        llm=VirtualLLM(),
        messages=[{"role": "user", "content": "inspect the selected target"}],
        context=context,
    )

    assert captured == [("image", "audio")]


@pytest.mark.asyncio
async def test_prepare_llm_for_context_reads_runtime_metadata_once() -> None:
    class PreparedLLM:
        pass

    class VirtualLLM:
        async def prepare_for_call(
            self,
            messages: list[dict[str, Any]],
            *,
            preferred_input_modalities: tuple[str, ...] = (),
        ) -> Any:
            assert preferred_input_modalities == ("image",)
            return PreparedLLM()

    class Metadata(dict[str, list[str]]):
        pass

    class Context:
        metadata_reads = 0

        @property
        def metadata(self) -> Metadata:
            self.metadata_reads += 1
            return Metadata({PREFERRED_INPUT_MODALITIES_METADATA_KEY: ["image"]})

    context = Context()
    await prepare_llm_for_context(
        llm=VirtualLLM(),
        messages=[{"role": "user", "content": "inspect the selected target"}],
        context=context,
    )

    assert context.metadata_reads == 1


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
async def test_runtime_interrupt_cancels_active_tool_call() -> None:
    runtime = PatternRuntime()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_tool() -> str:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    task = asyncio.create_task(runtime.run_tool_call(slow_tool))
    await started.wait()
    runtime.request_interrupt("pause now")

    with pytest.raises(ToolCallInterrupted, match="pause now"):
        await task
    assert cancelled.is_set()
    assert not runtime._active_tool_tasks


@pytest.mark.asyncio
async def test_external_cancellation_cleans_up_active_tool_call() -> None:
    runtime = PatternRuntime()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_tool() -> str:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    outer_task = asyncio.create_task(runtime.run_tool_call(slow_tool))
    await started.wait()
    tool_task = next(iter(runtime._active_tool_tasks))

    outer_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer_task

    assert tool_task.cancelled()
    assert cancelled.is_set()
    assert not runtime._active_tool_tasks


@pytest.mark.asyncio
async def test_runtime_does_not_start_tool_after_interrupt() -> None:
    runtime = PatternRuntime()
    invoked = False

    async def tool() -> str:
        nonlocal invoked
        invoked = True
        return "never"

    runtime.request_interrupt("already paused")

    with pytest.raises(ToolCallInterrupted, match="already paused"):
        await runtime.run_tool_call(tool)
    assert invoked is False


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


class RecordingLogger:
    """Records info/warning calls without depending on the logging module's
    own configuration - per the project rule against log assertions that
    depend on caplog or other global logging state, this replaces the
    module-level ``logger`` binding directly."""

    def __init__(self) -> None:
        self.info_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.warning_calls: list[tuple[str, tuple[Any, ...]]] = []

    def info(self, msg: str, *args: Any) -> None:
        self.info_calls.append((msg, args))

    def warning(self, msg: str, *args: Any) -> None:
        self.warning_calls.append((msg, args))


_NEWLINE_AND_QUOTE_EXCEPTION_TEXT = (
    "Traceback (most recent call last):\n"
    '  File "provider.py", line 42, in call\n'
    "    raise ValueError(\"bad 'quoted' response\")\n"
    "ValueError: bad 'quoted' response"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,reason,expected_reason",
    [
        (
            "known_literal",
            "interrupted during LLM stream",
            "interrupted during LLM stream",
        ),
        (
            "tool_protocol_retry_template",
            "invalid some_provider_code tool protocol, retrying",
            "invalid tool protocol (code:18 chars), retrying",
        ),
        ("unparsed_provider_exception", "x" * 5000, "<unparsed>:5000"),
        (
            "unparsed_exception_with_newlines_and_quotes",
            _NEWLINE_AND_QUOTE_EXCEPTION_TEXT,
            f"<unparsed>:{len(_NEWLINE_AND_QUOTE_EXCEPTION_TEXT)}",
        ),
    ],
)
async def test_runtime_stream_close_log_records_reason_in_one_of_three_shapes(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    reason: str,
    expected_reason: str,
) -> None:
    """I-8: fail_final_answer_stream logs a normalized reason that is always
    one of three shapes - a known fixed string verbatim, the
    tool-protocol-retry template folded to a fixed shape that keeps only the
    provider error code's length, or <unparsed>:<length> for anything else -
    never the raw external text itself."""

    recording_logger = RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recording_logger)
    runtime = PatternRuntime(execution_id="task-1")
    runtime.last_final_answer_stream_message_id = "final_answer_abc"

    await runtime.fail_final_answer_stream("final_answer_abc", reason)

    assert len(recording_logger.warning_calls) == 1
    msg, args = recording_logger.warning_calls[0]
    logged = msg % args
    assert f"reason={expected_reason}" in logged
    if reason != expected_reason:
        assert reason[:40] not in logged


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "has_handler", [True, False], ids=["with_handler", "without_handler"]
)
async def test_runtime_stream_events_are_logged_once_each(
    monkeypatch: pytest.MonkeyPatch,
    has_handler: bool,
) -> None:
    """I-9: each of the three final_answer stream events logs exactly once
    per call, with the acting message_id present in the logged line - except
    start_final_answer_stream, which logs nothing at all (and returns None)
    when there is no outbound handler, because no stream was actually
    opened."""

    recording_logger = RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recording_logger)
    outbound = OutboundCollector() if has_handler else None
    runtime = PatternRuntime(execution_id="task-1", outbound_message_handler=outbound)

    message_id = await runtime.start_final_answer_stream()

    if not has_handler:
        assert message_id is None
        assert recording_logger.info_calls == []
        assert recording_logger.warning_calls == []
        return

    assert message_id is not None
    assert len(recording_logger.info_calls) == 1
    logged_start = recording_logger.info_calls[0][0] % recording_logger.info_calls[0][1]
    assert message_id in logged_start

    await runtime.end_final_answer_stream(message_id, "answer text")
    assert len(recording_logger.info_calls) == 2
    logged_end = recording_logger.info_calls[1][0] % recording_logger.info_calls[1][1]
    assert message_id in logged_end
    assert "content_chars=11" in logged_end
    # The answer text itself must never reach the log - only its length.
    assert "answer text" not in logged_end

    second_message_id = await runtime.start_final_answer_stream()
    assert second_message_id is not None
    assert len(recording_logger.info_calls) == 3

    await runtime.fail_final_answer_stream(
        second_message_id, "interrupted during LLM stream"
    )
    assert len(recording_logger.warning_calls) == 1
    logged_fail = (
        recording_logger.warning_calls[0][0] % recording_logger.warning_calls[0][1]
    )
    assert second_message_id in logged_fail
    assert "reason=interrupted during LLM stream" in logged_fail


@pytest.mark.asyncio
async def test_runtime_stream_log_suppressed_when_outbound_handler_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opened/closed/failed log lines sit after the outbound emit call in
    all three stream methods, not before it - so when the outbound handler
    itself raises, the exception propagates to the caller and the
    corresponding log line is never written, because the emit it was meant
    to describe never actually succeeded."""

    async def raising_handler(payload: dict[str, Any]) -> None:
        raise RuntimeError("outbound send failed")

    recording_logger = RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recording_logger)
    runtime = PatternRuntime(
        execution_id="task-1", outbound_message_handler=raising_handler
    )

    with pytest.raises(RuntimeError, match="outbound send failed"):
        await runtime.start_final_answer_stream()
    assert recording_logger.info_calls == []

    with pytest.raises(RuntimeError, match="outbound send failed"):
        await runtime.end_final_answer_stream("final_answer_abc", "answer text")
    assert recording_logger.info_calls == []

    with pytest.raises(RuntimeError, match="outbound send failed"):
        await runtime.fail_final_answer_stream("final_answer_abc", "some reason")
    assert recording_logger.warning_calls == []


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
async def test_on_tool_cancelled_closes_trace_without_error_event() -> None:
    tracer = CheckpointTracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-cancelled-tool")

    await runtime.on_tool_cancelled(
        tool_call={"name": "understand_images", "args": {}, "id": "vision-1"},
        reason="paused by user",
    )

    assert [event["event_type"] for event in tracer.events] == ["action_end_tool"]
    assert tracer.events[0]["data"] == {
        "tool_name": "understand_images",
        "tool_params": {},
        "tool_call_id": "vision-1",
        "success": False,
        "interrupted": True,
        "interrupt_reason": "paused by user",
    }


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
    ctx.add_message("user", "persisted-only")

    messages = [{"role": "user", "content": "x" * 400}]
    tools = [{"function": {"name": "save", "description": "d" * 400}}]
    await runtime.on_llm_start(context=ctx, messages=messages, tools=tools)

    usage = [e["data"] for e in tracer.events if "context_threshold" in e["data"]]
    assert usage, tracer.events
    assert usage[0]["context_threshold"] == 96000
    assert isinstance(usage[0]["context_tokens"], int)
    assert usage[0]["context_tokens"] == ctx.estimate_context_tokens(messages, tools)


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
async def test_on_tool_end_marks_classified_nested_wait_as_failed() -> None:
    """A classified nested-wait failure must route through on_tool_error.

    The ReAct loop records ``status="failed"`` for this tool call and
    continues (it does not stop the parent run — see
    ``react.py:2787-2803``). This test covers the runtime half only: that a
    classified failure dict, not an exception, is enough to route through
    ``on_tool_error`` and carry ``failure_code`` into the emitted trace
    event.
    """

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
    runtime = PatternRuntime(tracer=tracer, execution_id="task-nested-wait")
    result = {
        "success": False,
        "is_error": True,
        "status": "error",
        "failure_code": "unsupported_nested_interaction",
        "error": "Nested agent calls cannot forward interactive prompts.",
        "output": "Nested agent calls cannot forward interactive prompts.",
        "response": "Nested agent calls cannot forward interactive prompts.",
    }

    await runtime.on_tool_end(
        tool_call={"name": "delegated_agent", "id": "call-nested"},
        result=result,
    )

    assert [event["type"] for event in tracer.events] == ["action_error_tool"]
    event_data = tracer.events[0]["data"]
    assert event_data["failure_code"] == "unsupported_nested_interaction"
    assert event_data["result"] == result
    assert runtime._tool_result_success(result) is False


@pytest.mark.asyncio
async def test_on_tool_end_carries_missing_output_failure_code() -> None:
    """The missing-delegated-output code must propagate into trace data too.

    Same runtime-level contract as the nested-wait case: a classified failure
    dict is enough, no exception required, to carry a new allowlisted
    failure_code into the emitted trace event.
    """

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
    runtime = PatternRuntime(tracer=tracer, execution_id="task-missing-output")
    result = {
        "success": False,
        "is_error": True,
        "status": "error",
        "failure_code": "missing_delegated_output",
        "error": "The delegated agent reported completion without returning "
        "any usable output, so there is no answer from it to use.",
        "output": "The delegated agent reported completion without returning "
        "any usable output, so there is no answer from it to use.",
        "response": "The delegated agent reported completion without "
        "returning any usable output, so there is no answer from it to use.",
    }

    await runtime.on_tool_end(
        tool_call={"name": "delegated_agent", "id": "call-missing-output"},
        result=result,
    )

    assert [event["type"] for event in tracer.events] == ["action_error_tool"]
    event_data = tracer.events[0]["data"]
    assert event_data["failure_code"] == "missing_delegated_output"


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


class RaisingCompactLLM:
    """Stub whose ``chat`` always fails, driving the llm_summary -> truncate
    fallback in ``PatternRuntime.compact_context_if_needed``."""

    model_name = "raising-compact-llm"

    async def chat(self, **_: Any) -> Any:
        raise RuntimeError("compact llm exploded")


@pytest.mark.asyncio
async def test_compact_context_if_needed_falls_back_to_truncate_without_orphans() -> (
    None
):
    """When the LLM-summary compaction path raises, the runtime must fall back
    to ``truncate`` -- and the fallback must never leave a native ``tool``
    message without the assistant message that declared its ``tool_calls``
    immediately before it, since providers reject that shape outright.
    """
    runtime = PatternRuntime(execution_id="task-compact-fallback")
    ctx = ExecutionContext()
    ctx.compact_config.threshold = 1
    # 9 messages total (u0, assistant+2 tool results, tail-0..tail-4). A
    # max_messages of 4 would land the naive tail slice entirely inside the
    # 5 trailing user messages (tail-1..tail-4) -- no "tool" message would
    # ever survive, making assert_no_orphan_tool_messages below vacuous: it
    # would pass identically even if `_tail_window_preserving_tool_pairs`
    # were replaced by a naive `messages[-keep_count:]`. max_messages=6 puts
    # the boundary inside the assistant+tool block instead, so the
    # pair-preservation walk-back must expand the window back to include
    # both tool results and their declaring assistant message, giving the
    # assertions below something real to check.
    ctx.compact_config.max_messages = 6
    ctx.add_user_message("u0")
    ctx.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "type": "function", "function": {"name": "read_file"}},
            {"id": "call-2", "type": "function", "function": {"name": "write_file"}},
        ],
    )
    ctx.add_tool_result("read_file", {"output": "read"}, tool_call_id="call-1")
    ctx.add_tool_result("write_file", {"output": "written"}, tool_call_id="call-2")
    for i in range(5):
        ctx.add_user_message(f"tail-{i}")

    result = await runtime.compact_context_if_needed(
        context=ctx, llm=RaisingCompactLLM()
    )

    assert result is not None
    assert result.compacted is True
    assert result.strategy == "truncate"
    assert result.metadata["fallback_strategy"] == "truncate"
    assert "llm_compact_error" in result.metadata

    # This is the load-bearing check for this test: with a naive
    # `messages[-keep_count:]` slice (keep_count=6 here), the survivors would
    # be tail-1..tail-4 plus the two dangling tool results with no preceding
    # assistant message -- 0 messages would have role "assistant" among the
    # last 6. The real pair-preserving walk-back must expand the window left
    # to include the assistant message that declared both tool_calls.
    final_roles = [message.role for message in ctx.messages]
    assert final_roles.count("tool") == 2, (
        "expected both tool results to survive the pair-preserving "
        "walk-back -- if this is 0, `_tail_window_preserving_tool_pairs` "
        "has regressed to a naive `messages[-keep_count:]` slice"
    )
    assert final_roles[:3] == ["assistant", "tool", "tool"]

    def assert_no_orphan_tool_messages(messages: list[dict[str, Any]]) -> None:
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            assert index > 0, "tool message with no preceding message"
            # A parallel tool-call batch is one assistant(tool_calls) message
            # followed by a contiguous run of "tool" messages -- so a tool
            # message's own predecessor can itself be another "tool" message
            # from the same batch. Walk back over that run to find the
            # declaring assistant message.
            declaring_index = index - 1
            while (
                declaring_index >= 0 and messages[declaring_index].get("role") == "tool"
            ):
                declaring_index -= 1
            assert declaring_index >= 0, "tool message with no preceding message"
            declaring = messages[declaring_index]
            assert declaring.get("role") == "assistant" and declaring.get(
                "tool_calls"
            ), "tool message not immediately preceded by its assistant tool_calls"
            declared_ids = {
                str(tool_call.get("id"))
                for tool_call in declaring["tool_calls"]
                if isinstance(tool_call, dict) and tool_call.get("id")
            }
            assert message.get("tool_call_id") in declared_ids

    assert_no_orphan_tool_messages(ctx.get_messages_for_llm())
    assert_no_orphan_tool_messages(
        [
            {
                "role": message.role,
                "tool_calls": message.tool_calls,
                "tool_call_id": message.tool_call_id,
            }
            for message in ctx.messages
        ]
    )


@pytest.mark.asyncio
async def test_compaction_records_that_an_unusable_summary_was_discarded() -> None:
    """A summary that was requested and then discarded must leave a trace.

    The error path already records ``llm_compact_error``. Without an
    equivalent here, a summary that came back empty -- or that the client
    marked as a substituted reasoning trace -- is indistinguishable in the
    trace from compaction that never attempted to summarize, which is exactly
    the case an operator would want to see.
    """

    class CompactTracer:
        """Accepts the full trace_event signature, including ``step_id``,
        which the compaction events carry."""

        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def trace_event(
            self, event_type: Any, *, data: dict[str, Any] | None = None, **_: Any
        ) -> None:
            self.events.append(
                {
                    "event_type": getattr(event_type, "value", str(event_type)),
                    "data": data or {},
                }
            )

    tracer = CompactTracer()
    runtime = PatternRuntime(tracer=tracer)
    context = ExecutionContext(execution_id="unusable-summary")
    context.compact_config.threshold = 1
    context.add_user_message("current request")
    context.add_tool_result("read_file", {"output": "x" * 200}, tool_call_id="call-1")

    class EmptySummaryLLM:
        model_name = "compact-test"

        async def chat(self, **_: Any) -> Any:
            return {"content": ""}

    result = await runtime.compact_context_if_needed(
        context=context,
        llm=EmptySummaryLLM(),
        metadata={"phase": "test"},
    )

    assert result.compacted
    # Fell back to dropping messages, and said so.
    assert result.strategy == "truncate"
    assert result.metadata["llm_summary_unusable"] is True
    assert result.metadata["fallback_strategy"] == "truncate"
    # The emitted compact event carries it too, so this is visible without
    # reading the return value.
    assert any(
        event["data"].get("llm_summary_unusable") is True for event in tracer.events
    )


@pytest.mark.asyncio
async def test_compaction_retries_with_a_smaller_budget_before_truncating() -> None:
    """The requested budget is a guess; a wrong guess must not cost the summary.

    It follows the model's *input* window, while providers cap the *output*
    separately and much lower. Nothing in the model config records that limit
    -- the one number stored per model is a default to send, not a ceiling the
    model can produce -- so a model whose output cap sits below the requested
    budget would previously have lost its summary entirely and fallen back to
    dropping messages, the outcome this path exists to avoid.
    """
    context = ExecutionContext(execution_id="budget-retry")
    context.compact_config.threshold = 32000
    context.add_user_message("current request")
    context.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}}
        ],
    )
    context.add_tool_result(
        "read_file", {"output": "x" * 200_000}, tool_call_id="call-1"
    )

    class OutputCappedLLM:
        model_name = "compact-test"

        def __init__(self) -> None:
            self.budgets: list[int] = []

        async def chat(self, **kwargs: Any) -> Any:
            budget = kwargs["max_tokens"]
            self.budgets.append(budget)
            if budget > 4096:
                raise RuntimeError("max_tokens is too large for this model")
            return {"content": "summary within the model's output cap"}

    llm = OutputCappedLLM()
    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=llm, metadata={"phase": "test"}
    )

    # Asked big, was refused, stepped down, succeeded -- and summarized rather
    # than dropping messages. The step lands on the largest rung the cap
    # allows, not the smallest one available.
    assert llm.budgets == [8000, 4096]
    assert result.compacted
    assert result.strategy == "llm_summary"
    assert "output cap" in context.messages[0].content


@pytest.mark.asyncio
async def test_compaction_ladder_skips_a_budget_the_model_cannot_use() -> None:
    """The step down must not go straight to the floor.

    A reasoning model draws its reasoning from the same allowance, so a
    budget that is merely *accepted* can still produce no summary -- the
    client marks such a response as a substituted reasoning trace and
    compaction rejects it. 1024 is on the ladder because it was the ceiling
    before this PR raised it, and is therefore known to leave room for an
    answer; jumping from 8000 to 256 would spend a request to arrive at the
    same truncation.
    """
    context = ExecutionContext(execution_id="budget-ladder")
    context.compact_config.threshold = 32000
    context.add_user_message("current request")
    context.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}}
        ],
    )
    context.add_tool_result(
        "read_file", {"output": "x" * 200_000}, tool_call_id="call-1"
    )

    class CappedReasoningLLM:
        model_name = "compact-test"

        def __init__(self) -> None:
            self.budgets: list[int] = []

        async def chat(self, **kwargs: Any) -> Any:
            budget = kwargs["max_tokens"]
            self.budgets.append(budget)
            if budget > 4096:
                raise RuntimeError("max_tokens is too large for this model")
            if budget < 1024:
                # Whole allowance spent reasoning; the client surfaces the
                # trace in place of content and marks it.
                return {
                    "content": "thinking about the file",
                    CONTENT_SOURCE_KEY: CONTENT_SOURCE_REASONING_FALLBACK,
                    "reasoning_content": "thinking about the file",
                }
            return {"content": "a real summary of the prior work"}

    llm = CappedReasoningLLM()
    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=llm, metadata={"phase": "test"}
    )

    assert llm.budgets == [8000, 4096]
    assert result.compacted
    assert result.strategy == "llm_summary"
    assert "a real summary" in context.messages[0].content


@pytest.mark.asyncio
async def test_compaction_stops_descending_once_a_budget_is_accepted() -> None:
    """Accepted-but-unusable ends the ladder rather than continuing down.

    A response is unusable because the allowance was too small for the model
    to finish, so usability is monotone in the budget: every smaller rung is
    unusable too. Continuing to step down would spend requests to arrive at
    the same truncation. Here the model accepts anything but can never
    produce a summary, so the ladder must stop at the first accepted rung.
    """
    context = ExecutionContext(execution_id="budget-monotone")
    context.compact_config.threshold = 32000
    context.add_user_message("current request")
    context.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}}
        ],
    )
    context.add_tool_result(
        "read_file", {"output": "x" * 200_000}, tool_call_id="call-1"
    )

    class AlwaysReasoningLLM:
        model_name = "compact-test"

        def __init__(self) -> None:
            self.budgets: list[int] = []

        async def chat(self, **kwargs: Any) -> Any:
            self.budgets.append(kwargs["max_tokens"])
            return {
                "content": "still thinking",
                CONTENT_SOURCE_KEY: CONTENT_SOURCE_REASONING_FALLBACK,
                "reasoning_content": "still thinking",
            }

    llm = AlwaysReasoningLLM()
    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=llm, metadata={"phase": "test"}
    )

    # One request, not one per rung.
    assert llm.budgets == [8000]
    assert result.strategy == "truncate"
    assert result.metadata["llm_summary_unusable"] is True


@pytest.mark.asyncio
async def test_compaction_marks_messages_dropped_for_want_of_a_summary_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping messages because no summary model was reachable used to look
    exactly like dropping them after a summary failed -- both produced a bare
    ``truncate`` result, and only the failure path left a marker behind. That
    made "nothing could summarize" invisible in the trace.
    """
    recording_logger = RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recording_logger)
    context = ExecutionContext(execution_id="no-compact-model")
    context.compact_config.threshold = 1
    context.compact_config.max_messages = 2
    for index in range(6):
        context.add_user_message(f"m{index}")

    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=None, metadata={"phase": "test"}
    )

    assert result.compacted is True
    assert result.strategy == "truncate"
    # Without this the test cannot tell a real drop from the no-op window that
    # keeps every message, which is exactly what the marker must not claim.
    assert result.metadata["removed_count"] == 4
    assert result.metadata["llm_summary_unavailable"] is True
    assert result.metadata["fallback_strategy"] == "truncate"
    assert len(recording_logger.warning_calls) == 1
    msg, args = recording_logger.warning_calls[0]
    assert "no reachable summary path" in msg % args


@pytest.mark.asyncio
async def test_compaction_does_not_claim_a_drop_when_the_window_kept_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context can be over budget while holding fewer messages than the tail
    window keeps -- one huge tool result does it. ``_drop_oldest_messages``
    still reports ``compacted=True`` there, so keying the marker off that flag
    announced a drop that never happened, once per turn, forever.
    """
    recording_logger = RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recording_logger)
    context = ExecutionContext(execution_id="nothing-to-drop")
    context.compact_config.threshold = 1
    context.compact_config.max_messages = 20
    context.add_user_message("only message")

    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=None, metadata={"phase": "test"}
    )

    assert result.metadata["removed_count"] == 0
    assert "llm_summary_unavailable" not in result.metadata
    assert recording_logger.warning_calls == []


@pytest.mark.asyncio
async def test_compaction_does_not_blame_the_model_when_nothing_is_summarizable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A usable summary model plus a context that declines to build a request
    (over budget, but every message hidden) is not the same failure as having
    no summarizer at all, and must not be reported as one.

    This is the only test that reaches the line clearing
    ``summary_unavailable_metadata``: delete that line and the marker below
    reappears.
    """
    recording_logger = RecordingLogger()
    monkeypatch.setattr(runtime_module, "logger", recording_logger)

    class UnusedCompactLLM:
        model_name = "compact-test"

        async def chat(self, **_: Any) -> Any:  # pragma: no cover - never called
            raise AssertionError("no request should have been built")

    context = ExecutionContext(execution_id="nothing-summarizable")
    context.compact_config.threshold = 1
    context.compact_config.max_messages = 2
    for index in range(6):
        context.add_user_message(f"m{index}", hidden=True)

    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=UnusedCompactLLM(), metadata={"phase": "test"}
    )

    assert result.compacted is True
    assert result.metadata["removed_count"] == 4
    assert "llm_summary_unavailable" not in result.metadata
    assert recording_logger.warning_calls == []


class _SummarizingLLM:
    model_name = "compact-test"

    async def chat(self, **_: Any) -> Any:
        return {"content": "what happened earlier"}


def _oversized_context(execution_id: str) -> ExecutionContext:
    context = ExecutionContext(execution_id=execution_id)
    context.compact_config.threshold = 32000
    context.add_user_message("current request")
    context.add_assistant_message(
        "",
        tool_calls=[
            {
                "id": "call-1",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    )
    context.add_tool_result(
        "read_file", {"output": "x" * 200_000}, tool_call_id="call-1"
    )
    return context


@pytest.mark.asyncio
async def test_compaction_publishes_the_summary_and_its_watermark() -> None:
    """The summary has to leave the turn on the compact event.

    The in-memory context holding it is rebuilt from scratch next turn, and
    the checkpoint that also holds it is pruned within this one, so without
    this the work is paid for and then discarded every turn.
    """
    context = _oversized_context("watermark-publish")
    context.metadata[TRANSCRIPT_WATERMARK_METADATA_KEY] = 42

    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=_SummarizingLLM(), metadata={"phase": "test"}
    )

    assert result.strategy == "llm_summary"
    assert result.metadata[COMPACT_WATERMARK_METADATA_KEY] == 42
    # Byte-identical to the system message this turn actually ran on, so a
    # replay reproduces the context rather than an approximation of it.
    assert result.metadata[COMPACT_SUMMARY_METADATA_KEY] == context.messages[0].content


@pytest.mark.asyncio
async def test_compaction_omits_the_watermark_when_the_caller_issued_none() -> None:
    """A reader must be able to tell "covers rows up to N" from "cannot be
    positioned at all". Storing None would collapse those into one value."""
    context = _oversized_context("watermark-absent")

    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=_SummarizingLLM(), metadata={"phase": "test"}
    )

    assert result.strategy == "llm_summary"
    assert COMPACT_WATERMARK_METADATA_KEY not in result.metadata
    assert COMPACT_SUMMARY_METADATA_KEY in result.metadata


@pytest.mark.asyncio
async def test_dropping_messages_publishes_no_summary_to_replay() -> None:
    """The backstop stands in for nothing. Emitting a summary key here would
    let a later turn skip stored rows on the strength of a compaction that
    never wrote one."""
    context = ExecutionContext(execution_id="watermark-backstop")
    context.compact_config.threshold = 1
    context.compact_config.max_messages = 2
    context.metadata[TRANSCRIPT_WATERMARK_METADATA_KEY] = 42
    for index in range(6):
        context.add_user_message(f"m{index}")

    result = await PatternRuntime().compact_context_if_needed(
        context=context, llm=None, metadata={"phase": "test"}
    )

    assert result.strategy == "truncate"
    assert COMPACT_SUMMARY_METADATA_KEY not in result.metadata
    assert COMPACT_WATERMARK_METADATA_KEY not in result.metadata
