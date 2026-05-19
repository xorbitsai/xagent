from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, TraceEventCallback


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


@pytest.mark.asyncio
async def test_trace_callback_success_emits_user_assistant_and_completion() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Write summary"

    await callback.on_run_start(runner=runner, context=context)
    await callback.on_run_end(
        runner=runner,
        context=context,
        result={"success": True, "execution_id": "exec-trace", "answer": "Done"},
    )

    assert [event["event_type"] for event in tracer.events] == [
        "task_start_message",
        "task_end_message",
        "task_end_general",
    ]
    assert tracer.events[1]["data"]["content"] == "Done"


@pytest.mark.asyncio
async def test_trace_callback_uses_display_user_message_when_present() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Read file\n\n## UPLOADED FILES\nfile_id=file-123"
    context.metadata["request_context"] = {
        "display_user_message": "Read file",
    }

    await callback.on_run_start(runner=runner, context=context)

    assert tracer.events[0]["data"]["message"] == "Read file"


@pytest.mark.asyncio
async def test_trace_callback_prefers_latest_message_display_metadata() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Original task"
    context.metadata["display_user_message"] = "Original task"
    context.add_user_message(
        "Read file\n\n## UPLOADED FILES\nfile_id=file-123",
        metadata={"display_message": "Read file"},
    )

    await callback.on_run_start(runner=runner, context=context)

    assert tracer.events[0]["data"]["message"] == "Read file"


@pytest.mark.asyncio
async def test_trace_callback_failed_run_emits_error() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")

    await callback.on_run_end(
        runner=runner,
        context=context,
        result={
            "success": False,
            "execution_id": "exec-trace",
            "error": "failed",
        },
    )

    assert tracer.events[0]["event_type"] == "task_error_general"
    assert tracer.events[0]["data"]["error_message"] == "failed"


@pytest.mark.asyncio
async def test_trace_callback_resume_does_not_duplicate_task_start() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Resume task"

    await callback.on_run_start(runner=runner, context=context, resume=True)
    await callback.on_run_start(
        runner=runner, context=context, checkpoint={"context": {}}
    )

    assert tracer.events == []


@pytest.mark.asyncio
async def test_trace_callback_no_tracer_is_noop() -> None:
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=None)
    context = ExecutionContext(execution_id="exec-trace")

    await callback.on_run_start(runner=runner, context=context)
    await callback.on_run_end(
        runner=runner,
        context=context,
        result={"success": True, "output": "Done"},
    )
