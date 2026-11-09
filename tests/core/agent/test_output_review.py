import asyncio
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent import Agent, ExecutionContext, PatternRuntime
from xagent.core.agent.execution_adapter import (
    AgentExecutionAdapter,
    AgentExecutionConfig,
)
from xagent.core.agent.pattern import ReActPattern
from xagent.core.agent.trace import ACTION_START_TOOL, Tracer
from xagent.core.guardrails import output
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.user_context import current_user_id


class Reviewer:
    def open_stream(self):
        return None

    async def check(self, text, *, structured=False):
        if "private-value" in text:
            return output.OutputDecision("block", output.BLOCKED_OUTPUT)
        return output.OutputDecision("allow", text)


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(output, "_resolver", None)


@pytest.mark.asyncio
async def test_standalone_has_no_policy_and_nested_runs_inherit():
    assert await output.resolve_output_guardrail("1") is None
    reviewer = Reviewer()
    with output.bind_output_guardrail(reviewer):
        assert await output.resolve_output_guardrail("child") is reviewer
        output.set_output_review_resolver(AsyncMock(return_value=None))
        assert await output.resolve_output_guardrail("child") is reviewer


@pytest.mark.asyncio
async def test_host_review_covers_stream_and_returned_result():
    resolver = AsyncMock(return_value=Reviewer())
    output.set_output_review_resolver(resolver)
    outbound = AsyncMock()

    class LLM:
        async def stream_chat(self, **kwargs):
            yield StreamChunk(
                type=ChunkType.TOOL_CALL,
                tool_calls=[
                    {
                        "id": "final",
                        "function": {
                            "name": "final_answer",
                            "arguments": '{"answer":"private-value"}',
                        },
                    }
                ],
            )
            yield StreamChunk(type=ChunkType.END)

    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="test",
            pattern="react",
            llm=LLM(),
            workspace_enabled=False,
            outbound_message_handler=outbound,
        )
    )
    result = await adapter.execute(task="Test", task_id="1")
    assert result["output"] == output.BLOCKED_OUTPUT
    events = [call.args[0] for call in outbound.await_args_list]
    assert [event["type"] for event in events] == [
        "final_answer_start",
        "final_answer_end",
    ]
    assert events[-1]["content"] == result["output"]
    assert resolver.await_args.args[0].execution_id == "1"


@pytest.mark.asyncio
async def test_question_cannot_park_after_rejection():
    outbound = AsyncMock()
    runtime = PatternRuntime(
        output_guardrail=Reviewer(), outbound_message_handler=outbound
    )
    with pytest.raises(output.ProtectedInteractionError):
        await runtime.send_message(
            message="Choose",
            expect_response=True,
            metadata={"options": ["private-value"]},
        )
    outbound.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_unavailable_review_closes_stream_and_rejects_questions(
    monkeypatch, timeout
):
    from xagent.core.agent import runtime as runtime_module

    class UnavailableReviewer:
        def open_stream(self):
            return self

        async def feed(self, delta):
            return await self.check(delta)

        async def check(self, text, *, structured=False):
            if timeout:
                await asyncio.Event().wait()
            raise RuntimeError(text)

    monkeypatch.setattr(runtime_module, "OUTPUT_REVIEW_TIMEOUT_SECONDS", 0.01)
    events = []
    runtime = PatternRuntime(
        output_guardrail=UnavailableReviewer(), outbound_message_handler=events.append
    )
    message_id = await runtime.start_final_answer_stream()
    assert message_id is not None
    await runtime.emit_final_answer_delta(message_id, "private-value")
    await runtime.end_final_answer_stream(message_id, "private-value")
    assert (
        await runtime.prepare_final_answer("private-value") == output.UNAVAILABLE_OUTPUT
    )
    assert [event["type"] for event in events] == [
        "final_answer_start",
        "final_answer_error",
    ]
    assert events[-1]["error"] == output.UNAVAILABLE_OUTPUT
    with pytest.raises(output.ProtectedInteractionError) as exc:
        await runtime.send_message(message="private-value", expect_response=True)
    assert str(exc.value) == output.UNAVAILABLE_OUTPUT


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
async def test_rejected_question_checkpoint_allows_followup_to_replan(restored):
    class LLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "tool_calls": [
                        {
                            "id": "question",
                            "function": {
                                "name": "send_message",
                                "arguments": '{"message":"private-value","message_type":"question","expect_response":true}',
                            },
                        }
                    ]
                }
            return {"content": "safe follow-up", "done": True}

    llm = LLM()
    runner = Agent("reviewed", patterns=[ReActPattern()], llm=llm).get_runner(
        workspace_enabled=False
    )
    runtime = PatternRuntime(output_guardrail=Reviewer())
    if restored:
        unprotected = PatternRuntime()
        first = await runner.run("Start", runtime=unprotected)
        assert first["status"] == "waiting_for_user"
        rejected = await runner.run(
            None, runtime=runtime, checkpoint=unprotected.last_checkpoint
        )
    else:
        rejected = await runner.run("Start", runtime=runtime)
    assert rejected["status"] == "failed"
    assert rejected["error"] == output.BLOCKED_OUTPUT
    assert not runtime.outbound_messages
    assert llm.calls == 1
    checkpoint = runtime.last_checkpoint
    assert checkpoint["status"] == "failed"
    assert checkpoint["pattern_state"]["pending_tool_calls"] == []
    assert checkpoint["pattern_state"]["waiting_for_user_request"] is None

    context = ExecutionContext.from_dict(checkpoint["context"])
    context.add_user_message("Try again")
    followup = await runner.run(
        None,
        runtime=PatternRuntime(output_guardrail=Reviewer()),
        checkpoint={**checkpoint, "context": context.to_dict()},
    )
    assert followup["success"] is True
    assert followup["output"] == "safe follow-up"
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_runner_rejects_unreviewed_waiting_results_from_patterns():
    context = ExecutionContext()
    runtime = PatternRuntime(output_guardrail=Reviewer())
    runner = Agent("reviewed", patterns=[]).get_runner(workspace_enabled=False)
    result = {
        "success": False,
        "status": "waiting_for_user",
        "message": "private-value",
        "interactions": [{"label": "private-value"}],
    }
    await runner._finish_run(context, result, runtime=runtime)
    assert result["status"] == "failed"
    assert result["error"] == output.BLOCKED_OUTPUT
    assert "message" not in result
    assert "interactions" not in result


@pytest.mark.asyncio
async def test_protected_failures_keep_provider_classification_without_diagnostics():
    context = ExecutionContext()
    runtime = PatternRuntime(output_guardrail=Reviewer())
    runner = Agent("reviewed", patterns=[]).get_runner(workspace_enabled=False)
    result = {
        "success": False,
        "error": "private-value",
        "diagnostic_error": "private-value",
        "pattern_errors": [{"error": "private-value"}],
        "model_error": {
            "kind": "rate_limited",
            "status_code": 429,
            "message": "private-value",
            "provider_code": "private-value",
        },
    }
    await runner._finish_run(context, result, runtime=runtime)
    assert result["model_error"] == {"kind": "rate_limited", "status_code": 429}
    assert "private-value" not in repr(result)


@pytest.mark.asyncio
async def test_protected_trace_markers_reach_audit_writer():
    tracer = Tracer()
    tracer.event_writer = AsyncMock()
    runtime = PatternRuntime(output_guardrail=Reviewer(), tracer=tracer)
    with output.bind_output_guardrail(runtime.output_guardrail):
        await tracer.trace_event(
            ACTION_START_TOOL, step_id="step", data={"content": "private-value"}
        )
    await runtime._emit_trace_event(
        ACTION_START_TOOL, step_id="step", data={"content": "private-value"}
    )
    for call in tracer.event_writer.await_args_list:
        assert call.args[0].data == {"content": "private-value", "__audit_only__": True}
    assert tracer.event_writer.await_count == 2


@pytest.mark.asyncio
async def test_failed_policy_resolution_aborts_before_model_execution():
    output.set_output_review_resolver(
        AsyncMock(side_effect=RuntimeError("review unavailable"))
    )
    llm = AsyncMock()
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="test", pattern="react", llm=llm, workspace_enabled=False
        )
    )
    with pytest.raises(RuntimeError, match="review unavailable"):
        await adapter.execute(task="Test", task_id="1")
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_receives_authenticated_user_and_authorized_delegation():
    resolver = AsyncMock(return_value=None)
    output.set_output_review_resolver(resolver)
    token = current_user_id.set(12)
    try:
        with output.bind_output_review_agent(34):
            await output.resolve_output_guardrail("child")
        await output.resolve_output_guardrail("parent")
    finally:
        current_user_id.reset(token)
    assert resolver.await_args_list[0].args[0] == output.OutputReviewRequest(
        "child", 12, 34
    )
    assert resolver.await_args_list[1].args[0] == output.OutputReviewRequest(
        "parent", 12
    )
