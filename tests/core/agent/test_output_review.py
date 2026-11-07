from unittest.mock import AsyncMock

import pytest

from xagent.core.agent import PatternRuntime
from xagent.core.agent.execution_adapter import (
    AgentExecutionAdapter,
    AgentExecutionConfig,
)
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
