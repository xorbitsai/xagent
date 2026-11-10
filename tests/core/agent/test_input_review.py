import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from xagent.core.agent import Agent, ContextManager, TraceEventCallback
from xagent.core.agent.attachments import build_image_context_references
from xagent.core.agent.runner import (
    AgentRunner,
    InjectionDisposition,
    UserMessageInjectionOutcome,
    classify_injection,
    track_user_message_injection,
)
from xagent.core.guardrails import input as input_review
from xagent.core.guardrails.input import (
    INPUT_WITHHELD_METADATA_KEY,
    InputDecision,
    withheld_input_placeholder,
)
from xagent.core.user_context import current_user_id

PLATFORM = withheld_input_placeholder("platform_policy")


class Reviewer:
    def __init__(self) -> None:
        self.reviewed: list[str] = []

    async def check(self, text: str) -> InputDecision:
        self.reviewed.append(text)
        if "jailbreak" in text:
            return InputDecision("platform_policy")
        return InputDecision()


class RecordingPattern:
    """Records what the model would see and probes nested review."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.nested_review: str | None = "unset"

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        self.nested_review = await input_review.review_user_input(
            "child", "jailbreak from the parent agent"
        )
        return {"success": True, "output": "done"}


@pytest.fixture(autouse=True)
def reset_resolver(monkeypatch):
    monkeypatch.setattr(input_review, "_resolver", None)


@pytest.fixture
def reviewer() -> Reviewer:
    reviewer = Reviewer()
    input_review.set_input_review_resolver(AsyncMock(return_value=reviewer))
    return reviewer


def _runner(pattern: RecordingPattern, **kwargs: Any) -> AgentRunner:
    return AgentRunner(
        agent=Agent(name="test", patterns=[pattern], llm="fake-llm"),
        workspace_enabled=False,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_standalone_reviews_nothing():
    assert await input_review.review_user_input("1", "jailbreak") is None


@pytest.mark.asyncio
async def test_withheld_task_reaches_the_model_only_as_placeholder(reviewer):
    pattern = RecordingPattern()
    result = await _runner(pattern).run(
        task="jailbreak\n\n[uploaded file context]",
        execution_id="exec",
        metadata={
            "display_message": "jailbreak",
            "request_context": {"display_message": "jailbreak", "turn_id": "t1"},
        },
        task_context_refs=build_image_context_references(
            [{"file_id": "image-1", "name": "a.png", "type": "image/png"}]
        ),
    )

    context = result["context"]
    message = context.messages[0]
    assert pattern.calls[0]["task"] == PLATFORM
    assert message.content == PLATFORM
    assert message.metadata[INPUT_WITHHELD_METADATA_KEY] == "platform_policy"
    assert message.metadata["display_message"] == PLATFORM
    assert message.metadata["turn_id"] == "t1"
    assert not message.context_refs
    assert context.metadata["task"] == PLATFORM
    assert context.metadata["display_message"] == PLATFORM
    assert context.metadata["request_context"]["display_message"] == PLATFORM
    assert "jailbreak" not in json.dumps(context.to_dict(), default=str)
    # The user-authored text was reviewed, not the enriched task.
    assert reviewer.reviewed == ["jailbreak"]
    # Runs started inside the pattern carry agent-written input.
    assert pattern.nested_review is None


@pytest.mark.asyncio
async def test_only_user_authored_text_is_reviewed(reviewer):
    pattern = RecordingPattern()
    enriched = "summarize\n\n[thread history: jailbreak]"
    result = await _runner(pattern).run(
        task=enriched,
        execution_id="exec",
        metadata={"request_context": {"display_message": "summarize"}},
    )
    assert result["context"].messages[0].content == enriched
    assert reviewer.reviewed == ["summarize"]


@pytest.mark.asyncio
async def test_reviewer_failure_withholds_input():
    class Broken:
        async def check(self, text: str) -> InputDecision:
            raise RuntimeError(f"provider echoed {text}")

    input_review.set_input_review_resolver(AsyncMock(return_value=Broken()))
    pattern = RecordingPattern()
    await _runner(pattern).run(task="hello", execution_id="exec")
    assert pattern.calls[0]["task"] == withheld_input_placeholder("unavailable")


@pytest.mark.asyncio
async def test_resolver_failure_aborts_before_the_run():
    input_review.set_input_review_resolver(
        AsyncMock(side_effect=RuntimeError("policy unavailable"))
    )
    pattern = RecordingPattern()
    with pytest.raises(RuntimeError, match="policy unavailable"):
        await _runner(pattern).run(task="hello", execution_id="exec")
    assert not pattern.calls


@pytest.mark.asyncio
async def test_resolver_receives_execution_and_user():
    resolver = AsyncMock(return_value=None)
    input_review.set_input_review_resolver(resolver)
    token = current_user_id.set(7)
    try:
        await input_review.review_user_input("42", "hello")
    finally:
        current_user_id.reset(token)
    request = resolver.await_args.args[0]
    assert (request.execution_id, request.user_id) == ("42", 7)


@pytest.fixture
def live():
    manager = ContextManager()
    context = manager.create_context("live")
    context.add_user_message("original")
    tracer = SimpleNamespace(
        load_latest_checkpoint=AsyncMock(return_value=None), checkpoint=AsyncMock()
    )
    runner = AgentRunner(
        SimpleNamespace(llm=None), tracer=tracer, context_manager=manager
    )
    return runner, context, tracer


@pytest.mark.asyncio
async def test_withheld_injection_is_accepted_as_placeholder(live, reviewer):
    runner, context, _ = live
    files = [{"file_id": "image-1", "name": "a.png", "type": "image/png"}]
    result = await runner.inject_user_message(
        "live",
        execution_message="jailbreak\n\n[uploaded file context]",
        display_message="jailbreak",
        files=files,
        turn_id="turn",
        request_interrupt=False,
    )
    assert result.outcome is UserMessageInjectionOutcome.POSTED_FRESH
    added = context.messages[-1]
    assert added.content == PLATFORM
    assert added.metadata["display_message"] == PLATFORM
    assert added.metadata[INPUT_WITHHELD_METADATA_KEY] == "platform_policy"
    assert added.metadata["files"] == files
    assert not added.context_refs
    assert reviewer.reviewed == ["jailbreak"]

    # A retry is reviewed again and replays the accepted placeholder.
    retry = await runner.inject_user_message(
        "live",
        execution_message="jailbreak\n\n[uploaded file context]",
        display_message="jailbreak",
        turn_id="turn",
        request_interrupt=False,
    )
    assert retry.outcome is UserMessageInjectionOutcome.POSTED_REPLAY
    assert [m.content for m in context.messages] == ["original", PLATFORM]


@pytest.mark.asyncio
async def test_injection_reviews_display_text_not_enrichment(live, reviewer):
    runner, context, _ = live
    await runner.inject_user_message(
        "live",
        execution_message="continue\n\n[thread history: jailbreak]",
        display_message="continue",
        turn_id="turn",
        request_interrupt=False,
    )
    assert context.messages[-1].content.startswith("continue")
    assert INPUT_WITHHELD_METADATA_KEY not in context.messages[-1].metadata
    assert reviewer.reviewed == ["continue"]


@pytest.mark.asyncio
async def test_injection_resolver_failure_writes_nothing(live):
    runner, context, tracer = live
    input_review.set_input_review_resolver(
        AsyncMock(side_effect=RuntimeError("policy unavailable"))
    )
    with track_user_message_injection() as attempt:
        with pytest.raises(RuntimeError) as raised:
            await runner.inject_user_message("live", "hello", turn_id="turn")
    assert (
        classify_injection(attempt.outcome, error=raised.value)
        is InjectionDisposition.FAILED_BEFORE_WRITE
    )
    tracer.checkpoint.assert_not_awaited()
    assert [m.content for m in context.messages] == ["original"]


@pytest.mark.asyncio
async def test_user_message_trace_carries_withhold_reason():
    context = ContextManager().create_context("traced")
    message = context.add_user_message(
        PLATFORM,
        metadata={
            "display_message": PLATFORM,
            "turn_id": "turn",
            INPUT_WITHHELD_METADATA_KEY: "platform_policy",
        },
    )
    tracer = SimpleNamespace(trace_event=AsyncMock())
    await TraceEventCallback().on_user_message_posted(
        runner=SimpleNamespace(tracer=tracer), context=context, message=message
    )
    data = tracer.trace_event.await_args.kwargs["data"]
    assert data["message"] == PLATFORM
    assert data["turn_id"] == "turn"
    assert data[INPUT_WITHHELD_METADATA_KEY] == "platform_policy"
