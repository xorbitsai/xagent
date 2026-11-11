"""The system message is identical from turn to turn, so providers can cache it.

DeepSeek-style chat templates render the system message, then the tool schemas,
then the conversation, and cache the longest unchanged prefix. Anything that
varies by turn must therefore stay out of the system message: it is sent in the
turn context, appended to the request message it belongs to.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from xagent.core.agent import AgentExecutionAdapter, AgentExecutionConfig
from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.context.enrichment import MEMORY_CONTEXT_METADATA_KEY
from xagent.core.agent.context.execution import (
    TURN_CONTEXT_HEADER,
    TURN_CONTEXT_METADATA_KEY,
)
from xagent.core.agent.language import OUTPUT_LANGUAGE_METADATA_KEY
from xagent.core.agent.pattern.react import ReActPattern

STATIC_PROMPT = "You are Toby, an AI teammate.\n\nConduct rules stay the same."


def _turn(
    *,
    started: datetime,
    run_id: str,
    raw_text: str,
    memory: str,
    output_language: str | None = None,
) -> ExecutionContext:
    """One Toby-shaped turn: a fresh context whose request carries enriched text
    (thread history, sender line) while the typed text rides as the display
    message, plus a host run id and a recalled memory."""
    ctx = ExecutionContext(execution_id=f"turn-{run_id}", system_prompt=STATIC_PROMPT)
    ctx.created_at = started
    ctx.metadata[TURN_CONTEXT_METADATA_KEY] = f"Run ID for this execution: {run_id}"
    ctx.metadata[MEMORY_CONTEXT_METADATA_KEY] = memory
    if output_language:
        ctx.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = output_language
    ctx.add_user_message(
        f"[Conversation: channel]\n[From: Ann <@U1>] {raw_text}",
        metadata={"display_message": raw_text},
    )
    return ctx


def _first_turn() -> ExecutionContext:
    return _turn(
        started=datetime(2026, 11, 11, 3, 4, 5, tzinfo=timezone.utc),
        run_id="run-aaaa",
        raw_text="summarise the launch thread",
        memory="Ann prefers bullet points.",
    )


def _second_turn() -> ExecutionContext:
    return _turn(
        started=datetime(2026, 11, 11, 9, 58, 41, tzinfo=timezone.utc),
        run_id="run-bbbb",
        raw_text="帮我把这个翻成英文",
        memory="Ann works on the billing team.",
    )


def _contents(messages: list[dict[str, Any]]) -> list[str]:
    return [str(message.get("content") or "") for message in messages]


def test_system_message_is_byte_identical_across_turns() -> None:
    first = _first_turn().get_messages_for_llm()
    second = _second_turn().get_messages_for_llm()

    assert first[0]["role"] == second[0]["role"] == "system"
    assert first[0]["content"] == second[0]["content"]
    # Nothing that varies by turn is in it.
    system = first[0]["content"]
    for per_turn in (
        "run-aaaa",
        "Turn started at",
        "03:04:05",
        "summarise the launch thread",
        "Ann prefers bullet points.",
    ):
        assert per_turn not in system


def test_turn_values_follow_the_request_text_in_the_same_message() -> None:
    ctx = _second_turn()
    messages = ctx.get_messages_for_llm()

    # No extra user turn: some chat templates require user/assistant to alternate.
    assert [message["role"] for message in messages] == ["system", "user"]
    request = ctx.messages[0].content
    assert messages[1]["content"].startswith(f"{request}\n\n{TURN_CONTEXT_HEADER}")
    turn = messages[1]["content"][len(request) :]
    assert "Run ID for this execution: run-bbbb" in turn
    assert "Turn started at: 2026-11-11 09:58:41 UTC" in turn
    assert '"independent_user_request": "帮我把这个翻成英文"' in turn
    assert "Ann works on the billing team." in turn
    # The stored request message is unchanged.
    assert ctx.messages[0].content == request
    # The system message says where the language evidence went.
    assert (
        "given in the turn context after the request text" in (messages[0]["content"])
    )


def test_prefix_through_the_request_stays_fixed_within_a_turn() -> None:
    ctx = _first_turn()
    before = ctx.get_messages_for_llm()

    ctx.add_assistant_message(
        "",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }
        ],
    )
    ctx.add_tool_result("web_search", {"output": "results"}, tool_call_id="call-1")
    after = ctx.get_messages_for_llm()

    assert after[: len(before)] == before
    assert len(after) == len(before) + 2


def test_pinned_language_evidence_stays_in_the_system_message() -> None:
    """Evidence that names only a pinned language is the same every turn."""
    first = _turn(
        started=datetime(2026, 11, 11, 1, 0, 0, tzinfo=timezone.utc),
        run_id="run-1",
        raw_text="one",
        memory="m1",
        output_language="English",
    ).get_messages_for_llm()
    second = _turn(
        started=datetime(2026, 11, 11, 2, 0, 0, tzinfo=timezone.utc),
        run_id="run-2",
        raw_text="two",
        memory="m2",
        output_language="English",
    ).get_messages_for_llm()

    assert first[0]["content"] == second[0]["content"]
    assert '{"output_language": "English"}' in first[0]["content"]
    assert "Canonical request-language evidence" not in first[1]["content"]


def test_react_system_message_is_byte_identical_across_turns() -> None:
    """The ReAct rules appended to the system message keep it identical too."""

    def system_for(ctx: ExecutionContext) -> str:
        pattern = ReActPattern(max_iterations=3)
        messages = pattern._messages_for_llm(
            ctx, has_tools=True, tool_names=["web_search", "final_answer"]
        )
        assert TURN_CONTEXT_HEADER in messages[1]["content"]
        return str(messages[0]["content"])

    first = system_for(_first_turn())
    second = system_for(_second_turn())

    assert first == second
    assert "Turn started at" not in first


class _RecordingLLM:
    model_name = "recording-llm"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "done"


class _NoSkills:
    async def list_skills(self) -> list[Any]:
        return []

    async def get_skill(self, _name: str) -> None:
        return None


def _adapter(llm: _RecordingLLM) -> AgentExecutionAdapter:
    return AgentExecutionAdapter(
        AgentExecutionConfig(
            name="prefix-host",
            pattern="react",
            llm=llm,
            tools=[],
            system_prompt="Base host instructions.",
            service_id="prefix-host-test",
            skill_manager=_NoSkills(),
        )
    )


def _system(call: dict[str, Any]) -> str:
    return next(m["content"] for m in call["messages"] if m["role"] == "system")


def _non_system(call: dict[str, Any]) -> str:
    return "\n\n".join(
        str(m["content"]) for m in call["messages"] if m["role"] != "system"
    )


@pytest.mark.asyncio
async def test_host_turn_notes_stay_out_of_the_system_message() -> None:
    llm = _RecordingLLM()
    adapter = _adapter(llm)

    for run_id in ("run-first", "run-second"):
        await adapter.execute(
            task="Say hi",
            task_id=f"prefix-{run_id}",
            context={TURN_CONTEXT_METADATA_KEY: f"Run ID: `{run_id}`"},
        )

    assert _system(llm.calls[0]) == _system(llm.calls[1])
    assert "run-first" not in _system(llm.calls[0])
    assert "Run ID: `run-first`" in _non_system(llm.calls[0])
    assert "Run ID: `run-second`" in _non_system(llm.calls[1])
    assert "run-first" not in _non_system(llm.calls[1])


@pytest.mark.asyncio
async def test_a_host_using_system_prompt_still_extends_the_system_message() -> None:
    """Hosts that pass per-run text as ``system_prompt`` (older xagent-saas)
    keep today's behaviour: it is appended to the system prompt."""
    llm = _RecordingLLM()
    adapter = _adapter(llm)

    await adapter.execute(
        task="Say hi",
        task_id="prefix-legacy",
        context={"system_prompt": "## Current execution identity\nRun ID `run-old`."},
    )

    system = _system(llm.calls[0])
    assert system.startswith(
        "Base host instructions.\n\n## Current execution identity\nRun ID `run-old`."
    )
    assert "run-old" not in _non_system(llm.calls[0])


@pytest.mark.asyncio
async def test_a_host_without_turn_notes_gets_only_the_clock() -> None:
    llm = _RecordingLLM()
    adapter = _adapter(llm)

    await adapter.execute(task="Say hi", task_id="prefix-plain", context={})

    user = _non_system(llm.calls[0])
    request, turn = user.split("\n\n", 1)
    assert request == "Say hi"
    assert turn.startswith(f"{TURN_CONTEXT_HEADER}\n\nTurn started at: ")
