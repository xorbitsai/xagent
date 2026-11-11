"""The current request text is sent once, not in both system and user messages."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.language import OUTPUT_LANGUAGE_METADATA_KEY

LONG_REQUEST = "[Memory: digest]\nuser: earlier line\n\nwhy is the sky blue UNIQUE-REQ"
FOCUS = "Conversation focus rules:"
LANGUAGE = "Canonical request-language evidence"


def _count(messages: list[dict[str, Any]], needle: str) -> int:
    return sum(str(m.get("content") or "").count(needle) for m in messages)


def _system(messages: list[dict[str, Any]]) -> str:
    return str(messages[0]["content"])


def test_normal_turn_sends_request_once_with_focus_and_language_rules() -> None:
    ctx = ExecutionContext(execution_id="dedupe-normal")
    ctx.add_user_message(LONG_REQUEST)

    messages = ctx.get_messages_for_llm()

    assert _count(messages, "UNIQUE-REQ") == 1
    assert messages[-1]["content"] == LONG_REQUEST
    system = _system(messages)
    assert "the latest user message" in system
    assert FOCUS in system
    assert "answer the latest user message" in system
    assert LANGUAGE in system
    assert "Latest user message" in system


def test_system_prefix_is_stable_across_different_requests() -> None:
    first = ExecutionContext(execution_id="a")
    first.add_user_message("first question")
    second = ExecutionContext(execution_id="a")
    second.created_at = first.created_at
    second.add_user_message("a completely different, much longer question")

    assert _system(first.get_messages_for_llm()) == _system(
        second.get_messages_for_llm()
    )


def test_request_stays_deduped_after_tool_turns() -> None:
    ctx = ExecutionContext(execution_id="dedupe-tools")
    ctx.add_user_message(LONG_REQUEST)
    ctx.add_assistant_message("calling a tool")

    messages = ctx.get_messages_for_llm()

    assert _count(messages, "UNIQUE-REQ") == 1


def test_pinned_language_keeps_focus_rules_and_directive() -> None:
    ctx = ExecutionContext(execution_id="dedupe-pinned")
    ctx.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "English"
    ctx.add_user_message(LONG_REQUEST)

    system = _system(ctx.get_messages_for_llm())

    assert FOCUS in system
    assert "English" in system
    assert "UNIQUE-REQ" not in system


def test_request_kept_in_system_when_user_message_is_hidden() -> None:
    ctx = ExecutionContext(execution_id="dedupe-hidden")
    ctx.metadata["task"] = LONG_REQUEST
    ctx.add_user_message("something else")
    ctx.messages[-1] = replace(ctx.messages[-1], hidden=True)

    messages = ctx.get_messages_for_llm()

    assert "UNIQUE-REQ" in _system(messages)
    assert FOCUS in _system(messages)
    assert "the current user request above" in _system(messages)


def test_request_kept_in_system_when_only_task_metadata_has_it() -> None:
    ctx = ExecutionContext(execution_id="dedupe-task-only")
    ctx.metadata["task"] = LONG_REQUEST

    messages = ctx.get_messages_for_llm()

    assert _count(messages, "UNIQUE-REQ") == 1
    assert "UNIQUE-REQ" in _system(messages)


def test_request_kept_when_waiting_answer_follows_it() -> None:
    ctx = ExecutionContext(execution_id="dedupe-waiting")
    ctx.add_user_message(LONG_REQUEST)
    ctx.add_assistant_message("Which city?")
    ctx.add_user_message(
        "Beijing",
        metadata={"response_to_waiting_for_user": {"question": "Which city?"}},
    )

    messages = ctx.get_messages_for_llm()

    assert "UNIQUE-REQ" in _system(messages)
    assert FOCUS in _system(messages)


def test_request_kept_when_sent_user_message_differs_from_request() -> None:
    ctx = ExecutionContext(execution_id="dedupe-differs")
    ctx.metadata["task"] = LONG_REQUEST
    ctx.add_user_message(LONG_REQUEST)
    ctx.messages[-1] = replace(
        ctx.messages[-1],
        content=[{"type": "text", "text": LONG_REQUEST}],  # type: ignore[arg-type]
    )

    system = _system(ctx.get_messages_for_llm())

    assert "UNIQUE-REQ" in system
    assert "the current user request above" in system


def test_dag_step_context_unchanged() -> None:
    ctx = ExecutionContext(execution_id="dedupe-dag")
    ctx.metadata["dag_step_id"] = "s1"
    ctx.add_user_message(LONG_REQUEST)

    system = _system(ctx.get_messages_for_llm())

    assert "Current user request" not in system
    assert "DAG step execution scope:" in system
