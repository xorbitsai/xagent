"""A host settle only answers the waiting request it names."""

from types import SimpleNamespace

from xagent.core.agent.attachments import CLIENT_METADATA_KEY
from xagent.core.agent.context.message import Message
from xagent.core.agent.pattern.react.react import ReActPattern


def _pattern(*interaction_ids: str) -> ReActPattern:
    pattern = ReActPattern.__new__(ReActPattern)
    pattern.waiting_for_user_request = {
        "kind": "tool_waiting_for_user",
        "message": "connect an app",
        "message_type": "question",
        "requests": [{"interaction_id": i} for i in interaction_ids],
    }
    return pattern


def _settle(interaction_id: str) -> Message:
    return Message(
        role="user",
        content="[interaction settled]",
        metadata={
            CLIENT_METADATA_KEY: {
                "hidden": True,
                "kind": "interaction_settled",
                "interaction_id": interaction_id,
            }
        },
    )


def test_a_settle_for_a_pending_request_is_an_answer() -> None:
    pattern = _pattern("a", "b")
    context = SimpleNamespace(
        messages=[Message(role="assistant", content="q"), _settle("b")]
    )
    assert not pattern._is_stale_settle_message(context.messages[1])
    assert pattern._has_answer_after(context, 1)


def test_a_settle_for_another_interaction_is_not_an_answer() -> None:
    pattern = _pattern("a")
    context = SimpleNamespace(
        messages=[Message(role="assistant", content="q"), _settle("old")]
    )
    assert pattern._is_stale_settle_message(context.messages[1])
    assert not pattern._has_answer_after(context, 1)


def test_a_settle_never_answers_a_plain_question() -> None:
    pattern = ReActPattern.__new__(ReActPattern)
    pattern.waiting_for_user_request = {"message": "which one?"}
    assert pattern._is_stale_settle_message(_settle("a"))


def test_ordinary_user_messages_are_unaffected() -> None:
    pattern = _pattern("a")
    plain = Message(role="user", content="blue")
    assert not pattern._is_stale_settle_message(plain)
    context = SimpleNamespace(messages=[Message(role="assistant", content="q"), plain])
    assert pattern._has_answer_after(context, 1)


def test_the_real_answer_is_picked_over_an_earlier_stale_settle() -> None:
    pattern = _pattern("a")
    context = SimpleNamespace(
        messages=[
            Message(role="assistant", content="q"),
            _settle("old"),
            Message(role="user", content="blue"),
        ]
    )
    answer = pattern._mark_latest_user_message_as_waiting_response(
        context=context, after_message_count=1
    )
    assert answer == "blue"
    assert "response_to_waiting_for_user" not in context.messages[1].metadata
    assert "response_to_waiting_for_user" in context.messages[2].metadata


async def test_a_stale_settle_leaves_the_run_waiting() -> None:
    from unittest.mock import AsyncMock

    pattern = _pattern("a")
    pattern.status = "waiting_for_user"
    pattern.waiting_for_user_request["message_count"] = 1
    runtime = SimpleNamespace(
        approve_waiting_result=AsyncMock(return_value=True),
        checkpoint=AsyncMock(),
    )
    context = SimpleNamespace(
        messages=[Message(role="assistant", content="q"), _settle("old")],
        execution_id="e1",
    )
    result = await pattern._resume_waiting_for_user_if_needed(
        context=context, runtime=runtime, tools=[]
    )
    assert result is not None and result["status"] == "waiting_for_user"
    assert "response_to_waiting_for_user" not in context.messages[1].metadata
