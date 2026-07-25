"""Reading a user's answer to a confirmation prompt.

Answers arrive both as a widget value and as free-form chat, in either of the
product's languages. An unrecognized answer must never read as approval, but it
also must not be indistinguishable from a refusal — otherwise the model
re-proposes the action and the run pauses again.
"""

from __future__ import annotations

import pytest

from xagent.core.tools.confirmation import (
    RESERVED_TOOL_ARG_PREFIX,
    strip_reserved_tool_args,
    tool_result_waits_for_user,
    user_approved_confirmation,
)


@pytest.mark.parametrize(
    "response",
    [
        "approve",
        "Approve this action",
        "computer_action_decision: approve",
        "yes",
        "Yes, go ahead",
        "ok",
        "Confirm.",
        "同意",
        "好的",
        "可以，继续",
    ],
)
def test_recognizes_approval(response: str) -> None:
    assert user_approved_confirmation(response) is True


@pytest.mark.parametrize(
    "response",
    [
        "deny",
        "Deny this action",
        "computer_action_decision: deny",
        "no",
        "No, stop",
        "cancel",
        "拒绝",
        "不要",
        "算了",
    ],
)
def test_recognizes_refusal(response: str) -> None:
    assert user_approved_confirmation(response) is False


@pytest.mark.parametrize(
    "response",
    [
        "",
        "what does that button do?",
        "先看看别的",
    ],
)
def test_unclear_answers_are_neither(response: str) -> None:
    assert user_approved_confirmation(response) is None


def test_mixed_signals_are_never_read_as_consent() -> None:
    """Matching is whole-phrase, so an ambiguous sentence stays undecided.

    Guessing from keywords inside prose is how "don't approve" becomes an
    approval. Undecided is the safe answer: the caller asks the user plainly
    instead of acting or silently refusing.
    """
    assert user_approved_confirmation("yes but cancel it") is not True
    assert user_approved_confirmation("approve, no wait") is not True


def test_waiting_status_detection() -> None:
    assert tool_result_waits_for_user({"status": "waiting_for_user"}) is True
    assert tool_result_waits_for_user({"status": "WAITING_FOR_USER "}) is True
    assert tool_result_waits_for_user({"status": "completed"}) is False
    assert tool_result_waits_for_user("waiting_for_user") is False
    assert tool_result_waits_for_user(None) is False


def test_reserved_arguments_cannot_be_supplied_by_a_model() -> None:
    args = {
        "expected_frame_id": "frame-1",
        f"{RESERVED_TOOL_ARG_PREFIX}tool_approval": {"decision": "approve"},
        f"{RESERVED_TOOL_ARG_PREFIX}computer_approval": {"decision": "approve"},
        f"{RESERVED_TOOL_ARG_PREFIX}step_id": "forged",
    }

    assert strip_reserved_tool_args(args) == {"expected_frame_id": "frame-1"}
