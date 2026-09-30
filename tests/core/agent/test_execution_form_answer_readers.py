"""``ExecutionContext.latest_form_answer_message`` and
``form_answer_continuation_target``: the single pair of readers that decide
whether the form-answer-continuation text may render for a turn.

Both reuse ``_latest_visible_user_message`` -- they only ever look at the
truly latest visible user message, never search further back for one that
happens to carry a form marker. An independent message typed after the
answer must turn this off, not fall back to the older answer.
"""

from __future__ import annotations

import pytest

from xagent.core.agent.context.execution import ExecutionContext


def _form_marker() -> dict[str, object]:
    return {
        "question": "What is the shift name?",
        "message_type": "question",
        "form": True,
    }


def _context_with_no_messages() -> ExecutionContext:
    return ExecutionContext()


def _context_with_no_metadata() -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    return context


def _context_with_a_non_form_marker() -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "B",
        metadata={
            "response_to_waiting_for_user": {
                "question": "A or B?",
                "message_type": "question",
            }
        },
    )
    return context


def _context_with_form_false() -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Proceed",
        metadata={
            "response_to_waiting_for_user": {
                "question": "Proceed?",
                "message_type": "question",
                "form": False,
            }
        },
    )
    return context


@pytest.mark.parametrize(
    "build_context",
    [
        _context_with_no_messages,
        _context_with_no_metadata,
        _context_with_a_non_form_marker,
        _context_with_form_false,
    ],
    ids=["no-messages", "no-metadata", "non-form-marker", "form-false"],
)
def test_latest_form_answer_message_returns_none(build_context) -> None:
    assert build_context().latest_form_answer_message() is None


def test_latest_user_message_with_form_true_is_returned() -> None:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    answer = context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    assert context.latest_form_answer_message() is answer


def test_a_hidden_latest_user_message_is_skipped_for_a_visible_one() -> None:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    answer = context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    context.add_user_message("(internal note)", hidden=True)
    assert context.latest_form_answer_message() is answer


def test_a_new_independent_message_after_the_answer_turns_it_off() -> None:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    context.add_user_message("Actually, do something else entirely.")
    assert context.latest_form_answer_message() is None


def test_removing_the_answer_message_from_the_context_returns_none() -> None:
    # Stands in for compaction dropping the answer (_drop_oldest_messages):
    # the reader has no cache, so once the message is gone from
    # context.messages it simply stops seeing it.
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    context.messages.pop()
    assert context.latest_form_answer_message() is None


def _dag_step_context() -> ExecutionContext:
    context = ExecutionContext()
    context.metadata["dag_step_id"] = "step-1"
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    return context


def _empty_task_context() -> ExecutionContext:
    # The only user message is itself the answer (skipped by
    # top_level_user_request), and no metadata task is set, so execution_text
    # is empty -- the same condition _system_context uses to suppress its
    # "Current user request" block.
    context = ExecutionContext()
    context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    return context


@pytest.mark.parametrize(
    "build_context",
    [_dag_step_context, _empty_task_context],
    ids=["dag-step", "empty-task"],
)
def test_target_is_none_even_with_a_form_answer(build_context) -> None:
    context = build_context()
    assert context.latest_form_answer_message() is not None
    assert context.form_answer_continuation_target() is None


def test_target_returns_the_form_answer_message_otherwise() -> None:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    answer = context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    assert context.form_answer_continuation_target() is answer
