"""``ExecutionContext.get_messages_for_llm(form_answer_continuation=...)``:
the system-context instruction and the answer-message framing clause, and
their gating.

``form_answer_continuation`` defaults to False and every existing caller
keeps that default (only ReAct's final-messages build site ever passes
True), so the whole matrix below re-proves that default path is untouched
before testing the flag turned on.
"""

from __future__ import annotations

import json
from typing import Callable

import pytest

from xagent.core.agent.context.enrichment import PendingUserResponse
from xagent.core.agent.context.execution import (
    FORM_ANSWER_CONTINUATION_INSTRUCTION,
    FORM_ANSWER_FRAMING,
    ExecutionContext,
)
from xagent.core.agent.language import serialize_pending_user_response


def _form_marker(question: str = "What is the shift name?") -> dict[str, object]:
    return {"question": question, "message_type": "question", "form": True}


def _system_message(messages: list[dict[str, object]]) -> str:
    assert messages[0]["role"] == "system"
    return str(messages[0]["content"])


def _build_form_answer_context() -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    return context


def _build_blank_question_context() -> ExecutionContext:
    # question blank (message_type is still "question"): pending_user_response
    # (strict) refuses it, so this exercises the lifecycle branch, not the
    # latest-pending one.
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift",
        metadata={
            "response_to_waiting_for_user": {
                "question": "",
                "message_type": "question",
                "form": True,
            }
        },
    )
    return context


def _build_two_form_answers_context() -> ExecutionContext:
    """An older answered question, then a second one with the same answer
    text -- the older becomes the historical-evidence branch, the newer
    stays the latest-pending one."""
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift",
        metadata={
            "response_to_waiting_for_user": _form_marker("What is the shift name?")
        },
    )
    context.add_user_message(
        "Morning shift",
        metadata={
            "response_to_waiting_for_user": _form_marker(
                "Do you want to notify the team?"
            )
        },
    )
    return context


def _build_empty_task_context() -> ExecutionContext:
    # The only user message is itself the answer: top_level_user_request
    # skips it, so execution_text is empty.
    context = ExecutionContext()
    context.add_user_message(
        "Morning shift",
        metadata={"response_to_waiting_for_user": _form_marker()},
    )
    return context


def _latest_pending_content(framing: str) -> str:
    return (
        "This user message is the answer to a pending agent question and is "
        "the primary response, not an independent task. "
        f"{framing}"
        "The exact allowlisted question and clean answer are in the canonical "
        "request-language evidence in the system context.\n"
        "Execution-enriched message content follows:\nMorning shift"
    )


def _lifecycle_content(framing: str) -> str:
    return (
        "This user message is the primary response in a waiting lifecycle "
        "whose question text is unavailable or blank, not an independent "
        "task. "
        f"{framing}"
        "Execution-enriched message content follows:\nMorning shift"
    )


def _historical_content() -> str:
    evidence = serialize_pending_user_response(
        PendingUserResponse(
            answer="Morning shift",
            question="What is the shift name?",
            message_type="question",
        )
    )
    return (
        "This user message is the answer to a pending agent question and is "
        "the primary response, not an independent task. "
        "Historical pending-response evidence (JSON):\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
        "\nExecution-enriched message content follows:\nMorning shift"
    )


@pytest.mark.parametrize(
    ("build_context", "flag", "index", "expected"),
    [
        pytest.param(
            _build_form_answer_context,
            False,
            -1,
            _latest_pending_content(""),
            id="latest-pending-flag-off",
        ),
        pytest.param(
            _build_form_answer_context,
            True,
            -1,
            _latest_pending_content(FORM_ANSWER_FRAMING),
            id="latest-pending-flag-on",
        ),
        pytest.param(
            _build_blank_question_context,
            False,
            -1,
            _lifecycle_content(""),
            id="lifecycle-flag-off",
        ),
        pytest.param(
            _build_blank_question_context,
            True,
            -1,
            _lifecycle_content(FORM_ANSWER_FRAMING),
            id="lifecycle-flag-on",
        ),
        pytest.param(
            _build_two_form_answers_context,
            False,
            -2,
            _historical_content(),
            id="historical-flag-off",
        ),
        pytest.param(
            _build_empty_task_context,
            False,
            -1,
            _latest_pending_content(""),
            id="empty-task-flag-off",
        ),
    ],
)
def test_answer_message_content_matches_the_gate_across_every_shape(
    build_context: Callable[[], ExecutionContext],
    flag: bool,
    index: int,
    expected: str,
) -> None:
    """One exact-content pin per (flag, branch) combination. Full-string
    equality already implies ``FORM_ANSWER_FRAMING`` is present or absent, so
    no separate substring assertion is needed alongside it. The historical
    and empty-task rows are byte-identical by construction when the flag is
    False (``form_answer_target`` is always None then), but both are run
    explicitly here rather than left to inference.
    """
    context = build_context()
    messages = context.get_messages_for_llm(form_answer_continuation=flag)
    assert str(messages[index]["content"]) == expected


def test_instruction_renders_exactly_once_between_the_task_and_the_focus_rules() -> (
    None
):
    context = _build_form_answer_context()
    messages = context.get_messages_for_llm(form_answer_continuation=True)
    system = _system_message(messages)
    assert system.count(FORM_ANSWER_CONTINUATION_INSTRUCTION) == 1
    # The instruction sits between the task text's blank line and the
    # focus-rules sentence.
    task_marker = "Do the thing\n\n"
    focus_rules = "Conversation focus rules: answer the current user request above."
    instruction_index = system.index(FORM_ANSWER_CONTINUATION_INSTRUCTION)
    assert system.index(task_marker) + len(task_marker) == instruction_index
    assert instruction_index + len(
        FORM_ANSWER_CONTINUATION_INSTRUCTION
    ) == system.index(focus_rules)


def test_flag_false_is_identical_whether_or_not_it_is_passed_explicitly() -> None:
    context = _build_form_answer_context()
    default_messages = context.get_messages_for_llm()
    explicit_messages = context.get_messages_for_llm(form_answer_continuation=False)
    assert default_messages == explicit_messages


def _dag_step_context() -> ExecutionContext:
    context = _build_form_answer_context()
    context.metadata["dag_step_id"] = "step-1"
    return context


@pytest.mark.parametrize(
    "build_context",
    [_dag_step_context, _build_empty_task_context],
    ids=["dag-step", "empty-task"],
)
def test_flag_true_still_suppresses_both_texts_when_target_is_none(
    build_context: Callable[[], ExecutionContext],
) -> None:
    """Rendering follows the target, not the flag alone: with the flag True,
    a DAG step or an empty current task still renders neither text, because
    ``form_answer_continuation_target()`` is None in both cases."""
    context = build_context()
    messages = context.get_messages_for_llm(form_answer_continuation=True)
    system = _system_message(messages)
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION not in system
    answer_content = str(messages[-1]["content"])
    assert FORM_ANSWER_FRAMING not in answer_content


def test_flag_true_puts_framing_only_on_the_latest_of_two_identical_text_answers() -> (
    None
):
    context = _build_two_form_answers_context()
    messages = context.get_messages_for_llm(form_answer_continuation=True)
    # messages: [system, "Do the thing", older answer, latest answer]
    older_answer_content = str(messages[-2]["content"])
    latest_answer_content = str(messages[-1]["content"])

    assert FORM_ANSWER_FRAMING not in older_answer_content
    assert "Historical pending-response evidence (JSON):" in older_answer_content
    assert FORM_ANSWER_FRAMING in latest_answer_content
    assert (
        "The exact allowlisted question and clean answer are in the canonical "
        "request-language evidence in the system context."
    ) in latest_answer_content


def test_both_texts_are_pinned_literally() -> None:
    """Every other test takes its expected text from these two constants, so a
    reworded constant -- for instance one that drops the approval sentences --
    would leave them all green. This is the one place the model-facing wording
    is written out in full.
    """
    assert FORM_ANSWER_FRAMING == "It replies to the form you asked for. "
    assert FORM_ANSWER_CONTINUATION_INSTRUCTION == (
        "The latest user message answers the form you asked for. Use those "
        "answers to continue the current user request. Do not ask again for "
        "information those answers already provide; if something required is "
        "still missing or unusable, ask only for that. If the answers change or "
        "cancel the request, follow the answers. This does not replace any "
        "confirmation or approval your instructions require before an action. "
        "Approving one proposal does not approve a changed or different "
        "action.\n\n"
    )
