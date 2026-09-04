from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.context.enrichment import (
    PendingUserResponse,
    TopLevelUserRequest,
    pending_user_response,
    top_level_user_request,
)
from xagent.core.agent.language import (
    canonical_unpinned_request_language_policy,
    render_request_language_harness,
    serialize_pending_user_response,
)
from xagent.core.agent.pattern.dag.dag import DAGPattern


def _request(text: str) -> TopLevelUserRequest:
    return TopLevelUserRequest(text, text, "text")


def _marked_message(answer: str, marker: Any) -> Any:
    context = ExecutionContext()
    return context.add_user_message(
        answer,
        metadata={"response_to_waiting_for_user": marker},
    )


def test_pending_response_serializer_exposes_only_allowlisted_exact_fields() -> None:
    answer = "ANSWER_BEGIN_" + "答" * 8_000 + "_ANSWER_END"
    question = "Which output language? " + "Q" * 8_000
    message = _marked_message(
        answer,
        {
            "question": question,
            "message_type": "question",
            "tool_name": "private_connector",
            "tool_call_id": "secret-id",
            "interactions": [{"options": ["Spanish"]}],
            "requests": [{"internal": True}],
        },
    )

    response = pending_user_response(message)
    assert response is not None
    serialized = serialize_pending_user_response(response)

    assert serialized == {
        "answer": answer,
        "question": question,
        "message_type": "question",
    }
    serialized_text = json.dumps(serialized, ensure_ascii=False)
    assert serialized_text.count(answer) == 1
    assert serialized_text.count(question) == 1
    assert "private_connector" not in serialized_text
    assert "secret-id" not in serialized_text
    assert "options" not in serialized_text


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"response_to_waiting_for_user": True},
        {"response_to_waiting_for_user": False},
        {"response_to_waiting_for_user": "legacy"},
        {"response_to_waiting_for_user": 1},
        {"response_to_waiting_for_user": None},
        {"response_to_waiting_for_user": {}},
        {"response_to_waiting_for_user": {"question": " \n"}},
        {"response_to_waiting_for_user": {"question": 7}},
    ],
)
def test_pending_response_rejects_malformed_or_blank_marker(
    metadata: dict[str, Any] | None,
) -> None:
    context = ExecutionContext()
    message = context.add_user_message("Spanish", metadata=metadata)
    assert pending_user_response(message) is None


def test_pending_response_defaults_invalid_message_type_without_leaking_it() -> None:
    response = pending_user_response(
        _marked_message(
            "Spanish",
            {"question": "Which output language?", "message_type": ["internal"]},
        )
    )

    assert response == PendingUserResponse(
        answer="Spanish",
        question="Which output language?",
        message_type="question",
    )


@pytest.mark.parametrize(
    "marker",
    [True, "legacy", {"question": " \n"}, {"question": 7}],
)
def test_strict_parser_does_not_change_layer_a_marker_compatibility(
    marker: Any,
) -> None:
    context = ExecutionContext()
    context.add_user_message("Draft the email.")
    message = context.add_user_message(
        "Spanish",
        metadata={"response_to_waiting_for_user": marker},
    )

    assert pending_user_response(message) is None
    assert top_level_user_request(context).language_text == "Draft the email."


def test_language_question_and_terse_selection_are_preserved_for_policy() -> None:
    response = pending_user_response(
        _marked_message(
            "Spanish",
            {"question": "Which output language?", "message_type": "question"},
        )
    )
    assert response is not None
    harness = render_request_language_harness(_request("Draft the email."), response)
    evidence = json.loads(harness.split("\n", 2)[1])

    assert evidence["pending_response"] == {
        "answer": "Spanish",
        "question": "Which output language?",
        "message_type": "question",
    }
    assert "question explicitly asks for the output language or script" in harness
    assert "answer is an unambiguous selection" in harness


def test_caller_pin_and_explicit_answer_override_share_one_policy() -> None:
    policy = canonical_unpinned_request_language_policy()

    assert "request_context.output_language is the sole hard language authority" in (
        policy
    )
    assert "answer explicitly asks to translate, rewrite, or continue" in policy


def test_city_question_and_language_name_are_not_a_language_override() -> None:
    policy = canonical_unpinned_request_language_policy()

    assert '"Which city should the email mention?" followed by "Spanish"' in policy
    assert "remains ordinary conversation context" in policy


def test_harness_preserves_large_request_and_answer_exactly_once() -> None:
    request = "REQUEST_BEGIN_" + "請" * 8_000 + "_REQUEST_END"
    answer = "Continue in Spanish. " + "A" * 8_000
    response = PendingUserResponse(answer, "Which language?", "question")

    harness = render_request_language_harness(_request(request), response)

    assert harness.count(request) == 1
    assert harness.count(answer) == 1
    assert harness.count("Which language?") == 1


def test_request_language_harness_is_not_active_in_root_consumers() -> None:
    context = ExecutionContext()
    context.add_user_message("Draft the email.")

    assert "Canonical request-language evidence" not in context._system_context()
    assert "Canonical request-language evidence" not in json.dumps(
        context.get_messages_for_llm()
    )


def test_request_language_representation_is_not_active_in_dag_forwarding() -> None:
    root = ExecutionContext()
    root.add_user_message("Draft the email.")
    child = root.create_child_context(execution_id="step")
    pattern = DAGPattern(lambda **_: None)
    pattern.status = "waiting_for_user"
    pattern.active_step_id = "draft"
    pattern.active_step_ids = ["draft"]
    pattern.active_step_contexts = {"draft": child.to_dict()}
    pattern.active_step_pattern_states = {
        "draft": {
            "status": "waiting_for_user",
            "waiting_for_user_request": {
                "message": "Which output language?",
                "message_type": "question",
                "tool_call_id": "secret-id",
            },
        }
    }
    pattern.planned_user_message_count = 1
    root.add_user_message("Spanish")

    assert pattern._forward_user_response_to_waiting_step(root)
    assert "response_to_waiting_for_user" not in root.messages[-1].metadata
    restored_child = ExecutionContext.from_dict(pattern.active_step_contexts["draft"])
    assert "response_to_waiting_for_user" not in restored_child.messages[-1].metadata
