"""Truth table for the model-authored "form" flag on a suspended
``ask_user_question`` turn (see the ``waiting_for_user_request["form"]``
comment in ``ReActPattern._handle_control_tool``).

Only ``ask_user_question`` writes this flag: it is the one suspending path
whose ``interactions`` are the model's own structured question.
``send_message(expect_response=True)`` and a tool's own waiting request never
write it, because neither carries a model-authored interaction list to judge.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses

    async def chat(self, **kwargs: Any) -> Any:
        return self.responses.pop(0)


def _control_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


async def _run_ask_user_question(arguments: dict[str, Any]) -> ReActPattern:
    llm = FakeLLM(responses=[_control_call("ask_user_question", arguments)])
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-form-flag")
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)
    assert result["status"] == "waiting_for_user"
    return pattern


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interactions", "expected"),
    [
        pytest.param(
            [{"type": "text_input", "field": "city", "label": "City"}],
            True,
            id="single-answerable-field",
        ),
        pytest.param(
            # An options-less select_one is degraded to a text_input by
            # degrade_options_less_pickers before this flag is computed.
            [{"type": "select_one", "field": "city", "label": "City", "options": []}],
            True,
            id="degraded-picker",
        ),
        pytest.param(
            [
                {"type": "text_input", "field": "shift_name", "label": "Shift name"},
                {"type": "confirm", "field": "notify", "label": "Notify the team?"},
            ],
            True,
            id="mixed-with-confirm",
        ),
        pytest.param(
            [{"type": "confirm", "field": "ok", "label": "Proceed?"}],
            False,
            id="confirm-only",
        ),
        pytest.param(
            [{"type": "connect_apps", "field": "apps", "label": "Connect"}],
            False,
            id="connect-apps-only",
        ),
        pytest.param(
            [{"type": "something_new", "field": "x", "label": "X"}],
            False,
            id="invalid-type-only",
        ),
        pytest.param(
            [{"type": "file_upload", "field": "doc", "label": "Upload"}],
            False,
            id="file-upload-only",
        ),
        pytest.param(
            [
                {
                    "type": "select_one",
                    "field": "city",
                    "label": "City",
                    "options": [
                        {"label": "Sydney", "value": "sydney"},
                        {"label": "Perth", "value": "perth"},
                    ],
                }
            ],
            True,
            id="select-one-with-options",
        ),
        pytest.param(
            # Intended: the rule excludes only confirm and file_upload by
            # type, so a select_one that happens to offer Yes/No is a form.
            [
                {
                    "type": "select_one",
                    "field": "proceed",
                    "label": "Proceed?",
                    "options": [
                        {"label": "Yes", "value": "yes"},
                        {"label": "No", "value": "no"},
                    ],
                }
            ],
            True,
            id="select-one-yes-no-options",
        ),
        pytest.param(
            [
                {
                    "type": "select_multiple",
                    "field": "days",
                    "label": "Days",
                    "options": [
                        {"label": "Monday", "value": "mon"},
                        {"label": "Tuesday", "value": "tue"},
                    ],
                }
            ],
            True,
            id="select-multiple",
        ),
        pytest.param(
            [{"type": "number_input", "field": "hours", "label": "Hours"}],
            True,
            id="number-input",
        ),
        pytest.param(
            # Intended: a pick among the model's own proposed actions is
            # still a form under the rule (only confirm and file_upload are
            # excluded).
            [
                {
                    "type": "action_cards",
                    "field": "plan",
                    "label": "Pick a plan",
                    "options": [
                        {"label": "Plan A", "value": "a"},
                        {"label": "Plan B", "value": "b"},
                    ],
                }
            ],
            True,
            id="action-cards",
        ),
        pytest.param(
            # "boolean" is an alias that normalizes to confirm before the flag
            # is computed, so it is excluded like confirm.
            [{"type": "boolean", "field": "ok", "label": "Proceed?"}],
            False,
            id="boolean-alias-is-confirm",
        ),
        pytest.param(
            [
                {"type": "confirm", "field": "ok", "label": "Proceed?"},
                {"type": "file_upload", "field": "doc", "label": "Upload"},
            ],
            False,
            id="confirm-plus-file-upload",
        ),
        pytest.param(
            # The engine appends a default text_input in _send_waiting_message,
            # but the flag is computed from the model-authored list
            # beforehand, which was empty.
            None,
            False,
            id="no-interactions",
        ),
    ],
)
async def test_form_flag_truth_table(
    interactions: list[dict[str, Any]] | None, expected: bool
) -> None:
    arguments: dict[str, Any] = {"message": "Well?"}
    if interactions is not None:
        arguments["interactions"] = interactions
    pattern = await _run_ask_user_question(arguments)
    assert pattern.waiting_for_user_request["form"] is expected


@pytest.mark.asyncio
async def test_send_message_expecting_a_response_never_writes_form() -> None:
    llm = FakeLLM(
        responses=[
            _control_call(
                "send_message",
                {
                    "message": "Shall I proceed?",
                    "message_type": "question",
                    "expect_response": True,
                },
            )
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(execution_id="exec-send-message"),
    )
    assert result["status"] == "waiting_for_user"
    assert "form" not in pattern.waiting_for_user_request


@pytest.mark.asyncio
async def test_tool_requested_interaction_never_writes_form() -> None:
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-tool-wait")
    context = ExecutionContext()
    context.add_user_message("Ask")

    outcome = await pattern._pause_for_tool_results(
        waiting_pairs=[
            ({"id": "call_a", "name": "tool_a"}, {"message": "Need a value"})
        ],
        context=context,
        runtime=runtime,
    )

    assert outcome["status"] == "waiting_for_user"
    assert "form" not in pattern.waiting_for_user_request


@pytest.mark.asyncio
async def test_form_survives_pause_checkpoint_round_trip_and_resume() -> None:
    """The full path: a real ask_user_question pause, a checkpoint
    serialize/restore through the same JSON encoding the checkpoint store
    uses (``get_state`` is documented JSON-serializable), then a resumed run
    that marks the user's answer message. ``form: True`` must reach that
    marker.

    ``waiting_for_user_request`` is rebound (not mutated) once the pause is
    recorded (react.py's post-pause ``{**self.waiting_for_user_request,
    "message_count": ...}``); if that rebind ever stopped spreading the prior
    dict, ``form`` would be dropped and this test would catch it.
    """

    llm = FakeLLM(
        responses=[
            _control_call(
                "ask_user_question",
                {
                    "message": "What is the shift name?",
                    "interactions": [
                        {
                            "type": "text_input",
                            "field": "shift_name",
                            "label": "Shift name",
                        }
                    ],
                },
            )
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext()
    context.add_user_message("Do the thing")

    first = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(execution_id="exec-form-pause"),
    )
    assert first["status"] == "waiting_for_user"
    assert pattern.waiting_for_user_request["form"] is True

    # The user's answer arrives as an ordinary, unmarked user message; marking
    # it is the resumed run's job (_mark_latest_user_message_as_waiting_response).
    context.add_user_message("Morning shift")

    serialized_state = json.loads(json.dumps(pattern.get_state()))
    resumed_pattern = ReActPattern(max_iterations=2)
    resumed_pattern.load_state(serialized_state)
    resumed_llm = FakeLLM(responses=[{"content": "Noted: Morning shift."}])

    result = await resumed_pattern.run(
        context=context,
        tools=[],
        llm=resumed_llm,
        runtime=PatternRuntime(execution_id="exec-form-resume"),
    )

    assert result["success"] is True
    answer_message = context.messages[-2]
    assert answer_message.content == "Morning shift"
    assert answer_message.metadata["response_to_waiting_for_user"]["form"] is True
