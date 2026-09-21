"""An options-less picker is published as a free-text field (xagent#2529).

``_send_waiting_message``'s guarantee (xagent#2322) is list-level: at least
one answerable field. A form that mixes an options-less ``select_one`` with
ordinary text inputs passes it untouched, and the picker -- usually the
field the task depends on -- reaches the user as a label over an empty
dropdown (``Select`` renders "No options available", ``select.tsx``). Both
publishing call sites now apply the same rule per field, after field
deduplication and before the list-level check: a ``select_one``,
``select_multiple`` or ``action_cards`` with no usable options is replaced
by a ``text_input`` carrying the same ``field``, ``label`` and
``placeholder``. The ``ask_user_question`` tool result names the replaced
fields so the model can supply options -- by calling the listing tool first
-- next time.

The form below is the one xagent#2529 reports from a production deployment,
asked without having called the tools that would have listed the clients
and staff.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.tools.adapters.vibe.interaction_types import TYPES_REQUIRING_OPTIONS

PICKER_TYPES = sorted(TYPES_REQUIRING_OPTIONS)


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


async def _ask(
    interactions: Any, *, message: str = "Fill in the booking"
) -> tuple[Any, ReActPattern, PatternRuntime, ExecutionContext]:
    llm = FakeLLM(
        responses=[
            _control_call(
                "ask_user_question",
                {"message": message, "interactions": interactions},
            )
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-degrade")
    context = ExecutionContext()
    context.add_user_message("Book it")
    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)
    assert result["status"] == "waiting_for_user"
    return result, pattern, runtime, context


def _published_interactions(runtime: PatternRuntime) -> list[dict[str, Any]]:
    waiting = [
        payload
        for payload in runtime.outbound_messages
        if payload.get("expect_response")
    ]
    assert len(waiting) == 1
    return waiting[0]["metadata"]["interactions"]


def _tool_result_text(context: ExecutionContext) -> str:
    return "".join(
        message.content
        for message in context.messages
        if getattr(message, "role", "") == "tool"
    )


PUBLISHED_OPTIONS = [
    {"label": "Published", "value": "true"},
    {"label": "Draft", "value": "false"},
]

# The production form: two pickers with no options, three text inputs, one
# picker with options.
PRODUCTION_FORM = [
    {"type": "select_one", "field": "client_id", "label": "Client"},
    {
        "type": "select_one",
        "field": "staff_ids",
        "label": "Staff",
        "placeholder": "Pick the assigned staff",
        "options": [],
    },
    {"type": "text_input", "field": "start_time", "label": "Start"},
    {"type": "text_input", "field": "end_time", "label": "End"},
    {"type": "text_input", "field": "address", "label": "Address"},
    {
        "type": "select_one",
        "field": "published",
        "label": "Published?",
        "options": PUBLISHED_OPTIONS,
    },
]


@pytest.mark.asyncio
async def test_the_production_form_reaches_the_user_fully_answerable() -> None:
    """The two options-less pickers become text inputs under their own
    names, labels and placeholder; the rest of the form is untouched; and
    because every field is now answerable, the list-level fallback appends
    nothing."""

    result, _, runtime, _ = await _ask(PRODUCTION_FORM)

    published = _published_interactions(runtime)
    assert [(item["field"], item["type"]) for item in published] == [
        ("client_id", "text_input"),
        ("staff_ids", "text_input"),
        ("start_time", "text_input"),
        ("end_time", "text_input"),
        ("address", "text_input"),
        ("published", "select_one"),
    ]
    assert published[0] == {
        "type": "text_input",
        "field": "client_id",
        "label": "Client",
    }
    assert published[1] == {
        "type": "text_input",
        "field": "staff_ids",
        "label": "Staff",
        "placeholder": "Pick the assigned staff",
    }
    assert published[5]["options"] == PUBLISHED_OPTIONS
    assert result["interactions"] == published


@pytest.mark.asyncio
async def test_the_tool_result_names_the_degraded_fields() -> None:
    """The echoed list already shows ``text_input`` where the model wrote
    ``select_one``, but a model does not diff its own call against the echo.
    The result says so outright, under the final (post-dedup) field names,
    and says what to do instead."""

    _, pattern, _, context = await _ask(PRODUCTION_FORM)

    tool_result = pattern.tool_ledger["call_1"].result
    assert tool_result["degraded_fields"] == ["client_id", "staff_ids"]
    assert tool_result["interactions"][0]["type"] == "text_input"

    echoed = _tool_result_text(context)
    assert "'degraded_fields': ['client_id', 'staff_ids']" in echoed
    assert "shown to the user as free-text inputs" in echoed
    assert "non-empty options" in echoed
    assert "'type': 'text_input', 'field': 'client_id'" in echoed
    assert "Your response" not in echoed


@pytest.mark.asyncio
async def test_a_clean_form_is_published_verbatim_and_unremarked() -> None:
    """No ``degraded_fields`` key at all, rather than an empty list: the key
    is a signal, and a model that sees it on every call learns to ignore
    it."""

    form = [
        {"type": "text_input", "field": "note", "label": "Note"},
        {
            "type": "select_one",
            "field": "published",
            "label": "Published?",
            "options": PUBLISHED_OPTIONS,
        },
    ]
    _, pattern, runtime, context = await _ask(form)

    assert _published_interactions(runtime) == form
    assert "degraded_fields" not in pattern.tool_ledger["call_1"].result
    assert "degraded_fields" not in _tool_result_text(context)


@pytest.mark.asyncio
async def test_a_text_input_with_an_empty_options_list_is_not_a_picker() -> None:
    """The rule is keyed on ``TYPES_REQUIRING_OPTIONS``, not on the presence
    of an ``options`` key: a ``text_input`` renders without options, so an
    empty list on it is noise the normalizer keeps, not a dead control."""

    form = [{"type": "text_input", "field": "note", "label": "Note", "options": []}]
    _, pattern, runtime, _ = await _ask(form)

    assert _published_interactions(runtime) == form
    assert "degraded_fields" not in pattern.tool_ledger["call_1"].result


@pytest.mark.asyncio
@pytest.mark.parametrize("picker_type", PICKER_TYPES)
@pytest.mark.parametrize(
    "options",
    [None, [], [{"label": " ", "value": ""}], "auto"],
    ids=["key-omitted", "empty", "all-blank", "not-a-list"],
)
async def test_every_pick_from_a_list_type_is_degraded(
    picker_type: str, options: Any
) -> None:
    """All three picker types, under every shape that leaves no usable
    option after ``_normalize_ask_user_interactions``: absent, empty,
    filtered to empty, and the non-list it leaves verbatim."""

    picker: dict[str, Any] = {"type": picker_type, "field": "choice", "label": "Which?"}
    if options is not None:
        picker["options"] = options
    _, _, runtime, _ = await _ask([picker])

    expected: dict[str, Any] = {
        "type": "text_input",
        "field": "choice",
        "label": "Which?",
    }
    if picker_type == "select_multiple":
        expected["multiline"] = True
    assert _published_interactions(runtime) == [expected]


@pytest.mark.asyncio
async def test_degradation_reports_the_deduplicated_field_names() -> None:
    """Degradation runs after ``_unique_field`` so the names in the tool
    result are the names the answers will come back under."""

    _, pattern, runtime, _ = await _ask(
        [
            {"type": "select_one", "field": "choice", "label": "Region", "options": []},
            {"type": "select_one", "field": "choice", "label": "City", "options": []},
        ]
    )

    assert pattern.tool_ledger["call_1"].result["degraded_fields"] == [
        "choice",
        "choice_2",
    ]
    assert [item["field"] for item in _published_interactions(runtime)] == [
        "choice",
        "choice_2",
    ]


@pytest.mark.asyncio
async def test_the_tool_path_degrades_its_pickers_too() -> None:
    """``_pause_for_tool_results`` is the other producer of a suspending
    message. Its per-request attribution copies must carry the degraded
    field too, or the structured row (built from them) and the published
    list disagree about what the user was shown."""

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-tool-degrade")
    context = ExecutionContext()
    context.add_user_message("Ask")

    picker = {"type": "select_one", "field": "choice", "label": "Region", "options": []}
    outcome = await pattern._pause_for_tool_results(
        waiting_pairs=[
            (
                {"id": "call_a", "name": "tool_a"},
                {"message": "Need a region", "interactions": [picker]},
            )
        ],
        context=context,
        runtime=runtime,
    )

    degraded = {"type": "text_input", "field": "choice", "label": "Region"}
    assert _published_interactions(runtime) == [degraded]
    assert pattern.waiting_for_user_request is not None
    assert pattern.waiting_for_user_request["requests"][0]["interactions"] == [degraded]
    assert outcome["interactions"] == [degraded]


@pytest.mark.asyncio
async def test_the_degraded_form_passes_the_write_side_validator() -> None:
    """``validate_v1_write_payload`` refuses a ``text_input`` that carries a
    non-empty ``options`` and a picker that carries none. The degraded shape
    has to pass it, or the engine publishes a form the structured row cannot
    hold."""

    from xagent.core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
    from xagent.web.services.task_interaction_service import validate_v1_write_payload

    _, _, runtime, _ = await _ask(PRODUCTION_FORM)
    published = _published_interactions(runtime)
    validate_v1_write_payload(
        AskUserQuestionArgs.model_validate(
            {"message": "Fill in the booking", "interactions": published}
        )
    )
