"""A suspended ReAct run always publishes at least one answerable field.

xagent#1528 declared an empty ``interactions`` list a legitimate shape and
left "no controls means answer in free text" to the reader. No reader
implements it, so a model that emits ``ask_user_question`` with an empty or
missing ``interactions`` -- or that suspends through ``send_message``, which
has no ``interactions`` parameter at all -- produced a ``waiting_for_user``
turn the frontend rendered as ordinary assistant prose with no form. The
composer stays open during a wait (``waiting_for_user`` is a stopped status,
``ChatInput.tsx``), so a free-text reply was still possible there; what was
missing is the form, and with it every surface built on the structured
interaction row, which is built from this same list.

``ReActPattern._send_waiting_message`` is the one place every suspending
path publishes through, and it appends a ``text_input`` field whenever
nothing already in the list is answerable -- an empty list, but equally a
list whose every entry the render surface would drop or render with nothing
to pick. Appended, not substituted for the list: an options-less picker's
``label`` is the question text, and the run has no other copy of it. These
cells pin that on all three paths and pin the reverse: a model that did
supply an answerable field gets exactly its own, untouched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.pattern.react import react
from xagent.core.tools.adapters.vibe.interaction_types import INTERACTION_TYPES


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


async def _run(name: str, arguments: dict[str, Any]) -> tuple[Any, PatternRuntime]:
    llm = FakeLLM(responses=[_control_call(name, arguments)])
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-waiting-default")
    context = ExecutionContext()
    context.add_user_message("Do the thing")

    result = await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)
    assert result["status"] == "waiting_for_user"
    return result, runtime


def _published_interactions(runtime: PatternRuntime) -> list[dict[str, Any]]:
    waiting = [
        payload
        for payload in runtime.outbound_messages
        if payload.get("expect_response")
    ]
    assert len(waiting) == 1
    return waiting[0]["metadata"]["interactions"]


DEFAULT_FIELD = {
    "type": "text_input",
    "field": "response",
    "label": "Your response",
    "placeholder": "Type your answer",
    "multiline": True,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"message": "Which one?", "interactions": []},
        {"message": "Which one?"},
    ],
    ids=["empty-list", "key-omitted"],
)
async def test_ask_user_question_with_no_interactions(
    arguments: dict[str, Any],
) -> None:
    """``key-omitted`` is the shape Kimi K2.5 was observed producing:
    ``interactions`` is in the schema's ``required`` list and the model omits
    it anyway."""

    result, runtime = await _run("ask_user_question", arguments)
    assert _published_interactions(runtime) == [DEFAULT_FIELD]
    assert result["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interactions",
    ["pick one", {"type": "text_input", "field": "city"}, 7],
    ids=["string", "dict", "int"],
)
async def test_ask_user_question_with_a_non_list_interactions(
    interactions: Any,
) -> None:
    """``_normalize_ask_user_interactions`` already flattens every non-list to
    ``[]``; the appended field is what keeps that from reaching the user as an
    unanswerable message."""

    _, runtime = await _run(
        "ask_user_question", {"message": "Which one?", "interactions": interactions}
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_send_message_that_expects_a_response() -> None:
    """``send_message`` has no ``interactions`` parameter, so its suspending
    branch had no ``interactions`` metadata key at all before this guard."""

    result, runtime = await _run(
        "send_message",
        {
            "message": "Here is my plan. Shall I proceed?",
            "message_type": "question",
            "expect_response": True,
        },
    )
    assert _published_interactions(runtime) == [DEFAULT_FIELD]
    assert result["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_send_message_pause_carries_the_field_into_the_structured_row() -> None:
    """A task with ``interaction_protocol_version`` set reads its controls off
    the interaction row, which is built from ``clarification_draft`` -- not off
    the chat message. Publishing the field only in the outbound metadata would
    leave that surface empty, so the waiting request has to carry it too.

    The draft must still attribute itself to ``send_message``: the field is
    appended by the engine, it does not turn the call into an
    ``ask_user_question``.
    """

    result, _ = await _run(
        "send_message",
        {
            "message": "Here is my plan. Shall I proceed?",
            "message_type": "question",
            "expect_response": True,
        },
    )
    draft = result["clarification_draft"]
    assert draft is not None
    assert draft.source == "send_message"
    assert list(draft.interactions) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_send_message_records_the_field_without_echoing_it_to_the_model() -> None:
    """Two audiences, two answers. The trace ledger has to show what the pause
    actually published or a trace cannot be used to check this invariant at
    all. The model's own tool result must not: ``send_message`` has no
    ``interactions`` parameter, and handing one back reads as a structured form
    it produced."""

    llm = FakeLLM(
        responses=[
            _control_call(
                "send_message", {"message": "Shall I?", "expect_response": True}
            )
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-waiting-ledger")
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert pattern.tool_ledger["call_1"].result["interactions"] == [DEFAULT_FIELD]
    tool_messages = [
        message.content
        for message in context.messages
        if getattr(message, "role", "") == "tool"
    ]
    assert tool_messages
    assert "Your response" not in "".join(tool_messages)


@pytest.mark.asyncio
async def test_ask_user_question_echoes_only_what_the_model_itself_supplied() -> None:
    """``ask_user_question`` does have an ``interactions`` parameter, so its
    tool result echoes the model's own list back -- but only that list. The
    appended field is the engine's, and handing it back would read on a
    retry/replan as a control the model had written itself."""

    llm = FakeLLM(
        responses=[
            _control_call(
                "ask_user_question",
                {
                    "message": "Which one?",
                    "interactions": [
                        {
                            "type": "select_one",
                            "field": "choice",
                            "label": "Which region?",
                            "options": [],
                        }
                    ],
                },
            )
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-ask-echo")
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert _published_interactions(runtime) == [
        {
            "type": "select_one",
            "field": "choice",
            "label": "Which region?",
            "options": [],
        },
        DEFAULT_FIELD,
    ]
    echoed = "".join(
        message.content
        for message in context.messages
        if getattr(message, "role", "") == "tool"
    )
    assert "Which region?" in echoed
    assert "Your response" not in echoed


@pytest.mark.asyncio
async def test_a_non_waiting_send_message_records_no_interactions() -> None:
    """The ledger key is added on the suspending branch only: an ordinary
    message publishes no field and must not claim to have published one."""

    llm = FakeLLM(
        responses=[
            _control_call("send_message", {"message": "Working on it."}),
            {"content": "done", "tool_calls": []},
        ]
    )
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(execution_id="exec-plain-send"),
    )

    assert "interactions" not in pattern.tool_ledger["call_1"].result


@pytest.mark.asyncio
async def test_a_hidden_waiting_message_keeps_its_visible_flag() -> None:
    """``visible`` is passed through, not overridden: the fallback adds a
    field, it does not decide whether the message is shown."""

    _, runtime = await _run(
        "send_message",
        {
            "message": "Quiet question",
            "expect_response": True,
            "visible": False,
        },
    )
    waiting = [p for p in runtime.outbound_messages if p.get("expect_response")]
    assert len(waiting) == 1
    assert waiting[0]["visible"] is False
    assert waiting[0]["metadata"]["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_tool_requested_pause_with_no_interactions() -> None:
    """The third suspending path: a tool that reports ``waiting_for_user``
    without offering any interaction of its own."""

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
    assert outcome["interactions"] == [DEFAULT_FIELD]
    assert _published_interactions(runtime) == [DEFAULT_FIELD]
    assert pattern.waiting_for_user_request is not None
    assert pattern.waiting_for_user_request["interactions"] == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_tool_requested_pause_carries_the_field_into_the_draft() -> None:
    """The third path reaches the structured surface through the same
    ``clarification_draft``, which was only pinned on ``send_message``. The
    draft takes its ``interactions`` from the waiting request's top-level
    list -- the published one -- not from the per-tool ``requests`` copies."""

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-tool-wait-draft")
    context = ExecutionContext()
    context.add_user_message("Ask")

    outcome = await pattern._pause_for_tool_results(
        waiting_pairs=[
            ({"id": "call_a", "name": "tool_a"}, {"message": "Need a value"})
        ],
        context=context,
        runtime=runtime,
    )

    draft = outcome["clarification_draft"]
    assert draft is not None
    assert draft.source == "tool_waiting"
    assert list(draft.interactions) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_the_per_request_copies_exclude_the_appended_field() -> None:
    """``requests[*]["interactions"]`` is the per-tool attribution of the
    published list: it records what that tool itself asked for. The appended
    field is the engine's and belongs to no tool, so it stays out of every
    copy even though it is in the published one.

    ``tool_a`` brings a real (if unanswerable) entry so that ``tool_b``'s
    empty copy is a distinguishing assertion rather than a truism.
    """

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-tool-wait-copies")
    context = ExecutionContext()
    context.add_user_message("Ask")

    picker = {
        "type": "select_one",
        "field": "choice",
        "label": "Which region?",
        "options": [],
    }
    await pattern._pause_for_tool_results(
        waiting_pairs=[
            (
                {"id": "call_a", "name": "tool_a"},
                {"message": "Need a region", "interactions": [picker]},
            ),
            ({"id": "call_b", "name": "tool_b"}, {"message": "Need a value"}),
        ],
        context=context,
        runtime=runtime,
    )

    assert _published_interactions(runtime) == [picker, DEFAULT_FIELD]
    assert pattern.waiting_for_user_request is not None
    requests = pattern.waiting_for_user_request["requests"]
    assert [request["interactions"] for request in requests] == [[picker], []]


@pytest.mark.asyncio
async def test_send_message_without_expect_response_stays_field_free() -> None:
    """The fallback must not leak onto messages that do not suspend the run."""

    llm = FakeLLM(
        responses=[
            _control_call(
                "send_message",
                {"message": "Working on it.", "message_type": "progress"},
            ),
            _control_call(
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "Done.",
                    "outcome": "completed",
                },
            ),
        ]
    )
    pattern = ReActPattern(max_iterations=3)
    runtime = PatternRuntime(execution_id="exec-no-wait")
    context = ExecutionContext()
    context.add_user_message("Do the thing")

    await pattern.run(context=context, tools=[], llm=llm, runtime=runtime)

    assert runtime.outbound_messages
    for payload in runtime.outbound_messages:
        assert not payload.get("expect_response")
        assert "interactions" not in payload["metadata"]


@pytest.mark.asyncio
async def test_model_supplied_interactions_are_not_augmented() -> None:
    """The reverse assertion: a well-formed question keeps exactly its own
    fields, with nothing appended."""

    supplied = [
        {
            "type": "select_one",
            "field": "city",
            "label": "City",
            "options": [{"label": "Paris", "value": "paris"}],
        }
    ]
    result, runtime = await _run(
        "ask_user_question", {"message": "Which city?", "interactions": supplied}
    )
    published = _published_interactions(runtime)
    assert [item["field"] for item in published] == ["city"]
    assert published[0]["type"] == "select_one"
    assert result["interactions"] == published


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "picker_type", ["select_one", "select_multiple", "action_cards"]
)
async def test_a_picker_whose_options_were_all_blank_keeps_its_label(
    picker_type: str,
) -> None:
    """The list is non-empty and still unanswerable.

    ``_normalize_ask_user_interactions`` drops blank options but keeps the
    interaction itself, so a picker whose every option was blank arrives here
    as an entry with ``options == []`` -- a control with nothing to select,
    which is the same dead end an empty list is. An emptiness test lets it
    through; the answerability test appends a field to it. The picker itself
    stays: its ``label`` is the question, and dropping it leaves a bare input
    box asking nothing.
    """

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which one?",
            "interactions": [
                {
                    "type": picker_type,
                    "field": "choice",
                    "label": "Which region?",
                    "options": [{"label": "  ", "value": ""}],
                }
            ],
        },
    )
    assert _published_interactions(runtime) == [
        {
            "type": picker_type,
            "field": "choice",
            "label": "Which region?",
            "options": [],
        },
        DEFAULT_FIELD,
    ]


@pytest.mark.asyncio
async def test_a_picker_whose_options_are_not_a_list_gets_a_field() -> None:
    """``_normalize_ask_user_interactions`` warns about a non-list ``options``
    and then leaves it exactly as the model wrote it. A truthy non-list is
    still nothing the render surface can iterate, so answerability has to test
    the type, not just truthiness."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which one?",
            "interactions": [
                {
                    "type": "select_one",
                    "field": "choice",
                    "label": "Choice",
                    "options": "auto",
                }
            ],
        },
    )
    assert _published_interactions(runtime) == [
        {
            "type": "select_one",
            "field": "choice",
            "label": "Choice",
            "options": "auto",
        },
        DEFAULT_FIELD,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interaction", "expected_type"),
    [
        ({"type": "confirm", "field": "ok", "label": "Proceed?"}, "confirm"),
        ({"type": "file_upload", "field": "doc", "label": "Upload"}, "file_upload"),
        ({"type": "number_input", "field": "n", "label": "How many?"}, "number_input"),
        ({"type": "text_input", "field": "note", "label": "Note"}, "text_input"),
        ({"type": "text", "field": "note", "label": "Note"}, "text_input"),
        ({"type": "boolean", "field": "ok", "label": "Proceed?"}, "confirm"),
    ],
    ids=[
        "confirm",
        "file_upload",
        "number_input",
        "text_input",
        "alias_text",
        "alias_boolean",
    ],
)
async def test_a_type_that_needs_no_options_is_left_alone(
    interaction: dict[str, Any],
    expected_type: str,
) -> None:
    """The answerability test must not fire on the types that never render
    options -- appending beside one of those adds a second box asking nothing.
    The last two are aliases the schema ``enum`` never offers:
    ``_normalize_ask_user_interactions`` maps them onto a canonical name, so
    the answerability check and the write-side validator -- neither of which
    knows the aliases -- see the same seven."""

    _, runtime = await _run(
        "ask_user_question", {"message": "Well?", "interactions": [interaction]}
    )
    published = _published_interactions(runtime)
    # Field name too: an unmapped alias would land on the appended field,
    # whose type is itself ``text_input`` and would satisfy a type check alone.
    assert [(item["type"], item["field"]) for item in published] == [
        (expected_type, interaction["field"])
    ]


@pytest.mark.asyncio
async def test_one_answerable_field_keeps_the_whole_list() -> None:
    """The append is all-or-nothing: one usable control is enough, and the
    unusable one beside it is published as the model wrote it rather than
    being pruned. Pruning is the write side's decision, not this one's."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which city?",
            "interactions": [
                {"type": "select_one", "field": "empty", "label": "E", "options": []},
                {
                    "type": "select_one",
                    "field": "city",
                    "label": "City",
                    "options": [{"label": "Paris", "value": "paris"}],
                },
            ],
        },
    )
    assert [item["field"] for item in _published_interactions(runtime)] == [
        "empty",
        "city",
    ]


@pytest.mark.asyncio
async def test_each_append_publishes_its_own_copy() -> None:
    """The default lives in a module-level mapping. Publishing it by reference
    would let any reader that edits what it was handed corrupt every later
    waiting message in the process."""

    _, first = await _run("ask_user_question", {"message": "One?"})
    published = _published_interactions(first)
    published[0]["field"] = "mutated"

    _, second = await _run("ask_user_question", {"message": "Two?"})
    assert _published_interactions(second) == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_the_caller_s_own_list_is_not_appended_to() -> None:
    """The published list outlives the call on the waiting request, the
    tool-call record and the result dict, so it must not be a list the caller
    still holds a reference to."""

    pattern = ReActPattern(max_iterations=2)
    supplied: list[dict[str, Any]] = []
    _, published = await pattern._send_waiting_message(
        runtime=PatternRuntime(execution_id="exec-waiting-copies"),
        message="Which city?",
        message_type="question",
        interactions=supplied,
        metadata={},
    )
    assert supplied == []
    assert published == [DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_appended_field_passes_the_write_side_validator() -> None:
    """The appended field has to survive the same admissibility rules the
    model-supplied ones do, or the append buys nothing."""

    from xagent.core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
    from xagent.web.services.task_interaction_service import validate_v1_write_payload

    _, runtime = await _run("ask_user_question", {"message": "Which one?"})
    parsed = AskUserQuestionArgs.model_validate(
        {"message": "Which one?", "interactions": _published_interactions(runtime)}
    )
    validate_v1_write_payload(parsed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interaction_type",
    [
        "something_new",
        "",
        "connect_apps",
        ["select_one"],
        {"type": "select_one"},
        None,
        7,
    ],
    ids=["unknown", "blank", "connect_apps", "list", "dict", "none", "int"],
)
async def test_a_type_the_render_surface_cannot_use_gets_a_field(
    interaction_type: Any,
) -> None:
    """``normalizeInteractions`` drops anything outside its whitelist and a
    list it empties renders no form, so an unrecognized type is the very dead
    end this append exists to prevent. Nothing upstream refuses one: the
    schema ``enum`` is a prompt, the tool arguments are raw ``json.loads``, and
    ``validate_v1_write_payload`` -- called from ``create()`` -- is behind a
    seam no route reaches yet.

    ``connect_apps`` is the one the frontend does keep and still cannot answer:
    a form whose fields are all live widgets renders no Submit button at all
    (``isConnectAppsOnly``, ``clarification-form.tsx``). The entry is kept and
    the field is appended, so the widget still renders and the form gains
    something to submit."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Well?",
            "interactions": [
                {"type": interaction_type, "field": "x", "label": "X"},
            ],
        },
    )
    published = _published_interactions(runtime)
    assert published[0]["label"] == "X"
    assert published[1:] == [DEFAULT_FIELD]


def test_a_non_str_type_is_refused_even_against_a_hashing_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the ``isinstance`` guard buys. Today ``INTERACTION_TYPES`` is a
    tuple, so ``in`` compares by equality and an unhashable ``type`` merely
    fails to match. Stored as a set -- the obvious edit for a membership
    test -- the same value raises ``TypeError`` out of ``_is_answerable``,
    past the whole waiting path, and kills the run. The guard makes the
    answer independent of which container the seven names live in."""

    monkeypatch.setattr(react, "INTERACTION_TYPES", frozenset(INTERACTION_TYPES))
    assert react._is_answerable({"type": ["select_one"], "field": "x"}) is False
    assert react._is_answerable({"type": "text_input", "field": "x"}) is True


@pytest.mark.asyncio
async def test_the_appended_field_is_deduplicated_against_the_kept_ones() -> None:
    """The appended field goes through the same ``_unique_field`` dedup the
    model-supplied fields do, so an unanswerable entry that already occupies
    ``response`` does not end up sharing a name with it -- two fields under one
    name lose one of the two submitted answers."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which one?",
            "interactions": [
                {
                    "type": "select_one",
                    "field": "response",
                    "label": "Which region?",
                    "options": [],
                }
            ],
        },
    )
    published = _published_interactions(runtime)
    assert [item["field"] for item in published] == ["response", "response_2"]
    assert published[1] == {**DEFAULT_FIELD, "field": "response_2"}


@pytest.mark.asyncio
async def test_a_non_dict_entry_survives_the_used_name_scan() -> None:
    """``_is_answerable`` refuses a non-dict entry, and the name scan right
    after it has to tolerate the same shape or that refusal turns into an
    ``AttributeError`` at the only call site. Driven directly: every caller
    normalizes first, so no public path can deliver one today."""

    pattern = ReActPattern(max_iterations=2)
    runtime = PatternRuntime(execution_id="exec-non-dict-entry")

    _, published = await pattern._send_waiting_message(
        runtime=runtime,
        message="Which one?",
        message_type="question",
        interactions=["response"],  # type: ignore[list-item]
        metadata={},
    )
    assert published == ["response", DEFAULT_FIELD]


@pytest.mark.asyncio
async def test_the_appended_field_walks_past_two_taken_names() -> None:
    """``_unique_field`` keeps counting. One collision only proves the suffix
    is appended; two prove the counter advances instead of retrying ``_2``."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Which one?",
            "interactions": [
                {
                    "type": "select_one",
                    "field": "response",
                    "label": "Which region?",
                    "options": [],
                },
                {
                    "type": "select_one",
                    "field": "response",
                    "label": "Which city?",
                    "options": [],
                },
            ],
        },
    )
    published = _published_interactions(runtime)
    assert [item["field"] for item in published] == [
        "response",
        "response_2",
        "response_3",
    ]
    assert published[2] == {**DEFAULT_FIELD, "field": "response_3"}


@pytest.mark.asyncio
async def test_a_live_widget_survives_beside_an_answerable_field() -> None:
    """Appending to an unanswerable list must not become "drop every widget".
    ``connect_apps`` is unanswerable alone but is a real control the model may
    want shown next to a question, and the append is all-or-nothing."""

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Connect an app, then tell me which one.",
            "interactions": [
                {"type": "connect_apps", "field": "apps", "label": "Connect"},
                {"type": "text_input", "field": "which", "label": "Which one?"},
            ],
        },
    )
    assert [item["type"] for item in _published_interactions(runtime)] == [
        "connect_apps",
        "text_input",
    ]


@pytest.mark.asyncio
async def test_a_normalized_alias_passes_the_write_side_validator() -> None:
    """The alias mapping is what keeps the readers of a published type
    agreeing. Without it ``{"type": "text"}`` -- a question the frontend
    renders perfectly well -- would be published verbatim, read as
    unanswerable here, and rejected as an unsupported type by this
    validator."""

    from xagent.core.tools.adapters.vibe.ask_user_tool import AskUserQuestionArgs
    from xagent.web.services.task_interaction_service import validate_v1_write_payload

    _, runtime = await _run(
        "ask_user_question",
        {
            "message": "Note?",
            "interactions": [{"type": "text", "field": "note", "label": "Note"}],
        },
    )
    published = _published_interactions(runtime)
    assert published[0]["type"] == "text_input"
    assert published[0]["field"] == "note"
    validate_v1_write_payload(
        AskUserQuestionArgs.model_validate(
            {"message": "Note?", "interactions": published}
        )
    )
