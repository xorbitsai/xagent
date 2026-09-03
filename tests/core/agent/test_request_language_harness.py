from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from xagent.core.agent.context import CompactConfig, ExecutionContext
from xagent.core.agent.context.enrichment import (
    TOP_LEVEL_USER_REQUEST_METADATA_KEY,
    latest_user_text,
    top_level_user_request,
)
from xagent.core.agent.language import (
    OUTPUT_LANGUAGE_METADATA_KEY,
    final_answer_language_rule,
    output_language_directives,
    output_language_policy,
    request_only_language_harness,
)
from xagent.core.agent.pattern.auto.auto import AutoPattern
from xagent.core.agent.pattern.dag.dag import DAGPattern
from xagent.core.agent.pattern.dag.plan_generator import (
    ExecutionPlan,
    LLMPlanGenerator,
    PlanGenerationRequest,
    PlanStep,
)
from xagent.core.agent.pattern.react.react import ReActPattern

ENGLISH_REQUEST = "Summarize the latest customer email and draft a concise reply."
POLLUTED_EXECUTION_REQUEST = (
    f"{ENGLISH_REQUEST}\n\n"
    "[From: Gerard Santos <gerard.santos@example.es>]\n"
    "Connector context: bandeja de entrada, correo electrónico, responder.\n"
    "Attached file(s): correo-del-cliente.pdf"
)


def _polluted_context() -> ExecutionContext:
    context = ExecutionContext(execution_id="request-language-harness")
    context.add_user_message(
        POLLUTED_EXECUTION_REQUEST,
        metadata={"display_message": ENGLISH_REQUEST},
    )
    return context


def _language_surfaces(
    context: ExecutionContext,
) -> tuple[str, dict[str, object], dict[str, object]]:
    plan_payload = json.loads(
        LLMPlanGenerator()._build_prompt(
            PlanGenerationRequest(
                context=context,
                execution_id=context.execution_id,
                available_tool_names=[],
            )
        )
    )
    completion_payload = json.loads(
        DAGPattern(lambda **_: None)._completion_assessment_messages(context)[1][
            "content"
        ]
    )
    return context._system_context(), plan_payload, completion_payload


@pytest.mark.parametrize(
    "user_request",
    [
        pytest.param(
            "Translate the following note to Spanish: The launch is tomorrow.",
            id="explicit-spanish-target",
        ),
        pytest.param("请把最新的客户邮件整理成简短摘要。", id="simplified-chinese"),
        pytest.param("請把最新的客戶郵件整理成簡短摘要。", id="traditional-chinese"),
        pytest.param("OK?", id="short-request"),
        pytest.param(
            'Review este "draft"\\path and keep the product names unchanged.',
            id="mixed-language-special-characters",
        ),
    ],
)
def test_request_language_harness_serializes_each_request_exactly(
    user_request: str,
) -> None:
    harness = request_only_language_harness(user_request)
    quote = harness.split("User-authored request (JSON string):\n", 1)[1]

    assert quote.startswith(json.dumps(user_request, ensure_ascii=False))


def test_request_language_harness_preserves_soft_authority_invariants() -> None:
    harness = request_only_language_harness("Summarize this request.")

    assert "Request-only response language harness" in harness
    assert "explicit or implicit target-language intent" in harness
    assert "explicitly marked as the answer to a pending agent question" in harness
    assert "empty, too short, mixed-language, or depends on conversation" in harness
    assert "Output language:" not in harness


def test_every_unpinned_surface_uses_one_noncontradictory_language_policy() -> None:
    context = _polluted_context()
    root, plan_payload, completion_payload = _language_surfaces(context)
    missing_display = ExecutionContext(execution_id="missing-display-policy")
    missing_display.add_user_message(ENGLISH_REQUEST)
    final_rule = final_answer_language_rule()
    canonical_surfaces = [
        root,
        missing_display._system_context(),
        str(plan_payload["output_language_policy"]),
        str(completion_payload["output_language_policy"]),
        output_language_policy(),
        final_rule,
    ]
    forbidden = (
        "unless the current user request explicitly asks for that language change",
        "unless the user-authored request above explicitly asks",
        "unless the `latest_user_request` field explicitly asks",
        "unless the `user_authored_language_request` field explicitly asks",
    )

    for surface in canonical_surfaces:
        assert "explicit or implicit target-language intent" in surface
        assert "explicitly marked as the answer to a pending agent question" in surface
        assert all(clause not in surface for clause in forbidden)

    planner_description = LLMPlanGenerator()._plan_tool_schema()["function"][
        "parameters"
    ]["properties"]["response_language"]["description"]
    assert "explicit or implicit target-language intent" in planner_description
    assert "pending_agent_question_response" in planner_description
    retry_context = ExecutionContext(execution_id="language-retry-policy")
    retry_context.add_user_message("请用中文总结。")
    retry = LLMPlanGenerator._request_language_reminder(
        retry_context,
        ExecutionPlan(steps=[PlanStep(id="summary", task="Write an English summary")]),
    )
    assert retry is not None
    assert "explicit or implicit target-language intent" in retry
    assert "explicitly marked as a pending agent question response" in retry
    assert "from that request alone" not in retry
    assert (
        AutoPattern()
        ._decision_tool_schema()["function"]["description"]
        .endswith(final_rule)
    )
    assert (
        ReActPattern()
        ._final_answer_tool_schema()["function"]["description"]
        .endswith(final_rule)
    )


def test_root_language_harness_uses_only_the_user_authored_request() -> None:
    system_context = _polluted_context()._system_context()
    harness = system_context.split("Request-only response language harness:\n", 1)[1]

    assert json.dumps(ENGLISH_REQUEST) in harness
    assert "Gerard Santos" not in harness
    assert "example.es" not in harness
    assert "bandeja de entrada" not in harness


@pytest.mark.parametrize("display_message", ["", "   \n\t"])
def test_blank_display_message_is_an_authoritative_empty_language_request(
    display_message: str,
) -> None:
    context = ExecutionContext(execution_id="blank-display-language")
    context.metadata["task"] = "Responder al correo adjunto."
    context.add_user_message(
        POLLUTED_EXECUTION_REQUEST,
        metadata={"display_message": display_message},
    )

    system_context, plan_payload, completion_payload = _language_surfaces(context)

    assert context._current_user_request_text(prefer_display=True) == ""
    assert latest_user_text(context, prefer_display=True) == ""
    assert request_only_language_harness("") in system_context
    assert request_only_language_harness(POLLUTED_EXECUTION_REQUEST) not in (
        system_context
    )
    assert plan_payload["latest_user_request"] == ""
    assert "`latest_user_request` field" in plan_payload["output_language_policy"]
    assert "Gerard Santos" not in plan_payload["output_language_policy"]
    assert completion_payload["user_authored_language_request"] == ""
    assert (
        "`user_authored_language_request` field"
        in completion_payload["output_language_policy"]
    )
    assert "Gerard Santos" not in completion_payload["output_language_policy"]


def test_file_only_blank_display_keeps_an_empty_root_language_anchor() -> None:
    context = ExecutionContext(execution_id="file-only-blank-language")
    context.add_user_message("", metadata={"display_message": "  \n"})

    assert request_only_language_harness("") in context._system_context()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"display_message": 42},
        None,
    ],
)
def test_unsupported_display_metadata_preserves_execution_content_fallback(
    metadata: object,
) -> None:
    context = ExecutionContext(execution_id="legacy-display-language")
    context.messages.append(
        SimpleNamespace(
            role="user",
            hidden=False,
            content=POLLUTED_EXECUTION_REQUEST,
            metadata=metadata,
        )
    )

    assert (
        context._current_user_request_text(prefer_display=True)
        == POLLUTED_EXECUTION_REQUEST
    )
    assert latest_user_text(context, prefer_display=True) == POLLUTED_EXECUTION_REQUEST


def test_dag_language_consumers_receive_the_same_user_authored_request() -> None:
    context = _polluted_context()
    _, plan_payload, completion_payload = _language_surfaces(context)

    assert plan_payload["latest_user_request"] == ENGLISH_REQUEST
    assert "`latest_user_request` field" in plan_payload["output_language_policy"]
    assert ENGLISH_REQUEST not in plan_payload["output_language_policy"]
    assert completion_payload["user_authored_language_request"] == ENGLISH_REQUEST
    assert (
        "`user_authored_language_request` field"
        in completion_payload["output_language_policy"]
    )
    assert ENGLISH_REQUEST not in completion_payload["output_language_policy"]
    assert "Gerard Santos" not in completion_payload["output_language_policy"]


def test_new_independent_request_replaces_the_persisted_provenance() -> None:
    context = _polluted_context()
    assert top_level_user_request(context).language_text == ENGLISH_REQUEST

    follow_up = "Switch to Spanish now."
    context.add_user_message(
        f"{follow_up}\n[Connector context in English]",
        metadata={"display_message": follow_up},
    )

    request = top_level_user_request(context)
    assert request.language_text == follow_up
    assert context.metadata[TOP_LEVEL_USER_REQUEST_METADATA_KEY] == {
        "execution_text": f"{follow_up}\n[Connector context in English]",
        "language_text": follow_up,
        "display_state": "text",
    }


def test_structured_language_payloads_include_a_large_request_exactly_once() -> None:
    request = "LANGUAGE_SENTINEL_BEGIN_" + "背景" * 8_000 + "_LANGUAGE_SENTINEL_END"
    context = ExecutionContext(execution_id="large-request-language")
    context.add_user_message(
        "[Connector execution context in Spanish: archivo adjunto]",
        metadata={"display_message": request},
    )

    system_context, plan_payload, completion_payload = _language_surfaces(context)

    assert system_context.count(request) == 1
    assert plan_payload["latest_user_request"] == request
    assert json.dumps(plan_payload, ensure_ascii=False).count(request) == 1
    assert request not in plan_payload["output_language_policy"]
    assert "`latest_user_request` field" in plan_payload["output_language_policy"]
    assert completion_payload["user_authored_language_request"] == request
    assert json.dumps(completion_payload, ensure_ascii=False).count(request) == 1
    assert request not in completion_payload["output_language_policy"]
    assert (
        "`user_authored_language_request` field"
        in completion_payload["output_language_policy"]
    )


@pytest.mark.parametrize(
    ("content", "metadata", "expected_policy", "quotes_request"),
    [
        pytest.param(
            "{request}",
            {},
            output_language_directives("", section="root_existing_request"),
            False,
            id="missing-display-references-existing-request",
        ),
        pytest.param(
            "[Connector context: correo adjunto]",
            {"display_message": "{request}"},
            "{quoted_request}",
            True,
            id="different-display-keeps-isolated-quote",
        ),
        pytest.param(
            "{request}\n[Connector context: correo adjunto]",
            {"display_message": " \n\t"},
            request_only_language_harness(""),
            False,
            id="blank-display-keeps-empty-language-anchor",
        ),
    ],
)
def test_root_language_request_appears_once_for_every_display_shape(
    content: str,
    metadata: dict[str, object],
    expected_policy: str,
    quotes_request: bool,
) -> None:
    request = "ROOT_SENTINEL_BEGIN_" + "背景" * 8_000 + "_ROOT_SENTINEL_END"
    context = ExecutionContext(execution_id="root-request-cardinality")
    context.add_user_message(
        content.format(request=request),
        metadata={
            key: value.format(request=request) if isinstance(value, str) else value
            for key, value in metadata.items()
        },
    )

    system_context = context._system_context()
    provider_prompt = context.get_messages_for_llm()
    provider_text = "\n".join(message["content"] for message in provider_prompt)
    rendered_policy = expected_policy.format(
        quoted_request=request_only_language_harness(request)
    )

    assert system_context.count(request) == (1 if quotes_request else 0)
    assert provider_text.count(request) == system_context.count(
        request
    ) + content.format(request=request).count(request)
    assert rendered_policy in system_context
    if quotes_request:
        assert request_only_language_harness(request) in system_context
    else:
        assert request_only_language_harness(request) not in system_context


@pytest.mark.parametrize("display_message", ["", "  \n\t"])
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "restored"])
def test_dag_step_preserves_authoritative_blank_display_anchor(
    display_message: str,
    restored: bool,
) -> None:
    root = ExecutionContext(execution_id="dag-blank-root")
    root.add_user_message(
        POLLUTED_EXECUTION_REQUEST,
        metadata={"display_message": display_message},
    )
    child = root.create_child_context(
        execution_id="dag-blank-root:step",
        metadata={"dag_step_id": "draft", "dag_step_name": "Draft reply"},
    )
    if restored:
        child = ExecutionContext.from_dict(child.to_dict())

    system_context = child._system_context()

    assert request_only_language_harness("") in system_context
    assert (
        request_only_language_harness(POLLUTED_EXECUTION_REQUEST) not in system_context
    )


@pytest.mark.parametrize(
    ("display_message", "language_text", "display_state"),
    [
        pytest.param(ENGLISH_REQUEST, ENGLISH_REQUEST, "text", id="clean-display"),
        pytest.param("", "", "empty", id="blank-display"),
        pytest.param("  \n\t", "", "empty", id="whitespace-display"),
    ],
)
@pytest.mark.parametrize("compaction", ["fresh", "summary", "truncate"])
@pytest.mark.parametrize("cold_restore", [False, True], ids=["live", "restored"])
def test_dag_request_provenance_survives_compaction_and_restore(
    display_message: str,
    language_text: str,
    display_state: str,
    compaction: str,
    cold_restore: bool,
) -> None:
    root = ExecutionContext(
        execution_id="dag-provenance-root",
        metadata={"task": POLLUTED_EXECUTION_REQUEST},
    )
    root.add_user_message(
        POLLUTED_EXECUTION_REQUEST,
        metadata={"display_message": display_message},
    )
    child = root.create_child_context(
        execution_id="dag-provenance-step",
        metadata={"dag_step_id": "draft", "dag_step_name": "Redactar respuesta"},
    )
    child.add_user_message(
        "DAG step instruction in Spanish",
        metadata={"dag_step_id": "draft", "kind": "dag_step_instruction"},
    )

    if compaction == "summary":
        child.compact_with_llm_response({"content": "Resumen español del trabajo"})
    elif compaction == "truncate":
        child.compact_config = CompactConfig(
            enabled=True,
            threshold=1,
            max_messages=1,
        )
        assert child.compact_if_needed().compacted
    if cold_restore:
        child = ExecutionContext.from_dict(child.to_dict())

    request = top_level_user_request(child)
    snapshot = child.metadata[TOP_LEVEL_USER_REQUEST_METADATA_KEY]
    system_context = child._system_context()

    assert request.execution_text == POLLUTED_EXECUTION_REQUEST
    assert request.language_text == language_text
    assert request.display_state == display_state
    assert snapshot == {
        "execution_text": POLLUTED_EXECUTION_REQUEST,
        "language_text": language_text,
        "display_state": display_state,
    }
    assert request_only_language_harness(language_text) in system_context
    assert (
        request_only_language_harness(POLLUTED_EXECUTION_REQUEST) not in system_context
    )


def _waiting_dag_context(answer: str) -> tuple[DAGPattern, ExecutionContext]:
    root = ExecutionContext(execution_id="dag-language-wait")
    root.add_user_message(
        POLLUTED_EXECUTION_REQUEST,
        metadata={"display_message": ENGLISH_REQUEST},
    )
    child = root.create_child_context(execution_id="dag-language-wait:confirm")
    pattern = DAGPattern(lambda **_: None)
    pattern.status = "waiting_for_user"
    pattern.active_step_id = "confirm"
    pattern.active_step_ids = ["confirm"]
    pattern.active_step_contexts = {"confirm": child.to_dict()}
    pattern.active_step_pattern_states = {
        "confirm": {
            "status": "waiting_for_user",
            "waiting_for_user_request": {"message": "Which date?"},
        }
    }
    pattern.planned_user_message_count = 1
    root.add_user_message(answer)
    assert pattern._forward_user_response_to_waiting_step(root)
    return pattern, root


@pytest.mark.parametrize("cold_restore", [False, True], ids=["live", "cold-restored"])
@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("Solo el viernes.", id="cross-language-answer"),
        pytest.param(
            "Continúa la respuesta en español.",
            id="explicit-language-switch",
        ),
    ],
)
def test_dag_wait_response_keeps_top_level_language_boundary(
    answer: str,
    cold_restore: bool,
) -> None:
    pattern, root = _waiting_dag_context(answer)
    if cold_restore:
        restored_pattern = DAGPattern(lambda **_: None)
        restored_pattern.load_state(pattern.get_state())
        pattern = restored_pattern
        root = ExecutionContext.from_dict(root.to_dict())

    request = top_level_user_request(root)
    plan_payload = json.loads(
        LLMPlanGenerator()._build_prompt(
            PlanGenerationRequest(
                context=root,
                execution_id=root.execution_id,
                available_tool_names=[],
            )
        )
    )
    completion = json.loads(pattern._completion_assessment_messages(root)[1]["content"])

    assert request.language_text == ENGLISH_REQUEST
    assert request.has_pending_response is True
    assert plan_payload["latest_user_request"] == ENGLISH_REQUEST
    plan_answer = next(
        item for item in plan_payload["messages"] if item.get("content") == answer
    )
    assert plan_answer["user_message_context"] == "pending_agent_question_response"
    assert completion["user_authored_language_request"] == ENGLISH_REQUEST
    answer_payload = next(
        item for item in completion["messages"] if item.get("content") == answer
    )
    assert answer_payload["user_message_context"] == "pending_agent_question_response"
    policies = [
        root._system_context(),
        str(plan_payload["output_language_policy"]),
        str(completion["output_language_policy"]),
    ]
    for policy in policies:
        assert "explicitly marked as the answer to a pending agent question" in policy
        assert "only when the marked answer explicitly asks" in policy
        assert "unless the user-authored request above explicitly asks" not in policy


def test_large_split_request_has_canonical_provider_copies_and_budget() -> None:
    display = "DISPLAY_SENTINEL_" + "E" * 20_000
    execution = display + "\nCONNECTOR_SENTINEL_" + "F" * 20_000
    context = ExecutionContext(execution_id="canonical-language-budget")
    context.add_user_message(execution, metadata={"display_message": display})

    provider_messages = context.get_messages_for_llm()
    provider_text = "\n".join(message["content"] for message in provider_messages)

    assert provider_text.count(execution) == 1
    assert provider_text.count(request_only_language_harness(display)) == 1
    assert provider_text.count(display) == 2
    assert context.estimate_context_tokens() >= sum(
        max(1, len(message["content"]) // 4) for message in provider_messages
    )


def test_caller_pinned_language_remains_the_only_hard_authority() -> None:
    context = _polluted_context()
    context.metadata["request_context"] = {OUTPUT_LANGUAGE_METADATA_KEY: "French"}
    context.metadata[OUTPUT_LANGUAGE_METADATA_KEY] = "French"

    system_context, plan_payload, completion_payload = _language_surfaces(context)

    assert "Output language: French" in system_context
    assert "Request-only response language harness" not in system_context
    assert "Output language: French" in plan_payload["output_language_policy"]
    assert (
        "Request-only response language policy"
        not in plan_payload["output_language_policy"]
    )
    assert "Output language: French" in completion_payload["output_language_policy"]
    assert "user_authored_language_request" not in completion_payload
    assert "sole hard authority" in output_language_policy("French")
    assert "user-requested language changes" not in output_language_policy("French")


def test_final_answer_schemas_follow_the_shared_language_guidance() -> None:
    react_schema = ReActPattern()._final_answer_tool_schema()
    react_function = react_schema["function"]
    auto_function = AutoPattern()._decision_tool_schema()["function"]

    rule = final_answer_language_rule()
    assert rule.startswith(
        "The final answer must follow authoritative output language guidance in "
        "the system context when it is present. Otherwise determine the target "
        "language from user-authored request text and conversation context"
    )
    assert "connector metadata" in rule
    for function in (react_function, auto_function):
        answer_description = function["parameters"]["properties"]["answer"][
            "description"
        ]
        assert function["description"].endswith(rule)
        assert answer_description.endswith(rule)
        assert json.dumps(function, ensure_ascii=False).count(rule) == 2


def test_final_answer_guidance_is_self_contained_without_a_root_request() -> None:
    context = ExecutionContext(
        execution_id="attachment-only-language",
        metadata={"request_context": {"files": [{"name": "correo.pdf"}]}},
    )

    root_system = context._system_context()
    react_messages = ReActPattern()._messages_for_llm(
        context,
        has_tools=False,
        force_final_answer=True,
    )
    rule = final_answer_language_rule()

    assert "Request-only response language" not in root_system
    assert rule in react_messages[0]["content"]
    assert (
        AutoPattern()._decision_tool_schema()["function"]["description"].endswith(rule)
    )
    assert "if no such text is available, preserve the language established" in rule
