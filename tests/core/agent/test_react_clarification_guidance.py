"""Suggested answers use the existing persisted interaction contract."""

from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.model.chat.basic.claude import ClaudeLLM
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.tools.adapters.vibe.ask_user_tool import (
    AskUserQuestionArgs,
    AskUserQuestionTool,
)
from xagent.core.tools.adapters.vibe.interaction_types import (
    CLARIFICATION_GUIDANCE,
    SUGGESTED_VALUE_GUIDANCE,
)

from .test_react_ask_user_question_dedup import FakeLLM, _ask_user_question_call


def test_builtin_question_schema_builds_real_gemini_sdk_tool_config() -> None:
    # All ReAct requests carry this schema, even before asking a question.
    # Exercise the SDK's typed Tool/Schema construction without a network call.
    llm = GeminiLLM(api_key="test-key")
    schemas = ReActPattern()._builtin_tool_schemas()
    config = llm._build_gemini_tool_config(schemas, tool_choice="auto")
    declarations = config["tools"][0].function_declarations
    assert [item.name for item in declarations] == [
        schema["function"]["name"] for schema in schemas
    ]
    question = next(item for item in declarations if item.name == "ask_user_question")
    suggestion = question.parameters.properties["interactions"].items.properties[
        "default_value"
    ]
    assert {branch.type.value for branch in suggestion.any_of} == {
        "STRING",
        "NUMBER",
        "BOOLEAN",
        "NULL",
    }


@pytest.mark.parametrize("source", ["builtin", "adapter"])
def test_suggestion_guidance_survives_claude_tool_conversion(source: str) -> None:
    if source == "builtin":
        schemas = ReActPattern()._builtin_tool_schemas()
    else:
        tool = AskUserQuestionTool()
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_type().model_json_schema(),
                },
            }
        ]
    converted = ClaudeLLM(api_key="test-key")._convert_tools_to_anthropic_format(
        schemas
    )
    question = next(tool for tool in converted if tool["name"] == "ask_user_question")
    # Claude currently simplifies unions and drops field-level descriptions.
    # Keep the complete guidance available independently of that conversion.
    assert f"default_value: {SUGGESTED_VALUE_GUIDANCE}" in question["description"]
    assert CLARIFICATION_GUIDANCE in question["description"]


def test_both_question_schemas_expose_suggestions_and_the_same_guidance() -> None:
    pattern = ReActPattern()
    schema = next(
        s["function"]
        for s in pattern._builtin_tool_schemas()
        if s["function"]["name"] == "ask_user_question"
    )
    suggestion = schema["parameters"]["properties"]["interactions"]["items"][
        "properties"
    ]["default_value"]
    assert {branch["type"] for branch in suggestion["anyOf"]} == {
        "string",
        "number",
        "boolean",
        "null",
    }
    assert suggestion["description"] == SUGGESTED_VALUE_GUIDANCE
    assert CLARIFICATION_GUIDANCE in schema["description"]
    assert CLARIFICATION_GUIDANCE in AskUserQuestionTool().description
    assert (
        AskUserQuestionArgs.model_json_schema()["$defs"]["InteractionArg"][
            "properties"
        ]["default_value"]["description"]
        == SUGGESTED_VALUE_GUIDANCE
    )


@pytest.mark.asyncio
async def test_suggestions_and_all_questions_survive_pause_and_restore() -> None:
    interactions: list[dict[str, Any]] = [
        {
            "type": "select_one",
            "field": "cadence",
            "label": "Cadence",
            "default_value": "weekly",
            "options": [{"label": "Weekly", "value": "weekly"}],
        },
        {
            "type": "number_input",
            "field": "threshold",
            "label": "Threshold",
            "default_value": 0,
        },
        {
            "type": "confirm",
            "field": "approval",
            "label": "Approval",
            "default_value": False,
        },
        *[
            {"type": "text_input", "field": f"q{i}", "label": f"Question {i}"}
            for i in range(4)
        ],
    ]
    # Grouping is a view concern, never truncation of the paused request.
    args = AskUserQuestionArgs(message="Questions", interactions=interactions)
    llm = FakeLLM(
        responses=[_ask_user_question_call("ask", args.message, interactions)]
    )
    pattern = ReActPattern(max_iterations=2)
    context = ExecutionContext()
    context.add_user_message("Ask about this task")
    result = await pattern.run(
        context=context,
        tools=[],
        llm=llm,
        runtime=PatternRuntime(execution_id="guided-clarification"),
    )
    assert result["status"] == "waiting_for_user"
    assert result["interactions"] == interactions
    assert result["clarification_draft"].to_dict()["interactions"] == interactions
    restored = ReActPattern()
    restored.load_state(pattern.get_state())
    assert restored.waiting_for_user_request["interactions"] == interactions
