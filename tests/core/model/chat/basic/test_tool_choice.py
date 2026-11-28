"""Regression coverage for auto-only tool choice on OpenAI-style transports."""

import pytest

from xagent.core.model.chat.basic.openai import OpenAICompatibleLLM
from xagent.core.model.chat.basic.openrouter import OpenRouterLLM


async def _empty_stream():
    if False:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat", "vision_chat", "stream_chat"])
@pytest.mark.parametrize(
    ("tool_choice", "is_forced"),
    [
        ("required", True),
        ({"type": "function", "function": {"name": "lookup"}}, True),
        ("auto", False),
        ("none", False),
        (None, False),
    ],
)
@pytest.mark.parametrize(
    ("adapter", "model_name", "abilities", "auto_only"),
    [
        (OpenRouterLLM, "anthropic/claude-opus-5.5", None, True),
        (OpenRouterLLM, "anthropic/claude-sonnet-5.5", None, True),
        (OpenRouterLLM, "anthropic/claude-sonnet-5.50", None, False),
        (
            OpenRouterLLM,
            "deepseek/deepseek-v4-flash",
            ["auto_tool_choice_only"],
            True,
        ),
        (
            OpenAICompatibleLLM,
            "custom-deployment",
            ["auto_tool_choice_only"],
            True,
        ),
        (OpenAICompatibleLLM, "custom-deployment", None, False),
    ],
)
async def test_openai_style_tool_choice_sent_on_first_request(
    adapter,
    model_name,
    abilities,
    auto_only,
    method,
    tool_choice,
    is_forced,
    mock_chat_completion,
    mocker,
):
    abilities = ["chat", "vision", "tool_calling"] + (abilities or [])
    llm = adapter(model_name, base_url=None, api_key="test-key", abilities=abilities)
    mock_client = mocker.AsyncMock()
    create = mock_client.chat.completions.create
    create.return_value = (
        _empty_stream() if method == "stream_chat" else mock_chat_completion
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI", return_value=mock_client
    )
    messages = [{"role": "user", "content": "Look it up"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    result = getattr(llm, method)(messages, tools=tools, tool_choice=tool_choice)
    if method == "stream_chat":
        async for _ in result:
            pass
    else:
        await result

    assert llm.supports_forced_tool_choice == (not auto_only)
    create.assert_awaited_once()
    assert create.call_args.kwargs["tools"] == tools
    if tool_choice is None:
        assert "tool_choice" not in create.call_args.kwargs
    else:
        expected = "auto" if auto_only and is_forced else tool_choice
        assert create.call_args.kwargs["tool_choice"] == expected
